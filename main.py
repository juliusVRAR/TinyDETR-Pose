# ------------------------------------------------------------------------
# LW-DETR
# Copyright (c) 2024 Baidu. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from Conditional DETR (https://github.com/Atten4Vis/ConditionalDETR)
# Copyright (c) 2021 Microsoft. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# ------------------------------------------------------------------------

"""
cleaned main file
"""
import argparse
import datetime
import json
import inspect
import random
import time
import ast
import copy
import textwrap
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, DistributedSampler
from datasets import build_train_dataset, get_coco_api_from_dataset, build_dataset as build_dataset_coco
from data_utils import build_dataset
from engine import evaluate, train_one_epoch, pose_evaluate, bop_evaluate
from models import build_model
from util.drop_scheduler import drop_scheduler
from util.get_param_dicts import get_param_dict
import util.misc as utils
from util.utils import ModelEma, BestMetricHolder, clean_state_dict
from util.benchmark import benchmark
from evaluation_tools.pose_evaluator_init import build_pose_evaluator, build_better_pose_evaluator
from torch.utils.tensorboard import SummaryWriter


COCO_BBOX_METRIC_NAMES = (
    'AP',
    'AP50',
    'AP75',
    'AP_small',
    'AP_medium',
    'AP_large',
    'AR_1',
    'AR_10',
    'AR_100',
    'AR_small',
    'AR_medium',
    'AR_large',
)
def normalize_loss_weight_name(loss_name: str) -> str:
    if loss_name.endswith('_enc'):
        return loss_name[:-4]

    base_name, separator, suffix = loss_name.rpartition('_')
    if separator and suffix.isdigit():
        return base_name

    return loss_name


def extract_criterion_loss_dispatch(criterion) -> dict:
    try:
        source = textwrap.dedent(inspect.getsource(type(criterion).get_loss))
    except (OSError, TypeError):
        return {}

    tree = ast.parse(source)
    dispatch = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if not isinstance(target, ast.Name) or target.id != 'loss_map':
                continue
            if not isinstance(node.value, ast.Dict):
                continue
            for key_node, value_node in zip(node.value.keys, node.value.values):
                if not isinstance(key_node, ast.Constant):
                    continue
                if not isinstance(value_node, ast.Attribute):
                    continue
                if not isinstance(value_node.value, ast.Name) or value_node.value.id != 'self':
                    continue
                dispatch[str(key_node.value)] = value_node.attr
    return dispatch


def extract_nested_loss_calls(method) -> list:
    try:
        source = textwrap.dedent(inspect.getsource(method))
    except (OSError, TypeError):
        return []

    tree = ast.parse(source)
    nested_losses = []
    method_name = getattr(method, '__name__', '')
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if not isinstance(node.func, ast.Attribute):
            continue
        if not isinstance(node.func.value, ast.Name) or node.func.value.id != 'self':
            continue
        callee_name = node.func.attr
        if not callee_name.startswith('loss_') or callee_name == method_name:
            continue
        nested_losses.append(callee_name)

    return sorted(dict.fromkeys(nested_losses))


def get_loss_configuration(args, criterion) -> dict:
    dispatch = extract_criterion_loss_dispatch(criterion)
    requested_losses = []
    for loss_entry in getattr(criterion, 'losses', []):
        entry = {'name': loss_entry}
        implementation = dispatch.get(loss_entry)
        if implementation is not None:
            entry['implementation'] = implementation
            method = getattr(type(criterion), implementation, None)
            if method is not None:
                nested_losses = extract_nested_loss_calls(method)
                if nested_losses:
                    entry['nested_losses'] = nested_losses
        requested_losses.append(entry)

    weighted_loss_terms = {}
    for loss_name, weight in getattr(criterion, 'weight_dict', {}).items():
        normalized_name = normalize_loss_weight_name(loss_name)
        weighted_loss_terms.setdefault(normalized_name, float(weight))

    return {
        'requested_loss_entries': requested_losses,
        'weighted_loss_terms': weighted_loss_terms,
        'criterion_flags': {
            'aux_loss': bool(args.aux_loss),
            'sum_group_losses': bool(getattr(criterion, 'sum_group_losses', False)),
            'use_varifocal_loss': bool(getattr(criterion, 'use_varifocal_loss', False)),
            'use_position_supervised_loss': bool(getattr(criterion, 'use_position_supervised_loss', False)),
            'ia_bce_loss': bool(getattr(criterion, 'ia_bce_loss', False)),
        },
        'training_behavior': {
            'warm_up_epochs': args.warm_up_epochs,
            'reduce_det_loss_epochs': args.reduce_det_loss_epochs,
        },
    }


def append_summary_log(summary_log_path: Path, payload: dict) -> None:
    summary_log_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_log_path.open('a') as log_file:
        log_file.write(json.dumps(payload, indent=2))
        log_file.write('\n\n')


def write_summary_run_config(summary_log_path: Path, args, criterion) -> None:
    append_summary_log(summary_log_path, {
        'event': 'run_config',
        'timestamp': str(datetime.datetime.now()),
        'dataset_file': args.dataset_file,
        'train_set': args.train_set,
        'models': args.models,
        'eval_set': args.eval_set,
        'loss_config': get_loss_configuration(args, criterion),
        'evaluation_config': {
            'eval_interval': args.eval_interval,
            'quick_eval': args.quick_eval,
            'skip_coco_eval': args.skip_coco_eval,
            'skip_pose_eval': args.skip_pose_eval,
        },
    })


def build_coco_eval_summary(test_stats: dict, image_set: str, phase: str, epoch=None,
                            eval_name: str = 'coco') -> dict:
    metrics = {}
    if 'loss' in test_stats:
        metrics['loss'] = float(test_stats['loss'])
    if 'class_error' in test_stats:
        metrics['class_error'] = float(test_stats['class_error'])

    bbox_stats = test_stats.get('coco_eval_bbox')
    if bbox_stats is not None:
        bbox_values = [float(value) for value in bbox_stats]
        metrics['bbox'] = {
            name: value for name, value in zip(COCO_BBOX_METRIC_NAMES, bbox_values)
        }

    summary = {
        'event': 'evaluation',
        'timestamp': str(datetime.datetime.now()),
        'phase': phase,
        'eval_type': eval_name,
        'image_set': image_set,
        'metrics': metrics,
    }
    if epoch is not None:
        summary['epoch'] = epoch
        summary['epoch_1based'] = epoch + 1
    return summary


def optional_float(value):
    if value is None:
        return None
    return float(value)


def build_pose_eval_summary(pose_metrics: dict,
                            image_set: str, bbox_mode: str, phase: str,
                            quick_mode: bool, epoch=None) -> dict:
    summary = {
        'event': 'evaluation',
        'timestamp': str(datetime.datetime.now()),
        'phase': phase,
        'eval_type': 'pose',
        'image_set': image_set,
        'bbox_mode': bbox_mode,
        'quick_mode': quick_mode,
        'metrics': {
            'ADD': optional_float(pose_metrics.get('ADD')),
            'ADI': optional_float(pose_metrics.get('ADI')),
            'ADD_minus_S': optional_float(pose_metrics.get('ADD_minus_S')),
            'avg_translation_error': optional_float(pose_metrics.get('avg_translation_error')),
            'avg_rotation_error': optional_float(pose_metrics.get('avg_rotation_error')),
            'avg_rotation_error_symmetry_aware': optional_float(
                pose_metrics.get('avg_rotation_error_symmetry_aware')
            ),
            'avg_rotation_error_nonsymmetric_only': optional_float(
                pose_metrics.get('avg_rotation_error_nonsymmetric_only')
            ),
        },
    }
    if epoch is not None:
        summary['epoch'] = epoch
        summary['epoch_1based'] = epoch + 1
    return summary


def serialize_best_metric_single(metric) -> dict:
    return {
        'best_res': float(metric.best_res),
        'best_ep': int(metric.best_ep),
        'better': metric.better,
        'init_res': float(metric.init_res),
    }


def serialize_best_metric_holder(best_metric_holder) -> dict:
    if best_metric_holder is None:
        return {}

    state = {
        'use_ema': bool(best_metric_holder.use_ema),
        'best_all': serialize_best_metric_single(best_metric_holder.best_all),
    }
    if best_metric_holder.use_ema:
        state['best_regular'] = serialize_best_metric_single(best_metric_holder.best_regular)
        state['best_ema'] = serialize_best_metric_single(best_metric_holder.best_ema)
    return state


def restore_best_metric_single(metric, state: dict) -> None:
    if not state:
        return

    if 'better' in state:
        metric.better = state['better']
    if 'init_res' in state:
        metric.init_res = float(state['init_res'])
    if 'best_res' in state:
        metric.best_res = float(state['best_res'])
    if 'best_ep' in state:
        metric.best_ep = int(state['best_ep'])


def restore_best_metric_holder(best_metric_holder, state: dict) -> None:
    if not state:
        return

    restore_best_metric_single(best_metric_holder.best_all, state.get('best_all', {}))
    if best_metric_holder.use_ema:
        restore_best_metric_single(best_metric_holder.best_regular, state.get('best_regular', {}))
        restore_best_metric_single(best_metric_holder.best_ema, state.get('best_ema', {}))


def build_training_checkpoint(args, model_state: dict, optimizer, lr_scheduler, epoch: int,
                              ema_m=None, best_metric_holder=None, best_adds_score: float = 0.0,
                              extra_payload: dict | None = None) -> dict:
    checkpoint = {
        'model': model_state,
        'optimizer': optimizer.state_dict(),
        'lr_scheduler': lr_scheduler.state_dict(),
        'epoch': epoch,
        'args': args,
        'best_map_holder': serialize_best_metric_holder(best_metric_holder),
        'best_adds_score': float(best_adds_score),
    }
    if ema_m is not None:
        checkpoint['ema_model'] = ema_m.module.state_dict()
    if extra_payload:
        checkpoint.update(extra_payload)
    return checkpoint


def make_lr_drop_lambda(base_lr: float, lr_after_drop: float, drop_epoch: int | None):
    if drop_epoch is None or drop_epoch <= 0 or base_lr <= 0:
        return lambda scheduler_epoch: 1.0

    drop_factor = lr_after_drop / base_lr
    return lambda scheduler_epoch: 1.0 if scheduler_epoch < drop_epoch else drop_factor


def set_scheduler_epoch_lrs(lr_scheduler, scheduler_epoch: int) -> None:
    lr_scheduler.last_epoch = scheduler_epoch
    lrs = [
        base_lr * lr_lambda(scheduler_epoch)
        for base_lr, lr_lambda in zip(lr_scheduler.base_lrs, lr_scheduler.lr_lambdas)
    ]
    for param_group, lr in zip(lr_scheduler.optimizer.param_groups, lrs):
        param_group['lr'] = lr
    lr_scheduler._last_lr = lrs

def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    # Learnining hyperparameters
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--lr_pose_heads', default=None, type=float,
                        help='Learning rate for newly added pose prediction heads. Defaults to --lr when omitted.')
    parser.add_argument('--lr_encoder', default=1.5e-6, type=float) 
    parser.add_argument('--lr_backbone', default=1e-6, type=float) 
    parser.add_argument('--lr_transformer', default=1e-5, type=float)
    parser.add_argument('--batch_size', default=2, type=int,
                        help='per-GPU batch size')
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=12, type=int)
    parser.add_argument('--lr_drop', default=11, type=int)
    parser.add_argument('--lr_after_drop', default=5e-5, type=float,
                        help='Base learning rate to use once --lr_drop has been reached.')
    parser.add_argument('--lr_drop_pose_heads', default=None, type=int,
                        help='Epoch interval for dropping pose-head learning rate. Defaults to --lr_drop when omitted.')
    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='gradient clipping max norm')
    parser.add_argument('--lr_vit_layer_decay', default=0.8, type=float)
    parser.add_argument('--lr_component_decay', default=1.0, type=float)

    # drop args 
    # dropout and stochastic depth drop rate; set at most one to non-zero
    parser.add_argument('--dropout', type=float, default=0,
                        help='Drop path rate (default: 0.0)')
    parser.add_argument('--drop_path', type=float, default=0,
                        help='Drop path rate (default: 0.0)')

    # early / late dropout and stochastic depth settings
    parser.add_argument('--drop_mode', type=str, default='standard',
                        choices=['standard', 'early', 'late'], help='drop mode')
    parser.add_argument('--drop_schedule', type=str, default='constant',
                        choices=['constant', 'linear'],
                        help='drop schedule for early dropout / s.d. only')
    parser.add_argument('--cutoff_epoch', type=int, default=0,
                        help='if drop_mode is early / late, this is the epoch where dropout ends / starts')

    # Model parameters
    parser.add_argument('--pretrained_encoder', type=str, default=None, 
                        help="Path to the pretrained encoder.")
    parser.add_argument('--pretrain_weights', type=str, default=None, 
                        help="Path to the pretrained model.")
    parser.add_argument('--pretrain_exclude_keys', type=str, default=None, nargs='+', 
                        help="Keys you do not want to load.")
    parser.add_argument('--pretrain_keys_modify_to_load', type=str, default=None, nargs='+',
                        help="Keys you want to modify to load. Only used when loading objects365 pre-trained weights.")

    # * Backbone
    parser.add_argument('--encoder', default='vit_tiny', type=str,
                        help="Name of the transformer or convolutional encoder to use")
    parser.add_argument('--vit_encoder_num_layers', default=12, type=int,
                        help="Number of layers used in ViT encoder")
    parser.add_argument('--window_block_indexes', default=None, type=int, nargs='+')
    parser.add_argument('--position_embedding', default='sine', type=str, 
                        choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")
    parser.add_argument('--out_feature_indexes', default=[-1], type=int, nargs='+', help='only for vit now')

    # * Transformer
    parser.add_argument('--dec_layers', default=3, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=2048, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=256, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--sa_nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's self-attentions")
    parser.add_argument('--ca_nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's cross-attentions")
    parser.add_argument('--num_queries', default=300, type=int,
                        help="Number of query slots")
    parser.add_argument('--group_detr', default=13, type=int,
                        help="Number of groups to speed up detr training")
    parser.add_argument('--two_stage', action='store_true')
    parser.add_argument('--projector_scale', default='P4', type=str, nargs='+', choices=('P3', 'P4', 'P5', 'P6'))
    parser.add_argument('--lite_refpoint_refine', action='store_true', help='lite refpoint refine mode for speed-up')
    parser.add_argument('--num_select', default=100, type=int,
                        help='the number of predictions selected for evaluation')
    parser.add_argument('--dec_n_points', default=4, type=int,
                        help='the number of sampling points')
    parser.add_argument('--decoder_norm', default='LN', type=str)
    parser.add_argument('--bbox_reparam', action='store_true')

    # * Matcher
    parser.add_argument('--set_cost_class', default=2., type=float,
                        help="Class coefficient in the matching cost")
    parser.add_argument('--set_cost_bbox', default=5., type=float,
                        help="L1 box coefficient in the matching cost")
    parser.add_argument('--set_cost_giou', default=2., type=float,
                        help="giou box coefficient in the matching cost")
    parser.add_argument('--set_cost_rotation', default=2.0, type=float,
                        help="rotation coefficient in the matching cost")
    parser.add_argument('--set_cost_translation', default=5., type=float,
                        help="translation coefficient in the matching cost")
    parser.add_argument('--set_cost_keypoint', default=5., type=float,
                        help="keypoint coefficient in the matching cost")
    parser.add_argument('--matcher_type', default='6d', choices=['6d', '6d_rot_trans', 'ablation', 'hungarian', 'yopo'], type=str,
                        help="Type of matcher to use, hungarian is the 3d match from lwdetr and will probably not work")
    parser.add_argument('--ablation_topk_candidates', default=3, type=int,
                        help='Per-target top-k detection candidates forwarded to the second-stage pose matcher.')
    parser.add_argument('--matcher_symmetry_stride', default=1, type=int,
                        help='Use every Nth symmetry transform in pose-aware matcher rotation cost.')
    
    # PoET Config
    parser.add_argument('--bbox_mode', default='backbone', type=str, choices=('gt', 'backbone', 'jitter'),
                        help='Defines which bounding boxes should be used for PoET to determine query embeddings.')
    parser.add_argument('--im_size', type=int, nargs=2, default=(512, 640)) # Must be divisible by 64
    parser.add_argument('--num_feature_levels', default=4, type=int, help='number of feature levels')
    parser.add_argument('--reference_points', default='bbox', type=str, choices=('bbox', 'learned'),
                        help='Defines whether the transformer reference points are learned or extracted from the bounding boxes')
    parser.add_argument('--class_mode', default='specific', type=str, choices=('agnostic', 'specific'),
                        help="Determine whether PoET ist trained class-specific or class-agnostic")
    parser.add_argument('--rotation_representation', default='6d', type=str, choices=('6d', 'sarr'),
                        help="Determine the rotation representation with which PoET is trained.")
    
    parser.add_argument('--query_embedding', default='bbox', type=str, choices=('bbox', 'learned'),
                        help='Defines whether the transformer query embeddings are learned or determined by the bounding boxes')
    # * Uncertainty Configs
    parser.add_argument('--aleatoric', action='store_true', help="Extend PoET for aleatoric uncertainty estimation by adding dedicated aleatoric uncertainty heads.")
    parser.add_argument('--calibrate', action='store_true', help="Only train the aleatoric uncertainty heads, freeze all other weights.")
    # * Loss coefficients
    parser.add_argument('--cls_loss_coef', default=2, type=float)
    parser.add_argument('--bbox_loss_coef', default=5, type=float)
    parser.add_argument('--giou_loss_coef', default=2, type=float)
    parser.add_argument('--focal_alpha', default=0.25, type=float)
    parser.add_argument('--reduce_det_loss_epochs', default=10000000, type=int, help='Number of epochs after which detection losses are reduced to half.')
    
    # * Loss coefficients
    # Pose Estimation losses
    parser.add_argument('--keypoint_loss_coef', default=10.0, type=float, help='Loss weighing parameter for the keypoints')
    parser.add_argument('--trans_z_loss_coef', default=1.0, type=float, help='Loss weighing parameter for the translation z component')
    parser.add_argument('--trans_xy_loss_coef', default=1.0, type=float, help='Loss weighing parameter for the translation')
    parser.add_argument('--rot_loss_coef', default=1.0, type=float, help='Loss weighing parameter for the rotation')
    parser.add_argument('--adds_loss_coef', default=1.0, type=float, help='Loss weighing parameter for the ADD-S metric.')
    parser.add_argument('--warm_up_epochs', default=0, type=int, help='Number of epochs to train ADD-S at 10% weight before using the full ADD-S coefficient.')
    # Loss
    parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false',
                        help="Disables auxiliary decoding losses (loss at each layer)")
    parser.add_argument('--sum_group_losses', action='store_true',
                        help="To sum losses across groups or mean losses.")
    parser.add_argument('--use_varifocal_loss', action='store_true')
    parser.add_argument('--use_position_supervised_loss', action='store_true')
    parser.add_argument('--ia_bce_loss', action='store_true')



    # dataset parameters
    parser.add_argument('--dataset_file', default='ycbv', type=str, choices=('ycbv', 'lmo'))
    parser.add_argument('--square_resize_div_64', action='store_true')
    parser.add_argument('--dataset_path', default='/data', type=str,
                        help='Path to the dataset')
    parser.add_argument('--train_set', default="train_real", type=str, help="Determine on which dataset split to train")
    parser.add_argument('--eval_set', default="test", type=str, help="Determine on which dataset split to evaluate")
    parser.add_argument('--synt_background', default=None, type=str,
                        help="Directory containing the background images from which to sample")
    parser.add_argument('--camera', default='camera_uw.json', type=str,
                        help='Camera intrisics file. This should be loacted at the root of the dataset.')
    parser.add_argument('--n_classes', default=21, type=int,
                        help='number of object categories, this is ignored if you set dataset_file to ycbv')
    parser.add_argument('--jitter_probability', default=0.5, type=float,
                        help='If bbox_mode is set to jitter, this value indicates the probability '
                             'that jitter is applied to a bounding box.')
    parser.add_argument('--rgb_augmentation', action='store_true',
                        help='Activate image augmentation for training pose estimation.')
    parser.add_argument('--grayscale', action='store_true', help='Activate grayscale augmentation.')
    parser.add_argument('--n_mesh_points', default=128, type=int, help='Number of mesh points to sample for symmetry-aware loss For debugging 128 for train 512.')
    # Data augmentations TODO: add yolox6d 
    parser.add_argument('--mosaic_augmentation', action='store_true',
                        help='Whether to use mosaic augmentation (from yolox6d).')
    # output and logging
    parser.add_argument('--output_dir', default='output',
                        help='path where to save, empty for no saving')
    parser.add_argument('--checkpoint_interval', default=5, type=int,
                        help='epoch interval to save checkpoint')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default=None, type=str, 
                        help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--use_ema', action='store_true')
    parser.add_argument('--ema_decay', default=0.9997, type=float)
    parser.add_argument('--num_workers', default=2, type=int)

    parser.add_argument('--eval_bop', action='store_true', help="Run model in BOP challenge evaluation mode")

    parser.add_argument('--cache_mode', default=False, action='store_true', help='whether to cache images on memory')
    # distributed training parameters
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', 
                        help='url used to set up distributed training')
    parser.add_argument('--sync_bn', default=True, type=bool,
                        help='setup synchronized BatchNorm for distributed training')
    
    # fp16
    parser.add_argument('--fp16_eval', default=False, action='store_true',
                        help='evaluate in fp16 precision.')
    # * Evaluator
    parser.add_argument('--eval_interval', type=int, default=3,
                        help="Epoch interval after which the current model is evaluated")
    parser.add_argument('--class_info', type=str, default='/annotations/classes.json',
                        help='path to .txt-file containing the class names')
    parser.add_argument('--models', type=str, default='/models/',
                        help='path to a directory containing the classes models')
    parser.add_argument('--model_symmetry', type=str, default='/annotations/symmetries.json',
                        help='path to .json-file containing the class symmetries')
    parser.add_argument("--quick_eval", action='store_true',
                        help="Enable quick evaluation mode (process only 10% of batches for faster evaluation)")
    # Tensorboard
    parser.add_argument('--tensorboard', action='store_true',
                        help='Enable TensorBoard logging')
    parser.add_argument('--tb_iter_freq', type=int, default=100,
                        help='Iteration frequency for TB logging (optional)')
    ## Eval modes
    parser.add_argument('--eval_only', action='store_true',
                        help='Run standard evaluation and exit (no training)')
    parser.add_argument('--pose_eval_only', action='store_true',
                        help='Run pose evaluation (requires --resume) and exit')
    parser.add_argument('--eval_batches', type=int, default=-1,
                        help='Limit number of eval batches for quick debug (-1 = all)')
    parser.add_argument('--skip_pose_eval', action='store_true',
                        help='Skip pose evaluation during --eval / training')
    parser.add_argument('--skip_coco_eval', action='store_true',
                        help='Skip box/coco evaluation, run pose only')


    # subparsers
    subparsers = parser.add_subparsers(title='sub-commands', dest='subcommand',
        description='valid subcommands', help='additional help')
    # subparser for export model
    parser_export = subparsers.add_parser('export_model', help='LWDETR model export')
    parser_export.add_argument('--shape', type=int, nargs=2, default=(640, 640), help="input shape (width, height)")
    parser_export.add_argument('--infer_dir', type=str, default=None)
    parser_export.add_argument('--verbose', type=ast.literal_eval, default=False, nargs="?", const=True)
    parser_export.add_argument('--opset_version', type=int, default=17)
    parser_export.add_argument('--simplify', action='store_true', help="Simplify onnx model")
    parser_export.add_argument('--tensorrt', '--trtexec', '--trt', action='store_true',
                               help="build tensorrt engine")
    parser_export.add_argument('--dry-run', '--test', '-t', action='store_true', help="just print command")
    return parser

def should_run_pose_eval(epoch: int, total_epochs: int, warmup_epochs: int) -> bool:
    """
    Adaptive pose-eval schedule (epoch is 0-based):
      - < warmup_epochs         : no pose eval
      - warmup..80% of training : every 10 epochs
      - 80%..95% of training    : every 5 epochs
      - >=95% of training       : every epoch
      - always run on last epoch
    """
    e = epoch + 1  # convert to 1-based for readability

    if e < warmup_epochs:
        return False

    # Always evaluate on final epoch
    if e == total_epochs:
        return True

    frac = e / float(total_epochs)

    if frac >= 0.95:
        # 95–100%: every epoch
        return True
    elif frac >= 0.80:
        # 80–95%: every 5 epochs
        return (e % 5) == 0
    else:
        # After warmup, before 80%: every eval_interval epochs
        return (e - warmup_epochs) % args.eval_interval == 0

def main(args):
    if args.eval_only:
        args.eval = True
    # if args.lr_pose_heads is None:
    #     args.lr_pose_heads = args.lr
    # if args.lr_drop_pose_heads is None:
    #     args.lr_drop_pose_heads = args.lr_drop

    utils.init_distributed_mode(args)
    print("git:\n  {}\n".format(utils.get_sha()))
    print(args)

    device = torch.device(args.device)
     # TensorBoard writer
    writer = None
    if getattr(args, "tensorboard", False):
        log_dir = Path(args.output_dir) / "tb" if args.output_dir else Path("./tb")
        log_dir.mkdir(parents=True, exist_ok=True)
        writer = SummaryWriter(str(log_dir))
    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    model, criterion, postprocessors, matcher = build_model(args)
    model.to(device)
    # TODO: Check which one is better in terms of runtime speed
    pose_evaluator = build_pose_evaluator(args)
    #pose_evaluator = build_better_pose_evaluator(args)
    if args.use_ema:
        ema_m = ModelEma(model, decay=args.ema_decay)
    else:
        ema_m = None
    model_without_ddp = model
    
    if args.distributed:
        if args.sync_bn:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=False)
        model_without_ddp = model.module
    
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)
    param_dicts = get_param_dict(args, model_without_ddp)

    # TODO: Check if the other one is needed
    print(f"LR1: {args.lr}")
    optimizer = torch.optim.AdamW(param_dicts, lr=args.lr, 
                                  weight_decay=args.weight_decay)
    
    
    # # Separate the parameters
    # backbone_params = []
    # transformer_params = []
    # new_head_params = []

    # for name, param in model.named_parameters():
    #     if not param.requires_grad:
    #         continue
            
    #     if "backbone" in name:
    #         # The Pretrained ViT
    #         backbone_params.append(param)
    #     elif "transformer" in name:
    #         # The Pretrained Transformer Decoder
    #         transformer_params.append(param)
    #     else:
    #         # Your New Heads (Class, Box, Z, XY, Rot)
    #         new_head_params.append(param)

    # # Define the Optimizer
    # optimizer = torch.optim.AdamW([
    #     # Backbone: VERY Slow (Preserve Objects365 knowledge from CAEv2)
    #     {'params': backbone_params, 'lr': args.lr_encoder}, 
    #     # Transformer: Slow
    #     {'params': transformer_params, 'lr': args.lr_transformer},
    #     # New Heads: Fast (They are learning from scratch!)
    #     {'params': new_head_params, 'lr': args.lr} 
    #     ], weight_decay=args.weight_decay)
    
    pose_lr_drop = args.lr_drop if args.lr_drop_pose_heads is None else args.lr_drop_pose_heads
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        [
            make_lr_drop_lambda(
                args.lr,
                args.lr_after_drop,
                pose_lr_drop if group.get('lr_group') == 'pose_heads' else args.lr_drop,
            )
            for group in optimizer.param_groups
        ],
    )
    
    # Build the dataset for training and validation
    dataset_train = build_dataset(image_set=args.train_set, args=args)
    # dataset_train = build_train_dataset(image_set=args.train_set, args=args)
    dataset_val = build_dataset(image_set=args.eval_set, args=args)
    # dataset_val_coco = build_dataset_coco(image_set='val', args=args)

    if args.distributed:
        sampler_train = DistributedSampler(dataset_train)
        sampler_val = DistributedSampler(dataset_val, shuffle=False)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    batch_sampler_train = torch.utils.data.BatchSampler(
        sampler_train, args.batch_size, drop_last=True)

    data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train,
                                   collate_fn=utils.collate_fn, num_workers=args.num_workers)
    data_loader_val = DataLoader(dataset_val, args.batch_size, sampler=sampler_val,
                                 drop_last=False, collate_fn=utils.collate_fn, 
                                 num_workers=args.num_workers)
    # Pose evaluation should be world-size invariant; run it on full val set from rank 0.
    data_loader_pose_val = None
    if utils.is_main_process():
        pose_sampler_val = torch.utils.data.SequentialSampler(dataset_val)
        data_loader_pose_val = DataLoader(
            dataset_val,
            args.batch_size,
            sampler=pose_sampler_val,
            drop_last=False,
            collate_fn=utils.collate_fn,
            num_workers=args.num_workers,
        )
    
    base_ds = get_coco_api_from_dataset(dataset_val)

   
    # if args.pretrain_weights is not None:
    #     checkpoint = torch.load(args.pretrain_weights, map_location='cpu')
    #     # add support to exclude_keys
    #     # e.g., when load object365 pretrain, do not load `class_embed.[weight, bias]`
    #     if args.pretrain_exclude_keys is not None:
    #         assert isinstance(args.pretrain_exclude_keys, list)
    #         for exclude_key in args.pretrain_exclude_keys:
    #             checkpoint['model'].pop(exclude_key)
    #     if args.pretrain_keys_modify_to_load is not None:
    #         from util.obj365_to_coco_model import get_coco_pretrain_from_obj365
    #         assert isinstance(args.pretrain_keys_modify_to_load, list)
    #         for modify_key_to_load in args.pretrain_keys_modify_to_load:
    #             checkpoint['model'][modify_key_to_load] = get_coco_pretrain_from_obj365(
    #                 model_without_ddp.state_dict()[modify_key_to_load],
    #                 checkpoint['model'][modify_key_to_load]
    #             )
    #     model_without_ddp.load_state_dict(checkpoint['model'], strict=False)
    #     if args.use_ema:
    #         del ema_m
    #         ema_m = ModelEma(model_without_ddp)
    if args.use_ema:
        del ema_m
        ema_m = ModelEma(model_without_ddp, decay=args.ema_decay)

    output_dir = Path(args.output_dir)
    summary_log_path = output_dir / 'summary.log' if args.output_dir else None
    best_map_holder = BestMetricHolder(use_ema=args.use_ema)
    best_adds_score = 0.0
    if summary_log_path is not None and utils.is_main_process():
        write_summary_run_config(summary_log_path, args, criterion)

    is_training_run = not (args.eval or args.pose_eval_only or args.eval_bop)
    resume_checkpoint = None
    resume_epoch = None
    
    if args.resume:
        resume_checkpoint = torch.load(args.resume, map_location='cpu')
        model_without_ddp.load_state_dict(clean_state_dict(resume_checkpoint['model']), strict=True)
        resume_epoch = resume_checkpoint.get('epoch')
        if resume_epoch is not None:
            resume_epoch = int(resume_epoch)
        if args.use_ema:
            if 'ema_model' in resume_checkpoint:
                ema_m.module.load_state_dict(clean_state_dict(resume_checkpoint['ema_model']))
            else:
                ema_m.set(model_without_ddp)
        if is_training_run:
            if 'optimizer' in resume_checkpoint:
                optimizer.load_state_dict(resume_checkpoint['optimizer'])
            else:
                print('Warning: checkpoint missing optimizer state; resuming weights only.')

            if 'lr_scheduler' in resume_checkpoint:
                try:
                    lr_scheduler.load_state_dict(resume_checkpoint['lr_scheduler'])
                except (KeyError, TypeError, ValueError):
                    if resume_epoch is not None:
                        lr_scheduler.last_epoch = resume_epoch
                        if hasattr(lr_scheduler, '_step_count'):
                            lr_scheduler._step_count = resume_epoch + 1
                        if hasattr(lr_scheduler, '_last_lr'):
                            lr_scheduler._last_lr = [group['lr'] for group in optimizer.param_groups]
                    print('Warning: checkpoint lr_scheduler state is incompatible; reconstructed scheduler epoch from checkpoint.')
            elif resume_epoch is not None:
                lr_scheduler.last_epoch = resume_epoch
                if hasattr(lr_scheduler, '_step_count'):
                    lr_scheduler._step_count = resume_epoch + 1
                if hasattr(lr_scheduler, '_last_lr'):
                    lr_scheduler._last_lr = [group['lr'] for group in optimizer.param_groups]
                print('Warning: checkpoint missing lr_scheduler state; reconstructed scheduler epoch from checkpoint.')

            if resume_epoch is not None:
                args.start_epoch = resume_epoch + 1
                set_scheduler_epoch_lrs(lr_scheduler, args.start_epoch)

            restore_best_metric_holder(best_map_holder, resume_checkpoint.get('best_map_holder', {}))
            best_adds_score = float(resume_checkpoint.get('best_adds_score', resume_checkpoint.get('score', 0.0)))

            if utils.is_main_process() and resume_epoch is not None:
                print(f'Resumed training from checkpoint epoch {resume_epoch} (next epoch: {args.start_epoch}).')

    if utils.is_main_process():
        print("Get benchmark")
        benchmark_model = copy.deepcopy(model_without_ddp)
        bm = benchmark(benchmark_model.float(), dataset_val, output_dir)
        print(json.dumps(bm, indent=2))
        del benchmark_model

    eval_epoch = resume_epoch

    test_stats, coco_evaluator = {}, None
    if args.eval and not args.pose_eval_only:
        if args.skip_coco_eval:
            print("Skipping COCO Eval (--skip_coco_eval).")
        else:
            print("COCO Eval.")
            test_stats, coco_evaluator = evaluate(
                model, criterion, postprocessors, data_loader_val, base_ds, device, args)
            if args.output_dir:
                utils.save_on_master(coco_evaluator.coco_eval["bbox"].eval, output_dir / "eval.pth")
            if summary_log_path is not None and utils.is_main_process():
                append_summary_log(
                    summary_log_path,
                    build_coco_eval_summary(test_stats, args.eval_set, phase='eval_only', epoch=eval_epoch),
                )
    
    if args.eval or args.pose_eval_only:
        print("Pose Eval.")
        if utils.is_main_process():
            pose_loader = data_loader_pose_val if data_loader_pose_val is not None else data_loader_val
            pose_eval_results = pose_evaluate(model=model, 
                        matcher=matcher, 
                        pose_evaluator=pose_evaluator,
                        data_loader=pose_loader, 
                        image_set=args.eval_set, bbox_mode=args.bbox_mode,
                        quick_mode=args.quick_eval, 
                        device=device, output_dir=args.output_dir, epoch=eval_epoch)
            if summary_log_path is not None:
                append_summary_log(
                    summary_log_path,
                    build_pose_eval_summary(
                        pose_eval_results,
                        args.eval_set,
                        args.bbox_mode,
                        phase='eval_only',
                        quick_mode=args.quick_eval,
                        epoch=eval_epoch,
                    ),
                )
        if args.distributed:
            torch.distributed.barrier()
        return
    # Evaluate the model for the BOP challenge
    if args.eval_bop:
        print(args.dataset)
        bop_evaluate(model, matcher, data_loader_val, args.eval_set, args.bbox_mode,
                     args.rotation_representation, device, args.output_dir, args.dataset)
        return


    # for drop
    total_batch_size = args.batch_size * utils.get_world_size()
    if utils.is_main_process():
        print(f"Batch config: per_gpu={args.batch_size}, world_size={utils.get_world_size()}, total={total_batch_size}")
    num_training_steps_per_epoch = (len(dataset_train) + total_batch_size - 1) // total_batch_size
    schedules = {}
    if args.dropout > 0:
        schedules['do'] = drop_scheduler(
            args.dropout, args.epochs, num_training_steps_per_epoch,
            args.cutoff_epoch, args.drop_mode, args.drop_schedule)
        print("Min DO = %.7f, Max DO = %.7f" % (min(schedules['do']), max(schedules['do'])))

    if args.drop_path > 0:
        schedules['dp'] = drop_scheduler(
            args.drop_path, args.epochs, num_training_steps_per_epoch,
            args.cutoff_epoch, args.drop_mode, args.drop_schedule)
        print("Min DP = %.7f, Max DP = %.7f" % (min(schedules['dp']), max(schedules['dp'])))
    if args.start_epoch >= args.epochs:
        raise ValueError(f"start_epoch ({args.start_epoch}) should be less than epochs ({args.epochs})")
    print("Start training")
    start_time = time.time()
    if args.skip_coco_eval and utils.is_main_process():
        print("Skipping COCO eval during training (--skip_coco_eval).")
    for epoch in range(args.start_epoch, args.epochs):
        epoch_start_time = time.time()
        if args.distributed:
            sampler_train.set_epoch(epoch)
        train_stats = train_one_epoch(
                                    model, 
                                    criterion, 
                                    data_loader_train, 
                                    optimizer, 
                                    device, 
                                    epoch,
                                    args.clip_max_norm, 
                                    ema_m=ema_m, 
                                    schedules=schedules, 
                                    num_training_steps_per_epoch=num_training_steps_per_epoch,
                                    vit_encoder_num_layers=args.vit_encoder_num_layers, 
                                    args=args, 
                                    writer=writer)
        # TensorBoard logging
        # Per-epoch scalars
        if writer:
            # Global losses (reduced)
            for k, v in train_stats.items():
                if isinstance(v, (int, float)):
                    writer.add_scalar(f"train/{k}", v, epoch)
            # Individual loss components
            for k, v in train_stats.get("loss_components", {}).items():
                writer.add_scalar(f"train/{k}", v, epoch)

        train_epoch_time = time.time() - epoch_start_time
        train_epoch_time_str = str(datetime.timedelta(seconds=int(train_epoch_time)))
        
        lr_scheduler.step()

        if args.skip_coco_eval:
            test_stats, coco_evaluator = {}, None
        else:
            test_stats, coco_evaluator = evaluate(
                model, criterion, postprocessors, data_loader_val, base_ds, device, args=args
            )
            if summary_log_path is not None and utils.is_main_process():
                append_summary_log(
                    summary_log_path,
                    build_coco_eval_summary(test_stats, args.eval_set, phase='train', epoch=epoch),
                )
            if 'coco_eval_bbox' in test_stats:
                map_regular = test_stats['coco_eval_bbox'][0]
                is_best_regular = best_map_holder.update(map_regular, epoch, is_ema=False)
                if is_best_regular and args.output_dir:
                    checkpoint_path = output_dir / 'checkpoint_best_regular.pth'
                    utils.save_on_master(
                        build_training_checkpoint(
                            args,
                            model_without_ddp.state_dict(),
                            optimizer,
                            lr_scheduler,
                            epoch,
                            ema_m=ema_m if args.use_ema else None,
                            best_metric_holder=best_map_holder,
                            best_adds_score=best_adds_score,
                        ),
                        checkpoint_path,
                    )
        
        # Adaptive pose evaluation schedule
        run_pose_eval = (
            not args.skip_pose_eval
            and should_run_pose_eval(epoch, args.epochs, args.warm_up_epochs)
        )

        if run_pose_eval:
            if utils.is_main_process():
                # Keep scheduled validation cheap during training, but always run the final epoch in full.
                quick_mode = args.quick_eval and (epoch + 1) != args.epochs
                pose_loader = data_loader_pose_val if data_loader_pose_val is not None else data_loader_val
                pose_eval_results = pose_evaluate(
                    model=model,
                    matcher=matcher,
                    pose_evaluator=pose_evaluator,
                    data_loader=pose_loader,
                    image_set=args.eval_set,
                    bbox_mode=args.bbox_mode,
                    quick_mode=quick_mode,
                    device=device,
                    output_dir=args.output_dir,
                    epoch=epoch,
                    
                )
                current_add_score = pose_eval_results['ADD']
                current_adi_score = pose_eval_results['ADI']
                current_adds_score = pose_eval_results['ADD_minus_S']
                current_avg_translation_error = pose_eval_results['avg_translation_error']
                current_avg_rotation_error = pose_eval_results['avg_rotation_error']
                current_avg_rotation_error_symmetry_aware = pose_eval_results['avg_rotation_error_symmetry_aware']
                current_avg_rotation_error_nonsymmetric_only = pose_eval_results['avg_rotation_error_nonsymmetric_only']
                print(f"Epoch {epoch} Validation ADD(-S): {current_adds_score:.2f}%")

                # TensorBoard logging for pose metrics
                if writer:
                    writer.add_scalar("val/pose_ADD", current_add_score, epoch)
                    writer.add_scalar("val/pose_ADI", current_adi_score, epoch)
                    writer.add_scalar("val/pose_ADD_minus_S", current_adds_score, epoch)
                    if current_avg_translation_error is not None:
                        writer.add_scalar("val/pose_avg_translation_error", current_avg_translation_error, epoch)
                    if current_avg_rotation_error is not None:
                        writer.add_scalar("val/pose_avg_rotation_error", current_avg_rotation_error, epoch)
                    if current_avg_rotation_error_symmetry_aware is not None:
                        writer.add_scalar(
                            "val/pose_avg_rotation_error_symmetry_aware",
                            current_avg_rotation_error_symmetry_aware,
                            epoch,
                        )
                    if current_avg_rotation_error_nonsymmetric_only is not None:
                        writer.add_scalar(
                            "val/pose_avg_rotation_error_nonsymmetric_only",
                            current_avg_rotation_error_nonsymmetric_only,
                            epoch,
                        )
                
                # Save best model by the mixed ADD(-S) validation score.
                if current_adds_score > best_adds_score:
                    best_adds_score = current_adds_score
                    print("🚀 New Best Model found! Saving checkpoint...")
                    checkpoint_path = output_dir / 'checkpoint_best_adds.pth' if args.output_dir else Path('checkpoint_best_adds.pth')
                    utils.save_on_master(
                        build_training_checkpoint(
                            args,
                            model_without_ddp.state_dict(),
                            optimizer,
                            lr_scheduler,
                            epoch,
                            ema_m=ema_m if args.use_ema else None,
                            best_metric_holder=best_map_holder,
                            best_adds_score=best_adds_score,
                            extra_payload={'score': float(best_adds_score)},
                        ),
                        checkpoint_path,
                    )
                if summary_log_path is not None:
                    append_summary_log(
                        summary_log_path,
                        build_pose_eval_summary(
                            pose_eval_results,
                            args.eval_set,
                            args.bbox_mode,
                            phase='train',
                            quick_mode=quick_mode,
                            epoch=epoch,
                        ),
                    )
            if args.distributed:
                torch.distributed.barrier()
        if writer:
            # Validation metrics
            for k, v in test_stats.items():
                if isinstance(v, (list, tuple)):
                    # e.g. coco_eval_bbox[0]
                    writer.add_scalar(f"val/{k}", v[0], epoch)
                elif isinstance(v, (int, float)):
                    writer.add_scalar(f"val/{k}", v, epoch)

        ema_test_stats = None
        if args.use_ema and not args.skip_coco_eval:
            ema_test_stats, _ = evaluate(
                ema_m.module, criterion, postprocessors, data_loader_val, base_ds, device, args=args
            )
            if summary_log_path is not None and utils.is_main_process():
                append_summary_log(
                    summary_log_path,
                    build_coco_eval_summary(
                        ema_test_stats,
                        args.eval_set,
                        phase='train_ema',
                        epoch=epoch,
                        eval_name='coco_ema',
                    ),
                )
            if 'coco_eval_bbox' in ema_test_stats:
                map_ema = ema_test_stats['coco_eval_bbox'][0]
                is_best_ema = best_map_holder.update(map_ema, epoch, is_ema=True)
                if is_best_ema and args.output_dir:
                    checkpoint_path = output_dir / 'checkpoint_best_ema.pth'
                    utils.save_on_master(
                        build_training_checkpoint(
                            args,
                            ema_m.module.state_dict(),
                            optimizer,
                            lr_scheduler,
                            epoch,
                            ema_m=ema_m,
                            best_metric_holder=best_map_holder,
                            best_adds_score=best_adds_score,
                        ),
                        checkpoint_path,
                    )

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     **{f'test_{k}': v for k, v in test_stats.items()},
                     'epoch': epoch,
                     'n_parameters': n_parameters,
                     'skip_coco_eval': args.skip_coco_eval}

        if ema_test_stats is not None:
            log_stats.update({f'ema_test_{k}': v for k, v in ema_test_stats.items()})
        if not args.skip_coco_eval:
            log_stats.update(best_map_holder.summary())

        log_stats.update({
            'now_time': str(datetime.datetime.now()),
            'train_epoch_time': train_epoch_time_str,
        })
        epoch_time = time.time() - epoch_start_time
        epoch_time_str = str(datetime.timedelta(seconds=int(epoch_time)))
        log_stats['epoch_time'] = epoch_time_str

        if args.output_dir and utils.is_main_process():
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")

            if coco_evaluator is not None:
                (output_dir / 'eval').mkdir(exist_ok=True)
                if "bbox" in coco_evaluator.coco_eval:
                    filenames = ['latest.pth']
                    if epoch % 50 == 0:
                        filenames.append(f'{epoch:03}.pth')
                    for name in filenames:
                        torch.save(coco_evaluator.coco_eval["bbox"].eval,
                                   output_dir / "eval" / name)

        if args.output_dir:
            checkpoint_paths = [output_dir / 'checkpoint.pth']
            lr_drop_checkpoint = any(
                (epoch + 1) % drop_epoch == 0
                for drop_epoch in {args.lr_drop, pose_lr_drop}
                if drop_epoch > 0
            )
            if lr_drop_checkpoint or (epoch + 1) % args.checkpoint_interval == 0:
                checkpoint_paths.append(output_dir / f'checkpoint{epoch:04}_new.pth')
            for checkpoint_path in checkpoint_paths:
                utils.save_on_master(
                    build_training_checkpoint(
                        args,
                        model_without_ddp.state_dict(),
                        optimizer,
                        lr_scheduler,
                        epoch,
                        ema_m=ema_m if args.use_ema else None,
                        best_metric_holder=best_map_holder,
                        best_adds_score=best_adds_score,
                    ),
                    checkpoint_path,
                )

    if writer:
        writer.close()

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('LWDETR training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()

    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    if args.subcommand is None:
        main(args)
    elif args.subcommand == 'export_model':
        from deploy.export import main
        if args.batch_size != 1:
            args.batch_size = 1
            print(f"Only batch_size 1 is supported for onnx export, \
                 but got batchsize = {args.batch_size}. batch_size is forcibly set to 1.")
        main(args)
