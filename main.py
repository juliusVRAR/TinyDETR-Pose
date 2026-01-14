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
import random
import time
import ast
import copy
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

def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    # Learnining hyperparameters
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--lr_encoder', default=1.5e-6, type=float) 
    parser.add_argument('--lr_backbone', default=1e-6, type=float) 
    parser.add_argument('--lr_transformer', default=1e-5, type=float)
    parser.add_argument('--batch_size', default=2, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=12, type=int)
    parser.add_argument('--lr_drop', default=11, type=int)
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
    parser.add_argument('--set_cost_keypoint', default=10., type=float,
                        help="keypoint coefficient in the matching cost")
    parser.add_argument('--matcher_type', default='6d', choices=['6d', 'hungarian', 'yopo'], type=str,
                        help="Type of matcher to use, hungarian is the 3d match from lwdetr and will probably not work")
    
    # PoET Config
    parser.add_argument('--bbox_mode', default='backbone', type=str, choices=('gt', 'backbone', 'jitter'),
                        help='Defines which bounding boxes should be used for PoET to determine query embeddings.')
    parser.add_argument('--im_size', type=int, nargs=2, default=(512, 640)) # Must be divisible by 64
    parser.add_argument('--num_feature_levels', default=4, type=int, help='number of feature levels')
    parser.add_argument('--reference_points', default='bbox', type=str, choices=('bbox', 'learned'),
                        help='Defines whether the transformer reference points are learned or extracted from the bounding boxes')
    parser.add_argument('--class_mode', default='specific', type=str, choices=('agnostic', 'specific'),
                        help="Determine whether PoET ist trained class-specific or class-agnostic")
    parser.add_argument('--rotation_representation', default='6d', type=str, choices=('6d', 'quat', 'silho_quat'),
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
    parser.add_argument('--adds_loss_coef', default=1.0, type=float, help='Loss weighing parameter for the ADD-S metric. Active after warmup epochs.')
    parser.add_argument('--warm_up_epochs', default=15, type=int, help='Number of epochs before ADD-S loss multiplier is activated.')
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
    parser.add_argument('--checkpoint_interval', default=10, type=int,
                        help='epoch interval to save checkpoint')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='', type=str, 
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
    parser.add_argument('--eval_interval', type=int, default=10,
                        help="Epoch interval after which the current model is evaluated")
    parser.add_argument('--class_info', type=str, default='/annotations/classes.json',
                        help='path to .txt-file containing the class names')
    parser.add_argument('--models', type=str, default='/models_eval/',
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
        # After warmup, before 80%: every 10 epochs
        return (e - warmup_epochs) % 10 == 0

def main(args):
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
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module
    
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)
    param_dicts = get_param_dict(args, model_without_ddp)

    # TODO: Check if the other one is needed
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
    
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)
    
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
        ema_m = ModelEma(model_without_ddp)

    output_dir = Path(args.output_dir)
    
    if  utils.is_main_process():
        print("Get benchmark")
        benchmark_model = copy.deepcopy(model_without_ddp)
        bm = benchmark(benchmark_model.float(), dataset_val, output_dir)
        print(json.dumps(bm, indent=2))
        del benchmark_model
    
    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'], strict=True)
        if args.use_ema:
            if 'ema_model' in checkpoint:
                ema_m.module.load_state_dict(clean_state_dict(checkpoint['ema_model']))
            else:
                del ema_m
                ema_m = ModelEma(model) 
        if not args.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
            checkpoint['optimizer']["param_groups"] = optimizer.state_dict()["param_groups"]
            checkpoint['lr_scheduler'].pop("step_size")
            checkpoint['lr_scheduler'].pop("_last_lr")
            
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            args.start_epoch = checkpoint['epoch'] + 1

    if args.eval:
        print("COCO Eval.")
        test_stats, coco_evaluator = evaluate(
            model, criterion, postprocessors, data_loader_val, base_ds, device, args)
        if args.output_dir:
            utils.save_on_master(coco_evaluator.coco_eval["bbox"].eval, output_dir / "eval.pth")
    if args.resume:
        eval_epoch = checkpoint['epoch']
    
    if args.eval or args.pose_eval_only:
        print("Pose Eval.")
        pose_evaluate(model=model, 
                    matcher=matcher, 
                    pose_evaluator=pose_evaluator,
                    data_loader=data_loader_val, 
                    image_set=args.eval_set, bbox_mode=args.bbox_mode,
                    quick_mode=args.quick_eval, 
                    device=device, output_dir=args.output_dir, epoch=eval_epoch)
        return
    # Evaluate the model for the BOP challenge
    if args.eval_bop:
        print(args.dataset)
        bop_evaluate(model, matcher, data_loader_val, args.eval_set, args.bbox_mode,
                     args.rotation_representation, device, args.output_dir, args.dataset)
        return


    # for drop
    total_batch_size = args.batch_size * utils.get_world_size()
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
    assert args.start_epoch - args.epochs !=0, "start_epoch should be less than epochs"
    print("Start training")
    start_time = time.time()
    best_map_holder = BestMetricHolder(use_ema=args.use_ema)
    best_adds_score = 0.0
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
        
        if args.output_dir:
            checkpoint_paths = [output_dir / 'checkpoint.pth']
            # extra checkpoint before LR drop and every `checkpoint_interval` epochs
            if (epoch + 1) % args.lr_drop == 0 or (epoch + 1) % args.checkpoint_interval == 0:
                checkpoint_paths.append(output_dir / f'checkpoint{epoch:04}_new.pth')
            for checkpoint_path in checkpoint_paths:
                weights = {
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'args': args,
                }
                if args.use_ema:
                    weights.update({
                        'ema_model': ema_m.module.state_dict(),
                    })
                utils.save_on_master(weights, checkpoint_path)

        test_stats, coco_evaluator = evaluate(
            model, criterion, postprocessors, data_loader_val, base_ds, device, args=args
        )
        if args.resume:
            eval_epoch = checkpoint['epoch']
        else:
            eval_epoch = None
        
        # Adaptive pose evaluation schedule
        run_pose_eval = (
            not args.skip_pose_eval
            and should_run_pose_eval(epoch, args.epochs, args.warm_up_epochs)
        )

        if run_pose_eval and utils.is_main_process():
            # Last epoch: force full evaluation (ignore quick_eval flag)
            if (epoch + 1) == args.epochs:
                quick_mode = False
            else:
                quick_mode = True
            # ADD, ADD-S, ADD(-S)
            current_add_score, current_adi_score, current_adds_score, current_avg_translation_error, current_avg_rotation_error = pose_evaluate(
                model=model,
                matcher=matcher,
                pose_evaluator=pose_evaluator,
                data_loader=data_loader_val,
                image_set=args.eval_set,
                bbox_mode=args.bbox_mode,
                quick_mode=quick_mode,
                device=device,
                output_dir=args.output_dir,
                epoch=epoch,
                
            )
            print(f"Epoch {epoch} Validation ADD-S: {current_adds_score:.2f}%")

            # TensorBoard logging for pose metrics
            if writer:
                writer.add_scalar("val/pose_ADD", current_add_score, epoch)
                writer.add_scalar("val/pose_ADI", current_adi_score, epoch)
                writer.add_scalar("val/pose_ADDS", current_adds_score, epoch)
                writer.add_scalar("val/pose_avg_translation_error", current_avg_translation_error, epoch)
                writer.add_scalar("val/pose_avg_rotation_error", current_avg_rotation_error, epoch)
            
            # Save Best Model (maximize ADD-S)
            if current_adds_score > best_adds_score:
                best_adds_score = current_adds_score
                print("🚀 New Best Model found! Saving checkpoint...")
                torch.save(
                    {
                        'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'epoch': epoch,
                        'score': best_adds_score,
                    },
                    "checkpoint_best_adds.pth",
                )
        if writer:
            # Validation metrics
            for k, v in test_stats.items():
                if isinstance(v, (list, tuple)):
                    # e.g. coco_eval_bbox[0]
                    writer.add_scalar(f"val/{k}", v[0], epoch)
                elif isinstance(v, (int, float)):
                    writer.add_scalar(f"val/{k}", v, epoch)

    if writer:
        writer.close()
        map_regular = test_stats['coco_eval_bbox'][0]
        _isbest = best_map_holder.update(map_regular, epoch, is_ema=False)
        if _isbest:
            checkpoint_path = output_dir / 'checkpoint_best_regular.pth'
            utils.save_on_master({
                'model': model_without_ddp.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'epoch': epoch,
                'args': args,
            }, checkpoint_path)
        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     **{f'test_{k}': v for k, v in test_stats.items()},
                     'epoch': epoch,
                     'n_parameters': n_parameters}
        if args.use_ema:
            ema_test_stats, _ = evaluate(
                ema_m.module, criterion, postprocessors, data_loader_val, base_ds, device, args=args
            )
        if args.resume:
            eval_epoch = checkpoint['epoch']
        else:
            eval_epoch = None
            #pose_evaluate(model, matcher, pose_evaluator, data_loader_val, args.eval_set, args.bbox_mode,
             #         args.rotation_representation, device, args.output_dir, eval_epoch)
            log_stats.update({f'ema_test_{k}': v for k,v in ema_test_stats.items()})
            map_ema = ema_test_stats['coco_eval_bbox'][0]
            _isbest = best_map_holder.update(map_ema, epoch, is_ema=True)
            if _isbest:
                checkpoint_path = output_dir / 'checkpoint_best_ema.pth'
                utils.save_on_master({
                    'model': ema_m.module.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'args': args,
                }, checkpoint_path)
        log_stats.update(best_map_holder.summary())
        
        # epoch parameters
        ep_paras = {
                'epoch': epoch,
                'n_parameters': n_parameters
            }
        log_stats.update(ep_paras)
        try:
            log_stats.update({'now_time': str(datetime.datetime.now())})
        except:
            pass
        log_stats['train_epoch_time'] = train_epoch_time_str
        epoch_time = time.time() - epoch_start_time
        epoch_time_str = str(datetime.timedelta(seconds=int(epoch_time)))
        log_stats['epoch_time'] = epoch_time_str
        
        if args.output_dir and utils.is_main_process():
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")

            # for evaluation logs
            if coco_evaluator is not None:
                (output_dir / 'eval').mkdir(exist_ok=True)
                if "bbox" in coco_evaluator.coco_eval:
                    filenames = ['latest.pth']
                    if epoch % 50 == 0:
                        filenames.append(f'{epoch:03}.pth')
                    for name in filenames:
                        torch.save(coco_evaluator.coco_eval["bbox"].eval,
                                   output_dir / "eval" / name)

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
