# ------------------------------------------------------------------------
# LW-DETR
# Copyright (c) 2024 Baidu. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Conditional DETR
# Copyright (c) 2021 Microsoft. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Copied from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# ------------------------------------------------------------------------

"""
Train and eval functions used in main.py
"""
import math
import sys
from typing import Iterable
import datetime
import torch
import torchvision
import util.misc as utils
from datasets.coco_eval import CocoEvaluator
from pathlib import Path
import os
import numpy as np
import time
from evaluation_tools.metrics import get_src_permutation_idx, calc_rotation_error, calc_translation_error
DEBUG = False
DEBUG_OUT=Path("debug")

def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float=0,
                    ema_m: torch.nn.Module=None, schedules: dict={}, 
                    num_training_steps_per_epoch=None, 
                    vit_encoder_num_layers=None, 
                    args=None,
                    writer=None):
    model.train()
    criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10
    start_steps=epoch * num_training_steps_per_epoch

    for data_iter_step, (samples, targets) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        it = start_steps + data_iter_step
        if 'dp' in schedules:
            if args.distributed:
                model.module.update_drop_path(schedules['dp'][it], vit_encoder_num_layers)
            else:
                model.update_drop_path(schedules['dp'][it], vit_encoder_num_layers)
        if 'do' in schedules:
            if args.distributed:
                model.module.update_dropout(schedules['do'][it])
            else:
                model.update_dropout(schedules['do'][it])
        
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        if DEBUG:
            if not os.path.exists(DEBUG_OUT):
                os.makedirs(DEBUG_OUT)
            nrow = len(samples.tensors)
            #filename = Path(DEBUG_OUT,f"batch_iter_{it}.png")
            filename = Path(DEBUG_OUT,f"batch.png")
            normalize = True
            if samples.mask is not None:
                # Invert so valid area is white, padding black (optional)
                mask = (~samples.mask).float().unsqueeze(1)  # [B, 1, H, W]
                mask_grid = torchvision.utils.make_grid(
                    mask, nrow=nrow, padding=2, normalize=False
                )
            mask_filename = Path(DEBUG_OUT, "batch_mask.png")
            torchvision.utils.save_image(mask_grid, mask_filename, nrow=nrow, padding=2)
            grid = torchvision.utils.make_grid(samples.tensors, nrow=nrow, padding=2, normalize=normalize)
            torchvision.utils.save_image(grid, filename, nrow=nrow, padding=2, normalize=normalize)
            # print(f"Grid saved as {filename}")
        outputs = model(samples, targets)

        # Update the weight in the criterion's dictionary BEFORE summing
        if epoch < args.warm_up_epochs:
            criterion.weight_dict['loss_adds'] = 0.0
        else:
            criterion.weight_dict['loss_adds'] = args.adds_loss_coef

        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

        loss_value = losses_reduced_scaled.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        if False:
            unused = []
            for n,p in model.named_parameters():
                if p.requires_grad and p.grad is None:
                    unused.append(n)
            print("Unused this iter:", unused)

        if max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        
              
        optimizer.step()

        # TensorBoard logging
        if writer is not None and getattr(args, "tensorboard", False) and \
           data_iter_step % getattr(args, "tb_iter_freq", 100) == 0 and utils.is_main_process():
            global_step = epoch * len(data_loader) + data_iter_step
            writer.add_scalar("iter/loss_total", loss_value, global_step)
            for k, v in loss_dict_reduced_scaled.items():
                writer.add_scalar(f"iter/{k}", v.item(), global_step)
            for k, v in loss_dict_reduced_unscaled.items():
                writer.add_scalar(f"iter/{k}", v.item(), global_step)

        if ema_m is not None:
            if epoch >= 0:
                ema_m.update(model)
        metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
        metric_logger.update(class_error=loss_dict_reduced['class_error'])
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(model, criterion, postprocessors, data_loader, base_ds, device, args = None):
    model.eval()
    if args.fp16_eval:
        model.half()
    criterion.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Test:'

    iou_types = tuple(k for k in ('segm', 'bbox') if k in postprocessors.keys())
    coco_evaluator = CocoEvaluator(base_ds, iou_types)
    # DEBUG
    
    for samples, targets in metric_logger.log_every(data_loader, 10, header):
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        if args.fp16_eval:
            samples.tensors = samples.tensors.half()
        
        outputs = model(samples)

        if args.fp16_eval:
            for key in outputs.keys():
                if key == 'enc_outputs':
                    for sub_key in outputs[key].keys():
                        outputs[key][sub_key] = outputs[key][sub_key].float()
                elif key == 'aux_outputs':
                    for idx in range(len(outputs[key])):
                        for sub_key in outputs[key][idx].keys():
                            outputs[key][idx][sub_key] = outputs[key][idx][sub_key].float()
                else:
                    outputs[key] = outputs[key].float()

        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        metric_logger.update(loss=sum(loss_dict_reduced_scaled.values()),
                             **loss_dict_reduced_scaled,
                             **loss_dict_reduced_unscaled)
        metric_logger.update(class_error=loss_dict_reduced['class_error'])

        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        results = postprocessors['bbox'](outputs, orig_target_sizes)
        res = {target['image_id'].item(): output for target, output in zip(targets, results)}
        if coco_evaluator is not None:
            coco_evaluator.update(res)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    if coco_evaluator is not None:
        coco_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    if coco_evaluator is not None:
        coco_evaluator.accumulate()
        coco_evaluator.summarize()
    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    if coco_evaluator is not None:
        if 'bbox' in postprocessors.keys():
            stats['coco_eval_bbox'] = coco_evaluator.coco_eval['bbox'].stats.tolist()
        if 'segm' in postprocessors.keys():
            stats['coco_eval_masks'] = coco_evaluator.coco_eval['segm'].stats.tolist()
    return stats, coco_evaluator

@torch.no_grad()
def pose_evaluate(model, 
                  matcher, 
                  pose_evaluator, 
                  data_loader, 
                  image_set, 
                  bbox_mode,  
                  device, 
                  output_dir, 
                  quick_mode,
                  epoch=None):
    """
    Evaluate PoET on the whole dataset, calculate the evaluation metrics and store the final performance.
    """
    model.eval()
    matcher.eval()

    # Reset pose evaluator to be empty
    pose_evaluator.reset()

    # Check whether the evaluation folder exists, otherwise create it
    if epoch is not None:
        output_eval_dir = output_dir + "/eval_" + image_set + "_" + bbox_mode + "_" + str(epoch) + "/"
    else:
        output_eval_dir = output_dir + "/eval_" + image_set + "_" + bbox_mode + "/"
    Path(output_eval_dir).mkdir(parents=True, exist_ok=True)

    print(f"Process validation dataset (Quick Mode: {quick_mode}):")
    
    # Calculate total images to process for accurate progress bar
    total_len = len(data_loader)
    n_images = len(data_loader.dataset.ids)
    
    if quick_mode:
        # If quick mode, we essentially skip 90% of batches
        # Adjust n_images for accurate logging
        n_images = int(n_images * 0.1)

    bs = data_loader.batch_size
    start_time = time.time()
    processed_images = 0
    # TODO: Add quick mode with 10% of the val dataset for fast eval while training.
    for i, (samples, targets) in enumerate(data_loader):

        # Skip 9 out of every 10 batches
        if quick_mode and i % 10 != 0:
            continue

        batch_start_time = time.time()
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        outputs = model(samples, targets)
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs' and k != 'enc_outputs'}

        # Extract final predictions and store them
        indices = matcher(outputs_without_aux, targets)
        idx = get_src_permutation_idx(indices)

        pred_translations = outputs_without_aux["pred_translations"][idx].detach().cpu().numpy()
        pred_rotations = outputs_without_aux["pred_rotations"][idx].detach().cpu().numpy()


        tgt_translations = torch.cat([t['relative_position'][i] for t, (_, i) in zip(targets, indices)], dim=0).detach().cpu().numpy()
        tgt_rotations = torch.cat([t['relative_rotation'][i] for t, (_, i) in zip(targets, indices)], dim=0).detach().cpu().numpy()

        obj_classes_idx = torch.cat([t['labels'][i] for t, (_, i) in zip(targets, indices)], dim=0).detach().cpu().numpy()
        intrinsics = torch.cat([t['intrinsics'][i] for t, (_, i) in zip(targets, indices)], dim=0).detach().cpu().numpy()
        img_files = [data_loader.dataset.coco.loadImgs(t["image_id"].item())[0]['file_name'] for t, (_, i) in zip(targets, indices) for _ in range(0, len(i))]

        # Iterate over all predicted objects and save them in the pose evaluator
        for cls_idx, img_file, intrinsic, pred_translation, pred_rotation, tgt_translation, tgt_rotation in \
                zip(obj_classes_idx, img_files, intrinsics, pred_translations, pred_rotations, tgt_translations, tgt_rotations):
            cls = pose_evaluator.classes[cls_idx - 1]
            pose_evaluator.poses_pred[cls].append(
                np.concatenate((pred_rotation, pred_translation.reshape(3, 1)), axis=1))
            pose_evaluator.poses_gt[cls].append(
                np.concatenate((tgt_rotation, tgt_translation.reshape(3, 1)), axis=1))
            pose_evaluator.poses_img[cls].append(img_file)
            pose_evaluator.num[cls] += 1
            pose_evaluator.camera_intrinsics[cls].append(intrinsic)

        batch_total_time = time.time() - batch_start_time
        batch_total_time_str = str(datetime.timedelta(seconds=int(batch_total_time)))
        processed_images = processed_images + len(targets)
        # Logic to estimate remaining batches correctly
        remaining_images = max(0, n_images - processed_images)
        remaining_batches = remaining_images / bs
        eta = batch_total_time * remaining_batches
        eta_str = str(datetime.timedelta(seconds=int(eta)))
        print("Processed {}/{} \t Batch Time: {} \t ETA: {}".format(processed_images, n_images, batch_total_time_str, eta_str))
    # At this point iterated over all validation images and for each object the result is fed into the pose evaluator
    total_time = time.time() - start_time
    # Avoid division by zero if n_images is 0 (rare edge case)
    n_images = max(1, n_images)
    time_per_img = total_time / n_images
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    time_per_img_str = str(datetime.timedelta(seconds=int(time_per_img)))
    print("Network Processing Time\nTotal Time: {}\t\tImages: {}\t\ts/img: {}".format(total_time_str, n_images, time_per_img_str))
    print("Start results evaluation")
    start_time = time.time()
    print("Start Calculating ADD")
    results_add = pose_evaluator.evaluate_pose_adds(output_eval_dir)

    print("Start Calculating ADD-S")
    results_adi= pose_evaluator.evaluate_pose_adi(output_eval_dir)

    print("Start Calculating ADD(-S)")
    results_adds = pose_evaluator.evaluate_pose_adds(output_eval_dir)
    print("Start Calculating Average Translation Error")
    pose_evaluator.calculate_class_avg_translation_error(output_eval_dir)
    print("Start Calculating Average Rotation Error")
    pose_evaluator.calculate_class_avg_rotation_error(output_eval_dir)
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("Evaluation time: {}".format(total_time_str))
    return results_add, results_adi, results_adds # ADD, ADD-S, ADD(-S)

@torch.no_grad()
def bop_evaluate(model, matcher, data_loader, image_set, bbox_mode, rotation_mode, device, output_dir):
    """
    Evaluate PoET on the dataset and store the results in the BOP format
    """
    model.eval()
    matcher.eval()

    output_eval_dir = output_dir + "/bop_" + bbox_mode + "/"
    Path(output_eval_dir).mkdir(parents=True, exist_ok=True)

    out_csv_file = open(output_eval_dir + 'ycbv.csv', 'w')
    out_csv_file.write("scene_id,im_id,obj_id,score,R,t,time")
    n_images = len(data_loader.dataset.ids)

    # CSV format: scene_id, im_id, obj_id, score, R, t, time
    counter = 1
    for samples, targets in data_loader:
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        pred_start_time = time.time()
        outputs, n_boxes_per_sample = model(samples, targets)
        pred_end_time = time.time() - pred_start_time
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs' and k != 'enc_outputs'}

        indices = matcher(outputs_without_aux, targets, n_boxes_per_sample)
        idx = get_src_permutation_idx(indices)

        pred_translations = outputs_without_aux["pred_translation"][idx].detach().cpu().numpy()
        pred_rotations = outputs_without_aux["pred_rotation"][idx].detach().cpu().numpy()

        obj_classes_idx = torch.cat([t['labels'][i] for t, (_, i) in zip(targets, indices)],
                                    dim=0).detach().cpu().numpy()

        img_files = [data_loader.dataset.coco.loadImgs(t["image_id"].item())[0]['file_name'] for t, (_, i) in
                     zip(targets, indices) for _ in range(0, len(i))]

        for cls_idx, img_file, pred_translation, pred_rotation in zip(obj_classes_idx, img_files, pred_translations, pred_rotations):
            file_info = img_file.split("/")
            scene_id = int(file_info[1])
            img_id = int(file_info[3][:file_info[3].rfind(".")])
            obj_id = cls_idx
            score = 1.0
            csv_str = "{},{},{},{},{} {} {} {} {} {} {} {} {}, {} {} {}, {}\n".format(scene_id, img_id, obj_id, score,
                                                                                    pred_rotation[0, 0], pred_rotation[0, 1], pred_rotation[0, 2],
                                                                                    pred_rotation[1, 0], pred_rotation[1, 1], pred_rotation[1, 2],
                                                                                    pred_rotation[2, 0], pred_rotation[2, 1], pred_rotation[2, 2],
                                                                                    pred_translation[0] * 1000, pred_translation[1] * 1000, pred_translation[2] * 1000,
                                                                                    pred_end_time)
            out_csv_file.write(csv_str)
        print("Processed {}/{}".format(counter, n_images))
        counter += 1

    out_csv_file.close()