#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.
"""
Data augmentation functionality. Passed as callable transformations to
Dataset classes.

The data augmentation procedures were interpreted from @weiliu89's SSD paper
http://arxiv.org/abs/1512.02325
"""

import math
import random
from matplotlib import image
from torchvision.io import write_png
import cv2
import numpy as np
import torch
from util.box_ops import box_xyxy_to_cxcywh
from util.quaternion_ops import rot2quat
from util.rotation_utils import (
    get_sarr_symmetry_vectors,
    rotation_matrix_to_gram_schmidt_6d,
    rotation_matrix_to_sarr,
)
import data_utils.transforms as T
from PIL import Image
from pathlib import Path
from util.utils import save_annotated_image, camera_params_to_K, pad_to_size
from util.visualize_object_pose import visualize_object_keypoints, YCBVVisualizer, save_image_with_bboxes, draw_6d_pose  
DEBUG = False
DEBUG_OUT=Path("debug")
CAD_MODELS = Path("/workspace/LWDETR/data/datasets/bop/models")


def draw_object_centers(img_input, centers_xy, radius=4, color=(0, 255, 0)):
    """
    Draw circles for object centers.
    Accepts:
      - torch.Tensor (C,H,W) in [0,1]
      - PIL.Image
      - np.ndarray (H,W,C) uint8 RGB or BGR
    centers_xy: (N,2) pixel coords (u,v)
    Returns BGR uint8 ndarray.
    """
    if centers_xy is None or len(centers_xy) == 0:
        # Just convert and return
        if isinstance(img_input, torch.Tensor):
            img = (img_input.permute(1,2,0).numpy().clip(0,1) * 255).astype(np.uint8)
        elif isinstance(img_input, Image.Image):
            img = np.array(img_input)
        else:
            img = img_input
        return img[..., ::-1].copy()  # BGR
    # Normalize input to RGB uint8
    if isinstance(img_input, torch.Tensor):
        img = (img_input.permute(1,2,0).numpy().clip(0,1) * 255).astype(np.uint8)
    elif isinstance(img_input, Image.Image):
        img = np.array(img_input)
    else:
        img = img_input
    # If grayscale expand
    if img.ndim == 2:
        img = np.repeat(img[:, :, None], 3, axis=2)
    # Assume current img is RGB; convert to BGR for OpenCV drawing
    img_bgr = img[..., ::-1].copy()
    for (u, v) in centers_xy.tolist():
        cv2.circle(img_bgr, (int(round(u)), int(round(v))), radius, color, -1, lineType=cv2.LINE_AA)
    return img_bgr

def augment_hsv(img, hgain=5, sgain=30, vgain=30):
    hsv_augs = np.random.uniform(-1, 1, 3) * [hgain, sgain, vgain]  # random gains
    hsv_augs *= np.random.randint(0, 2, 3)  # random selection of h, s, v
    hsv_augs = hsv_augs.astype(np.int16)
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.int16)

    img_hsv[..., 0] = (img_hsv[..., 0] + hsv_augs[0]) % 180
    img_hsv[..., 1] = np.clip(img_hsv[..., 1] + hsv_augs[1], 0, 255)
    img_hsv[..., 2] = np.clip(img_hsv[..., 2] + hsv_augs[2], 0, 255)

    cv2.cvtColor(img_hsv.astype(img.dtype), cv2.COLOR_HSV2BGR, dst=img)  # no return needed


def get_aug_params(value, center=0):
    if isinstance(value, (int, float)):
        return random.uniform(center - value, center + value)
    elif len(value) == 2:
        return random.uniform(value[0], value[1])
    else:
        raise ValueError(
            "Affine params should be either a sequence containing two values\
                          or single float values. Got {}".format(
                value
            )
        )

def get_affine_matrix(
    target_size,
    degrees=10,
    translate=0.1,
    scales=0.1,
    shear=10,
    camera_matrix=None
):
    theight,twidth = target_size

    # Rotation and Scale
    angle = get_aug_params(degrees)
    scale = get_aug_params(scales, center=1.0)

    if scale <= 0.0:
        raise ValueError("Argument scale should be positive")
    
    center = (camera_matrix['cx'], camera_matrix['cy'])
    R = cv2.getRotationMatrix2D(angle=angle, center=center, scale=scale) #Rotate around the principle axis
    M = np.ones([2, 3])
    # Shear
    shear_x = math.tan(get_aug_params(shear) * math.pi / 180)
    shear_y = math.tan(get_aug_params(shear) * math.pi / 180)

    M[0] = R[0] + shear_y * R[1]
    M[1] = R[1] + shear_x * R[0]

    # Translation
    translation_x = get_aug_params(translate) * twidth  # x translation (pixels)
    translation_y = get_aug_params(translate) * theight  # y translation (pixels)

    M[0, 2] += translation_x
    M[1, 2] += translation_y

    return M, scale, angle


def apply_affine_to_bboxes(targets, target_size, M):
    num_gts = len(targets)

    # warp corner points
    twidth, theight = target_size
    corner_points = np.ones((4 * num_gts, 3))
    corner_points[:, :2] = targets[:, [0, 1, 2, 3, 0, 3, 2, 1]].reshape(
        4 * num_gts, 2
    )  # x1y1, x2y2, x1y2, x2y1
    corner_points = corner_points @ M.T  # apply affine transform
    corner_points = corner_points.reshape(num_gts, 8)

    # create new boxes
    corner_xs = corner_points[:, 0::2]
    corner_ys = corner_points[:, 1::2]
    new_bboxes = (
        np.concatenate(
            (corner_xs.min(1), corner_ys.min(1), corner_xs.max(1), corner_ys.max(1))
        )
        .reshape(4, num_gts)
        .T
    )

    # clip boxes
    new_bboxes[:, 0::2] = new_bboxes[:, 0::2].clip(0, twidth)
    new_bboxes[:, 1::2] = new_bboxes[:, 1::2].clip(0, theight)

    targets[:, :4] = new_bboxes

    return targets

def apply_affine_to_kpts(targets, target_size, M, scale, num_kpts=17):
    num_gts = len(targets)
    # warp corner points
    twidth, theight = target_size
    xy_kpts = np.ones((num_gts * num_kpts, 3))
    xy_kpts[:, :2] = targets[:, 5:].reshape(num_gts * num_kpts, 2)  # num_kpt is hardcoded to 17
    xy_kpts = xy_kpts @ M.T  # transform
    xy_kpts = xy_kpts[:, :2].reshape(num_gts, num_kpts*2)  # perspective rescale or affine
    xy_kpts[targets[:, 5:] == 0] = 0
    x_kpts = xy_kpts[:, list(range(0, num_kpts*2, 2))]
    y_kpts = xy_kpts[:, list(range(1, num_kpts*2, 2))]

    x_kpts[np.logical_or.reduce((x_kpts < 0, x_kpts > twidth, y_kpts < 0, y_kpts > theight))] = 0
    y_kpts[np.logical_or.reduce((x_kpts < 0, x_kpts > twidth, y_kpts < 0, y_kpts > theight))] = 0
    xy_kpts[:, list(range(0, num_kpts*2, 2))] = x_kpts
    xy_kpts[:, list(range(1, num_kpts*2, 2))] = y_kpts

    targets[:, 5:] = xy_kpts

    return targets


def apply_affine_to_object_pose(obj_center_2d, 
                                translation, 
                                rotation,
                                K,
                                M, scale, angle, im_size):
    # Number of bboxes in targets, also the number of annotations in the current sample.
    
    obj_center_2d[:,0] = obj_center_2d[:,0] * im_size[0]    # u = x * W
    obj_center_2d[:,1] = obj_center_2d[:,1] * im_size[1]   # v = y * H
    ones = np.ones((obj_center_2d.shape[0], 1), dtype=obj_center_2d.dtype)
    pts_h = np.concatenate([obj_center_2d, ones], axis=1)   # (N,3)
    obj_center_2d = pts_h @ M.T  # Apply 2x3 affine matrix M: (x',y') = M * (x,y,1).
    # transform depth
    # There is no change in depth for rotation around z axis. Scaling reduces depth by the amount of scaling
    translation = translation * 1000.0  # The dataset is in m convert back to mm.
    # Adjust depth (tz) inversely to image scaling.
    translation_z = translation[:, 2] / scale
    translation_x = translation[:, 0:1]  # (N,2)
    translation_y = translation[:, 1:2]
    fx = K[0,0]
    fy = K[1,1]
    cx = K[0,2]
    cy = K[1,2]
    
    # Apply Pinhole Formula to get x,y in meters
    translation_x = (obj_center_2d[:,0]  - cx) * translation_z / fx
    translation_y = (obj_center_2d[:,1] - cy) * translation_z / fy
    # Ensure column vectors (N,1) then concatenate to (N,3)
    translation_x = translation_x[:, None]   # (N,1)
    translation_y = translation_y[:, None]   # (N,1)
    translation_z = translation_z[:, None]   # (N,1)
    translation = np.concatenate([translation_x, translation_y, translation_z], axis=1)
    
    

    # Back to mm
    translation = translation / 1000.0
    
    deltaR = cv2.getRotationMatrix2D(angle=angle, center=(0, 0), scale=1.0)  # 2D (in-plane Z) rotation (2x3).
    deltaR = np.vstack((deltaR, np.array([[0, 0, 1.0]])) ) # Extend to 3x3 by adding last row [0,0,1].
    rotation = deltaR[None, :, :] @ rotation # Left-multiply: rotate pose about camera Z axis.
    
    return obj_center_2d, translation, rotation

def make_divisible_by_64(hw: tuple[int, int]) -> tuple[int, int]:
    h, w = hw
    if DEBUG:
        None
        #print(f"Input hw {hw}")
    def up(x: int) -> int:
        return ((x + 63) // 64) * 64  # ceiling to next 64-multiple#
    if DEBUG:
        None
        #print(f"Output hw {hw}")
    return up(h), up(w)


def random_affine_single(
    img: torch.Tensor,
    target: dict, 
    target_size: tuple[int, int], # hw, resize the image to this size. TODO: Decide if padding is better and leave the img size as is.
    camera_matrix: dict=None,
    degrees:int=10,
    translate:float=0.1,
    scales:float=0.1,
    shear:int=10,
    ):
    
    # make sure the image will be divisible by 64 after affine transformation for the ViT backbone
    # TODO: Decide if padding is better and leave the img size as is.
    #make_divisible_by_64(target_size)
    # TODO: I K matrix always the right one?
    K = target['intrinsics'][0].reshape(3, 3).float().numpy()
    M, scale, angle = get_affine_matrix(target_size, 
                                        degrees, 
                                        translate, 
                                        scales, 
                                        shear,                    
                                        camera_matrix)
  
    
    # to CHW
    img = img.permute(1,2,0) # to HWC
    img = img.float() * 255.0
    img = img.numpy()
    img = img[:, :, ::-1]  # RGB to BGR
    # Apply affine transformation onto image.
    img = cv2.warpAffine(img, M, 
                         dsize=(target_size[1],target_size[0]), 
                         borderValue=(114, 114, 114))
    if DEBUG:
        out = Path(DEBUG_OUT, "after_random_affine", "affine_rgb.jpg")
        cv2.imwrite(out, img)
    # Transform 2D bbox coordinates
    affine_boxes = np.zeros((0, 4), dtype=np.float32)
    if len(target['boxes']) > 0:
        width = target['orig_size'][1].item()
        height = target['orig_size'][0].item()
        lst_boxes_xyxy = []
        for bbox in target['boxes']:
            # BBoxes comes in normalized (0,1) and xywh coords format
            # We need absolute pixels xyxy coords for mosaic
            x_center, y_center, w, h = bbox
            x1 = (x_center - w/2) * width
            y1 = (y_center - h/2) * height
            x2 = (x_center + w/2) * width
            y2 = (y_center + h/2) * height
            lst_boxes_xyxy.append([x1.item(), y1.item(), x2.item(), y2.item()])

        affine_boxes= np.array(lst_boxes_xyxy, dtype=np.float32) # xyxy and absolute pixel coords
        affine_boxes = apply_affine_to_bboxes(affine_boxes, 
                                         (target_size[1],target_size[0]), 
                                         M)
        
    # ----drop degenerate boxes produced by affine/mosaic ----
    # 1. FORCE SYNCHRONIZATION
    # Ensure affine_boxes and target['labels'] have the same length BEFORE we start.
    # If they mismatch, the labels are usually the "source of truth" (you can't have a box without a label).
    min_len = min(len(affine_boxes), len(target['labels']))

    if len(affine_boxes) > min_len:
        affine_boxes = affine_boxes[:min_len]
        
    per_object_keys = [
        'boxes', 'labels', 'relative_position', 'relative_translation_z',
        'relative_rotation', 'relative_rotation_gs', 'relative_quaternions',
        'relative_rotation_sarr', 'sarr_sym_v', 'object_center_2d',
        'intrinsics', 'model_points', 'is_symmetric', 'diameter',
        'symmetry_transforms', 'area', 'iscrowd'
    ]

    # Sync all target fields to this minimum length immediately
    # This prevents the "skip" bug where mismatched fields were ignored
    for key in per_object_keys:
        if key in target:
            t = target[key]
            if isinstance(t, torch.Tensor) and t.shape[0] > min_len:
                target[key] = t[:min_len]
            elif isinstance(t, list) and len(t) > min_len:
                target[key] = t[:min_len]

    # 2. PREPARE TENSORS
    device = target['boxes'].device
    if not isinstance(affine_boxes, torch.Tensor):
        affine_boxes = torch.from_numpy(affine_boxes).to(device, dtype=torch.float32)

    # 3. CALCULATE KEEP MASK
    # Now keep has length == min_len, so it works for all fields
    w = affine_boxes[:, 2] - affine_boxes[:, 0]
    h = affine_boxes[:, 3] - affine_boxes[:, 1]
    keep = (w > 1.0) & (h > 1.0)
    # 4. FILTER UNIFORMLY
    if keep.any():
        # Filter metadata fields
        for key in per_object_keys:
            if key in target:
                t = target[key]
                if isinstance(t, torch.Tensor):
                    target[key] = t[keep]
                elif isinstance(t, list):
                    target[key] = [item for item, k in zip(t, keep) if bool(k.item())]

        # Process Boxes (Normalize and Save)
        kept_boxes = affine_boxes[keep]
        final_boxes = box_xyxy_to_cxcywh(kept_boxes)
        scale_tensor = torch.tensor(
            [img.shape[1], img.shape[0], img.shape[1], img.shape[0]],
            device=device, dtype=torch.float32
        )
        target['boxes'] = final_boxes / scale_tensor

        # Keep pose supervision consistent with the affine-warped image.
        if (
            'object_center_2d' in target
            and 'relative_position' in target
            and 'relative_rotation' in target
            and len(target['labels']) > 0
        ):
            obj_center_2d, affine_trans, affine_rot = apply_affine_to_object_pose(
                obj_center_2d=target['object_center_2d'].detach().cpu().numpy().copy(),
                rotation=target['relative_rotation'].detach().cpu().numpy().copy(),
                translation=target['relative_position'].detach().cpu().numpy().copy(),
                M=M,
                K=K,
                scale=scale,
                angle=angle,
                im_size=(target_size[1], target_size[0])
            )

            obj_center_2d = torch.from_numpy(obj_center_2d).to(device=device, dtype=torch.float32)
            obj_center_2d = obj_center_2d / torch.tensor([img.shape[1], img.shape[0]], device=device, dtype=torch.float32)
            target['object_center_2d'] = obj_center_2d
            target['relative_position'] = torch.from_numpy(affine_trans).to(device=device, dtype=torch.float32)
            target['relative_rotation'] = torch.from_numpy(affine_rot).to(device=device, dtype=torch.float32)
            target['relative_translation_z'] = target['relative_position'][:, 2] / 2.5

            if 'relative_rotation_gs' in target:
                target['relative_rotation_gs'] = rotation_matrix_to_gram_schmidt_6d(target['relative_rotation'])
            if 'relative_quaternions' in target:
                quat = rot2quat(target['relative_rotation'].detach().cpu().numpy())
                target['relative_quaternions'] = torch.from_numpy(quat).to(device=device, dtype=torch.float32)
            if 'relative_rotation_sarr' in target:
                sarr_sym_v = target.get('sarr_sym_v', get_sarr_symmetry_vectors(target['labels']))
                sarr_sym_v = sarr_sym_v.to(device=device, dtype=torch.long)
                target['sarr_sym_v'] = sarr_sym_v
                target['relative_rotation_sarr'] = rotation_matrix_to_sarr(
                    target['relative_rotation'],
                    sarr_sym_v,
                    clamp=True,
                )

    else:
        # Handle empty case (no valid boxes left)
        target['boxes'] = torch.zeros((0, 4), device=device)
        # Clear other fields
        for key in per_object_keys: 
            if key in target:
                t = target[key]
                if isinstance(t, torch.Tensor):
                    target[key] = t.new_zeros((0,) + t.shape[1:])
                else:
                    target[key] = []
    # ---- END new block ----
    if DEBUG: 
        debug_img = img[:,:,::-1].copy()  # BGR to RGB
        debug_img = torch.from_numpy(debug_img)
        debug_img = debug_img.permute(2,0,1) # to CHW
        # Mosaic image with 2d annotations after affine
        save_image_with_bboxes(
                img=debug_img,
                boxes=affine_boxes,
                labels=target['labels'],
                out_path=Path(DEBUG_OUT,"after_random_affine","bboxes_ids.png")
            )
            
        assert len(target['boxes']) == len(target['labels']), f"Mismatch: boxes {len(target['boxes'])} vs labels {len(target['labels'])}"
        if DEBUG:
            
            debug_img_2 = img.copy()
             
            debug_img_2 = draw_object_centers(debug_img_2, target['object_center_2d'])
            
            cv2.imwrite(Path(DEBUG_OUT, "after_random_affine", f"keypoints.png"), debug_img_2)

            # Visualize 3D bbox and cad model overlay
            viz = YCBVVisualizer(CAD_MODELS)
            K = camera_params_to_K(camera_matrix)
            vis_img = img
            vis_img = viz.visualize_single_image(vis_img, 
                                                annotations={'labels': target['labels'], 
                                                            "relative_position":target['relative_position'], 
                                                            "relative_rotation":target['relative_rotation'],
                                                            },
                                                            K=K,
                                                            show_mesh=True,
                                                            sample_points=5000)

            cv2.imwrite(Path(DEBUG_OUT, "after_random_affine", "vis3d.png"), 
                        vis_img) # Visualization of 3D bboxes and overalayed objects
    
    
    img_tensor = torch.from_numpy(img).to(torch.uint8).permute(2, 0, 1)  # CHW
    img_tensor = img_tensor[[2, 1, 0], :, :]
    img_tensor = pad_to_size(img=img_tensor, fill=0)
    if DEBUG:
        write_png(img_tensor, Path(DEBUG_OUT, "after_random_affine", "img_tensor.png"))
    img_tensor = img_tensor.float() / 255.0
    return img_tensor, target

def _mirror(image, boxes, prob=0.5, human_pose=False, object_pose=False, human_kpts=None, flip_index=None):
    _, width, _ = image.shape
    if random.random() < prob or object_pose:
        image = image[:, ::-1]
        boxes[:, 0::2] = width - boxes[:, 2::-2]
        if human_pose:
            human_kpts[:, 0::2] = (width - human_kpts[:, 0::2])*(human_kpts[:, 0::2]!=0)
            human_kpts[:, 0::2] = human_kpts[:, 0::2][:, flip_index]
            human_kpts[:, 1::2] = human_kpts[:, 1::2][:, flip_index]
    if human_pose:
        return image, boxes, human_kpts
    else:
        return image, boxes


def preproc(img, input_size, swap=(2, 0, 1)):
    if len(img.shape) == 3:
        padded_img = np.ones((input_size[0], input_size[1], 3), dtype=np.uint8) * 114
    else:
        padded_img = np.ones(input_size, dtype=np.uint8) * 114

    r = min(input_size[0] / img.shape[0], input_size[1] / img.shape[1])
    resized_img = cv2.resize(
        img,
        (int(img.shape[1] * r), int(img.shape[0] * r)),
        interpolation=cv2.INTER_LINEAR,
    ).astype(np.uint8)
    
    padded_img[: int(img.shape[0] * r), : int(img.shape[1] * r)] = resized_img

    padded_img = padded_img.transpose(swap)
    padded_img = np.ascontiguousarray(padded_img, dtype=np.float32)
    return padded_img, r


class TrainTransform:
    def __init__(self, 
                 max_labels=50, flip_prob=0.5, hsv_prob=1.0, 
                 object_pose = False, human_pose=False, flip_index=None, num_kpts=17):
        self.max_labels = max_labels
        self.flip_prob = flip_prob
        self.hsv_prob = hsv_prob
        self.object_pose = object_pose
        self.human_pose = human_pose
        self.flip_index = flip_index
        self.num_kpts = num_kpts
        if self.object_pose:
            self.target_size = 14  #5 + 9
        elif self.human_pose:
            self.target_size = (5+2*self.num_kpts)  # 5+ 2*17
        else:
            self.target_size = 5

    def __call__(self, image, targets, input_dim):
        boxes = targets[:, :4].copy()
        labels = targets[:, 4].copy()
        if self.object_pose:
            object_poses = targets[:, 5:14].copy()
        if self.human_pose:
            human_kpts = targets[:, 5:].copy()
        else:
            human_kpts = None
        if len(boxes) == 0:
            targets = np.zeros((self.max_labels, self.target_size), dtype=np.float32)
            image, r_o = preproc(image, input_dim)
            return image, targets

        image_o = image.copy()
        targets_o = targets.copy()
        height_o, width_o, _ = image_o.shape
        boxes_o = targets_o[:, :4]
        labels_o = targets_o[:, 4]
        if self.object_pose:
            object_poses_o = targets_o[:, 5:14]
        elif self.human_pose:
            human_kpts_o = targets_o[:, 5:]
        # bbox_o: [xyxy] to [c_x,c_y,w,h]
        boxes_o = box_xyxy_to_cxcywh(boxes_o)

        if random.random() < self.hsv_prob:
            augment_hsv(image)
        if self.human_pose:
            image_t, boxes, human_kpts = _mirror(image, boxes, self.flip_prob, human_pose=self.human_pose, object_pose=self.object_pose, human_kpts=human_kpts, flip_index=self.flip_index)
        elif self.object_pose:
            image_t, boxes = image, boxes
        else:
            image_t, boxes = _mirror(image, boxes, self.flip_prob)

        height, width, _ = image_t.shape
        
        image_t, r_ = preproc(image_t, input_dim)
        # boxes [xyxy] 2 [cx,cy,w,h]
        boxes = box_xyxy_to_cxcywh(boxes)
        boxes *= r_
        if self.human_pose:
            human_kpts *= r_


        mask_b = np.minimum(boxes[:, 2], boxes[:, 3]) > 1
        boxes_t = boxes[mask_b]
        labels_t = labels[mask_b]
        if self.object_pose:
            object_poses_t = object_poses[mask_b]
        elif self.human_pose:
            human_kpts_t = human_kpts[mask_b]

        if len(boxes_t) == 0:
            image_t, r_o = preproc(image_o, input_dim)
            boxes_o *= r_o
            boxes_t = boxes_o
            labels_t = labels_o
            if self.object_pose:
                object_poses_t = object_poses_o
            elif self.human_pose:
                human_kpts_t = human_kpts_o
                human_kpts_t *= r_o

        labels_t = np.expand_dims(labels_t, 1)

        if self.object_pose:
            targets_t = np.hstack((labels_t, boxes_t, object_poses_t))
        elif self.human_pose:
            targets_t = np.hstack((labels_t, boxes_t, human_kpts_t))
        else:
            targets_t = np.hstack((labels_t, boxes_t))
        padded_labels = np.zeros((self.max_labels, self.target_size))
        padded_labels[range(len(targets_t))[: self.max_labels]] = targets_t[
            : self.max_labels
        ]
        padded_labels = np.ascontiguousarray(padded_labels, dtype=np.float32)
        image_t = torch.from_numpy(image_t)
        return image_t, padded_labels


class ValTransform:
    """
    Defines the transformations that should be applied to test PIL image
    for input into the network

    dimension -> tensorize -> color adj

    Arguments:
        resize (int): input dimension to SSD
        rgb_means ((int,int,int)): average RGB of the dataset
            (104,117,123)
        swap ((int,int,int)): final order of channels

    Returns:
        transform (transform) : callable transform to be applied to test/val
        data
    """

    def __init__(self, swap=(2, 0, 1), legacy=False, visualize = False):
        self.swap = swap
        self.legacy = legacy
        self.visualize = visualize

    # assume input is cv2 img for now
    def __call__(self, img, res, input_size):
        img, _ = preproc(img, input_size, self.swap)
        if self.legacy:
            img = img[::-1, :, :].copy()
            img /= 255.0
            img -= np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
            img /= np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)

        return img, res
