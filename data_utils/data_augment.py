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
from torchvision.io import write_png
import cv2
import numpy as np
import torch
from util.box_ops import box_xyxy_to_cxcywh
from PIL import Image
from pathlib import Path
from util.utils import save_annotated_image, camera_params_to_K, pad_to_size
from util.visualize_object_pose import visualize_object_keypoints, YCBVVisualizer, save_image_with_bboxes, draw_6d_pose  
DEBUG = False
DEBUG_OUT=Path("debug")
CAD_MODELS = Path("/workspace/LWDETR/data/datasets/bop/models")

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
    if isinstance(value, float):
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




def apply_affine_to_object_pose(targets, 
                                rotation,
                                translation, 
                                target_size, # wh
                                M, 
                                scale, 
                                angle):
    # Number of bboxes in targets, also the number of annotations in the current sample.
    num_gts = len(targets)
    test = translation.copy()
    # warp object center points [tx, ty]
    twidth, theight = target_size
    # Transform translation
    target_trans = np.ones((num_gts, 3)) # Create homogeneous 2D point array (x,y,1) per object.
    translation = translation * 1000.0  # The dataset is in mm convert back to m.
    target_trans[:, :2] = translation[:, -3:-1] # Copy current 2D translation (tx, ty) from last 3 columns (assumed [tx,ty,tz]).
    target_trans = target_trans @ M.T  # Apply 2x3 affine matrix M: (x',y') = M * (x,y,1).
    translation[:, -3:-1] = target_trans[:, :2] # Copy back transformed 2D translation (tx', ty').
    ###########
    #transform Rotation
    ### This is not necessary we have the full rotation matrix already
    r1 = rotation[:, 0:1] # Extract first rotation column (3 numbers) and add axis for stacking.
    r2 = rotation[:, 1:2] # Extract second rotation column (next 3 numbers).
    r3 = np.cross(r1, r2, axis=2) # Recompute third column via cross product to form orthonormal basis.
    rotation_mat = np.concatenate((r1, r2, r3), axis=1)  # Build full 3x3 rotation per object.
    #########
    deltaR = cv2.getRotationMatrix2D(angle=angle, center=(0, 0), scale=1.0)  # 2D (in-plane Z) rotation (2x3).
    deltaR = np.vstack((deltaR, np.array([[0, 0, 1.0]])) ) # Extend to 3x3 by adding last row [0,0,1].
    rotation_mat = deltaR @ rotation_mat # Left-multiply: rotate pose about camera Z axis.
    rotation[:, 0:1] = rotation_mat[:, 0:1] # Store updated first rotation column back.
    rotation[:, 1:2] = rotation_mat[:, 1:2] # Store updated second rotation column back.
    # transform depth
    # There is no change in depth for rotation around z axis. Scaling reduces depth by the amount of scaling
    translation[:, 2] = translation[:, 2] / (scale)  # Adjust depth (tz) inversely to image scaling.
    # Back to mm
    translation = translation / 1000.0
    
    return translation, rotation

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
    make_divisible_by_64(target_size)
    M, scale, angle = get_affine_matrix(target_size, 
                                        degrees, 
                                        translate, 
                                        scales, 
                                        shear,                    
                                        camera_matrix)
    # M = np.array([[1.0, 0.0, 12.34587156],
    #             [0.0, 1.0, 73.34484229]], dtype=float)
    
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

        affine_boxes= np.array(lst_boxes_xyxy) # xyxy and absolute pixel coords
        affine_boxes = apply_affine_to_bboxes(affine_boxes, 
                                         (target_size[1],target_size[0]), 
                                         M)
        # TODO: Convert back to normalized xywh adn update target['boxes']
        affine_boxes= torch.from_numpy(affine_boxes)   
        if DEBUG: # Visualize after bbox affine trafos
            debug_img = img[:,:,::-1].copy()  # BGR to RGB
            debug_img = torch.from_numpy(debug_img)
            debug_img = debug_img.permute(2,0,1) # to CHW
            # Mosaic image with 2d annotations after affine
            save_image_with_bboxes(
                    img=debug_img,
                    boxes=affine_boxes,
                    labels=target['labels'],
                    out_path=Path(DEBUG_OUT,"after_random_affine",f"bboxes_ids.png")
                )
                      
        affine_trans, affine_rot = apply_affine_to_object_pose(targets=target["boxes"],
                                                                   rotation=target["relative_rotation"].numpy(),
                                                                   translation=target["relative_position"].numpy(),
                                                                   target_size=(target_size[1],target_size[0]),
                                                                   M=M, 
                                                                   scale=scale, 
                                                                   angle=angle)
        
        # img_cuboid, img_mask, img_2dod = draw_6d_pose(img=img, 
        #                                             data_list=target,
        #                                             camera_matrix=camera_matrix)


        if DEBUG:
            debug_img_2 = img.copy()
            debug_img_2, _ = visualize_object_keypoints(
                cam=camera_matrix,
                targets= {"relative_position":torch.from_numpy(affine_trans), 
                          "relative_rotation":torch.from_numpy(affine_rot),
                          'labels': target['labels'],
                           "intrinsics": target['intrinsics']
                          },
                image=debug_img_2,
                obj_infos_by_label=target['labels']
            )
            cv2.imwrite(Path(DEBUG_OUT, "after_random_affine", f"keypoints.png"), debug_img_2)

           # Visualize 3D bbox and cad model overlay
            viz = YCBVVisualizer(CAD_MODELS)
            K = camera_params_to_K(camera_matrix)
            vis_img = img
            vis_img = viz.visualize_single_image(vis_img, 
                                                annotations={'labels': target['labels'], 
                                                            "relative_position":affine_trans, 
                                                            "relative_rotation":affine_rot,
                                                            },
                                                            K=K,
                                                            show_mesh=True,
                                                            sample_points=5000)

            cv2.imwrite(Path(DEBUG_OUT, "after_random_affine", "vis3d.png"), 
                        vis_img) # Visualization of 3D bboxes and overalayed objects
    # Finally update the target and image_tensor and return
    target['relative_position'] = torch.from_numpy(affine_trans)
    target['relative_rotation'] = torch.from_numpy(affine_rot)
    
    img_tensor = torch.from_numpy(img).to(torch.uint8).permute(2, 0, 1)  # CHW
    img_tensor = img_tensor[[2, 1, 0], :, :]
    img_tensor = pad_to_size(img=img_tensor, fill=0)
    if DEBUG:
        write_png(img_tensor, Path(DEBUG_OUT, "after_random_affine", "img_tensor.png"))
    img_tensor = img_tensor.float() / 255.0
    return img_tensor, target

def random_affine(
    img,
    rel_pos,
    rel_rot,
    rel_quats,
    labels,
    targets=(), # These are only the bboxes, TODO: rename
    target_size=((512, 640)), # hw
    degrees=10,
    translate=0.1,
    scales=0.1,
    shear=10,
    camera_matrix=None,
    mosaic_obj_ids=None,
    mosaic_intrinsics=None, 
    mosaic_rots=None,
    mosaic_trans=None
    ):
    
    # make sure the image will be divisible by 64 after affine transformation for the ViT backbone
    make_divisible_by_64(target_size)
    if mosaic_intrinsics is not None:
        camera_matrix['cx'] = mosaic_intrinsics[0][2].item()
        camera_matrix['cy'] = mosaic_intrinsics[0][3].item()
    M, scale, angle = get_affine_matrix(target_size, 
                                        degrees, 
                                        translate, 
                                        scales, 
                                        shear, 
                                        camera_matrix)
    # Apply affine transformation
    img = cv2.warpAffine(img, M, 
                         dsize=(target_size[1],target_size[0]), 
                         borderValue=(114, 114, 114))
    if DEBUG:
        out = Path(DEBUG_OUT, "after_affine_trafo.jpg")
        Image.fromarray(img).save(out)
    # Transform label coordinates
    if len(targets) > 0:
        affine_boxes = apply_affine_to_bboxes(targets, 
                                         (target_size[1],target_size[0]), 
                                         M, scale)
        if DEBUG: # Visualize after bbox affine trafos
            mosaic_img_debug_out=torch.from_numpy(img).to(torch.uint8)
            mosaic_img_debug_out= mosaic_img_debug_out.permute(2,0,1) # to CHW
            mosaic_img_debug_out = mosaic_img_debug_out.float() / 255.0
            # Mosaic image with 2d annotations after affine
            save_annotated_image(image=img, 
                                    targets={'boxes':torch.tensor(affine_boxes)}, 
                                    output_path="after_affine_mosaic_bboxes.png", 
                                    is_bbbox_coords_normalized=False,  
                                    is_corrected_bbx_coords=True)
                      
        if mosaic_intrinsics is not None:
            affine_trans, affine_rot = apply_affine_to_object_pose(targets=targets,
                                                                   rotation=mosaic_rots,
                                                                   translation=mosaic_trans,
                                                                   target_size=(target_size[1],target_size[0]),
                                                                   M=M, 
                                                                   scale=scale, 
                                                                   angle=angle)
        if DEBUG:
           
            # This can vizualize the keypoints 
            debug_img, projections = visualize_object_keypoints(cam=camera_matrix, 
                                                                targets={'labels':torch.from_numpy(labels), 
                                                                        "relative_position":torch.from_numpy(affine_trans), # (N,3)
                                                                        "relative_rotation":torch.from_numpy(affine_rot), # (N,3,3)
                                                                        "intrinsics": mosaic_intrinsics # (N,4)
                                                                        }, 
                                                                image=img[..., ::-1].copy(), # BGR to RGB
                                                                obj_infos_by_label=labels)
            cv2.imwrite(Path(DEBUG_OUT, "after_affine_mosaic_keypoints.png"), debug_img) # Vis of bboxes with objects center keypoints
            #Visualize 3d cad
            viz = YCBVVisualizer(CAD_MODELS)
            K = camera_params_to_K(camera_matrix)
            vis_img = viz.visualize_single_image(img[..., ::-1].copy(), 
                                                 annotations={'labels':labels, 
                                                              "relative_position":affine_trans, 
                                                              "relative_rotation":affine_rot,
                                                              "intrinsics": mosaic_intrinsics # (N,4)
                                                              }, 
                                                              K=K, 
                                                              show_mesh=False, 
                                                              sample_points=10000)
            cv2.imwrite(Path(DEBUG_OUT, "after_affine_mosaic_vis3d.png"), vis_img) # Visualization of 3D bboxes and overalayed objects
            
    return img, targets

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
