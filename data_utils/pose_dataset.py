# ------------------------------------------------------------------------
# PoET: Pose Estimation Transformer for Single-View, Multi-Object 6D Pose Estimation
# Copyright (c) 2022 Thomas Jantos (thomas.jantos@aau.at), University of Klagenfurt - Control of Networked Systems (CNS). All Rights Reserved.
# Licensed under the BSD-2-Clause-License with no commercial use [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from Deformable DETR (https://github.com/fundamentalvision/Deformable-DETR)
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE_DEFORMABLE_DETR in the LICENSES folder for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

"""
Build a dataset for the pose estimation task. This includes loading the images and annotations consisting of
class, bounding box, relative pose and absolute poses. Moreover, data augmentation and bounding box pertubation is possible.
"""
import copy
from pathlib import Path

import torch


import numpy as np
import random
from pycocotools import mask as coco_mask
# Poet
from .torchvision_datasets import CocoDetection
# LWDETR
#from datasets.coco import CocoDetection
from util.misc import get_local_rank, get_local_size
from util.quaternion_ops import quat2rot, rot2quat
from util.rotation_utils import rotation_matrix_to_gram_schmidt_6d, rotation_matrix_to_raw_6d, rotation_6d_to_matrix, precompute_points
import data_utils.transforms as T
from scipy.stats import truncnorm
from PIL import Image
import torch.nn.functional as F
from torchvision.transforms import functional as TF
from torchvision.transforms.functional import to_pil_image
from util import box_ops
import cv2
from PIL import Image
from pathlib import Path
import json
#mosaicdetection
from data_utils.data_augment import random_affine, random_affine_single
from util.utils import save_annotated_image, camera_params_to_K, pad_to_size
from util.visualize_object_pose import visualize_object_keypoints, YCBVVisualizer, save_image_with_bboxes
DEBUG = False
DEBUG_OUT=Path("debug")

def load_json(path: str | Path):
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

class PoseDataset(CocoDetection):
    """
    Pose Estimation Dataset. Returns samples consisting of images and the target containing the class, bounding box and
    the pose.
    """
    def __init__(self, 
                 img_folder, 
                 ann_file, 
                 synthetic_background, 
                 transforms, 
                 return_masks, 
                 im_size,
                 camera, 
                 cad_models_path,
                 jitter=False,
                 jitter_probability=0.5, 
                 std=0.02, 
                 cache_mode=False, 
                 local_rank=0, local_size=1,
                 image_set='train',
                 use_mosaic=False,
                 model_symmetry=None,
                 class_info=None,
                 sample_mesh_points = False, # Only true if we calulate symmetries because we have CAD model information
                 n_mesh_points=1400, # The higher the more VRAM we need but the better the symmetry-aware loss works (T6D samples 1500 points)
                 mesh_point_seed=0
                 ):
        """
        Args:
            img_folder (string): path to the directory containing the images
            ann_file (string): path to the file containing the annotations
            synthetic_background (string): path to the directory containing the background images for synthetic images
            transforms (callable): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.ToTensor``
            return_masks (bool): Whether to include the segmentation mask
            jitter (bool): Apply jitter to the bounding box
            jitter_probability (float): Probability with which jitter is applied to the bounding box
            std (float): standard deviation of the jitter.
        """
        super(PoseDataset, self).__init__(img_folder, 
                                            ann_file, 
                                            synthetic_background,  
                                            cache_mode=cache_mode, 
                                            local_rank=local_rank, 
                                            local_size=local_size, 
                                            )
        self._transforms = transforms
        self.prepare = ProcessPoseData(return_masks)
        self.jitter = jitter
        self.jitter_probability = jitter_probability
        self.std = std
        self.im_size = im_size
        self.image_set = image_set
        self.camera = camera
        self.use_mosaic = use_mosaic
        self.cad_model_path = cad_models_path
        self.models_info = load_json(Path(cad_models_path, "models_info.json"))
        
        # Load class-id ↔ name mapping
        self._class_id_to_name = {}
        if class_info is not None and Path(class_info).is_file():
            with open(class_info, 'r') as f:
                classes_json = json.load(f)
            # Accept either {name: id} or {id: name}
            # Normalize to id -> name
            for k, v in classes_json.items():
                if isinstance(v, int):
                    # {name: id}
                    self._class_id_to_name[v] = k
                else:
                    # {id: name}
                    try:
                        self._class_id_to_name[int(k)] = v
                    except Exception:
                        pass
        # Load symmetry info
        self._symmetry_info = {}
        
        if model_symmetry is not None and Path(model_symmetry).is_file():
            sample_mesh_points = True
            with open(model_symmetry, 'r') as f:
                sym_json = json.load(f)
            # Expected formats:
            # 1) { "obj_name": true/false }
            # 2) { "obj_name": { "symmetric": true, ... } }
            for name, val in sym_json.items():
                if isinstance(val, dict):
                    flag = bool(val.get("symmetric", False))
                else:
                    flag = bool(val)
                self._symmetry_info[name] = flag
        # Precompute model points for symmetry-aware loss
        self._model_points = {}
        if sample_mesh_points:
            cache_file = Path(cad_models_path) / f"mesh_points_n{n_mesh_points}_seed{mesh_point_seed}.pt"
            if cache_file.is_file():
                data = torch.load(cache_file)
                self._model_points = {int(k): v for k, v in data.items()}
            else:
                # Build list of mesh file paths keyed by class id
                # Assumes models_info keys are object ids (e.g. "1","2",...) and
                # meshes are named obj_000001.ply (adjust pattern if different).
                model_paths = {}
                for k in self.models_info.keys():
                    obj_id = int(k)
                    ply_name = f"obj_{obj_id:06d}.ply"
                    ply_path = Path(cad_models_path) / ply_name
                    if ply_path.is_file():
                        model_paths[obj_id] = str(ply_path)
                # Sample points
                sampled = precompute_points(model_paths, n=n_mesh_points, seed=mesh_point_seed)
                # sampled expected: dict[obj_id] -> (N,3) ndarray or tensor
                for obj_id, pts in sampled.items():
                    if not torch.is_tensor(pts):
                        pts = torch.as_tensor(pts, dtype=torch.float32)
                    self._model_points[int(obj_id)] = pts  # (N,3)
                torch.save(self._model_points, cache_file)

    def __getitem__(self, idx):
        img, target = super(PoseDataset, self).__getitem__(idx)
        if isinstance(img, torch.Tensor):
            img = to_pil_image(img)
        # We have to pad since the input for the ViT has to be dividable by 64.
        if self.image_set == "test" or self.use_mosaic is False:
            img = pad_to_size(img=img, fill=0)
        image_id = self.ids[idx]
        target = {'image_id': image_id, 'annotations': target}
        img, target = self.prepare(img, target)
        
        if self._transforms is not None:
            img, target = self._transforms(img, target)

        if self.jitter:
            # For the bounding box center we sample from a truncated normal distribution limited by the bounding box
            # width and height for x and y respectively. For the width and height jitter we assume a maximal error of
            # 10% and sample from this error range uniformly.
            jitter_boxes = copy.deepcopy(target["boxes"])
            for box in jitter_boxes:
                # Apply bounding box jitter only with probability
                if random.random() < self.jitter_probability:
                    cxa, cxb = -box[2] / (2 * self.std), box[2] / (2 * self.std)
                    cya, cyb = -box[3] / (2 * self.std), box[3] / (2 * self.std)
                    wa, wb = -0.3 / self.std, 0.3 / self.std
                    ha, hb = -0.3 / self.std, 0.3 / self.std

                    box[0] = truncnorm.rvs(cxa, cxb, loc=box[0], scale=self.std)
                    box[1] = truncnorm.rvs(cya, cyb, loc=box[1], scale=self.std)
                    box[2] = box[2] * (1 + truncnorm.rvs(wa, wb, loc=0, scale=self.std))
                    box[3] = box[3] * (1 + truncnorm.rvs(ha, hb, loc=0, scale=self.std))
            #TODO: if jitter the mosaic has to augment these...
            target["jitter_boxes"] = jitter_boxes
        if False:
            debug_img = img.permute(1,2,0).numpy()
            # Bring back to colorspace 0-255 and BGR -> RGB
            debug_img = (np.clip(debug_img,0,1) * 255).astype(np.uint8)
            debug_img_2 = debug_img[..., ::-1] 
            # This can vizualize the keypoints 
            debug_img, projections = visualize_object_keypoints(cam=self.camera, 
                                                    targets=target, 
                                                    image=debug_img, 
                                                    obj_infos_by_label=self.models_info)
            #Visualize
            viz = YCBVVisualizer(self.cad_model_path)
            K = camera_params_to_K(self.camera)
            vis_img = viz.visualize_single_image(debug_img_2, target, K, show_mesh=True, sample_points=10000)
            cv2.imwrite(Path(DEBUG_OUT, "vis3d.png"), vis_img) # Visualization of 3D bboxes and overalayed objects
            cv2.imwrite(Path(DEBUG_OUT, "keypoints.png"), debug_img) # Vis of bboxes with objects center keypoints
            # Save Visualization of the 2D bboxes
            save_annotated_image(image=img,targets=target)  
        
        # Build symmetry flags aligned with boxes / labels
        labels = target.get("labels", torch.empty(0, dtype=torch.int64))
        # Attach per-target model points (shared tensor per object)
        if len(labels):
            # Stack object-specific point sets: (num_objs, N, 3)
            pts_list = []
            for cid in labels.tolist():
                pts = self._model_points.get(int(cid), None)
                if pts is None:
                    # Fallback: zero cloud
                    pts = torch.zeros(1, 3, dtype=torch.float32)
                pts_list.append(pts)
            # Pad to same N if variable (optional). Here assume same N.
            target["model_points"] = torch.stack(pts_list, dim=0)
        else:
            target["model_points"] = torch.zeros(0, 1, 3, dtype=torch.float32)

        is_symmetric_list = []
        for cid in labels.tolist():
            name = self._class_id_to_name.get(cid, None)
            flag = False
            if name is not None:
                flag = self._symmetry_info.get(name, False)
            is_symmetric_list.append(flag)
        if len(is_symmetric_list):
            target["is_symmetric"] = torch.as_tensor(is_symmetric_list, dtype=torch.bool)
        else:
            target["is_symmetric"] = torch.zeros(0, dtype=torch.bool)

        return img, target 


def convert_coco_poly_to_mask(segmentations, height, width):
    masks = []
    for polygons in segmentations:
        rles = coco_mask.frPyObjects(polygons, height, width)
        mask = coco_mask.decode(rles)
        if len(mask.shape) < 3:
            mask = mask[..., None]
        mask = torch.as_tensor(mask, dtype=torch.uint8)
        mask = mask.any(dim=2)
        masks.append(mask)
    if masks:
        masks = torch.stack(masks, dim=0)
    else:
        masks = torch.zeros((0, height, width), dtype=torch.uint8)
    return masks



class ProcessPoseData(object):
    """
    Processes the annotation file and brings it in the right format for the pose estimation task.
    """
    def __init__(self, return_masks=False):
        self.return_masks = return_masks

    def __call__(self, image, target):
        w, h = image.size

        image_id = target["image_id"]
        image_id = torch.tensor([image_id])

        anno = target["annotations"]

        anno = [obj for obj in anno if 'iscrowd' not in obj or obj['iscrowd'] == 0]

        boxes = [obj["bbox"] for obj in anno]
        # guard against no boxes via resizing
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        boxes[:, 2:] += boxes[:, :2]
        boxes[:, 0::2].clamp_(min=0, max=w)
        boxes[:, 1::2].clamp_(min=0, max=h)

        classes = [obj["category_id"] for obj in anno]
        classes = torch.tensor(classes, dtype=torch.int64)

        if self.return_masks:
            segmentations = [obj["segmentation"] for obj in anno]
            masks = convert_coco_poly_to_mask(segmentations, h, w)

        keypoints = None
        if anno and "keypoints" in anno[0]:
            keypoints = [obj["keypoints"] for obj in anno]
            keypoints = torch.as_tensor(keypoints, dtype=torch.float32)
            num_keypoints = keypoints.shape[0]
            if num_keypoints:
                keypoints = keypoints.view(num_keypoints, -1, 3)

        # Load absolute camera pose
        # Only need to store the global camera pose from the first annotated object as it is the same for each object
        cam_position = None
        cam_rotation = None
        # TODO: Implement if rotation stored as quaternions
        if 'camera_pose' in anno[0]:
            if 'position' in anno[0]['camera_pose']:
                cam_position = anno[0]['camera_pose']['position']
                cam_position = torch.tensor(cam_position, dtype=torch.float32)
            if 'rotation' in anno[0]['camera_pose']:
                cam_rotation = anno[0]['camera_pose']['rotation']
                cam_rotation = torch.tensor(cam_rotation, dtype=torch.float32)
                cam_rotation = torch.reshape(cam_rotation, (3, 3))

        # Load absolute object pose
        obj_position = None
        obj_rotation = None
        if 'object_pose' in anno[0]:
            if 'position' in anno[0]['object_pose']:
                obj_position = [obj['object_pose']['position'] for obj in anno]
                obj_position = torch.tensor(obj_position, dtype=torch.float32)
            if 'rotation' in anno[0]['object_pose']:
                obj_rotation = [obj['object_pose']['rotation'] for obj in anno]
                obj_rotation = torch.tensor(obj_rotation, dtype=torch.float32)
                obj_rotation = torch.reshape(obj_rotation, (-1, 3, 3))

        # Load relative pose between camera and object
        rel_position = None
        rel_quaternion = None
        rel_rotation = None
        if 'relative_pose' in anno[0]:
            if 'position' in anno[0]['relative_pose']:
                rel_position = [obj["relative_pose"]['position'] for obj in anno]
                rel_position = torch.tensor(rel_position, dtype=torch.float32)
            if 'quaternions' in anno[0]['relative_pose']:
                rel_quaternion = [obj["relative_pose"]['quaternions'] for obj in anno]
                rel_quaternion = torch.tensor(rel_quaternion, dtype=torch.float32)
            if 'rotation' in anno[0]['relative_pose']:
                rel_rotation = [obj["relative_pose"]['rotation'] for obj in anno]
                rel_rotation = torch.tensor(rel_rotation, dtype=torch.float32)
                if rel_rotation.shape[1] == 9:
                    rel_rotation = torch.reshape(rel_rotation, (-1, 3, 3))
                    rel_gs_rotation = rotation_matrix_to_gram_schmidt_6d(rel_rotation)
                    # Checking the conversion 6D -> Matrix
                    #back_to_matrix = rotation_6d_to_matrix(rel_gs_rotation)
                    # Gives the same output.
                    #rel_raw_rotation = rotation_matrix_to_raw_6d(rel_rotation)
                    #back_to_matrix = rotation_6d_to_matrix(rel_raw_rotation)
                rel_quaternion = rot2quat(rel_rotation)
                rel_quaternion = torch.tensor(rel_quaternion, dtype=torch.float32)
            else:
                q = np.array([obj["relative_pose"]['quaternions'] for obj in anno])
                rel_rotation = quat2rot(q)
                rel_rotation = torch.tensor(rel_rotation, dtype=torch.float32)

        intrinsics = None
        if 'intrinsics' in anno[0]:
            intrinsics = [obj['intrinsics'] for obj in anno]
            intrinsics = torch.as_tensor(intrinsics, dtype=torch.float32)

        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        boxes = boxes[keep]
        classes = classes[keep]
        if self.return_masks:
            masks = masks[keep]
        if keypoints is not None:
            keypoints = keypoints[keep]
        if obj_position is not None:
            obj_position = obj_position[keep]
        if obj_rotation is not None:
            obj_rotation = obj_rotation[keep]
        if rel_position is not None:
            rel_position = rel_position[keep]
        if rel_quaternion is not None:
            rel_quaternion = rel_quaternion[keep]
        if rel_rotation is not None:
            rel_rotation = rel_rotation[keep]
        if rel_gs_rotation is not None:
            rel_gs_rotation = rel_gs_rotation[keep] 
        if intrinsics is not None:
            intrinsics = intrinsics[keep]

        target = {}
        target["boxes"] = boxes
        target["labels"] = classes
        if self.return_masks:
            target["masks"] = masks
        target["image_id"] = image_id
        if keypoints is not None:
            target["keypoints"] = keypoints
        if cam_position is not None:
            target["camera_position_w"] = cam_position
        if cam_rotation is not None:
            target["camera_rotation_w"] = cam_rotation
        if obj_position is not None:
            target["object_position_w"] = obj_position
        if obj_rotation is not None:
            target["object_rotation_w"] = obj_rotation
        if rel_position is not None:
            target["relative_position"] = rel_position
        if rel_quaternion is not None:
            target["relative_quaternions"] = rel_quaternion
        if rel_rotation is not None:
            target["relative_rotation"] = rel_rotation
        if rel_gs_rotation is not None:
            target["relative_rotation_gs"] = rel_gs_rotation
        if intrinsics is not None:
            target["intrinsics"] = intrinsics

        # for conversion to coco api
        area = torch.tensor([obj["area"] for obj in anno])
        iscrowd = torch.tensor([obj["iscrowd"] if "iscrowd" in obj else 0 for obj in anno])
        target["area"] = area[keep]
        target["iscrowd"] = iscrowd[keep]

        target["orig_size"] = torch.as_tensor([int(h), int(w)])
        target["size"] = torch.as_tensor([int(h), int(w)])



        return image, target
    
def make_pose_estimation_transform(image_set, use_rgb_augmentation, use_grayscale):
    """
    Apply transformations to the images and targets for the pose estimation task depending on the data split.
    """
    # TODO: Add proper data augmentation for pose estimation

    if use_grayscale and image_set not in ['keyframes', 'keyframes_bop', 'test']:
        normalize = T.Compose([
            T.GrayScale(),
            T.ToTensor(),
            T.To3DImage(),
            T.Normalize([0.0, 0.0, 0.0], [1.0, 1.0, 1.0])
        ])
    else:
        normalize = T.Compose([
            T.ToTensor(),
            T.Normalize([0.0, 0.0, 0.0], [1.0, 1.0, 1.0])
        ])

    rgb_augmentation = T.Compose([T.Blur(),
                                  T.Sharpness(),
                                  T.Contrast(),
                                  T.Brightness(),
                                T.Color()])
    
    
    if image_set == 'train':
        if use_rgb_augmentation:
            return T.Compose([rgb_augmentation, normalize, ])
        else:
            return T.Compose([normalize, ])

    if image_set == 'train_real':
        if use_rgb_augmentation:
            return T.Compose([rgb_augmentation, normalize, ])
        else:
            return T.Compose([normalize, ])

    if image_set == 'train_synt':
        if use_rgb_augmentation:
            return T.Compose([rgb_augmentation, normalize, ])
        else:
            return T.Compose([normalize, ])

    if image_set == 'train_pbr':
        if use_rgb_augmentation:
            return T.Compose([rgb_augmentation, normalize, ])
        else:
            return T.Compose([normalize, ])

    if image_set == 'val':
        return T.Compose([
            normalize,
        ])

    if image_set == 'test':
        return T.Compose([
            normalize,
        ])

    if image_set in ['keyframes', 'keyframes_bop']:
        return T.Compose([
            normalize,
        ])

    raise ValueError(f'unknown {image_set}')

def get_mosaic_coordinate(mosaic_image, mosaic_index, xc, yc, w, h, input_h, input_w):
    # TODO update doc
    # index0 to top left part of image
    if mosaic_index == 0:
        x1, y1, x2, y2 = max(xc - w, 0), max(yc - h, 0), xc, yc
        small_coord = w - (x2 - x1), h - (y2 - y1), w, h
    # index1 to top right part of image
    elif mosaic_index == 1:
        x1, y1, x2, y2 = xc, max(yc - h, 0), min(xc + w, input_w * 2), yc
        small_coord = 0, h - (y2 - y1), min(w, x2 - x1), h
    # index2 to bottom left part of image
    elif mosaic_index == 2:
        x1, y1, x2, y2 = max(xc - w, 0), yc, xc, min(input_h * 2, yc + h)
        small_coord = w - (x2 - x1), 0, w, min(y2 - y1, h)
    # index2 to bottom right part of image
    elif mosaic_index == 3:
        x1, y1, x2, y2 = xc, yc, min(xc + w, input_w * 2), min(input_h * 2, yc + h)  # noqa
        small_coord = 0, 0, min(w, x2 - x1), min(y2 - y1, h)
    return (x1, y1, x2, y2), small_coord


class MosaicDetection(PoseDataset):
    """Detection dataset wrapper that performs mixup for normal dataset."""

    def __init__(
        self, 
        dataset, 
        mosaic=True, 
        preproc=None,
        degrees=5.0, 
        translate=0.0,
        mosaic_scale=(1., 1.),
        mixup_scale=(0.5, 1.5), 
        shear=0.0, 
        enable_mixup=True,
        mosaic_prob=0.0, 
        mixup_prob=1.0,  
    ):
        """
        Args:
            dataset(Dataset) : PoseDataset.
            mosaic (bool): enable mosaic augmentation or not.
            preproc (func):
            degrees (float):
            translate (float):
            mosaic_scale (tuple):
            mixup_scale (tuple):
            shear (float):
            enable_mixup (bool):
        """
        self._dataset = dataset
        self.camera_matrix = dataset.camera
        self.preproc = preproc
        self.degrees = degrees
        self.translate = translate
        self.scale = mosaic_scale
        self.shear = shear
        self.mixup_scale = mixup_scale
        self.enable_mosaic = mosaic
        self.enable_mixup = enable_mixup
        self.mosaic_prob = mosaic_prob
        self.mixup_prob = mixup_prob
        self.local_rank = dataset.local_rank
        
    def __len__(self):
        return len(self._dataset)

    def __getitem__(self, idx):
        #if False: # To trigger it everytime for debugging
        if self.enable_mosaic and random.random() < self.mosaic_prob: # This should always be triggered as long as mosaic is 1.0
            mosaic_labels = []
            mosaic_rots = []
            mosaic_trans = []
            mosaic_obj_ids = []
            mosaic_intrinsics = []
            mosaic_keypoints = []
            input_dim = (self._dataset.im_size[0], self._dataset.im_size[1]) # (h, w)
            input_h, input_w = input_dim[0], input_dim[1]
            
            # yc, xc = s, s  # mosaic center x, y
            yc = int(random.uniform(0.5 * input_h, 1.5 * input_h))
            xc = int(random.uniform(0.5 * input_w, 1.5 * input_w))

            # 3 additional image indices, why 3 ?
            indices = [idx] + [random.randint(0, len(self._dataset) - 1) for _ in range(3)]
            for i_mosaic, index in enumerate(indices):
                img_tensor, _labels = self._dataset[index]
                img_id = _labels['image_id'].item()
                # To numpy 
                img = img_tensor.numpy()
                # Normalize it
                img = (img.clip(0,1) * 255).astype(np.uint8)
                img = img.transpose(1, 2, 0)  # to hwc
                h0, w0 = img.shape[:2]  # orig hw
                scale = min(1. * input_h / h0, 1. * input_w / w0)
                img = cv2.resize(img, 
                                 (int(w0 * scale), int(h0 * scale)), 
                                 interpolation=cv2.INTER_LINEAR)
                # generate output mosaic image
                (h, w, c) = img.shape[:3]
                if i_mosaic == 0:
                    mosaic_img = np.full((input_h * 2, input_w * 2, c), 114, dtype=np.uint8)
                    if DEBUG:
                        out = Path(DEBUG_OUT, "mosaic_background.png")
                        Image.fromarray(mosaic_img).save(out)
                    # Normalize since our img is alo normalized
                    #mosaic_img = mosaic_img.astype(np.float32) / 255.0
                # suffix l means large image, while s means small image in mosaic aug.
                (l_x1, l_y1, l_x2, l_y2), (s_x1, s_y1, s_x2, s_y2) = get_mosaic_coordinate(
                    mosaic_img, i_mosaic, xc, yc, w, h, input_h, input_w
                )
                # This has to be the camera intrinsics from the target not the given dataset camera??
                base_cam = self._dataset.camera
                base_scene_cam = self._dataset.camera
                mosaic_img[l_y1:l_y2, l_x1:l_x2] = img[s_y1:s_y2, s_x1:s_x2]
                padw, padh = l_x1 - s_x1, l_y1 - s_y1

                bboxes_xywh = _labels['boxes'].clone().numpy()
                relative_rotation = _labels['relative_rotation'].clone()
                relative_position = _labels['relative_position'].clone()
                relative_quaternions = _labels['relative_quaternions'].clone()
                labels = _labels["labels"]
                width, height = img.shape[1], img.shape[0]
                lst_boxes_xyxy = []
        
                # BBoxes comes in normalized (0,1) and xywh coords format
                # We need absolute pixes xyxy coords for mosaic
                for bbox in bboxes_xywh:
                    x_center, y_center, w, h = bbox
                    x1 = (x_center - w/2) * width
                    y1 = (y_center - h/2) * height
                    x2 = (x_center + w/2) * width
                    y2 = (y_center + h/2) * height
                    lst_boxes_xyxy.append([x1, y1, x2, y2])
                    boxes= np.array(lst_boxes_xyxy) # xyxy and absolute pixel coords
                
                if len(boxes) > 0:
                    boxes[:, 0] = scale * boxes[:, 0] + padw 
                    boxes[:, 1] = scale * boxes[:, 1] + padh 
                    boxes[:, 2] = scale * boxes[:, 2] + padw 
                    boxes[:, 3] = scale * boxes[:, 3] + padh
                   
                    if len(bboxes_xywh) > 0:
                        # per-image adjusted intrinsics (identical for all objects from this source image)
                        # We dont want to touch the camera matrix params from the dataset.
                        fx_adj = base_cam["fx"] * scale
                        fy_adj = base_cam["fy"] * scale
                        cx_adj = base_cam["cx"] * scale + padw
                        cy_adj = base_cam["cy"] * scale + padh
                        # Takes the intrinsics of every scene camera.
                        # fx_adj = base_cam["fx"]   
                        # fy_adj = base_cam["fy"]
                        # cx_adj = base_cam["cx"] 
                        # cy_adj = base_cam["cy"]
                        intrinsics_adj = torch.tensor([fx_adj, fy_adj, cx_adj, cy_adj], 
                                                      dtype=torch.float32)
                        intrinsics_adj = intrinsics_adj.unsqueeze(0).repeat(relative_position.shape[0], 1) 
                        
                # Append only matching groups
                assert relative_position.shape[0] == boxes.shape[0], \
                    f"Pose count {relative_position.shape[0]} != boxes count {boxes.shape[0]}"        
                mosaic_labels.append(boxes)
                mosaic_rots.append(relative_rotation)
                mosaic_trans.append(relative_position)
                mosaic_obj_ids.append(labels)
                mosaic_intrinsics.append(intrinsics_adj)

            if len(mosaic_labels):
                mosaic_labels = np.concatenate(mosaic_labels, 0)
                mosaic_rots = np.concatenate(mosaic_rots, 0)
                mosaic_trans = np.concatenate(mosaic_trans, 0)
                mosaic_obj_ids = np.concatenate(mosaic_obj_ids, 0)
                mosaic_intrinsics = torch.cat(mosaic_intrinsics, 0)
                # ## Vis without touching camera params
                # keypoints_2d = []
                # for i in range(mosaic_trans.shape[0]):
                #     label = int(mosaic_obj_ids[i])
                #     info = self._dataset.models_info.get(str(label), None)
                #     if info is None:
                #         keypoints_2d.append(np.zeros((8,2), dtype=np.float32))
                #         continue
                #     min_x, min_y, min_z = info["min_x"], info["min_y"], info["min_z"]
                #     sx, sy, sz = info["size_x"], info["size_y"], info["size_z"]
                #     max_x, max_y, max_z = min_x + sx, min_y + sy, min_z + sz
                #     corners_obj = np.array([
                #         [min_x, min_y, min_z],
                #         [max_x, min_y, min_z],
                #         [max_x, max_y, min_z],
                #         [min_x, max_y, min_z],
                #         [min_x, min_y, max_z],
                #         [max_x, min_y, max_z],
                #         [max_x, max_y, max_z],
                #         [min_x, max_y, max_z],
                #     ], dtype=np.float32)  # (8,3)
                #     R = mosaic_rots[i]
                #     t = mosaic_trans[i]
                #     fx, fy, cx, cy = mosaic_intrinsics[i].tolist()
                #     pts_cam = (R @ corners_obj.T + t.reshape(3,1)).T
                #     uv = np.zeros((8,2), dtype=np.float32)
                #     valid = pts_cam[:,2] > 1e-6
                #     uv[valid] = np.stack([
                #         fx * pts_cam[valid,0] / pts_cam[valid,2] + cx,
                #         fy * pts_cam[valid,1] / pts_cam[valid,2] + cy
                #     ], axis=1)
                #     keypoints_2d.append(uv)
                # mosaic_keypoints_2d = torch.from_numpy(np.stack(keypoints_2d, 0))  # (N,8,2)


                # Clip all boxes to be within the image bounds
                np.clip(mosaic_labels[:, 0], 0, 2 * input_w, out=mosaic_labels[:, 0])
                np.clip(mosaic_labels[:, 1], 0, 2 * input_h, out=mosaic_labels[:, 1])
                np.clip(mosaic_labels[:, 2], 0, 2 * input_w, out=mosaic_labels[:, 2])
                np.clip(mosaic_labels[:, 3], 0, 2 * input_h, out=mosaic_labels[:, 3])
            #### We need the object keypoint from the mosaic before random affine to ####
            if DEBUG:
                # Mosaic image 
                out = Path(DEBUG_OUT, "mosaic_img_b4_random_affine.jpg")
                Image.fromarray(mosaic_img).save(out)
                # random_affine expects a np.array so we put the mosaic into a tempraory tensor for debug vizualization
                mosaic_img_debug_out=torch.from_numpy(mosaic_img).to(torch.uint8)
                mosaic_img_debug_out= mosaic_img_debug_out.permute(2,0,1) # to CHW
                mosaic_img_debug_out = mosaic_img_debug_out.float() / 255.0
                # Mosaic image with annotations
                save_annotated_image(image=mosaic_img, 
                                        targets={'boxes':torch.tensor(mosaic_labels)}, 
                                        output_path="annotated_mosaic_b4_affine.png", 
                                                is_bbbox_coords_normalized=False,  
                                                is_corrected_bbx_coords=True)
                if DEBUG:
                    debug_img = mosaic_img
                    # Bring back to colorspace 0-255 and BGR -> RGB
                    #debug_img = (np.clip(debug_img,0,1) * 255).astype(np.uint8)
                    debug_img = debug_img[..., ::-1] 
                    debug_img_2 = debug_img
                    
                    debug_targets = {
                        'labels': torch.from_numpy(mosaic_obj_ids),
                        'relative_position': torch.from_numpy(mosaic_trans),
                        'relative_rotation': torch.from_numpy(mosaic_rots),
                        'intrinsics': mosaic_intrinsics,   # per-object intrinsics
                        #'keypoints_2d': mosaic_keypoints_2d
                    }
                    debug_img, _ = visualize_object_keypoints(
                        cam=None,
                        targets=debug_targets,
                        image=debug_img,
                        obj_infos_by_label=self._dataset.models_info
                    )
                    cv2.imwrite(Path(DEBUG_OUT, "keypoints_mosaic_b4_affine.png"), debug_img)
                    # Visualize 3D bbox and caod model overlay
                    viz = YCBVVisualizer(self._dataset.cad_model_path)
                    K = camera_params_to_K(self._dataset.camera)
                    vis_img = viz.visualize_single_image(debug_img_2, 
                                                        annotations={'labels': torch.from_numpy(mosaic_obj_ids), 
                                                                    "relative_position":torch.from_numpy(mosaic_trans), 
                                                                    "relative_rotation":torch.from_numpy(mosaic_rots),
                                                                    "intrinsics": mosaic_intrinsics # (N,4)
                                                                    },
                                                                    K=K,
                                                                    show_mesh=True,
                                                                    sample_points=5000)
                    cv2.imwrite(Path(DEBUG_OUT, "vis3d_mosaic_b4_affine.png"), vis_img) # Visualization of 3D bboxes and overalayed objects
                # Apply random affine transformations
                mosaic_img, mosaic_labels = random_affine(
                    img=mosaic_img,
                    rel_pos=relative_position,
                    rel_rot=relative_rotation,
                    rel_quats=relative_quaternions,

                    labels=mosaic_obj_ids,
                    targets=mosaic_labels,
                    # Can This be already be the image in the resolution we need it instead of padding it?
                    target_size=((input_h, input_w)), # Now it takes the input dim we need for the ViT to work with
                    degrees=self.degrees,
                    translate=self.translate,
                    scales=self.scale,
                    shear=self.shear,
                    camera_matrix=self.camera_matrix,
                    mosaic_obj_ids=mosaic_obj_ids,
                    mosaic_intrinsics=mosaic_intrinsics,
                    mosaic_rots=mosaic_rots,
                    mosaic_trans=mosaic_trans
                )  # border to remove

                # -----------------------------------------------------------------
                # CopyPaste: https://arxiv.org/abs/2012.07177
                # -----------------------------------------------------------------
                if (
                    self.enable_mixup
                    and not len(mosaic_labels) == 0
                    and random.random() < self.mixup_prob
                ):
                    mosaic_img, mosaic_labels = self.mixup(mosaic_img, mosaic_labels, input_dim)
                mix_img, padded_labels = self.preproc(mosaic_img, mosaic_labels, input_dim) 
                img_info = (mix_img.shape[1], mix_img.shape[0])

                # -----------------------------------------------------------------
                # img_info and img_id are not used for training.
                # They are also hard to be specified on a mosaic image.
                # -----------------------------------------------------------------
                return mix_img, padded_labels, img_info, img_id
        else:
            # This gets me an image and target from the base dataset.
            img, target = self._dataset[idx]
            input_dim = (self._dataset.im_size[0], self._dataset.im_size[1]) # (h, w)
            input_dim = (target['size'][0].item(), target['size'][1].item()) # (h, w) ycbv im size
            input_h, input_w = input_dim[0], input_dim[1]

            # when camera matrix is provided, we are doing 6d pose estimation.
            if self.camera_matrix is None:
                camera_matrix = None
            else:
                camera_matrix = self.camera_matrix
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
                lst_boxes_xyxy.append([x1, y1, x2, y2])
                boxes= np.array(lst_boxes_xyxy) # xyxy and absolute pixel coords
            # Image with bbox annotations
            if DEBUG:
                save_image_with_bboxes(
                    img=img,
                    boxes=torch.from_numpy(boxes),
                    labels=target['labels'],
                    out_path=Path(DEBUG_OUT,"before_random_affine","bboxes_ids.png")
                )
                debug_img = img.numpy().transpose(1,2,0)*255.0
                debug_img = debug_img[:,:,::-1]
                debug_img, _ = visualize_object_keypoints(
                    cam=camera_matrix,
                    targets=target,
                    image=debug_img,
                    obj_infos_by_label=self._dataset.models_info
                )
                cv2.imwrite(Path(DEBUG_OUT, "before_random_affine", "keypoints.png"), debug_img )
                # Visualize 3D bbox and cad model overlay
                viz = YCBVVisualizer(self._dataset.cad_model_path)
                K = camera_params_to_K(self._dataset.camera)  
                vis_img = img.numpy().transpose(1,2,0)*255.0
                vis_img = vis_img[:,:,::-1]
                vis_img = viz.visualize_single_image(vis_img, 
                                                    annotations={'labels': target['labels'], 
                                                                "relative_position":target['relative_position'], 
                                                                "relative_rotation":target['relative_rotation'],
                                                                },
                                                                K=K,
                                                                show_mesh=True,
                                                                sample_points=5000)
                
                cv2.imwrite(Path(DEBUG_OUT, "before_random_affine", "vis3d.png"), 
                            vis_img) # Visualization of 3D bboxes and overalayed objects



            if camera_matrix is not None and self.enable_mosaic:  # no aug training for 6d pose estimation.
                img, target = random_affine_single(
                    img=img,
                    target=target,
                    target_size=input_dim,
                    degrees=self.degrees,
                    translate=self.translate,
                    scales=self.scale,
                    shear=self.shear,
                    camera_matrix=camera_matrix
                ) 
            # Other augs?     
            #img, label = self.preproc(img, target, input_dim)
            
            return img, target


def build(image_set, args):
    root = Path(args.dataset_path)
    assert root.exists(), f'provided dataset path {root} does not exist'
    PATHS = {
        "train": (root , root / "annotations" / f'train.json'), # TODO: combine both train sets
        "train_real": (root , root / "annotations" / f'train.json'),
        "train_synt": (root, root / "annotations" / f'train_synt.json'),
        "train_pbr": (root , root / "annotations" / f'train_pbr.json'),
        "test": (root , root / "annotations" / f'test.json'), # TODO: Whats wrong here?
        "keyframes": (root, root / "annotations" / f'keyframes.json'),
        "keyframes_bop": (root, root / "annotations"/ f'keyframes_bop.json'),
        "val": (root , root / "annotations" / f'val.json'),
    }
    cad_model_path  = Path(root, "models")
    camera_intrinsics_file = Path(root,args.camera)
    # load intrinsics
    import json
    with open(camera_intrinsics_file) as f:
        camera = json.load(f)
        print(f'Camera matrix from: {camera_intrinsics_file.name}\n{camera}')

    img_folder, ann_file = PATHS[image_set]
    im_size = args.im_size
    # TODO: Replace 'transforms' by a proper data augmentation function suitable for pose estimation. Currently only
    #  image level augmentation possible (e.g. color augmentation, noise).
    if args.bbox_mode == 'jitter':
        jitter = True
    else:
        jitter = False
    
    if args.model_symmetry:
        model_symmetry = Path(str(root) + args.model_symmetry)
    else:
        model_symmetry = None
    if args.class_info:
        class_info = Path(str(root) + args.class_info)
    else:
        class_info = None
    dataset = PoseDataset(img_folder, ann_file, 
                          im_size=im_size, 
                          synthetic_background=args.synt_background,
                          transforms=make_pose_estimation_transform(image_set, 
                                                                    args.rgb_augmentation, 
                                                                    args.grayscale),
                          return_masks=False,
                          jitter=jitter, 
                          camera=camera,
                          jitter_probability=args.jitter_probability,
                          cache_mode=args.cache_mode,
                          local_rank=get_local_rank(),
                          local_size=get_local_size(),
                          image_set=image_set,
                          use_mosaic=args.mosaic,
                          cad_models_path=cad_model_path,
                          model_symmetry=model_symmetry,
                          class_info=class_info)
    if args.mosaic and 'train' in image_set:
        print("Creating Mosaic Augmentation")
        dataset_mosaic = MosaicDetection(
                dataset,
                mosaic=True,
                
        )
        return dataset_mosaic    
    else:
        return dataset
