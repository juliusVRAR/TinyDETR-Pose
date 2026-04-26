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
from util.misc import get_local_rank
from util.quaternion_ops import quat2rot, rot2quat
from util.rotation_utils import (
    rotation_matrix_to_gram_schmidt_6d,
    precompute_points,
    build_symmetry_transforms,
    pad_symmetry_transforms,
    get_sarr_symmetry_vectors,
    rotation_matrix_to_sarr,
)
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
from data_utils.data_augment import random_affine_single
from util.utils import save_annotated_image, camera_params_to_K, pad_to_size
from util.visualize_object_pose import visualize_object_keypoints, YCBVVisualizer, save_image_with_bboxes
from pycocotools.coco import COCO

DEBUG = False
DEBUG_OUT=Path("debug")

def load_json(path: str | Path):
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

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
                 local_rank=0,
                 image_set='train',
                 use_mosaic=False,
                 model_symmetry=None,
                 class_info=None,
                 sample_mesh_points = False, # Only true if we calulate symmetries because we have CAD model information
                 n_mesh_points=128, # The higher the more VRAM we need but the better the symmetry-aware loss works (T6D samples 1500 points)
                 mesh_point_seed=42
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
                                            )
        self._transforms = transforms
        self.prepare = ProcessPoseData(return_masks, camera)
        self.jitter = jitter
        self.jitter_probability = jitter_probability
        self.std = std
        self.im_size = im_size
        self.image_set = image_set
        self.camera = camera
        self.use_mosaic = use_mosaic
        self.cad_model_path = cad_models_path
        self.models_info = load_json(Path(cad_models_path, "models_info.json"))
        self.coco = COCO(ann_file)
        self.mesh_point_seed = mesh_point_seed
        self.n_mesh_points = n_mesh_points
        self._warned_missing_model_points = set()
        # Precompute diameter lookup (id -> float)
        self._diameters = {}
        for k, v in self.models_info.items():
            try:
                obj_id = int(k)
                if isinstance(v, dict) and 'diameter' in v:
                    self._diameters[obj_id] = float(v['diameter'])
            except Exception:
                continue

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
        
        # Build once needed for sym. aware L1 loss; dict[obj_id] -> (K, 3, 3) tensor of symmetry transforms
        raw = build_symmetry_transforms(cad_models_path, K_continuous=360)
        self.sym_transforms, self.max_K = pad_symmetry_transforms(raw) # self.sym_transforms: {obj_id: (K, 3, 3)}
        
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
            cache_file = Path(cad_models_path) / f"mesh_points_n{self.n_mesh_points}_seed{mesh_point_seed}.pt"
            if cache_file.is_file():
                data = torch.load(cache_file)
                self._model_points = {}
                for key, points in data.items():
                    if not torch.is_tensor(points):
                        points = torch.as_tensor(points, dtype=torch.float32)
                    else:
                        points = points.to(dtype=torch.float32)

                    # Older caches may store points in millimeters while newer caches
                    # already store meters. Only rescale when the magnitude indicates mm.
                    if points.numel() and points.abs().max().item() > 10.0:
                        points = points / 1000.

                    self._model_points[int(key)] = points
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
                sampled = precompute_points(model_paths, n=self.n_mesh_points, seed=mesh_point_seed)
                # sampled expected: dict[obj_id] -> (N,3) ndarray or tensor
                for obj_id, pts in sampled.items():
                    if not torch.is_tensor(pts):
                        pts = torch.as_tensor(pts, dtype=torch.float32)
                    self._model_points[int(obj_id)] = pts / 1000. # (N,3)
                torch.save(self._model_points, cache_file)

            if not self._model_points:
                print(
                    f"Warning: no CAD model points were loaded from {cad_models_path}. "
                    "Rotation loss will collapse to zero and ADD-S will only supervise translation."
                )
            else:
                tiny_point_ids = [
                    obj_id for obj_id, pts in self._model_points.items()
                    if pts.numel() and pts.abs().max().item() < 1e-4
                ]
                if tiny_point_ids:
                    preview = tiny_point_ids[:5]
                    print(
                        f"Warning: tiny cached CAD point clouds detected for object ids {preview}. "
                        f"This usually means a stale mesh cache was scaled twice. Delete {cache_file.name} "
                        "and regenerate it."
                    )

    def __getitem__(self, idx):
        img, target = super(PoseDataset, self).__getitem__(idx)
        if isinstance(img, torch.Tensor):
            img = to_pil_image(img)
        # We have to pad since the input for the ViT has to be dividable by 64.
        # if self.image_set == "test" or self.use_mosaic is False:
        #     img = pad_to_size(img=img, fill=0)
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
        # if DEBUG
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
        
        ### CAD Models needed for this part ### 
        # Build symmetry flags aligned with boxes / labels
        labels = target.get("labels", torch.empty(0, dtype=torch.int64))
        # Attach per-target model points (shared tensor per object)
        if len(labels) and self.models_info is not None:
            # Stack object-specific point sets: (num_objs, N, 3)
            pts_list = []
            for cid in labels.tolist():
                pts = self._model_points.get(int(cid), None)
                if pts is None:
                    if int(cid) not in self._warned_missing_model_points:
                        print(
                            f"Warning: missing CAD model points for class id {cid}. "
                            "Using a zero cloud, which makes the rotation loss zero for those samples."
                        )
                        self._warned_missing_model_points.add(int(cid))
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
        if len(is_symmetric_list) and self.models_info is not None:
            target["is_symmetric"] = torch.as_tensor(is_symmetric_list, dtype=torch.bool)
        else:
            target["is_symmetric"] = torch.zeros(0, dtype=torch.bool)

        if len(labels) and "relative_rotation" in target:
            sarr_sym_v = get_sarr_symmetry_vectors(labels).to(dtype=torch.long)
            rel_rot = target["relative_rotation"].float()
            target["sarr_sym_v"] = sarr_sym_v
            target["relative_rotation_sarr"] = rotation_matrix_to_sarr(
                rel_rot,
                sarr_sym_v.to(device=rel_rot.device),
                clamp=True,
            )
            if not torch.isfinite(target["relative_rotation_sarr"]).all():
                finite = torch.isfinite(target["relative_rotation_sarr"])
                print("Non-finite relative_rotation_sarr detected in dataset target creation.")
                print(f"labels={labels.tolist()}")
                print(f"sarr_sym_v={sarr_sym_v.tolist()}")
                print(
                    f"relative_rotation_sarr shape={tuple(target['relative_rotation_sarr'].shape)} "
                    f"finite={int(finite.sum().item())}/{target['relative_rotation_sarr'].numel()}"
                )
                print(
                    f"relative_rotation_sarr min="
                    f"{torch.nan_to_num(target['relative_rotation_sarr']).min().item():.6f}"
                )
                print(
                    f"relative_rotation_sarr max="
                    f"{torch.nan_to_num(target['relative_rotation_sarr']).max().item():.6f}"
                )
                raise RuntimeError("Non-finite relative_rotation_sarr")
        else:
            target["sarr_sym_v"] = torch.zeros(0, 3, dtype=torch.long)
            target["relative_rotation_sarr"] = torch.zeros(0, 6, dtype=torch.float32)

        if len(labels) and self.models_info is not None:
            diam_list = []
            for cid in labels.tolist():
                diam_list.append(self._diameters.get(int(cid), 0.0))  # fallback 0.0
            target["diameter"] = torch.as_tensor(diam_list, dtype=torch.float32)
        else:
            target["diameter"] = torch.zeros(0, dtype=torch.float32)
         

        if len(labels) and self.sym_transforms:
            sym_list = []
            for cid in labels.tolist():
                sym = self.sym_transforms.get(int(cid), None)
                if sym is None:
                    sym = torch.eye(3).unsqueeze(0).expand(self.max_K, 3, 3).clone()
                sym_list.append(sym)
            target["symmetry_transforms"] = torch.stack(sym_list, dim=0)
        else:
            target["symmetry_transforms"] = torch.zeros(0, self.max_K, 3, 3)
        

        # Sanaty check the symmetry transforms and print K for each object
        # if len(labels):
        #     print(f"sym_transforms shape: {target['symmetry_transforms'].shape}")
        #     # Expected: (num_objects, 360, 3, 3) with K_continuous=360
        #     for i, cid in enumerate(labels.tolist()):
        #         K_actual = (target['symmetry_transforms'][i].sum(dim=(-1,-2)) != 3.0).sum()
        #         # identity has trace=3, so non-identity count ≈ real symmetries
        #         name = self._class_id_to_name.get(cid, str(cid))
        #         print(f"  {name}: K={K_actual + 1}")

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
    def __init__(self, return_masks=False, camera=None):
        self.return_masks = return_masks
        self.camera = camera

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
                trans_z = rel_position[:, 2].unsqueeze(1)  # (N,1)

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


        # ------------------------------------------------------------------
        # Compute 2D projected object centers (pixel)
        # rel_position: (N,3) in camera coordinates (X,Y,Z)
        # u = fx * X/Z + cx, v = fy * Y/Z + cy
        # ------------------------------------------------------------------
        object_center_2d = None
        if rel_position is not None:
            fx = self.camera['fx']; fy = self.camera['fy']
            cx = self.camera['cx']; cy = self.camera['cy']
            X = rel_position[:, 0] * 1000.0
            Y = rel_position[:, 1] * 1000.0
            Z = rel_position[:, 2].clamp(min=1e-6) * 1000.0
            u = fx * (X / Z) + cx
            v = fy * (Y / Z) + cy
            object_center_2d = torch.stack([u, v], dim=1)  # pixels
        # Dont normalize here, do it later if needed in the model
        #centers_img = draw_object_centers(image, object_center_2d)
        #cv2.imwrite(str(Path(DEBUG_OUT, "object_centers.png")), centers_img)
        # ------------------------------------------------------------------

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
        if object_center_2d is not None:
            object_center_2d = object_center_2d[keep]
           

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
            #target["Z_LIMIT"] = 2.5 # For ycbv dataset we set a fixed z limit of 2.5 meters
            target["relative_translation_z"] = rel_position[:, 2] / 2.5 # Normalized Depth #TODO: Put into targets to use later in Criterion
        if rel_quaternion is not None:
            target["relative_quaternions"] = rel_quaternion
        if rel_rotation is not None:
            target["relative_rotation"] = rel_rotation
        if rel_gs_rotation is not None:
            target["relative_rotation_gs"] = rel_gs_rotation
        if intrinsics is not None:
            target["intrinsics"] = intrinsics
        if object_center_2d is not None:
            target["object_center_2d"] = object_center_2d              # pixels

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
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    else:
        normalize = T.Compose([
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
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
        degrees=10.0, 
        translate=0.1,
        mosaic_scale=(0.9, 1.1),
        mixup_scale=(1., 1.), 
        shear=0.0, 
        enable_mixup=True,
        mosaic_prob=0.5, 
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
        # This gets me an image and target from the base dataset.
        img, target = self._dataset[idx]
        if self.enable_mosaic and random.random() < self.mosaic_prob:
            
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
        else:
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
        "keyframesd": (root, root / "annotations" / f'keyframes.json'),
        "keyframes_bop": (root, root / "annotations"/ f'keyframes_bop.json'),
        "val": (root , root / "annotations" / f'val.json'),
    }
    if args.models:
        cad_model_path = Path(str(root) + args.models)
    else:
        cad_model_path = Path(root, "models")

    if not cad_model_path.exists():
        fallback_cad_model_path = Path(root, "models")
        if fallback_cad_model_path.exists():
            print(f"CAD model path {cad_model_path} not found. Falling back to {fallback_cad_model_path}.")
            cad_model_path = fallback_cad_model_path

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
    # Set seed when training for reproducibility
    seed = torch.manual_seed(0)
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
                          image_set=image_set,
                          use_mosaic=args.mosaic_augmentation,
                          cad_models_path=cad_model_path,
                          model_symmetry=model_symmetry,
                          class_info=class_info,
                          n_mesh_points=args.n_mesh_points,
                          mesh_point_seed=args.seed)
    
    if args.mosaic_augmentation and 'train' in image_set:
        print("Creating Mosaic Augmentation")
        dataset_mosaic = MosaicDetection(
                dataset,
                mosaic=True,  
        )
        return dataset_mosaic    
    else:
        return dataset
