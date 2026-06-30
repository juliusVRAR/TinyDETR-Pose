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
# Modified from Deformable DETR (https://github.com/fundamentalvision/Deformable-DETR)
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# ------------------------------------------------------------------------

"""
Modules to compute the matching cost and solve the corresponding LSAP.
"""
import numpy as np
from sympy import rotations
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from torch import nn
from util.box_ops import box_cxcywh_to_xyxy, generalized_box_iou
from util.rotation_utils import (
    normalize_sarr_pairs,
    rotation_matrix_to_gram_schmidt_6d,
    rotation_matrix_to_raw_6d,
    rotation_6d_to_matrix,
    rotation_6d_simple_to_matrix,
)


def pairwise_cosine_distance_cost(pred_vectors, gt_vectors, eps=1e-8):
    pred_vectors = pred_vectors / pred_vectors.norm(dim=-1, keepdim=True).clamp_min(eps)
    gt_vectors = gt_vectors / gt_vectors.norm(dim=-1, keepdim=True).clamp_min(eps)
    return 1.0 - torch.matmul(pred_vectors, gt_vectors.transpose(0, 1))

class HungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network
    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(self, cost_class: float = 1, 
                 cost_bbox: float = 1, 
                 cost_giou: float = 1, 
                 focal_alpha: float = 0.25, 
                 use_pos_only: bool = False,
                 use_position_modulated_cost: bool = False):
        """Creates the matcher
        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_bbox: This is the relative weight of the L1 error of the bounding box coordinates in the matching cost
            cost_giou: This is the relative weight of the giou loss of the bounding box in the matching cost
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        assert cost_class != 0 or cost_bbox != 0 or cost_giou != 0, "all costs cant be 0"
        self.focal_alpha = focal_alpha

    @torch.no_grad()
    def forward(self, outputs, targets, group_detr=1):
        """ Performs the matching
        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                 "pred_boxes": Tensor of dim [batch_size, num_queries, 4] with the predicted box coordinates
            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                           objects in the target) containing the class labels
                 "boxes": Tensor of dim [num_target_boxes, 4] containing the target box coordinates
            group_detr: Number of groups used for matching.
        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        bs, num_queries = outputs["pred_logits"].shape[:2]

        # We flatten to compute the cost matrices in a batch
        out_prob = outputs["pred_logits"].flatten(0, 1).sigmoid()  # [batch_size * num_queries, num_classes]
        out_bbox = outputs["pred_boxes"].flatten(0, 1)  # [batch_size * num_queries, 4]

        # Also concat the target labels and boxes
        tgt_ids = torch.cat([v["labels"] for v in targets])
        tgt_bbox = torch.cat([v["boxes"] for v in targets])

        # Compute the giou cost betwen boxes
        giou = generalized_box_iou(box_cxcywh_to_xyxy(out_bbox), box_cxcywh_to_xyxy(tgt_bbox))
        cost_giou = -giou

        # Compute the classification cost.
        alpha = 0.25
        gamma = 2.0
        
        neg_cost_class = (1 - alpha) * (out_prob ** gamma) * (-(1 - out_prob + 1e-8).log())
        pos_cost_class = alpha * ((1 - out_prob) ** gamma) * (-(out_prob + 1e-8).log())
        cost_class = pos_cost_class[:, tgt_ids] - neg_cost_class[:, tgt_ids]

        # Compute the L1 cost between boxes
        cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)

        # Final cost matrix
        C = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou
        C = C.view(bs, num_queries, -1).cpu()

        sizes = [len(v["boxes"]) for v in targets]
        indices = []
        g_num_queries = num_queries // group_detr
        C_list = C.split(g_num_queries, dim=1)
        for g_i in range(group_detr):
            C_g = C_list[g_i]
            indices_g = [linear_sum_assignment(c[i]) for i, c in enumerate(C_g.split(sizes, -1))]
            if g_i == 0:
                indices = indices_g
            else:
                indices = [
                    (np.concatenate([indice1[0], indice2[0] + g_num_queries * g_i]), np.concatenate([indice1[1], indice2[1]]))
                    for indice1, indice2 in zip(indices, indices_g)
                ]
        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]


class Matcher6D(nn.Module):
    """Hungarian matcher for 6D pose estimation tasks.
    This class computes an assignment between the targets and the predictions of the network
    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """
    def __init__(self, 
                 cost_class: float = 1, 
                 cost_bbox: float = 1, 
                 cost_giou: float = 1,
                 cost_rotation: float = 1,
                 cost_keypoint: float = 1, 
                 rotation_representation: str = '6d'):
        """Creates the matcher
        
        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_bbox: This is the relative weight of the L1 error of the bounding box coordinates in the matching cost
            cost_giou: This is the relative weight of the giou loss of the bounding box in the matching cost
            cost_rotation: This is the relative weight of the rotation error in the matching cost
            cost_translation: This is the relative weight of the translation error in the matching cost
            use_6d_pose: Whether to use 6D pose costs in matching
            rotation_representation: How rotation is represented ('matrix', 'quaternion', 'axis_angle', '6d')
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        self.cost_rotation = cost_rotation
        self.cost_keypoint = cost_keypoint
        self.rotation_representation = rotation_representation
        
        assert cost_class != 0 or cost_bbox != 0 or cost_giou != 0, "all costs cant be 0"

   
    def compute_keypoint_cost(self, pred_xy, target_xy):
        """
        pred_xy: [num_queries, 2] - predicted xy translations
        target_xy: [num_targets, 2] - ground truth xy translations
    
        Returns: cost matrix [num_queries, num_targets]
        """
        # L1 distance 
        cost_xy = torch.cdist(pred_xy, target_xy, p=1)  # Manhattan distance
    
        # Alternative: L2 distance
        # cost_xy = torch.cdist(pred_xy, target_xy, p=2)  # Euclidean distance
        return cost_xy
    
    
    @torch.no_grad()
    def forward(self, outputs, targets, group_detr=1):
        """Performs the matching
        
        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                 "pred_boxes": Tensor of dim [batch_size, num_queries, 4] with the predicted box coordinates
                 "pred_rotations": (optional) Tensor with predicted rotations
                 "pred_translations": (optional) Tensor with predicted translations [batch_size, num_queries, 3]
                 
            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                           objects in the target) containing the class labels
                 "boxes": Tensor of dim [num_target_boxes, 4] containing the target box coordinates
                 "rotations": (optional) Tensor containing target rotations
                 "translations": (optional) Tensor of dim [num_target_boxes, 3] containing target translations
                 
        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        bs, num_queries = outputs["pred_logits"].shape[:2]

        # We flatten to compute the cost matrices in a batch
        out_prob = outputs["pred_logits"].flatten(0, 1).softmax(-1)  # [batch_size * num_queries, num_classes]
        out_bbox = outputs["pred_boxes"].flatten(0, 1)  # [batch_size * num_queries, 4]
        
        pred_uv_norm = outputs["pred_uv_norm"].flatten(0, 1)  # [batch_size * num_queries, 2]
        out_keypoint = pred_uv_norm
        
        # Also concat the target labels and boxes
        tgt_ids = torch.cat([v["labels"] for v in targets])
        tgt_bbox = torch.cat([v["boxes"] for v in targets])

        if ("object_center_2d" in targets[0]) and (targets[0]["object_center_2d"] is not None):
            tgt_keypoint = torch.cat([v["object_center_2d"] for v in targets], dim=0)  # shape: [N, 2]
            # Compute the L1 cost between object keypoints (projected uv coords derived from xy translations)
            cost_keypoint = torch.cdist(out_keypoint, tgt_keypoint, p=1) 
        else:
            cost_keypoint = 0.0
        # Compute the classification cost. Contrary to the loss, we don't use the NLL,
        # but approximate it in 1 - proba[target class].
        # The 1 is a constant that doesn't change the matching, it can be ommitted.
        cost_class = -out_prob[:, tgt_ids]
        # Compute the L1 cost between boxes
        cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)
        # Compute the giou cost betwen boxes
        cost_giou = -generalized_box_iou(box_cxcywh_to_xyxy(out_bbox), box_cxcywh_to_xyxy(tgt_bbox))

        
        # Add to total cost
        # Initialize final cost matrix
        C = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou + self.cost_keypoint * cost_keypoint
        
        C = C.view(bs, num_queries, -1).cpu()

        sizes = [len(v["boxes"]) for v in targets]
        indices = []
        g_num_queries = num_queries // group_detr
        C_list = C.split(g_num_queries, dim=1)
        for g_i in range(group_detr):
            C_g = C_list[g_i]
            indices_g = [linear_sum_assignment(c[i]) for i, c in enumerate(C_g.split(sizes, -1))]
            if g_i == 0:
                indices = indices_g
            else:
                indices = [
                    (np.concatenate([indice1[0], indice2[0] + g_num_queries * g_i]), np.concatenate([indice1[1], indice2[1]]))
                    for indice1, indice2 in zip(indices, indices_g)
                ]
        # 
        # indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
        return [(torch.as_tensor(i, dtype=torch.int64), 
                 torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]


class Matcher6DRotTrans(nn.Module):
    """Hungarian matcher for 6D pose estimation with explicit pose costs."""

    def __init__(self,
                 cost_class: float = 1,
                 cost_bbox: float = 1,
                 cost_giou: float = 1,
                 cost_rotation: float = 1,
                 cost_translation: float = 1,
                 cost_keypoint: float = 1,
                 rotation_representation: str = '6d',
                 matcher_symmetry_stride: int = 1):
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        self.cost_rotation = cost_rotation
        self.cost_translation = cost_translation
        self.cost_keypoint = cost_keypoint
        self.rotation_representation = rotation_representation
        self.matcher_symmetry_stride = max(1, int(matcher_symmetry_stride))

        assert (
            cost_class != 0
            or cost_bbox != 0
            or cost_giou != 0
            or cost_rotation != 0
            or cost_translation != 0
            or cost_keypoint != 0
        ), "all costs cant be 0"

    @staticmethod
    def rotation_geodesic_distance_cost(pred_rotations, gt_rotations, eps=1e-7):
        pred_expanded = pred_rotations.unsqueeze(1)
        gt_expanded = gt_rotations.unsqueeze(0)
        relative_rotations = torch.matmul(pred_expanded.transpose(-2, -1), gt_expanded)
        traces = relative_rotations.diagonal(dim1=-2, dim2=-1).sum(-1)
        cos_angle = (traces - 1.0) / 2.0
        cos_angle = torch.clamp(cos_angle, -1.0 + eps, 1.0 - eps)
        return torch.acos(cos_angle)

    @staticmethod
    def rotation_symmetry_transform_min_cost(pred_rotations, gt_rotations, symmetry_transforms, symmetry_stride=1, eps=1e-6):
        columns = []
        for gt_rotation, transforms in zip(gt_rotations, symmetry_transforms):
            transforms = transforms[::symmetry_stride]
            equivalent_rotations = gt_rotation.unsqueeze(0) @ transforms
            relative_rotations = pred_rotations.unsqueeze(1).transpose(-2, -1) @ equivalent_rotations.unsqueeze(0)
            traces = relative_rotations.diagonal(dim1=-2, dim2=-1).sum(-1)
            cos_angle = ((traces - 1.0) / 2.0).clamp(-1.0 + eps, 1.0 - eps)
            columns.append(torch.acos(cos_angle).min(dim=1).values)
        return torch.stack(columns, dim=1)

    @torch.no_grad()
    def forward(self, outputs, targets, group_detr=1):
        bs, num_queries = outputs["pred_logits"].shape[:2]

        out_prob = outputs["pred_logits"].flatten(0, 1).softmax(-1)
        out_bbox = outputs["pred_boxes"].flatten(0, 1)
        out_keypoint = outputs["pred_uv_norm"].flatten(0, 1)
        out_trans = outputs["pred_translations"].flatten(0, 1) if self.cost_translation != 0 else None
        out_rot = outputs["pred_rotations"].flatten(0, 1) if self.cost_rotation != 0 else None

        tgt_ids = torch.cat([v["labels"] for v in targets])
        tgt_bbox = torch.cat([v["boxes"] for v in targets])

        if ("object_center_2d" in targets[0]) and (targets[0]["object_center_2d"] is not None):
            tgt_keypoint = torch.cat([v["object_center_2d"] for v in targets], dim=0).to(
                device=out_keypoint.device, dtype=out_keypoint.dtype
            )
            cost_keypoint = torch.cdist(out_keypoint, tgt_keypoint, p=1)
        else:
            cost_keypoint = 0.0

        cost_class = -out_prob[:, tgt_ids]
        cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)
        cost_giou = -generalized_box_iou(
            box_cxcywh_to_xyxy(out_bbox),
            box_cxcywh_to_xyxy(tgt_bbox),
        )
        if self.cost_translation != 0:
            tgt_trans = torch.cat([v["relative_position"] for v in targets], dim=0).to(
                device=out_trans.device, dtype=out_trans.dtype
            )
            cost_translation = torch.cdist(out_trans, tgt_trans, p=2)
            # Consider smooth l1 distacne ? 
            # out_trans: (B, M, D), tgt_trans: (B, N, D)
            # diff = out_trans.unsqueeze(2) - tgt_trans.unsqueeze(1)  # (B, M, N, D)
            # cost_translation = F.smooth_l1_loss(
            #     diff, 
            #     torch.zeros_like(diff), 
            #     reduction='none', 
            #     beta=1.0
            # ).sum(dim=-1)  # (B, M, N)
        else:
            cost_translation = 0.0
        if self.cost_rotation != 0:
            if self.rotation_representation == "sarr":
                tgt_rot = torch.cat([v["relative_rotation_sarr"] for v in targets], dim=0).to(
                    device=out_rot.device, dtype=out_rot.dtype
                )
                cost_rotation = pairwise_cosine_distance_cost(
                    normalize_sarr_pairs(out_rot),
                    normalize_sarr_pairs(tgt_rot),
                )
            else:
                tgt_rot = torch.cat([v["relative_rotation"] for v in targets], dim=0).to(
                    device=out_rot.device, dtype=out_rot.dtype
                )
                if "symmetry_transforms" in targets[0]:
                    tgt_symmetry_transforms = torch.cat([v["symmetry_transforms"] for v in targets], dim=0).to(
                        device=out_rot.device, dtype=out_rot.dtype
                    )
                    cost_rotation = self.rotation_symmetry_transform_min_cost(
                        out_rot,
                        tgt_rot,
                        tgt_symmetry_transforms,
                        symmetry_stride=self.matcher_symmetry_stride,
                    )
                else:
                    cost_rotation = self.rotation_geodesic_distance_cost(out_rot, tgt_rot)
        else:
            cost_rotation = 0.0

        C = (
            self.cost_bbox * cost_bbox
            + self.cost_class * cost_class
            + self.cost_giou * cost_giou
            + self.cost_keypoint * cost_keypoint
            + self.cost_translation * cost_translation
            + self.cost_rotation * cost_rotation
        )
        C = C.view(bs, num_queries, -1).cpu()

        sizes = [len(v["boxes"]) for v in targets]
        indices = []
        g_num_queries = num_queries // group_detr
        C_list = C.split(g_num_queries, dim=1)
        for g_i in range(group_detr):
            C_g = C_list[g_i]
            indices_g = [linear_sum_assignment(c[i]) for i, c in enumerate(C_g.split(sizes, -1))]
            if g_i == 0:
                indices = indices_g
            else:
                indices = [
                    (
                        np.concatenate([indice1[0], indice2[0] + g_num_queries * g_i]),
                        np.concatenate([indice1[1], indice2[1]]),
                    )
                    for indice1, indice2 in zip(indices, indices_g)
                ]
        return [
            (torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64))
            for i, j in indices
        ]


class MatcherAblation(nn.Module):
    """Two-stage matcher: 2D detection matching followed by pose matching."""

    def __init__(self,
                 cost_class: float = 1,
                 cost_bbox: float = 1,
                 cost_giou: float = 1,
                 cost_rotation: float = 1,
                 cost_translation: float = 1,
                 cost_keypoint: float = 1,
                 topk_candidates: int = 3,
                 rotation_representation: str = '6d'):
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        self.cost_rotation = cost_rotation
        self.cost_translation = cost_translation
        self.cost_keypoint = cost_keypoint
        self.rotation_representation = rotation_representation
        if topk_candidates < 1:
            raise ValueError("topk_candidates must be >= 1")
        self.topk_candidates = topk_candidates

    @staticmethod
    def rotation_geodesic_distance_cost(pred_rotations, gt_rotations, eps=1e-7):
        pred_expanded = pred_rotations.unsqueeze(1)
        gt_expanded = gt_rotations.unsqueeze(0)
        relative_rotations = torch.matmul(pred_expanded.transpose(-2, -1), gt_expanded)
        traces = relative_rotations.diagonal(dim1=-2, dim2=-1).sum(-1)
        cos_angle = (traces - 1.0) / 2.0
        cos_angle = torch.clamp(cos_angle, -1.0 + eps, 1.0 - eps)
        return torch.acos(cos_angle)

    def _build_detection_cost(self, outputs, targets):
        out_prob = outputs["pred_logits"].flatten(0, 1).softmax(-1)
        out_bbox = outputs["pred_boxes"].flatten(0, 1)
        out_keypoint = outputs["pred_uv_norm"].flatten(0, 1)

        tgt_ids = torch.cat([v["labels"] for v in targets])
        tgt_bbox = torch.cat([v["boxes"] for v in targets])

        if ("object_center_2d" in targets[0]) and (targets[0]["object_center_2d"] is not None):
            tgt_keypoint = torch.cat([v["object_center_2d"] for v in targets], dim=0).to(
                device=out_keypoint.device, dtype=out_keypoint.dtype
            )
            cost_keypoint = torch.cdist(out_keypoint, tgt_keypoint, p=1)
        else:
            cost_keypoint = 0.0

        cost_class = -out_prob[:, tgt_ids]
        cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)
        cost_giou = -generalized_box_iou(
            box_cxcywh_to_xyxy(out_bbox),
            box_cxcywh_to_xyxy(tgt_bbox),
        )

        return (
            self.cost_bbox * cost_bbox
            + self.cost_class * cost_class
            + self.cost_giou * cost_giou
            + self.cost_keypoint * cost_keypoint
        )

    def _build_pose_cost(self, outputs, targets):
        out_trans = outputs["pred_translations"].flatten(0, 1)
        out_rot = outputs["pred_rotations"].flatten(0, 1)

        tgt_trans = torch.cat([v["relative_position"] for v in targets], dim=0).to(
            device=out_trans.device, dtype=out_trans.dtype
        )
        if self.rotation_representation == "sarr":
            tgt_rot = torch.cat([v["relative_rotation_sarr"] for v in targets], dim=0).to(
                device=out_rot.device, dtype=out_rot.dtype
            )
        else:
            tgt_rot = torch.cat([v["relative_rotation"] for v in targets], dim=0).to(
                device=out_rot.device, dtype=out_rot.dtype
            )

        cost_translation = torch.cdist(out_trans, tgt_trans, p=2)
        if self.rotation_representation == "sarr":
            cost_rotation = pairwise_cosine_distance_cost(
                normalize_sarr_pairs(out_rot),
                normalize_sarr_pairs(tgt_rot),
            )
        else:
            cost_rotation = self.rotation_geodesic_distance_cost(out_rot, tgt_rot)
        return self.cost_translation * cost_translation + self.cost_rotation * cost_rotation

    def _build_candidate_mask(self, det_cost_img):
        """Keep the stage-1 detection matches plus per-target top-k detection candidates."""
        det_indices = linear_sum_assignment(det_cost_img.numpy())

        k = min(self.topk_candidates, det_cost_img.shape[0])
        topk_idx = torch.topk(det_cost_img, k=k, dim=0, largest=False).indices

        candidate_mask = torch.zeros_like(det_cost_img, dtype=torch.bool)
        candidate_mask.scatter_(0, topk_idx, True)
        candidate_mask[det_indices[0], det_indices[1]] = True
        return det_indices, candidate_mask

    @torch.no_grad()
    def forward(self, outputs, targets, group_detr=1):
        bs, num_queries = outputs["pred_logits"].shape[:2]

        detection_cost = self._build_detection_cost(outputs, targets)
        pose_cost = self._build_pose_cost(outputs, targets)

        C_det = detection_cost.view(bs, num_queries, -1).cpu()
        C_pose = pose_cost.view(bs, num_queries, -1).cpu()

        sizes = [len(v["boxes"]) for v in targets]
        indices = []
        g_num_queries = num_queries // group_detr
        det_groups = C_det.split(g_num_queries, dim=1)
        pose_groups = C_pose.split(g_num_queries, dim=1)

        for g_i in range(group_detr):
            C_det_g = det_groups[g_i]
            C_pose_g = pose_groups[g_i]
            split_det = C_det_g.split(sizes, -1)
            split_pose = C_pose_g.split(sizes, -1)
            indices_g = []

            for i, (det_cost_img, pose_cost_img) in enumerate(zip(split_det, split_pose)):
                det_cost_img = det_cost_img[i]
                pose_cost_img = pose_cost_img[i]
                n_targets = det_cost_img.shape[1]

                if n_targets == 0:
                    indices_g.append((np.array([], dtype=np.int64), np.array([], dtype=np.int64)))
                    continue

                _, candidate_mask = self._build_candidate_mask(det_cost_img)

                gated_pose_cost = pose_cost_img.clone()
                invalid_cost = pose_cost_img.max().item() + 1e6
                gated_pose_cost[~candidate_mask] = invalid_cost
                indices_g.append(linear_sum_assignment(gated_pose_cost.numpy()))

            if g_i == 0:
                indices = indices_g
            else:
                indices = [
                    (
                        np.concatenate([indice1[0], indice2[0] + g_num_queries * g_i]),
                        np.concatenate([indice1[1], indice2[1]]),
                    )
                    for indice1, indice2 in zip(indices, indices_g)
                ]

        return [
            (torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64))
            for i, j in indices
        ]

class MatcherYOPO(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network
    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(self, cost_class: float = 1, 
                 cost_bbox: float = 1, 
                 cost_giou: float = 1, 
                 cost_trans: float = 1,
                 cost_rot: float = 1,
                 rotation_representation: str = '6d',
        ):
        """Creates the matcher
        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_bbox: This is the relative weight of the L1 error of the bounding box coordinates in the matching cost
            cost_giou: This is the relative weight of the giou loss of the bounding box in the matching cost
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        self.cost_trans = cost_trans
        self.cost_rot = cost_rot
        self.rotation_representation = rotation_representation
        assert cost_class != 0 or cost_bbox != 0 or cost_giou != 0, "all costs cant be 0"
    
    def rotation_geodesic_distance_cost(self, pred_rotations, gt_rotations):
        """
        Compute geodesic distance cost matrix between two sets of rotation matrices.
        
        Args:
            pred_rotations: torch.Tensor of shape [n, 3, 3] - predicted rotations
            gt_rotations: torch.Tensor of shape [m, 3, 3] - ground truth rotations
        
        Returns:
            cost_matrix: torch.Tensor of shape [n, m] - pairwise geodesic distances
        """
       
        
        # Expand dimensions for broadcasting: [n, 1, 3, 3] and [1, m, 3, 3]
        pred_expanded = pred_rotations.unsqueeze(1)  # [n, 1, 3, 3]
        gt_expanded = gt_rotations.unsqueeze(0)      # [1, m, 3, 3]
        
        # Compute R_pred^T @ R_gt for all pairs
        relative_rotations = torch.matmul(
            pred_expanded.transpose(-2, -1),  # [n, 1, 3, 3]
            gt_expanded                        # [1, m, 3, 3]
        )  # Result: [n, m, 3, 3]
        
        # Compute trace of each matrix
        traces = relative_rotations.diagonal(dim1=-2, dim2=-1).sum(-1)  # [n, m]
        
        # Geodesic distance: arccos((trace(R) - 1) / 2)
        # Clamp to avoid numerical issues with arccos
        cos_angle = (traces - 1.0) / 2.0
        cos_angle = torch.clamp(cos_angle, -1.0, 1.0)
        
        geodesic_distances = torch.acos(cos_angle)  # [n, m] in radians
        
        return geodesic_distances

    def rotation_geodesic_distance(self, R1, R2, eps=1e-7):
        """
        R1, R2: (..., 3, 3)
        returns: (...,) angle in radians
        """
        R = R1.transpose(-1, -2) @ R2
        trace = R.diagonal(offset=0, dim1=-2, dim2=-1).sum(-1)
        cos_theta = (trace - 1.0) / 2.0
        cos_theta = torch.clamp(cos_theta, -1.0 + eps, 1.0 - eps)
        return torch.acos(cos_theta)
    
    @torch.no_grad()
    def forward(self, outputs, targets, group_detr=1):
        """ Performs the matching
        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                 "pred_boxes": Tensor of dim [batch_size, num_queries, 4] with the predicted box coordinates
            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                           objects in the target) containing the class labels
                 "boxes": Tensor of dim [num_target_boxes, 4] containing the target box coordinates
            group_detr: Number of groups used for matching.
        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        bs, num_queries = outputs["pred_logits"].shape[:2]

        # We flatten to compute the cost matrices in a batch
        out_prob = outputs["pred_logits"].flatten(0, 1).sigmoid()  # [batch_size * num_queries, num_classes]
        out_bbox = outputs["pred_boxes"].flatten(0, 1)  # [batch_size * num_queries, 4]
        out_trans = outputs["pred_translations"].flatten(0, 1)
        out_rot = outputs["pred_rotations"].flatten(0, 1)
        dev = out_prob.device
        dt = out_prob.dtype
        # Also concat the target labels, boxes, rotations and translations
        tgt_ids = torch.cat([v["labels"] for v in targets])
        tgt_bbox = torch.cat([v["boxes"] for v in targets])
        tgt_trans = torch.cat([v["relative_position"] for v in targets]).to(dev, dt) 
        if self.rotation_representation == "sarr":
            tgt_rot = torch.cat([v["relative_rotation_sarr"] for v in targets]).to(dev, out_rot.dtype)
        else:
            tgt_rot =  torch.cat([v["relative_rotation"] for v in targets]).to(dev, out_rot.dtype) 
        # Compute the giou cost betwen boxes
        giou = generalized_box_iou(box_cxcywh_to_xyxy(out_bbox), box_cxcywh_to_xyxy(tgt_bbox))
        cost_giou = -giou

        # Compute the classification cost.
        alpha = 0.25
        gamma = 2.0
        
        neg_cost_class = (1 - alpha) * (out_prob ** gamma) * (-(1 - out_prob + 1e-8).log())
        pos_cost_class = alpha * ((1 - out_prob) ** gamma) * (-(out_prob + 1e-8).log())
        cost_class = pos_cost_class[:, tgt_ids] - neg_cost_class[:, tgt_ids]

        # Compute the L1 cost between boxes
        cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)
        # Compute the L2 cost between translation vectors
        cost_trans = torch.cdist(out_trans, tgt_trans, p=2)
        # Compute the geodesic cost between rotation matrices (3x3)
        if self.rotation_representation == "sarr":
            cost_rot = pairwise_cosine_distance_cost(
                normalize_sarr_pairs(out_rot),
                normalize_sarr_pairs(tgt_rot),
            )
        else:
            cost_rot = self.rotation_geodesic_distance_cost(pred_rotations=out_rot, gt_rotations=tgt_rot)
        # Final cost matrix
        C = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou + self.cost_trans * cost_trans + self.cost_rot * cost_rot
        C = C.view(bs, num_queries, -1).cpu()

        sizes = [len(v["boxes"]) for v in targets]
        indices = []
        g_num_queries = num_queries // group_detr
        C_list = C.split(g_num_queries, dim=1)
        for g_i in range(group_detr):
            C_g = C_list[g_i]
            indices_g = [linear_sum_assignment(c[i]) for i, c in enumerate(C_g.split(sizes, -1))]
            if g_i == 0:
                indices = indices_g
            else:
                indices = [
                    (np.concatenate([indice1[0], indice2[0] + g_num_queries * g_i]), np.concatenate([indice1[1], indice2[1]]))
                    for indice1, indice2 in zip(indices, indices_g)
                ]
        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]


def build_matcher(args):
    if args.matcher_type == 'hungarian':
        return HungarianMatcher(
            cost_class=args.set_cost_class,
            cost_bbox=args.set_cost_bbox,
            cost_giou=args.set_cost_giou,
            focal_alpha=args.focal_alpha,)
    elif args.matcher_type == '6d':
        return Matcher6D(cost_bbox=args.set_cost_bbox, 
                         cost_class=args.set_cost_class, 
                         cost_rotation=args.set_cost_rotation,
                         cost_keypoint=args.set_cost_keypoint,
                         rotation_representation=args.rotation_representation)
    elif args.matcher_type == '6d_rot_trans':
        return Matcher6DRotTrans(
            cost_bbox=args.set_cost_bbox,
            cost_class=args.set_cost_class,
            cost_giou=args.set_cost_giou,
            cost_rotation=args.set_cost_rotation,
            cost_translation=args.set_cost_translation,
            cost_keypoint=args.set_cost_keypoint,
            rotation_representation=args.rotation_representation,
            matcher_symmetry_stride=getattr(args, 'matcher_symmetry_stride', 1),
        )
    elif args.matcher_type == 'ablation':
        return MatcherAblation(
            cost_bbox=args.set_cost_bbox,
            cost_class=args.set_cost_class,
            cost_giou=args.set_cost_giou,
            cost_rotation=args.set_cost_rotation,
            cost_translation=args.set_cost_translation,
            cost_keypoint=args.set_cost_keypoint,
            topk_candidates=args.ablation_topk_candidates,
            rotation_representation=args.rotation_representation,
        )
    elif args.matcher_type == 'yopo':
        return MatcherYOPO(cost_bbox=args.set_cost_bbox,
                           cost_class=args.set_cost_class,
                           cost_giou=args.set_cost_giou,
                           cost_trans=args.set_cost_translation,
                           cost_rot=args.set_cost_rotation,
                           rotation_representation=args.rotation_representation)
