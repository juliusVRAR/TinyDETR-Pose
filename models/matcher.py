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
from scipy.optimize import linear_sum_assignment
from torch import nn
from util.box_ops import box_cxcywh_to_xyxy, generalized_box_iou
from util.rotation_utils import rotation_matrix_to_gram_schmidt_6d, rotation_matrix_to_raw_6d, rotation_6d_to_matrix, rotation_6d_simple_to_matrix

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

class PoseMatcher(nn.Module):
    """
    This class computes an assignment between the network's predictions and targets. The matching is
    done based on the predicted bounding boxes. However, the predicted class is used to remove matches if the class is
    off.
    """

    def __init__(self,
                 cost_bbox: float = 1,
                 cost_class: float = 1,
                 bbox_mode: str = "gt",
                 class_mode: str = "specific"):
        """
        cost_bbox: weighting parameter for the bounding box cost
        cost_class: weighting parameter for the class cost
        bbox_mode: mode with which the bounding box information was fed to the transformer part of PoET
        class_mode: determines whether PoET is used in a class specific or agnostic way
        """
        super().__init__()
        self.cost_bbox = cost_bbox
        self.cost_class = cost_class
        self.bbox_mode = bbox_mode
        self.class_mode = class_mode

    def forward(self, outputs, targets, n_boxes, giou_thresh=0.5):
        """ Performs the matching

                Params:
                    outputs: This is a dict that contains at least these entries:
                         "pred_translation": Tensor of dim [batch_size, num_queries, 3 (*n_classes)] with the predicted translation
                         "pred_rotation": Tensor of dim [batch_size, num_queries, rot_dim (*n_classes)] with the predicted rotations
                         "pred_boxes": Tensor of dim [batch_size, num_queries, 4] with the predicted box coordinates
                         "pred_classes": Tensor of dim [batch_size, num_queries, 1] with the predicted classes


                    targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                         "relative_pose":
                            "position": Tensor of dim [num_target_boxes, 3 (*n_classes)] containing the target translation
                            "rotation": Tensor of dim [num_target_boxes, rot_dim (*n_classes)] containing the target rotation
                         "boxes": Tensor of dim [num_target_boxes, 4] containing the target box coordinates
                         "labels": Tensor of dim [num_target_boxes, 1] containing the target labels

                    n_boxes: This is a list of number of boxes (len(n_boxes) = batch_size) predicted per image.

                    giou_thresh: threshold value that the generalized IoU between predicted and target box
                    have to have for the matching

                Returns:
                    A list of size batch_size, containing tuples of (index_i, index_j) where:
                        - index_i is the indices of the selected predictions (in order)
                        - index_j is the indices of the corresponding selected targets (in order)
                    For each batch element, it holds:
                        len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
                """
        with torch.no_grad():
            bs, num_queries = outputs["pred_boxes"].shape[:2]

            # Flatten to compute cost matrices in a batch
            out_bbox = outputs["pred_boxes"].flatten(0, 1)  # [batch_size * num_queries, 4]
            out_class = outputs["pred_classes"].flatten(0, 1)

            # Concat target boxes
            tgt_bbox = torch.cat([t["boxes"] for t in targets])
            tgt_class = torch.cat([t["labels"].type(torch.float32) for t in targets])

            # Compute L1 cost between box centers
            cost_bbox = torch.cdist(out_bbox[:, 0:2], tgt_bbox[:, 0:2], p=1)

            # Compute classification cost
            cost_class = []
            for cls in out_class:
                cost_class.append(torch.where(cls == tgt_class, 0., 1.))
            cost_class = torch.stack(cost_class)

            # Final cost matrix
            # TODO: Find a better weighting between bounding box and class cost // e.g. normalization
            C = self.cost_bbox * cost_bbox + self.cost_class * cost_class
            C = C.view(bs, num_queries, -1).cpu()

            # PoET adds dummy query embeddings to allow for batch processing.
            # The transformer does not change the order of the queries, hence the indices of the dummy embeddings are known
            # Filter them out by taking only the first n_boxes boxes predicted per image in the batch
            sizes = [len(v["boxes"]) for v in targets]
            indices = []
            
            # Calculate the generalized IoU and remove matches if the boxes do not overlap at all --> no prediction
            new_indices = []
            for b, (out_box, out_cls, tgt) in enumerate(zip(out_bbox.split(num_queries), out_class.split(num_queries), targets)):
                tgt_box = tgt["boxes"]
                tgt_cls = tgt["labels"]
                gious = generalized_box_iou(box_cxcywh_to_xyxy(out_box[:n_boxes[b]]), box_cxcywh_to_xyxy(tgt_box))
                new_src_idx = []
                new_tgt_idx = []
                for idx, (i, j) in enumerate(zip(indices[b][0], indices[b][1])):
                    giou = gious[i, j]
                    if giou < giou_thresh:
                        # print("Match removed GIoU: {}".format(giou))
                        continue
                    else:
                        new_src_idx.append(i)
                        new_tgt_idx.append(j)
                new_indices.append((np.array(new_src_idx), np.array(new_tgt_idx)))
            indices = new_indices

            return [(torch.as_tensor(i, dtype=torch.int64), 
                     torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]

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
                 cost_translation: float = 1, 
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
        self.cost_translation = cost_translation
        self.rotation_representation = rotation_representation
        
        assert cost_class != 0 or cost_bbox != 0 or cost_giou != 0, "all costs cant be 0"

    # Compute distance of rotation in gram-schmidt representation
    def compute_rotation_cost(self, pred_rotations, target_rotations):
        """
        Pairwise L2 distance between rotation matrices.
        """
        if pred_rotations.shape[-1] != 6:
            pred_rotations = rotation_matrix_to_gram_schmidt_6d(pred_rotations)
        pred_expanded = pred_rotations.unsqueeze(1)  # [num_queries, 1, 6]
        target_expanded = target_rotations.unsqueeze(0)  # [1, num_targets, 6]
        rotation_cost = torch.norm(pred_expanded - target_expanded, p=2, dim=-1)

        return rotation_cost

    # From T6D: Lrot is the angular distance between the ground truth and predicted rotations
    def compute_rotation_cost_angular(self, pred_rotations: torch.Tensor, 
                                       target_rotations: torch.Tensor,
                                       eps: float = 1e-6):
        """
        Compute pairwise angular distance (radians) between rotation matrices.
        pred_rotations: (P, 3, 3) or (P, 6)
        target_rotations: (T, 3, 3) or (T, 6)
        eps: is a small safety margin to prevent numerical issues when taking acos.
        After floating point ops cos_theta can slightly exceed [-1,1]; clamping to [-1+eps, 1-eps] avoids NaNs (acos undefined outside [-1,1]) and unstable gradients at exactly ±1 (where angle derivative blows up). 
        Smaller eps → tighter range but higher risk of NaNs; 1e-6 is a typical choice.
        Returns:
            cost: (P, T) where cost[p,t] = angle(R_p, R_t)
        For two rotation matrices R₁ and R₂, the angle θ between them is: angle = arccos( (trace(R_p * R_t^T) - 1) / 2 ) 
        """
        # Expand for pairwise multiplication
        if pred_rotations.shape[-1] == 6:
            # Convert rotation matrices to 6D representation
            R_pred = rotation_6d_simple_to_matrix(pred_rotations) 
        else:
            R_pred = pred_rotations  # (P,3,3)
        if target_rotations.shape[-1] == 6:
            R_tgt  = rotation_6d_simple_to_matrix(target_rotations)
        else:
            R_tgt  = target_rotations  # (T,3,3)
        # Pairwise relative rotations
        R_pred_exp = R_pred.unsqueeze(1)                    # (P,1,3,3)
        R_tgt_exp  = R_tgt.unsqueeze(0)                     # (1,T,3,3)
        # Relative rotation R_rel = R_pred * R_tgt^T
        R_rel = R_pred_exp @ R_tgt_exp.transpose(-1, -2)    # (P,T,3,3)
        trace = R_rel[..., 0, 0] + R_rel[..., 1, 1] + R_rel[..., 2, 2]
        cos_theta = (trace - 1.0) * 0.5
        cos_theta = cos_theta.clamp(-1.0 + eps, 1.0 - eps)
        angle = torch.acos(cos_theta)
        return angle
    
    # Surrogate preserves ordering (smaller cost ↔ closer rotations) without expensive acos.
    def compute_rotation_cost_surrogate(self,
                                        pred_rotations: torch.Tensor,
                                        target_rotations: torch.Tensor,
                                        eps: float = 1e-6,
                                        use_geodesic: bool = False) -> torch.Tensor:
        """
        Rotation cost between all predicted and target rotations.

        Inputs:
            pred_rotations: (P,6) or (P,3,3)
            target_rotations: (T,6) or (T,3,3)
            eps: numerical clamp
            use_geodesic: if True returns angular distance (radians),
                          else returns surrogate cost (1 - cosθ) which is cheaper.

        Returns:
            cost: (P,T) rotation cost matrix
        """
        # Convert 6D to matrices if needed
        if pred_rotations.shape[-1] == 6 and pred_rotations.dim() == 2:
            R_pred = rotation_6d_simple_to_matrix(pred_rotations)      # (P,3,3)
        else:
            R_pred = pred_rotations
        if target_rotations.shape[-1] == 6 and target_rotations.dim() == 2:
            R_tgt = rotation_6d_simple_to_matrix(target_rotations)     # (T,3,3)
        else:
            R_tgt = target_rotations

        # Broadcast
        R_pred_exp = R_pred.unsqueeze(1)              # (P,1,3,3)
        R_tgt_exp  = R_tgt.unsqueeze(0)               # (1,T,3,3)

        # Relative rotation
        R_rel = R_pred_exp @ R_tgt_exp.transpose(-1, -2)  # (P,T,3,3)

        # Trace
        trace = R_rel[..., 0, 0] + R_rel[..., 1, 1] + R_rel[..., 2, 2]
        cos_theta = (trace - 1.0) * 0.5
        cos_theta = cos_theta.clamp(-1.0 + eps, 1.0 - eps)

        if use_geodesic:
            angle = torch.acos(cos_theta)          # (P,T)
            return angle
        else:
            # Monotonic surrogate (no acos) in [0,2]
            return 1.0 - cos_theta                 # (P,T)

    # TODO: What works best for translation cost L1 or L2?
    def compute_translation_cost_l1(self, pred_translations, target_translations):
        """Compute L1 translation cost"""
        pred_expanded = pred_translations.unsqueeze(1)  # [num_queries, 1, 3]
        target_expanded = target_translations.unsqueeze(0)  # [1, num_targets, 3]
        translation_cost = torch.norm(pred_expanded - target_expanded, p=1, dim=-1)
        return translation_cost
    
    # TODO: T6D uses L2 loss for translation cost
    def compute_translation_cost_l2(self, pred_translations, target_translations):
        """Compute L2 translation cost (Euclidean distance) using torch.cdist."""
        # pred_translations: (P,3), target_translations: (T,3)
        # Returns: (P,T) matrix of Euclidean distances
        return torch.cdist(pred_translations, target_translations, p=2)
    
    #TODO: YOLOX6d approach for translation cost. We split yx and z of the tranlation vector
    def compute_translation_xy_cost(self, pred_xy, target_xy):
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
    
    def compute_translation_z_cost(self, pred_z, target_z, epsilon=1e-6):
        """
        pred_z: [num_queries, 1]
        target_z: [num_targets, 1]
        """
        # L2 distance
        cost_z = torch.cdist(pred_z, target_z, p=2)
        
        return cost_z  # [num_queries, num_targets]
    
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
        
        # TODO: add support for yx and z translation predictions like YOLOX6D
        # Handle 6D pose predictions if available
        if self.rotation_representation == '6d':
            out_rotation = outputs["pred_rotations"].flatten(0, 1)  # [batch_size * num_queries, 9]
            out_translation = outputs["pred_translations"].flatten(0, 1)  # [batch_size * num_queries, 3]
        # Also concat the target labels and boxes
        tgt_ids = torch.cat([v["labels"] for v in targets])
        tgt_bbox = torch.cat([v["boxes"] for v in targets])
       
        # Handle 6D pose targets in GrammSchmidt representation
        if self.rotation_representation == "6d":
            tgt_rotation = torch.cat([v["relative_rotation"] for v in targets])
            # TODO: Check in dataloader which 6d conversion we need here (gram-schmidt or raw 6d)
            tgt_rotation = torch.cat([v["relative_rotation_gs"] for v in targets])  # shape: [N, 6]
        else: 
            NotImplementedError("Only 6D rotation representation is currently supported in Matcher6D")
        
        tgt_translation = torch.cat([v["relative_position"] for v in targets], dim=0)  # shape: [N, 2]

        # Compute the classification cost. Contrary to the loss, we don't use the NLL,
        # but approximate it in 1 - proba[target class].
        # The 1 is a constant that doesn't change the matching, it can be ommitted.
        cost_class = -out_prob[:, tgt_ids]

        # Compute the L1 cost between boxes
        cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)
        # TODO: Test if this performs better this is what poet does
        # Compute L1 cost between box centers
        # cost_bbox = torch.cdist(out_bbox[:, 0:2], tgt_bbox[:, 0:2], p=1)
        # Compute the giou cost betwen boxes
        cost_giou = -generalized_box_iou(box_cxcywh_to_xyxy(out_bbox), box_cxcywh_to_xyxy(tgt_bbox))

        # Initialize final cost matrix
        C = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou
        
        # Add 6D pose costs
        if self.rotation_representation == "6d" and "pred_rotations" in outputs and "pred_translations" in outputs:
            #cost_rotation = self.compute_rotation_cost(pred_rotations=out_rotation, 
            #                                               target_rotations=tgt_rotation)
            # From T6D paper
            cost_rotation = self.compute_rotation_cost_angular(pred_rotations=out_rotation,
                                                                    target_rotations=tgt_rotation)
            # Compute translation cost (l1)
            #cost_translation = self.compute_translation_cost_l1(out_translation, 
            #                                                   tgt_translation)
            # Compute translation cost (l2) T6D does this
            #cost_translation_l2 = self.compute_translation_cost_l2(out_translation, 
            #                                                    tgt_translation)
            cost_trans_xy = self.compute_translation_xy_cost(out_translation[:, :2],
                                                            tgt_translation[:, :2])
            cost_trans_z = self.compute_translation_z_cost(out_translation[:, 2:3],
                                                          tgt_translation[:, 2:3])
            lambda_xy = 1.0
            lambda_z = 1.0
            cost_translation = (lambda_xy * cost_trans_xy) + (lambda_z * cost_trans_z)

            # Add to total cost
            C += self.cost_rotation * cost_rotation + self.cost_translation * cost_translation

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

def build_matcher(args):
    if args.matcher_type == 'hungarian':
        return HungarianMatcher(
            cost_class=args.set_cost_class,
            cost_bbox=args.set_cost_bbox,
            cost_giou=args.set_cost_giou,
            focal_alpha=args.focal_alpha,)
    elif args.matcher_type == 'pose':
        return PoseMatcher(cost_bbox=args.set_cost_bbox, 
                           cost_class=args.set_cost_class, 
                           bbox_mode=args.bbox_mode,
                           class_mode=args.class_mode)
    elif args.matcher_type == '6d':
        return Matcher6D(cost_bbox=args.set_cost_bbox, 
                         cost_class=args.set_cost_class, 
                         cost_rotation=args.set_cost_rotation,
                         cost_translation=args.set_cost_translation)