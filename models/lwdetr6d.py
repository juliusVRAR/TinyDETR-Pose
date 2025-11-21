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
LW-DETR model and criterion classes
"""
import copy
import math
from typing import Callable
import torch
import torch.nn.functional as F
from torch import nn

from util import box_ops
from util.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       accuracy, get_world_size,
                       is_dist_avail_and_initialized)
from util.rotation_utils import rotation_6d_to_matrix, rotation_6d_simple_to_matrix, rotation_matrix_to_raw_6d
from .backbone import build_backbone
from .matcher import build_matcher
from .transformer import build_transformer


class LWDETR6D(nn.Module):
    """ This is the Group DETR v3 module that performs object detection """
    def __init__(self,
                 backbone,
                 transformer,
                 num_classes,
                 num_queries,
                 aux_loss=False,
                 group_detr=1,
                 two_stage=False,
                 lite_refpoint_refine=False,
                 bbox_reparam=False,
                 rotation_mode='6d'):
        """ Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            num_classes: number of object classes
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         Conditional DETR can detect in a single image. For COCO, we recommend 100 queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
            group_detr: Number of groups to speed detr training. Default is 1.
            lite_refpoint_refine: TODO
        """
        super().__init__()
        self.num_queries = num_queries
        self.transformer = transformer
        hidden_dim = transformer.d_model
        self.class_embed = nn.Linear(hidden_dim, num_classes)
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)

        query_dim=4
        self.refpoint_embed = nn.Embedding(num_queries * group_detr, query_dim)
        self.query_feat = nn.Embedding(num_queries * group_detr, hidden_dim)
        nn.init.constant_(self.refpoint_embed.weight.data, 0)

        self.backbone = backbone
        self.aux_loss = aux_loss
        self.group_detr = group_detr

        # iter update
        self.lite_refpoint_refine = lite_refpoint_refine
        if not self.lite_refpoint_refine:
            self.transformer.decoder.bbox_embed = self.bbox_embed
        else:
            self.transformer.decoder.bbox_embed = None

        self.bbox_reparam = bbox_reparam
        ################6D Heads##################
        
        # New: 6D pose estimation head
        self.rotation_mode = rotation_mode
        # Determine Translation and Rotation head output dimension
        self.t_dim = 3
        self.xy_dim = 2
        self.z_dim = 1
        if self.rotation_mode == '6d':
            self.rot_dim = 6 # GramSchmidt
        else:
            raise NotImplementedError('Rotational representation is not supported.')
        
        self.dec_rot_head  = MLP(input_dim=hidden_dim, 
                                 hidden_dim=hidden_dim, 
                                 output_dim=self.rot_dim, 
                                 num_layers=3)
        
        # YOLOX6d approach: split translation into 2D center (xy) + depth (z)
        self.dec_trans_head = MLP(input_dim=hidden_dim, 
                                  hidden_dim=hidden_dim, 
                                  output_dim=hidden_dim, 
                                  num_layers=2)
        
        self.dec_trans_xy_head = MLP(input_dim=hidden_dim, 
                                     hidden_dim=hidden_dim, 
                                     output_dim=self.xy_dim, 
                                     num_layers=1)
        
        self.dec_trans_z_head = MLP(input_dim=hidden_dim, 
                                    hidden_dim=hidden_dim, 
                                    output_dim=self.z_dim, 
                                    num_layers=1)
        ############################################

        # init prior_prob setting for focal loss
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        self.class_embed.bias.data = torch.ones(num_classes) * bias_value

        # init bbox_mebed
        nn.init.constant_(self.bbox_embed.layers[-1].weight.data, 0)
        nn.init.constant_(self.bbox_embed.layers[-1].bias.data, 0)

        # two_stage
        self.two_stage = two_stage
        if self.two_stage:
            self.transformer.enc_out_bbox_embed = nn.ModuleList(
                [copy.deepcopy(self.bbox_embed) for _ in range(group_detr)])
            self.transformer.enc_out_class_embed = nn.ModuleList(
                [copy.deepcopy(self.class_embed) for _ in range(group_detr)])
            # New: encoder pose heads
            self.transformer.enc_out_rot_embed = nn.ModuleList(
                [copy.deepcopy(self.dec_rot_head) for _ in range(group_detr)])
            self.transformer.enc_out_trans_embed = nn.ModuleList(
                [copy.deepcopy(self.dec_trans_head) for _ in range(group_detr)])
            self.transformer.enc_out_trans_xy_embed = nn.ModuleList(
                [copy.deepcopy(self.dec_trans_xy_head) for _ in range(group_detr)])
            self.transformer.enc_out_trans_z_embed = nn.ModuleList(
                [copy.deepcopy(self.dec_trans_z_head) for _ in range(group_detr)])

        self._export = False

    def export(self):
        self._export = True
        self._forward_origin = self.forward
        self.forward = self.forward_export
        for name, m in self.named_modules():
            if hasattr(m, "export") and isinstance(m.export, Callable) and hasattr(m, "_export") and not m._export:
                m.export()

    def forward(self, samples: NestedTensor, targets=None):
        """ The forward expects a NestedTensor, which consists of:
               - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
               - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels

            It returns a dict with the following elements:
               - "pred_logits": the classification logits (including no-object) for all queries.
                                Shape= [batch_size x num_queries x num_classes]
               - "pred_boxes": The normalized boxes coordinates for all queries, represented as
                               (center_x, center_y, width, height). These values are normalized in [0, 1],
                               relative to the size of each individual image (disregarding possible padding).
                               See PostProcess for information on how to retrieve the unnormalized bounding box.
               - "aux_outputs": Optional, only returned when auxilary losses are activated. It is a list of
                                dictionnaries containing the two above keys for each decoder layer.
        """
        if isinstance(samples, (list, torch.Tensor)):
            samples = nested_tensor_from_tensor_list(samples)
        features, poss = self.backbone(samples)

        srcs = []
        masks = []
        for l, feat in enumerate(features):
            src, mask = feat.decompose()
            srcs.append(src)
            masks.append(mask)
            assert mask is not None

        if self.training:
            refpoint_embed_weight = self.refpoint_embed.weight
            query_feat_weight = self.query_feat.weight
        else:
            # only use one group in inference
            refpoint_embed_weight = self.refpoint_embed.weight[:self.num_queries]
            query_feat_weight = self.query_feat.weight[:self.num_queries]

        hs, ref_unsigmoid, hs_enc, ref_enc = self.transformer(
            srcs, masks, poss, refpoint_embed_weight, query_feat_weight)

        if self.bbox_reparam:
            outputs_coord_delta = self.bbox_embed(hs)
            outputs_coord_cxcy = outputs_coord_delta[..., :2] * ref_unsigmoid[..., 2:] + ref_unsigmoid[..., :2]
            outputs_coord_wh = outputs_coord_delta[..., 2:].exp() * ref_unsigmoid[..., 2:]
            outputs_coord = torch.concat(
                [outputs_coord_cxcy, outputs_coord_wh], dim=-1
            )
        else:
            outputs_coord = (self.bbox_embed(hs) + ref_unsigmoid).sigmoid()

        outputs_class = self.class_embed(hs)
        pred_rots = self.dec_rot_head(hs)
        
        if self.training:
            rot_c1 = F.normalize(pred_rots[:, :, :, :3], dim=3) 
            rot_c2 = F.normalize(pred_rots[:, :, :, 3:] - torch.sum(rot_c1 * pred_rots[:, :, :, 3:], dim=3, keepdim=True) * rot_c1, dim=3) 
        else:
            rot_c1 = (pred_rots[:, :, :, :3] / pred_rots[:, :, :, :3].norm(p=2, dim=3, keepdim=True))
            # Old version:
            # rot_c2 =  pred_rots[:, :, :, 3:] - torch.sum(rot_c1 * pred_rots[:, :, :, 3:], dim=2, keepdim=True) * rot_c1
            # Shouldnt this be dim=3 
            rot_c2 =  pred_rots[:, :, :, 3:] - torch.sum(rot_c1 * pred_rots[:, :, :, 3:], dim=3, keepdim=True) * rot_c1
            rot_c2 = rot_c2 / rot_c2.norm(p=2, dim=3, keepdim=True)

        # Gram-Schmidt representation
        output_rots = torch.cat([rot_c1, rot_c2], dim=3)
        output_trans = self.dec_trans_head(hs)
        output_trans_xy = self.dec_trans_xy_head(output_trans)
        output_trans_z = self.dec_trans_z_head(output_trans)
        output_trans = torch.cat([output_trans_xy, output_trans_z], dim=3)

        # Postprocess output_rots for loss
        output_rots = self.process_rotation(output_rots)
        out = {'pred_logits': outputs_class[-1], 
               'pred_boxes': outputs_coord[-1],
               'pred_rotations': output_rots[-1],
               'pred_translations': output_trans[-1]
               }
        
        if self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(outputs_class, 
                                                    outputs_coord, 
                                                    output_rots, 
                                                    output_trans)
        
        if self.two_stage:
            hs_enc_list = hs_enc.split(self.num_queries, dim=1)
            cls_enc, rot_enc_list, trans_enc_list = [], [], []
            group_detr = self.group_detr if self.training else 1
            for g_idx in range(group_detr):
                enc_feat = hs_enc_list[g_idx]
                cls_enc_gidx = self.transformer.enc_out_class_embed[g_idx](enc_feat)
                cls_enc.append(cls_enc_gidx)
                rot_enc_list.append(self.transformer.enc_out_rot_embed[g_idx](enc_feat))      # (B, num_queries, rot_dim)
                #TODO: Is this correct? We predict trans and put these into xy and z heads separately
                trans_enc = self.transformer.enc_out_trans_embed[g_idx](enc_feat)  # (B, num_queries, 3)
                trans_enc_xy = self.transformer.enc_out_trans_xy_embed[g_idx](trans_enc)  # (B, num_queries, 2)
                trans_enc_z = (self.transformer.enc_out_trans_z_embed[g_idx](trans_enc))  # (B, num_queries, 1)
                trans_enc_list.append(torch.cat([trans_enc_xy, trans_enc_z], dim=-1))  # (B, num_queries, 3)
            cls_enc = torch.cat(cls_enc, dim=1)
            rot_enc = torch.cat(rot_enc_list, dim=1)            # (B, total_queries, rot_dim)
            # TODO: How do we achive the same as in outputs above?
            trans_enc = torch.cat(trans_enc_list, dim=1)      # (B, total_queries, 3)
            
            # What axis are these ?
            rot_c1 = F.normalize(rot_enc[..., :3], dim=-1) 
            rot_c2 = F.normalize(rot_enc[..., 3:] - torch.sum(rot_c1 * rot_enc[..., :3], dim=-1, keepdim=True) * rot_c1, dim=-1) 
            rot_enc_full = torch.cat([rot_c1, rot_c2], dim=-1)          # (B, total_queries, 6)
            rot_enc_full = self.process_rotation(rot_enc_full) # (B, total_queries, 3, 3)
            out['enc_outputs'] = {
                'pred_logits': cls_enc,
                'pred_boxes': ref_enc,
                'pred_rotations': rot_enc_full,
                'pred_translations': trans_enc
            }
        return out

    def forward_export(self, tensors):
        srcs, _, poss = self.backbone(tensors)
        # only use one group in inference
        refpoint_embed_weight = self.refpoint_embed.weight[:self.num_queries]
        query_feat_weight = self.query_feat.weight[:self.num_queries]

        hs, ref_unsigmoid, hs_enc, ref_enc = self.transformer(
            srcs, None, poss, refpoint_embed_weight, query_feat_weight)

        if self.bbox_reparam:
            outputs_coord_delta = self.bbox_embed(hs)
            outputs_coord_cxcy = outputs_coord_delta[..., :2] * ref_unsigmoid[..., 2:] + ref_unsigmoid[..., :2]
            outputs_coord_wh = outputs_coord_delta[..., 2:].exp() * ref_unsigmoid[..., 2:]
            outputs_coord = torch.concat(
                [outputs_coord_cxcy, outputs_coord_wh], dim=-1
            )
        else:
            outputs_coord = (self.bbox_embed(hs) + ref_unsigmoid).sigmoid()
        outputs_class = self.class_embed(hs)
        return outputs_coord, outputs_class

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord, output_rots, output_trans):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{'pred_logits': a, 'pred_boxes': b, 'pred_rotations': c, 'pred_translations': d}
                for a, b, c, d in zip(outputs_class[:-1], outputs_coord[:-1], output_rots[:-1], output_trans[:-1])]

    def update_drop_path(self, drop_path_rate, vit_encoder_num_layers):
        """ """
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, vit_encoder_num_layers)]
        for i in range(vit_encoder_num_layers):
            if hasattr(self.backbone[0].encoder.blocks[i].drop_path, 'drop_prob'):
                self.backbone[0].encoder.blocks[i].drop_path.drop_prob = dp_rates[i]

    def update_dropout(self, drop_rate):
        for module in self.transformer.modules():
            if isinstance(module, nn.Dropout):
                module.p = drop_rate

    
    def process_rotation(self, pred_rotation):
        """
        Processes the predicted output rotation given the rotation mode.
        '6d' --> Gram Schmidt

        """
        if self.rotation_mode == '6d':
            return rotation_6d_to_matrix(pred_rotation)


class SetCriterion(nn.Module):
    """ This class computes the loss for Conditional DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """
    def __init__(self,
                 num_classes,
                 matcher,
                 weight_dict,
                 focal_alpha,
                 losses,
                 group_detr=1,
                 sum_group_losses=False,
                 use_varifocal_loss=False,
                 use_position_supervised_loss=False,
                 ia_bce_loss=False,
                 ):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            losses: list of all the losses to be applied. See get_loss for list of available losses.
            focal_alpha: alpha in Focal Loss
            group_detr: Number of groups to speed detr training. Default is 1.
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.losses = losses
        self.focal_alpha = focal_alpha
        self.group_detr = group_detr
        self.sum_group_losses = sum_group_losses
        self.use_varifocal_loss = use_varifocal_loss
        self.use_position_supervised_loss = use_position_supervised_loss
        self.ia_bce_loss = ia_bce_loss
        # Needed for Yolox6d losses
        self.mae_loss = nn.L1Loss(reduction="none")
        self.mse_loss = nn.MSELoss(reduction="none")
        self.shape_loss = False
        self.warm_up_epochs = 5

    ###################PoET no CAD needed#########################
    # Pose losses
    def loss_translation(self, outputs, targets, indices, num_boxes=None):
        """
        Compute the loss related to the translation of pose estimation, namely the mean square error (MSE)/ L2 Loss.
        outputs must contain the key 'pred_translation', while targets must contain the key 'relative_position'
        Position / Translation are expected in [x, y, z] meters 
        """
        idx = self._get_src_permutation_idx(indices)
        src_translation = outputs["pred_translations"][idx]
        tgt_translation = torch.cat([t['relative_position'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        n_obj = len(tgt_translation)

        loss_translation = F.mse_loss(src_translation, tgt_translation, reduction='none')
        loss_translation = torch.sum(loss_translation, dim=1)
        loss_translation = torch.sqrt(loss_translation)
        losses = {}
        losses["loss_translation"] = loss_translation.sum() / n_obj
        return losses

    # Not symmetry/geometric aware
    def loss_rotation(self,
                      outputs,
                      targets, 
                      indices, 
                      num_boxes=None):
        """
        Compute the loss related to the rotation of pose estimation represented by a 6d rotation matrix.
        The function calculates the geodesic distance between the predicted and target rotation.
        L = arccos( 0.5 * (Trace(R\tilde(R)^T) -1)
        Calculates the loss in radiant.
        """
        eps = 1e-6
        idx = self._get_src_permutation_idx(indices)
        src_rot = outputs["pred_rotations"][idx]
        # TODO: Check in dataloader which 6d conversion we need here.
        # Does this loss also work on 6d representations directly?
        tgt_rot = torch.cat([t['relative_rotation'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        n_obj = len(tgt_rot)

        product = torch.bmm(src_rot, tgt_rot.transpose(1, 2))
        trace = torch.sum(product[:, torch.eye(3).bool()], 1)
        theta = torch.clamp(0.5 * (trace - 1), -1 + eps, 1 - eps)
        rad = torch.acos(theta)
        losses = {}
        losses["loss_rot"] = rad.sum() / n_obj
        return losses
    ###################PoET Losses#########################

    ############### Losses proposed in yolox6d ######################
    #### My ADD-S and Rot loss. TODO: Check if it differs from the YOLOX6d implementation
    def loss_adds(self, 
                  outputs, 
                  targets, 
                  indices, 
                  num_boxes=None, 
                  shape_loss=False):
        """
        ADD / ADD-S loss (mean normalized point distance).
        shape_loss: if True ignore translation (rotation-only shape alignment).
        outputs:
            pred_rotations: (B,Q,3,3)
            pred_translations: (B,Q,3)
        targets per batch element:
            relative_rotations or rotations: (N,3,3)
            relative_positions or translations: (N,3)
            model_points: (N,P,3)
            diameter: (N,)
            is_symmetric: (N,) bool
        indices: list of (pred_idx, tgt_idx)
        """
        # Accept both naming conventions in targets
        device = outputs['pred_rotations'].device
        pred_R = outputs['pred_rotations']
        pred_t = outputs['pred_translations']

        total_point_loss = 0.0
        total_instances = 0

        for b, (pi, ti) in enumerate(indices):
            if len(pi) == 0:
                continue
            tgt = targets[b]
            R_p = pred_R[b][pi]  # (M,3,3)
            t_p = pred_t[b][pi]  # (M,3)
            R_g = tgt.get('relative_rotation')[ti]      # (M,3,3)
            t_g = tgt.get('relative_position')[ti]   # (M,3)
            pts  = tgt['model_points'][ti]        # (M,P,3)
            diam = tgt['diameter'][ti]            # (M,)
            sym  = tgt['is_symmetric'][ti].to(torch.bool)  # (M,)

            # Transform points (omit translation if shape_loss)
            if shape_loss:
                pts_pred = (R_p @ pts.transpose(1,2)).transpose(1,2)  # (M,P,3)
                pts_gt   = (R_g @ pts.transpose(1,2)).transpose(1,2)  # (M,P,3)
            else:
                pts_pred = (R_p @ pts.transpose(1,2) + t_p.unsqueeze(-1)).transpose(1,2)
                pts_gt   = (R_g @ pts.transpose(1,2) + t_g.unsqueeze(-1)).transpose(1,2)

            # Non-symmetric distances
            diff = pts_pred - pts_gt
            d_nonsym = diff.norm(dim=-1).mean(dim=-1)  # (M,)

            # Symmetric subset (ADD-S)
            if sym.any():
                sym_idx = torch.where(sym)[0]
                pp = pts_pred[sym_idx]
                gg = pts_gt[sym_idx]
                d_pair = torch.cdist(pp, gg, p=2)                 # (Ms,P,P)
                d_sym = d_pair.min(dim=-1).values.mean(dim=-1)    # (Ms,)
                d_all = d_nonsym.clone()
                d_all[sym_idx] = d_sym
            else:
                d_all = d_nonsym

            diam_safe = torch.clamp(diam, min=1e-6)
            inst_loss = (d_all / diam_safe).mean()
            total_point_loss += inst_loss
            total_instances += 1

        if total_instances == 0:
            loss_adds = torch.zeros((), device=device, requires_grad=True)
        else:
            loss_adds = total_point_loss / total_instances

        # Key remains loss_adds (rotation-only if shape_loss True)
        return {'loss_adds': loss_adds}
    ### Is wrong version of ADD-S. ADD-S uses closest point distance, not squared distance
    def loss_adds_mse(self, 
                outputs, 
                targets, 
                indices, 
                num_boxes=None, 
                shape_loss=False):
        """
        Compute ADD-S loss for 6D pose estimation.
        ADD / ADD-S MSE loss (fully squared: distances² / diameter²).
        Note: This differs from standard ADD/ADD-S metric which uses linear distances.

        Args:
            outputs: Dict containing:

                - 'pred_rotation': Predicted rotation matrices [batch, num_queries, 3, 3]
                - 'pred_translation': Predicted translations [batch, num_queries, 3]

            targets: List of dicts, each containing:

                - 'rotation': Ground-truth rotation [num_objects, 3, 3]
                - 'translation': Ground-truth translation [num_objects, 3]

                - 'model_points': 3D model points [num_objects, num_points, 3]
                - 'diameter': Object diameter [num_objects]

                - 'symmetric': Boolean mask [num_objects]

                - 'shape_loss': if True ignore translation (use rotation-only shape alignment).

            indices: List of (pred_idx, target_idx) tuples from matcher
            num_boxes: Number of boxes for normalization
        
        Returns:
            loss_dict: Dictionary with ADD-S loss
        """
        # Extract predictions
        device = outputs['pred_rotation'].device
        pred_rotations = outputs['pred_rotations']  # [batch, num_queries, 3, 3]
        pred_translations = outputs['pred_translations']  # [batch, num_queries, 3]
        # Flag precedence: explicit param overrides self.shape_loss
        use_shape_loss = self.shape_loss if shape_loss is None else shape_loss
        total_loss = 0.0
        num_valid = 0
        
        # Iterate over batch
        for batch_idx, (pred_idx, target_idx) in enumerate(indices):
            if len(pred_idx) == 0:
                continue
                
            # Get matched predictions
            batch_pred_rot = pred_rotations[batch_idx][pred_idx]  # [num_matched, 3, 3]
            batch_pred_trans = pred_translations[batch_idx][pred_idx]  # [num_matched, 3]
            
            # Get ground truth
            target_data = targets[batch_idx]
            batch_gt_rot = target_data['relative_rotation'][target_idx]  # [num_matched, 3, 3]
            batch_gt_trans = target_data['relative_position'][target_idx]  # [num_matched, 3]
            batch_model_points = target_data['model_points'][target_idx]  # [num_matched, num_points, 3]
            batch_diameter = target_data['diameter'][target_idx]  # [num_matched]
            batch_symmetric = target_data['is_symmetric'][target_idx]  # [num_matched]
            
            # Compute loss for each matched pair
            for i in range(len(pred_idx)):
                R_p = batch_pred_rot[i]  # [3, 3]
                t_p = batch_pred_trans[i]  # [3]
                R_g = batch_gt_rot[i]  # [3, 3]
                t_g = batch_gt_trans[i]  # [3]
                model_pts = batch_model_points[i]  # [num_points, 3]
                diameter = batch_diameter[i]
                is_symmetric = batch_symmetric[i]
                
                # Transform model points with predicted pose
                # pts_pred = R_p @ model_pts.T + t_p -> [3, num_points]
                pts_pred = torch.matmul(R_p, model_pts.T) + t_p.unsqueeze(1)
                pts_pred = pts_pred.T  # [num_points, 3]
                
                # Transform model points with ground-truth pose
                pts_gt = torch.matmul(R_g, model_pts.T) + t_g.unsqueeze(1)
                pts_gt = pts_gt.T  # [num_points, 3]
                
                if is_symmetric:
                    # ADD-S: For each predicted point, find closest ground-truth point
                    # Compute pairwise distances: [num_points, num_points]
                    diff = pts_pred.unsqueeze(1) - pts_gt.unsqueeze(0)  # [num_points, num_points, 3]
                    distances = torch.norm(diff, dim=2)  # [num_points, num_points]
                    
                    # Take minimum distance for each predicted point
                    min_distances, _ = torch.min(distances, dim=1)  # [num_points]
                    point_loss = min_distances
                else:
                    # ADD: Direct point-to-point distance
                    diff = pts_pred - pts_gt  # [num_points, 3]
                    point_loss = torch.norm(diff, dim=1)  # [num_points]
                
                # Normalize by diameter and average over points
                normalized_loss = (point_loss / diameter).pow(2).mean()
                total_loss += normalized_loss
                num_valid += 1
        
        # Average over all matched objects
        if num_valid > 0:
            loss = total_loss / num_valid
        else:
            loss = torch.tensor(0.0, device=pred_rotations.device, requires_grad=True)
        # TODO: Rename this to loss_adds
        return {'loss_rot': loss}
    
    
    def loss_rot(self, 
                 outputs, 
                 targets, 
                 indices,  
                 num_boxes):
        """Rotation loss from YOLO-6D Paper."""

        idx = self._get_src_permutation_idx(indices)
        src_rot = outputs["pred_rotations"][idx]          # (N,6) or (N,3,3)
        tgt_rot = torch.cat([t["relative_rotation"][i] for t, (_, i) in zip(targets, indices)], dim=0)  # (N,6) or (N,3,3)

        # Convert to rotation matrices
        if  src_rot.dim() == 3 and src_rot.shape[-2:] == (3,3):
            R_pred = rotation_matrix_to_raw_6d(src_rot)         # (N,6)
        elif src_rot.dim() == 2 and src_rot.size(-1) == 6:
            R_pred = src_rot
        else:
            raise ValueError(f"Unsupported pred_rotations shape {src_rot.shape}")

        if tgt_rot.dim() == 3 and tgt_rot.shape[-2:] == (3,3):
            R_tgt = rotation_matrix_to_raw_6d(tgt_rot)          # (N,6)
        elif tgt_rot.dim() == 2 and tgt_rot.size(-1) == 6:
            R_tgt = tgt_rot
        else:
            raise ValueError(f"Unsupported target_rot shape {tgt_rot.shape}")
        
        losses = {}
        loss_rot = self.mae_loss(R_pred.view(-1, 6), R_tgt.view(-1, 6)).sum() / num_boxes
        #loss_rot = F.l1_loss(R_pred.view(-1, 6), R_tgt.view(-1, 6), reduction='mean')
        losses["loss_rot"] = loss_rot / 6.0
        return losses
    #### End My ADD-S and Rot loss. TODO: Check if it differs from the YOLOX6d implementation
    
    def kpts_loss(self, 
                  kpts_preds, 
                  kpts_targets, 
                  bbox_targets):
        sigmas = torch.tensor([.26], device=kpts_preds.device) / 10.0
        kpts_preds_x, kpts_targets_x = kpts_preds[:, 0:1], kpts_targets[:, 0:1]
        kpts_preds_y, kpts_targets_y = kpts_preds[:, 1:2], kpts_targets[:, 1:2]
        # OKS based loss
        d = (kpts_preds_x - kpts_targets_x) ** 2 + (kpts_preds_y - kpts_targets_y) ** 2
        bbox_scale = torch.prod(bbox_targets[:, -2:], dim=1, keepdim=True)  #scale derived from bbox gt
        oks = torch.exp(-d / (bbox_scale * (4 * sigmas**2) + 1e-9))
        lkpt = (1 - oks).mean(axis=1)
        return lkpt

    def loss_oks_vibed(self, outputs, targets, indices, num_boxes):
        
        device = outputs['pred_translations'].device 
        
        sigmas = torch.tensor([.26], device=device) / 10.0
        k = 2*sigmas
        EPSILON = torch.finfo(torch.float32).eps
        
        total = 0.0
        count = 0

        # Camera intrinsics (assumed pixels)
        K = {
            "cx": 312.9869,
            "cy": 241.3109,
            "depth_scale": 0.1,
            "fx": 1066.778,
            "fy": 1067.487,
            "height": 480,
            "width": 640
        }
        fx = K['fx']; fy = K['fy']; cx = K['cx']; cy = K['cy']
        W = torch.as_tensor(K['width'], device=device, dtype=torch.float32)
        H = torch.as_tensor(K['height'], device=device, dtype=torch.float32)
        WH = torch.stack([W, H])  # (2,)

        for b, (pred_idx, tgt_idx) in enumerate(indices):
            if len(pred_idx) == 0:
                continue

            # (M,3) translation in camera coords 
            pred_t = outputs['pred_translations'][b][pred_idx]  
            object_center_gt_norm = targets[b]['object_center_2d'][tgt_idx]  # (M,2) normalized
            # Preds mm to m ?
            X = pred_t[:, 0] * 1000.0
            Y = pred_t[:, 1] * 1000.0
            Z = pred_t[:, 2] * 1000.0
            # Project to pixel space
            u = fx * (X / Z) + cx
            v = fy * (Y / Z) + cy
            object_center_pred_px = torch.stack([u, v], dim=1)  # (M,2) pixels

            # GT to pixel space 
            object_center_gt_px = object_center_gt_norm * WH  # (M,2)

            kpts_preds_x, kpts_targets_x = object_center_pred_px[:, 0:1], object_center_gt_px[:, 0:1]
            kpts_preds_y, kpts_targets_y = object_center_pred_px[:, 1:2], object_center_gt_px[:, 1:2]
            # OKS based loss
            #dist_sq = (kpts_preds_x - kpts_targets_x) ** 2 + (kpts_preds_y - kpts_targets_y) ** 2
            dist_sq = (kpts_targets_x - kpts_preds_x) ** 2 + (kpts_targets_y - kpts_preds_y) ** 2
            # Box area in pixels
            boxes = targets[b]['boxes'][tgt_idx]  # (M,4) cx,cy,w,h normalized
            w_pix = boxes[:, 2] * W
            h_pix = boxes[:, 3] * H
            area_pix = (w_pix * h_pix)  #area in pixels

            # OKS: exp(-d² / (2 * s_b² * k²))
            denominator = 2*(k**2)* (area_pix + EPSILON)        
            exponent = dist_sq / denominator
            oks = torch.exp(-exponent)

            loss_vec = 1.0 - oks
            total += loss_vec.mean()
            count += 1

        if count == 0:
            loss = torch.zeros((), device=device, requires_grad=True)
        else:
            loss = total / count

        return {'loss_trans_xy': loss}

    def loss_oks(self, outputs, targets, indices, num_boxes=None):
        
        device = outputs['pred_translations'].device 
        
        k = 0.1
        k_squared = k ** 2
        
        total = 0.0
        count = 0

        # Project 3D → 2D Pixel-coordinates using camera intrinsics
        K = {
            "cx": 312.9869,
            "cy": 241.3109,
            "depth_scale": 0.1,
            "fx": 1066.778,
            "fy": 1067.487,
            "height": 480,
            "width": 640
            }
        if K is None:
            raise ValueError("Camera intrinsics 'K' required for OKS loss")
        fx = K['fx']
        fy = K['fy']
        cx = K['cx']
        cy = K['cy']
        # Image size
        W = torch.as_tensor(K['width'], device=device, dtype=torch.float32)
        H = torch.as_tensor(K['height'], device=device, dtype=torch.float32)
        WH = torch.tensor([W, H], dtype=torch.float32).to(device=device)
        for b, (pred_idx, tgt_idx) in enumerate(indices):
            if len(pred_idx) == 0:
                continue

            pred_t = outputs['pred_translations'][b][pred_idx]  # (M,3) preds are in camera coords and mm
            object_center_gt = targets[b]['object_center_2d'][tgt_idx]  # (M,2) object centers normalized just as boxes

            X = pred_t[:, 0] * 1000.0
            Y = pred_t[:, 1] * 1000.0
            Z = pred_t[:, 2] * 1000.0
            u = fx * (X / Z) + cx
            v = fy * (Y / Z) + cy
            object_center_pred = torch.stack([u, v], dim=1)  # (M,2) in pixel space
            # Normalized pixel-space 
            object_center_pred = object_center_pred / WH

            kpts_preds_x, kpts_targets_x = object_center_pred[:, 0:1], object_center_gt[:, 0:1]
            kpts_preds_y, kpts_targets_y = object_center_pred[:, 1:2], object_center_gt[:, 1:2]
            # OKS based loss
            d2 = (kpts_preds_x - kpts_targets_x) ** 2 + (kpts_preds_y - kpts_targets_y) ** 2

            #d2 = (object_center_pred - gt_centers_px).pow(2).sum(dim=-1)  # (M,)

            # Box area in pixels
            boxes = targets[b]['boxes'][tgt_idx]               # (M,4) cx,cy,w,h normalized
            area = boxes[:, 2] * boxes[:, 3]                   # (M,)
                                   
            area = area.clamp(min=1e-3)                     # avoid too tiny denominators
            s_b2 = area ** 2              
            exponent = d2 / (2.0 * s_b2 * k_squared)  # small epsilon for numerical stability
            oks = torch.exp(-(exponent))
            loss_vec = 1.0 - oks
            total += loss_vec.mean()
            count += 1

            # Optional debug (comment out in production)
            # if b == 0:
            #     print(f"[OKS] d_mean={d2.sqrt().mean():.2f}px area_mean={area_pix.mean():.1f} exp_min={exponent.min():.1f} exp_max={exponent.max():.1f}")

        if count == 0:
            loss = torch.zeros((), device=device, requires_grad=True)
        else:
            loss = total / count

        return {'loss_trans_xy': loss}
    
    def loss_oks_2(self, outputs, targets, indices, num_boxes=None):
        
        device = outputs['pred_translations'].device 
        sigmas = torch.tensor([.26], device=device) / 10.0
        k = 2*sigmas
        EPSILON = torch.finfo(torch.float32).eps
        
        total = 0.0
        count = 0

        # Project 3D → 2D Pixel-coordinates using camera intrinsics
        K = {
            "cx": 312.9869,
            "cy": 241.3109,
            "depth_scale": 0.1,
            "fx": 1066.778,
            "fy": 1067.487,
            "height": 480,
            "width": 640
            }
        if K is None:
            raise ValueError("Camera intrinsics 'K' required for OKS loss")
        fx = K['fx']
        fy = K['fy']
        cx = K['cx']
        cy = K['cy']
        # Image size
        W = torch.as_tensor(K['width'], device=device, dtype=torch.float32)
        H = torch.as_tensor(K['height'], device=device, dtype=torch.float32)
        WH = torch.tensor([W, H], dtype=torch.float32).to(device=device)
        for b, (pred_idx, tgt_idx) in enumerate(indices):
            if len(pred_idx) == 0:
                continue

            pred_t = outputs['pred_translations'][b][pred_idx]  # (M,3) preds are in camera coords and mm
            object_center_gt = targets[b]['object_center_2d'][tgt_idx]  # (M,2) object centers normalized just as boxes

            X = pred_t[:, 0] * 1000.0
            Y = pred_t[:, 1] * 1000.0
            Z = pred_t[:, 2] * 1000.0
            u = fx * (X / Z) + cx
            v = fy * (Y / Z) + cy
            object_center_pred = torch.stack([u, v], dim=1)  # (M,2) in pixel space
            # Normalized pixel-space just as bboxes
            object_center_pred = object_center_pred / WH

            kpts_preds_x, kpts_targets_x = object_center_pred[:, 0:1], object_center_gt[:, 0:1]
            kpts_preds_y, kpts_targets_y = object_center_pred[:, 1:2], object_center_gt[:, 1:2]
            # OKS based loss
            dist_sq = (kpts_preds_x - kpts_targets_x) ** 2 + (kpts_preds_y - kpts_targets_y) ** 2

            # Box area in pixels
            boxes = targets[b]['boxes'][tgt_idx]               # (M,4) cx,cy,w,h normalized
            area = boxes[:, 2] * boxes[:, 3]                   # (M,)

            denominator = 2*(k**2)* (area + EPSILON)        
            exponent = dist_sq / denominator 
            
            oks = torch.exp(-(exponent))
            loss_vec = 1.0 - oks
            total += loss_vec.mean()
            count += 1

            # Optional debug (comment out in production)
            # if b == 0:
            #     print(f"[OKS] d_mean={d2.sqrt().mean():.2f}px area_mean={area_pix.mean():.1f} exp_min={exponent.min():.1f} exp_max={exponent.max():.1f}")

        if count == 0:
            loss = torch.zeros((), device=device, requires_grad=True)
        else:
            loss = total / count

        return {'loss_trans_xy': loss}
    def loss_oks_yolo6d(self, 
                    outputs, 
                    targets, 
                    indices,
                    num_boxes=None):
        """
        Object Keypoint Similarity (OKS) loss for tx, ty components.
        
        From YOLO-6D-Pose paper (Equation 21):
        "Since our proposed network predicts the projection of [tx, ty] on the image, 
        we add a loss term to make the 2D prediction accurate. This task is similar 
        to predicting a keypoint per object. Hence, we use a simplified version of 
        Object Keypoint Similarity (OKS) loss."
        
        L_OKS = 1 - OKS = 1 - exp(-d²/(2*s_b²*k²))
        
        where:

            - d: Euclidean distance between GT and predicted projection of 3D centroid
            - s_b: square root of bounding box area (object scale)

            - k: keypoint-specific constant (empirically set to 0.1)
        
        Args:
            outputs: dict containing:

                - 'pred_translation': (B, Q, 3) predicted translations
                - Optional: 'pred_boxes': (B, Q, 4) for box area, else computed from targets

            targets: list of B dicts, each containing:

                - 'relative_position': (N, 3) ground truth translations
                - 'boxes': (N, 4) bounding boxes in [x1, y1, x2, y2] format (optional)

                - Optional: camera intrinsics for projection

            indices: list of B tuples (pred_idx, tgt_idx) matching predictions to targets
            num_boxes: optional, not used but kept for interface compatibility
        
        Returns:
            dict with key 'loss_oks': scalar loss value
        """
        device = outputs['pred_translations'].device
        
        # Keypoint-specific weight constant
        k = 0.1
        k_squared = k ** 2
        
        total_loss = 0.0
        num_valid = 0
        
        for b, (pred_idx, tgt_idx) in enumerate(indices):
            if len(pred_idx) == 0:
                continue
            
            # Extract matched predictions and ground truth
            t_pred_xy = outputs['pred_translations'][b][pred_idx][:, :2]       # (M, 2)
            t_gt_xy = targets[b]['object_center_2d'][tgt_idx][:, :2]
            
            # Compute Euclidean distance d
            
            # d² per matched pair
            d2 = (t_pred_xy - t_gt_xy).pow(2).sum(dim=-1)
            # Boxes are in format cxcywh
            # Boxes (cx,cy,w,h) normalized -> convert to pixels if image size known
            boxes = targets[b]['boxes'][tgt_idx]
            
            # Box areas (normalized)
            boxes = targets[b]['boxes'][tgt_idx]  # (M,4) cx,cy,w,h normalized
            w = boxes[:, 2]
            h = boxes[:, 3]
            area = (w * h)

           
            s_b_squared  = torch.clamp(area, min=1e-6).pow(2)

            # Compute OKS: exp(-d²/(2*s_b²*k²))
            # Prepare exponent
            denominator = (2 * s_b_squared * k_squared)
            exponent = (-(d2) / denominator + 1e-9 ) # small epsilon for numerical stability

            oks = torch.exp(exponent)
            # Loss: L_OKS = 1 - OKS
            loss = 1.0 - oks                        # (M,)
            total_loss += loss.mean()
            num_valid += 1
        
        # Average over all batch elements with valid matches
        if num_valid == 0:
            loss = torch.zeros((), device=device, requires_grad=True)
        else:
            loss = total_loss / num_valid
        #print(f"d_mean={d2.sqrt().mean():.3f} area_mean={area.mean():.3f} exp_min={exponent.min():.2f} exp_max={exponent.max():.2f}")
        return {'loss_trans_xy': loss}
    # From: Review of monocular depth estimation methods
    # https://www.spiedigitallibrary.org/journals/journal-of-electronic-imaging/volume-34/issue-02/020901/Review-of-monocular-depth-estimation-methods/10.1117/1.JEI.34.2.020901.full#r81
    def loss_ard(self, 
                outputs, 
                targets, 
                indices,
                num_boxes):
        eps=1e-8
        idx = self._get_src_permutation_idx(indices)
        pred_trans_z = outputs['pred_translations'][idx][:,-1]  # (N, 1)
        tgt_trans_z = torch.cat([t['relative_position'][i] for t, (_, i) in zip(targets, indices)], dim=0)[:,-1] # (N, 1)

        ard = torch.abs(tgt_trans_z - pred_trans_z)
        ard = ard / (tgt_trans_z + eps)
        loss_ard = ard
        
        loss = loss_ard.sum() / num_boxes

        return {'loss_trans_z': loss}

    # Adapted adds_z/translation z loss from yolo6d code to DETR structure.
    def loss_adds_z(self, 
                   outputs, 
                   targets, 
                   indices, 
                   num_boxes=None):
        """
        Depth (tz) ADD-style normalized loss.
        Original adds_loss_z logic adapted to DETR:
            |pred_z - gt_z| * 1000 / diameter(object)
        Returns:
            {'loss_add_z': scalar}
        """
        idx = self._get_src_permutation_idx(indices)
        pred_trans_z = outputs['pred_translations'][idx][:,-1]  # (N, 1)
        tgt_trans_z = torch.cat([t['relative_position'][i] for t, (_, i) in zip(targets, indices)], dim=0)[:,-1] # (N, 1)
        tgt_diam = torch.cat([t['diameter'][i] for t, (_, i) in zip(targets, indices)], dim=0)  # (N,) 
        loss_adds_z = (self.mae_loss(pred_trans_z, tgt_trans_z)*1000.0 / tgt_diam).sum() / num_boxes
       
        return {'loss_trans_z': loss_adds_z}
    
    def loss_trans_z(self, 
                    outputs, 
                    targets, 
                    indices,  
                    num_boxes):
        
        idx = self._get_src_permutation_idx(indices)
        pred_trans_z = outputs['pred_translations'][idx][:,-1]  # (N, 1)
        tgt_trans_z = torch.cat([t['relative_position'][i] for t, (_, i) in zip(targets, indices)], dim=0)[:,-1] # (N, 1)
        loss_trans_z = self.mae_loss(pred_trans_z, tgt_trans_z).sum() / num_boxes
       
        return {'loss_trans_z': loss_trans_z}
    

    def loss_trans_z_l2(self, outputs, targets, indices, num_boxes=None):
        """
        Compute the loss related to the translation of pose estimation, namely the mean square error (MSE)/ L2 Loss.
        outputs must contain the key 'pred_translation', while targets must contain the key 'relative_position'
        Position / Translation are expected in [x, y, z] meters 
        """
        idx = self._get_src_permutation_idx(indices)
        pred_trans_z = outputs['pred_translations'][idx][:,-1]  # (N, 1)
        tgt_trans_z = torch.cat([t['relative_position'][i] for t, (_, i) in zip(targets, indices)], dim=0)[:,-1] # (N, 1)
        n_obj = len(tgt_trans_z)

        loss_trans_z = F.mse_loss(pred_trans_z, tgt_trans_z, reduction='none')
        loss_trans_z = torch.sum(loss_trans_z, dim=1)
        loss_trans_z = torch.sqrt(loss_trans_z + 1e-6)  # to avoid NaN
        
        loss_trans_z = loss_trans_z.sum() / n_obj
        return {'loss_trans_z': loss_trans_z}
    

    # Deeper Depth Prediction with Fully Convolutional Residual Networks - BerHu Loss
    # https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=7785097
    def loss_berhu_trans_z(self, 
                        outputs, 
                        targets, 
                        indices,  
                        num_boxes):
        
        idx = self._get_src_permutation_idx(indices)
        pred_trans_z = outputs['pred_translations'][idx][:,-1]  # (N, 1)
        tgt_trans_z = torch.cat([t['relative_position'][i] for t, (_, i) in zip(targets, indices)], dim=0)[:,-1] # (N, 1)
        
        loss_trans_z = self.mae_loss(pred_trans_z, tgt_trans_z).sum() / num_boxes
        c = torch.abs(pred_trans_z - tgt_trans_z).max() * 0.2
        if loss_trans_z <= c:
            return {'loss_trans_z': loss_trans_z}
        else:
            loss_trans_z = ((tgt_trans_z - pred_trans_z)**2 + c**2 / 2*c).sum() / num_boxes
            return {'loss_trans_z': loss_trans_z}

            
    def adds_loss(self, 
                  pose_preds, 
                  pose_gt, 
                  cls_targets, 
                  shape_loss=False):
        """
        Find out the actual ADD(S) score that can be used as a loss
        shape_loss: if set to True, don't use the translation component of the loss. This is called shape loss.
        """
        pose_preds[:, 2] *= 100.0
        pose_gt[:, 2] *= 100.0
        R_pred = torch.cat([pose_preds[:, 3:6, None], pose_preds[:, 6:9, None],
                            torch.cross(pose_preds[:, 3:6, None], pose_preds[:, 6:9, None], dim=1)], dim=-1)
        R_gt = torch.cat([pose_gt[:, 3:6, None], pose_gt[:, 6:9, None],
                          torch.cross(pose_gt[:, 3:6, None], pose_gt[:, 6:9, None], dim=1)], dim=-1)
        loss_adds = None
        for model_idx, sparse_model in self.cad_models.class_to_sparse_model.items():
            cls_idx = cls_targets==model_idx
            sparse_model = torch.tensor(sparse_model, device=cls_targets.device, dtype=pose_preds.dtype)
            if not shape_loss:
                pred_transformed_model = torch.matmul(R_pred[cls_idx], sparse_model.T) + pose_preds[cls_idx][:, :3, None]
                gt_transformed_model = torch.matmul(R_gt[cls_idx], sparse_model.T) + pose_gt[cls_idx][:, :3, None]
            else:
                pred_transformed_model = torch.matmul(R_pred[cls_idx], sparse_model.T)
                gt_transformed_model = torch.matmul(R_gt[cls_idx], sparse_model.T)

            #if torch.sum(cls_idx) != 0:
            if model_idx not in self.cad_models.symmetric_objects.keys():
                mse = ((pred_transformed_model - gt_transformed_model) ** 2).mean(axis=-1).sum(axis=-1)
            else:
                mse = torch.min(((pred_transformed_model[:, :, None, :] - gt_transformed_model[:, :, :, None]) ** 2).sum(axis=1), dim=1)[0]
                mse = mse.mean(axis=-1)
            adds_0p1 = torch.sqrt(mse) / (self.cad_models.models_diameter[model_idx])  #adds_0.1
            if loss_adds is None:
                loss_adds = adds_0p1
            else:
                loss_adds = torch.hstack((loss_adds, adds_0p1))
        loss_adds = loss_adds.mean()
        return loss_adds
    
    def loss_pose(self, 
              outputs, 
              targets, 
              indices,
              num_boxes):
        """
        Complete pose loss from YOLO-6D-Pose paper (Equation 23):
        
        L_pose = λ_ADD(S) * L_ADD(S) + λ_rot * L_rot + λ_OKS * L_OKS + λ_ARD * L_ARD
        
        Args:
            outputs: dict containing:

                - 'pred_rotation': (B, Q, 3, 3) predicted rotation matrices
                - 'pred_translation': (B, Q, 3) predicted translations

            targets: list of B dicts, each containing:

                - 'relative_rotation': (N, 3, 3) ground truth rotations
                - 'relative_position': (N, 3) ground truth translations

                - 'model_points': (N, P, 3) 3D model points
                - 'diameter': (N,) object diameters

                - 'is_symmetric': (N,) boolean flags for symmetric objects
                - 'boxes': (N, 4) bounding boxes [x1, y1, x2, y2]

            indices: list of B tuples (pred_idx, tgt_idx) for matching
            num_boxes: optional, for interface compatibility
        
        Returns:
            dict with individual and total pose losses
        """
        
        # Loss weights from paper (empirically tuned)
        lambda_adds = 1.0
        lambda_rot = 1.0
        lambda_oks = 1.0
        lambda_ard = 1.0
        
        # Compute individual losses
        loss_adds_dict = self.loss_adds(outputs, targets, indices)
        loss_rot_dict = self.loss_rot(outputs, targets, indices, num_boxes)
        loss_oks_dict = self.loss_oks_vibed(outputs, targets, indices, num_boxes)
        loss_ard_dict = self.loss_trans_z(outputs, targets, indices, num_boxes)
        #loss_trans_dict = self.loss_translation(outputs, targets, indices)
        #loss_oks_dict = None
        #loss_ard_dict = None
        
        # Extract loss values
        loss_adds = loss_adds_dict['loss_adds']
        loss_rot = loss_rot_dict['loss_rot']
        #loss_trans = loss_trans_dict['loss_translation']
        loss_oks = loss_oks_dict['loss_trans_xy']
        loss_ard = loss_ard_dict['loss_trans_z']
        
        # Total weighted pose loss
        loss_pose_total = (
            lambda_adds * loss_adds +
            lambda_rot * loss_rot +
            lambda_oks * loss_oks +
            lambda_ard * loss_ard
        )
        
        # loss_pose_total = (
        #     lambda_adds * loss_adds +
        #     lambda_rot * loss_rot * 
        #     1.0 * loss_trans
        # )

        # Return all components for logging
        return {
            'loss_pose': loss_pose_total,
            'loss_rot': loss_rot,
            'loss_trans_xy': loss_oks,
            'loss_trans_z': loss_ard,
            'loss_adds': loss_adds
        }
        # return {
        #         'loss_pose': loss_pose_total,
        #         'loss_rot': loss_rot,
        #         'loss_translation': loss_trans,
        #         'loss_adds': loss_adds
        #     }

    ######################################################

    #####T6D Direct: Symmetric Aware loss for rotation | translation uses the same as PoET ########################
    def symmetric_aware_rot_loss_optimized(self,
                                 R_pred, 
                                 R_gt, 
                                 points, 
                                 symmetric_flags=None, 
                                 reduction='mean',
                                chunk_size=100):
        """
        Symmetry-aware rotation loss (Memory-optimized version)
        
        Improvements:
        - Uses chunked computation for symmetric loss
        - Numerical stability with epsilon
        - Better documentation
        - Optional use of torch.cdist for efficiency
        """
        if points.dim() == 2:
            B = R_pred.shape[0]
            points = points.unsqueeze(0).expand(B, -1, -1)
        
        # Transform points: X = P @ R^T
        X_gt   = points @ R_gt.transpose(-1, -2)   # (B, N, 3)
        X_pred = points @ R_pred.transpose(-1, -2) # (B, N, 3)
        
        B, N, _ = X_gt.shape
        device = X_gt.device
        
        if symmetric_flags is None:
            symmetric_flags = torch.zeros(B, dtype=torch.bool, device=device)
        else:
            symmetric_flags = torch.as_tensor(symmetric_flags, dtype=torch.bool, device=device)
        
        # Non-symmetric: mean ||R_gt x - R_pred x|| + epsilon for stability
        nonsym_dist = torch.norm(X_gt - X_pred, dim=-1)
        nonsym_dist = (nonsym_dist + 1e-8).mean(dim=-1)  # (B,)
        
        # Symmetric: Chunked computation to save memory
        # mean_{x1} min_{x2} ||R_gt x1 - R_pred x2||
        if symmetric_flags.any():
            # Option 1: Using torch.cdist (faster if available)
            # sym_dists = torch.cdist(X_gt, X_pred, p=2)  # (B, N, N)
            # sym_min = sym_dists.min(dim=-1).values.mean(dim=-1)
            
            # Option 2: Chunked computation (memory-efficient)
            sym_min_list = []
            for i in range(0, N, chunk_size):
                end_i = min(i + chunk_size, N)
                chunk = X_gt[:, i:end_i, None, :]  # (B, chunk_size, 1, 3)
                diffs = chunk - X_pred[:, None, :, :]  # (B, chunk_size, N, 3)
                chunk_dists = torch.norm(diffs, dim=-1) + 1e-8  # (B, chunk_size, N)
                chunk_min = chunk_dists.min(dim=-1).values  # (B, chunk_size)
                sym_min_list.append(chunk_min)
            
            sym_min = torch.cat(sym_min_list, dim=1).mean(dim=-1)  # (B,)
        else:
            # If no symmetric objects, skip computation
            sym_min = torch.zeros_like(nonsym_dist)
        
        # Select per instance
        loss_per_inst = torch.where(symmetric_flags, sym_min, nonsym_dist)  # (B,)
        
        if reduction == 'mean':
            return loss_per_inst.mean()
        elif reduction == 'sum':
            return loss_per_inst.sum()
        elif reduction == 'none':
            return loss_per_inst
        else:
            raise ValueError(f"Invalid reduction: {reduction}")
    
    def symmetric_aware_rot_loss(self,
                                 R_pred, 
                                 R_gt, 
                                 points, 
                                 symmetric_flags=None, 
                                 reduction='mean'):
        """
        Symmetry-aware rotation loss LR (no translation)
        Inputs:

        - R_pred: (B, 3, 3) predicted rotation matrices
        - R_gt:   (B, 3, 3) ground-truth rotation matrices

        - points: (B, N, 3) model points per instance, or (N, 3) shared across batch
        - symmetric_flags: (B,) bool tensor or list indicating if each instance is symmetric.
                        If None, all treated as non-symmetric.
        - reduction: 'mean' | 'sum' | 'none'
        Returns:
        - loss: scalar if reduction != 'none', else (B,) per-instance losses

        """
        if points.dim() == 2:
            # Broadcast shared points to batch
            B = R_pred.shape[0]
            points = points.unsqueeze(0).expand(B, -1, -1)

        # Transform points with R*x (row-vector convention: X = P @ R^T)
        X_gt   = points @ R_gt.transpose(-1, -2)   # (B, N, 3)
        X_pred = points @ R_pred.transpose(-1, -2) # (B, N, 3)

        B, N, _ = X_gt.shape
        device = X_gt.device

        if symmetric_flags is None:
            symmetric_flags = torch.zeros(B, dtype=torch.bool, device=device)
        else:
            symmetric_flags = torch.as_tensor(symmetric_flags, dtype=torch.bool, device=device)

        # Non-symmetric: mean ||R_gt x - R_pred x||
        nonsym_dist = torch.norm(X_gt - X_pred, dim=-1).mean(dim=-1)  # (B,)

        # Symmetric: mean_{x1} min_{x2} ||R_gt x1 - R_pred x2||
        # Fully vectorized pairwise distances: (B, N, N)
        diffs = X_gt[:, :, None, :] - X_pred[:, None, :, :]
        sym_dists = torch.norm(diffs, dim=-1)           # (B, N, N)
        sym_min = sym_dists.min(dim=-1).values.mean(dim=-1)  # (B,)

        # Select per instance
        loss_per_inst = torch.where(symmetric_flags, sym_min, nonsym_dist)  # (B,)

        if reduction == 'mean':
            return loss_per_inst.mean()
        elif reduction == 'sum':
            return loss_per_inst.sum()
        elif reduction == 'none':
            return loss_per_inst
        else:
            raise ValueError(f"Invalid reduction: {reduction}")

    def symmetric_aware_rotation_loss_bidirectional(
                                                    R_pred, 
                                                    R_gt, 
                                                    points, 
                                                    symmetric_flags=None,     
                                                    reduction='mean',
                                                    bidirectional=True,
                                                    squared=True
                                                ):
        """
        Rotation loss aware of object symmetry.
        
        Args:
            R_pred: (B, 3, 3) predicted rotation matrices
            R_gt: (B, 3, 3) ground truth rotation matrices
            points: (N, 3) or (B, N, 3) reference points
            symmetric_flags: (B,) bool tensor indicating symmetric objects
            reduction: 'mean', 'sum', or 'none'
            bidirectional: If True, use Chamfer distance for symmetric objects
            squared: If True, use squared distances
        
        Returns:
            Loss scalar (if reduction != 'none') or (B,) tensor
        """
        # Ensure points is (B, N, 3)
        if points.dim() == 2:
            B = R_pred.shape[0]
            points = points.unsqueeze(0).expand(B, -1, -1)
        
        # Rotate points: (B, N, 3) @ (B, 3, 3) -> (B, N, 3)
        X_gt = points @ R_gt.transpose(-1, -2)
        X_pred = points @ R_pred.transpose(-1, -2)
        
        B, N, _ = X_gt.shape
        device = X_gt.device
        
        # Handle symmetric flags
        if symmetric_flags is None:
            symmetric_flags = torch.zeros(B, dtype=torch.bool, device=device)
        else:
            symmetric_flags = torch.as_tensor(symmetric_flags, dtype=torch.bool, device=device)
        
        # --- Non-symmetric loss (point-to-point) ---
        diff = X_gt - X_pred  # (B, N, 3)
        if squared:
            nonsym = (diff ** 2).sum(dim=-1).mean(dim=-1)  # (B,)
        else:
            nonsym = diff.norm(dim=-1).mean(dim=-1)  # (B,)
        
        # --- Symmetric loss (Chamfer distance) ---
        if symmetric_flags.any():
            # Compute pairwise squared distances more efficiently
            # ||a - b||^2 = ||a||^2 + ||b||^2 - 2 * a·b
            X_gt_sq = (X_gt ** 2).sum(dim=-1, keepdim=True)      # (B, N, 1)
            X_pred_sq = (X_pred ** 2).sum(dim=-1, keepdim=True)  # (B, N, 1)
            
            # (B, N, N)
            pair_d_sq = (
                X_gt_sq +                                    # (B, N, 1)
                X_pred_sq.transpose(-1, -2) -                # (B, 1, N)
                2 * torch.bmm(X_gt, X_pred.transpose(-1, -2)) # (B, N, N)
            )
            
            # Clamp to avoid numerical issues with sqrt
            pair_d_sq = pair_d_sq.clamp(min=0)
            
            if squared:
                pair_d = pair_d_sq
            else:
                pair_d = pair_d_sq.sqrt()
            
            # GT → Pred: min over predicted points
            gt_to_pred = pair_d.min(dim=-1).values.mean(dim=-1)  # (B,)
            
            if bidirectional:
                # Pred → GT: min over GT points
                pred_to_gt = pair_d.min(dim=-2).values.mean(dim=-1)  # (B,)
                sym_loss = 0.5 * (gt_to_pred + pred_to_gt)
            else:
                sym_loss = gt_to_pred
        else:
            # No symmetric objects, create dummy tensor
            sym_loss = torch.zeros(B, device=device)
        
        # --- Combine ---
        loss_per = torch.where(symmetric_flags, sym_loss, nonsym)
        
        if reduction == 'mean':
            return loss_per.mean()
        elif reduction == 'sum':
            return loss_per.sum()
        elif reduction == 'none':
            return loss_per
        else:
            raise ValueError(f"Invalid reduction: {reduction}")

    def symmetric_aware_rotation_loss_bidirectional_optimized(
        R_pred, 
        R_gt, 
        points, 
        symmetric_flags=None,     
        reduction='mean',
        bidirectional=True,
        squared=True
    ):
        """
        Rotation loss aware of object symmetry.
        
        Args:
            R_pred: (B, 3, 3) predicted rotation matrices
            R_gt: (B, 3, 3) ground truth rotation matrices
            points: (N, 3) or (B, N, 3) reference points
            symmetric_flags: (B,) bool tensor indicating symmetric objects
            reduction: 'mean', 'sum', or 'none'
            bidirectional: If True, use bidirectional Chamfer distance for symmetric objects
            squared: If True, use squared distances
        
        Returns:
            Loss scalar (if reduction != 'none') or (B,) tensor
        """
        # Ensure points is (B, N, 3)
        if points.dim() == 2:
            B = R_pred.shape[0]
            points = points.unsqueeze(0).expand(B, -1, -1)
        
        B, N, _ = points.shape
        device = points.device
        
        # Rotate points: (B, N, 3) @ (B, 3, 3) -> (B, N, 3)
        X_pred = points @ R_pred.transpose(-1, -2)
        X_gt = points @ R_gt.transpose(-1, -2)
        
        # Handle symmetric flags
        if symmetric_flags is None:
            symmetric_flags = torch.zeros(B, dtype=torch.bool, device=device)
        else:
            symmetric_flags = symmetric_flags.to(device=device, dtype=torch.bool)
        
        has_symmetric = symmetric_flags.any()
        has_nonsymmetric = (~symmetric_flags).any()
        
        # Initialize loss tensor
        loss_per = torch.zeros(B, device=device)
        
        # --- Non-symmetric loss (point-to-point) ---
        if has_nonsymmetric:
            nonsym_mask = ~symmetric_flags
            diff = X_gt[nonsym_mask] - X_pred[nonsym_mask]  # (B_ns, N, 3)
            
            if squared:
                nonsym_loss = (diff ** 2).sum(dim=-1).mean(dim=-1)  # (B_ns,)
            else:
                nonsym_loss = diff.norm(dim=-1).mean(dim=-1)  # (B_ns,)
            
            loss_per[nonsym_mask] = nonsym_loss
        
        # --- Symmetric loss (Chamfer distance) ---
        if has_symmetric:
            sym_mask = symmetric_flags
            X_pred_sym = X_pred[sym_mask]  # (B_s, N, 3)
            X_gt_sym = X_gt[sym_mask]      # (B_s, N, 3)
            
            # Use torch.cdist for efficiency (handles sqrt internally)
            if squared:
                pair_d = torch.cdist(X_gt_sym, X_pred_sym, p=2).pow(2)  # (B_s, N, N)
            else:
                pair_d = torch.cdist(X_gt_sym, X_pred_sym, p=2)  # (B_s, N, N)
            
            # GT → Pred: min over predicted points
            gt_to_pred = pair_d.min(dim=-1).values.mean(dim=-1)  # (B_s,)
            
            if bidirectional:
                # Pred → GT: min over GT points
                pred_to_gt = pair_d.min(dim=-2).values.mean(dim=-1)  # (B_s,)
                sym_loss = 0.5 * (gt_to_pred + pred_to_gt)
            else:
                sym_loss = gt_to_pred
            
            loss_per[sym_mask] = sym_loss
        
        # --- Reduction ---
        if reduction == 'mean':
            return loss_per.mean()
        elif reduction == 'sum':
            return loss_per.sum()
        elif reduction == 'none':
            return loss_per
        else:
            raise ValueError(f"Invalid reduction: {reduction}")

    def loss_rotation_symmetric_aware(self, 
                                      outputs, 
                                      targets, 
                                      indices,
                                      num_boxes=None):
        """
        Symmetry-aware rotation loss.
        Expects:
          outputs["pred_rotations"]: (B, Q, 6) or (B, Q, 3, 3)
          targets[b]["relative_rotation"]: (Nt, 3, 3) or (Nt, 6)
          Optional per-target:
              targets[b]["model_points"]: 
                  EITHER shared (N,3) tensor
                  OR per-object (Nt, N, 3) tensor
              targets[b]["is_symmetric"]: (Nt,) bool tensor/list
        """
        device = outputs["pred_rotations"].device
        # Gather matched indices
        matched_pred = []
        matched_gt = []
        matched_points = []
        matched_sym_flags = []

        for b, (pred_idx, tgt_idx) in enumerate(indices):
            if len(pred_idx) == 0:
                continue

            # Pred rotations
            pred_rots_b = outputs["pred_rotations"][b]  # (Q,6) or (Q,3,3)

            if pred_rots_b.dim() == 2 and pred_rots_b.size(-1) == 6:
                pred_rots_b = rotation_6d_simple_to_matrix(pred_rots_b)  # (Q,3,3)

            # GT rotations
            gt_rots_b = targets[b]["relative_rotation"]  # (Nt,3,3) or (Nt,6)
            if gt_rots_b.dim() == 2 and gt_rots_b.size(-1) == 6:
                gt_rots_b = rotation_6d_simple_to_matrix(gt_rots_b)  # (Nt,3,3)

            pred_sel = pred_rots_b[pred_idx]  # (M,3,3)
            gt_sel = gt_rots_b[tgt_idx]       # (M,3,3)

            # Symmetry flags per target object
            if "is_symmetric" in targets[b]:
                sym_flags_full = torch.as_tensor(targets[b]["is_symmetric"], dtype=torch.bool, device=device)
                sym_flags_sel = sym_flags_full[tgt_idx]  # (M,)
            else:
                sym_flags_sel = torch.zeros(len(tgt_idx), dtype=torch.bool, device=device)

            # Model points
            if "model_points" in targets[b]:
                mp = targets[b]["model_points"]
                # Shared (N,3)
                if mp.dim() == 2:
                    points_sel = mp.unsqueeze(0).expand(len(tgt_idx), -1, -1)  # (M,N,3)
                # Per-object (Nt,N,3)
                elif mp.dim() == 3:
                    points_sel = mp[tgt_idx]  # (M,N,3)
                else:
                    raise ValueError(f"model_points unexpected shape {mp.shape}")
            else:
                raise KeyError("Targets missing 'model_points' required for symmetry-aware rotation loss.")

            matched_pred.append(pred_sel)
            matched_gt.append(gt_sel)
            matched_points.append(points_sel)
            matched_sym_flags.append(sym_flags_sel)

        if len(matched_pred) == 0:
            # No matches
            losses = {"loss_rot": torch.zeros((), device=device)}
            return losses

        R_pred = torch.cat(matched_pred, dim=0)        # (TotalM,3,3)
        R_gt   = torch.cat(matched_gt, dim=0)          # (TotalM,3,3)
        points = torch.cat(matched_points, dim=0)      # (TotalM,N,3)
        sym_flags = torch.cat(matched_sym_flags, dim=0)  # (TotalM,)

        # Compute per-instance symmetry-aware loss (returns scalar with reduction='mean')
        loss_val = self.symmetric_aware_rot_loss_optimized(R_pred,
                                                  R_gt,
                                                  points,
                                                  symmetric_flags=sym_flags)
        # loss_val = self.symmetric_aware_rotation_loss_bidirectional(R_pred,
        #                                          R_gt, 
        #                                          points,
        #                                          symmetric_flags=sym_flags,
        #                                          reduction='mean')

        losses = {"loss_rot": loss_val}  # Already mean-reduced
        return losses
    ############# T6D symmetric aware loss end  ########################
     

    def loss_labels(self, outputs, targets, indices, num_boxes, log=True):
        """Classification loss (Binary focal loss)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])

        if self.ia_bce_loss:
            alpha = self.focal_alpha
            gamma = 2 
            src_boxes = outputs['pred_boxes'][idx]
            target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)

            iou_targets=torch.diag(box_ops.box_iou(
                box_ops.box_cxcywh_to_xyxy(src_boxes.detach()),
                box_ops.box_cxcywh_to_xyxy(target_boxes))[0])
            pos_ious = iou_targets.clone().detach()
            prob = src_logits.sigmoid()
            #init positive weights and negative weights
            pos_weights = torch.zeros_like(src_logits)
            neg_weights =  prob ** gamma

            pos_ind=[id for id in idx]
            pos_ind.append(target_classes_o)

            t = prob[pos_ind].pow(alpha) * pos_ious.pow(1 - alpha)
            t = torch.clamp(t, 0.01).detach()

            pos_weights[pos_ind] = t
            neg_weights[pos_ind] = 1 - t
            loss_ce = - pos_weights * prob.log() - neg_weights * (1 - prob).log()
            loss_ce = loss_ce.sum() / num_boxes

        elif self.use_position_supervised_loss:
            src_boxes = outputs['pred_boxes'][idx]
            target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)

            iou_targets=torch.diag(box_ops.box_iou(
                box_ops.box_cxcywh_to_xyxy(src_boxes.detach()),
                box_ops.box_cxcywh_to_xyxy(target_boxes))[0])
            pos_ious = iou_targets.clone().detach()
            # pos_ious_func = pos_ious ** 2
            pos_ious_func = pos_ious

            cls_iou_func_targets = torch.zeros((src_logits.shape[0], src_logits.shape[1],self.num_classes),
                                        dtype=src_logits.dtype, device=src_logits.device)

            pos_ind=[id for id in idx]
            pos_ind.append(target_classes_o)
            cls_iou_func_targets[pos_ind] = pos_ious_func
            norm_cls_iou_func_targets = cls_iou_func_targets \
                / (cls_iou_func_targets.view(cls_iou_func_targets.shape[0], -1, 1).amax(1, True) + 1e-8)
            loss_ce = position_supervised_loss(src_logits, norm_cls_iou_func_targets, num_boxes, alpha=self.focal_alpha, gamma=2) * src_logits.shape[1]

        elif self.use_varifocal_loss:
            src_boxes = outputs['pred_boxes'][idx]
            target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)

            iou_targets=torch.diag(box_ops.box_iou(
                box_ops.box_cxcywh_to_xyxy(src_boxes.detach()),
                box_ops.box_cxcywh_to_xyxy(target_boxes))[0])
            pos_ious = iou_targets.clone().detach()

            cls_iou_targets = torch.zeros((src_logits.shape[0], src_logits.shape[1],self.num_classes),
                                        dtype=src_logits.dtype, device=src_logits.device)

            pos_ind=[id for id in idx]
            pos_ind.append(target_classes_o)
            cls_iou_targets[pos_ind] = pos_ious
            loss_ce = sigmoid_varifocal_loss(src_logits, cls_iou_targets, num_boxes, alpha=self.focal_alpha, gamma=2) * src_logits.shape[1]
        else:
            target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                        dtype=torch.int64, device=src_logits.device)
            target_classes[idx] = target_classes_o

            target_classes_onehot = torch.zeros([src_logits.shape[0], src_logits.shape[1], src_logits.shape[2]+1],
                                                dtype=src_logits.dtype, layout=src_logits.layout, device=src_logits.device)
            target_classes_onehot.scatter_(2, target_classes.unsqueeze(-1), 1)

            target_classes_onehot = target_classes_onehot[:,:,:-1]
            loss_ce = sigmoid_focal_loss(src_logits, target_classes_onehot, num_boxes, alpha=self.focal_alpha, gamma=2) * src_logits.shape[1]
        losses = {'loss_ce': loss_ce}

        if log:
            # TODO this should probably be a separate loss, not hacked in this one here
            losses['class_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]
        return losses
    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, indices, num_boxes):
        """ Compute the cardinality error, ie the absolute error in the number of predicted non-empty boxes
        This is not really a loss, it is intended for logging purposes only. It doesn't propagate gradients
        """
        pred_logits = outputs['pred_logits']
        device = pred_logits.device
        tgt_lengths = torch.as_tensor([len(v["labels"]) for v in targets], device=device)
        # Count the number of predictions that are NOT "no-object" (which is the last class)
        card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        losses = {'cardinality_error': card_err}
        return losses

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')

        losses = {}
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes

        loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(
            box_ops.box_cxcywh_to_xyxy(src_boxes),
            box_ops.box_cxcywh_to_xyxy(target_boxes)))
        losses['loss_giou'] = loss_giou.sum() / num_boxes
        return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx



    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {
            'labels': self.loss_labels,
            'cardinality': self.loss_cardinality,
            'boxes': self.loss_boxes,
            #'translation': self.loss_translation,
            'pose': self.loss_pose,
            #'rotation': self.loss_rotation,
            #'rotation': self.loss_rot,
            #'adds': self.loss_adds,
            #'adds': self.loss_adds_mse, # From yolox6d
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

    def forward(self, outputs, targets):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        group_detr = self.group_detr if self.training else 1
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}

        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets, group_detr=group_detr)

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_boxes = sum(len(t["labels"]) for t in targets)
        if not self.sum_group_losses:
            num_boxes = num_boxes * group_detr
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                indices = self.matcher(aux_outputs, targets, group_detr=group_detr)
                for loss in self.losses:
                    kwargs = {}
                    if loss == 'labels':
                        # Logging is enabled only for the last layer
                        kwargs = {'log': False}
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_boxes, **kwargs)
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        if 'enc_outputs' in outputs:
            enc_outputs = outputs['enc_outputs']
            indices = self.matcher(enc_outputs, targets, group_detr=group_detr)
            for loss in self.losses:
                kwargs = {}
                if loss == 'labels':
                    # Logging is enabled only for the last layer
                    kwargs['log'] = False
                l_dict = self.get_loss(loss, 
                                        enc_outputs,
                                        targets, 
                                        indices,
                                        num_boxes, 
                                        **kwargs)
                l_dict = {k + f'_enc': v for k, v in l_dict.items()}
                losses.update(l_dict)

        return losses


def sigmoid_focal_loss(inputs, targets, num_boxes, alpha: float = 0.25, gamma: float = 2):
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
    Returns:
        Loss tensor
    """
    prob = inputs.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    return loss.mean(1).sum() / num_boxes


def sigmoid_varifocal_loss(inputs, targets, num_boxes, alpha: float = 0.25, gamma: float = 2):
    prob = inputs.sigmoid()
    focal_weight = targets * (targets > 0.0).float() + \
            (1 - alpha) * (prob - targets).abs().pow(gamma) * \
            (targets <= 0.0).float()
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    loss = ce_loss * focal_weight

    return loss.mean(1).sum() / num_boxes


def position_supervised_loss(inputs, targets, num_boxes, alpha: float = 0.25, gamma: float = 2):
    prob = inputs.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    loss = ce_loss * (torch.abs(targets - prob) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * (targets > 0.0).float() + (1 - alpha) * (targets <= 0.0).float()
        loss = alpha_t * loss

    return loss.mean(1).sum() / num_boxes


class PostProcess(nn.Module):
    """ This module converts the model's output into the format expected by the coco api"""
    def __init__(self, num_select=300) -> None:
        super().__init__()
        self.num_select = num_select

    @torch.no_grad()
    def forward(self, outputs, target_sizes):
        """ Perform the computation
        Parameters:
            outputs: raw outputs of the model
            target_sizes: tensor of dimension [batch_size x 2] containing the size of each images of the batch
                          For evaluation, this must be the original image size (before any data augmentation)
                          For visualization, this should be the image size after data augment, but before padding
        """
        out_logits, out_bbox = outputs['pred_logits'], outputs['pred_boxes']

        assert len(out_logits) == len(target_sizes)
        assert target_sizes.shape[1] == 2

        prob = out_logits.sigmoid()
        topk_values, topk_indexes = torch.topk(prob.view(out_logits.shape[0], -1), self.num_select, dim=1)
        scores = topk_values
        topk_boxes = topk_indexes // out_logits.shape[2]
        labels = topk_indexes % out_logits.shape[2]
        boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)
        boxes = torch.gather(boxes, 1, topk_boxes.unsqueeze(-1).repeat(1,1,4))

        # and from relative [0, 1] to absolute [0, height] coordinates
        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        boxes = boxes * scale_fct[:, None, :]

        results = [{'scores': s, 'labels': l, 'boxes': b} for s, l, b in zip(scores, labels, boxes)]

        return results


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


def build(args):
    # the `num_classes` naming here is somewhat misleading.
    # it indeed corresponds to `max_obj_id + 1`, where max_obj_id
    # is the maximum id for a class in your dataset. For example,
    # COCO has a max_obj_id of 90, so we pass `num_classes` to be 91.
    # As another example, for a dataset that has a single class with id 1,
    # you should pass `num_classes` to be 2 (max_obj_id + 1).
    # For more details on this, check the following discussion
    # https://github.com/facebookresearch/detr/issues/108#issuecomment-650269223
    num_classes = args.num_classes +1 if args.dataset_file != 'ycbv' else 22
    
    device = torch.device(args.device)

    backbone = build_backbone(args) # Object detector

    args.num_feature_levels = len(args.projector_scale)
    transformer = build_transformer(args) # Feature extractor

    model = LWDETR6D(
        backbone,
        transformer,
        num_classes=num_classes,
        num_queries=args.num_queries,
        aux_loss=args.aux_loss,
        group_detr=args.group_detr,
        two_stage=args.two_stage,
        lite_refpoint_refine=args.lite_refpoint_refine,
        bbox_reparam=args.bbox_reparam,
        rotation_mode="6d",
    )
    matcher = build_matcher(args)
    weight_dict = {'loss_ce': args.cls_loss_coef, 
                   'loss_bbox': args.bbox_loss_coef, 
                   'loss_trans': args.translation_loss_coef, 
                   'loss_rot': args.rotation_loss_coef}
    weight_dict['loss_giou'] = args.giou_loss_coef

    if args.rotation_representation == '6d':
        #losses = ['translation', 'rotation']
        losses = ['pose']
    else:
        raise NotImplementedError('Rotation representation not implemented')

    # TODO this is a hack
    if args.aux_loss:
        aux_weight_dict = {}
        for i in range(args.dec_layers - 1):
            aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})
        if args.two_stage:
            aux_weight_dict.update({k + f'_enc': v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)

    losses.append('labels')
    losses.append('boxes')
    losses.append('cardinality')
    try:
        sum_group_losses = args.sum_group_losses
    except:
        sum_group_losses = False
    criterion = SetCriterion(num_classes, 
                             matcher=matcher, 
                             weight_dict=weight_dict,
                             focal_alpha=args.focal_alpha, 
                             losses=losses, 
                             group_detr=args.group_detr, 
                             sum_group_losses=sum_group_losses,
                             use_varifocal_loss = args.use_varifocal_loss,
                             use_position_supervised_loss=args.use_position_supervised_loss,
                             ia_bce_loss=args.ia_bce_loss,
                             )
    criterion.to(device)
    postprocessors = {'bbox': PostProcess(num_select=args.num_select)}

    return model, criterion, postprocessors, matcher
