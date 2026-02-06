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
from numpy import indices
from matplotlib.pylab import indices
from pyparsing import Path
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
                 rotation_mode='6d',
                 ):
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
        self.z_dim = 2
        self.max_depth = 2.5 # TODO: derive from dataset / input
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
    
    def init_pose_heads(self):
        """
        Manually initializes the prediction heads to prevent training collapse.
        Call this AFTER loading pretrained backbone weights.
        """
        import torch.nn as nn

        # --- Helper Function ---
        # Detects if your head is a simple nn.Linear, nn.Sequential, or the custom MLP class
        def get_last_linear_layer(module):
            if hasattr(module, 'layers'):  # Standard DETR MLP class uses .layers
                return module.layers[-1]
            elif isinstance(module, nn.Sequential):
                return module[-1]
            elif isinstance(module, nn.Linear):
                return module
            else:
                raise ValueError(f"Unknown head structure: {type(module)}")

        print(">> Running Manual Pose Head Initialization...")

        # ---------------------------------------------------------
        # 1. Initialize Z-Head (Depth + Uncertainty)
        # ---------------------------------------------------------
        z_layer = get_last_linear_layer(self.dec_trans_z_head)
        
        # Channel 0: Normalized Depth (Sigmoid output)
        # Setting bias to 0.0 -> Sigmoid(0.0) = 0.5
        # 0.5 * Max_Depth (2.5m) = 1.25m (Safe mean guess for YCB-V)
        nn.init.constant_(z_layer.bias[0], 0.0)
        
        # Channel 1: Log Uncertainty 's' (Linear output)
        # Setting bias to 2.0 -> Variance = exp(2.0) = 7.38
        # This tells the model: "I am very uncertain, dampen the gradients."
        nn.init.constant_(z_layer.bias[1], 2.0)
        
        # ---------------------------------------------------------
        # 2. Initialize XY-Head (Visual Center)
        # ---------------------------------------------------------
        xy_layer = get_last_linear_layer(self.dec_trans_xy_head)
        
        # Setting bias to 0.0 -> Sigmoid(0.0) = 0.5
        # This places the initial prediction at the center of the image crop.
        nn.init.constant_(xy_layer.bias, 0.0)

        # ---------------------------------------------------------
        # 3. Initialize Rotation Head (6D Continuous)
        # ---------------------------------------------------------
        rot_layer = get_last_linear_layer(self.dec_rot_head)
        
        # Weights: Make them very small so output vectors are small numbers.
        # This prevents large initial rotations that confuse the matcher.
        nn.init.xavier_uniform_(rot_layer.weight, gain=0.01)
        
        # Bias: Zero (Neutral)
        nn.init.constant_(rot_layer.bias, 0.0)
        # prior_prob = 0.01
        # bias_value = -math.log((1 - prior_prob) / prior_prob)
        # nn.init.constant_(self.class_head.bias, bias_value)
        # OR simply: 
        nn.init.constant_(self.class_embed.bias, -4.6)
        
        print(f">> Class Head initialized with bias -4.6 (Prob 0.01)")
        print(">> Pose Heads Initialized: Z-Uncertainty set high, XY centered.")
        
    def forward(self, samples: NestedTensor, targets = None):
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
        # -----------------------------------------------------------
        # 1. Backbone + Transformer (Standard DETR)
        # -----------------------------------------------------------
        if isinstance(samples, (list, torch.Tensor)):
            samples = nested_tensor_from_tensor_list(samples)
        features, poss = self.backbone(samples)
        # Backup camera intrinsics (assumed pixels) for benchmark testing
        K = {
            "cx": 312.9869,
            "cy": 241.3109,
            "depth_scale": 0.1,
            "fx": 1066.778,
            "fy": 1067.487,
            "height": 480,
            "width": 640
        }
        srcs = []
        masks = []
        metas = []
        for l, feat in enumerate(features):
            src, mask, meta = feat.decompose()
            srcs.append(src)
            masks.append(mask)
            metas.append(meta) # Camera matries 
            assert mask is not None
        # -----------------------------------------------------------
        # 2. Get Raw Head Outputs
        # -----------------------------------------------------------
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
        # Prepare pred rotation for criterion
        pred_rots = self.dec_rot_head(hs)
        
        if self.training:
            rot_c1 = F.normalize(pred_rots[:, :, :, :3], dim=3) 
            rot_c2 = F.normalize(pred_rots[:, :, :, 3:] - torch.sum(rot_c1 * pred_rots[:, :, :, 3:], dim=3, keepdim=True) * rot_c1, dim=3) 
        else:
            # TODO: This can lead to NaNs if the norm is zero, need to be careful. 
            rot_c1 = (pred_rots[:, :, :, :3] / pred_rots[:, :, :, :3].norm(p=2, dim=3, keepdim=True))
            # Old version:
            # rot_c2 =  pred_rots[:, :, :, 3:] - torch.sum(rot_c1 * pred_rots[:, :, :, 3:], dim=2, keepdim=True) * rot_c1
            # Shouldnt this be dim=3 
            rot_c2 =  pred_rots[:, :, :, 3:] - torch.sum(rot_c1 * pred_rots[:, :, :, 3:], dim=3, keepdim=True) * rot_c1
            rot_c2 = rot_c2 / rot_c2.norm(p=2, dim=3, keepdim=True)


        # Gram-Schmidt representation
        output_rots = torch.cat([rot_c1, rot_c2], dim=3)
        

        output_trans = self.dec_trans_head(hs)
        # Prepare pred translation for criterion
        output_uv_norm = self.dec_trans_xy_head(output_trans).sigmoid()
        #trans_z
        # Prepare pred translation z for monocular depth estimation.
        z_out = self.dec_trans_z_head(output_trans)
        # Split the output:
        # Channel 0: Normalized Depth (Sigmoid -> 0 to 1)
        output_norm_z = z_out[..., 0].sigmoid()
        # Channel 1: Uncertainty (Log Variance) - No activation needed
        output_z_log_var = z_out[..., 1]
        pred_tz = output_norm_z * self.max_depth # Pred z in meters
        #The backprojection has to happen on to the padded img not on the orginal size
        valid_h = (~samples.mask[0]).any(dim=1).sum()
        valid_w = (~samples.mask[0]).any(dim=0).sum()
        img_h, img_w = int(valid_h), int(valid_w)
        
        u = output_uv_norm[..., 0:1] * img_w
        v = output_uv_norm[..., 1:2] * img_h
        
        # -----------------------------------------------------------
        # 3. The "Anti-Bias" Math Layer (Back-Projection)
        # -----------------------------------------------------------
        # Unpack Intrinsics (broadcast to match num_queries)
        # Empty in test run (benchmark testing)
        if len(samples.meta) != 0:
            intrinsics = samples.meta['K']
            fx = intrinsics[:, 0, 0].reshape(1, -1, 1)
            fy = intrinsics[:, 1, 1].reshape(1, -1, 1)
            cx = intrinsics[:, 0, 2].reshape(1, -1, 1)
            cy = intrinsics[:, 1, 2].reshape(1, -1, 1)
            pred_tx = (u.squeeze(-1) - cx) * pred_tz / fx
            pred_ty = (v.squeeze(-1) - cy) * pred_tz / fy
            output_trans = torch.cat([pred_tx.unsqueeze(-1), pred_ty.unsqueeze(-1), pred_tz.unsqueeze(-1)], dim=-1)
        else: # Backup matrix for benchmark testing
            fx = K['fx']
            fy = K['fy']
            cx = K['cx']
            cy = K['cy']
            pred_tz = pred_tz.unsqueeze(-1)  # Add batch dim for broadcasting
            # Apply Pinhole Formula to get x,y in meters
            pred_tx = (u - cx) * pred_tz / fx
            pred_ty = (v - cy) * pred_tz / fy
            output_trans = torch.cat([pred_tx, pred_ty, pred_tz], dim=-1)

        # Postprocess output_rots for loss
        output_rots = self.process_rotation(output_rots)
        out = {'pred_logits': outputs_class[-1], 
               'pred_boxes': outputs_coord[-1],
               'pred_rotations': output_rots[-1],
               'pred_translations': output_trans[-1],   
               'pred_uv_norm': output_uv_norm[-1],
               'pred_trans_z': output_norm_z[-1],            # Normalized z input for Laplacian Loss 
               'pred_z_log_var': output_z_log_var[-1],      # Input for Laplacian Loss
               }
        
        if self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(outputs_class, 
                                                    outputs_coord, 
                                                    output_rots, 
                                                    output_trans,
                                                    output_uv_norm,
                                                    output_z_log_var,
                                                    output_norm_z
                                                    )

        if self.two_stage:
            hs_enc_list = hs_enc.split(self.num_queries, dim=1)
            cls_enc, rot_enc_list, trans_enc_list, kpt_enc_list, log_var_z_list, trans_enc_z_list = [], [], [], [], [], []
            group_detr = self.group_detr if self.training else 1
            for g_idx in range(group_detr):
                enc_feat = hs_enc_list[g_idx]
                cls_enc_gidx = self.transformer.enc_out_class_embed[g_idx](enc_feat)
                cls_enc.append(cls_enc_gidx)
                rot_enc_list.append(self.transformer.enc_out_rot_embed[g_idx](enc_feat))      # (B, num_queries, rot_dim)
                
    
                trans_enc = self.transformer.enc_out_trans_embed[g_idx](enc_feat)  # (B, num_queries, 3)
                uv_norm_enc = self.transformer.enc_out_trans_xy_embed[g_idx](trans_enc).sigmoid()  # (B, num_queries, 2)
                u = uv_norm_enc[..., 0:1] * img_w
                v = uv_norm_enc[..., 1:2] * img_h
                # Prepare pred translation z for monocular depth estimation.
                z_out_enc = (self.transformer.enc_out_trans_z_embed[g_idx](trans_enc))
                # Split the output:
                # Channel 0: Normalized Depth (Sigmoid -> 0 to 1)
                norm_z_enc = z_out_enc[..., 0].sigmoid()
                # Channel 1: Uncertainty (Log Variance) - No activation needed
                z_log_var_enc = z_out_enc[..., 1]
                trans_enc_z = norm_z_enc * self.max_depth # Pred z in meters
                
                if len(samples.meta) != 0:
                    trans_enc_x = (u.squeeze(-1) - cx.squeeze(0)) * trans_enc_z / fx.squeeze(0)
                    trans_enc_y = (v.squeeze(-1) - cy.squeeze(0)) * trans_enc_z / fy.squeeze(0)
                    trans_enc = torch.stack([trans_enc_x, trans_enc_y, trans_enc_z], dim=-1)
                    #trans_enc = torch.cat([trans_enc_x, trans_enc_y, trans_enc_z.unsqueeze(0)]).permute(1,2,0)
                else: # Backup matrix for benchmark testing
                    trans_enc_x = (u - cx) * trans_enc_z.unsqueeze(-1)  / fx
                    trans_enc_y = (v - cy) * trans_enc_z.unsqueeze(-1)  / fy
                    trans_enc = torch.cat([trans_enc_x, trans_enc_y, trans_enc_z.unsqueeze(-1)], dim=-1)

                kpt_enc_list.append(uv_norm_enc)
                trans_enc_list.append(trans_enc)
                log_var_z_list.append(z_log_var_enc)
                trans_enc_z_list.append(trans_enc_z)
            cls_enc = torch.cat(cls_enc, dim=1)
            rot_enc = torch.cat(rot_enc_list, dim=1)            # (B, total_queries, rot_dim)
            
            trans_enc = torch.cat(trans_enc_list, dim=1)      # (B, total_queries, 3)
            z_log_var_enc = torch.cat(log_var_z_list, dim=1)  # (B, total_queries, 1)
            trans_enc_z = torch.cat(trans_enc_z_list, dim=1)  # (B, total_queries, 1)
            kpt_enc = torch.cat(kpt_enc_list, dim=1)          # (B, total_queries, 2)
            
            rot_c1 = F.normalize(rot_enc[..., :3], dim=-1) 
            rot_c2 = F.normalize(rot_enc[..., 3:] - torch.sum(rot_c1 * rot_enc[..., 3:], dim=-1, keepdim=True) * rot_c1, dim=-1) 
            rot_enc_full = torch.cat([rot_c1, rot_c2], dim=-1)          # (B, total_queries, 6)
            rot_enc_full = self.process_rotation(rot_enc_full) # (B, total_queries, 3, 3)
            out['enc_outputs'] = {
                'pred_logits': cls_enc,
                'pred_boxes': ref_enc,
                'pred_rotations': rot_enc_full,
                'pred_translations': trans_enc,
                'pred_uv_norm': kpt_enc,
                'pred_z_log_var': z_log_var_enc,
                'pred_trans_z': trans_enc_z
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
    def _set_aux_loss(self, outputs_class, outputs_coord, output_rots, output_trans, output_uv_norm, output_z_log_var, output_trans_z):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{'pred_logits': a, 
                 'pred_boxes': b, 
                 'pred_rotations': c, 
                 'pred_translations': d, 
                 'pred_uv_norm': e, 
                 'pred_z_log_var': f, 
                 'pred_trans_z': g}

                for a, b, c, d, e, f, g in zip(outputs_class[:-1], 
                                               outputs_coord[:-1], 
                                               output_rots[:-1], 
                                               output_trans[:-1], 
                                               output_uv_norm[:-1], 
                                               output_z_log_var[:-1], 
                                               output_trans_z[:-1])]

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
        
        self.mae_loss = nn.L1Loss(reduction="none")
        self.mse_loss = nn.MSELoss(reduction="none")
        self.shape_loss = False
        

    ###################PoET no CAD needed#########################
    # Pose losses for translation and rotation expected in meters and radians
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
    # geodesic distance loss for rotation matrix but symmetry aware
    def loss_rotation_symmetry_aware(self, outputs, targets, indices, num_boxes):
        """
        Geodesic Loss (Safe Trace + Symmetry Handling)
        """
        eps = 1e-6
        idx = self._get_src_permutation_idx(indices)
        
        # 1. Get Predictions & Targets
        src_rot = outputs["pred_rotations"][idx] # [N, 3, 3]
        tgt_rot = torch.cat([t['relative_rotation'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        
        # Get Symmetry Labels (Optional but Recommended)
        # If your dataset doesn't have 'is_symmetric', assume all False
        if 'is_symmetric' in targets[0]:
            is_sym = torch.cat([t['is_symmetric'][i] for t, (_, i) in zip(targets, indices)], dim=0).bool()
        else:
            is_sym = torch.zeros(src_rot.shape[0], device=src_rot.device, dtype=torch.bool)

        # 2. Compute Product R_pred * R_gt^T
        # Ideally, this should be Identity if perfect match
        product = torch.bmm(src_rot, tgt_rot.transpose(1, 2)) # [N, 3, 3]

        # 3. Safe Trace Calculation
        # sum(diagonal) -> sum over dim 1 of the diagonal elements
        trace = product.diagonal(dim1=1, dim2=2).sum(-1)

        # 4. Compute Geodesic Distance (Radians)
        # formula: acos( (Tr - 1) / 2 )
        # Clamp is vital: Trace can slightly exceed 3.0 or -1.0 due to float errors
        cosine_val = 0.5 * (trace - 1)
        cosine_val = torch.clamp(cosine_val, -1 + eps, 1 - eps)
        rad = torch.acos(cosine_val)

        # 5. Symmetry Handling (Crucial for YCB)
        # For symmetric objects, R_pred vs R_gt is ambiguous.
        # Force loss to 0.0 for them, so we rely PURELY on ADD-S for those objects.
        # Otherwise, this loss fights the ADD-S loss.
        loss = torch.where(is_sym, torch.zeros_like(rad), rad)

        # 6. Normalize
        losses = {}
        # Use sum() / num_boxes as per DETR standard
        losses["loss_rot"] = loss.sum() / num_boxes
        
        return losses
    # geodesic distance loss for rotation matrix
    def loss_rotation(self,
                      outputs,
                      targets, 
                      indices, 
                      num_boxes):
        """
        Compute the loss related to the rotation of pose estimation represented by a 6d rotation matrix.
        The function calculates the geodesic distance between the predicted and target rotation.
        L = arccos( 0.5 * (Trace(R\tilde(R)^T) -1)
        Calculates the loss in radiant.
        """
        eps = 1e-6
        idx = self._get_src_permutation_idx(indices)
        src_rot = outputs["pred_rotations"][idx]
        tgt_rot = torch.cat([t['relative_rotation'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        # Get symmetry flags
        is_sym = torch.cat([t['is_symmetric'][i] for t, (_, i) in zip(targets, indices)], dim=0).bool()

        product = torch.bmm(src_rot, tgt_rot.transpose(1, 2))
        trace = torch.sum(product[:, torch.eye(3).bool()], 1)
        theta = torch.clamp(0.5 * (trace - 1), -1 + eps, 1 - eps)
        rad = torch.acos(theta)
        ####### Test #######
        # Sym aware
        # Zero out rotation loss for symmetric objects
        # Rely purely on ADD-S for those
        loss = torch.where(is_sym, torch.zeros_like(rad), rad)


        losses = {}
        losses["loss_rot"] = loss.sum() / num_boxes
        return losses
    ###################PoET Losses#########################
    ############### Losses proposed in yolox6d ######################
    #### My ADD-S and Rot loss.
    def loss_adds_sym_only(self, outputs, targets, indices, num_boxes):
        """
        ADD / ADD-S loss (Raw Meter Distance).
        """
        pred_R = outputs['pred_rotations']     # (B, Q, 3, 3)
        pred_t = outputs['pred_translations']  # (B, Q, 3)

        # We will accumulate the SUM of distances and divide by num_boxes at the end
        total_dist_sum = 0.0

        for b, (pi, ti) in enumerate(indices):
            if len(pi) == 0: continue

            # 1. Fetch Data
            # Note: We don't need 'diameter' anymore if we want raw meter loss
            tgt = targets[b]
            R_p, t_p = pred_R[b][pi], pred_t[b][pi]
            
            # Ensure targets are float32 to match preds
            R_g = tgt['relative_rotation'][ti].float() # (M, 3, 3)
            t_g = tgt['relative_position'][ti].float() # (M, 3)
            pts = tgt['model_points'][ti].float() # (M, P, 3)
            sym = tgt['is_symmetric'][ti].bool()  # (M,)

            # 2. Transform Points (Batch Matrix Multiplication)
            # Formula: (R @ points.T).T + t
            # pred: (M, 3, 3) @ (M, 3, P) -> (M, 3, P) -> (M, P, 3)
            pts_p = torch.bmm(R_p, pts.transpose(1, 2)).transpose(1, 2) + t_p.unsqueeze(1)
            pts_g = torch.bmm(R_g, pts.transpose(1, 2)).transpose(1, 2) + t_g.unsqueeze(1)

            if sym.any():
                # 3. Calculate Distances
                # Default: ADD (Non-Symmetric) - 1-to-1 distance
                # Shape: (M, P, 3) -> (M, P) -> (M,)
                diff = pts_p - pts_g
                dists = diff.norm(dim=-1).mean(dim=-1)

                # 4. Handle Symmetric Objects (ADD-S)
                sym_idx = torch.where(sym)[0]
                
                # Extract symmetric subset
                p_sym = pts_p[sym_idx] # (S, P, 3)
                g_sym = pts_g[sym_idx] # (S, P, 3)

                # Compute pairwise distance matrix (S, P, P)
                # This finds the closest point on GT for every point on Pred
                pairwise_dist = torch.cdist(p_sym, g_sym, p=2)
                
                # Min over GT points (dim 2), then Mean over Pred points (dim 1)
                min_dists = pairwise_dist.min(dim=2).values.mean(dim=1)
                
                # Overwrite the standard ADD distances with ADD-S distances
                dists[sym_idx] = min_dists

            # 5. Sum up the error for this batch
            total_dist_sum += dists.sum()

        # 6. Normalize by total number of matched objects in the batch (DETR standard)
        # Avoid division by zero
        num_boxes = max(num_boxes, 1)
        loss_adds = total_dist_sum / num_boxes

        return {'loss_adds': loss_adds}

    def loss_adds(self, outputs, targets, indices, num_boxes):
        """
        ADD / ADD-S loss (Raw Meter Distance).
        """
        pred_R = outputs['pred_rotations']     # (B, Q, 3, 3)
        pred_t = outputs['pred_translations']  # (B, Q, 3)

        # We will accumulate the SUM of distances and divide by num_boxes at the end
        total_dist_sum = 0.0

        for b, (pi, ti) in enumerate(indices):
            if len(pi) == 0: continue

            # 1. Fetch Data
            # Note: We don't need 'diameter' anymore if we want raw meter loss
            tgt = targets[b]
            R_p, t_p = pred_R[b][pi], pred_t[b][pi]
            
            # Ensure targets are float32 to match preds
            R_g = tgt['relative_rotation'][ti].float() # (M, 3, 3)
            t_g = tgt['relative_position'][ti].float() # (M, 3)
            pts = tgt['model_points'][ti].float() # (M, P, 3)
            sym = tgt['is_symmetric'][ti].bool()  # (M,)

            # 2. Transform Points (Batch Matrix Multiplication)
            # Formula: (R @ points.T).T + t
            # pred: (M, 3, 3) @ (M, 3, P) -> (M, 3, P) -> (M, P, 3)
            pts_p = torch.bmm(R_p, pts.transpose(1, 2)).transpose(1, 2) + t_p.unsqueeze(1)
            pts_g = torch.bmm(R_g, pts.transpose(1, 2)).transpose(1, 2) + t_g.unsqueeze(1)

            # 3. Calculate Distances
            # Default: ADD (Non-Symmetric) - 1-to-1 distance
            # Shape: (M, P, 3) -> (M, P) -> (M,)
            diff = pts_p - pts_g
            dists = diff.norm(dim=-1).mean(dim=-1)

            # 4. Handle Symmetric Objects (ADD-S)
            if sym.any():
                sym_idx = torch.where(sym)[0]
                
                # Extract symmetric subset
                p_sym = pts_p[sym_idx] # (S, P, 3)
                g_sym = pts_g[sym_idx] # (S, P, 3)

                # Compute pairwise distance matrix (S, P, P)
                # This finds the closest point on GT for every point on Pred
                pairwise_dist = torch.cdist(p_sym, g_sym, p=2)
                
                # Min over GT points (dim 2), then Mean over Pred points (dim 1)
                min_dists = pairwise_dist.min(dim=2).values.mean(dim=1)
                
                # Overwrite the standard ADD distances with ADD-S distances
                dists[sym_idx] = min_dists

            # 5. Sum up the error for this batch
            total_dist_sum += dists.sum()

        # 6. Normalize by total number of matched objects in the batch (DETR standard)
        # Avoid division by zero
        num_boxes = max(num_boxes, 1)
        loss_adds = total_dist_sum / num_boxes

        return {'loss_adds': loss_adds}
    # Rotation loss from YOLO-6D Paper (6D representation with L1 loss) 
    def loss_rot(self, 
                 outputs, 
                 targets, 
                 indices,  
                 num_boxes):
        """."""

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
    #### End My ADD-S and Rot loss.

    ## YOLOX6D Keypoint loss (2D projection of 3D center)
    # TODO: Check if this is implemented correctly
    def loss_oks(self, outputs, targets, indices, num_boxes):
        
        device = outputs['pred_translations'].device 
        sigmas = torch.tensor([.26], device=device) / 10.0
        k = 2*sigmas
        EPSILON = torch.finfo(torch.float32).eps
        
        total = 0.0
        
        idx = self._get_src_permutation_idx(indices)
        # Get the predicted Normalized UV for matched queries
        src_norm_uv = outputs['pred_uv_norm'][idx]
        tgt_norm_uv = torch.cat([t['object_center_2d'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        
        boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)              # (M,4) cx,cy,w,h normalized
        # Box area in normalized pixel space
        area = boxes[:, 2] * boxes[:, 3]                   # (M,)

        kpts_preds_x, kpts_targets_x = src_norm_uv[:, 0:1], tgt_norm_uv[:, 0:1]
        kpts_preds_y, kpts_targets_y = src_norm_uv[:, 1:2], tgt_norm_uv[:, 1:2]
        # OKS based loss
        dist_sq = (kpts_preds_x - kpts_targets_x) ** 2 + (kpts_preds_y - kpts_targets_y) ** 2

        denominator = 2*(k**2)* (area + EPSILON)        
        exponent = dist_sq / denominator

        oks = torch.exp(-exponent)
        loss_vec = 1.0 - oks
        total = loss_vec.sum()
        loss = total / num_boxes

        return {'loss_keypoint': loss}
    
    # From: Review of monocular depth estimation methods
    # https://www.spiedigitallibrary.org/journals/journal-of-electronic-imaging/volume-34/issue-02/020901/Review-of-monocular-depth-estimation-methods/10.1117/1.JEI.34.2.020901.full#r81
    def loss_ard_trans_z(self, 
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
        loss_adds_z = (self.mae_loss(pred_trans_z, tgt_trans_z) * 1000.0 / tgt_diam).sum() / num_boxes
       
        return {'loss_trans_z': loss_adds_z}
    # Same as above but not normalized by diameter
    def loss_trans_z_yolo6d(self, 
                    outputs, 
                    targets, 
                    indices,  
                    num_boxes):
        
        idx = self._get_src_permutation_idx(indices)
        pred_trans_z = outputs['pred_translations'][idx][:,-1]  # (N, 1)
        tgt_trans_z = torch.cat([t['relative_position'][i] for t, (_, i) in zip(targets, indices)], dim=0)[:,-1] # (N, 1)
        loss_trans_z = self.mae_loss(pred_trans_z, tgt_trans_z).sum() / num_boxes
       
        return {'loss_trans_z': loss_trans_z}
    
    def loss_trans_z_l2(self, outputs, targets, indices, num_boxes):
        """
        Compute the loss related to the translation of pose estimation, namely the mean square error (MSE)/ L2 Loss.
        outputs must contain the key 'pred_translation', while targets must contain the key 'relative_position'
        Position / Translation are expected in [x, y, z] meters 
        """
        idx = self._get_src_permutation_idx(indices)
        pred_trans_z = outputs['pred_translations'][idx][:,-1]  # (N, 1)
        tgt_trans_z = torch.cat([t['relative_position'][i] for t, (_, i) in zip(targets, indices)], dim=0)[:,-1] # (N, 1)
       
        loss_trans_z = F.mse_loss(pred_trans_z, tgt_trans_z, reduction='none') * 1000.0  # (N, 1)
        loss_trans_z = torch.sum(loss_trans_z)
        loss_trans_z = torch.sqrt(loss_trans_z + 1e-6)  # to avoid NaN
        
        loss_trans_z = loss_trans_z / num_boxes
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
    
    # Translation in xy in uv coords.
    # L1 Loss for 2D Keypoint (Object Center Projection)
    def loss_keypoint(self, 
                    outputs, 
                    targets, 
                    indices,
                    num_boxes):
        idx = self._get_src_permutation_idx(indices)
        # Get the predicted Normalized UV for matched queries
        src_norm_uv = outputs['pred_uv_norm'][idx]
        tgt_norm_uv = torch.cat([t['object_center_2d'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        loss_kpt = F.smooth_l1_loss(src_norm_uv, tgt_norm_uv, reduction='sum') / num_boxes
        return {'loss_keypoint': loss_kpt}

    # L1 Loss for translation in x,y in meters
    def loss_trans_xy(self, 
                      outputs, 
                      targets, 
                      indices,  
                      num_boxes):
        idx = self._get_src_permutation_idx(indices)
        src_trans = outputs['pred_translations'][idx]
        tgt_trans = torch.cat([t['relative_position'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        # Only consider x,y components (L1 loss)
        loss_trans = self.mae_loss(src_trans[:, :2], tgt_trans[:, :2]).sum() / num_boxes
        return {'loss_trans_xy': loss_trans}
    
    # The following trans_z losses expect z in normalized 0-1 space
    # Log-L1 Loss (Ablation Experiment)
    def loss_relative_log_l1(self, pred_z_meters, gt_z_meters):
        """
        Computes L1 loss in Log Space. 
        Returns the SUM of errors (not mean).
        """
        eps = 1e-6
        pred_z = torch.clamp(pred_z_meters, min=eps)
        gt_z = torch.clamp(gt_z_meters, min=eps)
        
        # Compute element-wise loss
        loss = torch.abs(torch.log(pred_z) - torch.log(gt_z))
        
        # Return Sum. Let the caller divide by global num_boxes.
        return loss.sum()

    # Laplacian Aleatoric Loss for translation in z
    def loss_trans_z(self, 
                    outputs, 
                    targets, 
                    indices,  
                    num_boxes):
        
        idx = self._get_src_permutation_idx(indices)
        
        # -------------------------------------------------------------
        # OPTION A: Laplacian Uncertainty
        # -------------------------------------------------------------
        # 1. Get Predictions (Specific to Z and Uncertainty)
        # We use the Normalized Z (0-1) for stability
        src_norm_z = outputs['pred_trans_z'][idx] # Normalized between 0-1
        # Get the Log Variance (s)
        src_log_var = outputs['pred_z_log_var'][idx]
        # safety guardrail because we are useing exp(-s) -> can explode/NaN
        src_log_var = torch.clamp(src_log_var, min=-10.0, max=10.0)
        # 2. Get Targets (Meters)
        # Extract the full translation vector from targets
        tgt_trans_z = torch.cat([t['relative_translation_z'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        # 4. Compute Laplacian Loss
        # Formula: L = |y - y_hat| * exp(-s) + s
        l1_error = torch.abs(src_norm_z - tgt_trans_z)
        loss = (l1_error * torch.exp(-src_log_var)) + src_log_var
        loss = loss.sum() / num_boxes
        
        # # -------------------------------------------------------------
        # # OPTION B: Log-L1 Loss (Ablation Experiment)
        # # -------------------------------------------------------------
        # # USE: Metric Z (Meters)
        # # Why: Physically intuitive log-ratio.
        # # -------------------------------------------------------------
        # # Extract Z from the Metric Translation vector
        # src_z_meters = outputs['pred_translations'][idx][:, 2] 
        # tgt_trans = torch.cat([t['relative_position'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        # tgt_z_meters = tgt_trans[:, 2] # Already in meters
        # # Calculate SUM of errors
        # loss_z_sum = self.loss_relative_log_l1(src_z_meters, tgt_z_meters)
        # # Normalize by global object count (Safety check for divide-by-zero)
        # # Note: num_boxes is usually synced across GPUs in DETR
        # loss = loss_z_sum / num_boxes
        
        return {'loss_trans_z': loss}
    ####################################################################################
    
    #####T6D Direct: Symmetric Aware loss for rotation | translation uses the same as PoET ########################
    def loss_rotation_sym_aware_T6D(self, outputs, targets, indices, num_boxes):
        """
        Compute symmetry-aware rotation loss as per equation (8).
        
        For symmetric objects: L_R = (1/|M|) * sum_{x1∈M} min_{x2∈M} ||R_gt @ x1 - R_pred @ x2||
        For non-symmetric:     L_R = (1/|M|) * sum_{x∈M} ||R_gt @ x - R_pred @ x||
        
        Args:
            outputs: dict containing 'pred_rotations' [batch_size, num_queries, 3, 3]
            targets: list of dicts, each containing:

                - 'rotation': [num_objects, 3, 3] ground truth rotation matrices
                - 'model_points': [num_objects, M, 3] canonical model points

                - 'is_symmetric': [num_objects] boolean tensor indicating symmetry

            indices: list of tuples (src_idx, tgt_idx) from Hungarian matching
            num_boxes: optional normalization factor
        
        Returns:
            dict with 'loss_rot'
        """
        idx = self._get_src_permutation_idx(indices)
        
        # Get predicted rotations [N_matched, 3, 3]
        src_rotations = outputs["pred_rotations"][idx]
        
        # Get target rotations [N_matched, 3, 3]
        tgt_rotations = torch.cat(
            [t['relative_rotation'][i] for t, (_, i) in zip(targets, indices)], dim=0
        )
        # Get model points [N_matched, M, 3]
        model_points = torch.cat(
            [t['model_points'][i] for t, (_, i) in zip(targets, indices)], dim=0
        )
        # Get symmetry flags [N_matched]
        is_symmetric = torch.cat(
            [t['is_symmetric'][i] for t, (_, i) in zip(targets, indices)], dim=0
        )
        
        # Rotate model points with ground truth and predicted rotations
        # [N, 3, 3] @ [N, 3, M] -> [N, 3, M] -> [N, M, 3]
        tgt_points = torch.bmm(tgt_rotations, model_points.transpose(1, 2)).transpose(1, 2)
        src_points = torch.bmm(src_rotations, model_points.transpose(1, 2)).transpose(1, 2)
        
        # === Non-symmetric loss: direct point correspondence ===
        diff_nonsym = tgt_points - src_points  # [N, M, 3]
        loss_nonsym = torch.norm(diff_nonsym, dim=2).mean(dim=1)  # [N]
        
        # === Symmetric loss: minimum over correspondences ===
        # Compute pairwise distances: ||R_gt @ x1 - R_pred @ x2|| for all x1, x2 in M
        diff_sym = tgt_points.unsqueeze(2) - src_points.unsqueeze(1)  # [N, M, M, 3]
        dist_sym = torch.norm(diff_sym, dim=3)  # [N, M, M]
        # For each x1, find min over x2, then average over x1
        loss_sym = dist_sym.min(dim=2)[0].mean(dim=1)  # [N]
        
        # Select loss based on symmetry flag
        loss_per_obj = torch.where(is_symmetric, loss_sym, loss_nonsym)
        
        losses = {}
        losses["loss_rot"] = loss_per_obj.sum() / num_boxes
        
        return losses    
    ############# T6D symmetric aware loss end  ########################
    
    # Put all pose losses together
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
        if "relative_rotation" in targets[0]:     
            # Loss weights from paper (empirically tuned)
            # Don't change this here unless you have a good reason!
            # You can also do this via the loss coefficients in the main training script
            lambda_adds = 1.0 
            lambda_kpt = 1.0
            lambda_trans_xy = 1.0
            lambda_trans_z = 1.0
            lambda_rot = 1.0

            # Compute individual losses
            loss_adds_dict = self.loss_adds(outputs, targets, indices, num_boxes)
            loss_rot_dict = self.loss_rotation(outputs, targets, indices, num_boxes)
            #loss_rot_dict = self.loss_rot(outputs, targets, indices, num_boxes)
            loss_kpt_dict = self.loss_keypoint(outputs, targets, indices, num_boxes)
            loss_trans_xy = self.loss_trans_xy(outputs, targets, indices, num_boxes)
            loss_trans_z = self.loss_trans_z(outputs, targets, indices, num_boxes)
            
            # Extract loss values
            loss_adds = loss_adds_dict['loss_adds']
            loss_rot = loss_rot_dict['loss_rot']
            #loss_trans = loss_trans_dict['loss_translation']
            loss_kpt = loss_kpt_dict['loss_keypoint']
            loss_trans_xy = loss_trans_xy['loss_trans_xy']
            loss_trans_z = loss_trans_z['loss_trans_z']

            # Total weighted pose loss
            loss_pose_total = (
                lambda_adds * loss_adds +
                lambda_rot * loss_rot +
                lambda_kpt * loss_kpt +
                lambda_trans_z * loss_trans_z +
                lambda_trans_xy * loss_trans_xy
            )
            
            # Return all components for logging
            return {
                'loss_pose': loss_pose_total,
                'loss_rot': loss_rot,
                'loss_keypoint': loss_kpt,
                'loss_trans_xy': loss_trans_xy,
                'loss_trans_z': loss_trans_z,
                'loss_adds': loss_adds
            }
        else:
            return {}

    ######################################################

    

    ############# 2D Detection Losses ########################
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
        # 6D Poses
        out_translations, out_trans_z, out_uv, out_rot, out_z_log_var = outputs['pred_translations'], outputs['pred_trans_z'], outputs['pred_uv_norm'], outputs['pred_rotations'], outputs['pred_z_log_var']
         
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

        # and 6D Poses
        topk_rotations = topk_indexes // out_logits.shape[2]
        rotations = torch.gather(out_rot, 1, topk_rotations.unsqueeze(-1).unsqueeze(-1).repeat(1,1,3,3))
        
        topk_uv = topk_indexes // out_logits.shape[2]
        uvs = torch.gather(out_uv, 1, topk_uv.unsqueeze(-1).repeat(1,1,2))
        # Denormalize uv
        uv_scale_fct = torch.stack([img_w, img_h], dim=1)
        keypoints = uvs * uv_scale_fct[:, None, :]

        topk_translations = topk_indexes // out_logits.shape[2]
        translations = torch.gather(out_translations, 1, topk_translations.unsqueeze(-1).repeat(1,1,3))

        topk_trans_z = topk_indexes // out_logits.shape[2]
        trans_z = torch.gather(out_trans_z.unsqueeze(-1), 1, topk_trans_z.unsqueeze(-1).repeat(1,1,1))
        
        topk_log_va_z = topk_indexes // out_logits.shape[2]
        z_log_var = torch.gather(out_z_log_var.unsqueeze(-1), 1, topk_log_va_z.unsqueeze(-1).repeat(1,1,1))

        results = [{'scores': s, 'labels': l, 'boxes': b, 'rotations': r, 'keypoints': u, 'trans': t, 'trans_z': tz, 'z_log_var': zlv} for s, l, b, r, u, t, tz, zlv in zip(scores, labels, boxes, rotations, keypoints, translations, trans_z, z_log_var)]

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
    
def load_pretrained_weights(model, ckpt_path):

    print(f"Loading weights from {ckpt_path}...")
    checkpoint = torch.load(ckpt_path, map_location='cpu')

    # Unpack state_dict
    if 'model' in checkpoint:
        pretrained_dict = checkpoint['model']
    elif 'state_dict' in checkpoint:
        pretrained_dict = checkpoint['state_dict']
    else:
        pretrained_dict = checkpoint
        
    model_dict = model.state_dict()
    
    # --- SURGERY STEP ---
    # We create a new dict containing ONLY the keys that match 
    # in both Name AND Shape.
    
    pretrained_dict_filtered = {
        k: v for k, v in pretrained_dict.items()
        if k in model_dict and model_dict[k].shape == v.shape
    }
    
    # Check what we are dropping (Mental check)
    dropped_keys = [k for k in pretrained_dict.keys() if k not in pretrained_dict_filtered]
    print(f"Dropped {len(dropped_keys)} keys (mismatched heads/layers).")
    
    # Specific check: Ensure Class Head was dropped
    # (Because Objects365 has 365 classes, you have 21)
    if any("class_embed" in k for k in dropped_keys):
        print(">> Success: Old Classification Head was removed.")
    else:
        print(">> Warning: Class head might have been loaded? Check shapes!")

    # Update the current model
    model_dict.update(pretrained_dict_filtered)
    model.load_state_dict(model_dict)

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
    if args.pretrain_weights is not None:
        load_pretrained_weights(model, Path(args.pretrain_weights))
    
    if args.resume is None:
        model.init_pose_heads()
    else: 
        print(f"Continue training at {args.resume}")

    print("Z-Head Bias:", model.dec_trans_z_head.layers[-1].bias.data) 
    matcher = build_matcher(args)
    weight_dict = {'loss_ce': args.cls_loss_coef, 
                   'loss_bbox': args.bbox_loss_coef, 
                   'loss_giou': args.giou_loss_coef,
                   'loss_trans_xy': args.trans_xy_loss_coef,
                   'loss_trans_z': args.trans_z_loss_coef,
                   'loss_keypoint': args.keypoint_loss_coef,
                   'loss_rot': args.rot_loss_coef
                   }
    
    # TODO this is a hack
    if args.aux_loss:
        aux_weight_dict = {}
        for i in range(args.dec_layers - 1):
            aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})
        if args.two_stage:
            aux_weight_dict.update({k + f'_enc': v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)

    losses = ['pose']
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
