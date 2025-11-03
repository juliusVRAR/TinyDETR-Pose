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

from .backbone import build_backbone
from .matcher import build_matcher
from .transformer import build_transformer
from .position_encoding import BoundingBoxEmbeddingSine
from util.rotation_utils import so3_log_map

class LWDETR6D(nn.Module):
    """ This is the Group DETR v3 module that performs object detection """
    def __init__(self,
                 backbone,
                 transformer,
                 num_classes,
                 num_queries,
                 num_feature_levels,
                 aux_loss=False,
                 group_detr=1,
                 two_stage=False,
                 lite_refpoint_refine=False,
                 bbox_reparam=False,
                 bbox_mode='gt',
                 ref_points_mode='bbox', 
                 query_embedding_mode='bbox', 
                 rotation_mode='6d',
                 class_mode='agnostic',
                 aleatoric=False
                 ):
        """ Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            num_classes: number of object classes
            num_feature_levels: number of feature levels that serve as input to the transformer.
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
        self.hidden_dim = hidden_dim
        self.class_embed = nn.Linear(hidden_dim, num_classes)
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        self.bbox_mode = bbox_mode
        self.ref_points_mode = ref_points_mode
        self.query_embedding_mode = query_embedding_mode
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
        self.rotation_mode = rotation_mode
        
        # Determine Translation and Rotation head output dimension
        self.t_dim = 3
        if self.rotation_mode == '6d':
            self.rot_dim = 6
        elif self.rotation_mode in ['quat', 'silho_quat']:
            self.rot_dim = 4
        else:
            raise NotImplementedError('Rotational representation is not supported.')
       self.dec_rot_head  = nn.ModuleList([MLP(input_dim=d_model, 
                                                hidden_dim=d_model, 
                                                output_dim=self.rot_dim,
                                                num_layers=3) 
                                                for _ in range(group_detr)])
        self.dec_trans_head = nn.ModuleList([MLP(input_dim=d_model, 
                                                 hidden_dim=d_model, 
                                                 output_dim=self.t_dim, 
                                                 num_layers=2) 
                                                 for _ in range(group_detr)])
        # # I assume this will not work since we have a different backbone structure.
        # # TODO Verify if this works as intended.
        # # Whats going on here?
        # self.num_feature_levels = num_feature_levels
        # if num_feature_levels > 1:
        #     # Use multi-scale features as input to the transformer
        #     num_backbone_outs = len(backbone.strides)
        #     input_proj_list = []
        #     # If multi-scale then every intermediate backbone feature map is returned
        #     for n in range(num_backbone_outs):
        #         in_channels = backbone.num_channels[n]
        #         input_proj_list.append(nn.Sequential(
        #             nn.Conv2d(in_channels, hidden_dim, kernel_size=1),
        #             nn.GroupNorm(32, hidden_dim),
        #         ))
        #     # If more feature levels are required than backbone feature maps are available then the last feature map is
        #     # passed through an additional 3x3 Conv layer to create a new feature map.
        #     # This new feature map is then used as the baseline for the next feature map to calculate
        #     # For details refer to the Deformable DETR paper's appendix.
        #     for n in range(num_feature_levels - num_backbone_outs):
        #         input_proj_list.append(nn.Sequential(
        #             nn.Conv2d(in_channels, hidden_dim, kernel_size=3, stride=2, padding=1),
        #             nn.GroupNorm(32, hidden_dim),
        #         ))
        #         in_channels = hidden_dim
        #     self.input_proj = nn.ModuleList(input_proj_list)
        # else:
        #     # We only want to use the backbones last feature embedding map.
        #     self.input_proj = nn.ModuleList([
        #         nn.Sequential(
        #             nn.Conv2d(backbone.num_channels[0], hidden_dim, kernel_size=1),
        #             nn.GroupNorm(32, hidden_dim),
        #         )
        #     ])
        ######
        # Initialize the projection layers ???
        # for proj in self.input_proj:
        #     nn.init.xavier_uniform_(proj[0].weight, gain=1)
        #     nn.init.constant_(proj[0].bias, 0)
        
        # TODO: What is this doing? Where does this come from?
        # Positional Embedding for bounding boxes to generate query embeddings
        # We dont need this this comes from the transfomer i think
        if self.query_embedding_mode == 'bbox':
            self.bbox_embedding = BoundingBoxEmbeddingSine(num_pos_feats=hidden_dim / 8)
        elif self.query_embedding_mode == 'learned':
            self.query_embed = nn.Embedding(num_queries, hidden_dim * 2)
            # TODO: Optimize Code to not generate bounding box query embeddings, when query embed is in learning mode.
            self.bbox_embedding = BoundingBoxEmbeddingSine(num_pos_feats=hidden_dim / 8)
        else:
            raise NotImplementedError('This query embedding mode is not implemented.')

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

             Functions expects a list of length batch_size, where each element is a dict with the following entries:
            - boxes: tensor of size [n_obj, 4], contains the bounding box (x_c, y_c, w, h) of each object in each image
            normalized to image size
            - labels: tensor of size [n_obj, ], contains the label of each object in the image
            - image_id: tensor of size [1],  contains the image id to which this annotation belongs to
            - relative_position; tensor of size [n_obj, 3], contains the relative translation for each object present
            in the image w.r.t the camera.
            - relative_rotation: tensor of size [n_obj, 3, 3], contains the relative rotation for each object present
            in the image w.r.t. the camera.  

            It returns a dict with the following elements:
               - "pred_logits": the classification logits (including no-object) for all queries.
                                Shape= [batch_size x num_queries x num_classes]
               - "pred_boxes": The normalized boxes coordinates for all queries, represented as
                               (center_x, center_y, width, height). These values are normalized in [0, 1],
                               relative to the size of each individual image (disregarding possible padding).
                               See PostProcess for information on how to retrieve the unnormalized bounding box.
               - "aux_outputs": Optional, only returned when auxilary losses are activated. It is a list of
                                dictionnaries containing the two above keys for each decoder layer.
                - pred_translation: tensor of size [batch_size, num_queries, 3], predicted relative translation for each
                                object query w.r.t. camera
                - pred_rotation: tensor of size [batch_size, num_queries, 3, 3], predicted relative rotation for each
                                object query w.r.t. camera
                - pred_classes: tensor of size [batch_size, num_queries], predicted class for each
                                object query
            It returns a list "n_boxes_per_sample" of length [batch_size, 1], which contains the number of
        """
        if isinstance(samples, (list, torch.Tensor)):
            samples = nested_tensor_from_tensor_list(samples)
        features, poss = self.backbone(samples) # Feature extraction using backbone

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

        out = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1]}
        if self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_coord)

        if self.two_stage:
            hs_enc_list = hs_enc.split(self.num_queries, dim=1)
            cls_enc = []
            group_detr = self.group_detr if self.training else 1
            for g_idx in range(group_detr):
                cls_enc_gidx = self.transformer.enc_out_class_embed[g_idx](hs_enc_list[g_idx])
                cls_enc.append(cls_enc_gidx)
            cls_enc = torch.cat(cls_enc, dim=1)
            out['enc_outputs'] = {'pred_logits': cls_enc, 'pred_boxes': ref_enc}

        #return out
        ############
        # We did 2d detection up to this point. Now we add the 6D pose estimation heads.
        
        # Store the image size in HxW
        image_sizes = [[sample.shape[-2], sample.shape[-1]] for sample in samples.tensors]
        # Extract the bounding boxes for each batch element
        pred_boxes = []
        pred_classes = []
        query_embeds = []
        n_boxes_per_sample = []

        # Depending on the bbox mode, we either use ground truth bounding boxes or backbone predicted bounding boxes for
        # transformer query input embedding calculation.
        if self.bbox_mode in ['gt', 'jitter'] and targets is not None:
            for t, target in enumerate(targets):
                # GT from COCO loaded as x1,y1,x2,y2, but by data loader transformed to cx, cy, w, h and normalized
                if self.bbox_mode == 'gt':
                    t_boxes = target["boxes"]
                elif self.bbox_mode == 'jitter':
                    t_boxes = target["jitter_boxes"]
                n_boxes = len(t_boxes)
                n_boxes_per_sample.append(n_boxes)

                # Add classes
                t_classes = target["labels"]

                # For the current number of boxes determine the query embedding
                query_embed = self.bbox_embedding(t_boxes)
                # As the embedding will serve as the query and key for attention, duplicate it to be later splitted
                query_embed = query_embed.repeat(1, 2)

                # We always predict a fixed number of object poses per image set to the maximum number of objects
                # present in a single image throughout the whole dataset. Check whether this upper limit is reached,
                # otherwise fill up with dummy embeddings that are defined as cx,cy,w,h = [-1, -1, -1, -1]
                # Dummy boxes will later be filtered out by the matcher and not used for cost calculation
                if n_boxes < self.num_queries:
                    dummy_boxes = torch.tensor([[-1, -1, -1, -1] for i in range(self.num_queries-n_boxes)],
                                               dtype=torch.float32, device=t_boxes.device)

                    dummy_embed = torch.tensor([[-10] for i in range(self.num_queries-n_boxes)],
                                               dtype=torch.float32, device=t_boxes.device)
                    dummy_embed = dummy_embed.repeat(1, self.hidden_dim*2)
                    t_boxes = torch.vstack((t_boxes, dummy_boxes))
                    query_embed = torch.cat([query_embed, dummy_embed], dim=0)
                    dummy_classes = torch.tensor([-1 for i in range(self.num_queries-n_boxes)],
                                               dtype=torch.int, device=t_boxes.device)
                    t_classes = torch.cat((t_classes, dummy_classes))
                pred_boxes.append(t_boxes)
                query_embeds.append(query_embed)
                pred_classes.append(t_classes)
        
        elif self.bbox_mode == 'backbone':
            # Prepare the output predicted by the backbone
            # Iterate over batch and prepare each image in batch
            bs = out["pred_boxes"].shape[0]
            for b in range(bs):
                backbone_boxes = out["pred_boxes"][b]     # (Q,4) normalized cxcywh
                backbone_scores = out["pred_logits"][b]   # (Q,C) class logits
                if out is None:
                    # Case: Backbone has not predicted anything for image
                    # Add only dummy boxes, but mark that nothing has been predicted
                    n_boxes = 0
                    n_boxes_per_sample.append(n_boxes)
                    backbone_boxes = torch.tensor([[-1, -1, -1, -1] for i in range(self.num_queries - n_boxes)],
                                                  dtype=torch.float32, device=features[0].decompose()[0].device)
                    query_embed = torch.tensor([[-10] for i in range(self.num_queries - n_boxes)],
                                               dtype=torch.float32, device=features[0].decompose()[0].device)
                    query_embed = query_embed.repeat(1, self.hidden_dim * 2)
                    backbone_classes = torch.tensor([-1 for i in range(self.num_queries - n_boxes)], dtype=torch.int64,
                                                    device=features[0].decompose()[0].device)
                    
                    print("Backbone did not predict any boxes for this image!")
                else:
                    # Case: Backbone predicted something
                    backbone_boxes = box_ops.box_xyxy_to_cxcywh(backbone_boxes)
                    # TODO: Adapt to different image sizes as we assume constant image size across the batch
                    backbone_boxes = box_ops.box_normalize_cxcywh(backbone_boxes, image_sizes[0])
                    n_boxes = len(backbone_boxes)

                    # Predicted classes by backbone // class 0 is "background"
                    # Scores predicted by the backbone are needed for top-k selection
                    # TODO: Continue here
                    # Can this case even happen ???? and is this correct ????a
                    cls_logits = out['pred_logits']             # (Q,C)
                    probs = cls_logits.sigmoid()  
                    scores, backbone_classes = probs.max(dim=1)
                    backbone_classes = backbone_classes.type(torch.int64)

                    # For the current number of boxes determine the query embedding
                    query_embed = self.bbox_embedding(backbone_boxes)
                    # As the embedding will serve as the query and key for attention, duplicate it to be later splitted
                    query_embed = query_embed.repeat(1, 2)

                    if n_boxes < self.num_queries:
                        # Fill up with dummy boxes to match the query size and add dummy embeddings
                        dummy_boxes = torch.tensor([[-1, -1, -1, -1] for i in range(self.num_queries - n_boxes)],
                                                   dtype=torch.float32, device=backbone_boxes.device)
                        dummy_embed = torch.tensor([[-10] for i in range(self.num_queries - n_boxes)],
                                                   dtype=torch.float32, device=backbone_boxes.device)
                        dummy_embed = dummy_embed.repeat(1, self.hidden_dim * 2)
                        backbone_boxes = torch.cat([backbone_boxes, dummy_boxes], dim=0)
                        query_embed = torch.cat([query_embed, dummy_embed], dim=0)
                        dummy_classes = torch.tensor([-1 for i in range(self.num_queries - n_boxes)],
                                                     dtype=torch.int64, device=backbone_boxes.device)
                        backbone_classes = torch.cat([backbone_classes, dummy_classes], dim=0)
                    elif n_boxes > self.num_queries:
                        # Number of boxes will be limited to the number of queries
                        n_boxes = self.num_queries
                        # Case: backbone predicts more output objects than queries available --> take top num_queries
                        # Sort scores to get the post top performing ones
                        backbone_scores, indices = torch.sort(backbone_scores, dim=0, descending=True)
                        backbone_classes = backbone_classes[indices]
                        backbone_boxes = backbone_boxes[indices, :]
                        query_embed = query_embed[indices, :]

                        # Take the top n predictions
                        backbone_scores = backbone_scores[:self.num_queries]
                        backbone_classes = backbone_classes[:self.num_queries]
                        backbone_boxes = backbone_boxes[:self.num_queries]
                        query_embed = query_embed[:self.num_queries]
                    n_boxes_per_sample.append(n_boxes)
                pred_boxes.append(backbone_boxes)
                pred_classes.append(backbone_classes)
                query_embeds.append(query_embed)
        else:
            raise NotImplementedError("Bounding Box Mode not implemented!")
        query_embeds = torch.stack(query_embeds)
        pred_boxes = torch.stack(pred_boxes)
        pred_classes = torch.stack(pred_classes)
        # if out is not None:
        #     pred_classes = torch.squeeze(pred_classes, 1)
        print("Here")

        if self.ref_points_mode == 'bbox':
            reference_points = pred_boxes[:, :, :2]
        else:
            reference_points = None
        if self.query_embedding_mode == 'learned':
            query_embeds = self.query_embed.weight

        # Pass everything to the transformer
        hs, init_reference, _, _, _ = self.transformer(srcs, 
                                                       masks, 
                                                       poss, 
                                                       query_embeds, 
                                                       reference_points)
        outputs_translation = []
        outputs_rotation = []
        if self.aleatoric:
            outputs_translation_aleatoric = []
            outputs_rotation_aleatoric = []
        bs, _ = pred_classes.shape 
        output_idx = torch.where(pred_classes > 0, pred_classes, 0).view(-1)
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
    def _set_aux_loss(self, outputs_class, outputs_coord):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{'pred_logits': a, 'pred_boxes': b}
                for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]

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
            'quat' or 'silho_quat' --> L2 normalization
            else: Raise error
            """
            if self.rotation_mode == '6d':
                return self.rotation_6d_to_matrix(pred_rotation)
            elif self.rotation_mode in ['quat', 'silho_quat']:
                return F.normalize(pred_rotation, p=2, dim=2)
            else:
                raise NotImplementedError('Rotation mode is not supported')

    def rotation_6d_to_matrix(self, rot_6d):
        """
        Given a 6D rotation output, calculate the 3D rotation matrix in SO(3) using the Gramm Schmit process

        For details: https://openaccess.thecvf.com/content_CVPR_2019/papers/Zhou_On_the_Continuity_of_Rotation_Representations_in_Neural_Networks_CVPR_2019_paper.pdf
        """
        bs, n_q, _ = rot_6d.shape
        rot_6d = rot_6d.view(-1, 6)
        m1 = rot_6d[:, 0:3]
        m2 = rot_6d[:, 3:6]

        x = F.normalize(m1, p=2, dim=1)
        z = torch.cross(x, m2, dim=1)
        z = F.normalize(z, p=2, dim=1)
        y = torch.cross(z, x, dim=1)
        rot_matrix = torch.cat((x.view(-1, 3, 1), y.view(-1, 3, 1), z.view(-1, 3, 1)), 2)  # Rotation Matrix lying in the SO(3)
        rot_matrix = rot_matrix.view(bs, n_q, 3, 3)  #.transpose(2, 3)
        return rot_matrix


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
                 ia_bce_loss=False,):
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
    # Poet Losses
    def loss_translation(self, outputs, targets, indices):
        """
        Compute the loss related to the translation of pose estimation, namely the mean square error (MSE).
        outputs must contain the key 'pred_translation', while targets must contain the key 'relative_position'
        Position / Translation are expected in [x, y, z] meters
        """
        idx = self._get_src_permutation_idx(indices)
        src_translation = outputs["pred_translation"][idx]
        tgt_translation = torch.cat([t['relative_position'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        n_obj = len(tgt_translation)

        loss_translation = F.mse_loss(src_translation, tgt_translation, reduction='none')
        loss_translation = torch.sum(loss_translation, dim=1)
        loss_translation = torch.sqrt(loss_translation)
        losses = {}
        losses["loss_trans"] = loss_translation.sum() / n_obj
        return losses
    
    def loss_translation_aleatoric(self, outputs, targets, indices):
        """
        Extension of the translation loss to train for aleatoric uncertainty estimation.
        Loss is calculated according to: Aleatoric Uncertainty from AI-based 6D Object Pose Predictors for Object-relative State Estimation 
        (https://doi.org/10.1109/LRA.2025.3606700)(https://www.arxiv.org/abs/2509.01583)
        The paper also explains simplifications.
        """
        idx = self._get_src_permutation_idx(indices)
        src_translation = outputs["pred_translation"][idx]
        src_translation_aleatoric = outputs["pred_translation_aleatoric"][idx]
        tgt_translation = torch.cat([t['relative_position'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        n_obj = len(tgt_translation)

        diff_t = tgt_translation - src_translation
        # Special case: instead of sigma^2, we predict s = log(sigma^2) to ensure numerical stability and positiveness
        s_sum = torch.sum(src_translation_aleatoric, dim=1)
        exp_neg_s = torch.exp(-src_translation_aleatoric)
        scaled_squared_euclidean = exp_neg_s * torch.square(diff_t)
        scaled_squared_euclidean = torch.sum(scaled_squared_euclidean, dim=1)

        loss_translation_aleatoric = scaled_squared_euclidean + s_sum
        losses = {}
        losses["loss_trans"] = loss_translation_aleatoric.sum() / (2* n_obj)
        return losses
    def loss_rotation(self, outputs, targets, indices):
        """
        Compute the loss related to the rotation of pose estimation represented by a 3x3 rotation matrix.
        The function calculates the geodesic distance between the predicted and target rotation.
        L = arccos( 0.5 * (Trace(R\tilde(R)^T) -1)
        Calculates the loss in radiant.
        """
        eps = 1e-6
        idx = self._get_src_permutation_idx(indices)
        src_rot = outputs["pred_rotation"][idx]
        tgt_rot = torch.cat([t['relative_rotation'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        n_obj = len(tgt_rot)

        product = torch.bmm(src_rot, tgt_rot.transpose(1, 2))
        trace = torch.sum(product[:, torch.eye(3).bool()], 1)
        theta = torch.clamp(0.5 * (trace - 1), -1 + eps, 1 - eps)
        rad = torch.acos(theta)
        losses = {}
        losses["loss_rot"] = rad.sum() / n_obj
        return losses
    
    def loss_rotation_aleatoric(self, outputs, targets, indices):
        """
        Extension of the rotation loss to train for aleatoric uncertainty estimation.
        Loss is calculated according to: Aleatoric Uncertainty from AI-based 6D Object Pose Predictors for Object-relative State Estimation
        (https://doi.org/10.1109/LRA.2025.3606700)(https://www.arxiv.org/abs/2509.01583)
        The paper also explains simplifications.
        """
        eps = 1e-6
        idx = self._get_src_permutation_idx(indices)
        src_rot = outputs["pred_rotation"][idx]
        src_rot_aleatoric = outputs["pred_rotation_aleatoric"][idx]
        tgt_rot = torch.cat([t['relative_rotation'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        n_obj = len(tgt_rot)

        diff_matrix = torch.bmm(src_rot, tgt_rot.transpose(1, 2))
        # Special case: instead of sigma^2, we predict s = log(sigma^2) to ensure numerical stability and positiveness
        s_sum = torch.sum(src_rot_aleatoric, dim=1)
        exp_neg_s = torch.exp(-src_rot_aleatoric)

        # Transform diff matrices into the lie algebra so(3) using the logarithmic map
        v = so3_log_map(diff_matrix)
        scaled_squared_euclidean = exp_neg_s * torch.square(v)
        scaled_squared_euclidean = torch.sum(scaled_squared_euclidean, dim=1)
        loss_rotation_aleatoric = scaled_squared_euclidean + s_sum
        losses = {}
        losses["loss_rot"] = loss_rotation_aleatoric.sum() / (2 * n_obj)
        return losses

    def loss_quaternion(self, outputs, targets, indices):
        """
        Compute the loss related to the rotation of pose estimation represented in quaternions, namely the quaternion loss
        Q_loss = - log(<q_pred,pred_gt>² + eps), where eps is a small values for stability reasons

        outputs must contain the key 'pred_quaternion', while targets must contain the key 'relative_quaternions'
        Quaternions expected in representation [w, x, y, z]
        """
        eps = 1e-4
        idx = self._get_src_permutation_idx(indices)
        src_quaternion = outputs["pred_rotation"][idx]
        tgt_quaternion = torch.cat([t['relative_quaternions'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        n_obj = len(tgt_quaternion)
        bs, q_dim = tgt_quaternion.shape

        dot_product = torch.mul(src_quaternion, tgt_quaternion)
        dp_sum = torch.sum(dot_product, 1)
        dp_square = torch.square(dp_sum)
        loss_quat = - torch.log(dp_square + eps)

        losses = {}
        losses["loss_rot"] = loss_quat.sum() / n_obj
        return losses

    def loss_silho_quaternion(self, outputs, targets, indices):
        """
        Compute the loss related to the rotation of pose estimation represented in quaternions, namely the quaternion loss
        Q_loss = log(1 - |<q_pred,pred_gt>| + eps), where eps is a small values for stability reasons

        outputs must contain the key 'pred_quaternion', while targets must contain the key 'relative_quaternions'
        Quaternions expected in representation [w, x, y, z]
        """
        eps = 1e-4
        idx = self._get_src_permutation_idx(indices)
        src_quaternion = outputs["pred_rotation"][idx]
        tgt_quaternion = torch.cat([t['relative_quaternions'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        n_obj = len(tgt_quaternion)
        bs, q_dim = tgt_quaternion.shape

        dot_product = torch.mul(src_quaternion, tgt_quaternion)
        dp_sum = torch.sum(dot_product, 1)
        loss_quat = torch.log(1 - torch.abs(dp_sum) + eps)

        losses = {}
        losses["loss_rot"] = loss_quat.sum() / n_obj
        return losses
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

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {
            'labels': self.loss_labels,
            'cardinality': self.loss_cardinality,
            'boxes': self.loss_boxes,
            'translation': self.loss_translation,
            'rotation': self.loss_rotation,
            'quaternion': self.loss_quaternion,
            'silho_quaternion': self.loss_silho_quaternion,
            'aleatoric_translation': self.loss_translation_aleatoric,
            'aleatoric_rotation': self.loss_rotation_aleatoric,
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

    def forward(self, outputs, targets):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
             n_boxes: Number of predicted objects per image
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
                l_dict = self.get_loss(loss, enc_outputs, targets, indices, num_boxes, **kwargs)
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
        num_feature_levels=args.num_feature_levels,
        bbox_mode=args.bbox_mode,
        ref_points_mode=args.reference_points,
        rotation_mode="6d",
    )
    matcher = build_matcher(args)
    weight_dict = {'loss_ce': args.cls_loss_coef, 
                   'loss_bbox': args.bbox_loss_coef, 
                   'loss_trans': args.translation_loss_coef, 
                   'loss_rot': args.rotation_loss_coef}
    weight_dict['loss_giou'] = args.giou_loss_coef

    if args.rotation_representation == '6d':
        losses = ['translation', 'rotation']
    elif args.rotation_representation == 'quat':
        losses = ['translation', 'quaternion']
    elif args.rotation_representation == 'silho_quat':
        losses = ['translation', 'silho_quaternion']
    else:
        raise NotImplementedError('Rotation representation not implemented')
    
    if args.aleatoric:
        losses = ['aleatoric_' + loss for loss in losses]

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
                             ia_bce_loss=args.ia_bce_loss)
    criterion.to(device)
    postprocessors = {'bbox': PostProcess(num_select=args.num_select)}

    return model, criterion, postprocessors, matcher