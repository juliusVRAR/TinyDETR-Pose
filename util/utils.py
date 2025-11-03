from copy import deepcopy
#import copy
import torch
import json
from collections import OrderedDict

import os
import torchvision
import random
from pathlib import Path

from PIL import Image
import torchvision.transforms.functional as TF
import torch.nn.functional as F
import numpy as np
#from util.visualize_object_pose import project_3d_2d, draw_cuboid_2d, draw_bbox_2d, Colors





DEBUG_OUT=Path("debug")

class ModelEma(torch.nn.Module):
    """EMA Model"""
    def __init__(self, model, decay=0.9997, device=None):
        super(ModelEma, self).__init__()
        # make a copy of the model for accumulating moving average of weights
        self.module = deepcopy(model)
        self.module.eval()

        self.decay = decay
        self.device = device  # perform ema on different device from model if set
        if self.device is not None:
            self.module.to(device=device)

    def _update(self, model, update_fn):
        with torch.no_grad():
            for ema_v, model_v in zip(
                self.module.state_dict().values(), model.state_dict().values()):
                if self.device is not None:
                    model_v = model_v.to(device=self.device)
                ema_v.copy_(update_fn(ema_v, model_v))

    def update(self, model):
        self._update(model, update_fn=lambda e, m: self.decay * e + (1. - self.decay) * m)

    def set(self, model):
        self._update(model, update_fn=lambda e, m: m)


class BestMetricSingle():
    def __init__(self, init_res=0.0, better='large') -> None:
        self.init_res = init_res
        self.best_res = init_res
        self.best_ep = -1

        self.better = better
        assert better in ['large', 'small']

    def isbetter(self, new_res, old_res):
        if self.better == 'large':
            return new_res > old_res
        if self.better == 'small':
            return new_res < old_res

    def update(self, new_res, ep):
        if self.isbetter(new_res, self.best_res):
            self.best_res = new_res
            self.best_ep = ep
            return True
        return False

    def __str__(self) -> str:
        return "best_res: {}\t best_ep: {}".format(self.best_res, self.best_ep)

    def __repr__(self) -> str:
        return self.__str__()

    def summary(self) -> dict:
        return {
            'best_res': self.best_res,
            'best_ep': self.best_ep,
        }

class BestMetricHolder():
    def __init__(self, init_res=0.0, better='large', use_ema=False) -> None:
        self.best_all = BestMetricSingle(init_res, better)
        self.use_ema = use_ema
        if use_ema:
            self.best_ema = BestMetricSingle(init_res, better)
            self.best_regular = BestMetricSingle(init_res, better)

    def update(self, new_res, epoch, is_ema=False):
        """
        return if the results is the best.
        """
        if not self.use_ema:
            return self.best_all.update(new_res, epoch)
        else:
            if is_ema:
                self.best_ema.update(new_res, epoch)
                return self.best_all.update(new_res, epoch)
            else:
                self.best_regular.update(new_res, epoch)
                return self.best_all.update(new_res, epoch)

    def summary(self):
        if not self.use_ema:
            return self.best_all.summary()

        res = {}
        res.update({f'all_{k}':v for k,v in self.best_all.summary().items()})
        res.update({f'regular_{k}':v for k,v in self.best_regular.summary().items()})
        res.update({f'ema_{k}':v for k,v in self.best_ema.summary().items()})
        return res

    def __repr__(self) -> str:
        return json.dumps(self.summary(), indent=2)

    def __str__(self) -> str:
        return self.__repr__()

def clean_state_dict(state_dict):
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if k[:7] == 'module.':
            k = k[7:]  # remove `module.`
        new_state_dict[k] = v
    return new_state_dict

#### Visualize to make sure everything is setup correctly
def draw_annotations(image_tensor, targets, is_bbbox_coords_normalized=True, is_corrected_bbx_coords=False): 
    """
    Visualize image with bounding boxes
    Args:
        image: PIL Image
        targets: List of target dictionaries containing bbox information
        class_names: Optional list of class names
    """
    # Sample colors for each bounding box
    COLORS = []
    for bbox in targets["boxes"]:
        COLORS.append(tuple(random.sample(range(255), 3)))
    
    # Convert to tensor and normalize (0.,1.)
    if type(image_tensor) is not torch.Tensor:
        image_tensor = torchvision.transforms.ToTensor()(image_tensor)
    
    # Extract bounding boxes and labels
    boxes = []
    # TODO: Add labels to vizualization
    labels = []
    width, height = image_tensor.shape[2], image_tensor.shape[1]
    for bbox in targets["boxes"]:
        # Bbox format: [x, y, width, height] -> [x1, y1, x2, y2]
        x1, y1, w, h = bbox
        x2, y2 = x1 + w, y1 + h
         # de-normalize bbox coordinates and 
        if is_bbbox_coords_normalized:
            x_center, y_center, w, h = bbox
            x1 = (x_center - w/2) * width
            y1 = (y_center - h/2) * height
            x2 = (x_center + w/2) * width
            y2 = (y_center + h/2) * height
        boxes.append([x1, y1, x2, y2])
    # Convert to tensors
    boxes_tensor = torch.tensor(boxes, dtype=torch.float32)
    # Draw bounding boxes
    annotated_image = torchvision.utils.draw_bounding_boxes(
        image_tensor, 
        boxes_tensor, 
        colors=COLORS,
        width=5
    )
    # If jitter is applied we can viz that too
    if "jitter_boxes" in targets:
        jitter_boxes = []
        for jitter_bbox in targets["jitter_boxes"]:
            # Bbox format: [x, y, width, height] -> [x1, y1, x2, y2]
            x1, y1, w, h = jitter_bbox
            x2, y2 = x1 + w, y1 + h
            # de-normalize bbox coordinates
            x_center, y_center, w, h = bbox
            x1 = (x_center - w/2) * width
            y1 = (y_center - h/2) * height
            x2 = (x_center + w/2) * width
            y2 = (y_center + h/2) * height
            jitter_boxes.append([x1, y1, x2, y2])
        annotated_jitter_image = torchvision.utils.draw_bounding_boxes(
            image_tensor, 
            boxes_tensor, 
            colors=COLORS,
            width=2
            )
        return annotated_image, annotated_jitter_image
    if is_corrected_bbx_coords: 
        boxes = targets["boxes"]
        annotated_image = torchvision.utils.draw_bounding_boxes(
        image_tensor, 
        boxes, 
        colors=COLORS,
        width=5
        )
        return annotated_image, None
    return annotated_image, None
      
# If you want to save the annotated image
def save_annotated_image(image, targets, output_path="annotated_image.png", is_bbbox_coords_normalized=True,  is_corrected_bbx_coords=False):
    """
    Save image with bounding boxes
    """
    annotated_image, annotated_jitter_image = draw_annotations(image_tensor=image, 
                                                               targets=targets, 
                                                               is_bbbox_coords_normalized=is_bbbox_coords_normalized, 
                                                               is_corrected_bbx_coords=is_corrected_bbx_coords)
    
    if not os.path.exists(DEBUG_OUT):
        os.makedirs(DEBUG_OUT)
    
    output_path = Path(DEBUG_OUT,output_path)
    torchvision.utils.save_image(annotated_image, output_path)
    print(f"Annotated image saved as {output_path}")

    if annotated_jitter_image is not None:
        jitter_output = "annotated_jitter_image.png"
        jitter_output = Path(DEBUG_OUT,jitter_output)
        grid_output = "annotated_grid_left_normal_right_jitter.png"
        grid_output = Path(DEBUG_OUT,grid_output)
        torchvision.utils.save_image(annotated_jitter_image, jitter_output)
        grid = torchvision.utils.make_grid([annotated_image, annotated_jitter_image], nrow=2, padding=2)
        torchvision.utils.save_image(grid, grid_output, nrow=2, padding=2)
        print(f"Grid saved as {grid_output}")
        print(f"Annotated jitter image saved as {jitter_output}")
    return annotated_image

def camera_params_to_K(cam_params):
    """
    Convert camera parameters to intrinsic matrix K
    
    Args:
        cam_params: Dictionary with keys 'fx', 'fy', 'cx', 'cy'
    
    Returns:
        3x3 camera intrinsic matrix K
    """
    K = [
        [cam_params['fx'], 0, cam_params['cx']],
        [0, cam_params['fy'], cam_params['cy']],
        [0, 0, 1]
    ]
    return K

def pad_to_size(img, target_h=512, target_w=640, fill=0):
    # Returns an image padded to (target_h, target_w).
    # Pads on the bottom and right only.
    if isinstance(img, Image.Image):
        w, h = img.size
        pad_w = max(0, target_w - w)
        pad_h = max(0, target_h - h)
        # padding order: (left, top, right, bottom)
        return TF.pad(img, (0, 0, pad_w, pad_h), fill=fill)
    elif isinstance(img, torch.Tensor):
        # expects CHW
        if img.dim() != 3:
            raise ValueError("Tensor image must be CHW")
        _, h, w = img.shape
        pad_w = max(0, target_w - w)
        pad_h = max(0, target_h - h)
        # pad order (left, right, top, bottom) for 2D dims (W,H) on CHW
        return F.pad(img, (0, pad_w, 0, pad_h), mode="constant", value=fill)
    else:
        raise TypeError("img must be PIL.Image or CHW torch.Tensor")