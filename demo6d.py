# ------------------------------------------------------------------------
# PoseDETR Inference Demo (Full Script)
# ------------------------------------------------------------------------
import argparse

from pathlib import Path
import cv2
import numpy as np
import torch
import json
from PIL import Image
from torchvision import transforms
from util.get_param_dicts import get_param_dict
from util.visualize_object_pose import YCBVVisualizer
import util.misc as utils
# --- Import your model builder ---
# Ensure your PYTHONPATH is set correctly so python can find 'models'
from models import build_model
from util.misc import nested_tensor_from_tensor_list
from PIL import Image, ImageDraw, ImageFont
# YCB-Video Classes (Must match your training index 1-to-1)
YCB_CLASSES = [
    '__background__', '002_master_chef_can', '003_cracker_box', '004_sugar_box',
    '005_tomato_soup_can', '006_mustard_bottle', '007_tuna_fish_can', '008_pudding_box',
    '009_gelatin_box', '010_potted_meat_can', '011_banana', '019_pitcher_base',
    '021_bleach_cleanser', '024_bowl', '025_mug', '035_power_drill', '036_wood_block',
    '037_scissors', '040_large_marker', '051_large_clamp', '052_extra_large_clamp',
    '061_foam_brick'
]

def get_args_parser():
    parser = argparse.ArgumentParser('PoseDETR Inference', add_help=False)
    
    # Model parameters
    parser.add_argument('--weights', type=str, default=None, required=True,
                        help="Path to the model parameters.")
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--pretrained_encoder', type=str, default=None, 
                        help="Path to the pretrained encoder.")
    
    # * Backbone
    parser.add_argument('--encoder', default='vit_tiny', type=str,
                        help="Name of the transformer or convolutional encoder to use")
    parser.add_argument('--vit_encoder_num_layers', default=12, type=int,
                        help="Number of layers used in ViT encoder")
    parser.add_argument('--window_block_indexes', default=None, type=int, nargs='+')
    parser.add_argument('--position_embedding', default='sine', type=str, 
                        choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")
    parser.add_argument('--out_feature_indexes', default=[-1], type=int, nargs='+', help='only for vit now')
    parser.add_argument('--pretrain_weights', default=None, type=str,)
    parser.add_argument('--resume', default="inference", type=str,)
    # * Transformer
    parser.add_argument('--dec_layers', default=3, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=2048, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=256, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--sa_nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's self-attentions")
    parser.add_argument('--ca_nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's cross-attentions")
    parser.add_argument('--num_queries', default=50, type=int,
                        help="Number of query slots")
    parser.add_argument('--group_detr', default=13, type=int,
                        help="Number of groups to speed up detr training")
    parser.add_argument('--two_stage', action='store_true')
    parser.add_argument('--projector_scale', default='P4', type=str, nargs='+', choices=('P3', 'P4', 'P5', 'P6'))
    parser.add_argument('--lite_refpoint_refine', action='store_true', help='lite refpoint refine mode for speed-up')
    parser.add_argument('--num_select', default=100, type=int,
                        help='the number of predictions selected for evaluation')
    parser.add_argument('--dec_n_points', default=4, type=int,
                        help='the number of sampling points')
    parser.add_argument('--decoder_norm', default='LN', type=str)
    parser.add_argument('--bbox_reparam', action='store_true')

    # * Dataset infomation
    parser.add_argument('--dataset_file', default='coco')
    parser.add_argument('--num_classes', default=21, type=int,
                        help='number of object classes')
    parser.add_argument('--cad_models_path', default="None", type=str,
                        help='path to the folder containing 3D CAD models')
    # * Matcher
    parser.add_argument('--set_cost_class', default=2, type=float,
                        help="Class coefficient in the matching cost")
    parser.add_argument('--set_cost_bbox', default=5, type=float,
                        help="L1 box coefficient in the matching cost")
    parser.add_argument('--set_cost_giou', default=2, type=float,
                        help="giou box coefficient in the matching cost")
    parser.add_argument('--set_cost_rotation', default=2.0, type=float,
                        help="rotation coefficient in the matching cost")
    parser.add_argument('--set_cost_translation', default=5., type=float,
                        help="translation coefficient in the matching cost")
    parser.add_argument('--set_cost_keypoint', default=10., type=float,
                        help="keypoint coefficient in the matching cost")
    parser.add_argument('--matcher_type', default="6d", type=str, choices=['hungarian', 'yopo', '6d'],)
    # * Learning rate
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--lr_encoder', default=1.5e-4, type=float)
    parser.add_argument('--batch_size', default=2, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=12, type=int)
    parser.add_argument('--lr_drop', default=11, type=int)
    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='gradient clipping max norm')
    parser.add_argument('--lr_vit_layer_decay', default=0.8, type=float)
    parser.add_argument('--lr_component_decay', default=1.0, type=float)
    
    # * Drop args
    parser.add_argument('--dropout', type=float, default=0,
                        help='Drop path rate (default: 0.0)')
    parser.add_argument('--drop_path', type=float, default=0,
                        help='Drop path rate (default: 0.0)')

    # * Loss coefficients
    parser.add_argument('--cls_loss_coef', default=2, type=float)
    parser.add_argument('--bbox_loss_coef', default=5, type=float)
    parser.add_argument('--giou_loss_coef', default=2, type=float)
    parser.add_argument('--focal_alpha', default=0.25, type=float)
    
    # * Loss
    parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false',
                        help="Disables auxiliary decoding losses (loss at each layer)")
    parser.add_argument('--sum_group_losses', action='store_true',
                        help="To sum losses across groups or mean losses.")
    parser.add_argument('--use_varifocal_loss', action='store_true')
    parser.add_argument('--use_position_supervised_loss', action='store_true')
    parser.add_argument('--ia_bce_loss', action='store_true')
    parser.add_argument('--keypoint_loss_coef', default=10.0, type=float, help='Loss weighing parameter for the keypoints')
    parser.add_argument('--trans_z_loss_coef', default=1.0, type=float, help='Loss weighing parameter for the translation z component')
    parser.add_argument('--trans_xy_loss_coef', default=1.0, type=float, help='Loss weighing parameter for the translation')
    parser.add_argument('--rot_loss_coef', default=5.0, type=float, help='Loss weighing parameter for the rotation')
    parser.add_argument('--adds_loss_coef', default=1.0, type=float, help='Loss weighing parameter for the ADD-S metric. Active after warmup epochs.')
    parser.add_argument('--warm_up_epochs', default=15, type=int, help='Number of epochs before ADD-S loss multiplier is activated.')
    

    # * Input and output
    parser.add_argument('--input', default=None, required=True,
                        help='"Path to image file."')
    parser.add_argument('--output_dir', default='output',
                        help='Directory to save output visualizations.')
    parser.add_argument('--confidence_threshold', type=float, default=0.3,
                        help='Minimum score for instance predictions to be shown')

    # --- Camera Intrinsics (Defaults to YCB-Video Real Camera) ---
    # NOTE: Change these if testing on a Webcam or different dataset!
    parser.add_argument('--cam_matrix', type=str, default=None,
                        help='Optional path to a json file containing camera intrinsics matrix.')

    # Inference Options
    parser.add_argument('--uncertainty_threshold', type=float, default=None,
                        help='Optional threshold to filter predictions by uncertainty (Laplacian log-variance).')
   
    return parser

# -------------------------------------------------------------------------
# Helper: Resize & Pad while updating Intrinsics (The "Letterbox" Trick)
# -------------------------------------------------------------------------
def resize_pad_with_intrinsics(image, K, target_size=640, divisibility=64):
    """
    Resizes image (numpy) preserving aspect ratio, pads to be divisible by 64.
    Updates Intrinsics Matrix K automatically.
    """
    h, w = image.shape[:2]
    scale = min(target_size / h, target_size / w)
    
    # 1. New Dimensions
    new_w, new_h = int(w * scale), int(h * scale)
    
    # 2. Resize Image
    image_resized = cv2.resize(image, (new_w, new_h))
    
    # 3. Update Intrinsics (Scale fx, fy, cx, cy)
    # The math: If image shrinks by 0.5, focal length also shrinks by 0.5
    K_new = K.copy()
    K_new[0, 0] *= scale # fx
    K_new[1, 1] *= scale # fy
    K_new[0, 2] *= scale # cx
    K_new[1, 2] *= scale # cy
    
    # 4. Pad (Right-Bottom) to match target_size
    # We pad right/bottom so cx/cy (measured from top-left) DO NOT CHANGE further.
    delta_w = target_size - new_w
    delta_h = target_size - new_h
    
    image_padded = cv2.copyMakeBorder(
        image_resized, 
        0, delta_h, 0, delta_w, 
        cv2.BORDER_CONSTANT, 
        value=(0,0,0)
    )
    
    return image_padded, K_new

def preprocess_image(image_path):
    image = Image.open(image_path).convert("RGB")
    orig_image_size = torch.tensor(image.size[::-1])

    normalize = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    transform = transforms.Compose([
            transforms.Resize([640, 640]),
            normalize,
        ])
    image = transform(image)
    return image, orig_image_size

def visualize_detections(image, boxes, labels, scores, conf_thresh, output_path):
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()

    for box, label, score in zip(boxes, labels, scores):
        if score > conf_thresh:
            xmin, ymin, xmax, ymax = map(int, box)
            draw.rectangle([xmin, ymin, xmax, ymax], outline="green", width=2)
            text = f"{YCB_CLASSES[label]} {score:.2f}"
            draw.text((xmin, ymin - 10), text, fill="black", font=font)

    image.save(output_path)
def main(args):
    with open(args.cam_matrix) as f:
        K = json.load(f)
        print(K)
        # Define Original Intrinsics (K) from args
    K_orig = np.array([
        [K["fx"], 0, K["cx"]],
        [0, K["fy"], K["cy"]],
        [0, 0, 1]
    ], dtype=np.float32)    
    viz = YCBVVisualizer(args.cad_models_path)
    
    utils.init_distributed_mode(args)
    print(args)

    device = torch.device(args.device)

    model, _, postprocessors, _ = build_model(args)
    model.to(device)
    model.eval()

    param_dicts = get_param_dict(args, model)

    output_path = Path(args.output_dir) /  "visualize.jpg"

    if args.weights:
        checkpoint = torch.load(args.weights, map_location='cpu')
        model.load_state_dict(checkpoint['model'], strict=True)

    # preprocess
    image, orig_image_size = preprocess_image(args.input)
    image = image.to(device)
    orig_image_size = orig_image_size.to(device)

    images = nested_tensor_from_tensor_list([image])
    orig_image_sizes = torch.stack([orig_image_size])

    # forward
    with torch.no_grad():
        outputs = model(images)

    # postprocess
    predictions = postprocessors['bbox'](outputs, orig_image_sizes)

    # 2D Detections
    boxes = predictions[0]['boxes'].cpu().numpy()
    labels = predictions[0]['labels'].cpu().numpy()
    scores = predictions[0]['scores'].cpu().numpy()
    # 6D Poses 
    rots = predictions[0]['rotations'].cpu().numpy()
    trans = predictions[0]['trans'].cpu().numpy()
    trans_z = predictions[0]['trans_z'].cpu().numpy()
    z_log_var = predictions[0]['z_log_var'].cpu().numpy()
    keypoints = predictions[0]['keypoints'].cpu().numpy()
    

    print("Visualize 2D Detections...")
    original_image = Image.open(args.input).convert("RGB")
    visualize_detections(
        original_image,
        boxes,
        labels,
        scores,
        args.confidence_threshold,
        output_path)
    print("Visualize object poses...")
    im = np.array(original_image)
     # Visualization of 3D bboxes and overalayed objects
    vis_img = im[:,:,::-1].copy()
    vis_img = viz.visualize_single_image(vis_img                                                                                                         , 
                                        annotations={'labels': labels, 
                                                    "relative_position":trans, 
                                                    "relative_rotation":rots,
                                                    },
                                            K=K_orig,
                                            show_mesh=True,
                                            sample_points=5000,
                                            conf_threshold=args.confidence_threshold,
                                            scores=scores)
    cv2.imwrite(Path(args.output_dir, "vis3d.jpg"), vis_img) # Visualization of 3D bboxes and overalayed objects
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser('LWDETR infer script', parents=[get_args_parser()])
    args = parser.parse_args()

    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    main(args)