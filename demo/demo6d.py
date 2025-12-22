# ------------------------------------------------------------------------
# PoseDETR Inference Demo (Full Script)
# ------------------------------------------------------------------------
import argparse
import random
from pathlib import Path
import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

# --- Import your model builder ---
# Ensure your PYTHONPATH is set correctly so python can find 'models'
from models import build_model
from util.misc import nested_tensor_from_tensor_list

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
    
    # --- Model Params ---
    parser.add_argument('--weights', type=str, required=True, help="Path to checkpoint")
    parser.add_argument('--device', default='cuda', help='cuda or cpu')
    
    # Architecture (Must match your training config!)
    parser.add_argument('--backbone', default='vit_tiny', type=str)
    parser.add_argument('--num_queries', default=20, type=int, help="Reduced queries for YCB")
    parser.add_argument('--enc_layers', default=6, type=int)
    parser.add_argument('--dec_layers', default=6, type=int)
    parser.add_argument('--hidden_dim', default=256, type=int)
    parser.add_argument('--nheads', default=8, type=int)
    parser.add_argument('--dim_feedforward', default=2048, type=int)
    parser.add_argument('--dropout', default=0.1, type=float)
    parser.add_argument('--two_stage', action='store_true', help="Use Two-Stage hint generation")

    # --- Input / Output ---
    parser.add_argument('--input', required=True, help="Path to input image")
    parser.add_argument('--output_dir', default='output_pose', help="Output folder")
    parser.add_argument('--confidence_threshold', type=float, default=0.5)
    parser.add_argument('--uncertainty_threshold', type=float, default=None, 
                        help="Optional: Filter objects with log_var > threshold (e.g. -1.0)")

    # --- Camera Intrinsics (Defaults to YCB-Video Real Camera) ---
    # NOTE: Change these if testing on a Webcam or different dataset!
    parser.add_argument('--fx', type=float, default=1066.778)
    parser.add_argument('--fy', type=float, default=1067.487)
    parser.add_argument('--cx', type=float, default=312.9869)
    parser.add_argument('--cy', type=float, default=241.3109)

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

# -------------------------------------------------------------------------
# Helper: Draw 3D Axes
# -------------------------------------------------------------------------
def draw_axis(img, R, t, K, dist=None, scale=0.1):
    """
    Draw 3D axis on the image. 
    Red: X, Green: Y, Blue: Z
    scale: Length of the axis line in Meters (e.g. 0.1 = 10cm)
    """
    if dist is None:
        dist = np.zeros(4)

    # Define 3D points: Origin, X-end, Y-end, Z-end
    points_3d = np.float32([
        [0, 0, 0],      # Origin
        [scale, 0, 0],  # X
        [0, scale, 0],  # Y
        [0, 0, scale]   # Z
    ])

    # Project 3D points to 2D image plane
    # cv2.projectPoints expects rotation vector (Rodrigues), not matrix
    r_vec, _ = cv2.Rodrigues(R) 
    
    points_2d, _ = cv2.projectPoints(points_3d, r_vec, t, K, dist)
    points_2d = points_2d.astype(int).reshape(-1, 2)

    origin = tuple(points_2d[0])
    pt_x = tuple(points_2d[1])
    pt_y = tuple(points_2d[2])
    pt_z = tuple(points_2d[3])

    # Draw Lines (BGR Colors in OpenCV)
    # Origin -> X (Red)
    img = cv2.line(img, origin, pt_x, (0, 0, 255), 3)  
    # Origin -> Y (Green)
    img = cv2.line(img, origin, pt_y, (0, 255, 0), 3)  
    # Origin -> Z (Blue)
    img = cv2.line(img, origin, pt_z, (255, 0, 0), 3)  
    
    return img

# -------------------------------------------------------------------------
# Main Logic
# -------------------------------------------------------------------------
def main(args):
    # 1. Setup
    device = torch.device(args.device)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # 2. Build Model
    # Note: args.num_queries (20) will override the default here
    model, _, _ = build_model(args)
    model.to(device)
    model.eval()

    # 3. Load Weights
    print(f"Loading weights from {args.weights}...")
    checkpoint = torch.load(args.weights, map_location='cpu')
    state_dict = checkpoint['model'] if 'model' in checkpoint else checkpoint
    
    # Strict=False is CRITICAL because your checkpoint has Aux Heads (training only)
    # that are not used in inference.
    model.load_state_dict(state_dict, strict=False) 

    # 4. Prepare Input & Intrinsics
    img_bgr = cv2.imread(args.input)
    if img_bgr is None:
        raise ValueError(f"Could not load image: {args.input}")
    
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    
    # Define Original Intrinsics (K) from args
    K_orig = np.array([
        [args.fx, 0, args.cx],
        [0, args.fy, args.cy],
        [0, 0, 1]
    ], dtype=np.float32)
    
    # Resize & Pad (Updating K)
    img_padded, K_new = resize_pad_with_intrinsics(img_rgb, K_orig)
    
    # Convert to Tensor (Normalize with ImageNet stats)
    transform_norm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    tensor_img = transform_norm(img_padded).to(device)
    
    # Create Nested Tensor (Standard DETR input)
    samples = nested_tensor_from_tensor_list([tensor_img])
    
    # 5. Prepare Dummy Target (To pass intrinsics to Model)
    # The model needs 'intrinsics' in the targets dict to perform Back-Projection
    # inside the forward() pass.
    K_tensor = torch.from_numpy(K_new).to(device)
    
    # Important: Create a list of dicts (one per image in batch)
    dummy_targets = [{'intrinsics': K_tensor}]

    # 6. Inference
    print(f"Running Inference on {args.device}...")
    with torch.no_grad():
        outputs = model(samples, dummy_targets)

    # 7. Extract Predictions
    # outputs['pred_logits']:      [B, Q, NumClasses]
    # outputs['pred_rotations']:   [B, Q, 3, 3]
    # outputs['pred_translation']: [B, Q, 3] (Meters)
    # outputs['pred_z_log_var']:   [B, Q]    (Uncertainty)
    
    pred_logits = outputs['pred_logits'][0]
    pred_rot = outputs['pred_rotations'][0].cpu().numpy()
    pred_trans = outputs['pred_translation'][0].cpu().numpy()
    pred_log_var = outputs['pred_z_log_var'][0].cpu().numpy()
    
    # Softmax for probabilities
    probs = pred_logits.softmax(-1)[..., :-1] # Exclude background
    scores, labels = probs.max(-1)
    
    # 8. Visualization Loop
    # Prepare canvas (use padded image to align with K_new)
    # Convert back to BGR for OpenCV
    canvas = cv2.cvtColor(img_padded, cv2.COLOR_RGB2BGR)
    
    print("\n--- Detections ---")
    found_obj = False
    
    for i in range(pred_logits.shape[0]):
        score = scores[i].item()
        label = labels[i].item()
        uncertainty = pred_log_var[i].item() # Laplacian Log-Variance
        
        # Filter by Confidence Score
        if score > args.confidence_threshold:
            
            # Optional: Filter by Uncertainty
            if args.uncertainty_threshold is not None:
                if uncertainty > args.uncertainty_threshold:
                    print(f"Skipping Object {label} (Score: {score:.2f}) due to High Uncertainty: {uncertainty:.2f}")
                    continue

            found_obj = True
            obj_name = YCB_CLASSES[label] if label < len(YCB_CLASSES) else str(label)
            
            # Console Log
            print(f"Found {obj_name:<20} | Score: {score:.2f} | Dist: {pred_trans[i, 2]:.2f}m | Unc (σ): {uncertainty:.2f}")
            
            # Draw Axis
            # R=pred_rot[i], t=pred_trans[i], K=K_new
            draw_axis(canvas, pred_rot[i], pred_trans[i], K_new, scale=0.08) # 8cm axis length
            
            # Draw Text Label
            # Project Center to find where to put text
            center_3d = pred_trans[i]
            # Project 3D point to 2D
            center_2d, _ = cv2.projectPoints(center_3d.reshape(1,3), np.zeros(3), np.zeros(3), K_new, None)
            uv = center_2d.reshape(-1).astype(int)
            
            label_text = f"{obj_name} {score:.2f}"
            cv2.putText(canvas, label_text, (uv[0], uv[1]-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            cv2.putText(canvas, f"Unc: {uncertainty:.1f}", (uv[0], uv[1]+15), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

    if not found_obj:
        print("No objects found above threshold.")

    # 9. Save
    out_path = Path(args.output_dir) / "pose_result.jpg"
    cv2.imwrite(str(out_path), canvas)
    print(f"\nSaved visualization to {out_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser('PoseDETR Inference Script', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)