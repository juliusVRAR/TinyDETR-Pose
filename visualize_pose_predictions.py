#!/usr/bin/env python3
"""
Run LW-DETR 6D inference on a dataset split and render predicted and ground-truth
6D poses together with predicted 2D bounding boxes.
"""
from pathlib import Path
import argparse
import cv2
import numpy as np
import torch

from main import get_args_parser
from models import build_model
from data_utils import build_dataset
from util import box_ops
import util.misc as utils
from util.utils import clean_state_dict
from util.visualize_object_pose import YCBVVisualizer


GT_POSE_COLOR = (0, 255, 0)      # green in OpenCV BGR order
PRED_POSE_COLOR = (0, 0, 255)

def build_vis_parser():
    parser = get_args_parser()
    # Re-enable -h/--help because get_args_parser sets add_help=False
    parser.add_argument('-h', '--help', action='help', default=argparse.SUPPRESS,
                        help='Show this help message and exit')
    parser.add_argument('--checkpoint', required=True, help='Path to the trained checkpoint (.pth)')
    parser.add_argument('--split', default='test', help='Dataset split to visualize (e.g., test, val, keyframes)')
    parser.add_argument('--num_images', type=int, default=16, help='Number of images to render (-1 for all)')
    parser.add_argument('--score_threshold', type=float, default=0.4, help='Confidence threshold for drawing')
    parser.add_argument('--vis_output', default='output/vis_test', help='Directory to save visualizations')
    parser.add_argument('--show_mesh', action='store_true', help='Project sampled CAD vertices in addition to 3D bbox')
    parser.add_argument('--mesh_points', type=int, default=5000, help='Vertices to sample when --show_mesh is set')
    parser.add_argument('--no_ground_truth', action='store_true',
                        help='Disable ground-truth 6D pose overlays')
    parser.add_argument('--exclude_objects', nargs='+', default=[], metavar='ID_OR_NAME',
                        help='Object IDs or class names to omit from all overlays')
    return parser


def get_class_map(dataset):
    """
    Returns a {class_id: class_name} mapping if available, otherwise None.
    """
    if hasattr(dataset, "_class_id_to_name") and dataset._class_id_to_name:
        return {int(k): v for k, v in dataset._class_id_to_name.items()}
    if hasattr(dataset, "models_info") and dataset.models_info:
        mapping = {}
        for k, v in dataset.models_info.items():
            try:
                mapping[int(k)] = v.get("name", str(k))
            except Exception:
                continue
        if mapping:
            return mapping
    return None


def resolve_excluded_objects(values, class_map):
    """Resolve numeric IDs and case-insensitive class names to class IDs."""
    name_to_id = {
        str(name).casefold(): class_id
        for class_id, name in (class_map or {}).items()
    }
    excluded_ids = set()
    unknown = []
    for value in values:
        try:
            excluded_ids.add(int(value))
        except ValueError:
            class_id = name_to_id.get(value.casefold())
            if class_id is None:
                unknown.append(value)
            else:
                excluded_ids.add(class_id)

    if unknown:
        available_names = ", ".join(sorted(map(str, (class_map or {}).values())))
        detail = f" Available names: {available_names}." if available_names else ""
        raise ValueError(f"Unknown object(s): {', '.join(unknown)}.{detail}")
    return excluded_ids


def filter_pose_annotations(annotations, excluded_ids):
    """Filter aligned pose annotation tensors by object ID."""
    labels = annotations["labels"]
    keep = torch.tensor(
        [int(label) not in excluded_ids for label in labels],
        dtype=torch.bool,
        device=labels.device,
    )
    return {key: value[keep] for key, value in annotations.items()}


def select_topk_pose(outputs, target_sizes, num_select=100, score_threshold=0.4):
    """
    Mirror PostProcess to keep pose heads aligned with the chosen queries.
    Returns a list (len=batch) of detections with box/score/label/rotation/translation.
    """
    logits = outputs['pred_logits'].sigmoid()
    topk_values, topk_indexes = torch.topk(logits.view(logits.shape[0], -1), num_select, dim=1)
    labels = topk_indexes % logits.shape[2]
    query_indices = topk_indexes // logits.shape[2]

    boxes = box_ops.box_cxcywh_to_xyxy(outputs['pred_boxes'])
    boxes = torch.gather(boxes, 1, query_indices.unsqueeze(-1).repeat(1, 1, 4))

    img_h, img_w = target_sizes.unbind(1)
    scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
    boxes = boxes * scale_fct[:, None, :]

    batch_dets = []
    for b in range(logits.shape[0]):
        dets = []
        for score, label, box, q_idx in zip(topk_values[b], labels[b], boxes[b], query_indices[b]):
            if score < score_threshold:
                continue
            dets.append({
                "score": score.item(),
                "label": int(label.item()),
                "box": box.detach().cpu().numpy(),
                "rotation": outputs['pred_rotations'][b, q_idx].detach().cpu().numpy(),
                "translation": outputs['pred_translations'][b, q_idx].detach().cpu().numpy(),
            })
        batch_dets.append(dets)
    return batch_dets


def get_object_color(class_id):
    """Return the visualizer's stable BGR color for an object class."""
    rng = np.random.RandomState(int(class_id))
    return tuple(rng.randint(100, 255, 3).tolist())
    #return tuple([255,0,0])

def draw_2d_bboxes(img_bgr, detections, class_map):
    img = img_bgr.copy()
    for det in detections:
        x1, y1, x2, y2 = det["box"]
        #color = get_object_color(det["label"])
        color = PRED_POSE_COLOR
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        label_text = class_map.get(det["label"], str(det["label"])) if class_map else str(det["label"])
        cv2.putText(img, f"2D {label_text} {det['score']:.2f}", (int(x1), max(0, int(y1) - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    return img


def draw_pose_legend(img_bgr, show_ground_truth=True):
    """Add a compact legend for the projected 6D-pose sources."""
    entries = []
    if show_ground_truth:
        entries.append(("Ground truth 6D", GT_POSE_COLOR))
    entries.append(("Predicted 6D: object color", PRED_POSE_COLOR))
    legend_bottom = 14 + 22 * len(entries)
    overlay = img_bgr.copy()
    cv2.rectangle(overlay, (8, 8), (250, legend_bottom), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.65, img_bgr, 0.35, 0, img_bgr)
    for index, (text, color) in enumerate(entries):
        y = 26 + index * 22
        if color is None:
            for offset, class_id in zip((0, 8, 16), (1, 2, 3)):
                cv2.line(img_bgr, (16 + offset, y - 4), (23 + offset, y - 4),
                         get_object_color(class_id), 3, cv2.LINE_AA)
            text_color = (255, 255, 255)
        else:
            cv2.line(img_bgr, (16, y - 4), (38, y - 4), color, 3, cv2.LINE_AA)
            text_color = color
        cv2.putText(img_bgr, text, (46, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, text_color, 1, cv2.LINE_AA)
    return img_bgr


def main():
    parser = build_vis_parser()
    args = parser.parse_args()
    args.eval_set = args.split  # keep build_dataset happy

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    output_dir = Path(args.vis_output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Model
    model, _, _, _ = build_model(args)
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    model.load_state_dict(clean_state_dict(checkpoint['model']), strict=False)
    model.to(device)
    model.eval()

    # Data
    dataset = build_dataset(args.eval_set, args)
    class_map = get_class_map(dataset)
    try:
        excluded_ids = resolve_excluded_objects(args.exclude_objects, class_map)
    except ValueError as exc:
        parser.error(str(exc))
    sampler = torch.utils.data.RandomSampler(dataset)
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, sampler=sampler,
        drop_last=False, collate_fn=utils.collate_fn, num_workers=args.num_workers
    )

    visualizer = YCBVVisualizer(dataset.cad_model_path)

    max_images = len(dataset) if args.num_images < 0 else args.num_images
    processed = 0
    with torch.no_grad():
        for samples, targets in data_loader:
            samples = samples.to(device)
            outputs = model(samples)
            target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0).to(device)
            batch_dets = select_topk_pose(outputs, target_sizes, num_select=args.num_select,
                                          score_threshold=args.score_threshold)
            Ks = samples.meta.get("K")

            for dets, tgt, k_mat in zip(batch_dets, targets, Ks):
                if processed >= max_images:
                    break

                dets = [det for det in dets if det["label"] not in excluded_ids]

                img_info = dataset.coco.loadImgs(int(tgt["image_id"])).pop()
                img_path = Path(dataset.root) / img_info['file_name']
                img_bgr = cv2.imread(str(img_path))
                if img_bgr is None:
                    print(f"[warn] Could not read {img_path}, skipping.")
                    processed += 1
                    continue

                img_bgr = draw_2d_bboxes(img_bgr, dets, class_map)

                if not args.no_ground_truth:
                    # Draw GT first with a wider line. Predictions are then overlaid
                    # with a narrower line, leaving a green outline when they agree.
                    gt_ann = {
                        "labels": tgt["labels"],
                        "relative_position": tgt["relative_position"],
                        "relative_rotation": tgt["relative_rotation"],
                    }
                    gt_ann = filter_pose_annotations(gt_ann, excluded_ids)
                    img_bgr = visualizer.visualize_single_image(
                        img_bgr,
                        gt_ann,
                        K=k_mat.cpu().numpy(),
                        show_mesh=args.show_mesh,
                        sample_points=args.mesh_points,
                        conf_threshold=0.0,
                        color=GT_POSE_COLOR,
                        line_thickness=4,
                        label_prefix="GT")

                if dets:
                    ann = {
                        "labels": torch.tensor([d["label"] for d in dets]),
                        "relative_position": torch.tensor([d["translation"] for d in dets]),
                        "relative_rotation": torch.tensor([d["rotation"] for d in dets]),
                    }
                    img_bgr = visualizer.visualize_single_image(
                        img_bgr,
                        ann,
                        K=k_mat.cpu().numpy(),
                        show_mesh=args.show_mesh,
                        sample_points=args.mesh_points,
                        conf_threshold=args.score_threshold,
                        scores=[d["score"] for d in dets],
                        color=PRED_POSE_COLOR,
                        line_thickness=2,
                        label_prefix="Pred")

                img_bgr = draw_pose_legend(img_bgr, show_ground_truth=not args.no_ground_truth)

                out_file = output_dir / f"{args.split}_{int(tgt['image_id'])}.png"
                cv2.imwrite(str(out_file), img_bgr)
                processed += 1

            if processed >= max_images:
                break

    print(f"Saved {processed} visualizations to {output_dir}")


if __name__ == "__main__":
    main()
