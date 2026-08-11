"""Utilities for exporting BOP19 6D-localization results."""

from collections import Counter
import json
from pathlib import Path, PurePosixPath

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment


BOP19_HEADER = "scene_id,im_id,obj_id,score,R,t,time"
SUPPORTED_BOP_DATASETS = {"ycbv", "lmo"}


def require_6d_rotation(rotation_mode):
    if rotation_mode != "6d":
        raise ValueError("BOP evaluation only supports --rotation_representation 6d")


def select_localization_queries(pred_logits, target_labels):
    """Assign one unique model query to each BOP localization target slot.

    BOP19 localization provides the object IDs and instance counts for an image.
    Repeating each object ID ``inst_count`` times gives ``target_labels``.  Query
    selection must not use the ground-truth boxes or poses, so this assignment is
    based exclusively on the model's classification confidences.
    """
    if pred_logits.ndim != 2:
        raise ValueError(
            f"Expected pred_logits with shape [num_queries, num_classes], got {tuple(pred_logits.shape)}"
        )

    target_labels = torch.as_tensor(
        target_labels,
        dtype=torch.long,
        device=pred_logits.device,
    ).reshape(-1)
    if target_labels.numel() == 0:
        empty_indices = torch.empty(0, dtype=torch.long, device=pred_logits.device)
        empty_scores = pred_logits.new_empty((0,))
        return empty_indices, target_labels, empty_scores

    num_queries, num_classes = pred_logits.shape
    if target_labels.numel() > num_queries:
        raise ValueError(
            f"BOP target requires {target_labels.numel()} poses, but the model only has {num_queries} queries"
        )
    invalid_labels = target_labels[(target_labels < 0) | (target_labels >= num_classes)]
    if invalid_labels.numel():
        raise ValueError(
            f"BOP object IDs {sorted(set(invalid_labels.tolist()))} are outside the model class range "
            f"[0, {num_classes - 1}]"
        )

    probabilities = pred_logits.sigmoid()
    # Rows are target slots and columns are model queries. The linear assignment
    # prevents one query from being exported as two different object instances.
    slot_scores = probabilities[:, target_labels].transpose(0, 1)
    slot_indices, query_indices = linear_sum_assignment(
        -slot_scores.detach().cpu().numpy()
    )
    if len(slot_indices) != target_labels.numel():
        raise RuntimeError("Could not assign a unique model query to every BOP target")

    order = np.argsort(slot_indices)
    query_indices = torch.as_tensor(
        query_indices[order],
        dtype=torch.long,
        device=pred_logits.device,
    )
    scores = probabilities[query_indices, target_labels]
    return query_indices, target_labels, scores


def parse_bop_image_ids(file_name):
    """Extract ``scene_id`` and ``im_id`` from a BOP RGB image path."""
    parts = PurePosixPath(str(file_name).replace("\\", "/")).parts
    try:
        rgb_index = len(parts) - 1 - parts[::-1].index("rgb")
    except ValueError as exc:
        raise ValueError(f"BOP image path has no 'rgb' component: {file_name!r}") from exc
    if rgb_index == 0 or rgb_index + 1 >= len(parts):
        raise ValueError(f"Cannot extract BOP scene/image IDs from {file_name!r}")
    try:
        return int(parts[rgb_index - 1]), int(PurePosixPath(parts[rgb_index + 1]).stem)
    except ValueError as exc:
        raise ValueError(f"Non-numeric BOP scene/image ID in {file_name!r}") from exc


def validate_bop19_dataset_targets(dataset):
    """Ensure dataset annotations encode exactly the official BOP19 targets."""
    dataset_root = Path(dataset.root)
    candidates = [
        dataset_root / "test_targets_bop19.json",
        dataset_root / dataset_root.name / "test_targets_bop19.json",
    ]
    targets_path = next((path for path in candidates if path.is_file()), None)
    if targets_path is None:
        raise FileNotFoundError(
            "Could not validate BOP evaluation targets; expected test_targets_bop19.json at "
            + " or ".join(str(path) for path in candidates)
        )

    with targets_path.open("r", encoding="utf-8") as target_file:
        raw_targets = json.load(target_file)
    expected = {}
    for target in raw_targets:
        image_key = (int(target["scene_id"]), int(target["im_id"]))
        expected.setdefault(image_key, Counter())[int(target["obj_id"])] += int(
            target["inst_count"]
        )

    observed = {}
    for image_id in dataset.ids:
        image_info = dataset.coco.loadImgs(image_id)[0]
        image_key = parse_bop_image_ids(image_info["file_name"])
        annotation_ids = dataset.coco.getAnnIds(imgIds=[image_id], iscrowd=False)
        annotations = dataset.coco.loadAnns(annotation_ids)
        observed[image_key] = Counter(int(ann["category_id"]) for ann in annotations)

    if observed != expected:
        missing_images = sorted(set(expected) - set(observed))
        extra_images = sorted(set(observed) - set(expected))
        mismatched_images = sorted(
            key
            for key in set(expected) & set(observed)
            if expected[key] != observed[key]
        )
        details = []
        if missing_images:
            details.append(f"missing images {missing_images[:3]}")
        if extra_images:
            details.append(f"extra images {extra_images[:3]}")
        if mismatched_images:
            key = mismatched_images[0]
            details.append(
                f"target counts differ for {key}: expected {dict(expected[key])}, "
                f"got {dict(observed[key])}"
            )
        raise ValueError(
            "Evaluation annotations do not match test_targets_bop19.json ("
            + "; ".join(details)
            + ")"
        )
    return targets_path


def make_bop_result_filename(dataset_name, method_name="lwdetr6d", split="test"):
    """Return a filename accepted by ``bop_toolkit`` result parsing."""
    if dataset_name not in SUPPORTED_BOP_DATASETS:
        raise ValueError(
            f"Unsupported BOP dataset {dataset_name!r}; expected one of {sorted(SUPPORTED_BOP_DATASETS)}"
        )
    if not method_name or "_" in method_name:
        raise ValueError("BOP method_name must be non-empty and cannot contain underscores")
    return f"{method_name}_{dataset_name}-{split}.csv"


def format_bop_result(scene_id, im_id, obj_id, score, rotation, translation_mm, runtime):
    """Format one pose estimate in the official seven-column BOP19 format."""
    rotation = np.asarray(rotation, dtype=np.float64)
    translation_mm = np.asarray(translation_mm, dtype=np.float64).reshape(-1)
    if rotation.shape != (3, 3):
        raise ValueError(f"BOP rotation must have shape (3, 3), got {rotation.shape}")
    if translation_mm.shape != (3,):
        raise ValueError(f"BOP translation must have shape (3,), got {translation_mm.shape}")
    if not np.isfinite(rotation).all() or not np.isfinite(translation_mm).all():
        raise ValueError("BOP pose contains non-finite values")
    if not np.isfinite(score) or not np.isfinite(runtime):
        raise ValueError("BOP score/runtime contains a non-finite value")

    rotation_str = " ".join(f"{value:.9g}" for value in rotation.reshape(-1))
    translation_str = " ".join(f"{value:.9g}" for value in translation_mm)
    return (
        f"{int(scene_id)},{int(im_id)},{int(obj_id)},{float(score):.9g},"
        f"{rotation_str},{translation_str},{float(runtime):.9g}"
    )
