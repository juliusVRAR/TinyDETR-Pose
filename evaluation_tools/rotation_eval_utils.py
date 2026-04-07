import csv
import os

import numpy as np


def safe_mean(values):
    if not values:
        return None
    return float(np.mean(np.asarray(values, dtype=np.float64)))


def rotation_geodesic_degrees(rot_a, rot_b):
    relative_rotation = np.matmul(rot_a, rot_b.T)
    cosine = 0.5 * (np.trace(relative_rotation) - 1.0)
    cosine = float(np.clip(cosine, -1.0, 1.0))
    return float(np.degrees(np.arccos(cosine)))


def has_rotational_symmetry(model_info=None, symmetry_flag=False):
    model_info = model_info or {}
    return bool(
        symmetry_flag
        or model_info.get('symmetries_discrete')
        or model_info.get('symmetries_continuous')
    )


def extract_discrete_symmetry_rotations(model_info):
    rotations = []
    for symmetry_transform in (model_info or {}).get('symmetries_discrete', []):
        symmetry_matrix = np.asarray(symmetry_transform, dtype=np.float64)
        if symmetry_matrix.size == 16:
            symmetry_matrix = symmetry_matrix.reshape(4, 4)
        elif symmetry_matrix.shape != (4, 4):
            continue
        rotations.append(symmetry_matrix[:3, :3])
    return rotations


def extract_continuous_symmetry_axes(model_info):
    axes = []
    for symmetry_definition in (model_info or {}).get('symmetries_continuous', []):
        axis = np.asarray(symmetry_definition.get('axis', []), dtype=np.float64)
        if axis.shape != (3,):
            continue
        axis_norm = np.linalg.norm(axis)
        if axis_norm < 1e-8:
            continue
        axes.append(axis / axis_norm)
    return axes


def continuous_symmetry_geodesic_degrees(rot_pred, rot_gt, axis):
    pred_axis = np.matmul(rot_pred, axis)
    gt_axis = np.matmul(rot_gt, axis)

    pred_axis_norm = np.linalg.norm(pred_axis)
    gt_axis_norm = np.linalg.norm(gt_axis)
    if pred_axis_norm < 1e-8 or gt_axis_norm < 1e-8:
        return 0.0

    cosine = float(np.dot(pred_axis, gt_axis) / (pred_axis_norm * gt_axis_norm))
    cosine = float(np.clip(cosine, -1.0, 1.0))
    return float(np.degrees(np.arccos(cosine)))


def symmetry_aware_geodesic_degrees(rot_pred, rot_gt, model_info=None, symmetry_flag=False):
    if not has_rotational_symmetry(model_info=model_info, symmetry_flag=symmetry_flag):
        return rotation_geodesic_degrees(rot_pred, rot_gt)

    candidate_rotations = [rot_gt]
    for symmetry_rotation in extract_discrete_symmetry_rotations(model_info):
        candidate_rotations.append(np.matmul(rot_gt, symmetry_rotation))

    continuous_axes = extract_continuous_symmetry_axes(model_info)
    best_error = None

    for candidate_rotation in candidate_rotations:
        candidate_error = rotation_geodesic_degrees(rot_pred, candidate_rotation)
        if continuous_axes:
            axis_errors = [
                continuous_symmetry_geodesic_degrees(rot_pred, candidate_rotation, axis)
                for axis in continuous_axes
            ]
            if axis_errors:
                candidate_error = min(candidate_error, min(axis_errors))

        if best_error is None or candidate_error < best_error:
            best_error = candidate_error

    return best_error if best_error is not None else rotation_geodesic_degrees(rot_pred, rot_gt)


def compute_rotation_error_summary(classes, poses_pred, poses_gt, models_info, model_symmetry):
    per_class = {}
    naive_errors_all = []
    symmetry_aware_errors_all = []
    nonsymmetric_errors_all = []

    for cls in classes:
        cls_pose_pred = poses_pred[cls]
        cls_pose_gt = poses_gt[cls]
        cls_model_info = models_info.get(cls, {})
        cls_is_symmetric = has_rotational_symmetry(
            model_info=cls_model_info,
            symmetry_flag=model_symmetry.get(cls, False),
        )

        cls_naive_errors = []
        cls_symmetry_aware_errors = []

        for pose_est, pose_gt in zip(cls_pose_pred, cls_pose_gt):
            rot_est = pose_est[:3, :3]
            rot_gt = pose_gt[:3, :3]

            naive_error = rotation_geodesic_degrees(rot_est, rot_gt)
            symmetry_aware_error = symmetry_aware_geodesic_degrees(
                rot_est,
                rot_gt,
                model_info=cls_model_info,
                symmetry_flag=model_symmetry.get(cls, False),
            )

            cls_naive_errors.append(naive_error)
            cls_symmetry_aware_errors.append(symmetry_aware_error)
            naive_errors_all.append(naive_error)
            symmetry_aware_errors_all.append(symmetry_aware_error)

            if not cls_is_symmetric:
                nonsymmetric_errors_all.append(naive_error)

        per_class[cls] = {
            'n_poses': len(cls_naive_errors),
            'is_symmetric': cls_is_symmetric,
            'naive': safe_mean(cls_naive_errors),
            'symmetry_aware': safe_mean(cls_symmetry_aware_errors),
        }

    return {
        'per_class': per_class,
        'overall': {
            'naive_all': safe_mean(naive_errors_all),
            'symmetry_aware': safe_mean(symmetry_aware_errors_all),
            'nonsymmetric_only': safe_mean(nonsymmetric_errors_all),
        },
        'counts': {
            'all': len(naive_errors_all),
            'nonsymmetric_only': len(nonsymmetric_errors_all),
        },
    }


def compute_translation_error_summary(classes, poses_pred, poses_gt, models_info, model_symmetry):
    per_class = {}
    translation_errors_all = []

    for cls in classes:
        cls_pose_pred = poses_pred[cls]
        cls_pose_gt = poses_gt[cls]
        cls_model_info = models_info.get(cls, {})
        cls_is_symmetric = has_rotational_symmetry(
            model_info=cls_model_info,
            symmetry_flag=model_symmetry.get(cls, False),
        )

        cls_translation_errors = []
        for pose_est, pose_gt in zip(cls_pose_pred, cls_pose_gt):
            t_est = pose_est[:, 3]
            t_gt = pose_gt[:, 3]
            error = float(np.sqrt(np.sum(np.square((t_est - t_gt)))))
            cls_translation_errors.append(error)
            translation_errors_all.append(error)

        per_class[cls] = {
            'n_poses': len(cls_translation_errors),
            'is_symmetric': cls_is_symmetric,
            'avg_translation_error_m': safe_mean(cls_translation_errors),
        }

    return {
        'per_class': per_class,
        'overall': {
            'mean': safe_mean(translation_errors_all),
        },
        'counts': {
            'all': len(translation_errors_all),
        },
    }


def write_paper_rotation_metrics_csv(output_dir, rotation_summary):
    csv_path = os.path.join(output_dir, 'paper_rotation_metrics.csv')
    rows = [
        {
            'metric_key': 'naive_all',
            'metric_label': 'Naive geodesic over all objects',
            'value_deg': rotation_summary['overall']['naive_all'],
            'num_objects': rotation_summary['counts']['all'],
        },
        {
            'metric_key': 'symmetry_aware',
            'metric_label': 'Symmetry-aware geodesic over all objects',
            'value_deg': rotation_summary['overall']['symmetry_aware'],
            'num_objects': rotation_summary['counts']['all'],
        },
        {
            'metric_key': 'nonsymmetric_only',
            'metric_label': 'Naive geodesic over non-symmetric objects only',
            'value_deg': rotation_summary['overall']['nonsymmetric_only'],
            'num_objects': rotation_summary['counts']['nonsymmetric_only'],
        },
    ]

    with open(csv_path, 'w', newline='') as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=['metric_key', 'metric_label', 'value_deg', 'num_objects'],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    return csv_path


def write_paper_rotation_metrics_per_class_csv(output_dir, rotation_summary):
    csv_path = os.path.join(output_dir, 'paper_rotation_metrics_per_class.csv')

    with open(csv_path, 'w', newline='') as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                'class_name',
                'n_poses',
                'is_symmetric',
                'naive_deg',
                'symmetry_aware_deg',
            ],
        )
        writer.writeheader()
        for class_name, class_metrics in rotation_summary['per_class'].items():
            writer.writerow({
                'class_name': class_name,
                'n_poses': class_metrics['n_poses'],
                'is_symmetric': class_metrics['is_symmetric'],
                'naive_deg': class_metrics['naive'],
                'symmetry_aware_deg': class_metrics['symmetry_aware'],
            })

    return csv_path


def write_paper_translation_metrics_per_class_csv(output_dir, translation_summary):
    csv_path = os.path.join(output_dir, 'paper_translation_metrics_per_class.csv')

    with open(csv_path, 'w', newline='') as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                'class_name',
                'n_poses',
                'is_symmetric',
                'avg_translation_error_m',
            ],
        )
        writer.writeheader()
        for class_name, class_metrics in translation_summary['per_class'].items():
            writer.writerow({
                'class_name': class_name,
                'n_poses': class_metrics['n_poses'],
                'is_symmetric': class_metrics['is_symmetric'],
                'avg_translation_error_m': class_metrics['avg_translation_error_m'],
            })

    return csv_path