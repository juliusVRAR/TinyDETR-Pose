# ------------------------------------------------------------------------
# PoET: Pose Estimation Transformer for Single-View, Multi-Object 6D Pose Estimation
# Copyright (c) 2022 Thomas Jantos (thomas.jantos@aau.at), University of Klagenfurt - Control of Networked Systems (CNS). All Rights Reserved.
# Licensed under the BSD-2-Clause-License with no commercial use [see LICENSE for details]
# ------------------------------------------------------------------------

import numpy as np
import json
import evaluation_tools.model_tools as model_tools
from evaluation_tools.pose_evaluator import PoseEvaluator
from evaluation_tools.pose_evaluator_lmo import PoseEvaluatorLMO
from evaluation_tools.better_pose_eval import PoseEvaluator as BetterPoseEvaluator


# Functions to initialize the PoseEvaluator module
def load_classes(path):
    """
    Load the class information from a json file. This file contains a mapping between class ID and class name.
    """
    with open(path, 'r') as f:
        classes = json.load(f)
    return classes


def build_class_id_to_name(classes):
    """Convert the JSON class mapping to integer object IDs."""
    if not isinstance(classes, dict):
        raise TypeError("Class information must be a mapping from object ID to class name.")

    class_id_to_name = {}
    for class_id, class_name in classes.items():
        try:
            class_id = int(class_id)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid object ID in class information: {class_id!r}") from exc

        if class_id in class_id_to_name:
            raise ValueError(f"Duplicate object ID in class information: {class_id}")
        class_id_to_name[class_id] = class_name

    return class_id_to_name


def resolve_pose_class_name(pose_evaluator, class_id):
    """Resolve a dataset object ID without assuming that IDs are contiguous."""
    class_id = int(class_id)
    class_id_to_name = getattr(pose_evaluator, "class_id_to_name", None)
    if class_id_to_name is not None:
        try:
            return class_id_to_name[class_id]
        except KeyError as exc:
            available_ids = sorted(class_id_to_name)
            raise KeyError(
                f"Object ID {class_id} is missing from the pose evaluator class mapping. "
                f"Available IDs: {available_ids}"
            ) from exc

    # Preserve compatibility for evaluators created directly with contiguous
    # one-based class IDs instead of through build_pose_evaluator().
    class_index = class_id - 1
    if 0 <= class_index < len(pose_evaluator.classes):
        return pose_evaluator.classes[class_index]
    raise KeyError(
        f"Object ID {class_id} cannot be resolved because the pose evaluator "
        "has no class_id_to_name mapping."
    )


def load_model_info(points):
    """
    Load information about the 3D model from the BOP files
    """
    infos = {}
    extents = 2 * np.max(np.absolute(points), axis=0)
    infos['diameter'] = np.sqrt(np.sum(extents * extents))
    infos['min_x'], infos['min_y'], infos['min_z'] = np.min(points, axis=0)
    infos['max_x'], infos['max_y'], infos['max_z'] = np.min(points, axis=0)
    return infos


def load_models(path, classes):
    """
    Load the 3D model point cloud and store it in a dict.
    """

    with open(path + 'models_info.json', 'r') as f:
        models_info_data = json.load(f)

    models = {}
    models_info = {}

    for cls in classes:
        model_class = classes[cls]
        model_file = "obj_" + f'{int(cls):06d}' + ".ply"
        model = model_tools.load_ply(path + model_file)
        if not 'texture_file' in model: 
            print(f"{model_file} has no texture")
        models[model_class] = model
        models[model_class]['pts'] = models[model_class]['pts'] / 1000  # Scale the model to meters.
        models_info[model_class] = models_info_data[cls]
    return models, models_info


def load_model_symmetry(path, classes):
    """
    Load information whether objects are symmetric or not.
    """
    model_symmetry = {}

    with open(path, 'r') as f:
        symmetry_dict = json.load(f)

    for cls in classes:
        model_cls = classes[cls]
        model_symmetry[model_cls] = symmetry_dict[model_cls]

    return model_symmetry


def build_pose_evaluator(args):
    """
    Function to build the Pose Evaluator by loading the 3D point clouds and additional information.
    """
    classes_path = args.dataset_path + args.class_info
    classes = load_classes(classes_path)
    class_id_to_name = build_class_id_to_name(classes)

    models_path = args.dataset_path + args.models
    models, models_info = load_models(models_path, classes)

    symmetries_path = args.dataset_path + args.model_symmetry
    model_symmetry = load_model_symmetry(symmetries_path, classes)
    classes = [classes[k] for k in classes]
    if args.dataset_file == 'ycbv':
        evaluator = PoseEvaluator(models, classes, models_info, model_symmetry)
    elif args.dataset_file == 'lmo':
        evaluator = PoseEvaluatorLMO(models, classes, models_info, model_symmetry)
    else:
        raise ValueError("Unknown dataset.")
    evaluator.class_id_to_name = class_id_to_name
    return evaluator

def build_better_pose_evaluator(args):
    """
    Function to build the Pose Evaluator by loading the 3D point clouds and additional information.
    """
    classes_path = args.dataset_path + args.class_info
    classes = load_classes(classes_path)
    class_id_to_name = build_class_id_to_name(classes)

    models_path = args.dataset_path + args.models
    models, models_info = load_models(models_path, classes)

    symmetries_path = args.dataset_path + args.model_symmetry
    model_symmetry = load_model_symmetry(symmetries_path, classes)
    classes = [classes[k] for k in classes]
    if args.dataset_file == 'ycbv':
        evaluator = PoseEvaluator(models, classes, models_info, model_symmetry)
    elif args.dataset_file == 'lmo':
        evaluator = PoseEvaluatorLMO(models, classes, models_info, model_symmetry)
    else:
        raise ValueError("Unknown dataset.")
    evaluator.class_id_to_name = class_id_to_name
    return evaluator
