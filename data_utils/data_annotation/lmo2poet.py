# ------------------------------------------------------------------------
# PoET: Pose Estimation Transformer for Single-View, Multi-Object 6D Pose Estimation
# Copyright (c) 2022 Thomas Jantos (thomas.jantos@aau.at), University of Klagenfurt - Control of Networked Systems (CNS). All Rights Reserved.
# Licensed under the BSD-2-Clause-License with no commercial use [see LICENSE for details]
# ------------------------------------------------------------------------

import argparse
import json
import os


def parse_args():
    parser = argparse.ArgumentParser(description="Convert BOP LM-O annotations to PoET COCO format.")
    parser.add_argument(
        "--dataset-root",
        default="/lmo",
        help="LM-O root containing train (or train_synt), train_pbr, test, and models.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Annotation output directory (default: <dataset-root>/annotations).",
    )
    parser.add_argument(
        "--split",
        choices=("train", "test"),
        default="train",
        help="Convert the training sets or the test set.",
    )
    parser.add_argument(
        "--min-visibility",
        type=float,
        default=0.05,
        help="Discard training instances with visib_fract below this value.",
    )
    parser.add_argument(
        "--max-images-per-source",
        type=int,
        default=None,
        help="Write at most this many images from each source (useful for smoke tests).",
    )
    parser.add_argument(
        "--test-targets",
        default=None,
        help=(
            "Path to test_targets_bop19.json. For test conversion, the converter "
            "auto-detects it under the dataset root when omitted."
        ),
    )
    return parser.parse_args()


args = parse_args()
base_path = os.path.abspath(args.dataset_root)
output_base_path = args.output_dir or os.path.join(base_path, "annotations")
os.makedirs(output_base_path, exist_ok=True)

if args.max_images_per_source is not None and args.max_images_per_source <= 0:
    raise ValueError("--max-images-per-source must be positive")

if args.split == "train":
    synthetic_train_path = "train_synt" if os.path.isdir(os.path.join(base_path, "train_synt")) else "train"
    data_paths = [synthetic_train_path, "train_pbr"]
    img_types = ["synt", "pbr"]
    # train_synt.json is written after the first split; train.json contains both.
    annotation_paths = ["train_synt.json", "train.json"]
else:
    data_paths = ["test"]
    img_types = ["real"]
    annotation_paths = ["test.json"]


def load_test_targets():
    if args.split != "test":
        return None

    if args.test_targets is not None:
        targets_path = args.test_targets
        if not os.path.isabs(targets_path):
            targets_path = os.path.join(base_path, targets_path)
        candidates = [targets_path]
    else:
        candidates = [
            os.path.join(base_path, "test_targets_bop19.json"),
            # Handles archives extracted as <dataset-root>/lmo/test_targets_bop19.json.
            os.path.join(base_path, "lmo", "test_targets_bop19.json"),
        ]

    targets_path = next((path for path in candidates if os.path.isfile(path)), None)
    if targets_path is None:
        raise FileNotFoundError(
            "Could not find test_targets_bop19.json. Pass it explicitly with --test-targets. "
            f"Checked: {candidates}"
        )

    with open(targets_path, "r") as target_file:
        targets = json.load(target_file)

    targets_by_image = {}
    for target in targets:
        image_key = (int(target["scene_id"]), int(target["im_id"]))
        targets_by_image.setdefault(image_key, {})[int(target["obj_id"])] = int(target["inst_count"])

    print(f"Using BOP test targets from: {targets_path}")
    return targets_by_image


test_targets_by_image = load_test_targets()

categories = [
    {'supercategory': 'background', 'id': 0, 'name': 'background'},
    {'supercategory': 'ape', 'id': 1, 'name': 'ape'},
    {'supercategory': 'can', 'id': 5, 'name': 'can'},
    {'supercategory': 'cat', 'id': 6, 'name': 'cat'},
    {'supercategory': 'driller', 'id': 8, 'name': 'driller'},
    {'supercategory': 'duck', 'id': 9, 'name': 'duck'},
    {'supercategory': 'eggbox', 'id': 10, 'name': 'eggbox'},
    {'supercategory': 'glue', 'id': 11, 'name': 'glue'},
    {'supercategory': 'holepuncher', 'id': 12, 'name': 'holepuncher'},
]

cls_ids = [1, 5, 6, 8, 9, 10, 11, 12]

annotations = {'images': [],
               'categories': categories,
               'annotations': []}
image_id = 0
annotation_id = 0
annotations_removed = 0
for data_path, ann_path, img_type in zip(data_paths, annotation_paths, img_types):
    print("Annotating: {}".format(data_path))
    source_image_count = 0
    source_limit_reached = False
    # Get List of all subdirectories
    split_path = os.path.join(base_path, data_path)
    image_dirs = [d.name for d in os.scandir(split_path) if d.is_dir()]
    image_dirs.sort()

    for img_dir in image_dirs:
        if source_limit_reached:
            break
        print("Image Directory: {}".format(img_dir))
        img_dir_path = os.path.join(split_path, img_dir)
        rgb_path = os.path.join(img_dir_path, "rgb")
        img_names = [
            img for img in os.listdir(rgb_path)
            if os.path.splitext(img)[1].lower() in (".png", ".jpg", ".jpeg")
        ]
        img_names.sort()
        with open(os.path.join(img_dir_path, "scene_gt_info.json"), 'r') as f:
            bbox_annotations = json.load(f)
        with open(os.path.join(img_dir_path, "scene_gt.json"), 'r') as f:
            pose_annotations = json.load(f)
        with open(os.path.join(img_dir_path, "scene_camera.json"), 'r') as f:
            camera_annotations = json.load(f)

        # Iterate over all images and annotations and create dict entries
        for img_name in img_names:
            if args.max_images_per_source is not None and source_image_count >= args.max_images_per_source:
                source_limit_reached = True
                break

            image_key = str(int(os.path.splitext(img_name)[0]))
            if not all(
                image_key in source
                for source in (bbox_annotations, pose_annotations, camera_annotations)
            ):
                raise KeyError(
                    f"Missing BOP metadata for {data_path}/{img_dir}/rgb/{img_name} "
                    f"(image key {image_key})"
                )

            img_annotation_counter = 0
            file_name = "/".join((data_path, img_dir, "rgb", img_name))
            bbox_data = bbox_annotations[image_key]
            pose_data = pose_annotations[image_key]
            camera_data = camera_annotations[image_key]

            selected_test_indices = None
            if test_targets_by_image is not None:
                target_counts = test_targets_by_image.get((int(img_dir), int(image_key)))
                if target_counts is None:
                    continue

                # A BOP target specifies the number of instances per object rather
                # than GT indices. Pick the requested number of most-visible GT
                # instances. LM-O has at most one instance per target object.
                selected_test_indices = set()
                for target_obj_id, inst_count in target_counts.items():
                    candidates = [
                        index for index, pose in enumerate(pose_data)
                        if int(pose["obj_id"]) == target_obj_id
                    ]
                    if len(candidates) < inst_count:
                        raise ValueError(
                            f"BOP target requests {inst_count} instance(s) of object "
                            f"{target_obj_id} in scene {int(img_dir)}, image {int(image_key)}, "
                            f"but only {len(candidates)} GT instance(s) were found"
                        )
                    candidates.sort(
                        key=lambda index: bbox_data[index]["visib_fract"],
                        reverse=True,
                    )
                    selected_test_indices.update(candidates[:inst_count])

            if len(bbox_data) != len(pose_data):
                raise ValueError(
                    f"Mismatched GT and GT-info counts for {file_name}: "
                    f"{len(pose_data)} poses vs {len(bbox_data)} boxes"
                )

            for gt_index, (bbox, pose) in enumerate(zip(bbox_data, pose_data)):
                if selected_test_indices is not None and gt_index not in selected_test_indices:
                    continue
                # Check if object is in LM-O
                if pose['obj_id'] not in cls_ids:
                    continue
                obj_id_original = int(pose['obj_id'])
                # If percentage of visible pixels is close to 0 --> skip
                if args.split == "train" and bbox['visib_fract'] < args.min_visibility:
                    annotations_removed += 1
                    continue
                # Check if bbox starts / ends outside of image --> set to 0 or img boundary simply
                x1 = bbox['bbox_obj'][0]
                y1 = bbox['bbox_obj'][1]
                x2 = bbox['bbox_obj'][0] + bbox['bbox_obj'][2]
                y2 = bbox['bbox_obj'][1] + bbox['bbox_obj'][3]

                if x1 < 0:
                    # Adjust upper left and width
                    bbox['bbox_obj'][2] = bbox['bbox_obj'][2] + bbox['bbox_obj'][0]
                    bbox['bbox_obj'][0] = 0

                if y1 < 0:
                    # Adjust upper left and height
                    bbox['bbox_obj'][3] = bbox['bbox_obj'][3] + bbox['bbox_obj'][1]
                    bbox['bbox_obj'][1] = 0

                if x2 >= 640:
                    # Adjust width
                    bbox['bbox_obj'][2] = 640 - bbox['bbox_obj'][0] - 1

                if y2 >= 480:
                    # Adjust height
                    bbox['bbox_obj'][3] = 480 - bbox['bbox_obj'][1] - 1

                obj_annotation = {
                    'id': annotation_id,
                    'image_id': image_id,
                    'relative_pose': {
                        'position': [t / 1000.0 for t in pose['cam_t_m2c']],
                        'rotation': pose['cam_R_m2c']
                    },
                    'bbox': bbox['bbox_obj'],
                    'bbox_info': bbox,
                    'area': bbox['bbox_obj'][2] * bbox['bbox_obj'][3],
                    'iscrowd': 0,
                    'category_id': obj_id_original
                }
                annotations['annotations'].append(obj_annotation)
                img_annotation_counter += 1
                annotation_id += 1
            # Check if there are annotations for the image, otherwise skip
            if img_annotation_counter == 0:
                print("Image skipped! No annotations valid!")
                continue
            img_annotation = {
                'file_name': file_name,
                'id': image_id,
                'width': 640,
                'height': 480,
                'intrinsics': camera_data['cam_K'],
                'type': img_type
            }
            annotations['images'].append(img_annotation)
            image_id += 1
            source_image_count += 1

    print("Annotations: {}".format(annotation_id))
    print("Annotations Removed: {}".format(annotations_removed))
    output_path = os.path.join(output_base_path, ann_path)
    with open(output_path, 'w') as out_file:
        json.dump(annotations, out_file)
    print("Wrote annotations to: {}".format(output_path))
