import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from evaluation_tools.bop import (
    BOP19_HEADER,
    format_bop_result,
    make_bop_result_filename,
    parse_bop_image_ids,
    require_6d_rotation,
    select_localization_queries,
    validate_bop19_dataset_targets,
)


class FakeCoco:
    def __init__(self, annotations):
        self.annotations = annotations

    def loadImgs(self, image_id):
        return [{"file_name": "test/000002/rgb/000003.png"}]

    def getAnnIds(self, imgIds, iscrowd=False):
        return list(range(len(self.annotations)))

    def loadAnns(self, annotation_ids):
        return [self.annotations[index] for index in annotation_ids]


class BOPExportTest(unittest.TestCase):
    def test_localization_assignment_uses_sparse_target_ids_and_unique_queries(self):
        logits = torch.full((4, 13), -10.0)
        logits[0, 1] = 10.0
        logits[0, 5] = 9.0
        logits[1, 5] = 8.0
        logits[2, 5] = 7.0

        query_indices, object_ids, scores = select_localization_queries(
            logits,
            torch.tensor([1, 5, 5]),
        )

        self.assertEqual(object_ids.tolist(), [1, 5, 5])
        self.assertEqual(query_indices.tolist(), [0, 1, 2])
        self.assertEqual(len(set(query_indices.tolist())), 3)
        self.assertTrue(torch.all(scores > 0.99))

    def test_localization_assignment_rejects_unknown_object_id(self):
        with self.assertRaisesRegex(ValueError, "outside the model class range"):
            select_localization_queries(torch.zeros(2, 12), [12])

    def test_result_filenames_are_dataset_specific_and_toolkit_compatible(self):
        self.assertEqual(make_bop_result_filename("ycbv"), "lwdetr6d_ycbv-test.csv")
        self.assertEqual(make_bop_result_filename("lmo"), "lwdetr6d_lmo-test.csv")

    def test_image_ids_are_parsed_from_both_dataset_layouts(self):
        self.assertEqual(
            parse_bop_image_ids("test_bop/000048/rgb/000123.png"),
            (48, 123),
        )
        self.assertEqual(
            parse_bop_image_ids("test/000002/rgb/000003.png"),
            (2, 3),
        )

    def test_result_row_has_official_columns_and_metric_translation(self):
        row = format_bop_result(
            2,
            3,
            12,
            0.75,
            np.eye(3),
            np.array([100.0, -20.0, 850.0]),
            0.01,
        )
        columns = row.split(",")
        self.assertEqual(BOP19_HEADER.split(","), [
            "scene_id", "im_id", "obj_id", "score", "R", "t", "time"
        ])
        self.assertEqual(len(columns), 7)
        self.assertEqual(columns[:4], ["2", "3", "12", "0.75"])
        self.assertEqual(len(columns[4].split()), 9)
        self.assertEqual(columns[5], "100 -20 850")

    def test_bop_export_rejects_sarr(self):
        require_6d_rotation("6d")
        with self.assertRaisesRegex(ValueError, "only supports"):
            require_6d_rotation("sarr")

    def test_dataset_annotations_must_match_official_targets(self):
        with tempfile.TemporaryDirectory() as root:
            targets = [
                {"scene_id": 2, "im_id": 3, "obj_id": 5, "inst_count": 1},
                {"scene_id": 2, "im_id": 3, "obj_id": 12, "inst_count": 1},
            ]
            (Path(root) / "test_targets_bop19.json").write_text(json.dumps(targets))
            dataset = type(
                "FakeDataset",
                (),
                {
                    "root": root,
                    "ids": [0],
                    "coco": FakeCoco([{"category_id": 5}, {"category_id": 12}]),
                },
            )()
            self.assertEqual(
                validate_bop19_dataset_targets(dataset),
                Path(root) / "test_targets_bop19.json",
            )

            dataset.coco = FakeCoco([{"category_id": 5}])
            with self.assertRaisesRegex(ValueError, "target counts differ"):
                validate_bop19_dataset_targets(dataset)


if __name__ == "__main__":
    unittest.main()
