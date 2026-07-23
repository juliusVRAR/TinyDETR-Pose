import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from evaluation_tools.pose_evaluator import PoseEvaluator


def make_pose(rotation=None, translation=None):
    pose = np.zeros((3, 4), dtype=np.float32)
    pose[:, :3] = np.eye(3, dtype=np.float32) if rotation is None else rotation
    if translation is not None:
        pose[:, 3] = translation
    return pose


class PoseEvaluator01dTest(unittest.TestCase):
    def setUp(self):
        self.classes = ["asymmetric", "empty", "symmetric"]
        self.models = {
            "asymmetric": {"pts": np.array([[0.0, 0.0, 0.0]], dtype=np.float32)},
            "empty": {"pts": np.array([[0.0, 0.0, 0.0]], dtype=np.float32)},
            "symmetric": {
                "pts": np.array(
                    [[-0.05, 0.0, 0.0], [0.05, 0.0, 0.0]],
                    dtype=np.float32,
                )
            },
        }
        self.models_info = {
            cls_name: {"diameter": 100.0}
            for cls_name in self.classes
        }
        self.model_symmetry = {
            "asymmetric": False,
            "empty": False,
            "symmetric": True,
        }
        self.evaluator = PoseEvaluator(
            self.models,
            self.classes,
            self.models_info,
            self.model_symmetry,
        )

    def test_adds_01d_uses_diameter_symmetry_and_macro_average(self):
        identity = make_pose()
        self.evaluator.poses_gt["asymmetric"] = [identity, identity]
        self.evaluator.poses_pred["asymmetric"] = [
            make_pose(translation=[0.009, 0.0, 0.0]),
            make_pose(translation=[0.011, 0.0, 0.0]),
        ]

        rotate_180_z = np.array(
            [[-1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]],
            dtype=np.float32,
        )
        self.evaluator.poses_gt["symmetric"] = [identity]
        self.evaluator.poses_pred["symmetric"] = [make_pose(rotation=rotate_180_z)]

        with tempfile.TemporaryDirectory() as output_dir:
            score = self.evaluator.evaluate_pose_adds_01d(output_dir)
            result_path = Path(output_dir) / "adds_01d" / "adds_01d.json"
            results = json.loads(result_path.read_text())

        self.assertAlmostEqual(score, 75.0)
        self.assertAlmostEqual(
            results["classes"]["asymmetric"]["threshold_m"],
            0.01,
        )
        self.assertEqual(results["classes"]["asymmetric"]["correct"], 1)
        self.assertEqual(results["classes"]["symmetric"]["correct"], 1)
        self.assertIsNone(results["classes"]["empty"]["accuracy"])
        self.assertEqual(results["accuracy"]["n_classes"], 2)
        self.assertAlmostEqual(results["accuracy"]["macro"], 75.0)
        self.assertAlmostEqual(results["accuracy"]["micro"], 200.0 / 3.0)

    def test_adds_01d_rejects_invalid_diameter(self):
        self.evaluator.models_info["asymmetric"]["diameter"] = 0.0
        with tempfile.TemporaryDirectory() as output_dir:
            with self.assertRaisesRegex(ValueError, "Invalid model diameter"):
                self.evaluator.evaluate_pose_adds_01d(output_dir)


if __name__ == "__main__":
    unittest.main()
