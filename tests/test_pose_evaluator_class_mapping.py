import unittest
from types import SimpleNamespace
from unittest.mock import patch

from evaluation_tools.pose_evaluator_init import (
    build_pose_evaluator,
    resolve_pose_class_name,
)


class PoseEvaluatorClassMappingTest(unittest.TestCase):
    def test_lmo_sparse_object_ids_resolve_to_the_correct_classes(self):
        classes = {
            "1": "ape",
            "5": "can",
            "6": "cat",
            "8": "driller",
            "9": "duck",
            "10": "eggbox",
            "11": "glue",
            "12": "holepuncher",
        }
        evaluator = SimpleNamespace(classes=list(classes.values()))
        args = SimpleNamespace(
            dataset_path="/dataset",
            class_info="/annotations/classes.json",
            models="/models/",
            model_symmetry="/annotations/symmetries.json",
            dataset_file="lmo",
        )

        with (
            patch(
                "evaluation_tools.pose_evaluator_init.load_classes",
                return_value=classes,
            ),
            patch(
                "evaluation_tools.pose_evaluator_init.load_models",
                return_value=({}, {}),
            ),
            patch(
                "evaluation_tools.pose_evaluator_init.load_model_symmetry",
                return_value={},
            ),
            patch(
                "evaluation_tools.pose_evaluator_init.PoseEvaluatorLMO",
                return_value=evaluator,
            ),
        ):
            result = build_pose_evaluator(args)

        self.assertIs(result, evaluator)
        self.assertEqual(resolve_pose_class_name(result, 1), "ape")
        self.assertEqual(resolve_pose_class_name(result, 5), "can")
        self.assertEqual(resolve_pose_class_name(result, 12), "holepuncher")

    def test_unknown_object_id_has_a_clear_error(self):
        evaluator = SimpleNamespace(
            classes=["ape"],
            class_id_to_name={1: "ape"},
        )

        with self.assertRaisesRegex(KeyError, "Available IDs: \\[1\\]"):
            resolve_pose_class_name(evaluator, 5)


if __name__ == "__main__":
    unittest.main()
