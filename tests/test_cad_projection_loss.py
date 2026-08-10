import json
import math

import pytest
import torch

from models.lwdetr6d import SetCriterion
from util.rotation_utils import build_symmetry_transforms


def rotation_z(angle):
    cosine = math.cos(angle)
    sine = math.sin(angle)
    return torch.tensor(
        [
            [cosine, -sine, 0.0],
            [sine, cosine, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=torch.float32,
    )


def make_criterion():
    return SetCriterion(
        num_classes=2,
        matcher=None,
        weight_dict={},
        focal_alpha=0.25,
        losses=[],
        rotation_mode="6d",
    )


def make_case(pred_rotation=None, pred_translation=None):
    identity = torch.eye(3)
    if pred_rotation is None:
        pred_rotation = identity
    if pred_translation is None:
        pred_translation = torch.tensor([0.0, 0.0, 1.0])

    outputs = {
        "pred_rotations": pred_rotation.reshape(1, 1, 3, 3),
        "pred_translations": pred_translation.reshape(1, 1, 3),
    }
    targets = [
        {
            "relative_rotation": identity.reshape(1, 3, 3),
            "relative_position": torch.tensor([[0.0, 0.0, 1.0]]),
            "model_points": torch.tensor(
                [[
                    [-0.1, -0.05, 0.0],
                    [0.1, -0.05, 0.02],
                    [0.1, 0.05, 0.0],
                    [-0.1, 0.05, -0.02],
                ]]
            ),
            "intrinsics": torch.tensor(
                [[[500.0, 0.0, 320.0], [0.0, 500.0, 240.0], [0.0, 0.0, 1.0]]]
            ),
            "is_symmetric": torch.tensor([False]),
        }
    ]
    indices = [(torch.tensor([0]), torch.tensor([0]))]
    return outputs, targets, indices


def test_behind_camera_prediction_has_bounded_corrective_gradient():
    outputs, targets, indices = make_case(
        pred_translation=torch.tensor([0.0, 0.0, -1.0])
    )
    outputs["pred_translations"].requires_grad_()

    loss = make_criterion().loss_cad_projection(
        outputs, targets, indices, num_boxes=1
    )["loss_adds"]
    loss.backward()

    assert 0.0 < loss.item() < 1.0
    assert torch.isfinite(outputs["pred_translations"].grad).all()
    assert outputs["pred_translations"].grad[0, 0, 2] < 0.0


def test_nonfinite_prediction_fails_fast():
    outputs, targets, indices = make_case(
        pred_translation=torch.tensor([float("nan"), 0.0, 1.0])
    )

    with pytest.raises(FloatingPointError, match="non-finite"):
        make_criterion().loss_cad_projection(
            outputs, targets, indices, num_boxes=1
        )


def test_empty_matches_return_finite_differentiable_zero():
    translations = torch.tensor(
        [[[float("nan"), 0.0, 1.0]]],
        requires_grad=True,
    )
    outputs = {
        "pred_rotations": torch.eye(3).reshape(1, 1, 3, 3),
        "pred_translations": translations,
    }
    empty = torch.empty(0, dtype=torch.long)

    loss = make_criterion().loss_cad_projection(
        outputs, [{}], [(empty, empty)], num_boxes=1
    )["loss_adds"]
    loss.backward()

    assert loss.item() == 0.0
    assert loss.requires_grad
    assert torch.equal(translations.grad, torch.zeros_like(translations))


def test_translated_symmetry_pose_has_zero_projection_loss():
    symmetry_rotation = rotation_z(math.pi)
    symmetry_translation = torch.tensor([0.02, -0.01, 0.005])
    outputs, targets, indices = make_case(
        pred_rotation=symmetry_rotation,
        pred_translation=torch.tensor([0.02, -0.01, 1.005]),
    )
    targets[0]["is_symmetric"] = torch.tensor([True])
    targets[0]["symmetry_transforms"] = torch.stack(
        [torch.eye(3), symmetry_rotation]
    ).unsqueeze(0)
    targets[0]["symmetry_translations"] = torch.stack(
        [torch.zeros(3), symmetry_translation]
    ).unsqueeze(0)

    loss = make_criterion().loss_cad_projection(
        outputs, targets, indices, num_boxes=1
    )["loss_adds"]

    torch.testing.assert_close(loss, torch.tensor(0.0), atol=1e-6, rtol=0.0)


def test_adds_and_cad_projection_share_switching_contract():
    outputs, targets, indices = make_case()
    targets[0]["symmetry_transforms"] = torch.eye(3).reshape(1, 1, 3, 3)
    targets[0]["symmetry_translations"] = torch.zeros(1, 1, 3)
    criterion = make_criterion()

    adds_result = criterion.loss_adds(
        outputs, targets, indices, num_boxes=1
    )
    cad_result = criterion.loss_cad_projection(
        outputs, targets, indices, num_boxes=1
    )

    assert adds_result.keys() == cad_result.keys() == {"loss_adds"}
    torch.testing.assert_close(
        adds_result["loss_adds"],
        torch.tensor(0.0),
        atol=1e-6,
        rtol=0.0,
    )
    torch.testing.assert_close(
        cad_result["loss_adds"],
        torch.tensor(0.0),
        atol=1e-6,
        rtol=0.0,
    )


def test_bop_symmetry_translations_are_converted_to_meters(tmp_path):
    models_info = {
        "1": {
            "symmetries_discrete": [
                [
                    -1, 0, 0, 10,
                    0, -1, 0, 20,
                    0, 0, 1, 30,
                    0, 0, 0, 1,
                ]
            ]
        }
    }
    (tmp_path / "models_info.json").write_text(json.dumps(models_info))

    _, translations = build_symmetry_transforms(
        tmp_path,
        return_translations=True,
        missing_continuous={},
    )

    torch.testing.assert_close(
        translations[1][1],
        torch.tensor([0.01, 0.02, 0.03]),
    )