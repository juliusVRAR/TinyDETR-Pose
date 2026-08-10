import math

import torch

from models.lwdetr6d import SetCriterion


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


def test_symmetric_rotation_uses_nearest_equivalent():
    identity = torch.eye(3)
    half_turn = rotation_z(math.pi)
    outputs = {"pred_rotations": half_turn.reshape(1, 1, 3, 3)}
    targets = [
        {
            "relative_rotation": identity.reshape(1, 3, 3),
            "is_symmetric": torch.tensor([True]),
            "symmetry_transforms": torch.stack((identity, half_turn)).unsqueeze(0),
        }
    ]
    indices = [(torch.tensor([0]), torch.tensor([0]))]

    loss = make_criterion().loss_rotation_symmetry_transform_min(
        outputs, targets, indices, num_boxes=1
    )["loss_rot"]

    torch.testing.assert_close(loss, torch.tensor(0.0), atol=1e-6, rtol=0.0)


def test_non_symmetric_rotation_uses_ordinary_geodesic_distance():
    identity = torch.eye(3)
    half_turn = rotation_z(math.pi)
    outputs = {"pred_rotations": half_turn.reshape(1, 1, 3, 3)}
    targets = [
        {
            "relative_rotation": identity.reshape(1, 3, 3),
            "is_symmetric": torch.tensor([False]),
            "symmetry_transforms": torch.stack((identity, half_turn)).unsqueeze(0),
        }
    ]
    indices = [(torch.tensor([0]), torch.tensor([0]))]

    loss = make_criterion().loss_rotation_symmetry_transform_min(
        outputs, targets, indices, num_boxes=1
    )["loss_rot"]

    torch.testing.assert_close(loss, torch.tensor(math.pi), atol=1e-6, rtol=0.0)