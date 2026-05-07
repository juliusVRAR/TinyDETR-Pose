# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE_PYTORCH3D file in the LICENSES folder.

import math
import torch
import torch.nn.functional as F
from typing import Tuple
import trimesh
import json
import numpy as np
from pathlib import Path


SARR_YCBV_SYM_V = {
    # 1:  (1, 1, 1000),   # 002_master_chef_can  (only discrete 180° in JSON)
    # 4:  (0, 1, 1000),   # 005_tomato_soup_can  (nothing in JSON)
    # 6:  (0, 1, 1000),   # 007_tuna_fish_can    (nothing in JSON)
    13: (1, 1, 1000),  # 024_bowl
    16: (1, 1, 4),     # 036_wood_block
    19: (1, 1, 2),     # 051_large_clamp
    20: (1, 1, 2),     # 052_extra_large_clamp
    21: (1, 1, 2),     # 061_foam_brick
}
_SARR_YCBV_SYM_TABLE = torch.ones(max(SARR_YCBV_SYM_V) + 1, 3, dtype=torch.long)
for _cls_id, _sym_v in SARR_YCBV_SYM_V.items():
    _SARR_YCBV_SYM_TABLE[_cls_id] = torch.tensor(_sym_v, dtype=torch.long)

DEFAULT_ACOS_BOUND = 1.0 - 1e-4

def acos_linear_extrapolation(
    x: torch.Tensor,
    bounds: Tuple[float, float] = (-DEFAULT_ACOS_BOUND, DEFAULT_ACOS_BOUND),
) -> torch.Tensor:
    """
    Implements `arccos(x)` which is linearly extrapolated outside `x`'s original
    domain of `(-1, 1)`. This allows for stable backpropagation in case `x`
    is not guaranteed to be strictly within `(-1, 1)`.
    More specifically:
    ```
    bounds=(lower_bound, upper_bound)
    if lower_bound <= x <= upper_bound:
        acos_linear_extrapolation(x) = acos(x)
    elif x <= lower_bound: # 1st order Taylor approximation
        acos_linear_extrapolation(x)
            = acos(lower_bound) + dacos/dx(lower_bound) * (x - lower_bound)
    else:  # x >= upper_bound
        acos_linear_extrapolation(x)
            = acos(upper_bound) + dacos/dx(upper_bound) * (x - upper_bound)
    ```
    Args:
        x: Input `Tensor`.
        bounds: A float 2-tuple defining the region for the
            linear extrapolation of `acos`.
            The first/second element of `bound`
            describes the lower/upper bound that defines the lower/upper
            extrapolation region, i.e. the region where
            `x <= bound[0]`/`bound[1] <= x`.
            Note that all elements of `bound` have to be within (-1, 1).
    Returns:
        acos_linear_extrapolation: `Tensor` containing the extrapolated `arccos(x)`.
    """

    lower_bound, upper_bound = bounds

    if lower_bound > upper_bound:
        raise ValueError("lower bound has to be smaller or equal to upper bound.")

    if lower_bound <= -1.0 or upper_bound >= 1.0:
        raise ValueError("Both lower bound and upper bound have to be within (-1, 1).")

    # init an empty tensor and define the domain sets
    acos_extrap = torch.empty_like(x)
    x_upper = x >= upper_bound
    x_lower = x <= lower_bound
    x_mid = (~x_upper) & (~x_lower)

    # acos calculation for upper_bound < x < lower_bound
    acos_extrap[x_mid] = torch.acos(x[x_mid])
    # the linear extrapolation for x >= upper_bound
    acos_extrap[x_upper] = _acos_linear_approximation(x[x_upper], upper_bound)
    # the linear extrapolation for x <= lower_bound
    acos_extrap[x_lower] = _acos_linear_approximation(x[x_lower], lower_bound)

    return acos_extrap


def acos_linear_extrapolation(
    x: torch.Tensor,
    bounds: Tuple[float, float] = (-DEFAULT_ACOS_BOUND, DEFAULT_ACOS_BOUND),
) -> torch.Tensor:
    """
    Implements `arccos(x)` which is linearly extrapolated outside `x`'s original
    domain of `(-1, 1)`. This allows for stable backpropagation in case `x`
    is not guaranteed to be strictly within `(-1, 1)`.
    More specifically:
    ```
    bounds=(lower_bound, upper_bound)
    if lower_bound <= x <= upper_bound:
        acos_linear_extrapolation(x) = acos(x)
    elif x <= lower_bound: # 1st order Taylor approximation
        acos_linear_extrapolation(x)
            = acos(lower_bound) + dacos/dx(lower_bound) * (x - lower_bound)
    else:  # x >= upper_bound
        acos_linear_extrapolation(x)
            = acos(upper_bound) + dacos/dx(upper_bound) * (x - upper_bound)
    ```
    Args:
        x: Input `Tensor`.
        bounds: A float 2-tuple defining the region for the
            linear extrapolation of `acos`.
            The first/second element of `bound`
            describes the lower/upper bound that defines the lower/upper
            extrapolation region, i.e. the region where
            `x <= bound[0]`/`bound[1] <= x`.
            Note that all elements of `bound` have to be within (-1, 1).
    Returns:
        acos_linear_extrapolation: `Tensor` containing the extrapolated `arccos(x)`.
    """

    lower_bound, upper_bound = bounds

    if lower_bound > upper_bound:
        raise ValueError("lower bound has to be smaller or equal to upper bound.")

    if lower_bound <= -1.0 or upper_bound >= 1.0:
        raise ValueError("Both lower bound and upper bound have to be within (-1, 1).")

    # init an empty tensor and define the domain sets
    acos_extrap = torch.empty_like(x)
    x_upper = x >= upper_bound
    x_lower = x <= lower_bound
    x_mid = (~x_upper) & (~x_lower)

    # acos calculation for upper_bound < x < lower_bound
    acos_extrap[x_mid] = torch.acos(x[x_mid])
    # the linear extrapolation for x >= upper_bound
    acos_extrap[x_upper] = _acos_linear_approximation(x[x_upper], upper_bound)
    # the linear extrapolation for x <= lower_bound
    acos_extrap[x_lower] = _acos_linear_approximation(x[x_lower], lower_bound)

    return acos_extrap


def _acos_linear_approximation(x: torch.Tensor, x0: float) -> torch.Tensor:
    """
    Calculates the 1st order Taylor expansion of `arccos(x)` around `x0`.
    """
    return (x - x0) * _dacos_dx(x0) + math.acos(x0)


def _dacos_dx(x: float) -> float:
    """
    Calculates the derivative of `arccos(x)` w.r.t. `x`.
    """
    return (-1.0) / math.sqrt(1.0 - x * x)


def so3_rotation_angle(
    R: torch.Tensor,
    eps: float = 1e-4,
    cos_angle: bool = False,
    cos_bound: float = 1e-4,
) -> torch.Tensor:
    """
    Calculates angles (in radians) of a batch of rotation matrices `R` with
    `angle = acos(0.5 * (Trace(R)-1))`. The trace of the
    input matrices is checked to be in the valid range `[-1-eps,3+eps]`.
    The `eps` argument is a small constant that allows for small errors
    caused by limited machine precision.
    Args:
        R: Batch of rotation matrices of shape `(minibatch, 3, 3)`.
        eps: Tolerance for the valid trace check.
        cos_angle: If==True return cosine of the rotation angles rather than
            the angle itself. This can avoid the unstable
            calculation of `acos`.
        cos_bound: Clamps the cosine of the rotation angle to
            [-1 + cos_bound, 1 - cos_bound] to avoid non-finite outputs/gradients
            of the `acos` call. Note that the non-finite outputs/gradients
            are returned when the angle is requested (i.e. `cos_angle==False`)
            and the rotation angle is close to 0 or π.
    Returns:
        Corresponding rotation angles of shape `(minibatch,)`.
        If `cos_angle==True`, returns the cosine of the angles.
    Raises:
        ValueError if `R` is of incorrect shape.
        ValueError if `R` has an unexpected trace.
    """

    N, dim1, dim2 = R.shape
    if dim1 != 3 or dim2 != 3:
        raise ValueError("Input has to be a batch of 3x3 Tensors.")

    rot_trace = R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]

    if ((rot_trace < -1.0 - eps) + (rot_trace > 3.0 + eps)).any():
        raise ValueError("A matrix has trace outside valid range [-1-eps,3+eps].")

    # phi ... rotation angle
    phi_cos = (rot_trace - 1.0) * 0.5

    if cos_angle:
        return phi_cos
    else:
        if cos_bound > 0.0:
            bound = 1.0 - cos_bound
            return acos_linear_extrapolation(phi_cos, (-bound, bound))
        else:
            return torch.acos(phi_cos)


def so3_exp_map(log_rot: torch.Tensor, eps: float = 0.0001) -> torch.Tensor:
    """
    Convert a batch of logarithmic representations of rotation matrices `log_rot`
    to a batch of 3x3 rotation matrices using Rodrigues formula [1].
    In the logarithmic representation, each rotation matrix is represented as
    a 3-dimensional vector (`log_rot`) who's l2-norm and direction correspond
    to the magnitude of the rotation angle and the axis of rotation respectively.
    The conversion has a singularity around `log(R) = 0`
    which is handled by clamping controlled with the `eps` argument.
    Args:
        log_rot: Batch of vectors of shape `(minibatch, 3)`.
        eps: A float constant handling the conversion singularity.
    Returns:
        Batch of rotation matrices of shape `(minibatch, 3, 3)`.
    Raises:
        ValueError if `log_rot` is of incorrect shape.
    [1] https://en.wikipedia.org/wiki/Rodrigues%27_rotation_formula
    """
    return _so3_exp_map(log_rot, eps=eps)[0]


def _so3_exp_map(log_rot: torch.Tensor, eps: float = 0.0001) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    A helper function that computes the so3 exponential map and,
    apart from the rotation matrix, also returns intermediate variables
    that can be re-used in other functions.
    """
    _, dim = log_rot.shape
    if dim != 3:
        raise ValueError("Input tensor shape has to be Nx3.")

    nrms = (log_rot * log_rot).sum(1)
    # phis ... rotation angles
    rot_angles = torch.clamp(nrms, eps).sqrt()
    rot_angles_inv = 1.0 / rot_angles
    fac1 = rot_angles_inv * rot_angles.sin()
    fac2 = rot_angles_inv * rot_angles_inv * (1.0 - rot_angles.cos())
    skews = hat(log_rot)
    skews_square = torch.bmm(skews, skews)

    R = (
        # pyre-fixme[16]: `float` has no attribute `__getitem__`.
        fac1[:, None, None] * skews
        + fac2[:, None, None] * skews_square
        + torch.eye(3, dtype=log_rot.dtype, device=log_rot.device)[None]
    )

    return R, rot_angles, skews, skews_square


def so3_log_map(R: torch.Tensor, eps: float = 0.0001, cos_bound: float = 1e-4) -> torch.Tensor:
    """
    Convert a batch of 3x3 rotation matrices `R`
    to a batch of 3-dimensional matrix logarithms of rotation matrices
    The conversion has a singularity around `(R=I)` which is handled
    by clamping controlled with the `eps` and `cos_bound` arguments.
    Args:
        R: batch of rotation matrices of shape `(minibatch, 3, 3)`.
        eps: A float constant handling the conversion singularity.
        cos_bound: Clamps the cosine of the rotation angle to
            [-1 + cos_bound, 1 - cos_bound] to avoid non-finite outputs/gradients
            of the `acos` call when computing `so3_rotation_angle`.
            Note that the non-finite outputs/gradients are returned when
            the rotation angle is close to 0 or π.
    Returns:
        Batch of logarithms of input rotation matrices
        of shape `(minibatch, 3)`.
    Raises:
        ValueError if `R` is of incorrect shape.
        ValueError if `R` has an unexpected trace.
    """

    N, dim1, dim2 = R.shape
    if dim1 != 3 or dim2 != 3:
        raise ValueError("Input has to be a batch of 3x3 Tensors.")

    phi = so3_rotation_angle(R, cos_bound=cos_bound, eps=eps)

    phi_sin = torch.sin(phi)

    # We want to avoid a tiny denominator of phi_factor = phi / (2.0 * phi_sin).
    # Hence, for phi_sin.abs() <= 0.5 * eps, we approximate phi_factor with
    # 2nd order Taylor expansion: phi_factor = 0.5 + (1.0 / 12) * phi**2
    phi_factor = torch.empty_like(phi)
    ok_denom = phi_sin.abs() > (0.5 * eps)
    phi_factor[~ok_denom] = 0.5 + (phi[~ok_denom] ** 2) * (1.0 / 12)
    phi_factor[ok_denom] = phi[ok_denom] / (2.0 * phi_sin[ok_denom])

    log_rot_hat = phi_factor[:, None, None] * (R - R.permute(0, 2, 1))

    log_rot = hat_inv(log_rot_hat)

    return log_rot


def hat_inv(h: torch.Tensor) -> torch.Tensor:
    """
    Compute the inverse Hat operator [1] of a batch of 3x3 matrices.
    Args:
        h: Batch of skew-symmetric matrices of shape `(minibatch, 3, 3)`.
    Returns:
        Batch of 3d vectors of shape `(minibatch, 3, 3)`.
    Raises:
        ValueError if `h` is of incorrect shape.
        ValueError if `h` not skew-symmetric.
    [1] https://en.wikipedia.org/wiki/Hat_operator
    """

    N, dim1, dim2 = h.shape
    if dim1 != 3 or dim2 != 3:
        raise ValueError("Input has to be a batch of 3x3 Tensors.")

    ss_diff = torch.abs(h + h.permute(0, 2, 1)).max()

    HAT_INV_SKEW_SYMMETRIC_TOL = 1e-5
    if float(ss_diff) > HAT_INV_SKEW_SYMMETRIC_TOL:
        raise ValueError("One of input matrices is not skew-symmetric.")

    x = h[:, 2, 1]
    y = h[:, 0, 2]
    z = h[:, 1, 0]

    v = torch.stack((x, y, z), dim=1)

    return v


def hat(v: torch.Tensor) -> torch.Tensor:
    """
    Compute the Hat operator [1] of a batch of 3D vectors.
    Args:
        v: Batch of vectors of shape `(minibatch , 3)`.
    Returns:
        Batch of skew-symmetric matrices of shape
        `(minibatch, 3 , 3)` where each matrix is of the form:
            `[    0  -v_z   v_y ]
             [  v_z     0  -v_x ]
             [ -v_y   v_x     0 ]`
    Raises:
        ValueError if `v` is of incorrect shape.
    [1] https://en.wikipedia.org/wiki/Hat_operator
    """

    N, dim = v.shape
    if dim != 3:
        raise ValueError("Input vectors have to be 3-dimensional.")

    h = torch.zeros((N, 3, 3), dtype=v.dtype, device=v.device)

    x, y, z = v.unbind(1)

    h[:, 0, 1] = -z
    h[:, 0, 2] = y
    h[:, 1, 0] = z
    h[:, 1, 2] = -x
    h[:, 2, 0] = -y
    h[:, 2, 1] = x

    return h


# def calc_average_rotation(rotations, max_iter=25, eps=1e-10):
#     """
#     Given a batch of rotation matrices (bs, 3, 3) calculate the average rotation matrix
#     """
#     n_rot = len(rotations)
#     # Initialize the average rotation
#     avg_rot = rotations[0]
#     for i in range(max_iter):
#         dist_rot = []
#         for rot in rotations:
#             dist_rot.append(so3_log_map(torch.matmul(avg_rot.transpose(0, 1), rot).reshape(1, 3, 3)))
#         avg_rot_logm = torch.mean(torch.stack(dist_rot), dim=0)
#         if torch.norm(avg_rot_logm) < eps:
#             break
#
#         avg_rot = torch.matmul(avg_rot, so3_exponential_map(avg_rot_logm))[0]
#
#     return avg_rot

def flat9_to_matrix(R_flat: torch.Tensor) -> torch.Tensor:
    """
    Convert flattened rotations (N,9) to (N,3,3).
    Accepts also (B,Q,9) -> returns (B,Q,3,3).
    """
    if R_flat.shape[-1] != 9:
        raise ValueError("Last dim must be 9 to reshape into 3x3.")
    new_shape = R_flat.shape[:-1] + (3, 3)
    return R_flat.view(*new_shape)

def rotation_matrix_to_gram_schmidt_6d(R: torch.Tensor) -> torch.Tensor:
    """
    R: (B,3,3) rotation matrices
    Returns: (B,6) 6D Gram-Schmidt (Zhou et al.) representation.
    """
    if R.dim()!=3 or R.shape[1:] != (3,3):
        raise ValueError("Expected (B,3,3) rotation matrices.")
    c1 = F.normalize(R[:, :, 0], dim=1)
    b  = R[:, :, 1]
    proj = (c1 * b).sum(dim=1, keepdim=True) * c1
    c2 = F.normalize(b - proj, dim=1)
    return torch.cat([c1, c2], dim=1)

def rotation_matrix_to_raw_6d(R: torch.Tensor) -> torch.Tensor:
    """
    Extract the first two columns of rotation matrices and flatten to 6D (no re-orthogonalization).
    Equivalent to the numpy:
        R_gs = np.array(R).reshape(3,3)
        raw6d = np.squeeze(R_gs[:, :2].transpose().reshape(6,1))
    For a batch:
        Input:  R of shape (B, 3, 3)
        Output: raw6d of shape (B, 6) with ordering:
                [R[0,0], R[1,0], R[2,0], R[0,1], R[1,1], R[2,1]]
    """
    if R.dim() != 3 or R.shape[1:] != (3, 3):
        raise ValueError(f"Expected (B,3,3), got {tuple(R.shape)}")
    # Take first two columns -> (B, 3, 2)
    first_two = R[:, :, :2]
    # Permute to (B, 2, 3) so each row is one column, then flatten to (B,6)
    raw6d = first_two.permute(0, 2, 1).reshape(R.size(0), 6)
    return raw6d

def rotation_6d_to_matrix(rot_6d):
    """
    Converts 6D rotation representation to SO(3) rotation matrix via
    Gram-Schmidt orthogonalization (Zhou et al., CVPR 2019).
    https://openaccess.thecvf.com/content_CVPR_2019/papers/Zhou_On_the_Continuity_of_Rotation_Representations_in_Neural_Networks_CVPR_2019_paper.pdf

    Uses standard Zhou et al. convention:
        Free parameters: a1 (col1), a2 (col2)
        b1 = normalize(a1)
        b2 = normalize(a2 - (b1·a2) * b1)    ← GS orthogonalize then normalize
        b3 = cross(b1, b2)                    ← fully determined, no free param

    FIX 1: Switched from POET convention (x, z free → y derived) to
            standard Zhou convention (col1, col2 free → col3 derived).
            POET convention caused a column ordering mismatch against
            YCB-V ground truth rotations, explaining the 77° plateau.

    FIX 2: Replaced brittle manual shape unpacking with shape[-1] reshape
            pattern — handles (B,Q,6) and (D,B,Q,6) without explicit branches.

    FIX 3: Single Gram-Schmidt here only. Removed the redundant inline GS
            (rot_c1/rot_c2) from forward() — double GS was adding unnecessary
            operations to the gradient path near convergence.

    Args:
        rot_6d: (..., 6) — raw MLP output, any leading batch dims
    Returns:
        (..., 3, 3) rotation matrix in SO(3), columns = [b1, b2, b3]
    """
    # FIX 2: preserve all leading dims generically — no manual unpacking
    leading = rot_6d.shape[:-1]         # e.g. (D,B,Q) or (B,Q)
    rot_6d  = rot_6d.reshape(-1, 6)     # (N, 6)

    a1 = rot_6d[:, :3]                  # (N, 3) — first free parameter
    a2 = rot_6d[:, 3:]                  # (N, 3) — second free parameter

    # FIX 1: Standard Zhou et al. Gram-Schmidt
    # Column 1: normalize directly
    b1 = F.normalize(a1, p=2, dim=1)                                        # (N, 3)

    # Column 2: remove b1 component from a2, then normalize
    b2 = F.normalize(a2 - (b1 * a2).sum(dim=1, keepdim=True) * b1,
                     p=2, dim=1)                                             # (N, 3)

    # Column 3: fully determined by right-hand rule
    b3 = torch.cross(b1, b2, dim=1)                                         # (N, 3)

    # Stack as columns: R = [b1 | b2 | b3]
    # R @ p_obj = p_cam  (standard camera convention)
    R = torch.stack([b1, b2, b3], dim=-1)                                   # (N, 3, 3)

    # FIX 2: restore all leading dims generically
    return R.reshape(*leading, 3, 3)


def rotation_6d_to_matrix_old(rot_6d):
    """
    Given a 6D rotation output, calculate the 3D rotation matrix in SO(3) using the Gramm Schmit process

    For details: https://openaccess.thecvf.com/content_CVPR_2019/papers/Zhou_On_the_Continuity_of_Rotation_Representations_in_Neural_Networks_CVPR_2019_paper.pdf
    """
    in_shape = len(rot_6d.shape)
    if in_shape == 4:
        dec_layers, bs, n_q, _ = rot_6d.shape
    elif in_shape == 3:
        bs, n_q, _ = rot_6d.shape
    else:
        raise ValueError("Input rot_6d must be of shape (B,Q,6) or (D,B,Q,6)")
    
    rot_6d = rot_6d.view(-1, 6)
    x_raw = rot_6d[:, 0:3]
    y_raw = rot_6d[:, 3:6]
    # From poet, they assume x and z rotation vectors, and compute y as cross product.
    # TODO: Is that a problem?
    x = F.normalize(x_raw, p=2, dim=1)
    z = torch.cross(x, y_raw, dim=1)
    z = F.normalize(z, p=2, dim=1)
    y = torch.cross(z, x, dim=1)
    rot_matrix = torch.cat((x.view(-1, 3, 1), y.view(-1, 3, 1), z.view(-1, 3, 1)), 2)  # Rotation Matrix lying in the SO(3)
    if in_shape == 4:
        rot_matrix = rot_matrix.view(dec_layers, bs, n_q, 3, 3)  #.transpose(2, 3) 
    else:
        rot_matrix = rot_matrix.view(bs, n_q, 3, 3)
    return rot_matrix

def rotation_6d_simple_to_matrix(rot_6d: torch.Tensor) -> torch.Tensor:
    """
    Convert a batch of 6D Gram-Schmidt rotation representations (N,6)
    into rotation matrices (N,3,3).

    rot_6d: Tensor shape (N,6)
        First 3 values: raw first axis
        Last 3 values: raw second axis
    Returns:
        (N,3,3) valid rotation matrices in SO(3)

    This is a lightweight variant of rotation_6d_to_matrix that
    only accepts rank-2 input and applies standard Gram-Schmidt:
        c1 = normalize(a)
        c2 = normalize(b - proj_b_on_c1)
        c3 = cross(c1, c2)
    """
    if rot_6d.dim() != 2 or rot_6d.size(1) != 6:
        raise ValueError(f"Expected (N,6) got {tuple(rot_6d.shape)}")
    a = rot_6d[:, 0:3]
    b = rot_6d[:, 3:6]
    c1 = F.normalize(a, dim=1)
    # Remove projection of b onto c1
    proj = (c1 * b).sum(dim=1, keepdim=True) * c1
    c2 = F.normalize(b - proj, dim=1)
    c3 = torch.cross(c1, c2, dim=1)
    R = torch.stack((c1, c2, c3), dim=2)  # (N,3,3) columns = orthonormal basis
    return R


# Example usage with the symmetric-aware rotation loss from earlier:
# points_dict = precompute_points(ycbv_model_paths, n=1500)
# loss = symmetric_aware_rot_loss(R_pred, R_gt, points=points_dict[obj_id],
#                                symmetric_flags=[is_symmetric_for_this_instance])
def load_model_points(mesh_path, n=1500, seed=0):
    """
    Uniformly sample n points on the mesh surface.
    Returns (n,3) float32 tensor in the mesh/object coordinate frame.
    """
    mesh = trimesh.load(mesh_path, process=True)
    pts, _ = trimesh.sample.sample_surface(mesh, n, seed=seed)
    return torch.from_numpy(pts).float()

def precompute_points(model_paths, n=1500, seed=0):
    """
    model_paths: dict {obj_id: path_to_mesh}
    Returns: dict {obj_id: (n,3) tensor}
    """
    return {obj_id: load_model_points(path, n=n, seed=seed)
            for obj_id, path in model_paths.items()}


YCBV_MISSING_CONTINUOUS = {
    1: [0, 0, 1],   # 002_master_chef_can  (only discrete 180° in JSON)
    4: [0, 0, 1],   # 005_tomato_soup_can  (nothing in JSON)
    6: [0, 0, 1],   # 007_tuna_fish_can    (nothing in JSON)
}


def build_symmetry_transforms(cad_models_path, K_continuous=360,
                               missing_continuous=None):
    models_info_path = Path(cad_models_path) / "models_info.json"
    with open(models_info_path, 'r') as f:
        models_info = json.load(f)

    if missing_continuous is None:
        missing_continuous = YCBV_MISSING_CONTINUOUS

    sym_dict = {}

    for obj_id_str, info in models_info.items():
        obj_id = int(obj_id_str)
        rots = [torch.eye(3)]

        # --- Discrete symmetries (from JSON) ---
        for T_flat in info.get('symmetries_discrete', []):
            T = np.array(T_flat, dtype=np.float32).reshape(4, 4)
            rots.append(torch.from_numpy(T[:3, :3].copy()))

        # --- Continuous symmetries (from JSON) ---
        found_continuous = False
        for sym in info.get('symmetries_continuous', []):
            axis = np.array(sym['axis'], dtype=np.float32)
            axis = axis / np.linalg.norm(axis)
            rots = _add_continuous_rotations(rots, axis, K_continuous)
            found_continuous = True

        # --- Fallback: manual override for missing entries ---
        if not found_continuous and obj_id in missing_continuous:
            axis = np.array(missing_continuous[obj_id], dtype=np.float32)
            # Clear any discrete rotations — continuous supersedes them
            rots = [torch.eye(3)]
            rots = _add_continuous_rotations(rots, axis, K_continuous)
            print(f"  obj {obj_id}: added manual continuous "
                  f"symmetry (axis={axis.tolist()}, K={K_continuous})")

        sym_dict[obj_id] = torch.stack(rots)

    return sym_dict


def _add_continuous_rotations(rots, axis, K):
    """Append K-1 evenly spaced rotations around `axis` to `rots`."""
    axis = axis / np.linalg.norm(axis)
    K_skew = np.array([
        [0,        -axis[2],  axis[1]],
        [axis[2],   0,       -axis[0]],
        [-axis[1],  axis[0],  0       ]
    ], dtype=np.float32)
    angles = np.linspace(0, 2 * np.pi, K, endpoint=False)
    for angle in angles[1:]:  # skip 0 (identity)
        R = (np.eye(3, dtype=np.float32)
             + np.sin(angle) * K_skew
             + (1 - np.cos(angle)) * (K_skew @ K_skew))

        rots.append(torch.from_numpy(R))
    return rots


def pad_symmetry_transforms(sym_dict):
    """Pad all objects to same K with identity (safe for min operation)."""
    max_K = max(v.shape[0] for v in sym_dict.values())
    padded = {}
    for obj_id, rots in sym_dict.items():
        K_i = rots.shape[0]
        if K_i < max_K:
            # Duplicate identity — doesn't affect min
            pad = torch.eye(3).unsqueeze(0).expand(max_K - K_i, 3, 3)
            rots = torch.cat([rots, pad], dim=0)
        padded[obj_id] = rots
    return padded, max_K


def get_sarr_symmetry_vectors(class_ids: torch.Tensor) -> torch.Tensor:
    """
    Return per-class SARR symmetry vectors.
    Only the five requested YCB-V symmetric classes are mapped to non-trivial
    symmetry vectors; all other classes default to [1, 1, 1].
    """
    class_ids = torch.as_tensor(class_ids)
    device = class_ids.device
    table = _SARR_YCBV_SYM_TABLE.to(device=device)
    flat_ids = class_ids.reshape(-1).long()
    valid = (flat_ids >= 0) & (flat_ids < table.shape[0])
    safe_ids = flat_ids.clamp(0, table.shape[0] - 1)
    sym_v = table[safe_ids]
    sym_v = torch.where(valid.unsqueeze(-1), sym_v, torch.ones_like(sym_v))
    return sym_v.reshape(*class_ids.shape, 3)


def normalize_sarr_pairs(sarr: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Normalize each [sin(theta), cos(theta)] pair in a flattened 6D SARR vector.
    """
    if sarr.shape[-1] != 6:
        raise ValueError(f"Expected (..., 6), got {tuple(sarr.shape)}")
    shape = sarr.shape
    pairs = sarr.reshape(*shape[:-1], 3, 2)
    pairs = F.normalize(pairs, p=2, dim=-1, eps=eps)
    return pairs.reshape(shape)


def xyz_euler_to_matrix_torch(angles: torch.Tensor) -> torch.Tensor:
    """
    Convert XYZ intrinsic Tait-Bryan angles to rotation matrices using the
    same closed-form expression as the public SARR implementation.
    angles: (..., 3) ordered as [alpha, beta, gamma]
    returns: (..., 3, 3)
    """
    alpha, beta, gamma = angles.unbind(dim=-1)
    ca, cb, cg = torch.cos(alpha), torch.cos(beta), torch.cos(gamma)
    sa, sb, sg = torch.sin(alpha), torch.sin(beta), torch.sin(gamma)

    row0 = torch.stack([cb * cg, -cb * sg, sb], dim=-1)
    row1 = torch.stack([ca * sg + cg * sa * sb, ca * cg - sa * sb * sg, -cb * sa], dim=-1)
    row2 = torch.stack([sa * sg - ca * cg * sb, cg * sa + ca * sb * sg, ca * cb], dim=-1)
    return torch.stack([row0, row1, row2], dim=-2)


def rotation_matrix_to_xyz_euler_torch(R: torch.Tensor) -> torch.Tensor:
    """
    Inverse of xyz_euler_to_matrix_torch for the XYZ convention used by SARR.
    Returns angles ordered as [alpha, beta, gamma].
    """
    if R.shape[-2:] != (3, 3):
        raise ValueError(f"Expected (..., 3, 3), got {tuple(R.shape)}")

    sy = R[..., 0, 2].clamp(-1.0, 1.0)
    beta = torch.asin(sy)
    alpha = torch.atan2(-R[..., 1, 2], R[..., 2, 2])
    gamma = torch.atan2(-R[..., 0, 1], R[..., 0, 0])
    return torch.stack([alpha, beta, gamma], dim=-1)


def _sarr_effective_factor(sym_component: torch.Tensor) -> torch.Tensor:
    factor = sym_component.remainder(1000)
    return factor.to(dtype=torch.float32)


def clamp_rotations_sarr(angles: torch.Tensor, sym_v: torch.Tensor) -> torch.Tensor:
    """
    Torch variant of SARR's clamp_rot_adv helper.
    angles: (N, 3)
    sym_v:  (N, 3) integer symmetry vectors
    """
    two_pi = 2.0 * math.pi
    alpha, beta, gamma = angles.unbind(dim=-1)
    sym_v = sym_v.to(device=angles.device)

    y_sym_mask = sym_v[:, 1] > 1
    if y_sym_mask.any():
        alpha_mod = torch.remainder(alpha[y_sym_mask], two_pi)
        flip_mask = alpha_mod > math.pi
        if flip_mask.any():
            idx = torch.where(y_sym_mask)[0][flip_mask]
            alpha[idx] = torch.remainder(alpha[idx] - math.pi, two_pi)
            gamma[idx] = torch.remainder(math.pi - gamma[idx], two_pi)
            beta[idx] = -beta[idx]

    other_mask = ~y_sym_mask
    if other_mask.any():
        sym_other = sym_v[other_mask].to(dtype=angles.dtype)
        alpha_o = alpha[other_mask]
        beta_o = beta[other_mask]
        gamma_o = gamma[other_mask]

        k0 = sym_other[:, 0]
        k1 = sym_other[:, 1]
        k2 = sym_other[:, 2]

        f0 = _sarr_effective_factor(sym_v[other_mask, 0]).to(dtype=angles.dtype) / k0
        f1 = _sarr_effective_factor(sym_v[other_mask, 1]).to(dtype=angles.dtype) / k1
        f2 = _sarr_effective_factor(sym_v[other_mask, 2]).to(dtype=angles.dtype) / k2

        alpha[other_mask] = torch.remainder(alpha_o, two_pi / k0) * f0
        beta[other_mask] = torch.remainder(beta_o, two_pi / k1) * f1
        gamma[other_mask] = torch.remainder(gamma_o, two_pi / k2) * f2

    out = torch.stack([alpha, beta, gamma], dim=-1)
    close_to_zero = torch.isclose(out, torch.zeros_like(out), atol=1e-10, rtol=0.0)
    close_to_two_pi = torch.isclose(out, torch.full_like(out, two_pi), atol=1e-10, rtol=0.0)
    out = torch.where(close_to_zero | close_to_two_pi, torch.zeros_like(out), out)
    return out


def rotation_matrix_to_sarr(R: torch.Tensor, sym_v: torch.Tensor, clamp: bool = True) -> torch.Tensor:
    """
    Encode rotation matrices into 6D SARR vectors ordered as
    [s_a, c_a, s_b, c_b, s_g, c_g].
    """
    angles = rotation_matrix_to_xyz_euler_torch(R)
    if clamp:
        angles = clamp_rotations_sarr(angles, sym_v)

    alpha, beta, gamma = angles.unbind(dim=-1)
    sym_v = sym_v.to(device=R.device)
    c_a = torch.cos(alpha)
    c_b = torch.cos(beta)

    sarr = torch.zeros((*angles.shape[:-1], 6), device=R.device, dtype=R.dtype)

    all_one = (sym_v == 1).all(dim=-1)
    z_only = (sym_v[:, 2] > 1) & (sym_v[:, 0] == 1) & (sym_v[:, 1] == 1)
    y_only = (sym_v[:, 1] > 1) & (sym_v[:, 0] == 1) & (sym_v[:, 2] == 1)
    x_only = (sym_v[:, 0] > 1) & (sym_v[:, 1] == 1) & (sym_v[:, 2] == 1)
    mixed = ~(all_one | z_only | y_only | x_only)
    unsupported = mixed & (sym_v == 1).any(dim=-1)
    if unsupported.any():
        raise NotImplementedError("SARR mixed symmetry with a unit axis is not supported in this refactor.")

    if all_one.any():
        idx = all_one
        sarr[idx, 0] = torch.sin(alpha[idx])
        sarr[idx, 1] = torch.cos(alpha[idx])
        sarr[idx, 2] = torch.sin(beta[idx])
        sarr[idx, 3] = torch.cos(beta[idx])
        sarr[idx, 4] = torch.sin(gamma[idx])
        sarr[idx, 5] = torch.cos(gamma[idx])

    if z_only.any():
        idx = z_only
        k = _sarr_effective_factor(sym_v[idx, 2]).to(dtype=R.dtype)
        sarr[idx, 0] = torch.sin(alpha[idx])
        sarr[idx, 1] = torch.cos(alpha[idx])
        sarr[idx, 2] = torch.sin(beta[idx])
        sarr[idx, 3] = torch.cos(beta[idx])
        sarr[idx, 4] = torch.sin(gamma[idx] * k)
        sarr[idx, 5] = torch.cos(gamma[idx] * k)

    if y_only.any():
        idx = y_only
        k = _sarr_effective_factor(sym_v[idx, 1]).to(dtype=R.dtype)
        sarr[idx, 0] = torch.sin(alpha[idx])
        sarr[idx, 1] = torch.cos(alpha[idx])
        sarr[idx, 2] = torch.sin(beta[idx] * k)
        sarr[idx, 3] = torch.cos(beta[idx] * k)
        sarr[idx, 4] = torch.sin(gamma[idx]) * c_b[idx]
        sarr[idx, 5] = torch.cos(gamma[idx])

    if x_only.any():
        idx = x_only
        k = _sarr_effective_factor(sym_v[idx, 0]).to(dtype=R.dtype)
        sarr[idx, 0] = torch.sin(alpha[idx] * k)
        sarr[idx, 1] = torch.cos(alpha[idx] * k)
        sarr[idx, 2] = torch.sin(beta[idx]) * c_a[idx]
        sarr[idx, 3] = torch.cos(beta[idx])
        sarr[idx, 4] = torch.sin(gamma[idx]) * c_a[idx]
        sarr[idx, 5] = torch.cos(gamma[idx])

    if mixed.any():
        idx = mixed
        k0 = _sarr_effective_factor(sym_v[idx, 0]).to(dtype=R.dtype)
        k1 = _sarr_effective_factor(sym_v[idx, 1]).to(dtype=R.dtype)
        k2 = _sarr_effective_factor(sym_v[idx, 2]).to(dtype=R.dtype)
        sarr[idx, 0] = torch.sin(alpha[idx] * k0)
        sarr[idx, 1] = torch.cos(alpha[idx] * k0)
        sarr[idx, 2] = torch.sin(beta[idx] * k1) * c_a[idx]
        sarr[idx, 3] = torch.cos(beta[idx] * k1)
        sarr[idx, 4] = torch.sin(gamma[idx] * k2) * c_a[idx] * c_b[idx]
        sarr[idx, 5] = torch.cos(gamma[idx] * k2)

    return sarr


def _clamp_abs_preserve_sign(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return torch.where(x >= 0, x.clamp(min=eps), x.clamp(max=-eps))


def sarr_to_rotation_matrix_old(sarr: torch.Tensor, sym_v: torch.Tensor) -> torch.Tensor:
    """
    Decode 6D SARR vectors with the previous stable-acos branch implementation.
    Kept for comparisons against the GPU-friendlier sarr_to_rotation_matrix.
    """
    if sarr.shape[-1] != 6:
        raise ValueError(f"Expected (..., 6), got {tuple(sarr.shape)}")

    flat_sarr = normalize_sarr_pairs(sarr).reshape(-1, 6)
    flat_sym_v = sym_v.reshape(-1, 3).to(device=flat_sarr.device)
    sa, ca, sb, cb, sg, cg = flat_sarr.unbind(dim=-1)
    ca = ca.clamp(-1.0, 1.0)
    cb = cb.clamp(-1.0, 1.0)
    cg = cg.clamp(-1.0, 1.0)

    alpha = torch.zeros_like(ca)
    beta = torch.zeros_like(cb)
    gamma = torch.zeros_like(cg)

    def stable_acos(cos_term: torch.Tensor) -> torch.Tensor:
        return acos_linear_extrapolation(
            cos_term,
            bounds=(-DEFAULT_ACOS_BOUND, DEFAULT_ACOS_BOUND),
        )

    def signed_acos(sin_term: torch.Tensor, cos_term: torch.Tensor) -> torch.Tensor:
        acos_val = stable_acos(cos_term)
        return torch.where(sin_term < 0.0, 2.0 * math.pi - acos_val, acos_val)

    all_one = (flat_sym_v == 1).all(dim=-1)
    z_only = (flat_sym_v[:, 2] > 1) & (flat_sym_v[:, 0] == 1) & (flat_sym_v[:, 1] == 1)
    y_only = (flat_sym_v[:, 1] > 1) & (flat_sym_v[:, 0] == 1) & (flat_sym_v[:, 2] == 1)
    x_only = (flat_sym_v[:, 0] > 1) & (flat_sym_v[:, 1] == 1) & (flat_sym_v[:, 2] == 1)
    mixed = ~(all_one | z_only | y_only | x_only)
    unsupported = mixed & (flat_sym_v == 1).any(dim=-1)
    if unsupported.any():
        raise NotImplementedError("SARR mixed symmetry with a unit axis is not supported in this refactor.")

    if all_one.any():
        idx = all_one
        alpha[idx] = signed_acos(sa[idx], ca[idx])
        beta[idx] = signed_acos(sb[idx], cb[idx])
        gamma[idx] = signed_acos(sg[idx], cg[idx])

    if z_only.any():
        idx = z_only
        k = flat_sym_v[idx, 2].to(dtype=sarr.dtype)
        alpha[idx] = signed_acos(sa[idx], ca[idx])
        beta[idx] = signed_acos(sb[idx], cb[idx])
        gamma_base = signed_acos(sg[idx], cg[idx])
        gamma[idx] = torch.where(k == 1000, torch.zeros_like(gamma_base), gamma_base / k)

    if y_only.any():
        idx = y_only
        k = flat_sym_v[idx, 1].to(dtype=sarr.dtype)
        beta_neg = (2.0 * math.pi / k) - (stable_acos(cb[idx]) / k)
        beta_pos = stable_acos(cb[idx]) / k
        bf = torch.where(sb[idx] < 0.0, -torch.ones_like(sb[idx]), torch.ones_like(sb[idx]))
        beta[idx] = torch.where(sb[idx] < 0.0, beta_neg, beta_pos)
        alpha[idx] = signed_acos(sa[idx], ca[idx])
        gamma[idx] = signed_acos(sg[idx], cg[idx]) * bf

    if x_only.any():
        idx = x_only
        k = flat_sym_v[idx, 0].to(dtype=sarr.dtype)
        alpha_raw = stable_acos(ca[idx])
        alpha[idx] = torch.where(sa[idx] < 0.0, 2.0 * math.pi - (alpha_raw / k), alpha_raw / k)
        cos_alpha = _clamp_abs_preserve_sign(torch.cos(alpha[idx]))
        gamma_base = stable_acos(cg[idx])
        beta_base = stable_acos(cb[idx])
        gamma[idx] = torch.where((sg[idx] / cos_alpha) < 0.0, 2.0 * math.pi - gamma_base, gamma_base)
        beta[idx] = torch.where((sb[idx] / cos_alpha) < 0.0, 2.0 * math.pi - beta_base, beta_base)

    if mixed.any():
        idx = mixed
        k0 = flat_sym_v[idx, 0].to(dtype=sarr.dtype)
        k1 = flat_sym_v[idx, 1].to(dtype=sarr.dtype)
        k2 = flat_sym_v[idx, 2].to(dtype=sarr.dtype)

        alpha_raw = signed_acos(sa[idx], ca[idx])
        alpha[idx] = alpha_raw / k0

        cos_alpha = _clamp_abs_preserve_sign(torch.cos(alpha[idx]))
        beta_base = stable_acos(cb[idx])
        beta_raw = torch.where((sb[idx] / cos_alpha) < 0.0, 2.0 * math.pi - beta_base, beta_base)
        beta[idx] = beta_raw / k1

        cos_beta = torch.cos(beta[idx])
        denom = cos_beta * cos_alpha
        denom = torch.where(denom >= 0, denom.clamp(min=1e-8), denom.clamp(max=-1e-8))
        gamma_base = stable_acos(cg[idx])
        gamma_raw = torch.where((sg[idx] / denom) < 0.0, 2.0 * math.pi - gamma_base, gamma_base)
        gamma[idx] = torch.where(k2 == 1000, torch.zeros_like(gamma_raw), gamma_raw / k2)

    angles = torch.stack([alpha, beta, gamma], dim=-1)
    R = xyz_euler_to_matrix_torch(angles)
    return R.reshape(*sarr.shape[:-1], 3, 3)


def sarr_to_rotation_matrix(sarr: torch.Tensor, sym_v: torch.Tensor) -> torch.Tensor:
    """
    Decode 6D SARR vectors ordered as [s_a, c_a, s_b, c_b, s_g, c_g]
    into canonical rotation matrices.

    Network predictions are normalized per SARR column before decoding. For
    symmetry classes whose sine terms are scaled by nu factors, undo that
    scaling before applying atan2 so the inverse remains invariant to the
    normalization step.
    """
    if sarr.shape[-1] != 6:
        raise ValueError(f"Expected (..., 6), got {tuple(sarr.shape)}")

    flat_sarr = normalize_sarr_pairs(sarr).reshape(-1, 6)
    flat_sym_v = sym_v.reshape(-1, 3).to(device=flat_sarr.device)
    sa, ca, sb, cb, sg, cg = flat_sarr.unbind(dim=-1)
    ca = ca.clamp(-1.0, 1.0)
    cb = cb.clamp(-1.0, 1.0)
    cg = cg.clamp(-1.0, 1.0)

    two_pi = 2.0 * math.pi

    def angle_mod(sin_term: torch.Tensor, cos_term: torch.Tensor) -> torch.Tensor:
        return torch.remainder(torch.atan2(sin_term, cos_term), two_pi)

    def angle_abs(sin_term: torch.Tensor, cos_term: torch.Tensor) -> torch.Tensor:
        return torch.atan2(sin_term.abs(), cos_term)

    all_one = (flat_sym_v == 1).all(dim=-1)
    z_only = (flat_sym_v[:, 2] > 1) & (flat_sym_v[:, 0] == 1) & (flat_sym_v[:, 1] == 1)
    y_only = (flat_sym_v[:, 1] > 1) & (flat_sym_v[:, 0] == 1) & (flat_sym_v[:, 2] == 1)
    x_only = (flat_sym_v[:, 0] > 1) & (flat_sym_v[:, 1] == 1) & (flat_sym_v[:, 2] == 1)
    mixed = ~(all_one | z_only | y_only | x_only)
    unsupported = mixed & (flat_sym_v == 1).any(dim=-1)
    if unsupported.any():
        raise NotImplementedError("SARR mixed symmetry with a unit axis is not supported in this refactor.")

    k0 = flat_sym_v[:, 0].to(dtype=sarr.dtype)
    k1 = flat_sym_v[:, 1].to(dtype=sarr.dtype)
    k2 = flat_sym_v[:, 2].to(dtype=sarr.dtype)

    alpha_plain = angle_mod(sa, ca)
    beta_plain = angle_mod(sb, cb)
    gamma_plain = angle_mod(sg, cg)

    gamma_z = torch.where(k2 == 1000, torch.zeros_like(gamma_plain), gamma_plain / k2)

    beta_y_abs = angle_abs(sb, cb) / k1
    beta_y = torch.where(sb < 0.0, (two_pi / k1) - beta_y_abs, beta_y_abs)
    cos_beta_y = _clamp_abs_preserve_sign(torch.cos(beta_y))
    gamma_y = angle_mod(sg / cos_beta_y, cg)

    alpha_x_abs = angle_abs(sa, ca) / k0
    alpha_x = torch.where(sa < 0.0, two_pi - alpha_x_abs, alpha_x_abs)
    cos_alpha_x = _clamp_abs_preserve_sign(torch.cos(alpha_x))
    beta_x = angle_mod(sb / cos_alpha_x, cb)
    gamma_x = angle_mod(sg / cos_alpha_x, cg)

    alpha_mixed = alpha_plain / k0
    cos_alpha_mixed = _clamp_abs_preserve_sign(torch.cos(alpha_mixed))
    beta_mixed = angle_mod(sb / cos_alpha_mixed, cb) / k1
    cos_beta_mixed = torch.cos(beta_mixed)
    denom_mixed = cos_beta_mixed * cos_alpha_mixed
    denom_mixed = torch.where(
        denom_mixed >= 0,
        denom_mixed.clamp(min=1e-8),
        denom_mixed.clamp(max=-1e-8),
    )
    gamma_mixed_raw = angle_mod(sg / denom_mixed, cg)
    gamma_mixed = torch.where(k2 == 1000, torch.zeros_like(gamma_mixed_raw), gamma_mixed_raw / k2)

    alpha = torch.where(x_only, alpha_x, torch.where(mixed, alpha_mixed, alpha_plain))
    beta = torch.where(
        y_only,
        beta_y,
        torch.where(x_only, beta_x, torch.where(mixed, beta_mixed, beta_plain)),
    )
    gamma = torch.where(
        z_only,
        gamma_z,
        torch.where(y_only, gamma_y, torch.where(x_only, gamma_x, torch.where(mixed, gamma_mixed, gamma_plain))),
    )

    angles = torch.stack([alpha, beta, gamma], dim=-1)
    R = xyz_euler_to_matrix_torch(angles)
    return R.reshape(*sarr.shape[:-1], 3, 3)
