#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from util.cad_bbox import (  # noqa: E402
    BBOX_3D_EDGES,
    CORNER_ORDER_DESCRIPTION,
    compute_axis_aligned_bbox_keypoints,
    load_mesh_vertices,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Load CAD mesh(es), compute the 8 keypoints of the 3D axis-aligned "
            "bounding box, save keypoints to file, and export lightweight visualization artifacts."
        )
    )
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument("--mesh_path", type=Path, help="Path to a single CAD mesh file.")
    source_group.add_argument(
        "--models_dir",
        type=Path,
        help="Directory containing CAD meshes. One keypoint file will be exported per mesh.",
    )

    parser.add_argument(
        "--pattern",
        type=str,
        default="obj_*.ply",
        help="Glob pattern for --models_dir (default: obj_*.ply).",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        required=True,
        help="Directory where keypoint and visualization files are saved.",
    )
    parser.add_argument(
        "--save_npy",
        action="store_true",
        help="Also save keypoints as .npy arrays in addition to JSON.",
    )
    parser.add_argument(
        "--image_size",
        type=int,
        default=900,
        help="Output SVG canvas size in pixels (default: 900).",
    )
    parser.add_argument(
        "--max_mesh_points",
        type=int,
        default=12000,
        help="Max sampled mesh points used for SVG background points (default: 12000).",
    )
    parser.add_argument(
        "--view_angles_deg",
        type=float,
        nargs=3,
        default=[25.0, -35.0, 0.0],
        metavar=("RX", "RY", "RZ"),
        help="Euler view angles (degrees) for visualization projection.",
    )
    return parser.parse_args()


def build_rotation_matrix_xyz(rx_deg: float, ry_deg: float, rz_deg: float) -> np.ndarray:
    rx = np.deg2rad(rx_deg)
    ry = np.deg2rad(ry_deg)
    rz = np.deg2rad(rz_deg)

    cx, sx = np.cos(rx), np.sin(rx)
    cy, sy = np.cos(ry), np.sin(ry)
    cz, sz = np.cos(rz), np.sin(rz)

    rx_mat = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]], dtype=np.float32)
    ry_mat = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=np.float32)
    rz_mat = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], dtype=np.float32)
    return rz_mat @ ry_mat @ rx_mat


def build_canvas_transform(
    reference_points_3d: np.ndarray, rotation_mat: np.ndarray, image_size: int, margin: int
) -> tuple[np.ndarray, float]:
    rotated = reference_points_3d @ rotation_mat.T
    xy = rotated[:, :2]
    min_xy = xy.min(axis=0)
    max_xy = xy.max(axis=0)
    span = np.maximum(max_xy - min_xy, 1e-6)

    scale_x = (image_size - 2 * margin) / span[0]
    scale_y = (image_size - 2 * margin) / span[1]
    scale = float(min(scale_x, scale_y))
    return min_xy, scale


def project_to_canvas(
    points_3d: np.ndarray,
    rotation_mat: np.ndarray,
    image_size: int,
    margin: int,
    min_xy: np.ndarray,
    scale: float,
) -> tuple[np.ndarray, np.ndarray]:
    rotated = points_3d @ rotation_mat.T
    xy = rotated[:, :2]
    z = rotated[:, 2]

    canvas_xy = (xy - min_xy) * scale + margin
    canvas_xy[:, 1] = image_size - canvas_xy[:, 1]
    return canvas_xy, z


def render_bbox_visualization_svg(
    vertices: np.ndarray,
    corners_3d: np.ndarray,
    image_size: int,
    max_mesh_points: int,
    rotation_mat: np.ndarray,
) -> str:
    margin = 70
    min_xy, scale = build_canvas_transform(
        reference_points_3d=corners_3d, rotation_mat=rotation_mat, image_size=image_size, margin=margin
    )

    if len(vertices) > max_mesh_points:
        idx = np.linspace(0, len(vertices) - 1, num=max_mesh_points, dtype=np.int64)
        vertices_vis = vertices[idx]
    else:
        vertices_vis = vertices

    verts_2d, verts_z = project_to_canvas(
        vertices_vis,
        rotation_mat=rotation_mat,
        image_size=image_size,
        margin=margin,
        min_xy=min_xy,
        scale=scale,
    )
    corners_2d, _ = project_to_canvas(
        corners_3d,
        rotation_mat=rotation_mat,
        image_size=image_size,
        margin=margin,
        min_xy=min_xy,
        scale=scale,
    )
    corners_2d_i = np.round(corners_2d).astype(np.int32)

    svg_lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{image_size}" height="{image_size}" viewBox="0 0 {image_size} {image_size}">',
        f'<rect width="{image_size}" height="{image_size}" fill="#f7f7f7"/>',
        '<text x="20" y="35" fill="#222" font-size="24" font-family="monospace">3D BBox keypoints (0-7)</text>',
    ]
    draw_order = np.argsort(verts_z)
    for i in draw_order:
        x, y = verts_2d[i]
        svg_lines.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="1" fill="#bbbbbb" />')

    for a, b in BBOX_3D_EDGES:
        x1, y1 = corners_2d_i[a]
        x2, y2 = corners_2d_i[b]
        svg_lines.append(
            f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="#28b43c" stroke-width="2" />'
        )

    for i, (x, y) in enumerate(corners_2d_i):
        svg_lines.append(f'<circle cx="{int(x)}" cy="{int(y)}" r="5" fill="#1f34d9" />')
        svg_lines.append(
            f'<text x="{int(x) + 8}" y="{int(y) - 8}" fill="#111" font-size="16" font-family="monospace">{i}</text>'
        )
    svg_lines.append("</svg>")
    return "\n".join(svg_lines)


def write_bbox_wireframe_obj(path: Path, corners_3d: np.ndarray) -> None:
    lines = ["# 3D bounding box wireframe", "# vertices (8 keypoints)"]
    for x, y, z in corners_3d:
        lines.append(f"v {x:.8f} {y:.8f} {z:.8f}")
    lines.append("# edges")
    for a, b in BBOX_3D_EDGES:
        # OBJ is 1-based indexing.
        lines.append(f"l {a + 1} {b + 1}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def collect_mesh_paths(args: argparse.Namespace) -> list[Path]:
    if args.mesh_path is not None:
        if not args.mesh_path.is_file():
            raise FileNotFoundError(f"--mesh_path not found: {args.mesh_path}")
        return [args.mesh_path]

    if args.models_dir is None or not args.models_dir.is_dir():
        raise FileNotFoundError(f"--models_dir not found: {args.models_dir}")

    mesh_paths = sorted(args.models_dir.glob(args.pattern))
    if not mesh_paths:
        raise FileNotFoundError(
            f"No CAD meshes found in {args.models_dir} with pattern '{args.pattern}'."
        )
    return mesh_paths


def build_payload(mesh_path: Path, corners_3d: np.ndarray) -> dict:
    min_xyz = corners_3d.min(axis=0)
    max_xyz = corners_3d.max(axis=0)
    return {
        "mesh_path": str(mesh_path.resolve()),
        "bbox_type": "axis_aligned",
        "corner_order": CORNER_ORDER_DESCRIPTION,
        "min_xyz": min_xyz.tolist(),
        "max_xyz": max_xyz.tolist(),
        "size_xyz": (max_xyz - min_xyz).tolist(),
        "keypoints_3d": corners_3d.tolist(),
    }


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    mesh_paths = collect_mesh_paths(args)
    rotation_mat = build_rotation_matrix_xyz(*args.view_angles_deg)

    manifest = []
    for mesh_path in mesh_paths:
        vertices = load_mesh_vertices(mesh_path)
        corners_3d = compute_axis_aligned_bbox_keypoints(vertices)

        stem = mesh_path.stem
        json_path = args.output_dir / f"{stem}_bbox_keypoints.json"
        vis_svg_path = args.output_dir / f"{stem}_bbox_keypoints_vis.svg"
        wireframe_obj_path = args.output_dir / f"{stem}_bbox_wireframe.obj"

        payload = build_payload(mesh_path, corners_3d)
        with json_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

        if args.save_npy:
            np.save(args.output_dir / f"{stem}_bbox_keypoints.npy", corners_3d)

        vis_svg = render_bbox_visualization_svg(
            vertices=vertices,
            corners_3d=corners_3d,
            image_size=args.image_size,
            max_mesh_points=args.max_mesh_points,
            rotation_mat=rotation_mat,
        )
        vis_svg_path.write_text(vis_svg, encoding="utf-8")
        write_bbox_wireframe_obj(wireframe_obj_path, corners_3d)

        manifest.append(
            {
                "mesh_path": str(mesh_path.resolve()),
                "keypoints_json": str(json_path.resolve()),
                "visualization_svg": str(vis_svg_path.resolve()),
                "wireframe_obj": str(wireframe_obj_path.resolve()),
            }
        )
        print(
            f"[OK] {mesh_path.name} -> {json_path.name}, {vis_svg_path.name}, {wireframe_obj_path.name}"
        )

    manifest_path = args.output_dir / "bbox_keypoints_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"[DONE] Exported {len(mesh_paths)} object(s). Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
