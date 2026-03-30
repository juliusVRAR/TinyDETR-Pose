from __future__ import annotations

from pathlib import Path
import struct

import numpy as np

# Corner order is kept consistent with util.visualize_object_pose.draw_cuboid_2d.
CORNER_ORDER_DESCRIPTION = [
    "0: (min_x, min_y, min_z)",
    "1: (max_x, min_y, min_z)",
    "2: (max_x, max_y, min_z)",
    "3: (min_x, max_y, min_z)",
    "4: (min_x, min_y, max_z)",
    "5: (max_x, min_y, max_z)",
    "6: (max_x, max_y, max_z)",
    "7: (min_x, max_y, max_z)",
]

BBOX_3D_EDGES = (
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 0),
    (4, 5),
    (5, 6),
    (6, 7),
    (7, 4),
    (0, 4),
    (1, 5),
    (2, 6),
    (3, 7),
)

_PLY_NUMPY_DTYPES = {
    "char": np.int8,
    "int8": np.int8,
    "uchar": np.uint8,
    "uint8": np.uint8,
    "short": np.int16,
    "int16": np.int16,
    "ushort": np.uint16,
    "uint16": np.uint16,
    "int": np.int32,
    "int32": np.int32,
    "uint": np.uint32,
    "uint32": np.uint32,
    "float": np.float32,
    "float32": np.float32,
    "double": np.float64,
    "float64": np.float64,
}


def _parse_ply_header(file_obj) -> tuple[str, int, list[tuple[str, str]]]:
    file_format = "ascii"
    n_vertices = None
    vertex_props: list[tuple[str, str]] = []
    in_vertex_section = False

    while True:
        raw = file_obj.readline()
        if not raw:
            raise ValueError("Invalid PLY file: missing end_header.")
        line = raw.decode("utf-8", errors="ignore").strip()
        if line.startswith("format "):
            file_format = line.split()[1]
        elif line.startswith("element vertex"):
            n_vertices = int(line.split()[-1])
            in_vertex_section = True
        elif line.startswith("element"):
            in_vertex_section = False
        elif in_vertex_section and line.startswith("property "):
            parts = line.split()
            if len(parts) >= 3 and parts[1] != "list":
                prop_type = parts[1]
                prop_name = parts[2]
                vertex_props.append((prop_name, prop_type))
        elif line == "end_header":
            break

    if n_vertices is None:
        raise ValueError("Invalid PLY file: missing vertex element.")
    if not vertex_props:
        raise ValueError("Invalid PLY file: no vertex properties found.")
    return file_format, n_vertices, vertex_props


def _load_vertices_from_ply(mesh_path: Path) -> np.ndarray:
    with mesh_path.open("rb") as f:
        file_format, n_vertices, vertex_props = _parse_ply_header(f)

        prop_names = [name for name, _ in vertex_props]
        if not {"x", "y", "z"}.issubset(prop_names):
            raise ValueError(f"PLY is missing x/y/z vertex properties: {mesh_path}")
        x_idx = prop_names.index("x")
        y_idx = prop_names.index("y")
        z_idx = prop_names.index("z")

        vertices = np.zeros((n_vertices, 3), dtype=np.float32)
        if file_format == "ascii":
            for i in range(n_vertices):
                while True:
                    raw = f.readline()
                    if not raw:
                        raise ValueError(f"Unexpected EOF while reading ASCII vertices in {mesh_path}")
                    line = raw.decode("utf-8", errors="ignore").strip()
                    if line:
                        break
                values = line.split()
                if len(values) <= max(x_idx, y_idx, z_idx):
                    raise ValueError(f"Malformed ASCII vertex line in {mesh_path}: '{line}'")
                vertices[i, 0] = float(values[x_idx])
                vertices[i, 1] = float(values[y_idx])
                vertices[i, 2] = float(values[z_idx])
            return vertices

        if file_format not in {"binary_little_endian", "binary_big_endian"}:
            raise ValueError(f"Unsupported PLY format '{file_format}' in {mesh_path}")
        endian = "<" if file_format == "binary_little_endian" else ">"

        struct_formats: list[tuple[str, str]] = []
        for _, prop_type in vertex_props:
            if prop_type not in _PLY_NUMPY_DTYPES:
                raise ValueError(f"Unsupported PLY property type '{prop_type}' in {mesh_path}")
            np_dtype = np.dtype(_PLY_NUMPY_DTYPES[prop_type])
            fmt_char = {
                1: "b" if np_dtype.kind == "i" else "B",
                2: "h" if np_dtype.kind == "i" else "H",
                4: "f" if np_dtype.kind == "f" else ("i" if np_dtype.kind == "i" else "I"),
                8: "d" if np_dtype.kind == "f" else "q",
            }.get(np_dtype.itemsize)
            if fmt_char is None:
                raise ValueError(f"Unsupported PLY dtype size '{np_dtype.itemsize}' in {mesh_path}")
            if np_dtype.kind == "u" and np_dtype.itemsize == 8:
                fmt_char = "Q"
            struct_formats.append((fmt_char, prop_type))

        for i in range(n_vertices):
            values = []
            for fmt_char, _ in struct_formats:
                size = struct.calcsize(fmt_char)
                raw = f.read(size)
                if len(raw) != size:
                    raise ValueError(f"Unexpected EOF while reading vertices in {mesh_path}")
                values.append(struct.unpack(endian + fmt_char, raw)[0])
            vertices[i, 0] = float(values[x_idx])
            vertices[i, 1] = float(values[y_idx])
            vertices[i, 2] = float(values[z_idx])
        return vertices


def _load_vertices_from_obj(mesh_path: Path) -> np.ndarray:
    vertices = []
    with mesh_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if line.startswith("v "):
                parts = line.strip().split()
                if len(parts) < 4:
                    continue
                vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
    if not vertices:
        raise ValueError(f"OBJ file has no vertices: {mesh_path}")
    return np.asarray(vertices, dtype=np.float32)


def load_mesh_vertices(mesh_path: str | Path) -> np.ndarray:
    mesh_path = Path(mesh_path)
    if not mesh_path.is_file():
        raise FileNotFoundError(f"CAD mesh not found: {mesh_path}")

    suffix = mesh_path.suffix.lower()
    if suffix == ".ply":
        vertices = _load_vertices_from_ply(mesh_path)
    elif suffix == ".obj":
        vertices = _load_vertices_from_obj(mesh_path)
    else:
        raise ValueError(
            f"Unsupported CAD format '{suffix}'. Supported formats: .ply, .obj"
        )

    if vertices.ndim != 2 or vertices.shape[1] != 3 or len(vertices) == 0:
        raise ValueError(f"CAD mesh has invalid vertex shape: {vertices.shape} ({mesh_path})")

    return vertices


def compute_axis_aligned_bbox_keypoints(vertices: np.ndarray) -> np.ndarray:
    vertices = np.asarray(vertices, dtype=np.float32)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or len(vertices) == 0:
        raise ValueError(f"Expected vertices with shape (N, 3), got {vertices.shape}")

    min_coords = vertices.min(axis=0)
    max_coords = vertices.max(axis=0)

    corners = np.array(
        [
            [min_coords[0], min_coords[1], min_coords[2]],
            [max_coords[0], min_coords[1], min_coords[2]],
            [max_coords[0], max_coords[1], min_coords[2]],
            [min_coords[0], max_coords[1], min_coords[2]],
            [min_coords[0], min_coords[1], max_coords[2]],
            [max_coords[0], min_coords[1], max_coords[2]],
            [max_coords[0], max_coords[1], max_coords[2]],
            [min_coords[0], max_coords[1], max_coords[2]],
        ],
        dtype=np.float32,
    )
    return corners
