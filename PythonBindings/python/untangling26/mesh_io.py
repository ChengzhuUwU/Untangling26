"""Mesh loading and export, with one shared world-to-solver transform."""

from dataclasses import dataclass
from pathlib import Path
import re

import numpy as np
import trimesh


INPUT_FORMATS = {".obj", ".ply", ".stl", ".off", ".glb", ".gltf"}
OUTPUT_FORMATS = {".obj", ".glb"}


@dataclass
class Body:
    name: str
    vertices: np.ndarray
    faces: np.ndarray
    source: str


def validate_mesh(vertices, faces, name):
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or len(vertices) == 0:
        raise ValueError(f"{name}: expected a nonempty (N, 3) vertex array")
    if not np.isfinite(vertices).all():
        raise ValueError(f"{name}: vertices contain NaN or infinity")
    if faces.ndim != 2 or faces.shape[1] != 3 or len(faces) == 0:
        raise ValueError(f"{name}: expected a nonempty triangle mesh")
    if not np.issubdtype(faces.dtype, np.integer):
        raise ValueError(f"{name}: triangle indices must be integers")
    if faces.min() < 0 or faces.max() >= len(vertices):
        raise ValueError(f"{name}: triangle index is outside the vertex array")
    if len(vertices) > np.iinfo(np.int32).max:
        raise ValueError(f"{name}: too many vertices for the solver")
    return np.ascontiguousarray(vertices), np.ascontiguousarray(faces, dtype=np.int32)


def split_components(vertices, faces):
    """Split by shared vertex indices without requiring scipy or networkx."""
    parents = np.arange(len(vertices))

    def root(v):
        while parents[v] != v:
            parents[v] = parents[parents[v]]
            v = parents[v]
        return int(v)

    for a, b, c in faces:
        ra = root(a)
        parents[root(b)] = ra
        parents[root(c)] = ra
    groups = {}
    for i, face in enumerate(faces):
        groups.setdefault(root(face[0]), []).append(i)
    for indices in groups.values():
        selected = faces[indices]
        used, inverse = np.unique(selected, return_inverse=True)
        yield vertices[used], inverse.reshape(-1, 3).astype(np.int32)


def load_bodies(paths, *, split=False):
    bodies = []
    for path in paths:
        path = Path(path).resolve()
        if not path.is_file():
            raise ValueError(f"Input mesh not found: {path}")
        if path.suffix.lower() not in INPUT_FORMATS:
            raise ValueError(f"Unsupported input format: {path.suffix}")
        scene = trimesh.load_scene(path, process=False, group_material=False,
                                   split_objects=True, skip_materials=True,
                                   maintain_order=True)
        mesh_count = 0
        for node in scene.graph.nodes_geometry:
            transform, geometry_name = scene.graph[node]
            geometry = scene.geometry[geometry_name]
            if not isinstance(geometry, trimesh.Trimesh):
                raise ValueError(f"{path}: scene contains non-triangle geometry {geometry_name!r}")
            mesh = geometry.copy()
            # Keep OBJ position connectivity across UV/normal seams, but do not
            # give unreferenced vertices zero-mass solver degrees of freedom.
            mesh.remove_unreferenced_vertices()
            # STL stores a separate copy of each vertex in every triangle.
            if path.suffix.lower() == ".stl":
                mesh.merge_vertices()
            mesh.apply_transform(transform)
            name = re.sub(r"[^\w.-]+", "_", f"{path.stem}_{node}").strip("._") or "body"
            vertices, faces = validate_mesh(mesh.vertices, mesh.faces, name)
            pieces = split_components(vertices, faces) if split else [(vertices, faces)]
            for v, f in pieces:
                bodies.append(Body(f"{len(bodies):03d}_{name}", v, f, str(path)))
                mesh_count += 1
        if not mesh_count:
            raise ValueError(f"{path}: no triangle meshes found")
    return bodies


def normalize_bodies(bodies):
    lower = np.min([b.vertices.min(axis=0) for b in bodies], axis=0)
    upper = np.max([b.vertices.max(axis=0) for b in bodies], axis=0)
    center = lower * 0.5 + upper * 0.5
    scale = float(np.max(upper - lower))
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("The scene must have a finite, nonzero extent")
    normalized = [Body(b.name, (b.vertices - center) / scale, b.faces, b.source) for b in bodies]
    return normalized, center, scale


def output_paths(destination, bodies):
    destination = Path(destination).resolve()
    if destination.suffix.lower() in OUTPUT_FORMATS:
        return [destination], destination.with_suffix(".json")
    if destination.suffix and not destination.is_dir():
        raise ValueError("Output must be an .obj, a .glb, or a directory")
    meshes = [destination / "resolved.obj"]
    if len(bodies) > 1:
        meshes.extend(destination / f"{b.name}.obj" for b in bodies)
    return meshes, destination / "summary.json"


def check_outputs(paths, inputs, *, overwrite=False):
    input_paths = {Path(p).resolve() for p in inputs}
    for path in paths:
        if path.resolve() in input_paths:
            raise ValueError(f"Output would overwrite an input mesh: {path}")
        if path.exists() and (not overwrite or not path.is_file()):
            raise ValueError(f"Output already exists: {path}; choose another path or use --overwrite")


def write_mesh(path, bodies):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".obj":
        with path.open("w", encoding="utf-8", newline="\n") as handle:
            handle.write("# Repaired by untangling26; coordinates in original input units\n")
            offset = 1
            for body in bodies:
                handle.write(f"o {body.name}\n")
                np.savetxt(handle, body.vertices, fmt="v %.17g %.17g %.17g")
                np.savetxt(handle, body.faces.astype(np.int64) + offset, fmt="f %d %d %d")
                offset += len(body.vertices)
    else:
        scene = trimesh.Scene()
        for body in bodies:
            scene.add_geometry(trimesh.Trimesh(body.vertices, body.faces, process=False),
                               node_name=body.name, geom_name=body.name)
        path.write_bytes(scene.export(file_type="glb"))
