import math

import numpy as np


def _as_vertices(vertices):
    return np.asarray(vertices, dtype=np.float64)


def _as_faces(faces):
    return np.asarray(faces, dtype=np.int32)


def _safe_percentile(values, q):
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")
    return float(np.percentile(arr, q))


def _bbox_diag(vertices):
    verts = _as_vertices(vertices)
    if verts.size == 0:
        return 1.0
    return max(float(np.linalg.norm(verts.max(axis=0) - verts.min(axis=0))), 1e-12)


def _bbox_volume(vertices):
    verts = _as_vertices(vertices)
    if verts.size == 0:
        return 0.0
    extent = np.maximum(verts.max(axis=0) - verts.min(axis=0), 0.0)
    return float(extent[0] * extent[1] * extent[2])


def unique_edges(faces):
    tri = _as_faces(faces)
    if tri.size == 0:
        return np.zeros((0, 2), dtype=np.int32)
    edges = np.concatenate([
        tri[:, [0, 1]],
        tri[:, [1, 2]],
        tri[:, [2, 0]],
    ], axis=0)
    edges = np.sort(edges, axis=1)
    return np.unique(edges, axis=0)


def triangle_areas(vertices, faces):
    verts = _as_vertices(vertices)
    tri = _as_faces(faces)
    if tri.size == 0:
        return np.zeros((0,), dtype=np.float64)
    pts = verts[tri]
    cross = np.cross(pts[:, 1] - pts[:, 0], pts[:, 2] - pts[:, 0])
    return 0.5 * np.linalg.norm(cross, axis=1)


def face_normals(vertices, faces):
    verts = _as_vertices(vertices)
    tri = _as_faces(faces)
    if tri.size == 0:
        return np.zeros((0, 3), dtype=np.float64)
    pts = verts[tri]
    cross = np.cross(pts[:, 1] - pts[:, 0], pts[:, 2] - pts[:, 0])
    norm = np.linalg.norm(cross, axis=1, keepdims=True)
    safe_norm = np.where(norm > 1e-12, norm, 1.0)
    return cross / safe_norm


def edge_lengths(vertices, edges):
    verts = _as_vertices(vertices)
    edge_arr = np.asarray(edges, dtype=np.int32)
    if edge_arr.size == 0:
        return np.zeros((0,), dtype=np.float64)
    return np.linalg.norm(verts[edge_arr[:, 0]] - verts[edge_arr[:, 1]], axis=1)


def triangle_aspect_ratios(vertices, faces):
    verts = _as_vertices(vertices)
    tri = _as_faces(faces)
    if tri.size == 0:
        return np.zeros((0,), dtype=np.float64)
    pts = verts[tri]
    edge_len = np.stack([
        np.linalg.norm(pts[:, 1] - pts[:, 0], axis=1),
        np.linalg.norm(pts[:, 2] - pts[:, 1], axis=1),
        np.linalg.norm(pts[:, 0] - pts[:, 2], axis=1),
    ], axis=1)
    longest = edge_len.max(axis=1)
    areas = triangle_areas(vertices, faces)
    return longest * longest / np.maximum(2.0 * areas, 1e-12)


def compute_object_quality_metrics(reference_vertices, current_vertices, faces):
    ref_v = _as_vertices(reference_vertices)
    cur_v = _as_vertices(current_vertices)
    tri = _as_faces(faces)

    if ref_v.shape != cur_v.shape:
        raise ValueError("reference and current vertex arrays must have the same shape")

    ref_diag = _bbox_diag(ref_v)
    ref_edges = unique_edges(tri)
    ref_edge_len = edge_lengths(ref_v, ref_edges)
    cur_edge_len = edge_lengths(cur_v, ref_edges)
    edge_ratio = cur_edge_len / np.maximum(ref_edge_len, 1e-12)
    edge_abs_change_pct = np.abs(edge_ratio - 1.0) * 100.0

    ref_area = triangle_areas(ref_v, tri)
    cur_area = triangle_areas(cur_v, tri)
    area_ratio = cur_area / np.maximum(ref_area, 1e-12)
    area_abs_change_pct = np.abs(area_ratio - 1.0) * 100.0

    cur_aspect = triangle_aspect_ratios(cur_v, tri)
    ref_normals = face_normals(ref_v, tri)
    cur_normals = face_normals(cur_v, tri)
    normal_dot = np.einsum("ij,ij->i", ref_normals, cur_normals)
    normal_dot = np.clip(normal_dot, -1.0, 1.0)
    normal_deviation_deg = np.degrees(np.arccos(normal_dot))
    flip_mask = normal_dot < 0.0

    disp_norm = np.linalg.norm(cur_v - ref_v, axis=1) / ref_diag

    return {
        "vertex_count": int(ref_v.shape[0]),
        "face_count": int(tri.shape[0]),
        "reference_surface_area": float(ref_area.sum()),
        "current_surface_area": float(cur_area.sum()),
        "surface_area_ratio": float(cur_area.sum() / max(ref_area.sum(), 1e-12)),
        "reference_bbox_volume": _bbox_volume(ref_v),
        "current_bbox_volume": _bbox_volume(cur_v),
        "bbox_volume_ratio": float(_bbox_volume(cur_v) / max(_bbox_volume(ref_v), 1e-12)),
        "edge_length_ratio_mean": float(np.mean(edge_ratio)) if edge_ratio.size else float("nan"),
        "edge_length_abs_change_mean_pct": float(np.mean(edge_abs_change_pct)) if edge_abs_change_pct.size else float("nan"),
        "edge_length_abs_change_p95_pct": _safe_percentile(edge_abs_change_pct, 95.0),
        "face_area_abs_change_mean_pct": float(np.mean(area_abs_change_pct)) if area_abs_change_pct.size else float("nan"),
        "face_area_abs_change_p95_pct": _safe_percentile(area_abs_change_pct, 95.0),
        "triangle_aspect_ratio_mean": float(np.mean(cur_aspect)) if cur_aspect.size else float("nan"),
        "triangle_aspect_ratio_p95": _safe_percentile(cur_aspect, 95.0),
        "triangle_aspect_ratio_max": float(np.max(cur_aspect)) if cur_aspect.size else float("nan"),
        "face_flip_ratio_pct": float(np.mean(flip_mask) * 100.0) if flip_mask.size else 0.0,
        "normal_deviation_mean_deg": float(np.mean(normal_deviation_deg)) if normal_deviation_deg.size else float("nan"),
        "normal_deviation_p95_deg": _safe_percentile(normal_deviation_deg, 95.0),
        "normalized_vertex_disp_mean_pct": float(np.mean(disp_norm) * 100.0) if disp_norm.size else float("nan"),
        "normalized_vertex_disp_rms_pct": float(math.sqrt(np.mean(np.square(disp_norm))) * 100.0) if disp_norm.size else float("nan"),
        "normalized_vertex_disp_max_pct": float(np.max(disp_norm) * 100.0) if disp_norm.size else float("nan"),
        "_edge_length_ratio": edge_ratio,
        "_edge_length_abs_change_pct": edge_abs_change_pct,
        "_face_area_abs_change_pct": area_abs_change_pct,
        "_triangle_aspect_ratio": cur_aspect,
        "_normal_deviation_deg": normal_deviation_deg,
        "_normalized_vertex_disp": disp_norm,
        "_flip_count": int(np.count_nonzero(flip_mask)),
    }


def compute_case_geometry_metrics(reference_meshes, current_meshes):
    if len(reference_meshes) != len(current_meshes):
        raise ValueError("reference_meshes and current_meshes must have the same length")

    if not reference_meshes:
        return {
            "object_count": 0,
            "vertex_count": 0,
            "face_count": 0,
        }

    object_metrics = []
    edge_abs_changes = []
    area_abs_changes = []
    aspect_ratios = []
    normal_devs = []
    vertex_disps = []
    total_faces = 0
    total_flips = 0
    total_ref_area = 0.0
    total_cur_area = 0.0
    total_ref_bbox_volume = 0.0
    total_cur_bbox_volume = 0.0
    total_vertices = 0

    for ref_mesh, cur_mesh in zip(reference_meshes, current_meshes):
        faces = ref_mesh["faces"]
        metrics = compute_object_quality_metrics(ref_mesh["vertices"], cur_mesh["vertices"], faces)
        object_metrics.append({
            key: value for key, value in metrics.items() if not key.startswith("_")
        })

        edge_abs_changes.append(metrics["_edge_length_abs_change_pct"])
        area_abs_changes.append(metrics["_face_area_abs_change_pct"])
        aspect_ratios.append(metrics["_triangle_aspect_ratio"])
        normal_devs.append(metrics["_normal_deviation_deg"])
        vertex_disps.append(metrics["_normalized_vertex_disp"])
        total_faces += metrics["face_count"]
        total_flips += metrics["_flip_count"]
        total_vertices += metrics["vertex_count"]
        total_ref_area += metrics["reference_surface_area"]
        total_cur_area += metrics["current_surface_area"]
        total_ref_bbox_volume += metrics["reference_bbox_volume"]
        total_cur_bbox_volume += metrics["current_bbox_volume"]

    edge_abs_changes = np.concatenate([arr for arr in edge_abs_changes if arr.size], axis=0) if any(arr.size for arr in edge_abs_changes) else np.zeros((0,), dtype=np.float64)
    area_abs_changes = np.concatenate([arr for arr in area_abs_changes if arr.size], axis=0) if any(arr.size for arr in area_abs_changes) else np.zeros((0,), dtype=np.float64)
    aspect_ratios = np.concatenate([arr for arr in aspect_ratios if arr.size], axis=0) if any(arr.size for arr in aspect_ratios) else np.zeros((0,), dtype=np.float64)
    normal_devs = np.concatenate([arr for arr in normal_devs if arr.size], axis=0) if any(arr.size for arr in normal_devs) else np.zeros((0,), dtype=np.float64)
    vertex_disps = np.concatenate([arr for arr in vertex_disps if arr.size], axis=0) if any(arr.size for arr in vertex_disps) else np.zeros((0,), dtype=np.float64)

    return {
        "object_count": len(reference_meshes),
        "vertex_count": int(total_vertices),
        "face_count": int(total_faces),
        "surface_area_ratio": float(total_cur_area / max(total_ref_area, 1e-12)),
        "bbox_volume_ratio": float(total_cur_bbox_volume / max(total_ref_bbox_volume, 1e-12)),
        "edge_length_abs_change_mean_pct": float(np.mean(edge_abs_changes)) if edge_abs_changes.size else float("nan"),
        "edge_length_abs_change_p95_pct": _safe_percentile(edge_abs_changes, 95.0),
        "face_area_abs_change_mean_pct": float(np.mean(area_abs_changes)) if area_abs_changes.size else float("nan"),
        "face_area_abs_change_p95_pct": _safe_percentile(area_abs_changes, 95.0),
        "triangle_aspect_ratio_mean": float(np.mean(aspect_ratios)) if aspect_ratios.size else float("nan"),
        "triangle_aspect_ratio_p95": _safe_percentile(aspect_ratios, 95.0),
        "triangle_aspect_ratio_max": float(np.max(aspect_ratios)) if aspect_ratios.size else float("nan"),
        "face_flip_ratio_pct": float(total_flips / total_faces * 100.0) if total_faces > 0 else 0.0,
        "normal_deviation_mean_deg": float(np.mean(normal_devs)) if normal_devs.size else float("nan"),
        "normal_deviation_p95_deg": _safe_percentile(normal_devs, 95.0),
        "normalized_vertex_disp_mean_pct": float(np.mean(vertex_disps) * 100.0) if vertex_disps.size else float("nan"),
        "normalized_vertex_disp_rms_pct": float(math.sqrt(np.mean(np.square(vertex_disps))) * 100.0) if vertex_disps.size else float("nan"),
        "normalized_vertex_disp_max_pct": float(np.max(vertex_disps) * 100.0) if vertex_disps.size else float("nan"),
        "per_object": object_metrics,
    }