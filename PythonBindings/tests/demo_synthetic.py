import json
import os
import sys
import numpy as np
try:
    import triangle as triangle_lib
except ImportError:
    triangle_lib = None

root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, os.path.join(root, 'build', 'bin'))
import lcs_py as lcs

from shared_args import create_parser, add_backend_args, add_headless_args
from utils.display_interface import DisplayInterface
from utils.mesh_proc import write_obj
from utils.sweep_guard import check_collision_buffer_budget, BufferBudgetExceeded
from utils.mesh_deformation import (
	BezierCurve3D,
	curve_deform,
	laplacian_smooth,
	save_curves_as_obj,
	_compute_rmf,
	_smoothstep_arr,
)

DEFAULT_FOLD_ORDER = ("left", "right", "front", "back")
SIDE_SPECS = {
	"left": {"axis_rank": 0, "side": -1, "rise_sign": 1.0},
	"right": {"axis_rank": 0, "side": 1, "rise_sign": -1.0},
	"front": {"axis_rank": 1, "side": 1, "rise_sign": 1.0},
	"back": {"axis_rank": 1, "side": -1, "rise_sign": -1.0},
}
AXIS_NAMES = {0: "x", 1: "y", 2: "z"}
REMESH_EDGE_TO_AREA_SCALE = 1.5 * np.sqrt(3.0) / 4.0



def _resolve_fold_sequence(args):
	if args.fold_sides:
		return list(args.fold_sides)
	count = int(np.clip(args.fold_side_count, 0, len(DEFAULT_FOLD_ORDER)))
	return list(DEFAULT_FOLD_ORDER[:count])


def _build_fold_band(vertices, axis, side, coverage, bbox_min, bbox_max):
	axis_min = float(bbox_min[axis])
	axis_max = float(bbox_max[axis])
	axis_span = max(axis_max - axis_min, 1e-6)
	if side > 0:
		seam_coord = axis_max - coverage * axis_span
		edge_coord = axis_max
		local_t = np.clip((vertices[:, axis] - seam_coord) / max(edge_coord - seam_coord, 1e-6), 0.0, 1.0)
		mask = vertices[:, axis] >= seam_coord
	else:
		seam_coord = axis_min + coverage * axis_span
		edge_coord = axis_min
		local_t = np.clip((seam_coord - vertices[:, axis]) / max(seam_coord - edge_coord, 1e-6), 0.0, 1.0)
		mask = vertices[:, axis] <= seam_coord
	band_length = max(abs(edge_coord - seam_coord), 1e-6)
	return mask, local_t, seam_coord, edge_coord, band_length, axis_span


def _compute_reference_normal(vertices, faces):
	triangles = np.asarray(vertices, dtype=np.float64)[np.asarray(faces, dtype=np.int32)]
	normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
	lengths = np.linalg.norm(normals, axis=1)
	valid = lengths > 1e-12
	if not np.any(valid):
		raise ValueError("Cannot remesh a degenerate mesh.")
	ref_normal = normals[valid][0]
	return ref_normal / np.linalg.norm(ref_normal)


def _project_planar_vertices(vertices, faces):
	verts = np.asarray(vertices, dtype=np.float64)
	faces_arr = np.asarray(faces, dtype=np.int32)
	triangles = verts[faces_arr]
	edge_a = triangles[:, 1] - triangles[:, 0]
	edge_b = triangles[:, 2] - triangles[:, 0]
	normals = np.cross(edge_a, edge_b)
	lengths = np.linalg.norm(normals, axis=1)
	valid_face_idx = np.flatnonzero(lengths > 1e-12)
	if valid_face_idx.size == 0:
		raise ValueError("Cannot project a degenerate mesh.")
	face_idx = int(valid_face_idx[0])
	origin = triangles[face_idx, 0]
	basis_u = edge_a[face_idx]
	basis_u /= np.linalg.norm(basis_u)
	ref_normal = normals[face_idx] / lengths[face_idx]
	basis_v = np.cross(ref_normal, basis_u)
	basis_v /= np.linalg.norm(basis_v)
	projected = np.column_stack(((verts - origin) @ basis_u, (verts - origin) @ basis_v))
	return projected, origin, basis_u, basis_v, ref_normal


def _compute_planar_mesh_area(projected_vertices, faces):
	triangles = np.asarray(projected_vertices, dtype=np.float64)[np.asarray(faces, dtype=np.int32)]
	twice_area = (
		(triangles[:, 1, 0] - triangles[:, 0, 0]) * (triangles[:, 2, 1] - triangles[:, 0, 1])
		- (triangles[:, 1, 1] - triangles[:, 0, 1]) * (triangles[:, 2, 0] - triangles[:, 0, 0])
	)
	return float(0.5 * np.abs(twice_area).sum())


def _compute_planar_remesh_characteristic_length(projected_vertices, faces):
	planar_area = _compute_planar_mesh_area(projected_vertices, faces)
	if planar_area <= 0.0:
		raise ValueError("Cannot remesh a zero-area planar mesh.")
	return float(((2.0 + np.sqrt(2.0)) / 3.0) * np.sqrt(planar_area))


def _format_triangle_option_number(value):
	formatted = f"{float(value):.16f}".rstrip("0").rstrip(".")
	return formatted if formatted else "0"


def _extract_boundary_loops(faces):
	faces_arr = np.asarray(faces, dtype=np.int32)
	undirected_edges = np.concatenate((faces_arr[:, [0, 1]], faces_arr[:, [1, 2]], faces_arr[:, [2, 0]]), axis=0)
	undirected_edges = np.sort(undirected_edges, axis=1)
	unique_edges, counts = np.unique(undirected_edges, axis=0, return_counts=True)
	boundary_edges = unique_edges[counts == 1]
	if boundary_edges.size == 0:
		raise ValueError("Planar remesh requires at least one boundary loop.")

	adjacency = {}
	for start, end in boundary_edges:
		start = int(start)
		end = int(end)
		adjacency.setdefault(start, []).append(end)
		adjacency.setdefault(end, []).append(start)
	for vertex_id, neighbors in adjacency.items():
		if len(neighbors) != 2:
			raise ValueError(
				f"Planar remesh expects closed manifold boundary loops, but boundary vertex {vertex_id} has degree {len(neighbors)}."
			)

	unused_edges = {tuple(int(v) for v in edge) for edge in boundary_edges.tolist()}
	loops = []
	while unused_edges:
		start, current = next(iter(unused_edges))
		loop = [start, current]
		unused_edges.remove(tuple(sorted((start, current))))
		while True:
			neighbors = adjacency[current]
			next_vertex = neighbors[0] if neighbors[0] != loop[-2] else neighbors[1]
			if next_vertex == start:
				unused_edges.discard(tuple(sorted((current, start))))
				break
			edge = tuple(sorted((current, next_vertex)))
			if edge not in unused_edges:
				raise ValueError("Failed to reconstruct a consistent boundary loop for constrained triangulation.")
			unused_edges.remove(edge)
			loop.append(next_vertex)
			current = next_vertex
		loops.append(loop)
	return loops


def _compute_loop_signed_area(projected_vertices, loop):
	loop_points = np.asarray(projected_vertices, dtype=np.float64)[np.asarray(loop, dtype=np.int32)]
	next_points = np.roll(loop_points, -1, axis=0)
	return float(0.5 * np.sum(loop_points[:, 0] * next_points[:, 1] - next_points[:, 0] * loop_points[:, 1]))


def _build_triangle_pslg(projected_vertices, faces):
	loops = _extract_boundary_loops(faces)
	loop_areas = [_compute_loop_signed_area(projected_vertices, loop) for loop in loops]
	outer_loop_index = int(np.argmax(np.abs(loop_areas)))

	boundary_vertices = []
	segments = []
	holes = []
	old_to_new = {}

	for loop_index, loop in enumerate(loops):
		is_outer_loop = loop_index == outer_loop_index
		loop_area = loop_areas[loop_index]
		ordered_loop = list(loop)
		if is_outer_loop and loop_area < 0.0:
			ordered_loop.reverse()
		if not is_outer_loop and loop_area > 0.0:
			ordered_loop.reverse()

		for vertex_id in ordered_loop:
			vertex_id = int(vertex_id)
			if vertex_id not in old_to_new:
				old_to_new[vertex_id] = len(boundary_vertices)
				boundary_vertices.append(np.asarray(projected_vertices, dtype=np.float64)[vertex_id])
		for edge_index in range(len(ordered_loop)):
			start = old_to_new[int(ordered_loop[edge_index])]
			end = old_to_new[int(ordered_loop[(edge_index + 1) % len(ordered_loop)])]
			segments.append([start, end])
		if not is_outer_loop:
			holes.append(np.asarray(projected_vertices, dtype=np.float64)[np.asarray(ordered_loop, dtype=np.int32)].mean(axis=0))

	pslg = {
		"vertices": np.asarray(boundary_vertices, dtype=np.float64),
		"segments": np.asarray(segments, dtype=np.int32),
	}
	if holes:
		pslg["holes"] = np.asarray(holes, dtype=np.float64)
	return pslg


def _orient_faces_like_reference(vertices, faces, ref_normal):
	oriented_faces = np.asarray(faces, dtype=np.int32).copy()
	triangles = np.asarray(vertices, dtype=np.float64)[oriented_faces]
	tri_normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
	flip_mask = (tri_normals @ ref_normal) < 0.0
	oriented_faces[flip_mask, 1] = oriented_faces[flip_mask, 2]
	oriented_faces[flip_mask, 2] = faces[flip_mask, 1]
	return oriented_faces


def remesh_planar_mesh_cdt(vertices, faces, max_triangle_area, min_triangle_angle_deg=30.0):
	projected_vertices, origin, basis_u, basis_v, ref_normal = _project_planar_vertices(vertices, faces)
	if max_triangle_area <= 0.0:
		raise ValueError("Constrained triangulation requires a positive max triangle area.")
	if min_triangle_angle_deg <= 0.0 or min_triangle_angle_deg >= 34.0:
		raise ValueError("Constrained triangulation min triangle angle must be in the open interval (0, 34) degrees.")
	triangle_input = _build_triangle_pslg(projected_vertices, faces)
	triangle_options = (
		f"pq{_format_triangle_option_number(min_triangle_angle_deg)}"
		f"a{_format_triangle_option_number(max_triangle_area)}D"
	)
	triangulation = triangle_lib.triangulate(triangle_input, triangle_options)
	if "vertices" not in triangulation or "triangles" not in triangulation:
		raise ValueError("Triangle constrained triangulation did not produce a valid mesh.")
	remesh_vertices_2d = np.asarray(triangulation["vertices"], dtype=np.float64)
	remesh_faces = np.asarray(triangulation["triangles"], dtype=np.int32)
	remesh_vertices = origin + np.outer(remesh_vertices_2d[:, 0], basis_u) + np.outer(remesh_vertices_2d[:, 1], basis_v)
	remesh_faces = _orient_faces_like_reference(remesh_vertices, remesh_faces, ref_normal)
	return remesh_vertices.astype(np.float64), remesh_faces.astype(np.int32)


def _compute_mesh_triangle_angle_stats(vertices, faces):
	faces_arr = np.asarray(faces, dtype=np.int32)
	if faces_arr.size == 0:
		raise ValueError("Remeshed mesh has no faces.")
	triangles = np.asarray(vertices, dtype=np.float64)[faces_arr]
	edge_ab = triangles[:, 1] - triangles[:, 0]
	edge_bc = triangles[:, 2] - triangles[:, 1]
	edge_ca = triangles[:, 0] - triangles[:, 2]
	len_ab = np.linalg.norm(edge_ab, axis=1)
	len_bc = np.linalg.norm(edge_bc, axis=1)
	len_ca = np.linalg.norm(edge_ca, axis=1)

	denom_a = np.maximum(len_ab * len_ca, 1e-12)
	denom_b = np.maximum(len_ab * len_bc, 1e-12)
	denom_c = np.maximum(len_bc * len_ca, 1e-12)
	angle_a = np.degrees(np.arccos(np.clip(np.sum(edge_ab * (-edge_ca), axis=1) / denom_a, -1.0, 1.0)))
	angle_b = np.degrees(np.arccos(np.clip(np.sum((-edge_ab) * edge_bc, axis=1) / denom_b, -1.0, 1.0)))
	angle_c = np.degrees(np.arccos(np.clip(np.sum((-edge_bc) * edge_ca, axis=1) / denom_c, -1.0, 1.0)))
	angles = np.stack((angle_a, angle_b, angle_c), axis=1)
	min_angles = np.min(angles, axis=1)
	return {
		"min_triangle_angle_deg": float(min_angles.min()),
		"avg_min_triangle_angle_deg": float(min_angles.mean()),
		"p05_min_triangle_angle_deg": float(np.quantile(min_angles, 0.05)),
		"max_triangle_angle_deg": float(angles.max()),
	}


def _compute_mesh_edge_stats(vertices, faces):
	faces_arr = np.asarray(faces, dtype=np.int32)
	if faces_arr.size == 0:
		raise ValueError("Remeshed mesh has no faces.")
	edges = np.concatenate((faces_arr[:, [0, 1]], faces_arr[:, [1, 2]], faces_arr[:, [2, 0]]), axis=0)
	edges = np.sort(edges, axis=1)
	unique_edges = np.unique(edges, axis=0)
	edge_lengths = np.linalg.norm(
		np.asarray(vertices, dtype=np.float64)[unique_edges[:, 0]] - np.asarray(vertices, dtype=np.float64)[unique_edges[:, 1]],
		axis=1,
	)
	return {
		"edge_count": int(edge_lengths.size),
		"avg_edge_length": float(edge_lengths.mean()),
		"min_edge_length": float(edge_lengths.min()),
		"max_edge_length": float(edge_lengths.max()),
	}


def _remesh_planar_mesh_with_stats(vertices, faces, frequency, min_triangle_angle_deg=30.0):
	requested_frequency = max(1, int(frequency))
	projected_vertices, _, _, _, _ = _project_planar_vertices(vertices, faces)
	target_edge_length = _compute_planar_remesh_characteristic_length(projected_vertices, faces) / requested_frequency
	max_triangle_area = REMESH_EDGE_TO_AREA_SCALE * target_edge_length * target_edge_length
	remesh_vertices, remesh_faces = remesh_planar_mesh_cdt(
		vertices,
		faces,
		max_triangle_area=max_triangle_area,
		min_triangle_angle_deg=min_triangle_angle_deg,
	)
	edge_stats = _compute_mesh_edge_stats(remesh_vertices, remesh_faces)
	triangle_quality_stats = _compute_mesh_triangle_angle_stats(remesh_vertices, remesh_faces)
	edge_stats["frequency"] = requested_frequency
	edge_stats["num_vertices"] = int(remesh_vertices.shape[0])
	edge_stats["num_faces"] = int(remesh_faces.shape[0])
	edge_stats["max_triangle_area"] = float(max_triangle_area)
	edge_stats["target_avg_edge_length"] = float(target_edge_length)
	edge_stats["target_min_triangle_angle_deg"] = float(min_triangle_angle_deg)
	edge_stats.update(triangle_quality_stats)
	return remesh_vertices, remesh_faces, edge_stats


def _resolve_planar_remesh(vertices, faces, requested_frequency, target_avg_edge_length=None, max_frequency=4096, min_triangle_angle_deg=30.0):
	max_frequency = max(1, int(max_frequency))
	base_frequency = max(1, min(int(requested_frequency), max_frequency))
	cache = {}

	def evaluate(frequency):
		frequency = max(1, min(int(frequency), max_frequency))
		if frequency not in cache:
			cache[frequency] = _remesh_planar_mesh_with_stats(
				vertices,
				faces,
				frequency,
				min_triangle_angle_deg=min_triangle_angle_deg,
			)
		return cache[frequency]

	base_vertices, base_faces, base_stats = evaluate(base_frequency)
	base_stats["selection_mode"] = "frequency"

	if target_avg_edge_length is None:
		return base_vertices, base_faces, base_stats

	target_avg_edge_length = float(target_avg_edge_length)
	if target_avg_edge_length <= 0.0:
		raise ValueError("Target average edge length must be positive.")

	if base_stats["avg_edge_length"] <= target_avg_edge_length:
		low = 1
		low_vertices, low_faces, low_stats = evaluate(low)
		if low_stats["avg_edge_length"] <= target_avg_edge_length:
			selected_vertices, selected_faces, selected_stats = low_vertices, low_faces, low_stats
		else:
			high = base_frequency
			while low + 1 < high:
				mid = (low + high) // 2
				_, _, mid_stats = evaluate(mid)
				if mid_stats["avg_edge_length"] <= target_avg_edge_length:
					high = mid
				else:
					low = mid
			selected_vertices, selected_faces, selected_stats = evaluate(high)
	else:
		low = base_frequency
		high = max(low + 1, int(np.ceil(low * base_stats["avg_edge_length"] / target_avg_edge_length)))
		high = min(high, max_frequency)
		selected_vertices, selected_faces, selected_stats = evaluate(high)
		while selected_stats["avg_edge_length"] > target_avg_edge_length and high < max_frequency:
			low = high
			next_high = max(high + 1, int(np.ceil(high * selected_stats["avg_edge_length"] / target_avg_edge_length)))
			high = min(next_high, max_frequency)
			selected_vertices, selected_faces, selected_stats = evaluate(high)
		if selected_stats["avg_edge_length"] > target_avg_edge_length:
			raise ValueError(
				f"Could not reach target average edge length {target_avg_edge_length:.6f} m within max frequency {max_frequency}."
			)
		while low + 1 < high:
			mid = (low + high) // 2
			_, _, mid_stats = evaluate(mid)
			if mid_stats["avg_edge_length"] <= target_avg_edge_length:
				high = mid
			else:
				low = mid
		selected_vertices, selected_faces, selected_stats = evaluate(high)

	selected_stats["selection_mode"] = "target_avg_edge_length"
	selected_stats["target_avg_edge_length"] = float(target_avg_edge_length)
	return selected_vertices, selected_faces, selected_stats


def _build_length_preserved_deformed_curve(stage_data, eased_progress):
	raw_curve_keys = (1.0 - eased_progress) * stage_data["rest_keys"] + eased_progress * stage_data["final_keys"]
	raw_curve = BezierCurve3D.from_keypoints(raw_curve_keys, smoothness=stage_data["curve_smoothness"])
	raw_curve_length = float(raw_curve.arc_length(n_samples=512))
	rest_curve_length = float(stage_data["rest_curve_length"])
	if raw_curve_length > 1e-12:
		length_scale = rest_curve_length / raw_curve_length
	else:
		length_scale = 1.0
	curve_keys = stage_data["rest_start"] + length_scale * (raw_curve_keys - stage_data["rest_start"])
	deformed_curve = BezierCurve3D.from_keypoints(curve_keys, smoothness=stage_data["curve_smoothness"])
	curve_length = float(deformed_curve.arc_length(n_samples=512))
	return curve_keys, deformed_curve, {
		"rest_curve_length": rest_curve_length,
		"raw_curve_length": raw_curve_length,
		"curve_length": curve_length,
		"curve_length_scale": float(length_scale),
	}


def curve_deform_by_normalized_arc_length(vertices, rest_curve, deformed_curve, weights=None, sample_count=512):
	verts = np.array(vertices, dtype=np.float64)
	if weights is None:
		weights = np.ones(len(verts), dtype=np.float64)
	else:
		weights = np.asarray(weights, dtype=np.float64)

	rs = rest_curve.evaluate(0.0)
	re = rest_curve.evaluate(1.0)
	rd = re - rs
	rlen = np.linalg.norm(rd)
	if rlen < 1e-12:
		return verts
	rhat = rd / rlen

	ref = np.array([0.0, 1.0, 0.0]) if abs(rhat[1]) < 0.9 else np.array([1.0, 0.0, 0.0])
	U0 = np.cross(rhat, ref)
	U0 /= np.linalg.norm(U0)
	V0 = np.cross(rhat, U0)

	offsets = verts - rs[None, :]
	material_s = np.clip(offsets @ rhat / rlen, 0.0, 1.0)
	local_u = offsets @ U0
	local_v = offsets @ V0

	sample_count = max(4, int(sample_count))
	param_samples = np.linspace(0.0, 1.0, sample_count)
	curve_points = deformed_curve.evaluate_batch(param_samples)
	segment_lengths = np.linalg.norm(np.diff(curve_points, axis=0), axis=1)
	arc_lengths = np.concatenate(([0.0], np.cumsum(segment_lengths)))
	total_length = float(arc_lengths[-1])
	if total_length < 1e-12:
		return verts

	normalized_arc = arc_lengths / total_length
	unique_arc, unique_indices = np.unique(normalized_arc, return_index=True)
	normalized_samples = np.linspace(0.0, 1.0, sample_count)
	arc_length_params = np.interp(normalized_samples, unique_arc, param_samples[unique_indices])
	def_pos, def_T, def_U = _compute_rmf(deformed_curve, arc_length_params)
	def_V = np.cross(def_T, def_U)

	fi = material_s * (sample_count - 1)
	i0 = np.minimum(fi.astype(int), sample_count - 2)
	frac = (fi - i0)[:, None]
	p = (1.0 - frac) * def_pos[i0] + frac * def_pos[i0 + 1]
	ud = (1.0 - frac) * def_U[i0] + frac * def_U[i0 + 1]
	vd = (1.0 - frac) * def_V[i0] + frac * def_V[i0 + 1]

	ud_n = np.linalg.norm(ud, axis=1, keepdims=True)
	vd_n = np.linalg.norm(vd, axis=1, keepdims=True)
	ud = np.where(ud_n > 1e-12, ud / ud_n, ud)
	vd = np.where(vd_n > 1e-12, vd / vd_n, vd)

	new_pos = p + local_u[:, None] * ud + local_v[:, None] * vd
	w = weights[:, None]
	return (1.0 - w) * verts + w * new_pos


def _build_center_fold_stage(
	vertices,
	faces,
	side_name,
	fold_depth,
	smooth_iterations,
	flat_axis,
	span_axes,
	base_diag,
	fold_height_scale,
	stage_smooth_respect_weights=True,
):
	verts = np.array(vertices, dtype=np.float64, copy=True)
	faces_arr = np.asarray(faces, dtype=np.int32)
	bbox_min = verts.min(axis=0)
	bbox_max = verts.max(axis=0)
	center = (bbox_min + bbox_max) * 0.5
	fold_height_scale = float(max(fold_height_scale, 0.0))

	spec = SIDE_SPECS[side_name]
	deform_axis = span_axes[spec["axis_rank"]]
	side = int(spec["side"])

	coverage = float(np.clip(0.28 + 0.22 * fold_depth, 0.22, 0.65))
	mask, local_t, seam_coord, edge_coord, band_length, axis_span = _build_fold_band(
		verts,
		deform_axis,
		side,
		coverage,
		bbox_min,
		bbox_max,
	)
	if mask.sum() < 5:
		return None

	rest_start = center.copy()
	rest_end = center.copy()
	rest_start[deform_axis] = seam_coord
	rest_end[deform_axis] = edge_coord
	rest_curve = BezierCurve3D.make_line(rest_start, rest_end)

	curl_turns = float(np.clip(0.95 + 0.75 * fold_depth, 0.90, 1.50))
	curl_angle = np.pi * curl_turns
	curl_radius = band_length / max(curl_angle, 1e-6)
	center_pull = axis_span * float(np.clip(0.18 + 0.42 * fold_depth, 0.12, 0.48))
	center_pull_start_t = float(np.clip(0.82 - 0.10 * fold_depth, 0.70, 0.84))

	key_count = 9
	key_ts = np.linspace(0.0, 1.0, key_count)
	angles = key_ts * curl_angle
	center_pull_profile = _smoothstep_arr(
		np.clip((key_ts - center_pull_start_t) / max(1.0 - center_pull_start_t, 1e-6), 0.0, 1.0)
	)
	penetration_start_t = float(np.clip(0.21 - 0.10 * fold_depth, 0.10, 0.34))
	penetration_profile = _smoothstep_arr(
		np.clip((key_ts - penetration_start_t) / max(1.0 - penetration_start_t, 1e-6), 0.0, 1.0)
	)
	penetration_depth = float(base_diag * (0.04 + 0.14 * max(fold_depth, 0.0)))
	keys = np.repeat(rest_start[None, :], key_count, axis=0)
	keys[:, deform_axis] += (
		side * curl_radius * np.sin(angles)
		- side * center_pull * center_pull_profile
	)
	keys[:, flat_axis] += fold_height_scale * float(spec["rise_sign"]) * (
		curl_radius * (1.0 - np.cos(angles))
		- penetration_depth * penetration_profile
	)
	rest_keys = np.linspace(rest_start, rest_end, key_count)
	weights = mask.astype(np.float64)
	curve_smoothness = 0.7
	return {
		"reference_vertices": verts,
		"faces": faces_arr,
		"side_name": side_name,
		"axis": int(deform_axis),
		"axis_name": AXIS_NAMES[int(deform_axis)],
		"flat_axis": int(flat_axis),
		"flat_axis_name": AXIS_NAMES[int(flat_axis)],
		"side": side,
		"coverage": coverage,
		"seam_coord": float(seam_coord),
		"edge_coord": float(edge_coord),
		"band_length": float(band_length),
		"mask_count": int(mask.sum()),
		"rise_height": float(np.max(np.abs(keys[:, flat_axis] - rest_start[flat_axis]))),
		"inward_shift": float(center_pull),
		"fold_height_scale": float(fold_height_scale),
		"center_pull_start_t": float(center_pull_start_t),
		"penetration_depth": float(penetration_depth),
		"penetration_start_t": float(penetration_start_t),
		"flat_axis_min_offset": float(np.min(keys[:, flat_axis] - rest_start[flat_axis])),
		"flat_axis_max_offset": float(np.max(keys[:, flat_axis] - rest_start[flat_axis])),
		"curl_angle_deg": float(np.degrees(curl_angle)),
		"curl_radius": float(curl_radius),
		"local_t_min": float(local_t[mask].min()) if np.any(mask) else 0.0,
		"local_t_max": float(local_t[mask].max()) if np.any(mask) else 0.0,
		"rest_start": rest_start,
		"rest_end": rest_end,
		"rest_curve": rest_curve,
		"rest_curve_length": float(band_length),
		"rest_keys": rest_keys,
		"final_keys": keys,
		"weights": weights,
		"curve_smoothness": curve_smoothness,
		"smooth_iterations": max(0, int(smooth_iterations)),
		"smooth_lambda_factor": 0.18,
		"smooth_respect_weights": bool(stage_smooth_respect_weights),
	}


def _apply_center_fold_stage(stage_data, progress=1.0):
	progress = float(np.clip(progress, 0.0, 1.0))
	eased_progress = progress * progress * (3.0 - 2.0 * progress)
	reference_vertices = np.array(stage_data["reference_vertices"], dtype=np.float64, copy=True)
	curve_keys, deformed_curve, length_info = _build_length_preserved_deformed_curve(stage_data, eased_progress)
	if eased_progress <= 0.0:
		deformed_verts = reference_vertices
	else:
		deformed_verts = curve_deform_by_normalized_arc_length(
			reference_vertices,
			stage_data["rest_curve"],
			deformed_curve,
			weights=stage_data["weights"],
		)
		if stage_data["smooth_iterations"] > 0:
			smoothed_verts = laplacian_smooth(
				deformed_verts,
				stage_data["faces"],
				iterations=stage_data["smooth_iterations"],
				lambda_factor=stage_data["smooth_lambda_factor"] * eased_progress,
				pin_boundary=False,
			)
			if bool(stage_data.get("smooth_respect_weights", True)):
				stage_weights = np.asarray(stage_data["weights"], dtype=np.float64)[:, None]
				deformed_verts = reference_vertices + stage_weights * (smoothed_verts - reference_vertices)
			else:
				deformed_verts = smoothed_verts

	return deformed_verts, {
		"side_name": stage_data["side_name"],
		"axis": stage_data["axis"],
		"axis_name": stage_data["axis_name"],
		"flat_axis": stage_data["flat_axis"],
		"flat_axis_name": stage_data["flat_axis_name"],
		"side": stage_data["side"],
		"coverage": stage_data["coverage"],
		"seam_coord": stage_data["seam_coord"],
		"edge_coord": stage_data["edge_coord"],
		"band_length": stage_data["band_length"],
		"mask_count": stage_data["mask_count"],
		"rise_height": float(np.max(np.abs(curve_keys[:, stage_data["flat_axis"]] - stage_data["rest_start"][stage_data["flat_axis"]]))),
		"inward_shift": stage_data["inward_shift"] * eased_progress,
		"fold_height_scale": stage_data["fold_height_scale"],
		"center_pull_start_t": stage_data["center_pull_start_t"],
		"penetration_depth": stage_data["penetration_depth"] * eased_progress,
		"penetration_start_t": stage_data["penetration_start_t"],
		"flat_axis_min_offset": float(np.min(curve_keys[:, stage_data["flat_axis"]] - stage_data["rest_start"][stage_data["flat_axis"]])),
		"flat_axis_max_offset": float(np.max(curve_keys[:, stage_data["flat_axis"]] - stage_data["rest_start"][stage_data["flat_axis"]])),
		"curl_angle_deg": stage_data["curl_angle_deg"] * eased_progress,
		"curl_radius": stage_data["curl_radius"],
		"local_t_min": stage_data["local_t_min"],
		"local_t_max": stage_data["local_t_max"],
		"progress": progress,
		"eased_progress": eased_progress,
		"rest_curve_length": length_info["rest_curve_length"],
		"raw_curve_length": length_info["raw_curve_length"],
		"curve_length": length_info["curve_length"],
		"curve_length_scale": length_info["curve_length_scale"],
		"curve_length_ratio": length_info["curve_length"] / max(length_info["rest_curve_length"], 1e-12),
		"keypoints": curve_keys.tolist(),
	}


def _build_stage_progresses(stage_count, timeline_value):
	stage_count = max(0, int(stage_count))
	timeline_value = float(np.clip(timeline_value, 0.0, float(stage_count)))
	stage_progresses = np.zeros(stage_count, dtype=np.float64)
	completed_stage_count = min(int(np.floor(timeline_value)), stage_count)
	if completed_stage_count > 0:
		stage_progresses[:completed_stage_count] = 1.0
	if completed_stage_count < stage_count:
		stage_progresses[completed_stage_count] = timeline_value - completed_stage_count
	return stage_progresses


def _compose_fold_stage_vertices(reference_vertices, fold_stage_sequence, stage_progresses):
	reference_vertices = np.asarray(reference_vertices, dtype=np.float64)
	if len(fold_stage_sequence) != len(stage_progresses):
		raise ValueError("Stage progress count must match fold stage count.")
	accumulated_delta = np.zeros_like(reference_vertices)
	stage_infos = []
	for stage_data, progress in zip(fold_stage_sequence, stage_progresses):
		progress = float(np.clip(progress, 0.0, 1.0))
		if progress <= 0.0:
			continue
		stage_vertices, stage_info = _apply_center_fold_stage(stage_data, progress=progress)
		accumulated_delta += stage_vertices - np.asarray(stage_data["reference_vertices"], dtype=np.float64)
		stage_info["deform_index"] = int(stage_data["deform_index"])
		stage_info["cycle_index"] = int(stage_data["cycle_index"])
		stage_infos.append(stage_info)
	return reference_vertices + accumulated_delta, stage_infos


def _apply_final_fold_smoothing(vertices, faces, iterations=1, lambda_factor=0.03, pin_boundary=True):
	iterations = max(0, int(iterations))
	lambda_factor = float(lambda_factor)
	if iterations <= 0 or lambda_factor <= 0.0:
		return np.asarray(vertices, dtype=np.float64)
	return laplacian_smooth(
		np.asarray(vertices, dtype=np.float64),
		np.asarray(faces, dtype=np.int32),
		iterations=iterations,
		lambda_factor=lambda_factor,
		pin_boundary=bool(pin_boundary),
	)


def generate_ordered_center_folds(
	vertices,
	faces,
	fold_cycles,
	fold_depth,
	fold_sequence,
	smooth_iterations=1,
	fold_height_scale=1.12,
	stage_smooth_respect_weights=True,
	final_smooth_iterations=1,
	final_smooth_lambda_factor=0.03,
	final_smooth_pin_boundary=True,
):
	orig_verts = np.asarray(vertices, dtype=np.float64)
	bbox_size = orig_verts.max(axis=0) - orig_verts.min(axis=0)
	flat_axis = int(np.argmin(bbox_size))
	span_axes = [axis for axis in range(3) if axis != flat_axis]
	base_diag = float(np.linalg.norm(bbox_size))

	curves_info = []
	stage_sequence = []
	fold_index = 0
	for cycle_index in range(max(0, int(fold_cycles))):
		for side_name in fold_sequence:
			stage_data = _build_center_fold_stage(
				orig_verts,
				faces,
				side_name=side_name,
				fold_depth=fold_depth,
				smooth_iterations=smooth_iterations,
				flat_axis=flat_axis,
				span_axes=span_axes,
				base_diag=base_diag,
				fold_height_scale=fold_height_scale,
				stage_smooth_respect_weights=stage_smooth_respect_weights,
			)
			if stage_data is None:
				continue
			_, fold_info = _apply_center_fold_stage(stage_data, progress=1.0)
			fold_info["deform_index"] = fold_index
			fold_info["cycle_index"] = cycle_index
			stage_data["deform_index"] = fold_index
			stage_data["cycle_index"] = cycle_index
			curves_info.append(fold_info)
			stage_sequence.append(stage_data)
			fold_index += 1
	stage_progresses = np.ones(len(stage_sequence), dtype=np.float64)
	deformed_verts, _ = _compose_fold_stage_vertices(orig_verts, stage_sequence, stage_progresses)
	deformed_verts = _apply_final_fold_smoothing(
		deformed_verts,
		faces,
		iterations=final_smooth_iterations,
		lambda_factor=final_smooth_lambda_factor,
		pin_boundary=final_smooth_pin_boundary,
	)
	return deformed_verts, curves_info, flat_axis, span_axes, stage_sequence


def save_deformation_debug(output_dir, deformation_info, fold_sequence, flat_axis, span_axes):
	info_path = os.path.join(output_dir, "deformation_info.json")
	with open(info_path, "w", encoding="utf-8") as handle:
		json.dump(
			{
				"fold_sequence": list(fold_sequence),
				"flat_axis": AXIS_NAMES[int(flat_axis)],
				"span_axes": [AXIS_NAMES[int(axis)] for axis in span_axes],
				"folds": deformation_info,
			},
			handle,
			indent=2,
		)
	print(f"Saved deformation info to {info_path}")
	if deformation_info:
		curve_path = os.path.join(output_dir, "deformation_curves.obj")
		save_curves_as_obj(curve_path, deformation_info)


def save_flip_animation(
	output_dir,
	initial_vertices,
	faces,
	fold_stage_sequence,
	frame_count,
	final_smooth_iterations=1,
	final_smooth_lambda_factor=0.03,
	final_smooth_pin_boundary=True,
):
	frame_count = max(2, int(frame_count))
	animation_dir = os.path.join(output_dir, "flip_animation")
	os.makedirs(animation_dir, exist_ok=True)
	initial_vertices = np.asarray(initial_vertices, dtype=np.float64)
	faces_arr = np.asarray(faces, dtype=np.int32)
	stage_count = len(fold_stage_sequence)
	if stage_count == 0:
		for frame_index in range(frame_count):
			frame_path = os.path.join(animation_dir, f"frame_{frame_index:04d}.obj")
			write_obj(frame_path, initial_vertices, faces_arr)
	else:
		timeline = np.linspace(0.0, float(stage_count), frame_count)
		for frame_index, timeline_value in enumerate(timeline):
			stage_progresses = _build_stage_progresses(stage_count, timeline_value)
			frame_vertices, _ = _compose_fold_stage_vertices(initial_vertices, fold_stage_sequence, stage_progresses)
			if np.any(stage_progresses > 0.0):
				frame_vertices = _apply_final_fold_smoothing(
					frame_vertices,
					faces_arr,
					iterations=final_smooth_iterations,
					lambda_factor=final_smooth_lambda_factor,
					pin_boundary=final_smooth_pin_boundary,
				)
			frame_path = os.path.join(animation_dir, f"frame_{frame_index:04d}.obj")
			write_obj(frame_path, frame_vertices, faces_arr)
	metadata_path = os.path.join(animation_dir, "flip_animation_info.json")
	with open(metadata_path, "w", encoding="utf-8") as handle:
		json.dump(
			{
				"frame_count": frame_count,
				"start_state": "initial_remeshed",
				"end_state": "folded",
				"stage_count": stage_count,
				"interpolation": "sequential per-fold bezier-curve interpolation with smoothstep easing, normalized arc-length remapping, curve-length correction on a fixed initial reference mesh, and light final surface fairing",
				"composition": "sum of per-stage displacement fields evaluated from the initial reference mesh",
				"length_policy": "each intermediate fold curve is scaled back to the corresponding rest curve length before skinning",
				"final_surface_fairing": {
					"iterations": int(max(0, final_smooth_iterations)),
					"lambda_factor": float(final_smooth_lambda_factor),
					"pin_boundary": bool(final_smooth_pin_boundary),
				},
				"stages": [
					{
						"deform_index": int(stage["deform_index"]),
						"cycle_index": int(stage["cycle_index"]),
						"side_name": stage["side_name"],
						"axis_name": stage["axis_name"],
					}
					for stage in fold_stage_sequence
				],
				"output_pattern": "frame_####.obj",
			},
			handle,
			indent=2,
		)
	print(f"Saved flip animation to {animation_dir} ({frame_count} frames)")

def parse_args():
	parser = create_parser(description="LuisaCompute Python example")
	add_backend_args(parser, default="auto")  # platform-aware default
	add_headless_args(parser, advance_frames_default=1)
	parser.add_argument("--abc_export", action="store_true", help="Export an Alembic animation in headless mode")
	parser.add_argument("--abc_path", default="", help="Alembic output path (default: <case output>/simulation.abc)")
	parser.add_argument("--prp_direction_optimization_iterations", "--prp_optimize_direction_count", "--PRP_optimize_directon_count",
						dest="prp_direction_optimization_iterations", type=int, default=0,
						help="Projected-Newton direction trials after discrete screening (default: 0 = disabled)")
	parser.add_argument("--use_gpu_pcg", type=int, default=0, choices=[0, 1])
	parser.add_argument("--prp_debug", type=int, default=0, choices=[0, 1], help="Enable per-contour/per-combo PRP debug stats (k-ring trajectory, frontier_hop, coverage, IRLS trajectory)")
	parser.add_argument("--use_gpu_untangling", type=int, default=None, choices=[0, 1], help="Evaluate PRP contours with the GPU batched path (default: engine default = GPU; 0 = CPU)")
	parser.add_argument("--intrinsic_contour_side_candidates", type=int, default=None, choices=[0, 1], help="Enable the experimental contour-side candidate coordinate")
	parser.add_argument("--intrinsic_tau", type=float, default=0.25, help="Phi window threshold for intrinsic side ownership (default: 0.25)")
	parser.add_argument("--intrinsic_filter_mode", choices=["and", "blend"], default="and", help="Independent constraints or weighted normalized coordinates")
	parser.add_argument("--intrinsic_blend_weight", type=float, default=0.5, help="Rest-coordinate weight w in blend mode; unrelated to LM")
	parser.add_argument("--rest_geodesic_distance_for_intrinsic_candidates", type=int, choices=[0, 1], default=None, help="Require rest-geodesic ownership (default: engine default = on)")
	parser.add_argument("--deformed_boundary_distance_for_intrinsic_candidates", type=int, choices=[0, 1], default=None, help="Require current-Euclidean ownership (default: engine default = on)")
	parser.add_argument("--untangling_process_contours_count", type=int, default=256, help="Maximum number of contours to process per untangling step")
	parser.add_argument("--fold_cycles", type=int, default=1, help="How many times to repeat the ordered center-fold sequence")
	parser.add_argument("--fold_depth", type=float, default=0.5, help="Relative fold depth controlling cylindrical curl angle and center pull")
	parser.add_argument("--fold_height_scale", type=float, default=3.0, help="Scale factor applied to the fold's flat-axis displacement. Values slightly above 1 increase the visible up/down amplitude without changing the inward pull.")
	parser.add_argument("--fold_side_count", type=int, default=4, choices=[0, 1, 2, 3, 4], help="How many sides to fold from the default order: left, right, front, back")
	parser.add_argument("--fold_sides", type=str, nargs="+", default=None, choices=list(DEFAULT_FOLD_ORDER), help="Optional explicit side order override, e.g. --fold_sides left right front back")
	parser.add_argument("--fold_smooth_iterations", type=int, default=1, help="Laplacian smoothing passes after each fold")
	parser.add_argument("--fold_smooth_respect_weights", type=int, default=1, choices=[0, 1], help="Keep per-stage smoothing local to the fold band so flat regions outside that fold are not moved by smoothing leakage.")
	parser.add_argument("--fold_final_smooth_iterations", type=int, default=2, help="Light Laplacian fairing passes applied once after all fold stages are composed. Set to 0 to disable the final surface fairing step.")
	parser.add_argument("--fold_final_smooth_lambda", type=float, default=0.03, help="Lambda factor for the final post-compose surface fairing pass.")
	parser.add_argument("--fold_final_smooth_pin_boundary", type=int, default=1, choices=[0, 1], help="Whether the final post-compose surface fairing keeps the remeshed boundary pinned.")
	parser.add_argument("--remesh_frequency", type=int, default=2 ** 7, help="Base constrained-triangulation resolution hint. Larger values target shorter average edges.")
	parser.add_argument("--remesh_min_triangle_angle_deg", type=float, default=30.0, help="Minimum interior triangle angle target for constrained triangulation. Larger values reduce skinny triangles but can increase vertex count or fail near 34 degrees. A practical range is usually 30-33.5.")
	parser.add_argument("--remesh_target_edge_length_mm", type=float, default=None, help="Optional target average edge length in mm. When set, the script searches for the smallest constrained-triangulation resolution whose measured average edge length is at or below this target.")
	parser.add_argument("--remesh_max_frequency", type=int, default=4096, help="Maximum equivalent remesh resolution allowed when solving for --remesh_target_edge_length_mm.")
	parser.add_argument("--output_flip_animation", type=int, default=0, choices=[0, 1], help="Whether to export an interpolated flip animation from the initial remeshed state to the folded state (1: True, 0: False).")
	parser.add_argument("--flip_animation_frames", type=int, default=120, help="Number of interpolated OBJ frames to save in output_dir/flip_animation, including the start and end states.")
	parser.add_argument("--mode", type=int, default=0, choices=[0, 1], help="0: untangling phase, 1: make intersection phase")
	parser.add_argument("--load_state_frame", type=int, default=0, help="Load state from this frame at startup (0 = start from initial state)")
	parser.add_argument("--save_best_state", type=int, default=1, choices=[0, 1], help="Save best.state/best.obj whenever the EF-pair count reaches a new minimum")
	parser.add_argument("--load_best_state", type=int, default=0, choices=[0, 1], help="Load best.state and its frame index from best_checkpoint.json in output_dir")
	parser.add_argument("--start_from_initial", type=int, default=1, choices=[0, 1], help="Start from the generated folded mesh (default: 1)")
	parser.add_argument("--output_frequency", type=int, default=16, help="Frequency of saving output frames and debug info (in frames)")
	parser.add_argument("--export_initial_debug", action="store_true", help="Export the first solved contour/response data as frame 0 so it pairs with frame_0000.obj under debug_frame_lag=0")
	parser.add_argument("--output_dir", type=str, default=None, help="Output directory (default: output/paper_cases/synthetic)")
	parser.add_argument("--use_ccd_linesearch", type=int, default=0, choices=[-1, 0, 1], help="Initial CCD line search (-1: keep default, 0: off, 1: on); auto-enabled below 16 contours")
	parser.add_argument("--untangling_response_depth", type=float, default=0.005, help="Per-iteration untangling response depth scale. Lower it (e.g. 0.005) when CCD line search is off to bound each response step.")
	parser.add_argument("--pcg_iter_count", type=int, default=200, help="Maximum PCG iterations; use 0 for fixed-geometry RayCasting diagnostics")
	parser.add_argument("--max_ef_pairs", type=int, default=10000, help="Abort the run when the current EF-pair count exceeds this limit; use 0 to disable")
	parser.add_argument("--lasting_zero_validation_steps", type=int, default=0, help="After first EF=0, keep intersection detection enabled, set untangling response stiffness to zero, and require this many additional EF=0 steps")
	parser.add_argument("--consistent_solve", type=int, choices=[0, 1], default=None, help="Force deterministic fixed-order assembly/SpMV on the CPU physics path (use with --use_gpu_pcg 0)")
	return parser.parse_args()
args = parse_args()
if args.deformed_boundary_distance_for_intrinsic_candidates and args.intrinsic_contour_side_candidates == 0:
	raise ValueError(
		"--deformed_boundary_distance_for_intrinsic_candidates=1 requires "
		"--intrinsic_contour_side_candidates=1")
if not 0.0 <= args.intrinsic_tau <= 1.0:
	raise ValueError("--intrinsic_tau must lie in [0, 1]")
if not 0.0 <= args.intrinsic_blend_weight <= 1.0:
	raise ValueError("--intrinsic_blend_weight must lie in [0, 1]")
if args.lasting_zero_validation_steps < 0:
	raise ValueError("--lasting_zero_validation_steps must be non-negative")
if 0 < args.lasting_zero_validation_steps < 5:
	raise ValueError(
		"--lasting_zero_validation_steps must be 0 (not requested) or at least 5")

# Initialize LuisaCompute device
backend = args.backend  # backends: cuda, dx, vk, metal (if supported on the platform)
solver = lcs.NewtonSolver()
solver.init_device(backend_name=backend)

load_state_frame = args.load_state_frame

output_dir = args.output_dir or os.path.join(root, "output", "paper_cases", "synthetic")
os.makedirs(output_dir, exist_ok=True)

BEST_STATE_FILENAME = "best.state"
BEST_OBJ_FILENAME = "best.obj"
BEST_METADATA_FILENAME = "best_checkpoint.json"

if args.start_from_initial and (args.load_best_state or load_state_frame != 0):
	raise ValueError("--start_from_initial cannot be combined with --load_best_state or --load_state_frame")
if args.load_best_state and load_state_frame != 0:
	raise ValueError("--load_best_state and --load_state_frame are mutually exclusive")

loaded_best_checkpoint = None


# Register meshes


# Legacy square-mesh C++ sample retained for reference.

orig_verts = np.array([[-0.5, 0, -0.5], [0.5, 0, -0.5], [-0.5, 0, 0.5], [0.5, 0, 0.5]], dtype=np.float64)
orig_faces = np.array([[0, 3, 1], [0, 2, 3]], dtype=np.int32)

# Legacy square-mesh file-loading sample retained for reference.

# args.remesh_target_edge_length_mm = 10.0
target_avg_edge_length = None if args.remesh_target_edge_length_mm is None else args.remesh_target_edge_length_mm * 1e-3
orig_verts, orig_faces, remesh_stats = _resolve_planar_remesh(
	orig_verts,
	orig_faces,
	requested_frequency=args.remesh_frequency,
	target_avg_edge_length=target_avg_edge_length,
	max_frequency=args.remesh_max_frequency,
	min_triangle_angle_deg=args.remesh_min_triangle_angle_deg,
)
print(
	f"Remeshed planar mesh: mode={remesh_stats['selection_mode']}, frequency={remesh_stats['frequency']}, "
	f"verts={remesh_stats['num_vertices']}, faces={remesh_stats['num_faces']}, edges={remesh_stats['edge_count']}, "
	f"avg_edge={remesh_stats['avg_edge_length'] * 1e3:.3f} mm, "
	f"min_edge={remesh_stats['min_edge_length'] * 1e3:.3f} mm, "
	f"max_edge={remesh_stats['max_edge_length'] * 1e3:.3f} mm, "
	f"target_min_angle={remesh_stats['target_min_triangle_angle_deg']:.1f} deg, "
	f"min_angle={remesh_stats['min_triangle_angle_deg']:.2f} deg, "
	f"p05_min_angle={remesh_stats['p05_min_triangle_angle_deg']:.2f} deg"
)
if remesh_stats['selection_mode'] == "target_avg_edge_length":
	print(
		f"Requested target average edge length = {remesh_stats['target_avg_edge_length'] * 1e3:.3f} mm"
	)

fold_sequence = _resolve_fold_sequence(args)
deformed_verts, deformation_info, flat_axis, span_axes, fold_stage_sequence = generate_ordered_center_folds(
	orig_verts,
	orig_faces,
	fold_cycles=args.fold_cycles,
	fold_depth=args.fold_depth,
	fold_sequence=fold_sequence,
	smooth_iterations=args.fold_smooth_iterations,
	fold_height_scale=args.fold_height_scale,
	stage_smooth_respect_weights=bool(args.fold_smooth_respect_weights),
	final_smooth_iterations=args.fold_final_smooth_iterations,
	final_smooth_lambda_factor=args.fold_final_smooth_lambda,
	final_smooth_pin_boundary=bool(args.fold_final_smooth_pin_boundary),
)
print(
	f"Applying ordered center folds: cycles={args.fold_cycles}, depth={args.fold_depth}, height_scale={args.fold_height_scale:.3f}, "
	f"final_smooth={args.fold_final_smooth_iterations}x{args.fold_final_smooth_lambda:.3f}, "
	f"sides={fold_sequence}, flat_axis={AXIS_NAMES[flat_axis]}, "
	f"span_axes={[AXIS_NAMES[axis] for axis in span_axes]}"
)
for fold in deformation_info:
	print(
		f"  Fold {fold['deform_index']:02d}: cycle={fold['cycle_index']} side={fold['side_name']} "
		f"axis={fold['axis_name']} verts={fold['mask_count']} "
		f"rise={fold['rise_height']:.4f} inward={fold['inward_shift']:.4f}"
	)
folded_edge_stats = _compute_mesh_edge_stats(deformed_verts, orig_faces)
print(
	f"Folded mesh edge stats: avg_edge={folded_edge_stats['avg_edge_length'] * 1e3:.3f} mm, "
	f"min_edge={folded_edge_stats['min_edge_length'] * 1e3:.3f} mm, "
	f"max_edge={folded_edge_stats['max_edge_length'] * 1e3:.3f} mm, "
	f"avg_ratio={folded_edge_stats['avg_edge_length'] / max(remesh_stats['avg_edge_length'], 1e-12):.4f}"
)
if args.output_flip_animation:
	save_flip_animation(
		output_dir=output_dir,
		initial_vertices=orig_verts,
		faces=orig_faces,
		fold_stage_sequence=fold_stage_sequence,
		frame_count=args.flip_animation_frames,
		final_smooth_iterations=args.fold_final_smooth_iterations,
		final_smooth_lambda_factor=args.fold_final_smooth_lambda,
		final_smooth_pin_boundary=bool(args.fold_final_smooth_pin_boundary),
	)
save_deformation_debug(output_dir, deformation_info, fold_sequence, flat_axis, span_axes)

cube = solver.create_world_data_from_array('cube', orig_verts, orig_faces)
cube.set_simulation_type(lcs.MaterialType.Cloth)
cube.set_physics_material_cloth(stretch_model="Spring", area_bending_stiffness=1e-2)
# cube.set_translation(-0.15, -0.13, 0.85)
cube_id = solver.register_world_data(cube)

solver.init_solver()
solver.set_object_vertex_positions(cube_id, deformed_verts.astype(np.float32))
print(f"Injected deformed positions into solver (registration_id={cube_id})")

# Get mesh info
solver.print_registered_meshes_info()


if args.load_best_state:
	best_state_path = os.path.join(output_dir, BEST_STATE_FILENAME)
	best_metadata_path = os.path.join(output_dir, BEST_METADATA_FILENAME)
	if not os.path.isfile(best_state_path):
		raise FileNotFoundError(f"Best state file not found: {best_state_path}")
	if not os.path.isfile(best_metadata_path):
		raise FileNotFoundError(f"Best checkpoint metadata not found: {best_metadata_path}")
	with open(best_metadata_path, "r", encoding="utf-8") as f:
		loaded_best_checkpoint = json.load(f)
	for required_key in ("frame", "ef_pair_count", "contour_count"):
		if required_key not in loaded_best_checkpoint:
			raise ValueError(
				f"Best checkpoint metadata lacks '{required_key}': {best_metadata_path}")
	load_state_frame = int(loaded_best_checkpoint["frame"])
	solver.load_target_state(best_state_path)
	print(
		f"Loaded best checkpoint from {best_state_path}: "
		f"frame={load_state_frame}, "
		f"EF={int(loaded_best_checkpoint['ef_pair_count'])}, "
		f"contours={int(loaded_best_checkpoint['contour_count'])}")
elif load_state_frame != 0:
	solver.load_target_state(output_dir + f"/state_frame_{load_state_frame}.state")
elif not args.start_from_initial:
	raise ValueError(
		"No initial state was selected. Use --start_from_initial 1, "
		"--load_best_state 1, or --load_state_frame <frame>.")

# Set scene parameters
config_ref = solver.get_config()
from utils.untangling_config import init_config
init_config(config_ref)
config_ref.pcg_iter_count = int(args.pcg_iter_count)
config_ref.prp_debug = bool(args.prp_debug) # 是否输出 PRP 逐 contour/combo 调试统计
config_ref.collect_iteration_debug = True
config_ref.PRP_direction_optimization_iterations = int(args.prp_direction_optimization_iterations)
config_ref.PRP_use_intrinsic_contour_side_candidates = bool(
	args.intrinsic_contour_side_candidates) if args.intrinsic_contour_side_candidates is not None else config_ref.PRP_use_intrinsic_contour_side_candidates
config_ref.PRP_intrinsic_tau = float(args.intrinsic_tau)
config_ref.PRP_use_blended_intrinsic_coordinates = args.intrinsic_filter_mode == "blend"
config_ref.PRP_intrinsic_blend_weight = float(args.intrinsic_blend_weight)
if args.rest_geodesic_distance_for_intrinsic_candidates is not None:
	config_ref.PRP_use_rest_geodesic_distance_for_intrinsic_candidates = bool(args.rest_geodesic_distance_for_intrinsic_candidates)
config_ref.PRP_use_deformed_boundary_distance_for_intrinsic_candidates = bool(
	args.deformed_boundary_distance_for_intrinsic_candidates) if args.deformed_boundary_distance_for_intrinsic_candidates is not None else config_ref.PRP_use_deformed_boundary_distance_for_intrinsic_candidates
config_ref.use_gpu_untangling = bool(args.use_gpu_untangling) if args.use_gpu_untangling is not None else config_ref.use_gpu_untangling
config_ref.use_gpu = bool(args.use_gpu_pcg) # 是否使用 GPU 进行 PCG 求解
if args.consistent_solve is not None:
	config_ref.consistent_solve = bool(args.consistent_solve)
	# consistent_solve 的确定性装配/PCG 只在 CPU 物理路径上实现；开启时默认
	# 强制 use_gpu = 0（引擎层 physics_step_GPU 也会防御性降级）。
	config_ref.use_gpu = False

config_ref.untangling_process_contours_count = args.untangling_process_contours_count # 每次 untangling step 最多处理的轮廓数量


config_ref.use_quasi_static_mode = True # 是否使用 quasi-static 模式
config_ref.use_ccd_linesearch = False
config_ref.ignore_near_zero_dist_pairs = False
config_ref.use_ccd_linesearch = True



config_ref.untangling_response_depth = float(args.untangling_response_depth)
# CCD line search override: -1 keeps the PRP default set above.
if args.use_ccd_linesearch >= 0:
	config_ref.use_ccd_linesearch = bool(args.use_ccd_linesearch)

# Legacy display startup path retained for reference.

config_ref.current_frame = load_state_frame


def to_plain_debug_maps(prp_debug_info):
	return {
		"bool": {str(key): bool(value) for key, value in dict(prp_debug_info.bool_stats).items()},
		"uint": {str(key): int(value) for key, value in dict(prp_debug_info.uint_stats).items()},
		"float": {str(key): float(value) for key, value in dict(prp_debug_info.float_stats).items()},
	}


def collect_debug_frame_payload(solver, frame_number, contour_data=None):
	if contour_data is None:
		contour_data = solver.get_intersection_contour_data()

	num_pairs_arr = np.asarray(contour_data.get("num_pairs", np.array([0], dtype=np.uint32)))
	num_contours_arr = np.asarray(contour_data.get("num_contours", np.array([0], dtype=np.uint32)))
	ef_pair_count = int(num_pairs_arr.flat[0]) if num_pairs_arr.size > 0 else 0
	contour_count = int(num_contours_arr.flat[0]) if num_contours_arr.size > 0 else 0

	debug_maps = {"bool": {}, "uint": {}, "float": {}}
	untangling_time = 0.0
	newton_step = 0.0
	try:
		debug_info = solver.get_prp_debug_info()
		debug_maps = to_plain_debug_maps(debug_info)
		float_stats = debug_maps.get("float", {})
		untangling_time = float(float_stats.get("untangling_total_time", 0.0))
		newton_step = float(float_stats.get("newton_step_inf_norm", float_stats.get("max_dq", 0.0)))
	except Exception as exc:
		debug_maps = {
			"bool": {},
			"uint": {},
			"float": {},
			"error": str(exc),
		}

	return {
		"frame": int(frame_number),
		"contour_count": contour_count,
		"ef_pair_count": ef_pair_count,
		"untangling_time_ms": untangling_time,
		"newton_step_inf_norm": newton_step,
		"debug_maps": debug_maps,
	}


class DisplayInterfaceGarment(DisplayInterface):
	def __init__(self, solver: lcs.NewtonSolver, config_ref, args, output_dir: str):
		super().__init__(solver=solver, config_ref=config_ref, args=args, output_dir=output_dir)
		self.best_ef_count = None
		self.best_checkpoint_frame = None
		self.ef_trajectory = []
		self.first_zero_frame = None
		self.zero_validation_samples = []
		self.lasting_zero_validated = False
		self.untangling_response_stiffness_before_validation = None
		self.ef_trajectory_path = os.path.join(self.output_dir, "ef_trajectory.json")
		if loaded_best_checkpoint is not None:
			self.best_ef_count = int(loaded_best_checkpoint["ef_pair_count"])
			self.best_checkpoint_frame = int(loaded_best_checkpoint["frame"])
			if os.path.isfile(self.ef_trajectory_path):
				with open(self.ef_trajectory_path, "r", encoding="utf-8") as f:
					previous_trajectory_payload = json.load(f)
				previous_samples = previous_trajectory_payload.get("samples", [])
				if not isinstance(previous_samples, list):
					raise ValueError(
						f"Existing trajectory has non-list samples: {self.ef_trajectory_path}")
				self.ef_trajectory = [
					sample for sample in previous_samples
					if int(sample.get("frame", -1)) <= self.best_checkpoint_frame
				]
				if self.ef_trajectory and int(self.ef_trajectory[-1]["frame"]) != self.best_checkpoint_frame:
					raise ValueError(
						"Existing trajectory does not contain the loaded best-checkpoint frame "
						f"{self.best_checkpoint_frame}: {self.ef_trajectory_path}")

		if load_state_frame == 0:
			initial_state_path = os.path.join(self.output_dir, "state_frame_0.state")
			output_obj_path = os.path.join(self.output_dir, f"frame_{0:04}.obj")
			self.solver.save_current_state(initial_state_path)
			self.solver.save_sim_result(obj_path=output_obj_path)
			self._export_debug_info_json(0)
			print(f"Saved initial state to {initial_state_path}")
			print(f"Saved initial frame to {output_obj_path}")

	def _write_ef_trajectory(self):
		trajectory_tmp_path = self.ef_trajectory_path + ".tmp"
		positive_ef_counts = [sample["ef_pair_count"] for sample in self.ef_trajectory if sample["ef_pair_count"] > 0]
		payload = {
			"version": 1,
			"configuration": {
				"backend": str(args.backend),
				"advance_frames": int(args.advance_frames),
				"start_from_initial": int(args.start_from_initial),
				"load_best_state": int(args.load_best_state),
				"use_gpu_pcg": int(bool(config_ref.use_gpu)),
				"use_gpu_untangling": int(bool(config_ref.use_gpu_untangling)),
				"prp_debug": bool(config_ref.prp_debug),
				"consistent_solve": bool(config_ref.consistent_solve),
				"pcg_iter_count": int(args.pcg_iter_count),
				"pcg_lm_adaptive": bool(self.config_ref.pcg_lm_adaptive),
				"pcg_lm_escalation_factor": float(self.config_ref.pcg_lm_escalation_factor),
				"use_ccd_linesearch": int(args.use_ccd_linesearch),
				"process_contours_count": int(args.untangling_process_contours_count),
				"untangling_response_depth": float(args.untangling_response_depth),
				"intrinsic_contour_side_candidates": int(bool(config_ref.PRP_use_intrinsic_contour_side_candidates)),
				"intrinsic_tau": float(args.intrinsic_tau),
				"intrinsic_filter_mode": "blend" if config_ref.PRP_use_blended_intrinsic_coordinates else "and",
				"intrinsic_blend_weight": float(config_ref.PRP_intrinsic_blend_weight),
				"rest_geodesic_distance_for_intrinsic_candidates": int(bool(config_ref.PRP_use_rest_geodesic_distance_for_intrinsic_candidates)),
				"deformed_boundary_distance_for_intrinsic_candidates": int(bool(config_ref.PRP_use_deformed_boundary_distance_for_intrinsic_candidates)),
				"prp_direction_optimization_iterations": int(
					args.prp_direction_optimization_iterations),
				"lasting_zero_validation_steps": int(
					args.lasting_zero_validation_steps),
				"max_ef_pairs": int(args.max_ef_pairs),
			},
			"sample_count": len(self.ef_trajectory),
			"first_zero_frame": self.first_zero_frame,
			"lasting_zero_validated": bool(self.lasting_zero_validated),
			"converged": bool(
				self.lasting_zero_validated
				if args.lasting_zero_validation_steps > 0
				else self.ef_trajectory
				and self.ef_trajectory[-1]["ef_pair_count"] == 0),
			"minimum_nonzero_ef_pair_count": min(positive_ef_counts) if positive_ef_counts else 0,
			"maximum_ef_pair_count": max((sample["ef_pair_count"] for sample in self.ef_trajectory), default=0),
			"samples": self.ef_trajectory,
		}
		with open(trajectory_tmp_path, "w", encoding="utf-8") as f:
			json.dump(payload, f, indent=2, sort_keys=True)
		os.replace(trajectory_tmp_path, self.ef_trajectory_path)

	def _write_zero_audit(self, status: str):
		payload = {
			"first_zero_frame": self.first_zero_frame,
			"required_validation_steps": int(args.lasting_zero_validation_steps),
			"samples": self.zero_validation_samples,
			"status": str(status),
			"lasting_zero": status == "passed",
			"intersection_detection_kept_enabled": True,
			"untangling_response_disabled_by_zero_stiffness": (
				self.untangling_response_stiffness_before_validation is not None),
			"untangling_response_stiffness_before_validation": (
				self.untangling_response_stiffness_before_validation),
		}
		audit_path = os.path.join(self.output_dir, "lasting_zero_audit.json")
		audit_tmp_path = audit_path + ".tmp"
		with open(audit_tmp_path, "w", encoding="utf-8") as f:
			json.dump(payload, f, indent=2, sort_keys=True)
		os.replace(audit_tmp_path, audit_path)

	def _export_debug_info_json(self, frame_idx: int, contour_data=None):
		debug_payload = collect_debug_frame_payload(self.solver, frame_idx, contour_data)
		debug_json_path = os.path.join(self.output_dir, f"debug_info_{frame_idx:06d}.json")
		with open(debug_json_path, "w", encoding="utf-8") as f:
			json.dump(debug_payload, f, indent=2, sort_keys=True)

	def _save_best_checkpoint(self, curr_frame: int, ef_pair_count: int, contour_count: int):
		state_path = os.path.join(self.output_dir, BEST_STATE_FILENAME)
		obj_path = os.path.join(self.output_dir, BEST_OBJ_FILENAME)
		metadata_path = os.path.join(self.output_dir, BEST_METADATA_FILENAME)
		state_tmp_path = state_path + ".tmp"
		obj_tmp_path = obj_path + ".tmp"
		metadata_tmp_path = metadata_path + ".tmp"

		self.solver.save_current_state(state_tmp_path)
		self.solver.save_sim_result(obj_path=obj_tmp_path)
		os.replace(state_tmp_path, state_path)
		os.replace(obj_tmp_path, obj_path)

		metadata = {
			"version": 1,
			"frame": int(curr_frame),
			"ef_pair_count": int(ef_pair_count),
			"contour_count": int(contour_count),
			"state_file": BEST_STATE_FILENAME,
			"obj_file": BEST_OBJ_FILENAME,
		}
		with open(metadata_tmp_path, "w", encoding="utf-8") as f:
			json.dump(metadata, f, indent=2, sort_keys=True)
		os.replace(metadata_tmp_path, metadata_path)

		self.best_ef_count = int(ef_pair_count)
		self.best_checkpoint_frame = int(curr_frame)
		print(
			f"Saved new best checkpoint: frame={curr_frame}, "
			f"EF={ef_pair_count}, contours={contour_count}, state={state_path}")

	def after_step(self):
		curr_frame = getattr(self.config_ref, "current_frame", -1)

		# Abort before collision buffers overflow device memory and trigger host fallback/OOM.
		check_collision_buffer_budget(self.solver)

		should_export = curr_frame % args.output_frequency == 0
		if should_export:
			output_obj_path = os.path.join(self.output_dir, f"frame_{curr_frame:04}.obj")
			self.solver.save_sim_result(obj_path=output_obj_path)
			print(f"Saved frame {curr_frame} to {output_obj_path}")

		debug_export_frame = curr_frame if should_export else None
		if args.export_initial_debug and curr_frame == 1:
			debug_export_frame = 0

		if debug_export_frame is not None:
			contour_data = self.solver.get_intersection_contour_data()
			np.savez_compressed(
				os.path.join(self.output_dir, f"contour_debug_{debug_export_frame:06d}.npz"),
				**{k: np.asarray(v) for k, v in contour_data.items()})
			self._export_debug_info_json(debug_export_frame, contour_data)

			if self.config_ref.use_untangling:
				response_data = self.solver.get_response_pair_data()
				np.savez_compressed(
					os.path.join(self.output_dir, f"response_pair_debug_{debug_export_frame:06d}.npz"),
					**{k: np.asarray(v) for k, v in response_data.items()})

		uint_stats = dict(self.solver.get_prp_debug_info().uint_stats)
		count_key = "intersection_resolver_input_ef_pair_count"
		if count_key not in uint_stats:
			if curr_frame <= 1:
				print(f"Frame {curr_frame}: PRP EF diagnostics are not initialized yet")
				return False
			raise RuntimeError(
				f"The intersection resolver did not publish its input EF count; keys={sorted(uint_stats)}")
		curr_ef_count = int(uint_stats[count_key])
		if curr_ef_count == 0:
			curr_contour_count = 0
		elif "contour_count" in uint_stats:
			curr_contour_count = int(uint_stats["contour_count"])
		else:
			curr_contour_count = 0
		print(f"Frame {curr_frame}: #EF pairs = {curr_ef_count}, #contours = {curr_contour_count}")
		self.ef_trajectory.append({
			"frame": int(curr_frame),
			"ef_pair_count": curr_ef_count,
			"contour_count": curr_contour_count,
		})
		if args.max_ef_pairs > 0 and curr_ef_count > args.max_ef_pairs:
			self._write_ef_trajectory()
			raise RuntimeError(
				f"EF-pair count {curr_ef_count} exceeds --max_ef_pairs {args.max_ef_pairs}; "
				"aborting the divergent Synthetic run")

		if args.save_best_state and (
			self.best_ef_count is None or curr_ef_count < self.best_ef_count):
			self._save_best_checkpoint(curr_frame, curr_ef_count, curr_contour_count)

		if self.first_zero_frame is None:
			if curr_ef_count != 0:
				self._write_ef_trajectory()
				return False

			self.first_zero_frame = int(curr_frame)
			state_path = os.path.join(self.output_dir, f"state_frame_{curr_frame}.state")
			output_obj_path = os.path.join(self.output_dir, f"frame_{curr_frame:04}.obj")
			self.solver.save_current_state(state_path)
			self.solver.save_sim_result(obj_path=output_obj_path)

			if args.lasting_zero_validation_steps == 0:
				self._write_zero_audit("not_requested")
				self._write_ef_trajectory()
				print(
					f"Frame {curr_frame} is untangled with 0 EF pairs; "
					"lasting-zero validation was not requested.")
				return True

			self.untangling_response_stiffness_before_validation = float(
				self.config_ref.stiffness_untangling)
			self.config_ref.stiffness_untangling = 0.0
			self._write_zero_audit("pending")
			self._write_ef_trajectory()
			print(
				f"Frame {curr_frame} first reached 0 EF pairs; keeping intersection "
				f"detection enabled and setting untangling response stiffness to zero "
				f"for {args.lasting_zero_validation_steps} validation steps.")
			return False

		self.zero_validation_samples.append({
			"frame": int(curr_frame),
			"ef_pair_count": curr_ef_count,
		})
		if curr_ef_count != 0:
			self._write_zero_audit("failed")
			self._write_ef_trajectory()
			print(
				f"Lasting-zero validation failed at frame {curr_frame}: "
				f"EF={curr_ef_count}.")
			return True

		if len(self.zero_validation_samples) >= args.lasting_zero_validation_steps:
			self.lasting_zero_validated = True
			self._write_zero_audit("passed")
			self._write_ef_trajectory()
			print(
				f"Lasting-zero validation passed for "
				f"{args.lasting_zero_validation_steps} steps.")
			return True

		self._write_zero_audit("pending")
		self._write_ef_trajectory()
		return False

display = DisplayInterfaceGarment(solver, config_ref, args, output_dir)
display.run()

solver.cleanup_device()
