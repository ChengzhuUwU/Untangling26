"""Static self-intersection repair from a single OBJ mesh.

Loads one self-intersecting OBJ file, runs RayCasting Untangling in quasi-static
mode (no gravity, no floor), and writes the repaired surface back as an OBJ.
By default, Polyscope displays the mesh and provides run/pause and single-step
controls. Pass --headless for unattended repair. After the viewer closes, the
process exits 0 when the final collision report is empty, 1 otherwise.

Example:
	python PythonBindings/tests/test_load_from_obj.py `
	  --backend cuda --input_mesh path/to/self_intersecting.obj
"""

import argparse
import json
from utils.intrinsic_filter import add_blend_args, apply_blend_args, configuration_record
import os
import sys

import numpy as np

root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, os.path.join(root, "build", "bin"))
import lcs_py as lcs

from shared_args import (
	UNTANGLING_METHODS,
	configure_untangling_method,
)
from utils.demo_runner import run_simulation


def parse_args():
	parser = argparse.ArgumentParser(
		description="Static self-intersection repair: load an OBJ, untangle it in quasi-static mode, save the resolved OBJ.")
	parser.add_argument("--backend", type=str, default="cuda", help="LuisaCompute backend (cuda, dx, vk, metal; default: cuda)")
	parser.add_argument("--input_mesh", type=str, required=True, help="Self-intersecting input OBJ mesh")
	parser.add_argument("--output_obj", type=str, default=None, help="Repaired output OBJ (default: <output_dir>/<stem>_resolved.obj)")
	parser.add_argument("--output_dir", type=str, default=None, help="Directory for summary.json and optional frames (default: output/paper_cases/obj_repair/<stem>)")
	parser.add_argument("--advance_frames", type=int, default=200, help="Iteration budget (default: 200)")
	parser.add_argument("--headless", action="store_true", help="Run without Polyscope (default: interactive GUI)")
	parser.add_argument("--material", choices=("cloth", "rigid"), default="cloth", help="Input mesh material (default: cloth)")
	parser.add_argument("--auto_scaling", type=int, choices=(0, 1), default=1, help="Normalize the mesh into the unit cube (default: 1)")
	parser.add_argument("--use_floor", type=int, choices=(0, 1), default=0, help="Enable the ground plane (default: 0)")
	parser.add_argument("--gravity", type=float, nargs=3, default=(0.0, 0.0, 0.0), help="Gravity vector (default: 0 0 0)")
	parser.add_argument("--physics_step", choices=("cpu", "gpu"), default="cpu", help="Newton physics stepping backend (default: cpu)")
	parser.add_argument("--strict_zero_frames", type=int, default=1, help="Consecutive zero-EF reports required before success (default: 1)")
	parser.add_argument("--export_frames", type=int, choices=(0, 1), default=0, help="Also export every frame as frame_NNNN.obj (default: 0)")
	parser.add_argument("--method", type=str.upper, choices=UNTANGLING_METHODS, default="PRP", help="Untangling method to use (default: PRP)")
	parser.add_argument("--prp_direction_optimization_iterations", "--prp_optimize_direction_count", "--PRP_optimize_directon_count",
						dest="prp_direction_optimization_iterations", type=int, default=0,
						help="Projected-Newton direction trials after discrete screening (default: 0 = disabled)")
	parser.add_argument("--untangling_response_depth", type=float, default=1.0, help="Untangling response depth (default: 1.0)")
	parser.add_argument("--use_ccd_linesearch", type=int, choices=(0, 1), default=1, help="Initial CCD line search (default: 1); auto-enabled below 16 contours")
	parser.add_argument("--use_gpu_untangling", type=int, choices=(0, 1), default=0, help="Evaluate PRP contours with the GPU batched path (default: 0 = CPU; 1 = GPU)")
	parser.add_argument("--intrinsic_contour_side_candidates", type=int, choices=(0, 1), default=None, help="Intrinsic contour-side candidate attribution (default: engine default = on)")
	parser.add_argument("--intrinsic_tau", type=float, default=0.25, help="Phi window threshold tau for side ownership (default: 0.25)")
	parser.add_argument("--deformed_boundary_distance_for_intrinsic_candidates", type=int, choices=(0, 1), default=None, help="Require current-space ownership in addition to rest-geodesic ownership (default: engine default = on)")
	parser.add_argument("--prp_debug", type=int, choices=(0, 1), default=None, help="Enable detailed PRP debug stats (default: engine default = off)")
	parser.add_argument("--pcg_lm_adaptive", type=int, choices=(0, 1), default=None, help="Enable adaptive LM through the Newton delta gate (default: engine default = on)")
	parser.add_argument("--pcg_iter_count", type=int, default=200, help="PCG iteration budget per Newton solve (default: 200)")
	parser.add_argument("--use_quasi_static_mode", type=int, choices=(0, 1), default=1, help="Quasi-static stepping without velocity carry-over (default: 1)")
	parser.add_argument("--qs_inertia_mode", choices=("off", "hessian_only", "full"), default="hessian_only", help="Free-DOF inertia treatment in quasi-static mode (default: hessian_only)")
	parser.add_argument("--ignore_nearly_zero_distance", type=int, choices=(0, 1), default=0, help="1 = drop nearly-zero-distance EF pairs (default: 0)")
	parser.add_argument("--consistent_solve", type=int, choices=(0, 1), default=None)
	parser.add_argument("--max_ef_pairs", type=int, default=0, help="Abort above this EF count; 0 disables")
	parser.add_argument("--untangling_process_contours_count", type=int, default=256)
	add_blend_args(parser)
	return parser.parse_args()


def collision_counts(solver):
	contour_data = solver.get_intersection_contour_data()
	num_pairs = np.asarray(contour_data.get("num_pairs", np.zeros(1, dtype=np.uint32)))
	num_contours = np.asarray(contour_data.get("num_contours", np.zeros(1, dtype=np.uint32)))
	return (
		int(num_pairs.flat[0]) if num_pairs.size else 0,
		int(num_contours.flat[0]) if num_contours.size else 0,
	)


def main():
	args = parse_args()

	input_mesh = os.path.realpath(args.input_mesh)
	if not os.path.isfile(input_mesh):
		raise FileNotFoundError(f"Input mesh not found: {input_mesh}")
	mesh_stem = os.path.splitext(os.path.basename(input_mesh))[0]

	output_dir = args.output_dir or os.path.join(
		root, "output", "paper_cases", "obj_repair", mesh_stem)
	os.makedirs(output_dir, exist_ok=True)
	output_obj = args.output_obj or os.path.join(output_dir, f"{mesh_stem}_resolved.obj")

	backend = args.backend
	solver = lcs.NewtonSolver()
	solver.init_device(backend_name=backend)

	try:
		mesh = solver.create_world_data_from_file_path(mesh_stem, input_mesh)
		mesh.set_auto_scaling(bool(args.auto_scaling))
		if args.material == "rigid":
			mesh.set_simulation_type(lcs.MaterialType.Rigid)
			mesh.set_physics_material_rigid(thickness=1e-3, stiffness=1e4)
		else:
			mesh.set_physics_material_cloth()
		solver.register_world_data(mesh)

		solver.init_solver()
		solver.print_registered_meshes_info()

		config = solver.get_config()
		configure_untangling_method(config, args.method)
		config.PRP_direction_optimization_iterations = int(args.prp_direction_optimization_iterations)
		config.use_quasi_static_mode = bool(args.use_quasi_static_mode)
		config.qs_inertia_mode = {
			"off": lcs.QuasiStaticInertiaMode.Off,
			"hessian_only": lcs.QuasiStaticInertiaMode.HessianOnly,
			"full": lcs.QuasiStaticInertiaMode.Full,
		}[args.qs_inertia_mode]
		config.use_floor = bool(args.use_floor)
		config.gravity.x, config.gravity.y, config.gravity.z = (float(g) for g in args.gravity)
		config.untangling_response_depth = float(args.untangling_response_depth)
		config.use_ccd_linesearch = bool(args.use_ccd_linesearch)
		config.pcg_iter_count = int(args.pcg_iter_count)
		config.ignore_near_zero_dist_pairs = bool(args.ignore_nearly_zero_distance)
		if args.pcg_lm_adaptive is not None:
			config.pcg_lm_adaptive = bool(args.pcg_lm_adaptive)
		if args.use_gpu_untangling is not None:
			config.use_gpu_untangling = bool(args.use_gpu_untangling)
		if args.intrinsic_contour_side_candidates is not None:
			config.PRP_use_intrinsic_contour_side_candidates = bool(args.intrinsic_contour_side_candidates)
			config.PRP_intrinsic_tau = float(args.intrinsic_tau)
			config.PRP_use_deformed_boundary_distance_for_intrinsic_candidates = bool(
				args.deformed_boundary_distance_for_intrinsic_candidates)
		if args.prp_debug is not None:
			config.prp_debug = bool(args.prp_debug)

		apply_blend_args(config, args)
		config.untangling_process_contours_count = args.untangling_process_contours_count
		initial_configuration = configuration_record(config)
		solver.save_sim_result(obj_path=os.path.join(output_dir, "initial.obj"))

		physics_step = solver.physics_step_cpu if args.physics_step == "cpu" else solver.physics_step_gpu

		records = []
		zero_streak = 0
		converged = False
		converged_frame = None
		initial_ef = None

		def step(frame):
			nonlocal initial_ef, zero_streak, converged, converged_frame
			physics_step()
			ef, contours = collision_counts(solver)
			if frame == 1:
				initial_ef = ef
				print(f"[{mesh_stem}] method={args.method} material={args.material} | Initial EF={initial_ef}, contours={contours}", flush=True)
			records.append({"frame": frame, "ef": ef, "contours": contours})
			print(f"    frame {frame:3d}: EF={ef:5d}, contours={contours:3d}", flush=True)

			if args.export_frames:
				solver.save_sim_result(obj_path=os.path.join(output_dir, f"frame_{frame:04d}.obj"))

			if args.max_ef_pairs > 0 and ef > args.max_ef_pairs:
				return "ef_limit"
			zero_streak = zero_streak + 1 if (ef == 0 and contours == 0) else 0
			if zero_streak >= max(1, args.strict_zero_frames):
				converged = True
				converged_frame = frame
				return "resolved"

		stop_reason = run_simulation(
			solver, step, args.advance_frames, headless=args.headless,
			title=f"{mesh_stem} / {args.method}", output_dir=output_dir)

		# Export the final mesh regardless of convergence; the report tells which case it is.
		solver.save_sim_result(obj_path=output_obj)

		final_ef = records[-1]["ef"] if records else None
		summary = {
			"input_mesh": input_mesh,
			"configuration": initial_configuration,
			"frame_records": records,
			"output_obj": os.path.realpath(output_obj),
			"method": args.method,
			"material": args.material,
			"quasi_static": bool(args.use_quasi_static_mode),
			"initial_ef": initial_ef,
			"final_ef": final_ef,
			"converged": converged,
			"converged_frame": converged_frame,
			"frames": len(records),
			"budget": args.advance_frames,
			"stop_reason": stop_reason,
		}
		summary_path = os.path.join(output_dir, "summary.json")
		with open(summary_path, "w", encoding="utf-8") as f:
			json.dump(summary, f, indent=2)

		if converged:
			print(f"[{mesh_stem}] Resolved at frame {converged_frame} (initial EF {initial_ef}).")
		else:
			print(f"[{mesh_stem}] Stopped ({stop_reason}) after {len(records)} frames; residual EF={final_ef}.")
		print(f"Output mesh: {output_obj}")
		print(f"Summary: {summary_path}")

	finally:
		solver.cleanup_device()
	sys.exit(0 if converged else 1)


if __name__ == "__main__":
	main()
