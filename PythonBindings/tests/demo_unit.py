import trimesh
import numpy as np
import json
from utils.intrinsic_filter import add_blend_args, apply_blend_args, configuration_record
import os
import sys

root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, os.path.join(root, 'build', 'bin'))
import lcs_py as lcs

from shared_args import (
	UNTANGLING_METHODS,
	add_backend_args,
	add_headless_args,
	configure_untangling_method,
	create_parser,
)
from utils.display_interface import DisplayInterface

def parse_args():
	parser = create_parser(description="LuisaCompute Python example")
	add_backend_args(parser, default="auto")  # platform-aware default
	add_headless_args(parser, advance_frames_default=1)
	parser.add_argument("--use_subdivision", action="store_true", help="Subdivide the canonical input mesh at runtime")
	parser.add_argument("--subdiv_levels", type=int, default=3, help="Number of subdivision levels (default: 3)")
	parser.add_argument("--method", type=str.upper, choices=UNTANGLING_METHODS, default="PRP", help="Untangling method to use (default: PRP)")
	parser.add_argument("--use_gpu_untangling", type=int, choices=[0, 1], default=None, help="Evaluate PRP contours with the GPU batched path (default: engine default = GPU; 0 = CPU)")
	parser.add_argument("--intrinsic_contour_side_candidates", type=int, choices=[0, 1], default=None, help="Intrinsic contour-side candidate attribution (default: engine default = on)")
	parser.add_argument("--intrinsic_tau", type=float, default=0.25, help="Phi window threshold tau for side ownership (default: 0.25)")
	parser.add_argument("--deformed_boundary_distance_for_intrinsic_candidates", type=int, choices=[0, 1], default=None, help="Require current-space ownership in addition to rest-geodesic ownership (default: engine default = on)")
	parser.add_argument("--consistent_solve", type=int, choices=[0, 1], default=None, help="Force deterministic fixed-order assembly/SpMV on the CPU physics path (default: engine default = off)")
	parser.add_argument("--prp_direction_optimization_iterations", "--prp_optimize_direction_count", "--PRP_optimize_directon_count",
						dest="prp_direction_optimization_iterations", type=int, default=0,
						help="Projected-Newton direction trials after discrete screening (default: 0 = disabled)")
	parser.add_argument("--untangling_process_contours_count", type=int, default=256, help="Maximum number of contours to process per untangling step")
	parser.add_argument("--scene_id", type=int, default=0, help="ID of the scene to simulate")
	parser.add_argument("--mode", type=int, default=0, choices=[0, 1], help="0: untangling phase, 1: make intersection phase")
	parser.add_argument("--output_dir", type=str, default=None, help="Output directory (default: output/paper_cases/canonical_units/<method>/<case>)")
	parser.add_argument("--export_debug", action="store_true", help="Export per-frame OBJ and PRP debug arrays")
	parser.add_argument("--max_ef_pairs", type=int, default=0, help="Abort above this EF count; 0 disables")
	add_blend_args(parser)
	return parser.parse_args()


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


args = parse_args()
# args.subdiv_levels = 2

# Initialize LuisaCompute device
backend = args.backend  # backends: cuda, dx, vk, metal (if supported on the platform)
solver = lcs.NewtonSolver()
solver.init_device(backend_name=backend)

# Register meshes
mesh_dir = os.path.join(root, "Resources", "InputMesh", "SelfIntersectionUnit")
files = [
    "unit_BBII.obj",  # 0
	"unit_BIBI.obj",  # 1
	"unit_BLI.obj", # 2
	"unit_Closed.obj",  # 3
	"unit_Cross.obj",  # 4
	"unit_Eight.obj", # 5
	"unit_LL.obj"  # 6
]
file_idx = args.scene_id % len(files)
input_file_name = files[file_idx]
file_name = input_file_name
obj_dir = os.path.join(mesh_dir, file_name)

cube_mesh = trimesh.load(obj_dir, process=False)

if args.use_subdivision:
	file_name = file_name[:-4] + "_subdiv.obj"
	# Legacy subdivision mesh-loading sample retained for reference.
	for _ in range(args.subdiv_levels):
		cube_mesh = cube_mesh.subdivide()
cube = solver.create_world_data_from_array(file_name, cube_mesh.vertices, cube_mesh.faces)
cube.set_simulation_type(lcs.MaterialType.Cloth)
cube.set_physics_material_cloth(stretch_model="Spring",area_bending_stiffness=5e-1)
# Legacy cube transform sample retained for reference.
cube.set_rotation(-0.5235988, 0.0, -0.5235988)
cube.set_scale(0.2)
# Legacy cube scale samples retained for reference.
cube_id = solver.register_world_data(cube)

# Initialize the solver (builds internal data structures, compiles shaders, etc.)
solver.init_solver()


# Get mesh info
solver.print_registered_meshes_info()

# Set scene parameters
config_ref = solver.get_config()
from utils.untangling_config import init_config
init_config(config_ref)
configure_untangling_method(config_ref, args.method)
config_ref.use_gpu_untangling = bool(args.use_gpu_untangling) if args.use_gpu_untangling is not None else config_ref.use_gpu_untangling
config_ref.PRP_use_intrinsic_contour_side_candidates = bool(
	args.intrinsic_contour_side_candidates) if args.intrinsic_contour_side_candidates is not None else config_ref.PRP_use_intrinsic_contour_side_candidates
config_ref.PRP_intrinsic_tau = float(args.intrinsic_tau)
config_ref.PRP_use_deformed_boundary_distance_for_intrinsic_candidates = bool(
	args.deformed_boundary_distance_for_intrinsic_candidates) if args.deformed_boundary_distance_for_intrinsic_candidates is not None else config_ref.PRP_use_deformed_boundary_distance_for_intrinsic_candidates
config_ref.consistent_solve = bool(args.consistent_solve)
config_ref.PRP_direction_optimization_iterations = int(args.prp_direction_optimization_iterations)

config_ref.pcg_iter_count = 200

config_ref.untangling_process_contours_count = args.untangling_process_contours_count # 每次 untangling step 最多处理的轮廓数量


config_ref.use_quasi_static_mode = True # 是否使用 quasi-static 模式
config_ref.use_ccd_linesearch = args.method == "PRP"

config_ref.prp_debug = bool(args.export_debug) and args.method == "PRP"

config_ref.untangling_response_depth = 1e-0
# config_ref.untangling_response_depth = 5e-3
config_ref.d_hat = 5e-3 


apply_blend_args(config_ref, args)
initial_configuration = configuration_record(config_ref)

# Output directory
case_suffix = f"_subdiv{args.subdiv_levels}" if args.use_subdivision else "_raw"
case_name = os.path.splitext(input_file_name)[0] + case_suffix
output_dir = args.output_dir or os.path.join(
	root, "output", "paper_cases", "canonical_units", args.method, case_name)
os.makedirs(output_dir, exist_ok=True)
solver.save_sim_result(obj_path=os.path.join(output_dir, "initial.obj"))

class DisplayInterfaceGarment(DisplayInterface):
	def __init__(self, solver: lcs.NewtonSolver, config_ref, args, output_dir: str):
		super().__init__(solver=solver, config_ref=config_ref, args=args, output_dir=output_dir)
		self.frame_records = []

	def _write_summary(self, converged: bool):
		with open(os.path.join(self.output_dir, "summary.json"), "w", encoding="utf-8") as f:
			json.dump({
				"case": case_name,
				"configuration": initial_configuration,
				"method": args.method,
				"scene_id": int(args.scene_id),
				"input_mesh": os.path.relpath(obj_dir, root).replace(os.sep, "/"),
				"subdivision_levels": int(args.subdiv_levels) if args.use_subdivision else 0,
				"prp_direction_optimization_iterations": int(
					args.prp_direction_optimization_iterations),
				"consistent_solve": bool(getattr(args, "consistent_solve", 0)),
				"converged": bool(converged),
				"frames": self.frame_records,
			}, f, indent=2)

	def _export_debug_info_json(self, frame_idx: int, contour_data=None):
		import json
		debug_payload = collect_debug_frame_payload(self.solver, frame_idx, contour_data)
		debug_json_path = os.path.join(self.output_dir, f"debug_info_{frame_idx:06d}.json")
		with open(debug_json_path, "w", encoding="utf-8") as f:
			json.dump(debug_payload, f, indent=2, sort_keys=True)

	def after_step(self):
		curr_frame = getattr(self.config_ref, "current_frame", -1)

		contour_data = self.solver.get_intersection_contour_data()
		if args.export_debug:
			output_obj_path = os.path.join(self.output_dir, f"frame_{curr_frame:04}.obj")
			self.solver.save_sim_result(obj_path=output_obj_path)
			np.savez_compressed(
				os.path.join(self.output_dir, f"contour_debug_{curr_frame:06d}.npz"),
				**{k: np.asarray(v) for k, v in contour_data.items()})
			self._export_debug_info_json(curr_frame, contour_data)
			if self.config_ref.use_untangling_PRP:
				response_data = self.solver.get_response_pair_data()
				np.savez_compressed(
					os.path.join(self.output_dir, f"response_pair_debug_{curr_frame:06d}.npz"),
					**{k: np.asarray(v) for k, v in response_data.items()})
			
		num_pairs_arr = np.asarray(contour_data.get("num_pairs", np.array([0], dtype=np.uint32)))
		num_contours_arr = np.asarray(contour_data.get("num_contours", np.array([0], dtype=np.uint32)))
		curr_ef_count = int(num_pairs_arr.flat[0]) if num_pairs_arr.size > 0 else 0
		curr_contour_count = int(num_contours_arr.flat[0]) if num_contours_arr.size > 0 else 0
		print(f"Frame {curr_frame}: #EF pairs = {curr_ef_count}, #contours = {curr_contour_count}")
		self.frame_records.append({
			"frame": int(curr_frame),
			"ef_pair_count": curr_ef_count,
			"contour_count": curr_contour_count,
		})
		self._write_summary(curr_ef_count == 0)

		if args.max_ef_pairs > 0 and curr_ef_count > args.max_ef_pairs:
			raise RuntimeError(f"EF pair limit exceeded: {curr_ef_count} > {args.max_ef_pairs}")

		if curr_ef_count == 0:
			state_path = os.path.join(self.output_dir, "resolved.state")
			self.solver.save_current_state(state_path)
			self.solver.save_sim_result(obj_path=os.path.join(self.output_dir, "resolved.obj"))
			print(f"Frame {curr_frame} is untangled with 0 EF pairs. Stopping simulation.")
			return True

		return False

display = DisplayInterfaceGarment(solver, config_ref, args, output_dir)
display.run()

solver.cleanup_device()
