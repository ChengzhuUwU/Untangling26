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
	parser.add_argument("--abc_export", action="store_true", help="Export an Alembic animation in headless mode")
	parser.add_argument("--abc_path", default="", help="Alembic output path (default: <case output>/simulation.abc)")
	parser.add_argument("--method", type=str.upper, choices=UNTANGLING_METHODS, default="PRP", help="Untangling method to use (default: PRP)")
	parser.add_argument("--prp_direction_optimization_iterations", "--prp_optimize_direction_count", "--PRP_optimize_directon_count",
						dest="prp_direction_optimization_iterations", type=int, default=0,
						help="Projected-Newton direction trials after discrete screening (default: 0 = disabled)")
	parser.add_argument("--consistent_solve", type=int, choices=[0, 1], default=None, help="Force deterministic fixed-order assembly/SpMV on the CPU physics path (default: engine default = off)")
	parser.add_argument("--json_name", type=str, default="default_scene", help="Scene path relative to Resources/Scenes, without .json")
	parser.add_argument("--output_dir", type=str, default=None, help="Output directory (default: output/paper_cases/json_smoke/<method>/<scene>)")
	parser.add_argument("--ignore_nearly_zero_distance", type=int, choices=(0, 1), default=0, help="1 = drop nearly-zero-distance EF pairs, 0 = keep adjacency-qualified degenerate candidates (default: 0)")
	parser.add_argument("--use_gpu_untangling", type=int, choices=(0, 1), default=None, help="Evaluate PRP contours with the GPU batched path (default: engine default = GPU; 0 = CPU)")
	parser.add_argument("--intrinsic_contour_side_candidates", type=int, choices=(0, 1), default=None, help="Intrinsic contour-side candidate attribution (default: engine default = on)")
	parser.add_argument("--intrinsic_tau", type=float, default=0.25, help="Phi window threshold tau for side ownership (default: 0.25)")
	parser.add_argument("--deformed_boundary_distance_for_intrinsic_candidates", type=int, choices=(0, 1), default=None, help="Require current-space ownership in addition to rest-geodesic ownership (default: engine default = on)")
	parser.add_argument("--prp_debug", type=int, choices=(0, 1), default=None, help="Enable detailed PRP debug stats (default: engine default = off)")
	parser.add_argument("--pcg_lm_adaptive", type=int, choices=(0, 1), default=None, help="Enable adaptive LM through the Newton delta gate (default: engine default = on)")
	parser.add_argument("--use_quasi_static_mode", type=int, choices=(0, 1), default=None, help="Quasi-static mode override (1 = drop velocity carry-over in prediction; default: engine default = off)")
	parser.add_argument("--qs_inertia_mode", choices=("off", "hessian_only", "full"), default=None, help="Free-DOF inertia in quasi-static mode (default: hessian_only)")
	return parser.parse_args()

args = parse_args()

# Initialize LuisaCompute device
backend = args.backend  # backends: cuda, dx, vk, metal (if supported on the platform)
solver = lcs.NewtonSolver()
solver.init_device(backend_name=backend)

# Register meshes
scene_root = os.path.realpath(os.path.join(root, "Resources", "Scenes"))
input_dir = os.path.realpath(os.path.join(scene_root, f"{args.json_name}.json"))
if os.path.commonpath((scene_root, input_dir)) != scene_root:
	raise ValueError(f"Scene must be inside {scene_root}: {args.json_name}")
if not os.path.isfile(input_dir):
	raise FileNotFoundError(f"Scene JSON not found: {input_dir}")
solver.load_scene_from_json(input_dir)

# Initialize the solver (builds internal data structures, compiles shaders, etc.)
solver.init_solver()

# Get mesh info
solver.print_registered_meshes_info()

# Set scene parameters
config_ref = solver.get_config()
configure_untangling_method(config_ref, args.method)
config_ref.PRP_direction_optimization_iterations = int(args.prp_direction_optimization_iterations)
config_ref.consistent_solve = bool(args.consistent_solve)
config_ref.untangling_response_depth = 1.0
config_ref.pcg_iter_count = 200
config_ref.use_ccd_linesearch = args.method == "PRP"
config_ref.ignore_near_zero_dist_pairs = bool(args.ignore_nearly_zero_distance)
if args.pcg_lm_adaptive is not None:
	config_ref.pcg_lm_adaptive = bool(args.pcg_lm_adaptive)
if args.use_quasi_static_mode is not None:
	config_ref.use_quasi_static_mode = bool(args.use_quasi_static_mode)
if args.qs_inertia_mode is not None:
	config_ref.qs_inertia_mode = {
		"off": lcs.QuasiStaticInertiaMode.Off,
		"hessian_only": lcs.QuasiStaticInertiaMode.HessianOnly,
		"full": lcs.QuasiStaticInertiaMode.Full,
	}[args.qs_inertia_mode]
if args.use_gpu_untangling is not None:
	config_ref.use_gpu_untangling = bool(args.use_gpu_untangling)
if args.intrinsic_contour_side_candidates is not None:
	config_ref.PRP_use_intrinsic_contour_side_candidates = bool(args.intrinsic_contour_side_candidates)
	config_ref.PRP_intrinsic_tau = float(args.intrinsic_tau)
	config_ref.PRP_use_deformed_boundary_distance_for_intrinsic_candidates = bool(args.deformed_boundary_distance_for_intrinsic_candidates)
if args.prp_debug is not None:
	config_ref.prp_debug = bool(args.prp_debug)

# Output directory (for optional file saving)
scene_slug = args.json_name.replace("\\", "_").replace("/", "_")
output_dir = args.output_dir or os.path.join(
	root, "output", "paper_cases", "json_smoke", args.method, scene_slug)
os.makedirs(output_dir, exist_ok=True)

display = DisplayInterface(solver, config_ref, args, output_dir)
display.run()

solver.cleanup_device()
