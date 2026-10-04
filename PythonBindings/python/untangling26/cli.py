"""Headless single-mesh and multi-body untangling entry point."""

import argparse
import ctypes.util
from importlib.metadata import PackageNotFoundError, version
import json
import math
from pathlib import Path
import sys
import time


def positive_int(value):
    result = int(value)
    if result <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return result


def nonnegative_int(value):
    result = int(value)
    if result < 0:
        raise argparse.ArgumentTypeError("must be a nonnegative integer")
    return result


def positive_float(value):
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise argparse.ArgumentTypeError("must be finite and positive")
    return result


def make_parser():
    parser = argparse.ArgumentParser(
        prog="untangling26",
        description="Repair self-intersecting meshes or intersecting bodies without a viewer.",
        epilog="Examples: untangling26 folded.obj repaired.obj\n"
               "          untangling26 body_a.obj body_b.obj out/ --material rigid",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("inputs", nargs="+", type=Path, help="input OBJ/PLY/STL/OFF/GLB/GLTF files in shared world coordinates")
    parser.add_argument("output", type=Path, help="output OBJ, GLB, or directory (with per-body OBJ files)")
    parser.add_argument("--material", choices=("cloth", "rigid"), default="cloth",
                        help="cloth permits surface deformation; rigid separates bodies (default: cloth)")
    parser.add_argument("--split-components", action=argparse.BooleanOptionalAction, default=None,
                        help="register disconnected components as separate bodies (default: on for rigid)")
    parser.add_argument("--backend", choices=("auto", "cuda", "vk", "dx", "metal"), default="auto",
                        help="compute backend available in the installed wheel (default: auto)")
    parser.add_argument("--physics", choices=("cpu", "gpu"), default="cpu",
                        help="Newton/PCG path; both require a compute device (default: cpu)")
    parser.add_argument("--gpu-untangling", action="store_true", help="evaluate PRP contours on the GPU")
    parser.add_argument("--max-iters", type=nonnegative_int, default=200,
                        help="maximum repair updates; 0 checks and exports the input (default: 200)")
    parser.add_argument("--response-depth", type=positive_float, default=1.0,
                        help="maximum requested correction in normalized scene units (default: 1)")
    parser.add_argument("--direction-iters", type=nonnegative_int, default=0,
                        help="direction refinement trials per Newton iteration (default: 0)")
    parser.add_argument("--pcg-iters", type=positive_int, default=200, help="PCG iteration budget (default: 200)")
    parser.add_argument("--max-contours", type=positive_int, default=256,
                        help="contours processed per repair update (default: 256)")
    parser.add_argument("--ccd", action=argparse.BooleanOptionalAction, default=True,
                        help="CCD step-size checking (default: on)")
    parser.add_argument("--overwrite", action="store_true", help="replace existing outputs; input files are always protected")
    try:
        installed_version = version("untangling26")
    except PackageNotFoundError:
        installed_version = "source"
    parser.add_argument("--version", action="version", version=f"%(prog)s {installed_version}")
    return parser


def select_backend(lcs, requested):
    if requested != "auto":
        return requested
    runtime = Path(lcs.__file__).resolve().parent
    available = set()
    for name in ("cuda", "vk", "dx", "metal"):
        if any(runtime.glob(f"*backend-{name}.*")):
            available.add(name)
    if sys.platform == "darwin":
        preferences = ("metal", "vk")
    elif ctypes.util.find_library("nvcuda" if sys.platform == "win32" else "cuda"):
        preferences = ("cuda", "vk", "dx")
    else:
        preferences = ("vk", "dx", "cuda")
    for name in preferences:
        if name in available:
            return name
    raise RuntimeError("No supported compute backend found beside lcs_py; install a complete platform wheel or specify --backend")


def configure_solver(config, args):
    config.use_untangling = True
    config.use_untangling_PRP = True
    config.use_untangling_ICM = False
    config.use_untangling_GIA = False
    config.use_self_collision = True
    config.use_floor = False
    config.gravity.x = config.gravity.y = config.gravity.z = 0.0
    config.use_quasi_static_mode = True
    config.use_static_mode = False
    config.fix_scene = False
    if config.num_substep != 1:
        raise RuntimeError("CLI validation requires the default single-substep configuration")
    config.nonlinear_iter_count = 1
    config.ignore_near_zero_dist_pairs = False
    config.contact_energy_type = 0  # Quadratic contacts for initially intersecting surfaces.
    config.use_ccd_linesearch = args.ccd
    config.use_energy_linesearch = False
    config.use_gpu_untangling = args.gpu_untangling
    config.PRP_solid_interior_adjacency = True
    config.PRP_direction_optimization_iterations = args.direction_iters
    config.untangling_response_depth = args.response_depth
    config.untangling_process_contours_count = args.max_contours
    config.pcg_iter_count = args.pcg_iters
    config.print_collision_info = False
    config.print_pcg_info = False


def run_repair(solver, args):
    """Return the exact snapshot associated with the collision report.

    lcs_py reports intersections at Newton iteration START, not at the positions
    returned after physics_step(). This static, unanimated configuration performs
    exactly one Newton iteration per call. Retain its input snapshot, and discard
    the speculative update on success or at the budget. There is one extra solver
    call to validate the last accepted update, without needing new native bindings.
    """
    import numpy as np

    step = solver.physics_step_cpu if args.physics == "cpu" else solver.physics_step_gpu
    records = []
    for iteration in range(args.max_iters + 1):
        vertices, faces = solver.get_sim_result()
        snapshot = ([np.array(v, copy=True) for v in vertices],
                    [np.array(f, copy=True) for f in faces])
        step()
        data = solver.get_intersection_contour_data()
        # Missing reports must never be interpreted as successful repair.
        ef = int(np.asarray(data["num_pairs"]).reshape(-1)[0])
        contours = int(np.asarray(data["num_contours"]).reshape(-1)[0])
        if ef < 0 or contours < 0:
            raise RuntimeError("The solver returned an invalid intersection report")
        records.append({"iteration": iteration, "ef_pairs": ef, "contours": contours})
        print(f"Iteration {iteration}: {ef} EF intersections, {contours} contours", flush=True)
        if ef == 0 and contours == 0:
            return snapshot, records, True
    return snapshot, records, False


def repair(args):
    import numpy as np
    from .mesh_io import (Body, check_outputs, load_bodies, normalize_bodies,
                          output_paths, validate_mesh, write_mesh)

    split = args.material == "rigid" if args.split_components is None else args.split_components
    bodies = load_bodies(args.inputs, split=split)
    mesh_paths, report_path = output_paths(args.output, bodies)
    check_outputs([*mesh_paths, report_path], args.inputs, overwrite=args.overwrite)
    normalized, center, scale = normalize_bodies(bodies)
    try:
        import lcs_py as lcs
    except ImportError as exc:
        raise RuntimeError("Cannot import the native solver. Install the complete untangling26 wheel for this Python/platform.") from exc
    backend = select_backend(lcs, args.backend)
    print(f"Untangling {len(bodies)} bodies with {args.material} material; backend={backend}", flush=True)
    solver = lcs.NewtonSolver()
    start = time.perf_counter()
    try:
        solver.init_device(backend_name=backend)
        for body in normalized:
            world = solver.create_world_data_from_array(body.name, body.vertices, body.faces)
            world.set_auto_scaling(False)
            if args.material == "rigid":
                world.set_physics_material_rigid(thickness=1e-3, stiffness=1e4)
            else:
                world.set_physics_material_cloth()
            solver.register_world_data(world)
        solver.init_solver()
        configure_solver(solver.get_config(), args)
        (vertices, faces), records, converged = run_repair(solver, args)
    finally:
        solver.cleanup_device()
    if len(vertices) != len(bodies) or len(faces) != len(bodies):
        raise RuntimeError("Solver output does not match the registered body count")
    repaired = []
    for body, v, f in zip(bodies, vertices, faces):
        v, f = validate_mesh(np.asarray(v, dtype=np.float64) * scale + center, f, body.name)
        repaired.append(Body(body.name, v, f, body.source))
    write_mesh(mesh_paths[0], repaired)
    for path, body in zip(mesh_paths[1:], repaired):
        write_mesh(path, [body])
    report = {
        "inputs": [str(p.resolve()) for p in args.inputs],
        "outputs": [str(p) for p in mesh_paths],
        "bodies": [{"name": b.name, "source": b.source, "vertices": len(b.vertices), "faces": len(b.faces)} for b in repaired],
        "converged": converged,
        "stop_reason": "zero_intersections" if converged else "iteration_budget",
        "initial_ef_pairs": records[0]["ef_pairs"],
        "final_ef_pairs": records[-1]["ef_pairs"],
        "iterations": records[-1]["iteration"],
        "solver_calls": len(records),
        "elapsed_seconds": time.perf_counter() - start,
        "configuration": {"backend": backend, "material": args.material,
                          "split_components": split, "physics": args.physics,
                          "gpu_untangling": args.gpu_untangling, "ccd": args.ccd,
                          "max_iters": args.max_iters, "response_depth": args.response_depth,
                          "direction_iters": args.direction_iters, "pcg_iters": args.pcg_iters,
                          "max_contours": args.max_contours},
        "normalization": {"center": center.tolist(), "scale": scale, "largest_extent": 1.0},
        "history": records,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    status = "Resolved" if converged else "Iteration budget reached with remaining intersections"
    print(f"{status}. Mesh: {mesh_paths[0]}\nReport: {report_path}", flush=True)
    return 0 if converged else 1


def main(argv=None):
    args = make_parser().parse_args(argv)
    try:
        return repair(args)
    except (ImportError, OSError, ValueError, RuntimeError, KeyError, IndexError) as exc:
        print(f"untangling26: {exc}", file=sys.stderr)
        return 2
