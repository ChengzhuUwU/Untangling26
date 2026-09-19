"""Benchmark script for PRP Solid Interior Adjacency on Thingi10K rigid/solid bodies.

This script tests untangling of penetrating watertight solid bodies (MaterialType::Rigid)
from Thingi10K, comparing baseline (PRP_solid_interior_adjacency = False) vs
the solid interior adjacency feature (PRP_solid_interior_adjacency = True) on both CPU and GPU.
The default GUI previews one selected case/configuration; --headless runs the
complete comparison in isolated workers.
"""

import os
import sys
import time
import json
import argparse
import subprocess
import numpy as np

root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, os.path.join(root, "build", "bin"))
from utils.demo_runner import run_simulation
from utils.intrinsic_filter import add_blend_args, apply_blend_args, configuration_record


def create_rigid_penetration_scene(solver, mesh_path, offset=(0.0, 0.0, 0.0), scale=1.0, rot1=(0.0, 0.0, 0.0), rot2=(0.0, 0.0, 0.0)):
    """Create a two-body rigid penetration scenario using a watertight solid mesh."""
    import lcs_py as lcs

    # Body 1: Base object at origin
    mesh1 = solver.create_world_data_from_file_path("rigid_body_1", mesh_path)
    mesh1.set_auto_scaling(True)
    mesh1.set_simulation_type(lcs.MaterialType.Rigid)
    mesh1.set_physics_material_rigid(thickness=1e-3, stiffness=1e4)
    mesh1.set_scale(float(scale))
    mesh1.set_rotation(*rot1)
    mesh1.set_translation(0.0, 0.0, 0.0)
    id1 = solver.register_world_data(mesh1)

    # Body 2: Penetrating object
    mesh2 = solver.create_world_data_from_file_path("rigid_body_2", mesh_path)
    mesh2.set_auto_scaling(True)
    mesh2.set_simulation_type(lcs.MaterialType.Rigid)
    mesh2.set_physics_material_rigid(thickness=1e-3, stiffness=1e4)
    mesh2.set_scale(float(scale))
    mesh2.set_rotation(*rot2)
    mesh2.set_translation(*offset)
    id2 = solver.register_world_data(mesh2)

    return id1, id2


def collision_counts(solver):
    contour_data = solver.get_intersection_contour_data()
    num_pairs = np.asarray(contour_data.get("num_pairs", np.zeros(1, dtype=np.uint32)))
    num_contours = np.asarray(contour_data.get("num_contours", np.zeros(1, dtype=np.uint32)))
    return (
        int(num_pairs.flat[0]) if num_pairs.size else 0,
        int(num_contours.flat[0]) if num_contours.size else 0,
    )


def run_experiment(mesh_path, case_name, offset, rot2, use_gpu, solid_adj, max_frames=30, backend="cuda", *, headless=True, options=None):
    import lcs_py as lcs

    solver = lcs.NewtonSolver()
    solver.init_device(backend_name=backend)

    try:
        id1, id2 = create_rigid_penetration_scene(
            solver, mesh_path, offset=offset, rot2=rot2
        )

        solver.init_solver()
        config = solver.get_config()
        config.use_floor = False
        config.gravity.x = 0.0
        config.gravity.y = 0.0
        config.gravity.z = 0.0
        config.use_quasi_static_mode = True
        config.use_untangling = True
        config.use_untangling_PRP = True
        config.use_gpu_untangling = use_gpu
        config.PRP_solid_interior_adjacency = solid_adj
        config.pcg_iter_count = 100
        config.untangling_response_depth = 1.0
        config.d_hat = 5e-3
        config.prp_debug = True

        if options is not None:
            apply_blend_args(config, options)
            config.PRP_intrinsic_tau = options.intrinsic_tau
            config.PRP_use_intrinsic_contour_side_candidates = True
            config.PRP_use_deformed_boundary_distance_for_intrinsic_candidates = True
            config.untangling_process_contours_count = options.untangling_process_contours_count
            config.prp_debug = bool(options.prp_debug)
        initial_configuration = configuration_record(config)
        if options is not None and options.output_dir:
            os.makedirs(options.output_dir, exist_ok=True)
            solver.save_sim_result(obj_path=os.path.join(options.output_dir, "initial.obj"))
        records = []
        converged = False
        initial_ef = 0
        elapsed = 0.0

        def step(frame):
            nonlocal initial_ef, converged, elapsed
            start_time = time.perf_counter()
            solver.physics_step_cpu()
            elapsed += time.perf_counter() - start_time

            ef, contours = collision_counts(solver)
            if frame == 1:
                initial_ef = ef
                print(f"[{case_name}] SolidAdj={solid_adj} GPU={use_gpu} | Initial EF={initial_ef}, Contours={contours}", flush=True)

            records.append({"frame": frame, "ef": ef, "contours": contours})
            print(f"    frame {frame:2d}: EF={ef:4d}, contours={contours:2d}", flush=True)

            if options is not None and options.max_ef_pairs > 0 and ef > options.max_ef_pairs:
                return "ef_limit"
            if ef == 0 and contours == 0:
                converged = True
                return "resolved"

        stop_reason = run_simulation(
            solver, step, max_frames, headless=headless,
            title=f"{case_name} / GPU={use_gpu} / SolidAdj={solid_adj}")
        final_ef = records[-1]["ef"] if records else initial_ef
        ef_reduction = (initial_ef - final_ef) / max(initial_ef, 1) * 100.0

        result = {
            "case_name": case_name,
            "configuration": initial_configuration,
            "frame_records": records,
            "solid_adj": solid_adj,
            "use_gpu": use_gpu,
            "initial_ef": initial_ef,
            "final_ef": final_ef,
            "ef_reduction_pct": ef_reduction,
            "converged": converged,
            "frames": len(records),
            "elapsed_s": elapsed,
            "stop_reason": stop_reason,
        }
        print(f"  --> Result: Converged={converged}, Final EF={final_ef} (Reduction: {ef_reduction:.1f}%), Frames={len(records)}, Time={elapsed:.2f}s", flush=True)
        if options is not None and options.output_dir:
            solver.save_sim_result(obj_path=os.path.join(options.output_dir, "final.obj"))
            with open(os.path.join(options.output_dir, "summary.json"), "w", encoding="utf-8") as handle:
                json.dump(result, handle, indent=2)
        return result
    finally:
        try:
            solver.cleanup_device()
        except Exception:
            pass


def run_isolated(mesh_path, case_name, offset, rot2, use_gpu, solid_adj, max_frames, backend, options=None):
    cmd = [
        sys.executable,
        os.path.abspath(__file__),
        "--worker",
        "--headless",
        "--mesh_path", mesh_path,
        "--case_name", case_name,
        "--offset", str(offset[0]), str(offset[1]), str(offset[2]),
        "--rot2", str(rot2[0]), str(rot2[1]), str(rot2[2]),
        "--use_gpu", "1" if use_gpu else "0",
        "--solid_adj", "1" if solid_adj else "0",
        "--max_frames", str(max_frames),
        "--backend", backend,
    ]
    if options is not None:
        cmd += ["--intrinsic_filter_mode", options.intrinsic_filter_mode,
                "--intrinsic_blend_weight", str(options.intrinsic_blend_weight),
                "--intrinsic_tau", str(options.intrinsic_tau),
                "--prp_debug", str(options.prp_debug),
                "--max_ef_pairs", str(options.max_ef_pairs),
                "--untangling_process_contours_count", str(options.untangling_process_contours_count)]
        if options.consistent_solve is not None:
            cmd += ["--consistent_solve", str(options.consistent_solve)]
        if options.rest_geodesic_distance_for_intrinsic_candidates is not None:
            cmd += ["--rest_geodesic_distance_for_intrinsic_candidates", str(options.rest_geodesic_distance_for_intrinsic_candidates)]
        if options.output_dir:
            cmd += ["--output_dir", os.path.join(options.output_dir, case_name)]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    out = proc.stdout
    result = None
    for line in out.splitlines():
        if line.startswith("JSON_RESULT:"):
            try:
                result = json.loads(line[len("JSON_RESULT:"):].strip())
            except Exception:
                pass
    if result is None:
        print(f"[Worker Error for {case_name}] Return code {proc.returncode}:\n{out}", flush=True)
        result = {
            "case_name": case_name,
            "solid_adj": solid_adj,
            "use_gpu": use_gpu,
            "initial_ef": -1,
            "final_ef": -1,
            "ef_reduction_pct": 0.0,
            "converged": False,
            "frames": 0,
            "elapsed_s": 0.0,
            "error": out[-500:],
        }
    else:
        for line in out.splitlines():
            if line.startswith("    frame") or line.startswith("  --> Result") or line.startswith("["):
                print("  " + line, flush=True)
    return result


def benchmark_cases():
    thingi_dir = os.path.join(root, "Resources", "InputMesh", "thingi10k")
    return [
        {
            "name": "101560_VacuumFormerSmall",
            "mesh": os.path.join(thingi_dir, "101560_vacuum_former_small.obj"),
            "offset": (0.12, 0.05, 0.0),
            "rot2": (15.0, 30.0, 0.0),
        },
        {
            "name": "1020669_SDHolderClip",
            "mesh": os.path.join(thingi_dir, "1020669_sd_holder_clip.obj"),
            "offset": (0.08, 0.05, 0.0),
            "rot2": (20.0, 45.0, 0.0),
        },
        {
            "name": "1036312_EnginePart508",
            "mesh": os.path.join(thingi_dir, "1036312_engine_part_508.obj"),
            "offset": (0.12, 0.06, 0.0),
            "rot2": (10.0, 25.0, 0.0),
        },
        {
            "name": "101550_VacuumFormer660",
            "mesh": os.path.join(thingi_dir, "101550_vacuum_former_660.obj"),
            "offset": (0.15, 0.05, 0.0),
            "rot2": (30.0, 20.0, 0.0),
        },
        {
            "name": "1036309_EngineRocker963",
            "mesh": os.path.join(thingi_dir, "1036309_engine_part_963.obj"),
            "offset": (0.10, 0.08, 0.0),
            "rot2": (15.0, 40.0, 0.0),
        },
        {
            "name": "1016856_FlexyHand990",
            "mesh": os.path.join(thingi_dir, "1016856_flexy_hand_990.obj"),
            "offset": (0.12, 0.04, 0.0),
            "rot2": (25.0, 15.0, 0.0),
        },
        {
            "name": "102248_TankTread1124",
            "mesh": os.path.join(thingi_dir, "102248_tank_tread_1124.obj"),
            "offset": (0.14, 0.05, 0.0),
            "rot2": (20.0, 30.0, 0.0),
        },
        {
            "name": "1018276_SDHolderBracket1330",
            "mesh": os.path.join(thingi_dir, "1018276_sd_holder_bracket.obj"),
            "offset": (0.10, 0.06, 0.0),
            "rot2": (15.0, 35.0, 0.0),
        },
    ]


def main():
    parser = argparse.ArgumentParser(description="Thingi10K PRP Solid Interior Adjacency Benchmark")
    parser.add_argument("--backend", default="cuda", help="LuisaCompute backend (cuda, dx, cpu)")
    parser.add_argument("--max_frames", type=int, default=25, help="Max frames per test")
    parser.add_argument("--headless", action="store_true", help="Run the full benchmark without Polyscope (default: preview one case in GUI)")
    parser.add_argument("--case_index", type=int, choices=range(1, 9), default=1, help="1-based case to preview in GUI (default: 1)")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--mesh_path", type=str, help="Override the selected GUI mesh (also used by internal workers)")
    parser.add_argument("--case_name", type=str, help="Override the GUI case label")
    parser.add_argument("--offset", type=float, nargs=3, help="Override the second body's translation in GUI")
    parser.add_argument("--rot2", type=float, nargs=3, help="Override the second body's rotation in GUI")
    parser.add_argument("--use_gpu", type=int, default=0, help="Use GPU untangling")
    parser.add_argument("--solid_adj", type=int, default=0, help="Enable solid interior adjacency")
    add_blend_args(parser)
    parser.add_argument("--intrinsic_tau", type=float, default=0.25)
    parser.add_argument("--consistent_solve", type=int, choices=(0, 1), default=None)
    parser.add_argument("--prp_debug", type=int, choices=(0, 1), default=1)
    parser.add_argument("--untangling_process_contours_count", type=int, default=256)
    parser.add_argument("--max_ef_pairs", type=int, default=0)
    parser.add_argument("--output_dir", default=None)
    args = parser.parse_args()

    if args.worker:
        res = run_experiment(
            mesh_path=args.mesh_path,
            case_name=args.case_name,
            offset=args.offset,
            rot2=args.rot2,
            use_gpu=bool(args.use_gpu),
            solid_adj=bool(args.solid_adj),
            max_frames=args.max_frames,
            backend=args.backend,
            headless=True, options=args,
        )
        print("JSON_RESULT:" + json.dumps(res), flush=True)
        return

    test_cases = benchmark_cases()

    if not args.headless:
        case = test_cases[args.case_index - 1]
        mesh_path = args.mesh_path or case["mesh"]
        if not os.path.isfile(mesh_path):
            raise FileNotFoundError(f"Input mesh not found: {mesh_path}")
        result = run_experiment(
            mesh_path, args.case_name or case["name"],
            args.offset if args.offset is not None else case["offset"],
            args.rot2 if args.rot2 is not None else case["rot2"],
            bool(args.use_gpu), bool(args.solid_adj), args.max_frames,
            args.backend, headless=False, options=args)
        print("JSON_RESULT:" + json.dumps(result), flush=True)
        return

    all_results = []
    print("=" * 80)
    print(" Thingi10K Rigid Untangling Benchmark: Baseline vs Solid Interior Adjacency")
    print("=" * 80)

    for case in test_cases:
        mesh_path = case["mesh"]
        if not os.path.exists(mesh_path):
            print(f"Skipping {case['name']} (mesh not found: {mesh_path})")
            continue

        print(f"\n>>> Running Case: {case['name']}")
        modes = [
            ("CPU_Baseline", False, False),
            ("CPU_SolidAdj", False, True),
            ("GPU_Baseline", True, False),
            ("GPU_SolidAdj", True, True),
        ]

        for mode_name, use_gpu, solid_adj in modes:
            tag = f"{case['name']}_{mode_name}"
            print(f"\n--- [{mode_name}] {case['name']} ---")
            res = run_isolated(
                mesh_path=mesh_path,
                case_name=tag,
                offset=case["offset"],
                rot2=case["rot2"],
                use_gpu=use_gpu,
                solid_adj=solid_adj,
                max_frames=args.max_frames,
                backend=args.backend, options=args,
            )
            res["mode"] = mode_name
            res["model_name"] = case["name"]
            all_results.append(res)

    # Print summary table
    print("\n" + "=" * 105)
    print(f"{'Case Name':<30} | {'Mode':<14} | {'Init EF':<8} | {'Final EF':<8} | {'Reduct %':<9} | {'Conv':<5} | {'Frames':<6} | {'Time (s)':<8}")
    print("-" * 105)
    for r in all_results:
        conv_str = "YES" if r.get("converged") else "NO"
        red_str = f"{r.get('ef_reduction_pct', 0.0):.1f}%"
        print(f"{r.get('model_name', ''):<30} | {r.get('mode', ''):<14} | {r.get('initial_ef', 0):<8} | {r.get('final_ef', 0):<8} | {red_str:<9} | {conv_str:<5} | {r.get('frames', 0):<6} | {r.get('elapsed_s', 0.0):<8.2f}")
    print("=" * 105)

    # Dump JSON
    json_path = os.path.join(args.output_dir, "summary.json") if args.output_dir else os.path.join(root, "Resources", "thingi10k_rigid_benchmark_results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    print(f"Results saved to {json_path}")


if __name__ == "__main__":
    main()
