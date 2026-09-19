"""Reproduce the paper's Jang self-intersection benchmark.

The dataset is distributed by Jang et al. and is intentionally not copied into
this repository. Pass either the extracted ``Dataset_distributed`` directory or
its parent through ``--jang_dataset``. By default, selected cases open in
Polyscope one at a time; pass ``--headless`` for an unattended batch. Each case
runs in a child process so a native solver abort does not lose the batch.
"""

import argparse
import hashlib
import json
from utils.intrinsic_filter import add_blend_args, apply_blend_args, configuration_record
import os
import pickle
import platform
import subprocess
import sys
import tempfile
import time
import traceback

import numpy as np

root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, os.path.join(root, "build", "bin"))

# Bundled sample cases from the official ISIR repository (git submodule). The
# complete 59-case Dataset_distributed benchmark is available from the original
# authors (wonjong@postech.ac.kr); pass --jang_dataset to evaluate it.
DEFAULT_JANG_DATASET = os.path.join(
    root, "external", "instant-mesh-intersection-repair", "data", "misc")

from shared_args import (
    UNTANGLING_METHODS,
    add_backend_args,
    add_headless_args,
    configure_untangling_method,
    create_parser,
)
from utils.mesh_quality_metrics import compute_case_geometry_metrics
from utils.demo_runner import run_simulation


DEFAULT_OUTPUT_DIR = os.path.join(root, "output", "paper_cases", "jang59")
WORKER_REQUEST_ENV = "LCS_JANG_WORKER_REQUEST"
WORKER_RESULT_ENV = "LCS_JANG_WORKER_RESULT"


def parse_args():
    parser = create_parser(description="Jang benchmark for RayCasting Untangling")
    add_backend_args(parser, default="auto")
    add_headless_args(parser, advance_frames_default=100)
    parser.add_argument("--prp_direction_optimization_iterations", "--prp_optimize_direction_count", "--PRP_optimize_directon_count",
                        dest="prp_direction_optimization_iterations", type=int, default=0,
                        help="Projected-Newton direction trials after discrete screening (default: 0 = disabled)")
    parser.add_argument("--dataset_mode", choices=["jang"], default="jang", help=argparse.SUPPRESS)
    parser.add_argument("--methods", nargs="+", type=str.upper, choices=UNTANGLING_METHODS, default=["PRP"], help="Untangling methods to benchmark (default: PRP)")
    parser.add_argument("--jang_dataset", default=DEFAULT_JANG_DATASET,
                        help="Jang benchmark dataset directory (default: the sample cases bundled in the "
                             "instant-mesh-intersection-repair submodule; pass the extracted "
                             "Dataset_distributed directory to evaluate the full 59-case set)")
    parser.add_argument("--begin", type=int, default=1, help="1-based first case in deterministic path order")
    parser.add_argument("--end", type=int, default=0, help="1-based final case, inclusive; 0 selects all")
    parser.add_argument("--max_cases", type=int, default=0, help="Maximum selected cases after --begin/--end; 0 disables")
    parser.add_argument("--skip_names", nargs="*", default=[], help="Skip cases whose relative path contains any token")
    parser.add_argument("--timeout", type=int, default=300, help="Per-case headless worker timeout in seconds (GUI has no timeout)")
    parser.add_argument("--max_ef_pairs", type=int, default=10000, help="Abort a case above this EF count; 0 disables")
    parser.add_argument("--output_dir", "--export_obj_root", dest="output_dir", default=DEFAULT_OUTPUT_DIR, help="Report and optional frame output directory")
    parser.add_argument("--output_json_path", default=None, help="Summary JSON path (default: <output_dir>/summary.json)")
    parser.add_argument("--output_md", default=None, help="Summary Markdown path (default: <output_dir>/summary.md)")
    parser.add_argument("--export_json", type=int, default=1, help=argparse.SUPPRESS)
    parser.add_argument("--no_export_per_frame", action="store_true", help="Only save initial/resolved milestone meshes")
    parser.add_argument("--dry_run", action="store_true", help="Validate, hash, and list inputs without loading lcs_py")

    parser.add_argument("--use_gpu_pcg", type=int, choices=[0, 1], default=0)
    parser.add_argument("--use_gpu_untangling", type=int, choices=[0, 1], default=None)
    parser.add_argument("--intrinsic_contour_side_candidates", type=int, choices=[0, 1], default=None)
    parser.add_argument("--intrinsic_tau", type=float, default=0.25)
    parser.add_argument("--deformed_boundary_distance_for_intrinsic_candidates", type=int, choices=[0, 1], default=None)
    parser.add_argument("--pcg_iter_count", type=int, default=200)
    parser.add_argument("--use_ccd_linesearch", type=int, choices=[0, 1], default=1, help="Initial CCD line search (default: 1); auto-enabled below 16 contours")
    parser.add_argument("--untangling_response_depth", type=float, default=1.0)
    parser.add_argument("--PRP_cpu_contour_batch_size", type=int, default=32)
    parser.add_argument("--untangling_process_contours_count", type=int, default=256)
    parser.add_argument("--prp_debug", type=int, choices=[0, 1], default=0)
    parser.add_argument("--use_large_bending", action="store_true")
    parser.add_argument("--consistent_solve", type=int, choices=(0, 1), default=None)
    add_blend_args(parser)
    return parser.parse_args()


def collect_obj_files(dataset_root):
    paths = []
    for current_root, dir_names, file_names in os.walk(dataset_root):
        dir_names.sort()
        for file_name in sorted(file_names):
            if file_name.lower().endswith(".obj"):
                paths.append(os.path.join(current_root, file_name))
    return paths


def resolve_dataset_root(dataset_root):
    candidate = os.path.abspath(dataset_root)
    nested = os.path.join(candidate, "Dataset_distributed")
    if os.path.isdir(nested) and collect_obj_files(nested):
        candidate = nested
    hint = ""
    if os.path.abspath(dataset_root) == os.path.abspath(DEFAULT_JANG_DATASET):
        hint = (" The default dataset comes from the git submodule at "
                "external/instant-mesh-intersection-repair; run "
                "`git submodule update --init external/instant-mesh-intersection-repair`, "
                "or pass --jang_dataset pointing at an extracted Dataset_distributed directory "
                "(the full benchmark is available from the original authors, wonjong@postech.ac.kr).")
    if not os.path.isdir(candidate):
        raise FileNotFoundError(f"Jang dataset directory not found: {candidate}.{hint}")
    if not collect_obj_files(candidate):
        raise FileNotFoundError(f"No OBJ cases found under: {candidate}.{hint}")
    return candidate


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit():
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=root, check=True,
            capture_output=True, text=True)
        return completed.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def build_case_manifest(dataset_root, args):
    cases = []
    for path in collect_obj_files(dataset_root):
        case_name = os.path.splitext(os.path.relpath(path, dataset_root))[0].replace(os.sep, "/")
        if any(token in case_name for token in args.skip_names):
            continue
        cases.append({
            "case_name": case_name,
            "mesh_path": path,
            "sha256": sha256_file(path),
        })

    begin_index = max(0, int(args.begin) - 1)
    end_index = int(args.end) if int(args.end) > 0 else len(cases)
    selected = cases[begin_index:end_index]
    if args.max_cases > 0:
        selected = selected[:args.max_cases]
    if not selected:
        raise ValueError("No Jang cases selected")

    manifest_text = "\n".join(
        f"{case['case_name']}\t{case['sha256']}" for case in selected)
    return selected, hashlib.sha256(manifest_text.encode("utf-8")).hexdigest()


def to_plain_debug_maps(debug_info):
    return {
        "bool": {str(key): bool(value) for key, value in dict(debug_info.bool_stats).items()},
        "uint": {str(key): int(value) for key, value in dict(debug_info.uint_stats).items()},
        "float": {str(key): float(value) for key, value in dict(debug_info.float_stats).items()},
    }


def capture_mesh(solver, registration_id):
    vertices, faces = solver.get_object_sim_result_by_registration_id(int(registration_id))
    return {
        "registration_id": int(registration_id),
        "vertices": np.asarray(vertices, dtype=np.float64).copy(),
        "faces": np.asarray(faces, dtype=np.int32).copy(),
    }


def collision_counts(solver):
    contour_data = solver.get_intersection_contour_data()
    num_pairs = np.asarray(contour_data.get("num_pairs", np.zeros(1, dtype=np.uint32)))
    num_contours = np.asarray(contour_data.get("num_contours", np.zeros(1, dtype=np.uint32)))
    return (
        int(num_pairs.flat[0]) if num_pairs.size else 0,
        int(num_contours.flat[0]) if num_contours.size else 0,
    )


def configure_solver(config, args, method):
    config.use_floor = False
    config.pcg_iter_count = max(0, int(args.pcg_iter_count))
    config.nonlinear_iter_count = 1
    config.use_gpu = bool(args.use_gpu_pcg)
    config.gravity.x = 0.0
    config.gravity.y = 0.0
    config.gravity.z = 0.0
    config.contact_energy_type = 0
    config.use_quasi_static_mode = True
    configure_untangling_method(config, method)
    if method == "PRP":
        config.PRP_direction_optimization_iterations = int(args.prp_direction_optimization_iterations)
        config.use_gpu_untangling = bool(args.use_gpu_untangling) if args.use_gpu_untangling is not None else config.use_gpu_untangling
        config.PRP_use_intrinsic_contour_side_candidates = bool(
            args.intrinsic_contour_side_candidates) if args.intrinsic_contour_side_candidates is not None else config.PRP_use_intrinsic_contour_side_candidates
        config.PRP_intrinsic_tau = float(args.intrinsic_tau)
        config.PRP_use_deformed_boundary_distance_for_intrinsic_candidates = bool(
            args.deformed_boundary_distance_for_intrinsic_candidates) if args.deformed_boundary_distance_for_intrinsic_candidates is not None else config.PRP_use_deformed_boundary_distance_for_intrinsic_candidates
    apply_blend_args(config, args)
    config.use_ccd_linesearch = bool(args.use_ccd_linesearch)
    config.untangling_response_depth = float(args.untangling_response_depth)
    config.PRP_cpu_contour_batch_size = int(args.PRP_cpu_contour_batch_size)
    config.untangling_process_contours_count = int(args.untangling_process_contours_count)
    config.d_hat = 5e-3
    config.prp_debug = bool(args.prp_debug) and method == "PRP"
    config.collect_iteration_debug = True
    config.print_collision_info = False
    config.print_pcg_info = False


def frame_record(solver, frame):
    ef_count, contour_count = collision_counts(solver)
    record = {
        "frame": int(frame),
        "ef_pair_count": ef_count,
        "contour_count": contour_count,
    }
    try:
        debug_maps = to_plain_debug_maps(solver.get_prp_debug_info())
        float_stats = debug_maps["float"]
        record["untangling_time_ms"] = float(float_stats.get("untangling_total_time", 0.0))
        record["newton_step_inf_norm"] = float(
            float_stats.get("newton_step_inf_norm", float_stats.get("max_dq", 0.0)))
    except Exception:
        record["untangling_time_ms"] = 0.0
        record["newton_step_inf_norm"] = 0.0
    return record


def convergence_metrics(records):
    initial_ef = int(records[0]["ef_pair_count"]) if records else 0
    final_ef = int(records[-1]["ef_pair_count"]) if records else 0
    converged = bool(records and initial_ef > 0 and final_ef == 0)
    return {
        "converged": converged,
        "initial_ef_pairs": initial_ef,
        "final_ef_pairs": final_ef,
        "frames_run": int(records[-1]["frame"]) if records else 0,
        "ef_reduction_pct": (
            (initial_ef - final_ef) / initial_ef * 100.0 if initial_ef > 0 else 0.0),
    }


def run_case(args, case, method, case_output_dir, *, headless=True):
    import lcs_py as lcs

    os.makedirs(case_output_dir, exist_ok=True)
    solver = lcs.NewtonSolver()
    solver.init_device(backend_name=args.backend)
    started = time.perf_counter()
    try:
        mesh = solver.create_world_data_from_file_path(
            os.path.basename(case["mesh_path"]), case["mesh_path"])
        mesh.set_auto_scaling(True)
        mesh.set_simulation_type(lcs.MaterialType.Cloth)
        bending = 1e-1 if args.use_large_bending else 1e-2
        mesh.set_physics_material_cloth(
            stretch_model="Spring", bending_model="DihedralAngle",
            area_bending_stiffness=bending)
        registration_id = solver.register_world_data(mesh)
        solver.init_solver()
        configure_solver(solver.get_config(), args, method)

        initial_configuration = configuration_record(solver.get_config())
        reference_mesh = capture_mesh(solver, registration_id)
        solver.save_sim_result(obj_path=os.path.join(case_output_dir, "initial.obj"))
        records = []
        baseline_ef = None

        def step(frame):
            nonlocal baseline_ef
            solver.physics_step_cpu()
            record = frame_record(solver, frame)
            records.append(record)
            ef_count = record["ef_pair_count"]
            contour_count = record["contour_count"]
            if baseline_ef is None:
                baseline_ef = ef_count
            print(
                f"[{case['case_name']}] frame {frame}: "
                f"EF={ef_count}, contours={contour_count}", flush=True)

            if not args.no_export_per_frame:
                solver.save_sim_result(
                    obj_path=os.path.join(case_output_dir, f"frame_{frame:06d}.obj"))
            if args.max_ef_pairs > 0 and ef_count > args.max_ef_pairs:
                return "ef_limit"
            if ef_count == 0 and contour_count == 0:
                return "resolved"
            if baseline_ef and ef_count > 5 * baseline_ef:
                return "ef_explosion"

        stop_reason = run_simulation(
            solver, step, int(args.advance_frames), headless=headless,
            title=f"{case['case_name']} / {method}", output_dir=case_output_dir)

        current_mesh = capture_mesh(solver, registration_id)
        metrics = convergence_metrics(records)
        if metrics["converged"]:
            solver.save_sim_result(obj_path=os.path.join(case_output_dir, "resolved.obj"))
            solver.save_current_state(os.path.join(case_output_dir, "resolved.state"))
        try:
            debug_maps = to_plain_debug_maps(solver.get_prp_debug_info())
        except Exception:
            debug_maps = {"bool": {}, "uint": {}, "float": {}}

        return {
            "ok": True,
            "configuration": initial_configuration,
            "case_name": case["case_name"],
            "method": method,
            "mesh_sha256": case["sha256"],
            "elapsed_s": time.perf_counter() - started,
            "stop_reason": stop_reason,
            "convergence": metrics,
            "frame_records": records,
            "geometry_quality": compute_case_geometry_metrics(
                [reference_mesh], [current_mesh]),
            "prp_debug": debug_maps,
        }
    finally:
        solver.cleanup_device()


def worker_main():
    request_path = os.environ[WORKER_REQUEST_ENV]
    result_path = os.environ[WORKER_RESULT_ENV]
    with open(request_path, "rb") as handle:
        request = pickle.load(handle)
    args = argparse.Namespace(**request["args"])
    try:
        result = run_case(
            args, request["case"], request["method"], request["case_output_dir"],
            headless=request.get("headless", True))
    except Exception as exc:
        result = {
            "ok": False,
            "case_name": request["case"]["case_name"],
            "method": request["method"],
            "mesh_sha256": request["case"]["sha256"],
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
    with open(result_path, "wb") as handle:
        pickle.dump(result, handle)


def run_case_isolated(args, case, method, case_output_dir, *, headless=True):
    request_fd, request_path = tempfile.mkstemp(prefix="lcs_jang_", suffix=".request.pkl")
    os.close(request_fd)
    result_path = request_path.replace(".request.pkl", ".result.pkl")
    request = {
        "args": vars(args),
        "case": case,
        "method": method,
        "case_output_dir": case_output_dir,
        "headless": headless,
    }
    try:
        with open(request_path, "wb") as handle:
            pickle.dump(request, handle)
        env = os.environ.copy()
        env[WORKER_REQUEST_ENV] = request_path
        env[WORKER_RESULT_ENV] = result_path
        try:
            completed = subprocess.run(
                [sys.executable, os.path.abspath(__file__)], env=env,
                timeout=max(1, int(args.timeout)) if headless else None)
        except subprocess.TimeoutExpired:
            return {
                "ok": False,
                "case_name": case["case_name"],
                "method": method,
                "mesh_sha256": case["sha256"],
                "error": f"Timed out after {args.timeout} seconds",
            }
        if completed.returncode != 0:
            return {
                "ok": False,
                "case_name": case["case_name"],
                "method": method,
                "mesh_sha256": case["sha256"],
                "error": f"Native worker exited with code {completed.returncode}",
            }
        if not os.path.isfile(result_path):
            return {
                "ok": False,
                "case_name": case["case_name"],
                "method": method,
                "mesh_sha256": case["sha256"],
                "error": "Worker produced no result file",
            }
        with open(result_path, "rb") as handle:
            return pickle.load(handle)
    finally:
        for path in (request_path, result_path):
            try:
                if os.path.exists(path):
                    os.remove(path)
            except OSError:
                pass


def markdown_report(report):
    results = report["results"]
    converged = sum(
        1 for result in results
        if result.get("ok") and result.get("convergence", {}).get("converged"))
    elapsed = [float(result.get("elapsed_s", 0.0)) for result in results if result.get("ok")]
    lines = [
        "# Jang Benchmark Reproduction",
        "",
        f"- Git commit: `{report['provenance']['git_commit']}`",
        f"- Selected input manifest SHA-256: `{report['dataset']['manifest_sha256']}`",
        f"- Result: **{converged}/{len(results)} converged**",
        f"- Mean case time: **{np.mean(elapsed):.3f} s**" if elapsed else "- Mean case time: N/A",
        "",
        "| # | Case | Method | Input SHA-256 | EF initial | EF final | Frames | Time (s) | Status |",
        "|---:|---|---|---|---:|---:|---:|---:|---|",
    ]
    for index, result in enumerate(results, 1):
        metrics = result.get("convergence", {})
        status = "PASS" if result.get("ok") and metrics.get("converged") else "FAIL"
        if not result.get("ok"):
            status += f": {result.get('error', 'unknown error')}"
        lines.append(
            f"| {index} | `{result['case_name']}` | {result.get('method', 'N/A')} | "
            f"`{result['mesh_sha256'][:12]}` | "
            f"{metrics.get('initial_ef_pairs', 'N/A')} | {metrics.get('final_ef_pairs', 'N/A')} | "
            f"{metrics.get('frames_run', 'N/A')} | {result.get('elapsed_s', 0.0):.3f} | {status} |")
    lines.append("")
    return "\n".join(lines)


def write_report(args, dataset_root, manifest_hash, cases, results):
    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    json_path = os.path.abspath(args.output_json_path or os.path.join(output_dir, "summary.json"))
    markdown_path = os.path.abspath(args.output_md or os.path.join(output_dir, "summary.md"))
    report = {
        "schema_version": 1,
        "provenance": {
            "git_commit": git_commit(),
            "python": sys.version,
            "platform": platform.platform(),
            "backend": args.backend,
        },
        "dataset": {
            "root": dataset_root,
            "selected_case_count": len(cases),
            "manifest_sha256": manifest_hash,
            "cases": [
                {key: case[key] for key in ("case_name", "sha256")}
                for case in cases
            ],
        },
        "parameters": {
            key: value for key, value in vars(args).items()
            if key not in {"jang_dataset", "output_json_path", "output_md"}
        },
        "results": results,
    }
    for path in (json_path, markdown_path):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False, default=str)
    with open(markdown_path, "w", encoding="utf-8") as handle:
        handle.write(markdown_report(report))
    print(f"JSON report: {json_path}")
    print(f"Markdown report: {markdown_path}")
    return report


def main():
    args = parse_args()
    dataset_root = resolve_dataset_root(args.jang_dataset)
    cases, manifest_hash = build_case_manifest(dataset_root, args)
    print(f"Jang dataset: {dataset_root}")
    print(f"Selected cases: {len(cases)}")
    print(f"Input manifest SHA-256: {manifest_hash}")

    if args.dry_run:
        print(f"Methods: {', '.join(args.methods)}")
        for index, case in enumerate(cases, 1):
            print(f"[{index:02d}] {case['case_name']}  {case['sha256']}")
        return 0

    results = []
    cases_root = os.path.join(os.path.abspath(args.output_dir), "cases")
    total_runs = len(cases) * len(args.methods)
    run_index = 0
    for case in cases:
        for method in args.methods:
            run_index += 1
            print(
                f"\n[{run_index}/{total_runs}] {case['case_name']} / {method}",
                flush=True)
            case_output_dir = os.path.join(
                cases_root, method, case["case_name"].replace("/", "_"))
            result = run_case_isolated(
                args, case, method, case_output_dir, headless=args.headless)
            results.append(result)
            metrics = result.get("convergence", {})
            if result.get("ok"):
                print(
                    f"  EF {metrics.get('initial_ef_pairs')} -> {metrics.get('final_ef_pairs')} "
                    f"in {metrics.get('frames_run')} frames "
                    f"({result.get('elapsed_s', 0.0):.3f} s)")
            else:
                print(
                    f"  ERROR: {result.get('error', 'unknown error')}",
                    file=sys.stderr)
            if result.get("stop_reason") == "viewer_closed":
                break
        if results[-1].get("stop_reason") == "viewer_closed":
            break

    report = write_report(args, dataset_root, manifest_hash, cases, results)
    passed = all(
        result.get("ok") and result.get("convergence", {}).get("converged")
        for result in report["results"])
    return 0 if passed else 1


if __name__ == "__main__":
    if os.environ.get(WORKER_REQUEST_ENV) and os.environ.get(WORKER_RESULT_ENV):
        worker_main()
    else:
        raise SystemExit(main())
