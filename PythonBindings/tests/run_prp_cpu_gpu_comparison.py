"""Run matched PRP paths through the retained case drivers in isolated processes."""

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "PythonBindings" / "tests"
CONFIG_FIELDS = (
    "use_gpu", "use_gpu_untangling", "nonlinear_iter_count", "pcg_iter_count",
    "pcg_lm_adaptive",
    "use_ccd_linesearch", "untangling_response_depth",
    "PRP_use_intrinsic_contour_side_candidates", "PRP_intrinsic_tau",
    "PRP_use_deformed_boundary_distance_for_intrinsic_candidates",
    "PRP_direction_optimization_iterations", "PRP_cpu_contour_batch_size",
    "untangling_process_contours_count",
)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def worker(request_path):
    import numpy as np

    request = json.loads(Path(request_path).read_text(encoding="utf-8"))
    if request.get("worker_priority") == "above_normal":
        import ctypes
        from ctypes import wintypes
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.GetCurrentProcess.restype = wintypes.HANDLE
        kernel32.SetPriorityClass.argtypes = [wintypes.HANDLE, wintypes.DWORD]
        if not kernel32.SetPriorityClass(kernel32.GetCurrentProcess(), 0x8000):
            raise ctypes.WinError(ctypes.get_last_error())
    out = Path(request["output"])
    binding = Path(request["binding"])
    dll_handles = []
    if os.name == "nt":
        for directory in (ROOT / "build" / "bin", binding.parent):
            dll_handles.append(os.add_dll_directory(str(directory)))
    spec = importlib.util.spec_from_file_location("lcs_py", binding)
    lcs = importlib.util.module_from_spec(spec)
    sys.modules["lcs_py"] = lcs
    spec.loader.exec_module(lcs)
    sys.path.insert(0, str(TESTS))
    native_init = lcs.NewtonSolver.init_device

    def init_device(self, backend_name="cuda", binary_path=None):
        return native_init(self, backend_name, str(ROOT / "build" / "bin" / binding.name))

    lcs.NewtonSolver.init_device = init_device
    steps = []

    def save_geometry(solver, name):
        vertices, faces = solver.get_sim_result()
        arrays = {}
        for i, (v, f) in enumerate(zip(vertices, faces)):
            arrays[f"vertices_{i}"] = np.asarray(v).copy()
            arrays[f"faces_{i}"] = np.asarray(f).copy()
        np.savez_compressed(out / name, **arrays)

    def wrap_step(native):
        def step(self):
            config = self.get_config()
            first = not steps
            previous_debug = bool(config.prp_debug)
            if first:
                config.use_gpu = True
                config.use_gpu_untangling = request["path"] == "gpu"
                config.nonlinear_iter_count = 1
                config.pcg_lm_adaptive = not request.get("single_iteration", False)
                config.PRP_use_intrinsic_contour_side_candidates = True
                config.PRP_use_deformed_boundary_distance_for_intrinsic_candidates = True
                config.PRP_intrinsic_tau = 0.25
                config.PRP_direction_optimization_iterations = 0
                config.collect_iteration_debug = True
                if request.get("single_iteration", False):
                    config.untangling_response_depth = 0.005
                write_json(out / "effective_config.json", {
                    "binding": str(binding), "binding_sha256": digest(binding),
                    **{key: getattr(config, key) for key in CONFIG_FIELDS},
                })
                save_geometry(self, "initial_geometry.npz")
                config.prp_debug = True
            started = time.perf_counter()
            result = native(self)
            elapsed = time.perf_counter() - started
            data = self.get_intersection_contour_data()
            debug = self.get_prp_debug_info()
            floats = dict(debug.float_stats)
            uints = dict(debug.uint_stats)
            ef_pairs = int(np.asarray(data["num_pairs"]).flat[0])
            record = {
                "step": len(steps) + 1, "frame": int(config.current_frame),
                "ef_pairs": ef_pairs,
                "contours": int(np.asarray(data["num_contours"]).flat[0]),
                "step_s": elapsed,
                "newton_step_inf_norm": float(floats.get("newton_step_inf_norm", floats.get("max_dq", 0.0))),
                "ray_max_task_hits": int(uints.get("prp_ray_max_task_hits", 0)) if ef_pairs else 0,
                "ray_max_task_capacity": int(uints.get("prp_ray_max_task_capacity", 0)) if ef_pairs else 0,
            }
            steps.append(record)
            with (out / "trajectory.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record) + "\n")
            if first:
                maps = {"uint": uints, "float": floats, "bool": dict(debug.bool_stats)}
                (out / "first_iteration_debug.json").write_text(json.dumps(maps), encoding="utf-8")
                save_geometry(self, "first_iteration_geometry.npz")
                config.prp_debug = previous_debug
            if record["ef_pairs"] == 0 or len(steps) == request["frames"]:
                save_geometry(self, "final_geometry.npz")
            print("COMPARISON " + json.dumps(record), flush=True)
            return result
        return step

    native_gpu_step = lcs.NewtonSolver.physics_step_gpu
    for method in ("physics_step_cpu", "physics_step_gpu"):
        setattr(lcs.NewtonSolver, method, wrap_step(native_gpu_step))

    if request["suite"] == "json":
        from utils.display_interface import DisplayInterface
        native_after_step = DisplayInterface.after_step

        def after_step(self):
            native_after_step(self)
            return bool(steps and steps[-1]["ef_pairs"] == 0)

        DisplayInterface.after_step = after_step

    if request["suite"] == "synthetic" and request.get("resume_failed_synthetic_audit"):
        from utils.display_interface import DisplayInterface
        native_run = DisplayInterface.run

        def run_with_audit_continuation(self, *args, **kwargs):
            native_after_step = self.after_step
            attempts = []

            def after_step():
                stop = native_after_step()
                audit_path = out / "lasting_zero_audit.json"
                if not stop or not audit_path.exists():
                    return stop
                audit = json.loads(audit_path.read_text(encoding="utf-8"))
                attempts.append(audit)
                write_json(out / f"lasting_zero_attempt_{len(attempts):03d}.json", audit)
                if audit["status"] != "failed":
                    return stop
                stiffness = self.untangling_response_stiffness_before_validation
                if stiffness is None:
                    raise RuntimeError("Cannot resume Synthetic: original response stiffness is missing")
                self.config_ref.stiffness_untangling = stiffness
                self.first_zero_frame = None
                self.zero_validation_samples = []
                self.untangling_response_stiffness_before_validation = None
                print("Resuming Synthetic untangling after a failed zero audit", flush=True)
                return False

            self.after_step = after_step
            try:
                return native_run(self, *args, **kwargs)
            finally:
                write_json(out / "lasting_zero_attempts.json", {
                    "resume_after_failed_audit": True,
                    "first_zero_step": next((s["step"] for s in steps if s["ef_pairs"] == 0), None),
                    "lasting_zero": bool(self.lasting_zero_validated),
                    "attempts": attempts,
                })

        DisplayInterface.run = run_with_audit_continuation

    script = TESTS / request["script"]
    sys.argv = [str(script), *request["args"]]
    if request["suite"] == "jang":
        import test_batch_method_comparison_enhanced as jang
        args = jang.parse_args()
        result = jang.run_case(args, request["case"], "PRP", str(out))
        write_json(out / "case_summary.json", result)
    else:
        runpy.run_path(str(script), run_name="__main__")


def requests(args):
    common = ["--backend", "cuda", "--headless"]
    large = ["--use_gpu_pcg", "1", "--use_ccd_linesearch", "0",
             "--untangling_response_depth", "0.005", "--untangling_process_contours_count", "256",
             "--prp_debug", "0"]
    for path in args.paths:
        path_args = ["--use_gpu_untangling", str(int(path == "gpu"))]
        if "unit" in args.suites or "unit_raw" in args.suites:
            for raw in (True, False):
                suite = "unit_raw" if raw else "unit"
                if suite not in args.suites:
                    continue
                for scene, name in enumerate(("BBII", "BIBI", "BLI", "Closed", "Cross", "Eight", "LL")):
                    frames = 1 if raw else args.unit_frames
                    yield dict(path=path, suite=suite, name=name, frames=frames,
                               single_iteration=raw, script="demo_unit.py",
                               args=common + path_args + ["--scene_id", str(scene), "--advance_frames", str(frames)]
                               + ([] if raw else ["--use_subdivision", "--subdiv_levels", "3"]))
        if "json" in args.suites:
            scenes = ["default_scene", "Penetrations/2_cloth_conflict", "Penetrations/cloth_sorento",
                      "Penetrations/cloth_torus", "Penetrations/plane_cylinder",
                      "Penetrations/self_collision_from_bizier"]
            for scene in scenes:
                yield dict(path=path, suite="json", name=scene.split("/")[-1], frames=args.unit_frames,
                           script="test_load_from_json_untangling.py",
                           args=common + path_args + ["--json_name", scene, "--advance_frames", str(args.unit_frames), "--prp_debug", "0"])
        if "jang" in args.suites:
            sys.path.insert(0, str(TESTS))
            import test_batch_method_comparison_enhanced as jang
            native_argv = sys.argv
            # Default to the sample cases bundled in the instant-mesh-intersection-repair
            # submodule unless an explicit dataset directory is provided.
            dataset_arg = str(args.jang_dataset) if args.jang_dataset else jang.DEFAULT_JANG_DATASET
            case_args = ["--backend", "cuda", "--jang_dataset", dataset_arg, "--begin", "1", "--end", "59",
                         "--advance_frames", str(args.jang_frames), "--no_export_per_frame"] + path_args + large
            sys.argv = [str(TESTS / jang.__name__), *case_args]
            try:
                parsed = jang.parse_args()
                cases, manifest_hash = jang.build_case_manifest(jang.resolve_dataset_root(parsed.jang_dataset), parsed)
            finally:
                sys.argv = native_argv
            for case in cases:
                yield dict(path=path, suite="jang", name=case["case_name"], frames=args.jang_frames,
                           script="test_batch_method_comparison_enhanced.py", case=case,
                           dataset_manifest_sha256=manifest_hash, args=case_args)
        if "synthetic" in args.suites:
            yield dict(path=path, suite="synthetic", name="fold", frames=args.synthetic_frames,
                       script="demo_synthetic.py", args=common + path_args + large +
                       ["--advance_frames", str(args.synthetic_frames), "--start_from_initial", "1",
                        "--max_ef_pairs", "200000", "--lasting_zero_validation_steps", "5", "--output_frequency", "25"])
        if "dress" in args.suites:
            yield dict(path=path, suite="dress", name="Drop3", frames=args.dress_frames,
                       script="demo_dropping3.py", args=common + path_args + large +
                       ["--advance_frames", str(args.dress_frames), "--PRP_cpu_contour_batch_size", "32", "--no_export_per_frame"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker")
    parser.add_argument("--binding", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--jang_dataset", type=Path, help="Jang dataset directory; defaults to the sample cases bundled in the instant-mesh-intersection-repair submodule")
    parser.add_argument("--suites", nargs="+", default=["unit_raw", "unit", "json", "jang", "synthetic", "dress"])
    parser.add_argument("--paths", nargs="+", choices=["cpu", "gpu"], default=["cpu", "gpu"])
    parser.add_argument("--unit_frames", type=int, default=300)
    parser.add_argument("--jang_frames", type=int, default=100)
    parser.add_argument("--synthetic_frames", type=int, default=600)
    parser.add_argument("--dress_frames", type=int, default=300)
    parser.add_argument("--timeout", type=int, default=3600)
    parser.add_argument("--worker_priority", choices=["normal", "above_normal"], default="normal")
    parser.add_argument("--resume_failed_synthetic_audit", action="store_true")
    args = parser.parse_args()
    if args.worker:
        worker(args.worker)
        return
    if not args.binding or not args.output:
        parser.error("--binding and --output are required")
    if args.worker_priority == "above_normal" and os.name != "nt":
        parser.error("--worker_priority above_normal requires Windows")
    args.output.mkdir(parents=True, exist_ok=True)
    results = []
    for request in requests(args):
        name = request["name"].replace("/", "_").replace("\\", "_")
        out = args.output.resolve() / request["suite"] / request["path"] / name
        out.mkdir(parents=True, exist_ok=True)
        request.update(binding=str(args.binding.resolve()), output=str(out))
        request["worker_priority"] = args.worker_priority
        request["resume_failed_synthetic_audit"] = bool(args.resume_failed_synthetic_audit)
        request["args"] += ["--output_dir", str(out)]
        request_file = out / "request.json"
        write_json(request_file, request)
        print(f"START {request['suite']} {request['path']} {name}", flush=True)
        started = time.perf_counter()
        with (out / "run.log").open("w", encoding="utf-8") as log:
            command = [sys.executable, "-u", str(Path(__file__).resolve()), "--worker", str(request_file)]
            try:
                creationflags = subprocess.ABOVE_NORMAL_PRIORITY_CLASS if args.worker_priority == "above_normal" else 0
                completed = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                           timeout=args.timeout, creationflags=creationflags)
                code = completed.returncode
            except subprocess.TimeoutExpired:
                code = "timeout"
        trajectory = out / "trajectory.jsonl"
        steps = [json.loads(line) for line in trajectory.read_text(encoding="utf-8").splitlines()] if trajectory.exists() else []
        zero = next((row["step"] for row in steps if row["ef_pairs"] == 0), None)
        result = {"suite": request["suite"], "path": request["path"], "name": name,
                  "exit_code": code, "elapsed_s": time.perf_counter() - started,
                  "steps": len(steps), "first_zero_step": zero,
                  "initial_ef": steps[0]["ef_pairs"] if steps else None,
                  "last_ef": steps[-1]["ef_pairs"] if steps else None,
                  "mean_step_s_after_first": sum(x["step_s"] for x in steps[1:]) / (len(steps) - 1) if len(steps) > 1 else None,
                  "output": str(out)}
        write_json(out / "comparison_result.json", result)
        results.append(result)
        write_json(args.output / "comparison_results.json", results)
        print("DONE " + json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
