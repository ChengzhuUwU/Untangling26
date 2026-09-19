"""Paired CPU PRP blend tests on the other locally bundled README cases."""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

from test_thingi10k_rigid_untangling import benchmark_cases
from utils.intrinsic_filter import unit_interval

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "PythonBindings/tests"


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--weights", type=unit_interval, nargs="+", default=[0.25, 0.5])
    parser.add_argument("--tau", type=unit_interval, default=0.25)
    parser.add_argument("--workers", type=int, choices=(1, 2), default=2)
    parser.add_argument("--timeout_seconds", type=int, default=1800)
    parser.add_argument("--max_ef_pairs", type=int, default=20000)
    parser.add_argument("--jang_dataset", type=Path,
                        default=ROOT / "external/instant-mesh-intersection-repair/data/misc")
    parser.add_argument("--groups", nargs="+", choices=("canonical", "drop3", "jang", "obj", "thingi"),
                        default=["canonical", "drop3", "jang", "obj", "thingi"])
    args = parser.parse_args()
    if len(set(args.weights)) != len(args.weights):
        parser.error("weights must be unique")
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, str(ROOT / "build/bin"))
    import lcs_py
    binding = Path(lcs_py.__file__)
    sources = [*TESTS.glob("demo_*.py"), TESTS / "test_load_from_obj.py",
               TESTS / "test_batch_method_comparison_enhanced.py",
               TESTS / "test_thingi10k_rigid_untangling.py", Path(__file__).resolve(),
               *sorted((TESTS / "utils").glob("*.py")), TESTS / "shared_args.py",
               ROOT / "Solver/CollisionDetector/intersection_resolver2.cpp",
               ROOT / "Solver/CollisionDetector/intersection_resolver2.h",
               ROOT / "Solver/SimulationCore/scene_params.h", ROOT / "PythonBindings/src/python_bindings.cpp"]
    report = {"complete": False, "binding_sha256": sha256(binding),
              "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
              "source_sha256": {str(p.relative_to(ROOT)): sha256(p) for p in sources},
              "parameters": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
              "cases": []}
    for p in sources:
        destination = out / "source_snapshot" / p.relative_to(ROOT)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, destination)
    (out / "working_tree.patch").write_bytes(subprocess.check_output(["git", "diff"], cwd=ROOT))

    specs = []
    for i, label in enumerate(("BBII", "BIBI", "BLI", "Closed", "Cross", "Eight", "LL")):
        specs.append(("canonical", label, "demo_unit.py", 300,
                      ["--scene_id", str(i), "--use_subdivision", "--subdiv_levels", "3"],
                      [ROOT / f"Resources/InputMesh/SelfIntersectionUnit/unit_{label}.obj"]))
    DROP_INPUTS = ["Resources/PaperCases/Drop3/drop3_clothes.obj",
                   "Resources/PaperCases/Drop3/state_frame_68.state"]
    specs.append(("drop3", "Drop3", "demo_dropping3.py", 100,
                  ["--use_gpu_pcg", "0", "--prp_debug", "0", "--no_export_per_frame",
                   "--use_ccd_linesearch", "0", "--PRP_cpu_contour_batch_size", "64"],
                  [ROOT / p for p in DROP_INPUTS]))
    jang_dir = args.jang_dataset.resolve()
    for i, mesh in enumerate(sorted(jang_dir.glob("*.obj")), 1):
        specs.append(("jang", mesh.stem, "test_batch_method_comparison_enhanced.py", 100,
                      ["--jang_dataset", str(jang_dir), "--begin", str(i), "--end", str(i), "--timeout", str(args.timeout_seconds),
                       "--use_gpu_pcg", "0", "--prp_debug", "0", "--no_export_per_frame"], [mesh]))
    specs.append(("obj", "KleinBottle", "test_load_from_obj.py", 200,
                  ["--input_mesh", str(jang_dir / "disc_kleinbottle.obj"), "--prp_debug", "0"],
                  [jang_dir / "disc_kleinbottle.obj"]))
    for case in benchmark_cases():
        specs.append(("thingi", case["name"], "test_thingi10k_rigid_untangling.py", 35,
                      ["--worker", "--mesh_path", case["mesh"], "--case_name", case["name"],
                       "--offset", *map(str, case["offset"]), "--rot2", *map(str, case["rot2"]),
                       "--use_gpu", "0", "--solid_adj", "1", "--prp_debug", "0"], [Path(case["mesh"]).resolve()]))
    specs = [spec for spec in specs if spec[0] in args.groups]

    def save():
        (out / "results.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    def run(spec, weight):
        group, label, script, budget, extra, inputs = spec
        name = f"{group}_{label}_w{weight:g}".replace(".", "p")
        case_dir = out / name
        case_dir.mkdir()
        cmd = [sys.executable, "-u", str(TESTS / script), "--backend", "cuda", "--headless",
               "--max_frames" if group == "thingi" else "--advance_frames", str(budget),
               "--intrinsic_filter_mode", "blend", "--intrinsic_blend_weight", str(weight),
               "--intrinsic_tau", str(args.tau), "--rest_geodesic_distance_for_intrinsic_candidates", "1",
               "--consistent_solve", "1", "--untangling_process_contours_count", "256",
               "--max_ef_pairs", str(args.max_ef_pairs), "--output_dir", str(case_dir), *extra]
        if group != "thingi":
            cmd += ["--use_gpu_untangling", "0", "--intrinsic_contour_side_candidates", "1",
                    "--deformed_boundary_distance_for_intrinsic_candidates", "1"]
        record = {"name": name, "group": group, "case": label, "weight": weight, "budget": budget,
                  "command": cmd, "input_sha256": {str(p): sha256(p) for p in inputs}}
        (case_dir / "request.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
        print(f"START {name}", flush=True)
        start = time.monotonic()
        with (case_dir / "run.log").open("w", encoding="utf-8") as log:
            try:
                process = subprocess.run(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                         timeout=args.timeout_seconds + (30 if group == "jang" else 0))
                record["exit_code"] = process.returncode
            except subprocess.TimeoutExpired:
                record["timeout"] = True
                record["exit_code"] = None
        record["elapsed_s"] = time.monotonic() - start
        path = case_dir / "summary.json"
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            if group == "jang":
                data = data["results"][0]
                initial = case_dir / "cases/PRP" / label / "initial.obj"
            else:
                initial = case_dir / "initial.obj"
            record["initial_obj_sha256"] = sha256(initial) if initial.exists() else None
            samples = data.get("frame_records", data.get("frames", []))
            samples = samples if isinstance(samples, list) else []
            samples = [{"frame": s["frame"], "ef": s.get("ef", s.get("ef_pair_count")),
                        "contours": s.get("contours", s.get("contour_count"))} for s in samples]
            record["samples"] = samples
            record["configuration"] = data.get("configuration", {})
            expected = {"use_gpu": False, "use_gpu_untangling": False, "consistent_solve": True,
                        "prp_debug": False, "PRP_use_intrinsic_contour_side_candidates": True,
                        "PRP_use_rest_geodesic_distance_for_intrinsic_candidates": True,
                        "PRP_use_deformed_boundary_distance_for_intrinsic_candidates": True,
                        "PRP_use_blended_intrinsic_coordinates": True,
                        "PRP_intrinsic_blend_weight": weight, "PRP_intrinsic_tau": args.tau,
                        "untangling_process_contours_count": 256}
            record["configuration_verified"] = all(record["configuration"].get(k) == v for k, v in expected.items())
            record.update({"steps": len(samples), "initial_ef": samples[0]["ef"] if samples else None,
                           "final_ef": samples[-1]["ef"] if samples else None,
                           "peak_ef": max((s["ef"] for s in samples), default=None),
                           "first_zero_step": next((i for i, s in enumerate(samples, 1) if s["ef"] == 0), None),
                           "driver_stop_reason": data.get("stop_reason")})
            converged = data.get("converged", data.get("convergence", {}).get("converged", False))
            if record.get("timeout"):
                status = "timeout"
            elif data.get("ok") is False:
                status = "timeout" if "Timed out" in data.get("error", "") else "process_error"
                record["error"] = data.get("error")
            elif not record["configuration_verified"]:
                status = "configuration_mismatch"
            elif args.max_ef_pairs > 0 and (record["final_ef"] or 0) > args.max_ef_pairs:
                status = "ef_limit"
            elif converged and record["exit_code"] == 0:
                status = "passed"
            elif data.get("stop_reason") == "ef_explosion":
                status = "ef_explosion"
            elif len(samples) >= budget:
                status = "budget_exhausted"
            else:
                status = "process_error"
            record["status"] = status
        else:
            record["status"] = "timeout" if record.get("timeout") else "missing_summary"
        print(f"DONE {name}: {record['status']}, step={record.get('steps')}, EF={record.get('final_ef')}", flush=True)
        return record

    save()
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(run, spec, w) for spec in specs for w in args.weights]
        for future in as_completed(futures):
            report["cases"].append(future.result())
            save()
    report["cases"].sort(key=lambda r: (r["group"], r["case"], r["weight"]))
    report["binding_unchanged"] = sha256(binding) == report["binding_sha256"]
    report["sources_unchanged"] = all(sha256(ROOT / name) == value for name, value in report["source_sha256"].items())
    report["paired_checks"] = []
    for group, label, *_ in specs:
        rows = [r for r in report["cases"] if r["group"] == group and r["case"] == label]
        hashes = {r.get("initial_obj_sha256") for r in rows}
        configs = [{k: v for k, v in r.get("configuration", {}).items() if k != "PRP_intrinsic_blend_weight"} for r in rows]
        report["paired_checks"].append({"group": group, "case": label,
            "identical_initial_geometry": len(hashes) == 1 and None not in hashes,
            "matched_configuration_except_weight": bool(configs[0]) and all(c == configs[0] for c in configs[1:])})
    report["complete"] = True
    save()
    print(f"COMPLETE {out / 'results.json'}", flush=True)


if __name__ == "__main__":
    main()
