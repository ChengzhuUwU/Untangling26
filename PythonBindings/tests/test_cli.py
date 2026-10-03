"""CLI regressions: python -m unittest discover -s PythonBindings/tests -p test_cli.py."""

from contextlib import redirect_stderr, redirect_stdout
import io
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import trimesh

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
from untangling26.cli import main, make_parser, run_repair
from untangling26.mesh_io import (Body, check_outputs, load_bodies, normalize_bodies,
                                 output_paths, split_components, validate_mesh, write_mesh)


class MeshIOTests(unittest.TestCase):
    def setUp(self):
        self.vertices = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]])
        self.faces = np.array([[0, 1, 2]], dtype=np.int32)

    def test_shared_normalization_preserves_body_offsets_and_units(self):
        bodies = [Body("a", self.vertices * 100 + 200, self.faces, "a.obj"),
                  Body("b", self.vertices * 20 + 450, self.faces, "b.obj")]
        normalized, center, scale = normalize_bodies(bodies)
        all_vertices = np.concatenate([b.vertices for b in normalized])
        self.assertAlmostEqual(np.ptp(all_vertices, axis=0).max(), 1.0)
        for original, normal in zip(bodies, normalized):
            np.testing.assert_allclose(normal.vertices * scale + center, original.vertices)
        np.testing.assert_allclose(
            (normalized[1].vertices[0] - normalized[0].vertices[0]) * scale,
            bodies[1].vertices[0] - bodies[0].vertices[0])

    def test_rigid_components_have_local_face_indices(self):
        vertices = np.vstack([self.vertices, self.vertices + 4])
        faces = np.vstack([self.faces, self.faces + 3])
        pieces = list(split_components(vertices, faces))
        self.assertEqual(len(pieces), 2)
        for (v, f), expected in zip(pieces, [self.vertices, self.vertices + 4]):
            np.testing.assert_array_equal(v, expected)
            np.testing.assert_array_equal(f, self.faces)

    def test_scene_instances_and_world_transforms_are_preserved(self):
        scene = trimesh.Scene()
        scene.add_geometry(trimesh.Trimesh(self.vertices, self.faces, process=False),
                           geom_name="mesh", node_name="first")
        transform = np.eye(4)
        transform[:3, 3] = [4, 5, 6]
        scene.graph.update(frame_to="second", geometry="mesh", matrix=transform)
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "scene.glb"
            path.write_bytes(scene.export(file_type="glb"))
            bodies = load_bodies([path])
            self.assertEqual(len(bodies), 2)
            positions = sorted(tuple(b.vertices[0]) for b in bodies)
            self.assertEqual(positions, [(0., 0., 0.), (4., 5., 6.)])

    def test_export_two_bodies_preserves_obj_indices(self):
        bodies = [Body("a", self.vertices, self.faces, "a.obj"),
                  Body("b", self.vertices + 3, self.faces, "b.obj")]
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "result.obj"
            write_mesh(path, bodies)
            text = path.read_text()
            self.assertIn("o a\n", text)
            self.assertIn("o b\n", text)
            self.assertIn("f 1 2 3\n", text)
            self.assertIn("f 4 5 6\n", text)
            loaded = load_bodies([path])
            self.assertEqual(sum(len(b.faces) for b in loaded), 2)
            self.assertEqual(len(loaded), 2)

    def test_stl_vertices_are_welded_before_component_split(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "box.stl"
            trimesh.creation.box().export(path)
            bodies = load_bodies([path], split=True)
            self.assertEqual(len(bodies), 1)
            self.assertEqual(len(bodies[0].vertices), 8)

    def test_obj_normal_seams_do_not_split_surface_connectivity(self):
        text = ("v 0 0 0\nv 1 0 0\nv 0 1 0\nv 1 1 0\nv 9 9 9\n"
                "vn 0 0 1\nvn 0 0 -1\n"
                "f 1//1 2//1 3//1\nf 2//2 4//2 3//2\n")
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "seams.obj"
            path.write_text(text)
            bodies = load_bodies([path], split=True)
            self.assertEqual(len(bodies), 1)
            self.assertEqual(len(bodies[0].vertices), 4)

    def test_invalid_mesh_and_input_overwrite_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "NaN"):
            validate_mesh(self.vertices * np.nan, self.faces, "bad")
        with self.assertRaisesRegex(ValueError, "index"):
            validate_mesh(self.vertices, [[0, 1, 10]], "bad")
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "input.obj"
            path.touch()
            with self.assertRaisesRegex(ValueError, "input mesh"):
                check_outputs([path], [path], overwrite=True)
            with self.assertRaisesRegex(ValueError, "already exists"):
                check_outputs([path], [])

    def test_directory_outputs_have_one_file_per_body(self):
        bodies = [Body("a", self.vertices, self.faces, "a.obj"),
                  Body("b", self.vertices, self.faces, "b.obj")]
        paths, report = output_paths(Path("out"), bodies)
        self.assertEqual([p.name for p in paths], ["resolved.obj", "a.obj", "b.obj"])
        self.assertEqual(report.name, "summary.json")


class RepairLoopTests(unittest.TestCase):
    def solver(self, reports):
        class FakeSolver:
            def __init__(self):
                self.calls = 0
                self.vertices = np.zeros((3, 3))

            def get_sim_result(self):
                # Deliberately return mutable storage, as bindings may expose views.
                return [self.vertices], [np.array([[0, 1, 2]])]

            def physics_step_cpu(self):
                self.calls += 1
                self.vertices += 1

            physics_step_gpu = physics_step_cpu

            def get_intersection_contour_data(self):
                ef = reports[self.calls - 1]
                return {"num_pairs": [ef], "num_contours": [int(ef > 0)]}

        return FakeSolver()

    def test_success_exports_checked_start_state_not_speculative_update(self):
        for physics in ("cpu", "gpu"):
            solver = self.solver([2, 0])
            with redirect_stdout(io.StringIO()):
                (vertices, _), records, success = run_repair(
                    solver, SimpleNamespace(physics=physics, max_iters=5))
            self.assertTrue(success)
            np.testing.assert_array_equal(vertices[0], np.ones((3, 3)))
            self.assertEqual(solver.calls, 2)
            self.assertEqual(records[-1]["iteration"], 1)

    def test_budget_exports_last_checked_state_and_marks_unresolved(self):
        solver = self.solver([4, 2])
        with redirect_stdout(io.StringIO()):
            (vertices, _), records, success = run_repair(
                solver, SimpleNamespace(physics="cpu", max_iters=1))
        self.assertFalse(success)
        np.testing.assert_array_equal(vertices[0], np.ones((3, 3)))
        self.assertEqual(records[-1]["ef_pairs"], 2)

    def test_zero_budget_checks_input_without_exporting_motion(self):
        solver = self.solver([0])
        with redirect_stdout(io.StringIO()):
            (vertices, _), _, success = run_repair(
                solver, SimpleNamespace(physics="cpu", max_iters=0))
        self.assertTrue(success)
        np.testing.assert_array_equal(vertices[0], np.zeros((3, 3)))

    def test_missing_report_is_not_success(self):
        solver = self.solver([0])
        solver.get_intersection_contour_data = lambda: {}
        with self.assertRaises(KeyError):
            run_repair(solver, SimpleNamespace(physics="cpu", max_iters=0))


class CommandTests(unittest.TestCase):
    def test_multiple_inputs_and_output_parse_like_cuacd(self):
        args = make_parser().parse_args(["a.obj", "b.obj", "out/", "--material", "rigid"])
        self.assertEqual(args.inputs, [Path("a.obj"), Path("b.obj")])
        self.assertEqual(args.output, Path("out"))

    def test_help_does_not_import_native_solver(self):
        with patch.dict(sys.modules, {"lcs_py": None}), redirect_stdout(io.StringIO()):
            with self.assertRaises(SystemExit) as caught:
                main(["--help"])
            self.assertEqual(caught.exception.code, 0)

    def test_invalid_arguments_and_missing_input_return_two(self):
        for value in ("-1", "nan", "inf"):
            with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as caught:
                make_parser().parse_args(["a.obj", "out.obj", "--response-depth", value])
            self.assertEqual(caught.exception.code, 2)
        with redirect_stderr(io.StringIO()):
            self.assertEqual(main(["missing-input-84264.obj", "out.obj"]), 2)


if __name__ == "__main__":
    unittest.main()
