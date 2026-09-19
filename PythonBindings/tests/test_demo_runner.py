"""Controller regressions; run directly with Python, without lcs_py or a GPU."""

import sys
import unittest
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, patch

from utils.demo_runner import run_simulation


class DemoRunnerTests(unittest.TestCase):
    def run_headless(self, step, budget, solver=None):
        # A server installation must not need any Polyscope modules.
        with patch.dict(sys.modules, {"polyscope": None,
                                     "utils.polyscope_gui": None}):
            solver = solver or SimpleNamespace(
                get_config=lambda: SimpleNamespace(use_ccd_linesearch=True))
            return run_simulation(solver, step, budget, headless=True, title="test")

    def run_gui(self, step, budget, actions, solver=None):
        ps = ModuleType("polyscope")
        ps.clear_user_callback = Mock()
        ps.remove_all_structures = Mock()
        ps.imgui = ModuleType("polyscope.imgui")

        class FakeGUI:
            def __init__(self, *args):
                self._is_simulating = False

            def show(self):
                actions(self)

        module = ModuleType("utils.polyscope_gui")
        module.SimulationGUI = FakeGUI
        with patch.dict(sys.modules, {"polyscope": ps, "polyscope.imgui": ps.imgui,
                                     "utils.polyscope_gui": module}):
            try:
                solver = solver or SimpleNamespace(
                    get_config=lambda: SimpleNamespace(use_ccd_linesearch=True))
                return run_simulation(
                    solver, step, budget,
                    headless=False, title="test")
            finally:
                ps.clear_user_callback.assert_called_once()
                ps.remove_all_structures.assert_called_once()

    def test_headless_budget_and_no_viewer_dependency(self):
        step = Mock(return_value=None)
        self.assertEqual(self.run_headless(step, 3), "frame_budget")
        self.assertEqual([c.args[0] for c in step.call_args_list], [1, 2, 3])

    def test_zero_budget_never_steps(self):
        step = Mock()
        self.assertEqual(self.run_headless(step, 0), "frame_budget")
        step.assert_not_called()

    def test_gui_and_headless_stop_at_same_convergence_frame(self):
        def actions(gui):
            self.assertFalse(gui._is_simulating)
            gui._is_simulating = True
            for _ in range(8):
                gui._physics_step()  # Includes attempted stepping after completion.
            self.assertFalse(gui._is_simulating)

        headless_step = Mock(side_effect=lambda n: "resolved" if n == 2 else None)
        gui_step = Mock(side_effect=lambda n: "resolved" if n == 2 else None)
        self.assertEqual(self.run_headless(headless_step, 5), "resolved")
        self.assertEqual(self.run_gui(gui_step, 5, actions), "resolved")
        self.assertEqual(headless_step.call_args_list, gui_step.call_args_list)

    def test_gui_budget_cannot_be_overrun_by_single_step(self):
        step = Mock(return_value=None)
        result = self.run_gui(step, 2, lambda gui: [gui._physics_step() for _ in range(5)])
        self.assertEqual(result, "frame_budget")
        self.assertEqual(step.call_count, 2)

    def test_closing_paused_viewer_does_not_advance_or_claim_success(self):
        step = Mock()
        self.assertEqual(self.run_gui(step, 5, lambda gui: None), "viewer_closed")
        step.assert_not_called()

    def test_physics_failure_propagates_and_cleans_viewer(self):
        step = Mock(side_effect=RuntimeError("solver failed"))
        with self.assertRaisesRegex(RuntimeError, "solver failed"):
            self.run_gui(step, 5, lambda gui: gui._physics_step())

    def test_ccd_switch_applies_to_next_step_in_both_modes(self):
        for headless in (True, False):
            with self.subTest(headless=headless):
                config = SimpleNamespace(use_ccd_linesearch=False,
                                         use_untangling=True, current_frame=0)
                counts = [16, 15, 20]
                used_ccd = []
                solver = SimpleNamespace(
                    get_config=lambda: config,
                    get_intersection_contour_data=lambda: {
                        "num_contours": [counts[config.current_frame - 1]]})

                def step(frame):
                    used_ccd.append(config.use_ccd_linesearch)
                    config.current_frame = frame

                if headless:
                    self.run_headless(step, 3, solver)
                else:
                    self.run_gui(step, 3,
                                 lambda gui: [gui._physics_step() for _ in range(3)],
                                 solver)
                self.assertEqual(used_ccd, [False, False, True])
                self.assertTrue(config.use_ccd_linesearch)


if __name__ == "__main__":
    unittest.main()
