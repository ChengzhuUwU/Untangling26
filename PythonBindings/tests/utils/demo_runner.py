"""Run the same bounded simulation in a terminal or an interactive viewer.

``step(frame)`` performs one physical frame (including logging/export) and
returns a stop reason, or None to continue. Polyscope is imported only for GUI.
"""

from utils.auto_ccd import enable_ccd_for_few_contours


def run_simulation(solver, step, max_frames, *, headless, title, output_dir="."):
    frame = 0
    stop_reason = "frame_budget" if max_frames <= 0 else None

    def advance():
        nonlocal frame, stop_reason
        if stop_reason is not None:
            return
        frame += 1
        stop_reason = step(frame)
        enable_ccd_for_few_contours(solver)
        if stop_reason is None and frame >= max_frames:
            stop_reason = "frame_budget"

    if headless:
        while stop_reason is None:
            advance()
    else:
        from utils.polyscope_gui import SimulationGUI
        import polyscope as ps
        import polyscope.imgui as psim

        class DemoGUI(SimulationGUI):
            def _physics_step(self):
                advance()
                if stop_reason is not None:
                    self._is_simulating = False

            def _ui_callback(self):
                self._handle_keyboard()
                self._handle_selection()
                psim.TextUnformatted(title)
                psim.TextUnformatted(f"Frame {frame} / {max_frames}")
                if stop_reason is None:
                    if psim.Button("Advance Single Frame"):
                        self._physics_step()
                        self._update_gui_vertices()
                    if psim.Button("Pause" if self._is_simulating else "Run"):
                        self._is_simulating = not self._is_simulating
                else:
                    psim.TextUnformatted(f"Stopped: {stop_reason}")
                psim.TextUnformatted("Space: single step. Close window: finish preview.")
                self._panel_highlight()
                self._panel_collision()
                self._continuous_simulation_loop()

        gui = DemoGUI(solver, solver.get_config(), output_dir)
        try:
            gui.show()
        finally:
            ps.clear_user_callback()
            ps.remove_all_structures()

    return stop_reason or "viewer_closed"
