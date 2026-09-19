import os

from utils.abc_exporter import AlembicVertexExporter
from utils.auto_ccd import enable_ccd_for_few_contours


class DisplayInterface:
	"""
	Unified display/simulation loop for GUI and headless workflows.

	Features:
	- `--headless` headless stepping
	- GPU/CPU step dispatch
	- optional Alembic export via `--abc_export` / `--abc_path`

	Users usually subclass and override `update_animation()`.
	"""

	def __init__(
		self,
		solver,
		config_ref,
		args,
		output_dir: str = ".",
		init_obj_name: str = "init.obj",
		result_obj_name: str = "result.obj",
		abc_path_override: str | None = None,
	):
		self.solver = solver
		self.config_ref = config_ref
		self.args = args
		self.output_dir = output_dir
		self.init_obj_name = init_obj_name
		self.result_obj_name = result_obj_name
		self.abc_path_override = abc_path_override

		os.makedirs(self.output_dir, exist_ok=True)

	def update_animation(self):
		"""Override this in subclasses to update keyframes / constraints per frame."""
		return

	def before_step(self):
		"""Override this in subclasses to add hooks before each physics step."""
		return

	def after_step(self):
		"""Override this in subclasses to add hooks after each physics step. Return True to end the simulation."""
		return False

	def run(self):
		# self._run_headless()
		if bool(getattr(self.args, "headless", False)):
			self._run_headless()
		else:
			self._run_gui()

	def _run_headless(self):
		abc_exporter = None
		abc_path = self._resolve_abc_path()
		if abc_path:
			abc_exporter = AlembicVertexExporter()
			abc_exporter.add_frame_from_solver(self.solver)

		# Legacy per-frame OBJ export path retained for reference.

		for _ in range(self._advance_frames()):
			end_sim = self._physics_step_with_hooks()
			# Legacy per-frame OBJ export path retained for reference.
			if end_sim:
				break
			if abc_exporter is not None:
				abc_exporter.add_frame_from_solver(self.solver)

		if abc_exporter is not None:
			abc_exporter.save(abc_path, mesh_names=self.solver.get_mesh_names())
			print(f"Alembic file saved: {abc_path}")


	def _run_gui(self):
		import utils.polyscope_gui

		interface = self

		class HookedSimulationGUI(utils.polyscope_gui.SimulationGUI):
			def _physics_step(self):
				interface._physics_step_with_hooks()

		gui = HookedSimulationGUI(self.solver, self.config_ref, self.output_dir)
		gui.show()

	def _physics_step_with_hooks(self):
		self.update_animation()
		self.before_step()
		if self._use_gpu_step():
			self.solver.physics_step_gpu()
		else:
			self.solver.physics_step_cpu()
		enable_ccd_for_few_contours(self.solver)
		end_sim = self.after_step()
		return end_sim

	def _use_gpu_step(self) -> bool:
		if hasattr(self.config_ref, "use_gpu"):
			return bool(self.config_ref.use_gpu)
		backend = str(getattr(self.args, "backend", "")).lower()
		return backend in {"cuda", "dx", "vk", "metal"}

	def _advance_frames(self) -> int:
		return max(0, int(getattr(self.args, "advance_frames", 1)))

	def _resolve_abc_path(self) -> str:
		if self.abc_path_override is not None:
			path = str(self.abc_path_override).strip()
			return path

		path = str(getattr(self.args, "abc_path", "")).strip()
		if path:
			return path

		if bool(getattr(self.args, "abc_export", False)):
			return os.path.join(self.output_dir, "simulation.abc")
		return ""

	def _init_obj_path(self) -> str:
		return os.path.join(self.output_dir, self.init_obj_name)

	def _result_obj_path(self) -> str:
		return os.path.join(self.output_dir, self.result_obj_name)
