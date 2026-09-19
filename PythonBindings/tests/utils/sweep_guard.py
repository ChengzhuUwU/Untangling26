"""GPU collision-buffer guard for parameter-sweep runs.

The collision-detection buffers (broad/narrow phase, EF pairs, triplets, CSR
adjacency) grow with the number of self-intersections. On the 12 GB RTX 5070 a
147k-vertex garment with ~75k EF pairs wants ~14 GB for the broad-phase list
alone: LuisaCompute first spills to host memory (10x slowdown) and then dies
with CUDA_ERROR_OUT_OF_MEMORY, having written only a fraction of the frames.

This guard checks the buffer total each frame via the solver's
``get_collision_buffer_bytes`` binding (mirrors CollisionData::get_momery_bytes)
and aborts the run cleanly once it crosses a budget, so a crashed arm leaves a
recognizable partial result instead of a slow death.

Usage in a DisplayInterface.after_step override::

    from utils.sweep_guard import check_collision_buffer_budget
    ...
    check_collision_buffer_budget(self.solver)   # raises BufferBudgetExceeded
"""
import os


class BufferBudgetExceeded(RuntimeError):
	"""Raised when the collision buffer total exceeds the configured budget."""
	pass


# Default 8 GiB budget leaves headroom below the 12 GB device; override via SWEEP_BUFFER_BUDGET_GIB.
def _budget_bytes():
	env = os.environ.get("SWEEP_BUFFER_BUDGET_GIB")
	try:
		gib = float(env) if env else 8.0
	except ValueError:
		gib = 8.0
	return int(gib * (1024 ** 3))


def check_collision_buffer_budget(solver):
	"""Raise BufferBudgetExceeded if the collision buffers exceed the budget."""
	if not hasattr(solver, "get_collision_buffer_bytes"):
		return
	try:
		nbytes = solver.get_collision_buffer_bytes()
	except Exception:
		return
	budget = _budget_bytes()
	if nbytes > budget:
		raise BufferBudgetExceeded(
			f"collision buffer {nbytes / (1024 ** 3):.2f} GiB exceeds budget "
			f"{budget / (1024 ** 3):.1f} GiB; aborting run before device OOM"
		)
