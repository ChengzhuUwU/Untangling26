import os
from typing import Optional

import numpy as np


class AlembicVertexExporter:
	"""
	Collect simulation frames and export mesh animation to Alembic (.abc).

	Typical usage in headless mode:
		1) exporter.add_frame_from_solver(solver) each frame
		2) exporter.save("path/to/output.abc", mesh_names=solver.get_mesh_names())
	"""

	def __init__(self):
		self._frames_vertices: list[list[np.ndarray]] = []
		self._faces: Optional[list[np.ndarray]] = None
		self._vertex_counts: Optional[list[int]] = None

	def add_frame(self, vertices_list, faces_list=None):
		verts = [np.asarray(v, dtype=np.float32).copy() for v in vertices_list]
		if len(verts) == 0:
			raise ValueError("No mesh vertices provided for Alembic export.")

		for idx, mesh_v in enumerate(verts):
			if mesh_v.ndim != 2 or mesh_v.shape[1] != 3:
				raise ValueError(f"Mesh {idx} vertices must have shape (N, 3), got {mesh_v.shape}.")

		if self._faces is None:
			if faces_list is None:
				raise ValueError("faces_list is required for the first frame.")
			faces = [np.asarray(f, dtype=np.int32).copy() for f in faces_list]
			if len(faces) != len(verts):
				raise ValueError("faces_list size must match vertices_list size.")
			for idx, mesh_f in enumerate(faces):
				if mesh_f.ndim != 2 or mesh_f.shape[1] != 3:
					raise ValueError(f"Mesh {idx} faces must have shape (M, 3), got {mesh_f.shape}.")
			self._faces = faces
			self._vertex_counts = [int(mesh_v.shape[0]) for mesh_v in verts]
		else:
			if len(verts) != len(self._faces):
				raise ValueError("Mesh count changed across frames; Alembic exporter expects fixed topology.")
			for idx, mesh_v in enumerate(verts):
				if int(mesh_v.shape[0]) != int(self._vertex_counts[idx]):
					raise ValueError(
						f"Mesh {idx} vertex count changed from {self._vertex_counts[idx]} to {mesh_v.shape[0]}."
					)

		self._frames_vertices.append(verts)

	def add_frame_from_solver(self, solver):
		verts_list, faces_list = solver.get_sim_result()
		if self._faces is None:
			self.add_frame(verts_list, faces_list)
		else:
			self.add_frame(verts_list)

	def save(self, abc_path: str, mesh_names=None):
		if not self._frames_vertices:
			raise RuntimeError("No frames collected. Call add_frame()/add_frame_from_solver() before save().")
		if not abc_path:
			raise ValueError("abc_path is empty.")
		if not abc_path.lower().endswith(".abc"):
			raise ValueError(f"Alembic output path must end with .abc, got: {abc_path}")

		os.makedirs(os.path.dirname(os.path.abspath(abc_path)), exist_ok=True)
		self._write_alembic(abc_path, mesh_names)

	def clear(self):
		self._frames_vertices.clear()
		self._faces = None
		self._vertex_counts = None

	@staticmethod
	def _to_v3f_list(vertices: np.ndarray, imath_module):
		return [imath_module.V3f(float(v[0]), float(v[1]), float(v[2])) for v in vertices]

	@staticmethod
	def _sanitize_name(name: str, fallback_idx: int) -> str:
		clean = "".join(ch if (ch.isalnum() or ch in "_-") else "_" for ch in str(name))
		clean = clean.strip("_")
		if not clean:
			clean = f"mesh_{fallback_idx}"
		return clean

	def _write_alembic(self, abc_path: str, mesh_names=None):
		try:
			import imath
			from alembic.Abc import OArchive
			from alembic.AbcGeom import OPolyMesh, OPolyMeshSchemaSample
		except Exception as exc:
			raise RuntimeError(
				"Alembic export requires Python bindings for Alembic and Imath. "
				"Please install pyalembic + pyimath."
			) from exc

		archive = OArchive(abc_path)
		top = archive.getTop()

		mesh_objects = []
		mesh_schemas = []
		mesh_count = len(self._faces)
		for idx in range(mesh_count):
			raw_name = f"mesh_{idx}"
			if mesh_names is not None and idx < len(mesh_names):
				raw_name = str(mesh_names[idx])
			mesh_name = self._sanitize_name(raw_name, idx)
			mesh_obj = OPolyMesh(top, mesh_name)
			mesh_objects.append(mesh_obj)
			mesh_schemas.append(mesh_obj.getSchema())

		for frame_vertices in self._frames_vertices:
			for idx, (mesh_v, mesh_f) in enumerate(zip(frame_vertices, self._faces)):
				indices = mesh_f.reshape(-1).astype(np.int32).tolist()
				counts = np.full((mesh_f.shape[0],), 3, dtype=np.int32).tolist()
				sample = OPolyMeshSchemaSample(self._to_v3f_list(mesh_v, imath), indices, counts)
				mesh_schemas[idx].set(sample)

		# Keep handles alive until all samples are written.
		_ = mesh_objects

