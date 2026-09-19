"""
Polyscope-based GUI for LuisaComputeSimulator.

Encapsulates all polyscope initialisation, mesh registration, ImGui panels,
and the continuous-simulation callback.  Usage:

    from polyscope_gui import SimulationGUI
    gui = SimulationGUI(solver, config_ref, output_dir)
    gui.show()   # blocking – opens the polyscope window
"""

import bisect
import os
import numpy as np
import polyscope as ps
import polyscope.imgui as psim


class SimulationGUI:
    """Thin wrapper around polyscope that mirrors the C++ Polyscope GUI."""

    MERGE_RENDER_THRESHOLD = 20
    RESPONSE_VISUAL_LABEL_COUNT = 256
    UV_PNG_DPI = 300
    UV_PNG_WIDTH_PX = 4800
    UV_PNG_HEIGHT_PX = 3600
    CONTOUR_TYPE_NAMES = {
        0: "Undefined",
        1: "Closed",
        2: "Eight",
        3: "LL",
        4: "BLI",
        5: "Cross",
        6: "BIBI",
        7: "BBII",
    }

    def __init__(self, solver, config_ref, output_dir: str = "."):
        """
        Parameters
        ----------
        solver : lcs_py.NewtonSolver
            An already-initialised solver (``init_solver()`` must have been called).
        config_ref : lcs_py.SceneParams
            Reference returned by ``lcs.get_scene_params()``.
        output_dir : str
            Directory used when saving OBJ files from the GUI.
        """
        self._solver = solver
        self._config = config_ref
        self._output_dir = output_dir

        # UI state
        self._is_simulating = False

        # Polyscope handles
        self._surface_meshes: list = []
        self._mesh_names: list = []
        self._surface_mesh_name_to_idx: dict[str, int] = {}
        self._curve_network = None

        # Rendering mode state
        self._use_merged_render = False
        self._mesh_faces: list = []
        self._mesh_edges: list = []
        self._mesh_boundary_edges: list = []
        self._default_face_colors: list = []
        self._merged_default_face_colors = None
        self._vert_prefix = [0]
        self._face_prefix = [0]
        self._edge_prefix = [0]
        self._last_selection_signature = None

        # Highlight state
        self._highlight_input_id = 0
        self._highlight_element_type = None
        self._highlight_element_global_id = None
        self._highlight_mesh_idx = None
        self._highlight_local_idx = None
        self._highlight_vertex_adj_edges: list = []
        self._highlight_cloud = None
        self._highlight_curve = None
        self._last_highlight_adj_edge = None

        # Contour export state
        self._export_contour_idx = 0
        self._export_contour_ring_count = 0

        # 2D material-space visualization state
        self._uv_2d_enabled = False
        self._uv_contour_filter = -1
        self._uv_show_mesh_boundaries = True
        self._uv_status = "Set contour filter to -1 for all contours."
        self._uv_mesh_networks: list = []
        self._uv_boundary_networks: list = []
        self._uv_contour_network = None
        self._uv_contour_points = None
        self._uv_hit_network = None
        self._uv_hit_points = None
        self._uv_saved_structure_states: list = []
        self._uv_previous_view = None
        self._uv_previous_background = None
        self._uv_previous_projection = None
        self._uv_previous_bbox = None

    # Public API.

    def show(self):
        """Initialise polyscope, register meshes, set the callback and open the window."""
        ps.init()
        self._register_meshes()
        ps.update_scene_extents()
        ps.set_automatically_compute_scene_extents(False)
        ps.set_user_callback(self._ui_callback)
        ps.show()

    # Internal helpers.

    def _register_meshes(self):
        verts_list, faces_list = self._solver.get_sim_result()
        self._mesh_names = self._solver.get_mesh_names()
        self._surface_meshes = []
        self._surface_mesh_name_to_idx = {}
        self._mesh_faces = [np.asarray(faces, dtype=np.int32) for faces in faces_list]
        self._mesh_edges = [
            np.asarray(edges, dtype=np.int32)
            for edges in self._solver.get_surface_edges()
        ]
        self._mesh_boundary_edges = [
            self._extract_mesh_boundary_edges(faces) for faces in self._mesh_faces
        ]
        self._default_face_colors = []
        self._merged_default_face_colors = None
        self._vert_prefix = self._build_prefix(verts_list)
        self._face_prefix = self._build_prefix(self._mesh_faces)
        self._edge_prefix = self._build_prefix(self._mesh_edges)
        self._last_selection_signature = None

        mesh_count = len(verts_list)
        self._use_merged_render = mesh_count > self.MERGE_RENDER_THRESHOLD

        if self._use_merged_render:
            self._register_merged_mesh(verts_list, self._mesh_faces)
            self._update_mesh_visualization(verts_list)
            self._register_or_update_curve_network()
            return

        for idx, name in enumerate(self._mesh_names):
            faces_np = self._mesh_faces[idx]
            default_face_colors = np.tile(
                self._face_color_from_index(idx), (faces_np.shape[0], 1)
            ).astype(np.float32, copy=False)
            ps_mesh = ps.register_surface_mesh(
                f"{name}{idx}", verts_list[idx], faces_np
            )
            ps_mesh.set_back_face_policy("custom")
            self._surface_mesh_name_to_idx[ps_mesh.get_name()] = idx
            self._add_color_quantity(
                ps_mesh, "object_face_color", default_face_colors, defined_on="faces"
            )
            ps_mesh.set_enabled(True)
            self._enable_mesh_wireframe(ps_mesh)
            self._surface_meshes.append(ps_mesh)
            self._default_face_colors.append(default_face_colors)
        self._update_mesh_visualization(verts_list)
        # Keep curve rendering in sync with the C++ app GUI.
        self._register_or_update_curve_network()

    def _register_merged_mesh(self, verts_list, faces_list):
        """Merge all objects into one render mesh and mark each object's faces by color."""
        merged_verts = []
        merged_faces = []
        merged_face_colors = []

        vertex_offset = 0

        for mesh_idx, (verts, faces) in enumerate(zip(verts_list, faces_list)):
            verts_np = np.asarray(verts)
            faces_np = np.asarray(faces, dtype=np.int32)

            merged_verts.append(verts_np)
            merged_faces.append(faces_np + vertex_offset)

            # Stable pseudo-random color per object id.
            obj_color = self._face_color_from_index(mesh_idx)
            merged_face_colors.append(np.tile(obj_color, (faces_np.shape[0], 1)))

            vertex_offset += verts_np.shape[0]

        all_verts = np.concatenate(merged_verts, axis=0)
        all_faces = np.concatenate(merged_faces, axis=0)
        all_face_colors = np.concatenate(merged_face_colors, axis=0)
        self._merged_default_face_colors = all_face_colors

        merged_mesh = ps.register_surface_mesh("merged_scene", all_verts, all_faces)
        merged_mesh.set_back_face_policy("custom")
        self._surface_mesh_name_to_idx[merged_mesh.get_name()] = -1
        self._add_color_quantity(
            merged_mesh,
            "object_face_color",
            all_face_colors,
            defined_on="faces",
        )
        merged_mesh.set_enabled(True)
        self._enable_mesh_wireframe(merged_mesh)
        self._surface_meshes = [merged_mesh]

    @staticmethod
    def _enable_mesh_wireframe(mesh):
        """Enable visible surface edges across polyscope Python API variants."""
        setter = getattr(mesh, "set_edge_width", None)
        if callable(setter):
            setter(1.0)
            return

        # Compatibility fallback for bindings exposing C++ style naming.
        setter = getattr(mesh, "setEdgeWidth", None)
        if callable(setter):
            setter(1.0)

    @staticmethod
    def _build_prefix(items):
        prefix = [0]
        running = 0
        for item in items:
            running += int(len(item))
            prefix.append(running)
        return prefix

    @staticmethod
    def _extract_mesh_boundary_edges(faces):
        """Return exactly the edges incident to one triangle."""
        faces = np.asarray(faces, dtype=np.int32)
        if faces.size == 0:
            return np.empty((0, 2), dtype=np.int32)
        if faces.ndim != 2 or faces.shape[1] != 3:
            raise ValueError(f"triangle array must have shape (F, 3), got {faces.shape}")

        face_edges = np.concatenate(
            (faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]), axis=0
        )
        face_edges = np.sort(face_edges, axis=1)
        unique_edges, counts = np.unique(face_edges, axis=0, return_counts=True)
        return unique_edges[counts == 1].astype(np.int32, copy=False)

    @staticmethod
    def _resolve_prefixed_index(prefix, global_index):
        if global_index < 0 or global_index >= prefix[-1]:
            return None
        mesh_idx = bisect.bisect_right(prefix, global_index) - 1
        local_idx = global_index - prefix[mesh_idx]
        return mesh_idx, local_idx

    @staticmethod
    def _disable_structure(structure):
        if structure is not None:
            structure.set_enabled(False)

    def _resolve_pick_mesh_and_local_index(self, structure_name, element_type, element_index):
        mesh_idx = self._surface_mesh_name_to_idx.get(structure_name)
        if mesh_idx is None:
            return None

        if mesh_idx != -1:
            return mesh_idx, int(element_index)

        if element_type == "vertex":
            return self._resolve_prefixed_index(self._vert_prefix, int(element_index))
        if element_type == "face":
            return self._resolve_prefixed_index(self._face_prefix, int(element_index))
        if element_type == "edge":
            return self._resolve_prefixed_index(self._edge_prefix, int(element_index))
        return None

    def _get_global_element_index(self, mesh_idx, local_idx, element_type):
        if element_type == "vertex":
            return int(
                self._solver.query_global_vid_from_registration_id_and_local_vid(
                    mesh_idx, local_idx
                )
            )
        if element_type == "face":
            return int(self._face_prefix[mesh_idx] + local_idx)
        if element_type == "edge":
            return int(self._edge_prefix[mesh_idx] + local_idx)
        return None

    def _handle_selection(self):
        if not ps.have_selection():
            self._last_selection_signature = None
            return

        try:
            selection = ps.get_selection()
        except RuntimeError as exc:
            # Drop stale picks when a per-frame structure has been re-registered or removed.
            if "pick result is not from this structure" in str(exc):
                ps.reset_selection()
                self._last_selection_signature = None
                return
            raise
        if not selection.is_hit or selection.structure_type_name != "Surface Mesh":
            self._last_selection_signature = None
            return

        element_type = str(selection.structure_data.get("element_type", "")).lower()
        if element_type not in {"vertex", "face", "edge"}:
            self._last_selection_signature = None
            return

        element_index = selection.structure_data.get("index")
        if element_index is None:
            self._last_selection_signature = None
            return

        signature = (
            selection.structure_name,
            tuple(selection.screen_coords),
            element_type,
            int(element_index),
        )
        if signature == self._last_selection_signature:
            return
        self._last_selection_signature = signature

        resolved = self._resolve_pick_mesh_and_local_index(
            selection.structure_name, element_type, element_index
        )
        if resolved is None:
            return

        mesh_idx, local_idx = resolved
        global_idx = self._get_global_element_index(mesh_idx, local_idx, element_type)
        if global_idx is None:
            return

        label = {
            "vertex": "Vert",
            "face": "Face",
            "edge": "Edge",
        }[element_type]
        extra_info = ""
        if element_type == "vertex":
            adj_edges = [int(edge_id) for edge_id in self._solver.query_vert_adj_edges(global_idx)]
            extra_info = f", adj_edges={adj_edges}"

        print(
            f"Select {label} {global_idx:3d} on mesh {mesh_idx} (local {local_idx}){extra_info}"
        )

    def _face_color_from_index(self, idx: int):
        """Generate a deterministic, visually-separated RGB color."""
        # Golden-ratio hue stepping gives well-spaced colors for many objects.
        hue = (0.61803398875 * (idx + 1)) % 1.0
        sat = 0.65
        val = 0.95
        return np.asarray(self._hsv_to_rgb(hue, sat, val), dtype=np.float32)

    @staticmethod
    def _hsv_to_rgb(h, s, v):
        i = int(h * 6.0)
        f = h * 6.0 - i
        p = v * (1.0 - s)
        q = v * (1.0 - f * s)
        t = v * (1.0 - (1.0 - f) * s)
        i %= 6
        if i == 0:
            return [v, t, p]
        if i == 1:
            return [q, v, p]
        if i == 2:
            return [p, v, t]
        if i == 3:
            return [p, q, v]
        if i == 4:
            return [t, p, v]
        return [v, p, q]

    def _add_color_quantity(self, structure, name, values, defined_on):
        values = np.asarray(values, dtype=np.float32)
        try:
            structure.add_color_quantity(
                name, values, defined_on=defined_on, enabled=True
            )
        except TypeError:
            structure.add_color_quantity(name, values, enabled=True)

    @staticmethod
    def _require_debug_array(data, key, ndim, width=None, dtype=None):
        if key not in data:
            raise KeyError(f"missing debug array '{key}'")
        array = np.asarray(data[key], dtype=dtype)
        if array.ndim != ndim:
            raise ValueError(
                f"debug array '{key}' must have {ndim} dimensions, got {array.shape}"
            )
        if width is not None and array.shape[1] != width:
            raise ValueError(
                f"debug array '{key}' must have shape (*, {width}), got {array.shape}"
            )
        return array

    @staticmethod
    def _embed_uv_points(points, z):
        points = np.asarray(points, dtype=np.float32)
        if points.size == 0:
            return np.empty((0, 3), dtype=np.float32)
        if points.ndim != 2 or points.shape[1] != 2:
            raise ValueError(f"UV points must have shape (N, 2), got {points.shape}")
        result = np.empty((points.shape[0], 3), dtype=np.float32)
        result[:, :2] = points
        result[:, 2] = float(z)
        return result

    def _build_uv_contour_geometry(self, contour_data, contour_filter):
        pair_pos = self._require_debug_array(
            contour_data, "ef_pair_pos_2d", 2, width=4, dtype=np.float32
        )
        pair_count = pair_pos.shape[0]
        pair_roles = self._require_debug_array(
            contour_data, "ef_pair_mesh_index", 1, dtype=np.uint32
        )
        contour_ids = self._require_debug_array(
            contour_data, "ef_pair_contour_index", 1, dtype=np.uint32
        )
        adj_prefix = self._require_debug_array(
            contour_data, "ef_pair_adj_pairs_ext_prefix", 1, dtype=np.uint32
        )
        adj_data = self._require_debug_array(
            contour_data, "ef_pair_adj_pairs_ext_data", 1, dtype=np.uint32
        )
        contour_prefix = self._require_debug_array(
            contour_data, "intersection_contours_prefix", 1, dtype=np.uint32
        )
        contour_members = self._require_debug_array(
            contour_data, "intersection_contours_data", 1, dtype=np.uint32
        )

        if pair_roles.shape[0] != pair_count or contour_ids.shape[0] != pair_count:
            raise ValueError("EF pair position, role, and contour arrays have different lengths")
        if adj_prefix.shape[0] != pair_count + 1:
            raise ValueError(
                "ef_pair_adj_pairs_ext_prefix length must equal EF pair count + 1"
            )
        if adj_prefix.size == 0 or int(adj_prefix[0]) != 0:
            raise ValueError("EF adjacency prefix must start at zero")
        if np.any(adj_prefix[1:] < adj_prefix[:-1]):
            raise ValueError("EF adjacency prefix must be nondecreasing")
        if int(adj_prefix[-1]) != adj_data.shape[0]:
            raise ValueError("EF adjacency prefix does not cover its data array")
        if contour_prefix.size == 0 or int(contour_prefix[0]) != 0:
            raise ValueError("intersection contour prefix must start at zero")
        if np.any(contour_prefix[1:] < contour_prefix[:-1]):
            raise ValueError("intersection contour prefix must be nondecreasing")
        if int(contour_prefix[-1]) != contour_members.shape[0]:
            raise ValueError("intersection contour prefix does not cover its data array")

        num_contours = contour_prefix.shape[0] - 1
        contour_filter = int(contour_filter)
        if contour_filter < -1 or contour_filter >= num_contours:
            raise ValueError(
                f"contour filter {contour_filter} is outside [-1, {num_contours - 1}]"
            )

        if contour_filter == -1:
            selected_contours = list(range(num_contours))
            selected_pair_ids = np.unique(contour_members).astype(
                np.int64, copy=False
            )
        else:
            selected_contours = [contour_filter]
            begin = int(contour_prefix[contour_filter])
            end = int(contour_prefix[contour_filter + 1])
            selected_pair_ids = np.unique(contour_members[begin:end]).astype(
                np.int64, copy=False
            )

        if selected_pair_ids.size > 0:
            if int(selected_pair_ids[0]) < 0 or int(selected_pair_ids[-1]) >= pair_count:
                raise ValueError("intersection contour contains an out-of-range EF pair")
            selected_roles = pair_roles[selected_pair_ids]
            if np.any(selected_roles > 1):
                raise ValueError("EF pair mesh role must be either 0 or 1")
            if not np.all(np.isfinite(pair_pos[selected_pair_ids])):
                raise ValueError("selected EF pair material positions contain NaN or Inf")
            if contour_filter >= 0 and np.any(
                contour_ids[selected_pair_ids] != np.uint32(contour_filter)
            ):
                raise ValueError("contour membership and EF pair contour indices disagree")

        edge_uv = pair_pos[:, :2]
        face_uv = pair_pos[:, 2:]
        edge_is_side0 = pair_roles == 0
        side0_uv = np.where(edge_is_side0[:, None], edge_uv, face_uv)
        side1_uv = np.where(edge_is_side0[:, None], face_uv, edge_uv)
        pair_side_uv = np.stack((side0_uv, side1_uv), axis=1)

        selected_mask = np.zeros(pair_count, dtype=bool)
        selected_mask[selected_pair_ids] = True
        graph_edges = set()
        for pair_idx in selected_pair_ids:
            pair_idx = int(pair_idx)
            begin = int(adj_prefix[pair_idx])
            end = int(adj_prefix[pair_idx + 1])
            for adj_pair_idx in adj_data[begin:end]:
                adj_pair_idx = int(adj_pair_idx)
                if adj_pair_idx < 0 or adj_pair_idx >= pair_count:
                    raise ValueError(
                        f"EF pair {pair_idx} has out-of-range adjacency {adj_pair_idx}"
                    )
                if pair_idx == adj_pair_idx or not selected_mask[adj_pair_idx]:
                    continue
                if contour_ids[pair_idx] != contour_ids[adj_pair_idx]:
                    continue
                graph_edges.add(
                    (min(pair_idx, adj_pair_idx), max(pair_idx, adj_pair_idx))
                )

        curve_nodes = []
        curve_edges = []
        curve_colors = []
        for pair_a, pair_b in sorted(graph_edges):
            color = self._face_color_from_index(int(contour_ids[pair_a]))
            for side in range(2):
                base = len(curve_nodes)
                curve_nodes.extend(
                    (pair_side_uv[pair_a, side], pair_side_uv[pair_b, side])
                )
                curve_edges.append((base, base + 1))
                curve_colors.append(color)

        point_uv = pair_side_uv[selected_pair_ids].reshape(-1, 2)
        point_colors = []
        for pair_idx in selected_pair_ids:
            color = self._face_color_from_index(int(contour_ids[int(pair_idx)]))
            point_colors.extend((color, color))

        contour_types = None
        if "contour_types" in contour_data:
            contour_types = np.asarray(contour_data["contour_types"], dtype=np.uint32)
            if contour_types.ndim != 1 or contour_types.shape[0] != num_contours:
                raise ValueError(
                    "contour_types must have one entry per intersection contour"
                )

        return {
            "curve_uv": np.asarray(curve_nodes, dtype=np.float32).reshape(-1, 2),
            "curve_edges": np.asarray(curve_edges, dtype=np.int32).reshape(-1, 2),
            "curve_colors": np.asarray(curve_colors, dtype=np.float32).reshape(-1, 3),
            "point_uv": np.asarray(point_uv, dtype=np.float32).reshape(-1, 2),
            "point_colors": np.asarray(point_colors, dtype=np.float32).reshape(-1, 3),
            "selected_pair_count": int(selected_pair_ids.shape[0]),
            "selected_contours": selected_contours,
            "num_contours": int(num_contours),
            "contour_types": contour_types,
        }

    def _build_uv_hit_geometry(
        self, global_material_positions, response_data, contour_filter
    ):
        material_positions = np.asarray(global_material_positions, dtype=np.float32)
        if material_positions.ndim != 2 or material_positions.shape[1] != 2:
            raise ValueError(
                "global material positions must have shape (N, 2), got "
                f"{material_positions.shape}"
            )
        if not np.all(np.isfinite(material_positions)):
            raise ValueError("global material positions contain NaN or Inf")

        pair_ids = self._require_debug_array(
            response_data,
            "target_point_template_pair_indices",
            2,
            width=4,
            dtype=np.uint32,
        )
        pair_weights = self._require_debug_array(
            response_data,
            "target_point_template_pair_weights",
            2,
            width=4,
            dtype=np.float32,
        )
        pair_types = self._require_debug_array(
            response_data,
            "target_point_template_pair_collision_type",
            1,
            dtype=np.uint32,
        )
        pair_meta = self._require_debug_array(
            response_data,
            "target_point_template_pairs_indices",
            2,
            width=4,
            dtype=np.uint32,
        )

        pair_count = pair_ids.shape[0]
        if (
            pair_weights.shape[0] != pair_count
            or pair_types.shape[0] != pair_count
            or pair_meta.shape[0] != pair_count
        ):
            raise ValueError("response pair arrays have different lengths")

        visual_regions = pair_meta[:, 2].astype(np.uint64, copy=False) >> 1
        contour_ids = visual_regions // self.RESPONSE_VISUAL_LABEL_COUNT
        solution_labels = visual_regions % self.RESPONSE_VISUAL_LABEL_COUNT
        selected_mask = np.ones(pair_count, dtype=bool)
        if int(contour_filter) >= 0:
            selected_mask &= contour_ids == np.uint64(int(contour_filter))

        supported_mask = (pair_types == 4) | (pair_types == 8)
        unsupported_count = int(np.count_nonzero(selected_mask & ~supported_mask))
        selected_indices = np.flatnonzero(selected_mask & supported_mask)

        if selected_indices.size > 0:
            selected_ids = pair_ids[selected_indices].astype(np.int64, copy=False)
            if int(selected_ids.min()) < 0 or int(selected_ids.max()) >= material_positions.shape[0]:
                raise ValueError("response pair contains an out-of-range global vertex id")
            if not np.all(np.isfinite(pair_weights[selected_indices])):
                raise ValueError("response pair weights contain NaN or Inf")

        src_uv = np.empty((selected_indices.shape[0], 2), dtype=np.float32)
        dst_uv = np.empty((selected_indices.shape[0], 2), dtype=np.float32)
        selected_types = pair_types[selected_indices]
        vf_local = np.flatnonzero(selected_types == 4)
        ee_local = np.flatnonzero(selected_types == 8)

        if vf_local.size > 0:
            rows = selected_indices[vf_local]
            ids = pair_ids[rows]
            weights = pair_weights[rows]
            src_uv[vf_local] = material_positions[ids[:, 0]]
            dst_uv[vf_local] = (
                weights[:, 1, None] * material_positions[ids[:, 1]]
                + weights[:, 2, None] * material_positions[ids[:, 2]]
                + weights[:, 3, None] * material_positions[ids[:, 3]]
            )

        if ee_local.size > 0:
            rows = selected_indices[ee_local]
            ids = pair_ids[rows]
            weights = pair_weights[rows]
            src_uv[ee_local] = (
                weights[:, 0, None] * material_positions[ids[:, 0]]
                + weights[:, 1, None] * material_positions[ids[:, 1]]
            )
            dst_uv[ee_local] = (
                weights[:, 2, None] * material_positions[ids[:, 2]]
                + weights[:, 3, None] * material_positions[ids[:, 3]]
            )

        if not np.all(np.isfinite(src_uv)) or not np.all(np.isfinite(dst_uv)):
            raise ValueError("weighted response material positions contain NaN or Inf")

        hit_uv = np.stack((src_uv, dst_uv), axis=1).reshape(-1, 2)
        hit_edges = np.arange(hit_uv.shape[0], dtype=np.int32).reshape(-1, 2)
        selected_regions = visual_regions[selected_indices]
        color_cache = {}
        hit_colors = []
        for visual_region in selected_regions:
            visual_region = int(visual_region)
            if visual_region not in color_cache:
                color_cache[visual_region] = self._face_color_from_index(
                    visual_region
                )
            hit_colors.append(color_cache[visual_region])

        return {
            "hit_uv": hit_uv.astype(np.float32, copy=False),
            "hit_edges": hit_edges,
            "hit_colors": np.asarray(hit_colors, dtype=np.float32).reshape(-1, 3),
            "hit_count": int(selected_indices.shape[0]),
            "vf_count": int(vf_local.shape[0]),
            "ee_count": int(ee_local.shape[0]),
            "unsupported_count": unsupported_count,
            "solution_labels": np.unique(solution_labels[selected_indices]),
        }

    def _primary_scene_structures(self):
        structures = []
        structures.extend(self._surface_meshes)
        structures.extend(
            structure
            for structure in (
                self._curve_network,
                self._highlight_cloud,
                self._highlight_curve,
            )
            if structure is not None
        )

        unique_structures = []
        seen = set()
        for structure in structures:
            identity = id(structure)
            if identity in seen:
                continue
            seen.add(identity)
            unique_structures.append(structure)
        return unique_structures

    @staticmethod
    def _get_structure_enabled(structure):
        getter = getattr(structure, "is_enabled", None)
        if not callable(getter):
            getter = getattr(structure, "get_enabled", None)
        if not callable(getter):
            raise RuntimeError(
                f"Polyscope structure {structure!r} does not expose enabled state"
            )
        return bool(getter())

    def _save_and_hide_primary_scene(self):
        self._uv_saved_structure_states = []
        for structure in self._primary_scene_structures():
            self._uv_saved_structure_states.append(
                (structure, self._get_structure_enabled(structure))
            )
            structure.set_enabled(False)

    def _restore_primary_scene(self):
        for structure, was_enabled in self._uv_saved_structure_states:
            structure.set_enabled(was_enabled)
        self._uv_saved_structure_states = []

    def _remove_uv_structures(self):
        structures = []
        structures.extend(self._uv_mesh_networks)
        structures.extend(self._uv_boundary_networks)
        structures.extend(
            structure
            for structure in (
                self._uv_contour_network,
                self._uv_contour_points,
                self._uv_hit_network,
                self._uv_hit_points,
            )
            if structure is not None
        )
        for structure in structures:
            structure.remove()

        self._uv_mesh_networks = []
        self._uv_boundary_networks = []
        self._uv_contour_network = None
        self._uv_contour_points = None
        self._uv_hit_network = None
        self._uv_hit_points = None

    def _refit_uv_camera(self, uv_points):
        uv_points = np.asarray(uv_points, dtype=np.float32)
        if uv_points.ndim != 2 or uv_points.shape[1] != 2 or uv_points.shape[0] == 0:
            raise ValueError("cannot fit the 2D camera without material-space points")
        if not np.all(np.isfinite(uv_points)):
            raise ValueError("cannot fit the 2D camera to NaN or Inf material positions")

        bbox_min = uv_points.min(axis=0)
        bbox_max = uv_points.max(axis=0)
        extent = float(np.max(bbox_max - bbox_min))
        if extent <= 1e-8:
            extent = 1.0
        margin = 0.06 * extent
        center = 0.5 * (bbox_min + bbox_max)
        low = (bbox_min[0] - margin, bbox_min[1] - margin, -0.01 * extent)
        high = (bbox_max[0] + margin, bbox_max[1] + margin, 0.01 * extent)

        ps.set_bounding_box(low, high)
        ps.set_background_color((1.0, 1.0, 1.0))
        ps.set_ground_plane_mode("none")
        ps.reset_camera_to_home_view()
        ps.set_view_projection_mode("orthographic")
        ps.look_at_dir(
            (float(center[0]), float(center[1]), 2.5 * extent),
            (float(center[0]), float(center[1]), 0.0),
            (0.0, 1.0, 0.0),
        )

    def _register_uv_structures(
        self, mesh_positions, contour_geometry, hit_geometry, uv_span
    ):
        self._remove_uv_structures()

        mesh_radius = max(4.5e-4 * uv_span, 1e-7)
        boundary_radius = max(1.0e-3 * uv_span, 2e-7)
        contour_radius = max(8.0e-4 * uv_span, 2e-7)
        hit_radius = max(5.0e-4 * uv_span, 1e-7)
        hit_point_radius = max(5.0e-4 * uv_span, 2e-7)
        mesh_color = (0.72, 0.86, 0.96)
        boundary_color = (0.26, 0.56, 0.78)

        for mesh_idx, uv in enumerate(mesh_positions):
            uv = np.asarray(uv, dtype=np.float32)
            edges = self._mesh_edges[mesh_idx]
            if edges.shape[0] > 0:
                network = ps.register_curve_network(
                    f"UV 2D | Mesh {mesh_idx} Wireframe",
                    self._embed_uv_points(uv, 0.0),
                    edges,
                )
                network.set_radius(mesh_radius, relative=False)
                network.set_color(mesh_color)
                network.set_enabled(True)
                self._uv_mesh_networks.append(network)

            boundary_edges = self._mesh_boundary_edges[mesh_idx]
            if self._uv_show_mesh_boundaries and boundary_edges.shape[0] > 0:
                boundary_network = ps.register_curve_network(
                    f"UV 2D | Mesh {mesh_idx} Boundary",
                    self._embed_uv_points(uv, 2.0e-4 * uv_span),
                    boundary_edges,
                )
                boundary_network.set_radius(boundary_radius, relative=False)
                boundary_network.set_color(boundary_color)
                boundary_network.set_enabled(True)
                self._uv_boundary_networks.append(boundary_network)

        if contour_geometry["curve_edges"].shape[0] > 0:
            self._uv_contour_network = ps.register_curve_network(
                "UV 2D | Intersection Contours",
                self._embed_uv_points(
                    contour_geometry["curve_uv"], 5.0e-4 * uv_span
                ),
                contour_geometry["curve_edges"],
            )
            self._uv_contour_network.set_radius(contour_radius, relative=False)
            self._uv_contour_network.set_enabled(True)
            self._add_color_quantity(
                self._uv_contour_network,
                "Contour Colors",
                contour_geometry["curve_colors"],
                defined_on="edges",
            )

        if hit_geometry["hit_edges"].shape[0] > 0:
            soft_hit_colors = np.clip(
                0.42 * hit_geometry["hit_colors"] + 0.58, 0.0, 1.0
            ).astype(np.float32, copy=False)
            self._uv_hit_network = ps.register_curve_network(
                "UV 2D | RayCasting Hits",
                self._embed_uv_points(hit_geometry["hit_uv"], 3.0e-4 * uv_span),
                hit_geometry["hit_edges"],
            )
            self._uv_hit_network.set_radius(hit_radius, relative=False)
            self._uv_hit_network.set_enabled(True)
            self._add_color_quantity(
                self._uv_hit_network,
                "Hit Solution Colors",
                soft_hit_colors,
                defined_on="edges",
            )

            soft_hit_point_colors = np.clip(
                0.60 * hit_geometry["hit_colors"] + 0.40, 0.0, 1.0
            ).astype(np.float32, copy=False)
            hit_point_colors = np.repeat(soft_hit_point_colors, 2, axis=0)
            self._uv_hit_points = ps.register_point_cloud(
                "UV 2D | RayCasting Hit Endpoints",
                self._embed_uv_points(
                    hit_geometry["hit_uv"], 4.0e-4 * uv_span
                ),
            )
            self._uv_hit_points.set_radius(hit_point_radius, relative=False)
            self._uv_hit_points.add_color_quantity(
                "Hit Endpoint Colors", hit_point_colors, enabled=True
            )
            self._uv_hit_points.set_enabled(True)

    def _collect_uv_visualization_data(self):
        getter = getattr(self._solver, "get_material_position_data", None)
        if not callable(getter):
            raise RuntimeError(
                "lcs_py does not expose get_material_position_data(); rebuild the Python bindings"
            )

        material_data = getter()
        if "global_positions" not in material_data or "mesh_positions" not in material_data:
            raise KeyError("material position data is missing global or per-mesh positions")
        global_positions = np.asarray(
            material_data["global_positions"], dtype=np.float32
        )
        mesh_positions = [
            np.asarray(uv, dtype=np.float32)
            for uv in material_data["mesh_positions"]
        ]
        if len(mesh_positions) != len(self._mesh_edges):
            raise ValueError(
                "material-space mesh count does not match the registered render meshes"
            )

        finite_mesh_positions = []
        for mesh_idx, uv in enumerate(mesh_positions):
            if uv.ndim != 2 or uv.shape[1] != 2:
                raise ValueError(
                    f"mesh {mesh_idx} material positions must have shape (N, 2), got {uv.shape}"
                )
            if not np.all(np.isfinite(uv)):
                raise ValueError(
                    f"mesh {mesh_idx} material positions contain NaN or Inf"
                )
            edges = self._mesh_edges[mesh_idx]
            if edges.shape[0] > 0 and (
                int(edges.min()) < 0 or int(edges.max()) >= uv.shape[0]
            ):
                raise ValueError(
                    f"mesh {mesh_idx} wireframe references an out-of-range material vertex"
                )
            if uv.shape[0] > 0:
                finite_mesh_positions.append(uv)

        if not finite_mesh_positions:
            raise ValueError("the scene has no material-space vertices")

        contour_data = self._solver.get_intersection_contour_data()
        response_data = self._solver.get_response_pair_data()
        contour_geometry = self._build_uv_contour_geometry(
            contour_data, self._uv_contour_filter
        )
        hit_geometry = self._build_uv_hit_geometry(
            global_positions, response_data, self._uv_contour_filter
        )

        all_mesh_uv = np.concatenate(finite_mesh_positions, axis=0)
        bbox_min = all_mesh_uv.min(axis=0)
        bbox_max = all_mesh_uv.max(axis=0)
        uv_span = max(float(np.max(bbox_max - bbox_min)), 1e-6)

        return {
            "mesh_positions": mesh_positions,
            "all_mesh_uv": all_mesh_uv,
            "uv_span": uv_span,
            "contour_geometry": contour_geometry,
            "hit_geometry": hit_geometry,
        }

    def _uv_filter_text(self, contour_geometry):
        if self._uv_contour_filter < 0:
            return "all"

        contour_type_text = "unknown"
        contour_types = contour_geometry["contour_types"]
        if contour_types is not None:
            contour_type = int(contour_types[self._uv_contour_filter])
            contour_type_text = self.CONTOUR_TYPE_NAMES.get(
                contour_type, f"type {contour_type}"
            )
        return f"{self._uv_contour_filter} ({contour_type_text})"

    def _format_uv_summary(self, contour_geometry, hit_geometry):
        summary = (
            f"contour={self._uv_filter_text(contour_geometry)}; visible contours="
            f"{len(contour_geometry['selected_contours'])}/"
            f"{contour_geometry['num_contours']}; EF pairs="
            f"{contour_geometry['selected_pair_count']}; contour segments="
            f"{contour_geometry['curve_edges'].shape[0]}; hits="
            f"{hit_geometry['hit_count']} (VF {hit_geometry['vf_count']}, "
            f"EE {hit_geometry['ee_count']})"
        )
        if hit_geometry["unsupported_count"]:
            summary += (
                f"; skipped unsupported response types="
                f"{hit_geometry['unsupported_count']}"
            )
        return summary

    def _refresh_uv_2d_visualization(self, refit_camera=False):
        uv_data = self._collect_uv_visualization_data()
        self._register_uv_structures(
            uv_data["mesh_positions"],
            uv_data["contour_geometry"],
            uv_data["hit_geometry"],
            uv_data["uv_span"],
        )

        if refit_camera:
            self._refit_uv_camera(uv_data["all_mesh_uv"])

        self._uv_status = self._format_uv_summary(
            uv_data["contour_geometry"], uv_data["hit_geometry"]
        )

    def _next_uv_png_path(self):
        frame_idx = int(self._config.current_frame)
        contour_token = (
            "all"
            if self._uv_contour_filter < 0
            else f"{self._uv_contour_filter:04d}"
        )
        output_dir = os.path.abspath(os.path.join(self._output_dir, "uv_2d"))
        os.makedirs(output_dir, exist_ok=True)
        stem = f"frame_{frame_idx:06d}_contour_{contour_token}"
        path = os.path.join(output_dir, f"{stem}.png")
        if not os.path.exists(path):
            return path

        for suffix in range(1, 10000):
            path = os.path.join(output_dir, f"{stem}_{suffix:02d}.png")
            if not os.path.exists(path):
                return path
        raise RuntimeError(f"too many PNG files share the output stem '{stem}'")

    def _write_uv_png(self, output_path, uv_data):
        try:
            from matplotlib.backends.backend_agg import FigureCanvasAgg
            from matplotlib.collections import LineCollection
            from matplotlib.figure import Figure
            from matplotlib.lines import Line2D
        except ImportError as exc:
            raise RuntimeError(
                "matplotlib is required to save the 2D material-space PNG"
            ) from exc

        contour_geometry = uv_data["contour_geometry"]
        hit_geometry = uv_data["hit_geometry"]
        width_px = self.UV_PNG_WIDTH_PX
        height_px = self.UV_PNG_HEIGHT_PX
        dpi = self.UV_PNG_DPI
        figure = Figure(
            figsize=(width_px / dpi, height_px / dpi),
            dpi=dpi,
            facecolor="white",
        )
        FigureCanvasAgg(figure)
        axes = figure.add_subplot(1, 1, 1)
        axes.set_facecolor("white")
        figure.subplots_adjust(left=0.065, right=0.985, bottom=0.075, top=0.91)

        mesh_color = "#b7dbf2"
        boundary_color = "#438fbd"
        mesh_line_width = 0.38
        boundary_line_width = 0.9

        for mesh_idx, uv in enumerate(uv_data["mesh_positions"]):
            edges = self._mesh_edges[mesh_idx]
            if edges.shape[0] > 0:
                axes.add_collection(
                    LineCollection(
                        uv[edges],
                        colors=mesh_color,
                        linewidths=mesh_line_width,
                        alpha=0.82,
                        zorder=1,
                    )
                )

            boundary_edges = self._mesh_boundary_edges[mesh_idx]
            if self._uv_show_mesh_boundaries and boundary_edges.shape[0] > 0:
                axes.add_collection(
                    LineCollection(
                        uv[boundary_edges],
                        colors=boundary_color,
                        linewidths=boundary_line_width,
                        alpha=0.95,
                        zorder=2,
                    )
                )

        if hit_geometry["hit_edges"].shape[0] > 0:
            hit_segments = hit_geometry["hit_uv"][hit_geometry["hit_edges"]]
            axes.add_collection(
                LineCollection(
                    hit_segments,
                    colors=hit_geometry["hit_colors"],
                    linewidths=0.38,
                    alpha=0.20,
                    zorder=3,
                )
            )

        if hit_geometry["hit_uv"].shape[0] > 0:
            soft_hit_point_colors = np.clip(
                0.60 * hit_geometry["hit_colors"] + 0.40, 0.0, 1.0
            )
            hit_point_colors = np.repeat(soft_hit_point_colors, 2, axis=0)
            axes.scatter(
                hit_geometry["hit_uv"][:, 0],
                hit_geometry["hit_uv"][:, 1],
                c=hit_point_colors,
                s=3.5,
                edgecolors="none",
                alpha=0.72,
                zorder=4,
            )

        if contour_geometry["curve_edges"].shape[0] > 0:
            contour_segments = contour_geometry["curve_uv"][
                contour_geometry["curve_edges"]
            ]
            axes.add_collection(
                LineCollection(
                    contour_segments,
                    colors=contour_geometry["curve_colors"],
                    linewidths=0.72,
                    alpha=0.98,
                    zorder=5,
                )
            )

        all_mesh_uv = uv_data["all_mesh_uv"]
        mesh_bbox_min = all_mesh_uv.min(axis=0)
        mesh_bbox_max = all_mesh_uv.max(axis=0)
        mesh_span = max(float(np.max(mesh_bbox_max - mesh_bbox_min)), 1e-6)
        focus_arrays = []
        if self._uv_contour_filter >= 0:
            if contour_geometry["point_uv"].shape[0] > 0:
                focus_arrays.append(contour_geometry["point_uv"])
            if hit_geometry["hit_uv"].shape[0] > 0:
                focus_arrays.append(hit_geometry["hit_uv"])

        if focus_arrays:
            focus_uv = np.concatenate(focus_arrays, axis=0)
            bbox_min = focus_uv.min(axis=0)
            bbox_max = focus_uv.max(axis=0)
            span = max(
                float(np.max(bbox_max - bbox_min)), 0.02 * mesh_span, 1e-6
            )
            margin = 0.07 * span
        else:
            bbox_min = mesh_bbox_min
            bbox_max = mesh_bbox_max
            margin = 0.035 * mesh_span
        axes.set_xlim(float(bbox_min[0] - margin), float(bbox_max[0] + margin))
        axes.set_ylim(float(bbox_min[1] - margin), float(bbox_max[1] + margin))
        axes.set_aspect("equal", adjustable="box")
        axes.set_xlabel("material u")
        axes.set_ylabel("material v")
        axes.tick_params(colors="#4b5563", labelsize=8)
        for spine in axes.spines.values():
            spine.set_color("#cbd5e1")
            spine.set_linewidth(0.7)

        frame_idx = int(self._config.current_frame)
        filter_text = self._uv_filter_text(contour_geometry)
        axes.set_title(
            f"Frame {frame_idx} | Contour {filter_text} | "
            f"{hit_geometry['hit_count']} hits "
            f"(VF {hit_geometry['vf_count']}, EE {hit_geometry['ee_count']})",
            pad=12.0,
        )

        contour_sample_color = (
            contour_geometry["curve_colors"][0]
            if contour_geometry["curve_colors"].shape[0] > 0
            else (0.85, 0.24, 0.24)
        )
        hit_sample_color = (
            hit_geometry["hit_colors"][0]
            if hit_geometry["hit_colors"].shape[0] > 0
            else (0.36, 0.24, 0.85)
        )
        soft_hit_sample_color = np.clip(
            0.42 * np.asarray(hit_sample_color) + 0.58, 0.0, 1.0
        )
        soft_hit_point_sample_color = np.clip(
            0.60 * np.asarray(hit_sample_color) + 0.40, 0.0, 1.0
        )
        legend_items = [
            Line2D(
                [0],
                [0],
                color=mesh_color,
                linewidth=1.2,
                label="Mesh wireframe",
            )
        ]
        if self._uv_show_mesh_boundaries:
            legend_items.append(
                Line2D(
                    [0],
                    [0],
                    color=boundary_color,
                    linewidth=1.8,
                    label="Mesh boundary (one adjacent face)",
                )
            )
        legend_items.extend(
            (
                Line2D(
                    [0],
                    [0],
                    color=contour_sample_color,
                    linewidth=1.25,
                    label=(
                        "Intersection contour "
                        f"({contour_geometry['selected_pair_count']} EF pairs)"
                    ),
                ),
                Line2D(
                    [0],
                    [0],
                    color=soft_hit_sample_color,
                    linewidth=1.0,
                    label="RayCasting hit link (faint)",
                ),
                Line2D(
                    [0],
                    [0],
                    color="none",
                    marker="o",
                    markerfacecolor=soft_hit_point_sample_color,
                    markeredgecolor="none",
                    markersize=3.0,
                    alpha=0.72,
                    label="Hit endpoints (color = contour/Solution)",
                ),
            )
        )
        axes.legend(
            handles=legend_items,
            loc="upper right",
            frameon=True,
            facecolor="white",
            edgecolor="#d1d5db",
            framealpha=0.92,
            fontsize=8,
        )

        figure.savefig(
            output_path,
            format="png",
            dpi=dpi,
            facecolor="white",
            metadata={"Software": "LuisaComputeSimulator"},
        )
        figure.clear()
        return width_px, height_px

    def _save_uv_2d_png(self):
        uv_data = self._collect_uv_visualization_data()
        output_path = self._next_uv_png_path()
        width_px, height_px = self._write_uv_png(output_path, uv_data)
        self._uv_status = (
            f"Saved {width_px}x{height_px} PNG: {output_path}; "
            + self._format_uv_summary(
                uv_data["contour_geometry"], uv_data["hit_geometry"]
            )
        )
        print(f"[SimulationGUI] {self._uv_status}")
        return output_path

    def _enter_uv_2d_view(self):
        if self._uv_2d_enabled:
            self._refresh_uv_2d_visualization(refit_camera=True)
            return

        self._uv_previous_view = ps.get_view_as_json()
        self._uv_previous_background = ps.get_background_color()
        self._uv_previous_projection = ps.get_view_projection_mode()
        self._uv_previous_bbox = ps.get_bounding_box()
        self._save_and_hide_primary_scene()
        self._uv_2d_enabled = True

        try:
            self._refresh_uv_2d_visualization(refit_camera=True)
        except Exception:
            self._uv_2d_enabled = False
            self._remove_uv_structures()
            self._restore_primary_scene()
            if self._uv_previous_bbox is not None:
                ps.set_bounding_box(*self._uv_previous_bbox)
            if self._uv_previous_background is not None:
                ps.set_background_color(self._uv_previous_background)
            if self._uv_previous_projection is not None:
                ps.set_view_projection_mode(self._uv_previous_projection)
            if self._uv_previous_view is not None:
                ps.set_view_from_json(self._uv_previous_view)
            self._clear_uv_view_snapshot()
            self._apply_ground_plane()
            raise

    def _clear_uv_view_snapshot(self):
        self._uv_previous_view = None
        self._uv_previous_background = None
        self._uv_previous_projection = None
        self._uv_previous_bbox = None

    def _leave_uv_2d_view(self):
        if not self._uv_2d_enabled:
            return

        self._uv_2d_enabled = False
        self._remove_uv_structures()
        self._restore_primary_scene()
        if self._uv_previous_bbox is not None:
            ps.set_bounding_box(*self._uv_previous_bbox)
        if self._uv_previous_background is not None:
            ps.set_background_color(self._uv_previous_background)
        if self._uv_previous_projection is not None:
            ps.set_view_projection_mode(self._uv_previous_projection)
        if self._uv_previous_view is not None:
            ps.set_view_from_json(self._uv_previous_view)
        self._clear_uv_view_snapshot()
        self._apply_ground_plane()
        self._update_gui_vertices()
        self._uv_status = "Returned to the 3D scene."

    def _update_mesh_visualization(self, v_list):
        """Push the latest simulation vertices to polyscope."""
        if self._use_merged_render:
            merged_v = np.concatenate([np.asarray(v) for v in v_list], axis=0)
            self._surface_meshes[0].update_vertex_positions(merged_v)
            return

        for idx, mesh in enumerate(self._surface_meshes):
            mesh.update_vertex_positions(v_list[idx])

    def _update_gui_vertices(self):
        """Fetch latest simulation vertices and push them to polyscope."""
        if self._uv_2d_enabled:
            try:
                self._refresh_uv_2d_visualization(refit_camera=False)
            except Exception as exc:
                self._uv_status = f"2D material-space refresh failed: {exc}"
                print(f"[SimulationGUI] {self._uv_status}")
            return

        v_list, _ = self._solver.get_sim_result()
        self._update_mesh_visualization(v_list)

        self._register_or_update_curve_network()

        self._update_highlight_marker(v_list)

    def _register_or_update_curve_network(self):
        curve_vertices, curve_edges, curve_edge_colors = self._solver.get_visualize_curves()

        if curve_vertices.shape[0] == 0 or curve_edges.shape[0] == 0:
            if self._curve_network is not None:
                self._curve_network.set_enabled(False)
            return

        nodes = np.asarray(curve_vertices, dtype=np.float32)
        edges = np.asarray(curve_edges, dtype=np.int32)
        edge_colors = np.asarray(curve_edge_colors, dtype=np.float32)

        # Re-register each frame to match C++ GUI behavior and changing edge topology.
        self._curve_network = ps.register_curve_network(
            "Visualize Curves", nodes, edges
        )
        self._curve_network.set_radius(5e-4, relative=False)
        self._curve_network.set_enabled(True)

        if edge_colors.shape[0] == edges.shape[0]:
            try:
                self._curve_network.add_color_quantity(
                    "Curve Colors", edge_colors, defined_on="edges", enabled=True
                )
            except TypeError:
                # Backward-compatible fallback for older polyscope Python bindings.
                self._curve_network.add_color_quantity(
                    "Curve Colors", edge_colors, enabled=True
                )

    def _resolve_global_vid(self, global_vid: int):
        if self._resolve_prefixed_index(self._vert_prefix, int(global_vid)) is None:
            raise ValueError(f"vertex id {global_vid} is out of range")
        world_data_idx = int(self._solver.query_registration_id_from_global_vid(global_vid))
        local_vid = int(self._solver.query_local_vid_from_global_vid(global_vid))
        return world_data_idx, local_vid

    def _resolve_global_element(self, element_type: str, global_id: int):
        global_id = int(global_id)
        if element_type == "vertex":
            return self._resolve_global_vid(global_id)

        prefix = self._edge_prefix if element_type == "edge" else self._face_prefix
        resolved = self._resolve_prefixed_index(prefix, global_id)
        if resolved is None:
            raise ValueError(f"{element_type} id {global_id} is out of range")
        return resolved

    def _set_highlight_element(self, element_type: str, global_id: int):
        mesh_idx, local_idx = self._resolve_global_element(element_type, global_id)
        global_id = int(global_id)

        self._highlight_input_id = global_id
        self._highlight_element_type = element_type
        self._highlight_element_global_id = global_id
        self._highlight_mesh_idx = mesh_idx
        self._highlight_local_idx = local_idx
        self._highlight_vertex_adj_edges = []
        self._last_highlight_adj_edge = None

        label = {
            "vertex": "Vert",
            "edge": "Edge",
            "face": "Face",
        }[element_type]
        extra_info = ""
        if element_type == "vertex":
            self._highlight_vertex_adj_edges = [
                int(edge_id) for edge_id in self._solver.query_vert_adj_edges(global_id)
            ]
            extra_info = f", adj_edges={self._highlight_vertex_adj_edges}"

        print(
            f"Highlight {label} {global_id:3d} on mesh {mesh_idx} (local {local_idx}){extra_info}"
        )

    def _set_highlight_global_vid(self, global_vid: int):
        self._set_highlight_element("vertex", global_vid)

    def _build_highlight_edge_curve(self, v_list, mesh_idx: int, local_edge_idx: int):
        if mesh_idx < 0 or mesh_idx >= len(v_list):
            return None

        verts = np.asarray(v_list[mesh_idx], dtype=np.float32)
        edges = self._mesh_edges[mesh_idx]
        if local_edge_idx < 0 or local_edge_idx >= edges.shape[0]:
            return None

        edge = np.asarray(edges[local_edge_idx], dtype=np.int32)
        nodes = np.asarray([verts[int(edge[0])], verts[int(edge[1])]], dtype=np.float32)
        segments = np.asarray([[0, 1]], dtype=np.int32)
        return nodes, segments

    def _build_highlight_face_curve(self, v_list, mesh_idx: int, local_face_idx: int):
        if mesh_idx < 0 or mesh_idx >= len(v_list):
            return None

        verts = np.asarray(v_list[mesh_idx], dtype=np.float32)
        faces = self._mesh_faces[mesh_idx]
        if local_face_idx < 0 or local_face_idx >= faces.shape[0]:
            return None

        face = np.asarray(faces[local_face_idx], dtype=np.int32)
        nodes = np.asarray(
            [verts[int(face[0])], verts[int(face[1])], verts[int(face[2])]],
            dtype=np.float32,
        )
        segments = np.asarray([[0, 1], [1, 2], [2, 0]], dtype=np.int32)
        return nodes, segments

    def _get_vertex_adj_edge_curve(self, v_list):
        if not self._highlight_vertex_adj_edges:
            self._last_highlight_adj_edge = None
            return None

        adj_idx = int(self._config.current_frame) % len(self._highlight_vertex_adj_edges)
        global_edge_id = self._highlight_vertex_adj_edges[adj_idx]
        resolved = self._resolve_prefixed_index(self._edge_prefix, global_edge_id)
        if resolved is None:
            return None

        if self._last_highlight_adj_edge != global_edge_id:
            print(
                f"Highlight Vert {self._highlight_element_global_id:3d} -> adj Edge {global_edge_id:3d} ({adj_idx + 1}/{len(self._highlight_vertex_adj_edges)})"
            )
            self._last_highlight_adj_edge = global_edge_id

        return self._build_highlight_edge_curve(v_list, resolved[0], resolved[1])

    def _update_highlight_curve(self, curve_data, color):
        if curve_data is None:
            self._disable_structure(self._highlight_curve)
            return

        nodes, segments = curve_data
        edge_colors = np.tile(np.asarray(color, dtype=np.float32), (segments.shape[0], 1))
        self._highlight_curve = ps.register_curve_network(
            "highlight_element", nodes, segments
        )
        self._highlight_curve.set_radius(5e-3, relative=False)
        self._highlight_curve.set_enabled(True)
        self._add_color_quantity(
            self._highlight_curve,
            "highlight_color",
            edge_colors,
            defined_on="edges",
        )

    def _update_highlight_marker(self, v_list):
        if self._highlight_element_type is None:
            self._disable_structure(self._highlight_cloud)
            self._disable_structure(self._highlight_curve)
            return

        if self._highlight_mesh_idx is None or self._highlight_local_idx is None:
            return

        if self._highlight_mesh_idx < 0 or self._highlight_mesh_idx >= len(v_list):
            return

        if self._highlight_element_type == "vertex":
            verts = np.asarray(v_list[self._highlight_mesh_idx], dtype=np.float32)
            if self._highlight_local_idx < 0 or self._highlight_local_idx >= verts.shape[0]:
                return

            point = np.asarray([verts[self._highlight_local_idx]], dtype=np.float32)
            if self._highlight_cloud is None:
                self._highlight_cloud = ps.register_point_cloud("highlight_vertex", point)
                color = np.asarray([[1.0, 0.15, 0.1]], dtype=np.float32)
                self._highlight_cloud.add_color_quantity(
                    "highlight_color", color, enabled=True
                )
            else:
                self._highlight_cloud.update_point_positions(point)
            self._highlight_cloud.set_enabled(True)

            self._update_highlight_curve(
                self._get_vertex_adj_edge_curve(v_list),
                color=np.asarray([1.0, 0.6, 0.1], dtype=np.float32),
            )
            return

        self._disable_structure(self._highlight_cloud)

        if self._highlight_element_type == "edge":
            curve_data = self._build_highlight_edge_curve(
                v_list, self._highlight_mesh_idx, self._highlight_local_idx
            )
        elif self._highlight_element_type == "face":
            curve_data = self._build_highlight_face_curve(
                v_list, self._highlight_mesh_idx, self._highlight_local_idx
            )
        else:
            curve_data = None

        self._update_highlight_curve(
            curve_data,
            color=np.asarray([1.0, 0.15, 0.1], dtype=np.float32),
        )

    def _physics_step(self):
        """Run one simulation step (GPU or CPU depending on config)."""
        if self._config.use_gpu:
            self._solver.physics_step_gpu()
        else:
            self._solver.physics_step_cpu()

    # ImGui callback (called every frame by polyscope).

    def _ui_callback(self):
        self._handle_keyboard()
        self._handle_selection()
        self._panel_parameters()
        self._panel_simulation()
        self._panel_material_space_2d()
        self._panel_highlight()
        self._panel_untangling()
        self._panel_collision()
        self._panel_data_io()
        self._continuous_simulation_loop()

    # ---- Keyboard shortcuts ----------------------------------------------

    def _handle_keyboard(self):
        # Escape -> close window  (ImGuiKey_Escape = 526)
        if psim.IsKeyPressed(psim.ImGuiKey_Escape):
            ps.unshow()
        # Space -> advance single frame  (ImGuiKey_Space = 525)
        if psim.IsKeyPressed(psim.ImGuiKey_Space):
            self._physics_step()
            self._update_gui_vertices()

    # ---- Parameters panel ------------------------------------------------

    def _panel_parameters(self):
        cfg = self._config
        if psim.TreeNode("Parameters"):
            _, cfg.nonlinear_iter_count = psim.InputInt(
                "Num Nonlinear-Iteration", cfg.nonlinear_iter_count
            )
            _, cfg.pcg_iter_count = psim.InputInt(
                "Num PCG-Iteration", cfg.pcg_iter_count
            )
            _, cfg.implicit_dt = psim.SliderFloat(
                "Implicit Timestep", cfg.implicit_dt, v_min=0.0001, v_max=0.2
            )
            _, cfg.use_energy_linesearch = psim.Checkbox(
                "Use Energy LineSearch", cfg.use_energy_linesearch
            )
            _, cfg.use_ccd_linesearch = psim.Checkbox(
                "Use CCD LineSearch", cfg.use_ccd_linesearch
            )
            _, cfg.use_global_ccd = psim.Checkbox(
                "Use Global CCD", cfg.use_global_ccd
            )
            _, cfg.print_system_energy = psim.Checkbox(
                "Print Energy", cfg.print_system_energy
            )
            _, cfg.use_gpu = psim.Checkbox("Use GPU Solver", cfg.use_gpu)
            _, cfg.use_self_collision = psim.Checkbox(
                "Use Self-Collision", cfg.use_self_collision
            )
            _, cfg.use_quasi_static_mode = psim.Checkbox(
                "Use Quasi-Static Mode", cfg.use_quasi_static_mode
            )
            psim.TreePop()
        if cfg.print_system_energy:
            cfg.use_energy_linesearch = True 
            # Legacy system-energy overlay retained for reference.

    # ---- Simulation panel ------------------------------------------------

    def _panel_simulation(self):
        cfg = self._config
        # Prefer TreeNodeEx default-open; gracefully fallback for older bindings.
        tree_node_ex = getattr(psim, "TreeNodeEx", None)
        if tree_node_ex is not None:
            default_open_flag = getattr(psim, "ImGuiTreeNodeFlags_DefaultOpen", 0)
            is_open = tree_node_ex("Simulation", default_open_flag)
        else:
            set_next_item_open = getattr(psim, "SetNextItemOpen", None)
            if set_next_item_open is not None:
                set_next_item_open(True, getattr(psim, "ImGuiCond_Once", 0))
            is_open = psim.TreeNode("Simulation")

        if is_open:
            psim.TextUnformatted(f"Frame {cfg.current_frame}")

            if psim.Button("Reset"):
                cfg.current_frame = 0
                self._solver.restart_system()
                self._update_gui_vertices()

            if psim.Button("Advance Single Frame"):
                self._physics_step()
                self._update_gui_vertices()

            if psim.Button("Start Simulation"):
                self._is_simulating = True

            if psim.Button("End Simulation"):
                self._is_simulating = False

            psim.TreePop()

    # ---- Highlight panel ------------------------------------------------
    def _panel_highlight(self):
        if psim.TreeNode("Highlight"):
            changed, highlight_id = psim.InputInt(
                "Highlight Element ID", self._highlight_input_id
            )
            if changed:
                self._highlight_input_id = max(0, int(highlight_id))

            if psim.Button("Highlight Vertex"):
                try:
                    self._set_highlight_element("vertex", self._highlight_input_id)
                    self._update_gui_vertices()
                except Exception as exc:
                    print(f"[SimulationGUI] failed to highlight vertex: {exc}")

            if psim.Button("Highlight Edge"):
                try:
                    self._set_highlight_element("edge", self._highlight_input_id)
                    self._update_gui_vertices()
                except Exception as exc:
                    print(f"[SimulationGUI] failed to highlight edge: {exc}")

            if psim.Button("Highlight Face"):
                try:
                    self._set_highlight_element("face", self._highlight_input_id)
                    self._update_gui_vertices()
                except Exception as exc:
                    print(f"[SimulationGUI] failed to highlight face: {exc}")

            if (
                self._highlight_element_type is not None
                and self._highlight_element_global_id is not None
                and self._highlight_mesh_idx is not None
                and self._highlight_local_idx is not None
            ):
                psim.TextUnformatted(
                    f"type={self._highlight_element_type}, global={self._highlight_element_global_id}, mesh_idx={self._highlight_mesh_idx}, local_idx={self._highlight_local_idx}"
                )
                if self._highlight_element_type == "vertex":
                    if self._highlight_vertex_adj_edges:
                        adj_idx = int(self._config.current_frame) % len(
                            self._highlight_vertex_adj_edges
                        )
                        curr_edge = self._highlight_vertex_adj_edges[adj_idx]
                        psim.TextUnformatted(
                            f"adj_edge={curr_edge} ({adj_idx + 1}/{len(self._highlight_vertex_adj_edges)})"
                        )
                    else:
                        psim.TextUnformatted("adj_edge=none")

            psim.TreePop()

    def _panel_material_space_2d(self):
        if psim.TreeNode("2D Material Space"):
            changed, contour_filter = psim.InputInt(
                "Contour Filter (-1 = All)", self._uv_contour_filter
            )
            if changed:
                self._uv_contour_filter = max(-1, int(contour_filter))

            changed_boundary, self._uv_show_mesh_boundaries = psim.Checkbox(
                "Emphasize Mesh Boundary (One Adjacent Face)",
                self._uv_show_mesh_boundaries,
            )
            if changed_boundary and self._uv_2d_enabled:
                try:
                    self._refresh_uv_2d_visualization(refit_camera=False)
                except Exception as exc:
                    self._uv_status = f"2D material-space refresh failed: {exc}"
                    print(f"[SimulationGUI] {self._uv_status}")

            if self._uv_2d_enabled:
                if psim.Button("Refresh / Refit 2D View"):
                    try:
                        self._refresh_uv_2d_visualization(refit_camera=True)
                    except Exception as exc:
                        self._uv_status = (
                            f"2D material-space refresh failed: {exc}"
                        )
                        print(f"[SimulationGUI] {self._uv_status}")
                psim.SameLine()
                if psim.Button("Return to 3D"):
                    self._leave_uv_2d_view()
            elif psim.Button("Show 2D Material Space"):
                try:
                    self._enter_uv_2d_view()
                except Exception as exc:
                    self._uv_status = f"Cannot open 2D material space: {exc}"
                    print(f"[SimulationGUI] {self._uv_status}")

            if psim.Button(
                f"Save High-Resolution PNG "
                f"({self.UV_PNG_WIDTH_PX}x{self.UV_PNG_HEIGHT_PX})"
            ):
                try:
                    self._save_uv_2d_png()
                except Exception as exc:
                    self._uv_status = f"Cannot save 2D material-space PNG: {exc}"
                    print(f"[SimulationGUI] {self._uv_status}")

            psim.TextWrapped(
                "Light blue: mesh wireframe; blue: true mesh boundary; "
                "thin contour edges: intersection_contour index; faint Hit "
                "links with highlighted endpoints: packed "
                "(contour, selected Solution) label."
            )
            psim.TextWrapped(
                "PNG export uses the current contour filter and is saved under "
                "the output directory's uv_2d folder. A single-contour export "
                "automatically fits its contour and Hit extent."
            )
            psim.TextWrapped(self._uv_status)
            psim.TreePop()

    # ---- Untangling panel ------------------------------------------------

    def _panel_untangling(self):
        cfg = self._config
        if psim.TreeNode("Untangling"):
            _, cfg.use_untangling = psim.Checkbox("Use Untangling", cfg.use_untangling)
            _, cfg.untangling_process_contours_count = psim.InputInt(
                "Process Contours Count", cfg.untangling_process_contours_count
            )
            _, cfg.untangling_response_depth = psim.SliderFloat(
                "Untangling Response Depth Large",
                cfg.untangling_response_depth,
                v_min=1e-3,
                v_max=2e1,
            )
            _, cfg.untangling_response_depth = psim.SliderFloat(
                "Untangling Response Depth Small",
                cfg.untangling_response_depth,
                v_min=1e-3,
                v_max=1e-2,
            )

            psim.TreePop()

    # ---- Collision panel -------------------------------------------------

    def _apply_ground_plane(self):
        if self._uv_2d_enabled:
            ps.set_ground_plane_mode("none")
        elif self._config.use_floor:
            ps.set_ground_plane_mode("tile_reflection")
            ps.set_ground_plane_height_mode("manual")
            ps.set_ground_plane_height(self._config.floor.y)
        else:
            ps.set_ground_plane_mode("none")

    def _panel_collision(self):
        cfg = self._config
        if psim.TreeNode("Collision"):
            _, cfg.use_floor = psim.Checkbox("Use Ground Collision", cfg.use_floor)
            if cfg.use_floor:
                changed, new_floor_y = psim.SliderFloat(
                    "Floor Y", cfg.floor.y, v_min=-1.0, v_max=1.0
                )
                if changed:
                    floor_vec = cfg.floor
                    floor_vec.y = new_floor_y
                    cfg.floor = floor_vec
            psim.TreePop()
        self._apply_ground_plane()

    # ---- Data IO panel ---------------------------------------------------

    def _panel_data_io(self):
        cfg = self._config
        if psim.TreeNode("Data IO"):
            if psim.Button("Save State"):
                state_path = os.path.join(self._output_dir, f"state_frame_{cfg.current_frame}.state")
                self._solver.save_current_state(state_path)
                print(f"Saved state to {state_path}")
            if psim.Button("Load State"):
                state_path = os.path.join(self._output_dir, f"state_frame_{cfg.load_state_frame}.state")
                if os.path.exists(state_path):
                    self._solver.load_target_state(state_path)
                    self._update_gui_vertices()
                    print(f"Loaded state from {state_path}")
                else:
                    print(f"State file not found: {state_path}")
            _, cfg.load_state_frame = psim.InputInt(
                "Load State Frame", cfg.load_state_frame
            )
            
            if psim.Button("Save mesh combined"):
                obj_path = os.path.join(self._output_dir, f"frame_{cfg.current_frame}_combined.obj")
                self._solver.save_sim_result(obj_path=obj_path)
                print(f"Saved combined mesh to {obj_path}")
            if psim.Button("Save mesh separate"):
                from utils.mesh_proc import write_obj
                v_list, f_list = self._solver.get_sim_result()
                for idx in range(len(v_list)):
                    verts = np.asarray(v_list[idx])
                    faces = np.asarray(f_list[idx], dtype=np.int32)
                    obj_path = os.path.join(self._output_dir, f"frame_{cfg.current_frame}_mesh{idx}.obj")
                    write_obj(
                        obj_path,
                        verts,
                        faces,
                    )
                    print(f"Saved mesh {idx} to {obj_path}")

            changed, contour_idx = psim.InputInt(
                "Export Contour Idx", self._export_contour_idx
            )
            if changed:
                self._export_contour_idx = max(0, int(contour_idx))

            changed, ring_count = psim.InputInt(
                "Export Contour Ring Count", self._export_contour_ring_count
            )
            if changed:
                self._export_contour_ring_count = max(0, int(ring_count))

            if psim.Button("Export Contour Submeshes"):
                try:
                    output_paths = self._solver.export_contour_submeshes(
                        self._output_dir,
                        self._export_contour_idx,
                        self._export_contour_ring_count,
                    )
                    for output_path in output_paths:
                        print(f"Exported contour submesh to {output_path}")
                except Exception as exc:
                    print(
                        f"[SimulationGUI] failed to export contour submeshes: {exc}"
                    )

            _, cfg.output_per_frame = psim.Checkbox(
                "Output Each Frame", cfg.output_per_frame
            )
            psim.TreePop()

    # ---- Continuous simulation loop --------------------------------------

    def _continuous_simulation_loop(self):
        if not self._is_simulating:
            return

        cfg = self._config

        if cfg.output_per_frame and cfg.current_frame == 0:
            obj_path = os.path.join(self._output_dir, f"frame_0_init.obj")
            self._solver.save_sim_result(obj_path=obj_path)

        self._physics_step()
        self._update_gui_vertices()

        if cfg.output_per_frame:
            animation_fps = 60.0
            output_freq = max(1, int((1.0 / animation_fps) / cfg.implicit_dt))
            if cfg.current_frame % output_freq == 0:
                obj_path = os.path.join(self._output_dir, f"frame_{cfg.current_frame}.obj")
                self._solver.save_sim_result(obj_path=obj_path)
