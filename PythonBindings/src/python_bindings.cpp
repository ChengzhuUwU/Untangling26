#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <cstdint>
#include <array>
#include <memory>
#include <string>
#include <string_view>

#include <luisa/luisa-compute.h>

#include "SimulationCore/world_data.h"
#include "SimulationSolver/newton_solver.h"
#include "SimulationCore/scene_params.h"
#include "app_simulation_demo_config.h"

namespace py = pybind11;
using namespace lcs;
using namespace lcs::Initializer;
using namespace lcs::Material;

// Helper wrapper to hold WorldData pointer and expose chainable methods
struct WorldDataWrapper
{
	std::shared_ptr<WorldData> wd;
	WorldDataWrapper(std::shared_ptr<WorldData> w)
		: wd(std::move(w))
	{
	}

	WorldDataWrapper& set_name(const std::string_view& name)
	{
		wd->set_name(name);
		return *this;
	}
	WorldDataWrapper& set_simulation_type(lcs::Material::MaterialType t)
	{
		wd->set_material_type(t);
		return *this;
	}
	WorldDataWrapper& set_auto_scaling(bool enabled)
	{
		wd->set_auto_scaling(enabled);
		return *this;
	}

	// Expose cloth material setter by accepting keyword args
	WorldDataWrapper& set_physics_material_cloth(
		const std::string_view& stretch_model = cloth_stretch_model_to_string(ClothMaterial::default_stretch_model()),
		const std::string_view& bending_model = cloth_bending_model_to_string(ClothMaterial::default_bending_model()),
		float					thickness = ClothMaterial::default_thickness(),
		float					youngs_modulus = ClothMaterial::default_youngs_modulus(),
		float					poisson_ratio = ClothMaterial::default_poisson_ratio(),
		float					area_bending_stiffness = ClothMaterial::default_area_bending_stiffness(),
		float					d_hat = ClothMaterial::default_d_hat(),
		float					contact_offset = ClothMaterial::default_contact_offset())
	{
		wd->set_material_type(lcs::Material::MaterialType::Cloth);
		ClothMaterial mat;
		mat.stretch_model = parse_cloth_stretch_model(stretch_model);
		mat.bending_model = parse_cloth_bending_model(bending_model);
		mat.thickness = thickness;
		mat.youngs_modulus = youngs_modulus;
		mat.poisson_ratio = poisson_ratio;
		mat.area_bending_stiffness = area_bending_stiffness;
		mat.d_hat = d_hat;
		mat.contact_offset = contact_offset;
		wd->set_physics_material(mat);
		return *this;
	}

	// Expose tetrahedral material setter
	WorldDataWrapper& set_physics_material_tet(
		const std::string_view& model = tet_model_to_string(TetMaterial::default_model()),
		float					youngs_modulus = TetMaterial::default_youngs_modulus(),
		float					poisson_ratio = TetMaterial::default_poisson_ratio(),
		float					density = TetMaterial::default_density(),
		float					mass = TetMaterial::default_mass(),
		float					d_hat = ClothMaterial::default_d_hat(),
		float					contact_offset = ClothMaterial::default_contact_offset())
	{
		wd->set_material_type(lcs::Material::MaterialType::Tetrahedral);
		TetMaterial mat;
		mat.model = parse_tet_model(model);
		mat.youngs_modulus = youngs_modulus;
		mat.poisson_ratio = poisson_ratio;
		mat.density = density;
		mat.mass = mass;
		mat.is_shell = false;
		mat.d_hat = d_hat;
		mat.contact_offset = contact_offset;
		wd->set_physics_material(mat);
		return *this;
	}

	// Expose rigid material setter
	WorldDataWrapper& set_physics_material_rigid(
		const std::string_view& model = rigid_model_to_string(RigidMaterial::default_model()),
		float					thickness = RigidMaterial::default_thickness(),
		float					stiffness = RigidMaterial::default_stiffness(),
		float					density = RigidMaterial::default_density(),
		float					mass = RigidMaterial::default_mass(),
		float					d_hat = ClothMaterial::default_d_hat(),
		float					contact_offset = ClothMaterial::default_contact_offset())
	{
		wd->set_material_type(lcs::Material::MaterialType::Rigid);
		RigidMaterial mat;
		mat.model = parse_rigid_model(model);
		mat.thickness = thickness;
		mat.stiffness = stiffness;
		mat.density = density;
		mat.mass = mass;
		mat.d_hat = d_hat;
		mat.contact_offset = contact_offset;
		wd->set_physics_material(mat);
		return *this;
	}

	// Expose rod material setter
	WorldDataWrapper& set_physics_material_rod(
		const std::string_view& model = rod_model_to_string(RodMaterial::default_model()),
		float					radius = RodMaterial::default_radius(),
		float					bending_stiffness = RodMaterial::default_bending_stiffness(),
		float					twisting_stiffness = RodMaterial::default_twisting_stiffness(),
		float					density = RodMaterial::default_density(),
		float					mass = RodMaterial::default_mass(),
		float					d_hat = ClothMaterial::default_d_hat(),
		float					contact_offset = ClothMaterial::default_contact_offset())
	{
		wd->set_material_type(lcs::Material::MaterialType::Rod);
		RodMaterial mat;
		mat.model = parse_rod_model(model);
		mat.radius = radius;
		mat.bending_stiffness = bending_stiffness;
		mat.twisting_stiffness = twisting_stiffness;
		mat.density = density;
		mat.mass = mass;
		wd->set_physics_material(mat);
		return *this;
	}

	// Convenience: add fixed-point rule by name and optional numeric range/list
	WorldDataWrapper& add_fixed_point_by_method(const std::string_view& method, float stiffness, float range)
	{
		MakeFixedPointsInterface mfp;
		mfp.method = parse_fixed_method_py(method);
		mfp.range = range;

		wd->add_fixed_point_info(mfp, stiffness);
		return *this;
	}

	// Convenience: add explicit vertex indices as fixed points
	WorldDataWrapper& add_fixed_point_by_indices(py::array_t<int, py::array::c_style | py::array::forcecast> indices, float stiffness)
	{
		if (indices.ndim() != 1)
			throw std::runtime_error("indices must be a 1-D array of ints");
		auto		 buf = indices.unchecked<1>();
		const size_t n = indices.shape(0);
		for (size_t i = 0; i < n; ++i)
		{
			int v = buf(i);
			if (v >= 0)
			{
				wd->fixed_point_indices.push_back(static_cast<uint>(v));
				wd->fixed_point_stiffness.push_back(stiffness);
			}
		}
		return *this;
	}

	WorldDataWrapper& set_translation(float x, float y, float z)
	{
		wd->set_translation(x, y, z);
		return *this;
	}

	WorldDataWrapper& set_rotation(float x, float y, float z)
	{
		wd->set_rotation(x, y, z);
		return *this;
	}

	WorldDataWrapper& set_scale(float s)
	{
		wd->set_scale(s);
		return *this;
	}
	WorldDataWrapper& set_face_orientation(bool along_triangle_winding)
	{
		wd->set_face_orientation(along_triangle_winding);
		return *this;
	}

	std::array<float, 3> get_rest_translation() const
	{
		auto t = wd->translation;
		return std::array<float, 3>{ t.x, t.y, t.z };
	}

	std::array<float, 3> get_rest_rotation() const
	{
		auto r = wd->rotation;
		return std::array<float, 3>{ r.x, r.y, r.z };
	}

	std::array<float, 3> get_rest_scale() const
	{
		auto s = wd->scale;
		return std::array<float, 3>{ s.x, s.y, s.z };
	}

	std::string get_name() const
	{
		return wd->get_model_name();
	}
	uint get_registration_index() const
	{
		return wd->get_registration_index();
	}

	py::list get_fixed_point_indices() const
	{
		const auto& indices = wd->fixed_point_indices;
		py::list	out;
		for (auto idx : indices)
			out.append(static_cast<uint32_t>(idx));
		return out;
	}

	py::array_t<float> get_rest_positions() const
	{
		const auto		   rest = wd->get_rest_positions();
		py::array_t<float> out({ rest.size(), static_cast<size_t>(3) });
		auto			   buf = out.mutable_unchecked<2>();
		for (size_t i = 0; i < rest.size(); ++i)
		{
			buf(i, 0) = rest[i][0];
			buf(i, 1) = rest[i][1];
			buf(i, 2) = rest[i][2];
		}
		return out;
	}
};

// Read-only wrapper for APIs that return const WorldData&.
struct ConstWorldDataWrapper
{
	const WorldData* wd;
	ConstWorldDataWrapper(const WorldData* w)
		: wd(w)
	{
	}

	std::string get_name() const
	{
		return wd->get_model_name();
	}
	uint get_registration_index() const
	{
		return wd->get_registration_index();
	}

	py::list get_fixed_point_indices() const
	{
		const auto& indices = wd->fixed_point_indices;
		py::list	out;
		for (auto idx : indices)
			out.append(static_cast<uint32_t>(idx));
		return out;
	}

	py::array_t<float> get_rest_positions() const
	{
		const auto		   rest = wd->get_rest_positions();
		py::array_t<float> out({ rest.size(), static_cast<size_t>(3) });
		auto			   buf = out.mutable_unchecked<2>();
		for (size_t i = 0; i < rest.size(); ++i)
		{
			buf(i, 0) = rest[i][0];
			buf(i, 1) = rest[i][1];
			buf(i, 2) = rest[i][2];
		}
		return out;
	}

	std::array<float, 3> get_rest_translation() const
	{
		auto t = wd->translation;
		return std::array<float, 3>{ t.x, t.y, t.z };
	}

	std::array<float, 3> get_rest_rotation() const
	{
		auto r = wd->rotation;
		return std::array<float, 3>{ r.x, r.y, r.z };
	}

	std::array<float, 3> get_rest_scale() const
	{
		auto s = wd->scale;
		return std::array<float, 3>{ s.x, s.y, s.z };
	}
};

// Python-facing Newton-like builder that stores a vector<WorldData>
struct PyNewtonBuilder
{
	std::unique_ptr<lcs::NewtonSolver> solver_ptr;

	PyNewtonBuilder()
		: solver_ptr(std::make_unique<lcs::NewtonSolver>())
	{
	}

	// register_mesh accepts numpy arrays (vertices Nx3, triangles Mx3)
	WorldDataWrapper create_world_data_from_array(const std::string_view& name,
		py::array_t<double, py::array::c_style | py::array::forcecast>	  vertices,
		py::array_t<int, py::array::c_style | py::array::forcecast>		  triangles)
	{
		// Validate shapes
		if (vertices.ndim() != 2 || vertices.shape(1) != 3)
			throw std::runtime_error("vertices must be a (N,3) array of floats");
		if (triangles.ndim() != 2 || triangles.shape(1) != 3)
			throw std::runtime_error("triangles must be a (M,3) array of ints");

		using InputVertexType = std::array<float, 3>;
		using InputFaceType = std::array<uint32_t, 3>;

		const size_t nverts = vertices.shape(0);
		const size_t nfaces = triangles.shape(0);

		std::vector<InputVertexType> input_vertices(nverts);
		std::vector<InputFaceType>	 input_triangles(nfaces);

		auto buf_v = vertices.unchecked<2>();
		auto buf_t = triangles.unchecked<2>();
		for (size_t i = 0; i < nverts; ++i)
		{
			InputVertexType p;
			p[0] = static_cast<float>(buf_v(i, 0));
			p[1] = static_cast<float>(buf_v(i, 1));
			p[2] = static_cast<float>(buf_v(i, 2));
			input_vertices[i] = p;
		}
		for (size_t i = 0; i < nfaces; ++i)
		{
			InputFaceType f;
			f[0] = static_cast<uint32_t>(buf_t(i, 0));
			f[1] = static_cast<uint32_t>(buf_t(i, 1));
			f[2] = static_cast<uint32_t>(buf_t(i, 2));
			input_triangles[i] = f;
		}

		auto world_data = std::make_shared<WorldData>();
		world_data->set_name(name);
		world_data->load_mesh_from_array(input_vertices, input_triangles);
		return WorldDataWrapper(world_data);
	}
	// create world data from an obj file path; call register_world_data() to add into solver
	WorldDataWrapper create_world_data_from_file_path(const std::string_view& name, const std::string_view& obj_file_path)
	{
		auto world_data = std::make_shared<WorldData>();
		world_data->set_name(name);
		world_data->load_mesh_from_path(obj_file_path);
		return WorldDataWrapper(world_data);
	}

	uint register_world_data(const WorldDataWrapper& world_data)
	{
		if (!world_data.wd)
			throw std::runtime_error("Invalid world data handle.");
		return solver_ptr->register_world_data(*world_data.wd);
	}

	// expose method to get number of registered meshes
	size_t num_meshes() const { return solver_ptr->get_sorted_world_data().size(); }

	// expose a method to export registered meshes as python lists (simple)
	py::list get_mesh_names() const
	{
		py::list				 out;
		const auto&				 world_data = solver_ptr->get_sorted_world_data();
		std::vector<std::string> names(world_data.size());
		// Legacy world-data registration debug logging retained for reference.
		for (uint registration_id = 0; registration_id < world_data.size(); ++registration_id)
		{
			const uint	sorted_idx = solver_ptr->query_sorted_index_from_registration_id(registration_id);
			const auto& w = world_data[sorted_idx];
			names[registration_id] = w.get_model_name();
		}
		for (const auto& name : names)
			out.append(name);
		return out;
	}

	// Debug helper to print registered meshes info in C++ logs
	void print_registered_meshes_info() const
	{
		auto& world_data = solver_ptr->get_sorted_world_data();
		for (auto& w : world_data)
		{
			auto& mesh = w.get_mesh();
			LUISA_INFO("Mesh '{}': registration_id={}, num_verts={}, num_faces={}",
				w.get_model_name(), w.get_registration_index(), mesh.model_positions.size(), mesh.faces.size());
		}
	}

	// Initialize underlying NewtonSolver using the device previously set via init_device()/set_device().
	void init_solver()
	{
		solver_ptr->init_solver();
		LUISA_INFO("Solver initialized.");
	}

	// Load a full scene from JSON, including world_data and scene params. This should be called before init_solver().
	void load_scene_from_json(const std::string_view& json_path)
	{
		Demo::Simulation::load_scene_params_from_json(
			[&](const lcs::Initializer::WorldData& wd)
			{
				solver_ptr->register_world_data(wd);
			},
			std::string(json_path));
	}

	void physics_step_cpu()
	{
		if (!solver_ptr)
			throw std::runtime_error("Solver not initialized. Call init_solver() first.");
		solver_ptr->physics_step_CPU();
	}

	void physics_step_gpu()
	{
		if (!solver_ptr)
			throw std::runtime_error("Solver not initialized. Call init_solver() first.");
		solver_ptr->physics_step_GPU();
	}

	void restart_system()
	{
		if (!solver_ptr)
			throw std::runtime_error("Solver not initialized. Call init_solver() first.");
		solver_ptr->lcs::SolverInterface::restart_system();
	}

	// Update a pinned vertex position on the solver (mesh local vertex id, target position)
	void update_per_vertex_animation(const unsigned int				  mesh_idx,
		const unsigned int											  local_vid,
		py::array_t<float, py::array::c_style | py::array::forcecast> target_pos)
	{
		if (!solver_ptr)
			throw std::runtime_error("Solver not initialized. Call init_solver() first.");

		if (target_pos.ndim() != 1 || target_pos.shape(0) != 3)
			throw std::runtime_error("target_pos must be a 1-D array of length 3 (x,y,z)");

		auto				 buf = target_pos.unchecked<1>();
		std::array<float, 3> tp{ buf(0), buf(1), buf(2) };
		solver_ptr->update_per_vertex_animation(mesh_idx, local_vid, tp);
	}

	// Update a pinned body state on the solver (body id, target translation and rotation)
	void update_per_body_animation(const unsigned int				  mesh_idx,
		py::array_t<float, py::array::c_style | py::array::forcecast> target_translation,
		py::array_t<float, py::array::c_style | py::array::forcecast> target_rotation)
	{
		if (!solver_ptr)
			throw std::runtime_error("Solver not initialized. Call init_solver() first.");
		if (target_translation.ndim() != 1 || target_translation.shape(0) != 3)
			throw std::runtime_error("target_translation must be a 1-D array of length 3 (x,y,z)");
		if (target_rotation.ndim() != 1 || target_rotation.shape(0) != 3)
			throw std::runtime_error("target_rotation must be a 1-D array of length 3 (x,y,z)");

		auto				 buf_t = target_translation.unchecked<1>();
		std::array<float, 3> tt{ buf_t(0), buf_t(1), buf_t(2) };
		auto				 buf_r = target_rotation.unchecked<1>();
		std::array<float, 3> tr{ buf_r(0), buf_r(1), buf_r(2) };
		solver_ptr->update_per_body_animation(mesh_idx, tt, tr);
	}

	// Return simulation results as a tuple of (vertices_list, faces_list) of numpy arrays. Uses memcpy for efficient data transfer.
	py::tuple get_sim_result()
	{
		if (!solver_ptr)
			throw std::runtime_error("Solver not initialized. Call init_solver() first.");

		const uint num_meshes = solver_ptr->get_host_mesh_data().num_meshes;

		std::vector<std::vector<std::array<float, 3>>> sa_rendering_vertices(num_meshes);
		std::vector<std::vector<std::array<uint, 3>>>  sa_rendering_triangles(num_meshes);
		solver_ptr->get_curr_vertices_to_host(sa_rendering_vertices);
		solver_ptr->get_triangles_to_host(sa_rendering_triangles);

		py::list py_verts;
		py::list py_faces;
		for (uint i = 0; i < num_meshes; ++i)
		{
			// vertices – contiguous std::array<float,3>, safe to memcpy
			const auto&		   mesh_verts = sa_rendering_vertices[i];
			py::array_t<float> v_arr({ (size_t)mesh_verts.size(), (size_t)3 });
			std::memcpy(v_arr.mutable_data(), mesh_verts.data(), mesh_verts.size() * 3 * sizeof(float));
			py_verts.append(v_arr);

			// faces
			const auto&			  mesh_faces = sa_rendering_triangles[i];
			py::array_t<uint32_t> f_arr({ (size_t)mesh_faces.size(), (size_t)3 });
			std::memcpy(f_arr.mutable_data(), mesh_faces.data(), mesh_faces.size() * 3 * sizeof(uint32_t));
			py_faces.append(f_arr);
		}
		return py::make_tuple(py_verts, py_faces);
	}

	// Return PRP material-space positions in global MeshData and registration orderings.
	py::dict get_material_position_data() const
	{
		if (!solver_ptr)
			throw std::runtime_error("Solver not initialized. Call init_solver() first.");

		const auto& mesh_data = solver_ptr->get_host_mesh_data();
		const auto& material_positions = mesh_data.sa_material_x;
		if (material_positions.size() != mesh_data.num_verts)
		{
			throw std::runtime_error("sa_material_x size does not match the global vertex count.");
		}
		if (mesh_data.prefix_num_verts.size() != static_cast<size_t>(mesh_data.num_meshes) + 1u
			|| mesh_data.sorted_to_input_mesh_id.size() != mesh_data.num_meshes)
		{
			throw std::runtime_error("Mesh prefix/order data is inconsistent while exporting sa_material_x.");
		}

		py::array_t<float> global_arr({ static_cast<size_t>(mesh_data.num_verts), static_cast<size_t>(2) });
		auto			   global_view = global_arr.mutable_unchecked<2>();
		for (uint global_vid = 0u; global_vid < mesh_data.num_verts; ++global_vid)
		{
			const auto uv = material_positions[global_vid];
			global_view(global_vid, 0) = uv.x;
			global_view(global_vid, 1) = uv.y;
		}

		py::list mesh_positions(mesh_data.num_meshes);
		for (uint sorted_idx = 0u; sorted_idx < mesh_data.num_meshes; ++sorted_idx)
		{
			const uint registration_idx = mesh_data.sorted_to_input_mesh_id[sorted_idx];
			if (registration_idx >= mesh_data.num_meshes)
			{
				throw std::runtime_error("Invalid sorted-to-registration mesh index while exporting sa_material_x.");
			}
			const uint prefix = mesh_data.prefix_num_verts[sorted_idx];
			const uint suffix = mesh_data.prefix_num_verts[sorted_idx + 1u];
			if (prefix > suffix || suffix > mesh_data.num_verts)
			{
				throw std::runtime_error("Invalid vertex prefix range while exporting sa_material_x.");
			}

			const size_t	   num_verts = static_cast<size_t>(suffix - prefix);
			py::array_t<float> mesh_arr({ num_verts, static_cast<size_t>(2) });
			auto			   mesh_view = mesh_arr.mutable_unchecked<2>();
			for (size_t local_vid = 0u; local_vid < num_verts; ++local_vid)
			{
				const auto uv = material_positions[prefix + static_cast<uint>(local_vid)];
				mesh_view(local_vid, 0) = uv.x;
				mesh_view(local_vid, 1) = uv.y;
			}
			mesh_positions[registration_idx] = std::move(mesh_arr);
		}

		py::dict out;
		out["global_positions"] = std::move(global_arr);
		out["mesh_positions"] = std::move(mesh_positions);
		return out;
	}

	py::list get_surface_edges()
	{
		if (!solver_ptr)
			throw std::runtime_error("Solver not initialized. Call init_solver() first.");

		const uint num_meshes = solver_ptr->get_host_mesh_data().num_meshes;

		std::vector<std::vector<std::array<uint, 2>>> sa_surface_edges(num_meshes);
		solver_ptr->get_edges_to_host(sa_surface_edges);

		py::list py_edges;
		for (uint i = 0; i < num_meshes; ++i)
		{
			const auto&			  mesh_edges = sa_surface_edges[i];
			py::array_t<uint32_t> e_arr({ (size_t)mesh_edges.size(), (size_t)2 });
			if (!mesh_edges.empty())
				std::memcpy(e_arr.mutable_data(), mesh_edges.data(), mesh_edges.size() * 2 * sizeof(uint32_t));
			py_edges.append(e_arr);
		}

		return py_edges;
	}

	// Return visualize curves as a tuple of numpy arrays: (curve_vertices[N,3], curve_edges[M,2], curve_edge_colors[M,3]).
	py::tuple get_visualize_curves()
	{
		if (!solver_ptr)
			throw std::runtime_error("Solver not initialized. Call init_solver() first.");

		std::vector<std::array<float, 3>> curve_vertices;
		std::vector<std::array<uint, 2>>  curve_edges;
		std::vector<std::array<float, 3>> curve_edge_colors;
		solver_ptr->get_visualize_curves(curve_vertices, curve_edges, curve_edge_colors);

		py::array_t<float> curve_v_arr({ (size_t)curve_vertices.size(), (size_t)3 });
		if (!curve_vertices.empty())
			std::memcpy(curve_v_arr.mutable_data(), curve_vertices.data(), curve_vertices.size() * 3 * sizeof(float));

		py::array_t<uint32_t> curve_e_arr({ (size_t)curve_edges.size(), (size_t)2 });
		if (!curve_edges.empty())
			std::memcpy(curve_e_arr.mutable_data(), curve_edges.data(), curve_edges.size() * 2 * sizeof(uint32_t));

		py::array_t<float> curve_c_arr({ (size_t)curve_edge_colors.size(), (size_t)3 });
		if (!curve_edge_colors.empty())
			std::memcpy(curve_c_arr.mutable_data(), curve_edge_colors.data(), curve_edge_colors.size() * 3 * sizeof(float));

		return py::make_tuple(curve_v_arr, curve_e_arr, curve_c_arr);
	}

	// Return EF-pair contour data, adjacency CSR arrays, contour types, and pair/contour counts.

	// Return total collision-buffer bytes for memory-budget guards.
	size_t get_collision_buffer_bytes() const
	{
		if (!solver_ptr)
			throw std::runtime_error("Solver not initialized. Call init_solver() first.");
		return solver_ptr->get_host_collision_data().get_momery_bytes();
	}

	py::dict get_intersection_contour_data() const
	{
		if (!solver_ptr)
			throw std::runtime_error("Solver not initialized. Call init_solver() first.");

		const auto& host_collision_data = solver_ptr->get_host_collision_data();
		const auto& host_untangling_data = solver_ptr->get_host_untangling_data();

		uint num_pairs = 0u;
		if (host_collision_data.narrow_phase_collision_count.size() > 1u)
			num_pairs = host_collision_data.narrow_phase_collision_count[1u];

		py::array_t<uint32_t> ef_pairs_arr({ (size_t)num_pairs, (size_t)5 });
		py::array_t<uint32_t> contour_index_arr(py::array::ShapeContainer{ static_cast<py::ssize_t>(num_pairs) });
		py::array_t<uint32_t> mesh_index_arr(py::array::ShapeContainer{ static_cast<py::ssize_t>(num_pairs) });
		py::array_t<float>	  ef_pos_2d_arr({ (size_t)num_pairs, (size_t)4 });
		py::array_t<float>	  ef_pos_3d_arr({ (size_t)num_pairs, (size_t)3 });

		auto ef_pairs_view = ef_pairs_arr.mutable_unchecked<2>();
		auto contour_index_view = contour_index_arr.mutable_unchecked<1>();
		auto mesh_index_view = mesh_index_arr.mutable_unchecked<1>();
		auto ef_pos_3d_view = ef_pos_3d_arr.mutable_unchecked<2>();
		auto ef_pos_2d_view = ef_pos_2d_arr.mutable_unchecked<2>();

		for (uint pair_idx = 0u; pair_idx < num_pairs; ++pair_idx)
		{
			ef_pairs_view(pair_idx, 0) = static_cast<uint32_t>(-1);
			ef_pairs_view(pair_idx, 1) = static_cast<uint32_t>(-1);
			ef_pairs_view(pair_idx, 2) = static_cast<uint32_t>(-1);
			ef_pairs_view(pair_idx, 3) = static_cast<uint32_t>(-1);
			ef_pairs_view(pair_idx, 4) = static_cast<uint32_t>(-1);
			contour_index_view(pair_idx) = static_cast<uint32_t>(-1);
			mesh_index_view(pair_idx) = static_cast<uint32_t>(-1);
			ef_pos_3d_view(pair_idx, 0) = 0.0f;
			ef_pos_3d_view(pair_idx, 1) = 0.0f;
			ef_pos_3d_view(pair_idx, 2) = 0.0f;
			ef_pos_2d_view(pair_idx, 0) = 0.0f;
			ef_pos_2d_view(pair_idx, 1) = 0.0f;
			ef_pos_2d_view(pair_idx, 2) = 0.0f;
			ef_pos_2d_view(pair_idx, 3) = 0.0f;

			if ((size_t)pair_idx < host_collision_data.narrow_phase_list_ef.size())
			{
				const auto ids = host_collision_data.narrow_phase_list_ef[pair_idx].get_indices();
				ef_pairs_view(pair_idx, 0) = ids[0];
				ef_pairs_view(pair_idx, 1) = ids[1];
				ef_pairs_view(pair_idx, 2) = ids[2];
				ef_pairs_view(pair_idx, 3) = ids[3];
				ef_pairs_view(pair_idx, 4) = ids[4];
			}
			if ((size_t)pair_idx < host_untangling_data.ef_pair_contour_index.size())
			{
				contour_index_view(pair_idx) = host_untangling_data.ef_pair_contour_index[pair_idx];
			}
			if ((size_t)pair_idx < host_untangling_data.ef_pair_mesh_index.size())
			{
				mesh_index_view(pair_idx) = host_untangling_data.ef_pair_mesh_index[pair_idx];
			}
			// Note: Using the position in iteration start
			if ((size_t)pair_idx < host_untangling_data.ef_pair_pos_2D.size())
			{
				const auto p = host_untangling_data.ef_pair_pos_2D[pair_idx];
				ef_pos_2d_view(pair_idx, 0) = p.x;
				ef_pos_2d_view(pair_idx, 1) = p.y;
				ef_pos_2d_view(pair_idx, 2) = p.z;
				ef_pos_2d_view(pair_idx, 3) = p.w;
			}
			if ((size_t)pair_idx < host_untangling_data.ef_pair_pos_3D.size())
			{
				const auto p = host_untangling_data.ef_pair_pos_3D[pair_idx];
				ef_pos_3d_view(pair_idx, 0) = p.x;
				ef_pos_3d_view(pair_idx, 1) = p.y;
				ef_pos_3d_view(pair_idx, 2) = p.z;
			}
		}

		std::vector<uint32_t> ef_adj_prefix((size_t)num_pairs + 1u, 0u);
		for (uint pair_idx = 0u; pair_idx < num_pairs; ++pair_idx)
		{
			uint32_t degree = 0u;
			if ((size_t)pair_idx < host_untangling_data.ef_pair_adj_pairs_ext.size())
				degree = static_cast<uint32_t>(host_untangling_data.ef_pair_adj_pairs_ext[pair_idx].size());
			ef_adj_prefix[pair_idx + 1u] = ef_adj_prefix[pair_idx] + degree;
		}
		std::vector<uint32_t> ef_adj_data;
		ef_adj_data.reserve((size_t)ef_adj_prefix.back());
		for (uint pair_idx = 0u; pair_idx < num_pairs; ++pair_idx)
		{
			if ((size_t)pair_idx >= host_untangling_data.ef_pair_adj_pairs_ext.size())
				continue;
			for (const auto adj_pair_idx : host_untangling_data.ef_pair_adj_pairs_ext[pair_idx])
			{
				ef_adj_data.push_back(adj_pair_idx);
			}
		}

		py::array_t<uint32_t> ef_adj_prefix_arr(py::array::ShapeContainer{ static_cast<py::ssize_t>(ef_adj_prefix.size()) });
		if (!ef_adj_prefix.empty())
		{
			std::memcpy(ef_adj_prefix_arr.mutable_data(), ef_adj_prefix.data(), ef_adj_prefix.size() * sizeof(uint32_t));
		}
		py::array_t<uint32_t> ef_adj_data_arr(py::array::ShapeContainer{ static_cast<py::ssize_t>(ef_adj_data.size()) });
		if (!ef_adj_data.empty())
		{
			std::memcpy(ef_adj_data_arr.mutable_data(), ef_adj_data.data(), ef_adj_data.size() * sizeof(uint32_t));
		}

		uint num_contours = static_cast<uint>(host_untangling_data.intersection_contours.size());

		std::vector<uint32_t> contour_prefix((size_t)num_contours + 1u, 0u);
		for (uint contour_idx = 0u; contour_idx < num_contours; ++contour_idx)
		{
			if ((size_t)contour_idx >= host_untangling_data.intersection_contours.size())
				break;
			const auto& contour = host_untangling_data.intersection_contours[contour_idx];
			contour_prefix[contour_idx + 1u] = contour_prefix[contour_idx] + static_cast<uint32_t>(contour.size());
		}
		std::vector<uint32_t> contour_data;
		contour_data.reserve((size_t)contour_prefix.back());
		for (uint contour_idx = 0u; contour_idx < num_contours; ++contour_idx)
		{
			if ((size_t)contour_idx >= host_untangling_data.intersection_contours.size())
				break;
			const auto& contour = host_untangling_data.intersection_contours[contour_idx];
			for (const auto pair_idx : contour)
			{
				contour_data.push_back(pair_idx);
			}
		}

		py::array_t<uint32_t> contour_prefix_arr(py::array::ShapeContainer{ static_cast<py::ssize_t>(contour_prefix.size()) });
		if (!contour_prefix.empty())
		{
			std::memcpy(contour_prefix_arr.mutable_data(), contour_prefix.data(), contour_prefix.size() * sizeof(uint32_t));
		}

		py::array_t<uint32_t> contour_data_arr(py::array::ShapeContainer{ static_cast<py::ssize_t>(contour_data.size()) });
		if (!contour_data.empty())
		{
			std::memcpy(contour_data_arr.mutable_data(), contour_data.data(), contour_data.size() * sizeof(uint32_t));
		}

		py::array_t<uint32_t> num_pairs_arr(py::array::ShapeContainer{ static_cast<py::ssize_t>(1) });
		num_pairs_arr.mutable_unchecked<1>()(0) = num_pairs;
		py::array_t<uint32_t> num_contours_arr(py::array::ShapeContainer{ static_cast<py::ssize_t>(1) });
		num_contours_arr.mutable_unchecked<1>()(0) = num_contours;
		py::array_t<uint32_t> contour_types_arr(py::array::ShapeContainer{ static_cast<py::ssize_t>(num_contours) });
		auto				  contour_types_view = contour_types_arr.mutable_unchecked<1>();
		for (uint contour_idx = 0u; contour_idx < num_contours; ++contour_idx)
		{
			const auto contour_type = contour_idx < host_untangling_data.intersection_contours_info.size()
				? host_untangling_data.intersection_contours_info[contour_idx].contour_type
				: ContourType::Undifined;
			contour_types_view(contour_idx) = static_cast<uint32_t>(contour_type);
		}
		py::dict out;
		out["narrow_phase_list_ef"] = ef_pairs_arr;
		out["ef_pair_contour_index"] = contour_index_arr;
		out["ef_pair_mesh_index"] = mesh_index_arr;
		out["ef_pair_pos_3d"] = ef_pos_3d_arr;
		out["ef_pair_pos_2d"] = ef_pos_2d_arr;
		out["ef_pair_adj_pairs_ext_prefix"] = ef_adj_prefix_arr;
		out["ef_pair_adj_pairs_ext_data"] = ef_adj_data_arr;
		out["intersection_contours_prefix"] = contour_prefix_arr;
		out["intersection_contours_data"] = contour_data_arr;
		out["contour_types"] = contour_types_arr;
		// Preserve the historical debug NPZ schema; disabled projected-winding fields remain sentinels.
		py::array_t<uint32_t> disabled_uv_index(py::array::ShapeContainer{ static_cast<py::ssize_t>(1) });
		disabled_uv_index.mutable_unchecked<1>()(0) = std::numeric_limits<uint32_t>::max();
		out["prp_debug_uv_contour_idx"] = disabled_uv_index;
		out["prp_debug_uv_combo_idx"] = disabled_uv_index;
		for (uint side = 0u; side < 2u; ++side)
		{
			const std::string side_key = std::to_string(side);
			out[py::str("prp_debug_uv_side" + side_key + "_vertex_ids")] = py::array_t<uint32_t>(py::array::ShapeContainer{ 0 });
			out[py::str("prp_debug_uv_side" + side_key + "_vertex_uv")] = py::array_t<float>({ 0, 2 });
			out[py::str("prp_debug_uv_side" + side_key + "_vertex_has_uv")] = py::array_t<luisa::ubyte>(py::array::ShapeContainer{ 0 });
			out[py::str("prp_debug_uv_side" + side_key + "_vertex_is_boundary")] = py::array_t<luisa::ubyte>(py::array::ShapeContainer{ 0 });
			out[py::str("prp_debug_uv_side" + side_key + "_faces")] = py::array_t<uint32_t>({ 0, 3 });
			out[py::str("prp_debug_uv_side" + side_key + "_segments")] = py::array_t<float>({ 0, 4 });
		}
		out["prp_debug_uv_hit_src_uv"] = py::array_t<float>({ 0, 2 });
		out["prp_debug_uv_hit_dst_uv"] = py::array_t<float>({ 0, 2 });
		out["prp_debug_uv_hit_src_has_uv"] = py::array_t<luisa::ubyte>(py::array::ShapeContainer{ 0 });
		out["prp_debug_uv_hit_dst_has_uv"] = py::array_t<luisa::ubyte>(py::array::ShapeContainer{ 0 });
		out["prp_debug_uv_hit_src_key"] = py::array_t<uint32_t>(py::array::ShapeContainer{ 0 });
		out["prp_debug_uv_hit_dst_key"] = py::array_t<uint32_t>(py::array::ShapeContainer{ 0 });
		out["prp_debug_uv_hit_winding_key"] = py::array_t<uint32_t>(py::array::ShapeContainer{ 0 });
		out["prp_debug_uv_hit_is_valid"] = py::array_t<luisa::ubyte>(py::array::ShapeContainer{ 0 });
		out["num_pairs"] = num_pairs_arr;
		out["num_contours"] = num_contours_arr;
		return out;
	}

	// Return target-point response pair arrays and template/pair counts.
	py::dict get_response_pair_data() const
	{
		if (!solver_ptr)
			throw std::runtime_error("Solver not initialized. Call init_solver() first.");

		const auto& host_untangling_data = solver_ptr->get_host_untangling_data();
		const auto& template_pairs = host_untangling_data.target_point_template_pairs;
		const auto& template_pair_indices = host_untangling_data.target_point_template_pairs_indices;
		const auto& sa_x = solver_ptr->get_host_sim_data().sa_x;
		const auto& sa_x_step_start = solver_ptr->get_host_sim_data().sa_x_step_start;
		const auto& sa_x_iter_start = solver_ptr->get_host_sim_data().sa_x_iter_start;

		const size_t num_pairs = template_pairs.size();
		const size_t num_pair_indices = template_pair_indices.size();

		py::array_t<uint32_t> pair_indices_arr({ num_pairs, (size_t)4 });
		py::array_t<float>	  pair_weights_arr({ num_pairs, (size_t)4 });
		py::array_t<uint32_t> pair_type_arr(py::array::ShapeContainer{ static_cast<py::ssize_t>(num_pairs) });
		py::array_t<float>	  pair_normal_arr({ num_pairs, (size_t)3 });
		py::array_t<float>	  pair_stiff_arr({ num_pairs, (size_t)2 });
		py::array_t<float>	  pair_area_arr(py::array::ShapeContainer{ static_cast<py::ssize_t>(num_pairs) });
		py::array_t<float>	  pair_src_arr({ num_pairs, (size_t)3 });
		py::array_t<float>	  pair_dst_arr({ num_pairs, (size_t)3 });

		auto pair_indices_view = pair_indices_arr.mutable_unchecked<2>();
		auto pair_weights_view = pair_weights_arr.mutable_unchecked<2>();
		auto pair_type_view = pair_type_arr.mutable_unchecked<1>();
		auto pair_normal_view = pair_normal_arr.mutable_unchecked<2>();
		auto pair_stiff_view = pair_stiff_arr.mutable_unchecked<2>();
		auto pair_area_view = pair_area_arr.mutable_unchecked<1>();
		auto pair_src_view = pair_src_arr.mutable_unchecked<2>();
		auto pair_dst_view = pair_dst_arr.mutable_unchecked<2>();

		for (size_t i = 0u; i < num_pairs; ++i)
		{
			const auto& pair = template_pairs[i];
			const auto	ids = pair.get_indices();
			const auto	weights = luisa::abs(pair.get_weight());
			const auto	normal = pair.get_normal();
			const auto	stiff = pair.get_stiff();
			const auto	type = pair.get_collision_type();

			pair_indices_view(i, 0) = ids.x;
			pair_indices_view(i, 1) = ids.y;
			pair_indices_view(i, 2) = ids.z;
			pair_indices_view(i, 3) = ids.w;

			pair_weights_view(i, 0) = weights.x;
			pair_weights_view(i, 1) = weights.y;
			pair_weights_view(i, 2) = weights.z;
			pair_weights_view(i, 3) = weights.w;

			pair_normal_view(i, 0) = normal.x;
			pair_normal_view(i, 1) = normal.y;
			pair_normal_view(i, 2) = normal.z;

			pair_stiff_view(i, 0) = stiff.x;
			pair_stiff_view(i, 1) = stiff.y;

			pair_type_view(i) = pair.get_collision_type();
			pair_area_view(i) = pair.get_area();

			float3		 p1;
			float3		 p2;
			const float3 positions[4] = {
				sa_x_iter_start[ids.x],
				sa_x_iter_start[ids.y],
				sa_x_iter_start[ids.z],
				sa_x_iter_start[ids.w],
			};
			if (type == CollisionPair::type_vf())
			{
				p1 = positions[0];
				p2 = weights[1] * positions[1] + weights[2] * positions[2] + weights[3] * positions[3];
			}
			else if (type == CollisionPair::type_ee())
			{
				p1 = weights[0] * positions[0] + weights[1] * positions[1];
				p2 = weights[2] * positions[2] + weights[3] * positions[3];
			}
			pair_src_view(i, 0) = p1.x;
			pair_src_view(i, 1) = p1.y;
			pair_src_view(i, 2) = p1.z;
			pair_dst_view(i, 0) = p2.x;
			pair_dst_view(i, 1) = p2.y;
			pair_dst_view(i, 2) = p2.z;
		}

		py::array_t<uint32_t> pair_indices_meta_arr({ num_pair_indices, (size_t)4 });
		auto				  pair_indices_meta_view = pair_indices_meta_arr.mutable_unchecked<2>();
		for (size_t i = 0u; i < num_pair_indices; ++i)
		{
			const auto idx_pair = template_pair_indices[i];
			pair_indices_meta_view(i, 0) = idx_pair.x;
			pair_indices_meta_view(i, 1) = idx_pair.y;
			pair_indices_meta_view(i, 2) = idx_pair.z;
			pair_indices_meta_view(i, 3) = idx_pair.w;
		}

		py::array_t<uint32_t> num_pairs_arr(py::array::ShapeContainer{ static_cast<py::ssize_t>(1) });
		num_pairs_arr.mutable_unchecked<1>()(0) = static_cast<uint32_t>(num_pairs);
		py::array_t<uint32_t> num_pair_indices_arr(py::array::ShapeContainer{ static_cast<py::ssize_t>(1) });
		num_pair_indices_arr.mutable_unchecked<1>()(0) = static_cast<uint32_t>(num_pair_indices);

		py::dict out;
		out["target_point_template_pair_indices"] = std::move(pair_indices_arr);
		out["target_point_template_pair_weights"] = std::move(pair_weights_arr);
		out["target_point_template_pair_collision_type"] = std::move(pair_type_arr);
		out["target_point_template_pair_normal"] = std::move(pair_normal_arr);
		out["target_point_template_pair_stiff"] = std::move(pair_stiff_arr);
		out["target_point_template_pair_area"] = std::move(pair_area_arr);
		out["target_point_template_pair_src"] = std::move(pair_src_arr);
		out["target_point_template_pair_dst"] = std::move(pair_dst_arr);
		out["target_point_template_pairs_indices"] = std::move(pair_indices_meta_arr);
		out["num_template_pairs"] = std::move(num_pairs_arr);
		out["num_template_pairs_indices"] = std::move(num_pair_indices_arr);
		return out;
	}

	uint query_local_vid_from_global_vid(const uint global_vid) const
	{
		if (!solver_ptr)
			throw std::runtime_error("Solver not initialized. Call init_solver() first.");

		return solver_ptr->query_local_vid_from_global_vid(global_vid);
	}

	uint query_registration_id_from_global_vid(const uint global_vid) const
	{
		if (!solver_ptr)
			throw std::runtime_error("Solver not initialized. Call init_solver() first.");

		return solver_ptr->query_registration_id_from_global_vid(global_vid);
	}

	uint query_registration_id_from_sorted_index(const uint sorted_index) const
	{
		if (!solver_ptr)
			throw std::runtime_error("Solver not initialized. Call init_solver() first.");

		return solver_ptr->query_registration_id_from_sorted_index(sorted_index);
	}

	uint query_global_vid_from_registration_id_and_local_vid(const uint registration_id, const uint local_vid) const
	{
		if (!solver_ptr)
			throw std::runtime_error("Solver not initialized. Call init_solver() first.");

		return solver_ptr->query_global_vid_from_registration_id_and_local_vid(registration_id, local_vid);
	}

	std::vector<uint> query_vert_adj_edges(const uint global_vid) const
	{
		if (!solver_ptr)
			throw std::runtime_error("Solver not initialized. Call init_solver() first.");

		return solver_ptr->query_vert_adj_edges(global_vid);
	}

	py::tuple get_object_sim_result_by_registration_id(uint registration_id)
	{
		if (!solver_ptr)
			throw std::runtime_error("Solver not initialized. Call init_solver() first.");

		std::vector<std::array<float, 3>> object_vertices;
		std::vector<std::array<uint, 3>>  object_triangles;
		solver_ptr->get_object_sim_result_by_registration_id(registration_id, object_vertices, object_triangles);

		py::array_t<float> v_arr({ (size_t)object_vertices.size(), (size_t)3 });
		if (!object_vertices.empty())
			std::memcpy(v_arr.mutable_data(), object_vertices.data(), object_vertices.size() * 3 * sizeof(float));

		py::array_t<uint32_t> f_arr({ (size_t)object_triangles.size(), (size_t)3 });
		if (!object_triangles.empty())
			std::memcpy(f_arr.mutable_data(), object_triangles.data(), object_triangles.size() * 3 * sizeof(uint32_t));

		return py::make_tuple(v_arr, f_arr);
	}

	ConstWorldDataWrapper get_object_by_registration_id(uint registration_id) const
	{
		return ConstWorldDataWrapper(&solver_ptr->get_object_by_registration_id(registration_id));
	}

	void set_object_vertex_positions(uint							  registration_id,
		py::array_t<float, py::array::c_style | py::array::forcecast> positions)
	{
		if (!solver_ptr)
			throw std::runtime_error("Solver not initialized. Call init_solver() first.");

		auto buf = positions.request();
		if (buf.ndim != 2 || buf.shape[1] != 3)
			throw std::runtime_error("positions must be (N, 3) float array");

		const size_t					  num_verts = static_cast<size_t>(buf.shape[0]);
		std::vector<std::array<float, 3>> input_positions(num_verts);
		const float*					  ptr = static_cast<const float*>(buf.ptr);
		for (size_t i = 0; i < num_verts; ++i)
		{
			input_positions[i] = { ptr[i * 3 + 0], ptr[i * 3 + 1], ptr[i * 3 + 2] };
		}

		solver_ptr->set_object_sim_result_by_registration_id(registration_id, input_positions);
	}

	void update_object_vertex_animation_targets(uint				  registration_id,
		py::array_t<float, py::array::c_style | py::array::forcecast> positions)
	{
		if (!solver_ptr)
			throw std::runtime_error("Solver not initialized. Call init_solver() first.");

		auto buf = positions.request();
		if (buf.ndim != 2 || buf.shape[1] != 3)
			throw std::runtime_error("positions must be (N, 3) float array");

		const auto&	 world_object = solver_ptr->get_object_by_registration_id(registration_id);
		const size_t num_verts = static_cast<size_t>(buf.shape[0]);
		const size_t expected_num_verts = world_object.get_mesh().model_positions.size();
		if (num_verts != expected_num_verts)
		{
			throw std::runtime_error(
				"positions vertex count does not match registered object vertex count");
		}

		const float* ptr = static_cast<const float*>(buf.ptr);
		for (const uint local_vid : world_object.fixed_point_indices)
		{
			const size_t		 base = static_cast<size_t>(local_vid) * 3u;
			std::array<float, 3> target_position{ ptr[base + 0], ptr[base + 1], ptr[base + 2] };
			solver_ptr->update_per_vertex_animation(registration_id, local_vid, target_position);
		}
	}

	void set_object_vertex_positions_with_animation_targets(uint	  registration_id,
		py::array_t<float, py::array::c_style | py::array::forcecast> positions)
	{
		set_object_vertex_positions(registration_id, positions);
		update_object_vertex_animation_targets(registration_id, positions);
	}

	void save_sim_result(const std::string_view& full_path)
	{
		if (!solver_ptr)
			throw std::runtime_error("Solver not initialized. Call init_solver() first.");

		solver_ptr->save_mesh_to_obj(full_path);
	}

	void save_current_state(const std::string_view& full_path)
	{
		if (!solver_ptr)
			throw std::runtime_error("Solver not initialized. Call init_solver() first.");

		solver_ptr->save_current_frame_state_file(full_path);
	}

	std::vector<std::string> export_contour_submeshes(const std::string_view& output_dir,
		const uint															  contour_idx,
		const uint															  ring_count)
	{
		if (!solver_ptr)
			throw std::runtime_error("Solver not initialized. Call init_solver() first.");

		return solver_ptr->export_contour_submeshes(output_dir, contour_idx, ring_count);
	}

	void load_target_state(const std::string_view& full_path)
	{
		if (!solver_ptr)
			throw std::runtime_error("Solver not initialized. Call init_solver() first.");

		solver_ptr->load_state_from_state_file(full_path);
	}

	void load_target_state_from_obj(const std::string_view& full_path)
	{
		if (!solver_ptr)
			throw std::runtime_error("Solver not initialized. Call init_solver() first.");

		solver_ptr->load_state_from_obj_file(full_path);
	}

	// Device management (mirrors lcs::SolverInterface methods).

	// Create the Luisa device/stream; backend and binary path are optional.
	void init_device(py::object backend_name_obj = py::none(), py::object binary_path_obj = py::none())
	{
		// Resolve binary path
		std::string binary_path;
		if (!binary_path_obj.is_none())
		{
			binary_path = binary_path_obj.cast<std::string>();
		}
		else
		{
			try
			{
				py::module_ self = py::module_::import("lcs_py");
				if (py::hasattr(self, "__file__"))
					binary_path = self.attr("__file__").cast<std::string>();
			}
			catch (...)
			{
			}
		}

		// Resolve backend name
		std::string backend;
		if (!backend_name_obj.is_none())
			backend = backend_name_obj.cast<std::string>();

		solver_ptr->create_device(binary_path, backend);
	}

	// Borrow an external device/stream (non-owning). The caller must ensure they outlive this solver.
	void set_device(uintptr_t device_ptr, uintptr_t stream_ptr)
	{
		solver_ptr->set_device_from_pointers(device_ptr, stream_ptr);
	}

	// Release owned device resources.
	void cleanup_device()
	{
		solver_ptr->cleanup_device();
	}

	// Return raw pointer (as int) to the active luisa::compute::Device.
	uintptr_t get_device_ptr() const
	{
		return solver_ptr->get_device_ptr();
	}

	// Return raw pointer (as int) to the active luisa::compute::Stream.
	uintptr_t get_stream_ptr() const
	{
		return solver_ptr->get_stream_ptr();
	}

	lcs::SceneParams& get_config() const
	{
		if (!solver_ptr)
			throw std::runtime_error("Solver not initialized.");
		return solver_ptr->get_config();
	}

	struct PyPRPDebugInfo
	{
		py::dict bool_stats;
		py::dict uint_stats;
		py::dict float_stats;
	};

	PyPRPDebugInfo get_prp_debug_info() const
	{
		if (!solver_ptr)
			throw std::runtime_error("Solver not initialized.");

		const auto&	   prp_debug_info = solver_ptr->get_config().prp_debug_info;
		PyPRPDebugInfo out;
		{
			for (const auto& [key, value] : prp_debug_info.bool_stats)
			{
				out.bool_stats[key.c_str()] = value;
			}
			for (const auto& [key, value] : prp_debug_info.uint_stats)
			{
				out.uint_stats[key.c_str()] = value;
			}
			for (const auto& [key, value] : prp_debug_info.float_stats)
			{
				out.float_stats[key.c_str()] = value;
			}
		}
		return out;
	}
};

PYBIND11_MODULE(lcs_py, m)
{
	py::enum_<lcs::Material::MaterialType>(m, "MaterialType")
		.value("Cloth", lcs::Material::MaterialType::Cloth)
		.value("Tetrahedral", lcs::Material::MaterialType::Tetrahedral)
		.value("Rigid", lcs::Material::MaterialType::Rigid)
		.value("Rod", lcs::Material::MaterialType::Rod)
		.export_values();

	py::enum_<lcs::QuasiStaticInertiaMode>(m, "QuasiStaticInertiaMode")
		.value("Off", lcs::QuasiStaticInertiaMode::Off)
		.value("HessianOnly", lcs::QuasiStaticInertiaMode::HessianOnly)
		.value("Full", lcs::QuasiStaticInertiaMode::Full);

	// Legacy ClothMaterial bindings retained for reference.

	// Legacy TetMaterial bindings retained for reference.

	// Legacy RigidMaterial bindings retained for reference.

	// Legacy RodMaterial bindings retained for reference.

	py::class_<MakeFixedPointsInterface>(m, "MakeFixedPointsInterface")
		.def(py::init<>())
		.def_readwrite("method", &MakeFixedPointsInterface::method)
		.def_readwrite("range", &MakeFixedPointsInterface::range);

	py::class_<WorldDataWrapper>(m, "WorldData")
		.def("set_name", &WorldDataWrapper::set_name)
		.def("set_simulation_type", &WorldDataWrapper::set_simulation_type)
		.def("set_auto_scaling", &WorldDataWrapper::set_auto_scaling, py::arg("enabled"),
			"Scale meshes whose maximum AABB span exceeds 1.5 m down to 0.5 m")
		.def("set_physics_material_cloth",
			&WorldDataWrapper::set_physics_material_cloth,
			py::arg("stretch_model") = std::string(cloth_stretch_model_to_string(ClothMaterial::default_stretch_model())),
			py::arg("bending_model") = std::string(cloth_bending_model_to_string(ClothMaterial::default_bending_model())),
			py::arg("thickness") = ClothMaterial::default_thickness(),
			py::arg("youngs_modulus") = ClothMaterial::default_youngs_modulus(),
			py::arg("poisson_ratio") = ClothMaterial::default_poisson_ratio(),
			py::arg("area_bending_stiffness") = ClothMaterial::default_area_bending_stiffness(),
			py::arg("d_hat") = ClothMaterial::default_d_hat(),
			py::arg("contact_offset") = ClothMaterial::default_contact_offset())
		.def("set_physics_material_tet",
			&WorldDataWrapper::set_physics_material_tet,
			py::arg("model") = std::string(tet_model_to_string(TetMaterial::default_model())),
			py::arg("youngs_modulus") = TetMaterial::default_youngs_modulus(),
			py::arg("poisson_ratio") = TetMaterial::default_poisson_ratio(),
			py::arg("density") = TetMaterial::default_density(),
			py::arg("mass") = TetMaterial::default_mass(),
			py::arg("d_hat") = ClothMaterial::default_d_hat(),
			py::arg("contact_offset") = ClothMaterial::default_contact_offset())
		.def("set_physics_material_rigid",
			&WorldDataWrapper::set_physics_material_rigid,
			py::arg("model") = std::string(rigid_model_to_string(RigidMaterial::default_model())),
			py::arg("thickness") = RigidMaterial::default_thickness(),
			py::arg("stiffness") = RigidMaterial::default_stiffness(),
			py::arg("density") = RigidMaterial::default_density(),
			py::arg("mass") = RigidMaterial::default_mass(),
			py::arg("d_hat") = ClothMaterial::default_d_hat(),
			py::arg("contact_offset") = ClothMaterial::default_contact_offset())
		.def("set_physics_material_rod",
			&WorldDataWrapper::set_physics_material_rod,
			py::arg("model") = std::string(rod_model_to_string(RodMaterial::default_model())),
			py::arg("radius") = RodMaterial::default_radius(),
			py::arg("bending_stiffness") = RodMaterial::default_bending_stiffness(),
			py::arg("twisting_stiffness") = RodMaterial::default_twisting_stiffness(),
			py::arg("density") = RodMaterial::default_density(),
			py::arg("mass") = RodMaterial::default_mass(),
			py::arg("d_hat") = ClothMaterial::default_d_hat(),
			py::arg("contact_offset") = ClothMaterial::default_contact_offset())
		.def("add_fixed_point_by_method", &WorldDataWrapper::add_fixed_point_by_method,
			py::arg("method"),
			py::arg("stiffness") = 1.0f,
			py::arg("range") = 0.001f)
		.def("add_fixed_point_by_indices", &WorldDataWrapper::add_fixed_point_by_indices, py::arg("indices"), py::arg("stiffness") = 1.0f)
		.def("set_translation", &WorldDataWrapper::set_translation)
		.def("set_rotation", &WorldDataWrapper::set_rotation)
		.def("set_scale", &WorldDataWrapper::set_scale)
		.def("set_face_orientation", &WorldDataWrapper::set_face_orientation, py::arg("along_triangle_winding") = true)
		.def("get_rest_translation", &WorldDataWrapper::get_rest_translation)
		.def("get_rest_rotation", &WorldDataWrapper::get_rest_rotation)
		.def("get_rest_scale", &WorldDataWrapper::get_rest_scale)
		.def("get_name", &WorldDataWrapper::get_name)
		.def("get_id", &WorldDataWrapper::get_registration_index)
		.def("get_registration_index", &WorldDataWrapper::get_registration_index)
		.def("get_fixed_point_indices", &WorldDataWrapper::get_fixed_point_indices,
			"Return currently registered fixed-point local vertex indices as a Python list")
		.def("get_rest_positions", &WorldDataWrapper::get_rest_positions,
			"Return rest positions (after object transform) as an (N,3) float32 numpy array");

	py::class_<ConstWorldDataWrapper>(m, "ConstWorldData")
		.def("get_name", &ConstWorldDataWrapper::get_name)
		.def("get_registration_index", &ConstWorldDataWrapper::get_registration_index)
		.def("get_fixed_point_indices", &ConstWorldDataWrapper::get_fixed_point_indices,
			"Return currently registered fixed-point local vertex indices as a Python list")
		.def("get_rest_positions", &ConstWorldDataWrapper::get_rest_positions,
			"Return rest positions (after object transform) as an (N,3) float32 numpy array")
		.def("get_rest_translation", &ConstWorldDataWrapper::get_rest_translation)
		.def("get_rest_rotation", &ConstWorldDataWrapper::get_rest_rotation)
		.def("get_rest_scale", &ConstWorldDataWrapper::get_rest_scale);

	// disambiguate overloaded world_data creation signatures
	using VertArr = py::array_t<double, py::array::c_style | py::array::forcecast>;
	using TriArr = py::array_t<int, py::array::c_style | py::array::forcecast>;

	py::class_<PyNewtonBuilder::PyPRPDebugInfo>(m, "PyPRPDebugInfo")
		.def(py::init<>())
		.def_readwrite("bool_stats", &PyNewtonBuilder::PyPRPDebugInfo::bool_stats)
		.def_readwrite("uint_stats", &PyNewtonBuilder::PyPRPDebugInfo::uint_stats)
		.def_readwrite("float_stats", &PyNewtonBuilder::PyPRPDebugInfo::float_stats);

	py::class_<PyNewtonBuilder>(m, "NewtonSolver")
		.def(py::init<>())
		.def("create_world_data_from_array", &PyNewtonBuilder::create_world_data_from_array, py::arg("name"), py::arg("vertices"), py::arg("triangles"))
		.def("create_world_data_from_file_path", &PyNewtonBuilder::create_world_data_from_file_path, py::arg("name"), py::arg("obj_file_path"))
		.def("register_world_data", &PyNewtonBuilder::register_world_data, py::arg("world_data"), "Register configured WorldData and return object registration id")
		.def("load_scene_from_json",
			&PyNewtonBuilder::load_scene_from_json,
			py::arg("json_path"),
			"Load world_data and scene params from a JSON scene file (same format as app_simulation).")
		.def("num_meshes", &PyNewtonBuilder::num_meshes)
		.def("get_mesh_names", &PyNewtonBuilder::get_mesh_names)
		.def("print_registered_meshes_info", &PyNewtonBuilder::print_registered_meshes_info, "Print registered meshes info")
		.def("init_device",
			&PyNewtonBuilder::init_device,
			py::arg("backend_name") = py::none(),
			py::arg("binary_path") = py::none(),
			"Create and own a luisa compute device/stream.\n\n"
			"backend_name: optional backend string (e.g. 'cuda','metal','dx')\n"
			"binary_path: optional binary path passed to luisa::compute::Context")
		.def("set_device",
			&PyNewtonBuilder::set_device,
			py::arg("device_ptr"),
			py::arg("stream_ptr"),
			"Borrow an existing luisa Device/Stream (non-owning).\n\n"
			"device_ptr: integer address of a luisa::compute::Device object\n"
			"stream_ptr: integer address of a luisa::compute::Stream object\n"
			"The caller must ensure these objects outlive this solver.")
		.def("cleanup_device", &PyNewtonBuilder::cleanup_device, "Release owned device resources (no-op for borrowed device).")
		.def("get_device_ptr", &PyNewtonBuilder::get_device_ptr, "Return the raw pointer (as int) to the active luisa::compute::Device.")
		.def("get_stream_ptr", &PyNewtonBuilder::get_stream_ptr, "Return the raw pointer (as int) to the active luisa::compute::Stream.")
		.def("get_config",
			&PyNewtonBuilder::get_config,
			py::return_value_policy::reference_internal,
			"Return reference to solver-owned SceneParams config")
		.def("init_solver", &PyNewtonBuilder::init_solver, "Initialize the underlying solver using the device set via init_device()/set_device()")
		.def("physics_step_cpu", &PyNewtonBuilder::physics_step_cpu)
		.def("physics_step_gpu", &PyNewtonBuilder::physics_step_gpu)
		.def("restart_system", &PyNewtonBuilder::restart_system, "Reset positions/velocities to initial rest state")
		.def("update_per_vertex_animation", &PyNewtonBuilder::update_per_vertex_animation, py::arg("mesh_idx"), py::arg("local_vid"), py::arg("target_pos"))
		.def("update_per_body_animation", &PyNewtonBuilder::update_per_body_animation, py::arg("mesh_idx"), py::arg("target_translation"), py::arg("target_rotation"))
		.def("get_sim_result", &PyNewtonBuilder::get_sim_result, "Return simulation results as a tuple (vertices_list, faces_list) of numpy arrays")
		.def("get_material_position_data",
			&PyNewtonBuilder::get_material_position_data,
			"Return PRP material-space positions in both global vertex order and per-mesh registration order")
		.def("get_visualize_curves",
			&PyNewtonBuilder::get_visualize_curves,
			"Return visualize curves as a tuple (curve_vertices, curve_edges, curve_edge_colors) of numpy arrays")
		.def("get_intersection_contour_data",
			&PyNewtonBuilder::get_intersection_contour_data,
			"Return contour data dict with EF pairs, adjacency (CSR), contour index and contour membership")
		.def("get_response_pair_data",
			&PyNewtonBuilder::get_response_pair_data,
			"Return response pair dict from target_point_template_pairs and target_point_template_pairs_indices")
		.def("get_collision_buffer_bytes",
			&PyNewtonBuilder::get_collision_buffer_bytes,
			"Return total bytes of all collision-detection buffers (GPU working set)")
		.def("query_local_vid_from_global_vid",
			&PyNewtonBuilder::query_local_vid_from_global_vid,
			"Return global-vertex-id to local-vertex-id mapping as a 1-D uint32 numpy array")
		.def("query_registration_id_from_global_vid",
			&PyNewtonBuilder::query_registration_id_from_global_vid,
			"Return global-vertex-id to world_data(sorted mesh) index mapping as a 1-D uint32 numpy array")
		.def("query_global_vid_from_registration_id_and_local_vid",
			&PyNewtonBuilder::query_global_vid_from_registration_id_and_local_vid,
			py::arg("registration_id"),
			py::arg("local_vid"),
			"Return MeshData global vertex id from registration id and mesh-local vertex id")
		.def("query_vert_adj_edges",
			&PyNewtonBuilder::query_vert_adj_edges,
			py::arg("global_vid"),
			"Return adjacent global edge ids for a global vertex id")
		.def("get_object_sim_result_by_registration_id",
			&PyNewtonBuilder::get_object_sim_result_by_registration_id,
			py::arg("registration_id"),
			"Return one object simulation result as tuple (vertices, faces) by registration id")
		.def("get_object_by_registration_id", &PyNewtonBuilder::get_object_by_registration_id, py::arg("registration_id"))
		.def("set_object_vertex_positions",
			&PyNewtonBuilder::set_object_vertex_positions,
			py::arg("registration_id"),
			py::arg("positions"),
			"Set vertex positions for a registered object by registration id. positions: (N,3) float32 array.")
		.def("update_object_vertex_animation_targets",
			&PyNewtonBuilder::update_object_vertex_animation_targets,
			py::arg("registration_id"),
			py::arg("positions"),
			"Update fixed-point animation targets for one registered object from a (N,3) float32 array.")
		.def("set_object_vertex_positions_with_animation_targets",
			&PyNewtonBuilder::set_object_vertex_positions_with_animation_targets,
			py::arg("registration_id"),
			py::arg("positions"),
			"Set vertex positions and synchronize fixed-point animation targets for one registered object from a (N,3) float32 array.")
		.def("save_sim_result", &PyNewtonBuilder::save_sim_result, py::arg("obj_path"))
		.def("save_current_state", &PyNewtonBuilder::save_current_state, py::arg("state_path"))
		.def("export_contour_submeshes",
			&PyNewtonBuilder::export_contour_submeshes,
			py::arg("output_dir"),
			py::arg("contour_idx"),
			py::arg("ring_count"),
			"Export both contour sides as OBJ submeshes grown by the given vertex-ring count.")
		.def("load_target_state", &PyNewtonBuilder::load_target_state, py::arg("state_path"))
		.def("load_target_state_from_obj", &PyNewtonBuilder::load_target_state_from_obj, py::arg("obj_path"))
		.def("get_prp_debug_info", &PyNewtonBuilder::get_prp_debug_info)
		.def("get_surface_edges", &PyNewtonBuilder::get_surface_edges, "Return surface edge topology as a list of (E,2) uint32 numpy arrays, one per mesh.");

	// Expose luisa::float3 so Python can access .x/.y/.z on floor, gravity, etc.
	py::class_<luisa::float3>(m, "Float3")
		.def(py::init<>())
		.def(py::init<float, float, float>())
		.def_readwrite("x", &luisa::float3::x)
		.def_readwrite("y", &luisa::float3::y)
		.def_readwrite("z", &luisa::float3::z)
		.def("__repr__", [](const luisa::float3& v)
			{ return "Float3(" + std::to_string(v.x) + ", " + std::to_string(v.y) + ", " + std::to_string(v.z) + ")"; });

	// Expose SceneParams and accessors so Python can read/modify global scene settings
	py::class_<lcs::SceneParams>(m, "SceneParams")
		.def("update_dt", &lcs::SceneParams::update_dt)
		.def_readwrite("use_gpu", &lcs::SceneParams::use_gpu)
		.def_readwrite("fix_scene", &lcs::SceneParams::fix_scene)
		.def_readwrite("use_energy_linesearch", &lcs::SceneParams::use_energy_linesearch)
		.def_readwrite("use_ccd_linesearch", &lcs::SceneParams::use_ccd_linesearch)
		.def_readwrite("use_global_ccd", &lcs::SceneParams::use_global_ccd)
		.def_readwrite("use_quasi_static_mode", &lcs::SceneParams::use_quasi_static_mode)
		.def_readwrite("use_static_mode", &lcs::SceneParams::use_static_mode)
		.def_readwrite("qs_inertia_mode", &lcs::SceneParams::qs_inertia_mode)
		.def_readwrite("print_system_energy", &lcs::SceneParams::print_system_energy)
		.def_readwrite("print_pcg_info", &lcs::SceneParams::print_pcg_info)
		.def_readwrite("print_collision_info", &lcs::SceneParams::print_collision_info)
		.def_readwrite("use_floor", &lcs::SceneParams::use_floor)
		.def_readwrite("use_self_collision", &lcs::SceneParams::use_self_collision)
		.def_readwrite("output_per_frame", &lcs::SceneParams::output_per_frame)
		.def_readwrite("output_per_iteration", &lcs::SceneParams::output_per_iteration)
		.def_readwrite("scene_id", &lcs::SceneParams::scene_id)
		.def_readwrite("load_state_frame", &lcs::SceneParams::load_state_frame)
		.def_readonly("num_substep", &lcs::SceneParams::num_substep)
		.def_readwrite("nonlinear_iter_count", &lcs::SceneParams::nonlinear_iter_count)
		.def_readwrite("pcg_iter_count", &lcs::SceneParams::pcg_iter_count)
		.def_readwrite("current_frame", &lcs::SceneParams::current_frame)
		.def_readonly("current_nonlinear_iter", &lcs::SceneParams::current_nonlinear_iter)
		.def_readonly("current_pcg_it", &lcs::SceneParams::current_pcg_it)
		.def_readonly("current_substep", &lcs::SceneParams::current_substep)
		.def_readwrite("collision_detection_frequece", &lcs::SceneParams::collision_detection_frequece)
		.def_readwrite("contact_energy_type", &lcs::SceneParams::contact_energy_type)
		.def_readwrite("implicit_dt", &lcs::SceneParams::implicit_dt)
		.def_readwrite("explicit_dt", &lcs::SceneParams::explicit_dt)
		.def_readonly("dt", &lcs::SceneParams::dt)
		.def_readwrite("floor", &lcs::SceneParams::floor)
		.def_readwrite("gravity", &lcs::SceneParams::gravity)
		.def_readwrite("stiffness_bending_ui", &lcs::SceneParams::stiffness_bending_ui)
		.def_readwrite("stiffness_collision", &lcs::SceneParams::stiffness_collision)
		.def_readwrite("stiffness_untangling", &lcs::SceneParams::stiffness_untangling)
		.def_readwrite("stiffness_dirichlet", &lcs::SceneParams::stiffness_dirichlet)
		.def_readwrite("damping_rate", &lcs::SceneParams::damping_rate)
		.def_readwrite("d_hat", &lcs::SceneParams::d_hat)
		.def_readwrite("use_untangling", &lcs::SceneParams::use_untangling)
		.def_readwrite("use_untangling_ICM", &lcs::SceneParams::use_untangling_ICM)
		.def_readwrite("use_untangling_GIA", &lcs::SceneParams::use_untangling_GIA)
		.def_readwrite("use_untangling_PRP", &lcs::SceneParams::use_untangling_PRP)
		.def_readwrite("use_gpu_untangling", &lcs::SceneParams::use_gpu_untangling)
		.def_readwrite("PRP_use_intrinsic_contour_side_candidates", &lcs::SceneParams::PRP_use_intrinsic_contour_side_candidates)
		.def_readwrite("PRP_use_rest_geodesic_distance_for_intrinsic_candidates", &lcs::SceneParams::PRP_use_rest_geodesic_distance_for_intrinsic_candidates)
		.def_readwrite("PRP_intrinsic_tau", &lcs::SceneParams::PRP_intrinsic_tau)
		.def_readwrite("PRP_use_blended_intrinsic_coordinates", &lcs::SceneParams::PRP_use_blended_intrinsic_coordinates)
		.def_readwrite("PRP_intrinsic_blend_weight", &lcs::SceneParams::PRP_intrinsic_blend_weight)
		.def_readwrite("PRP_use_deformed_boundary_distance_for_intrinsic_candidates", &lcs::SceneParams::PRP_use_deformed_boundary_distance_for_intrinsic_candidates)
		.def_readwrite("PRP_direction_optimization_iterations", &lcs::SceneParams::PRP_direction_optimization_iterations)
		.def_readwrite("accumulate_ICM_correction", &lcs::SceneParams::accumulate_ICM_correction)
		.def_readwrite("GIA_use_EF_response", &lcs::SceneParams::GIA_use_EF_response)
		.def_readwrite("ignore_near_zero_dist_pairs", &lcs::SceneParams::ignore_near_zero_dist_pairs)
		.def_readwrite("PRP_cpu_contour_batch_size", &lcs::SceneParams::PRP_cpu_contour_batch_size)
		.def_readwrite("untangling_process_contours_count", &lcs::SceneParams::untangling_process_contours_count)
		.def_readwrite("untangling_response_depth", &lcs::SceneParams::untangling_response_depth)
		.def_readwrite("pcg_lm_adaptive", &lcs::SceneParams::pcg_lm_adaptive)
		.def_readwrite("pcg_lm_escalation_factor", &lcs::SceneParams::pcg_lm_escalation_factor)
		.def_readwrite("consistent_solve", &lcs::SceneParams::consistent_solve)
		.def_readwrite("collect_iteration_debug", &lcs::SceneParams::collect_iteration_debug)
		.def_readwrite("prp_debug", &lcs::SceneParams::prp_debug)
		.def_readwrite("PRP_solid_interior_adjacency", &lcs::SceneParams::PRP_solid_interior_adjacency)
		.def_readwrite("PRP_solid_interior_adjacency_all_meshes", &lcs::SceneParams::PRP_solid_interior_adjacency_all_meshes)
		.def_readwrite("PRP_solid_interior_adjacency_normal_eps", &lcs::SceneParams::PRP_solid_interior_adjacency_normal_eps)
		.def_readwrite("PRP_solid_interior_adjacency_dist_eps", &lcs::SceneParams::PRP_solid_interior_adjacency_dist_eps)
		.def("get_substep_dt", &lcs::SceneParams::get_substep_dt)
		.def("get_bending_stiffness_scaling", &lcs::SceneParams::get_bending_stiffness_scaling);

	m.doc() = "Python bindings for basic NewtonSolver scene building (lightweight)";
}
