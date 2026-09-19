#include <iostream>
#include <luisa/luisa-compute.h>
#include <string>

#include "SimulationSolver/newton_solver.h"
#include "SimulationCore/scene_params.h"
#include "SimulationSolver/solver_interface.h"
#include "Utils/cpu_parallel.h"
#include "app_simulation_demo_config.h"
#include "Utils/color_utils.h"

#if defined(SIMULATION_APP_USE_GUI)
	#include <polyscope/polyscope.h>
	#include <polyscope/surface_mesh.h>
	#include <polyscope/curve_network.h>
	#include <Eigen/Dense>
#endif

template <typename T>
using Buffer = luisa::compute::Buffer<T>;

int main(int argc, char** argv)
{

#ifndef NDEBUG
	luisa::log_level_info(); // log_level_verbose
#else
	luisa::log_level_info();
#endif

	LUISA_INFO("Hello, LuisaComputeSimulation!");

	// Init GPU system
#if defined(__APPLE__)
	std::string backend = "metal";
#elif defined(_WIN32)
	std::string backend = "cuda";
#else
	std::string backend = "cuda";
#endif
	if (argc >= 2)
	{
		backend = argv[1];
	}

	const std::string binary_path(argv[0]);

	lcs::NewtonSolver solver;
	solver.create_device(binary_path, backend);

	// Read Mesh
	std::string scene_json_path = "default_scene.json";
	if (argc >= 3)
	{
		LUISA_INFO("Load target scene {}", argv[2]);
		scene_json_path = argv[2];
	}
	else
	{
		LUISA_INFO("Load default scene {}", scene_json_path);
	}

	auto fn_register_mesh_from_json_file = [&](const lcs::Initializer::WorldData& wd)
	{
		return solver.register_world_data(wd);
	};
	Demo::Simulation::load_scene_params_from_json(fn_register_mesh_from_json_file, scene_json_path);

	// Init Solver
	solver.init_solver();

	auto fn_update_pinned_verts = [&](const uint curr_frame)
	{
		solver.update_default_animations();
		// You can also set animation for specific vertices or bodies by calling
		// 		`solver.update_per_vertex_animation()` or `solver.update_per_body_animation()`
		// solver.update_per_vertex_animation(meshIdx, local_vid, target_position);
		// solver.update_per_body_animation(body_id, target_translation, target_rotation);
	};

	auto fn_physics_step = [&]()
	{
		fn_update_pinned_verts(lcs::get_scene_params().current_frame);

		if (lcs::get_scene_params().use_gpu)
			solver.physics_step_GPU();
		else
			solver.physics_step_CPU();
	};

	uint		   max_frame = 0;
	uint		   optimize_frames = 20;
	constexpr bool draw_bounding_box = false;
	constexpr bool use_ui = true;

	const auto& sorted_data = solver.get_sorted_world_data();
	const uint	num_meshes = sorted_data.size();

	// Init surface rendering data
	std::vector<std::vector<std::array<uint, 2>>>  surface_edges(num_meshes);
	std::vector<std::vector<std::array<float, 3>>> surface_vertices(num_meshes);
	std::vector<std::vector<std::array<uint, 3>>>  surface_triangles(num_meshes);
	std::vector<std::vector<std::array<float, 3>>> surface_default_face_colors(num_meshes);
	{
		solver.get_rest_vertices_to_host(surface_vertices);
		solver.get_triangles_to_host(surface_triangles);
		solver.get_edges_to_host(surface_edges);
		for (uint meshIdx = 0; meshIdx < num_meshes; meshIdx++)
		{
			surface_default_face_colors[meshIdx].resize(surface_triangles[meshIdx].size(), { 0.7f, 0.2f, 0.3f });
		}
	}

	// Init curve rendering data
	std::vector<std::array<float, 3>> curve_vertices;
	std::vector<std::array<uint, 2>>  curve_edges;
	std::vector<std::array<float, 3>> curve_edge_colors;

	auto fn_get_vertex_position = [&](const uint global_vid) -> std::array<float, 3>
	{
		const uint mesh_idx = solver.query_registration_id_from_global_vid(global_vid);
		const uint local_vid = solver.query_local_vid_from_global_vid(global_vid);
		return surface_vertices[mesh_idx][local_vid];
	};

	auto fn_update_rendering_vertices = [&]()
	{
		// Get simulation results
		solver.get_curr_vertices_to_host(surface_vertices);
		solver.get_visualize_curves(curve_vertices, curve_edges, curve_edge_colors);
		// LUISA_INFO("curve_vertices = {}", curve_vertices);
		// LUISA_INFO("curve_edges = {}", curve_edges);
		// LUISA_INFO("curve_edge_colors = {}", curve_edge_colors);
	};

	auto fn_save_frame_to_obj = [&](const uint frame_id, const auto& sim_result, const std::string& additional_info = "")
	{
		std::string full_path = luisa::format("{}/OutputMesh/0{}/frame_{}{}.obj",
			LCSV_RESOURCE_PATH,
			lcs::get_scene_params().scene_id,
			frame_id, additional_info);
		SimMesh::saveToOBJ_combined(sim_result, surface_triangles, full_path);
		// solver.save_mesh_to_obj(full_path);
	};

#if !defined(SIMULATION_APP_USE_GUI)
	{
		constexpr bool												   use_merge_writing = true;
		std::map<uint, std::vector<std::vector<std::array<float, 3>>>> per_frame_rendering_vertices;
		auto														   fn_save_frame_to_obj_merge =
			[&](const std::pair<uint, std::vector<std::vector<std::array<float, 3>>>>& curr_frame_result,
				const std::string&													   additional_info = "")
		{
			fn_save_frame_to_obj(curr_frame_result.first, curr_frame_result.second, additional_info);
		};

		auto fn_single_step_without_ui = [&]()
		{ fn_physics_step(); };

		const uint frame_start = lcs::get_scene_params().current_frame;
		const uint frame_end = frame_start + 50;

		fn_save_frame_to_obj(0, surface_vertices, "_init");
		for (uint frame = frame_start; frame < frame_end; frame++)
		{
			fn_single_step_without_ui();

			fn_update_rendering_vertices();

			continue;

			const uint curr_frame = lcs::get_scene_params().current_frame - 1;
			if (use_merge_writing)
			{
				if (curr_frame % 2 == 1)
					per_frame_rendering_vertices[curr_frame] = surface_vertices;

				if (curr_frame % 100 == 99 || frame == frame_end - 1)
				{
					CpuParallel::parallel_for_each_core(0,
						per_frame_rendering_vertices.size(),
						[&](uint idx)
						{
							auto iter = per_frame_rendering_vertices.begin();
							std::advance(iter, idx);
							fn_save_frame_to_obj_merge(*iter);
						});
					per_frame_rendering_vertices.clear();
					if (curr_frame != frame_start)
					{
						std::string scene_directory = luisa::format("{}/OutputMesh/0{}", LCSV_RESOURCE_PATH, lcs::get_scene_params().scene_id);
						std::string save_state_full_path = scene_directory + "/saved_state.state";
						solver.lcs::SolverInterface::save_current_frame_state_file(save_state_full_path);
					}
				}
			}
			else
			{
				fn_save_frame_to_obj(frame, surface_vertices, "");
			}
		}
	}
#else
	{
		// Init Polyscope
		polyscope::init("openGL3_glfw");
		std::vector<polyscope::SurfaceMesh*> visualize_meshes;
		polyscope::CurveNetwork*			 visualize_curves = nullptr;

		for (uint meshIdx = 0; meshIdx < num_meshes; meshIdx++)
		{
			const std::string&		curr_mesh_name = solver.get_object_by_registration_id(meshIdx).get_model_name() + std::to_string(meshIdx);
			polyscope::SurfaceMesh* curr_mesh_ptr = polyscope::registerSurfaceMesh(
				curr_mesh_name, surface_vertices[meshIdx], surface_triangles[meshIdx]);
			curr_mesh_ptr->setEnabled(true);
			curr_mesh_ptr->setEdgeWidth(1.0);
			curr_mesh_ptr->addFaceColorQuantity("Surface Colors", surface_default_face_colors[meshIdx])->setEnabled(true);
			visualize_meshes.push_back(curr_mesh_ptr);
		}

		auto fn_update_GUI_vertices = [&]()
		{
			for (uint meshIdx = 0; meshIdx < num_meshes; meshIdx++)
			{
				visualize_meshes[meshIdx]->updateVertexPositions(surface_vertices[meshIdx]);
			}

			if (!curve_vertices.empty() && !curve_edges.empty())
			{
				visualize_curves =
					polyscope::registerCurveNetwork("Visualize Curves", curve_vertices, curve_edges);
				visualize_curves->setRadius(5e-4f);
				visualize_curves->addEdgeColorQuantity("Curve Colors", curve_edge_colors)->setEnabled(true);
			}
			else
			{
				if (visualize_curves != nullptr)
				{
					polyscope::removeStructure("Visualize Curves");
					visualize_curves = nullptr;
				}
			}
		};
		auto fn_single_step_with_ui = [&]()
		{
			// LUISA_INFO("     Sync frame {}", lcs::get_scene_params().current_frame);
			fn_physics_step();

			fn_update_rendering_vertices();
			fn_update_GUI_vertices();
		};

		bool is_simulate_frame = false;

		polyscope::state::userCallback = [&]()
		{
			if (ImGui::IsKeyPressed(ImGuiKey_Escape))
				polyscope::unshow();

			// Selection
			{
				static polyscope::PickResult prev_selection;
				if (polyscope::haveSelection())
				{
					const auto& mesh_data = solver.lcs::SolverInterface::get_host_mesh_data();

					polyscope::PickResult selection = polyscope::getSelection();
					if (selection.isHit
						&& !(selection.screenCoords.x == prev_selection.screenCoords.x
							&& selection.screenCoords.y == prev_selection.screenCoords.y))
					{
						prev_selection = selection;
						for (uint meshIdx = 0; meshIdx < visualize_meshes.size(); meshIdx++)
						{
							polyscope::SurfaceMesh* mesh = visualize_meshes[meshIdx];
							if (mesh == selection.structure)
							{
								polyscope::SurfaceMeshPickResult meshPickResult =
									mesh->interpretPickResult(selection);
								if (meshPickResult.elementType == polyscope::MeshElement::VERTEX)
								{
									uint prefix = mesh_data.prefix_num_verts[meshIdx];
									uint vid = prefix + meshPickResult.index;
									LUISA_INFO("Select Vert {:3} on mesh {}", vid, meshIdx);
								}
								else if (meshPickResult.elementType == polyscope::MeshElement::FACE)
								{
									uint prefix = mesh_data.prefix_num_faces[meshIdx];
									uint vid = prefix + meshPickResult.index;
									LUISA_INFO("Select Face {:3} on mesh {}", vid, meshIdx);
								}
								else if (meshPickResult.elementType == polyscope::MeshElement::EDGE)
								{
									uint prefix = mesh_data.prefix_num_edges[meshIdx];
									uint vid = prefix + meshPickResult.index;
									LUISA_INFO("Select Edge {:3} on mesh {}", vid, meshIdx);
								}
							}
						}
					}
				}
			}

			uint& current_frame = lcs::get_scene_params().current_frame;

			if (ImGui::CollapsingHeader("Parameters", ImGuiTreeNodeFlags_DefaultOpen))
			{
				// ImGui::InputScalar("Num Substep", ImGuiDataType_U32, &lcs::get_scene_params().num_substep);
				ImGui::InputScalar("Num Nonliear-Iteration", ImGuiDataType_U32, &lcs::get_scene_params().nonlinear_iter_count);
				ImGui::InputScalar("Num PCG-Iteration", ImGuiDataType_U32, &lcs::get_scene_params().pcg_iter_count);
				// ImGui::SliderFloat("Implicit Timestep", &lcs::get_scene_params().implicit_dt, 0.0001f, 0.2f);
				ImGui::Checkbox("Use Quasi-Static Mode", &lcs::get_scene_params().use_quasi_static_mode);
				// ImGui::Checkbox("Use 	   Static Mode", &lcs::get_scene_params().use_static_mode);
				ImGui::Checkbox("Use Energy LineSearch", &lcs::get_scene_params().use_energy_linesearch);
				ImGui::Checkbox("Use CCD LineSearch", &lcs::get_scene_params().use_ccd_linesearch);
				ImGui::Checkbox("Use Global CCD", &lcs::get_scene_params().use_global_ccd);
				if (lcs::get_scene_params().contact_energy_type == uint(lcs::ContactEnergyType::Barrier)
					&& (lcs::get_scene_params().use_self_collision || lcs::get_scene_params().use_floor))
					lcs::get_scene_params().use_ccd_linesearch = true;

				// ImGui::Checkbox("Use Bending", &lcs::get_scene_params().use_bending);
				// ImGui::Checkbox("Use Quadratic Bending", &lcs::get_scene_params().use_quadratic_bending_model);
				ImGui::SliderFloat("Bending Stiffness", &lcs::get_scene_params().stiffness_bending_ui, 0.0f, 10.0f);
				// static int stiffness_bending_exp = 0;
				// ImGui::InputInt("Bending Stiffness's Exp", &stiffness_bending_exp);
				// lcs::get_scene_params().stiffness_bending_ui = pow(10.0f, (float)stiffness_bending_exp);

				static uint stiffness_spring_exp = 4;
				ImGui::Checkbox("Print Energy", &lcs::get_scene_params().print_system_energy);
				ImGui::Checkbox("Use GPU Solver", &lcs::get_scene_params().use_gpu);
				ImGui::Checkbox("Use Self-Collision", &lcs::get_scene_params().use_self_collision);

				// static const char* items[] = { "A", "B", "C" };
				// static int current_item = 0;
				// ImGui::Combo("Combo", &current_item, items, IM_ARRAYSIZE(items));
			}

			if (ImGui::CollapsingHeader("Simulation", ImGuiTreeNodeFlags_DefaultOpen))
			{
				ImGui::Text("Frame %d", current_frame);
				if (ImGui::Button("Reset", ImVec2(-1, 0)))
				{
					current_frame = 0;
					max_frame = 0;
					solver.lcs::SolverInterface::restart_system();
					fn_update_rendering_vertices();
					fn_update_GUI_vertices();
				}
				if (ImGui::Button("Advance Single Frame", ImVec2(-1, 0)))
				{
					fn_single_step_with_ui();
				}
				if (ImGui::Button("Start Simulation", ImVec2(-1, 0)))
				{
					is_simulate_frame = true;
					max_frame = 10000;
				}
				if (ImGui::Button("End Simulation", ImVec2(-1, 0)))
				{
					is_simulate_frame = false;
					max_frame = current_frame;
				}
			}

			if (ImGui::CollapsingHeader("Untangling", ImGuiTreeNodeFlags_DefaultOpen))
			{
				ImGui::Checkbox("Use Untangling", &lcs::get_scene_params().use_untangling);
				ImGui::InputScalar("Process Contour Count", ImGuiDataType_U32, &lcs::get_scene_params().untangling_process_contours_count);
				ImGui::SliderFloat("Untangling Response Depth", &lcs::get_scene_params().untangling_response_depth, 1e-3f, 1e-1f);
			}
			// Ground Collision
			{
				if (lcs::get_scene_params().use_floor)
				{
					ImGui::SliderFloat("Floor Y", &lcs::get_scene_params().floor.y, -1.0f, 1.0f);
					polyscope::options::groundPlaneMode = polyscope::GroundPlaneMode::TileReflection;
					polyscope::options::groundPlaneHeightMode = polyscope::GroundPlaneHeightMode::Manual;
					polyscope::options::groundPlaneHeight = lcs::get_scene_params().floor.y;
				}
				else
				{
					polyscope::options::groundPlaneMode = polyscope::GroundPlaneMode::None;
				}
			}
			if (ImGui::CollapsingHeader("Collision"))
			{
				ImGui::Checkbox("Use Ground Collision", &lcs::get_scene_params().use_floor);
				const uint	offset_pairs = lcs::CollisionPair::CollisionCount::total_adj_pairs_offset();
				const uint	offset_verts = lcs::CollisionPair::CollisionCount::total_adj_verts_offset();
				const auto& host_collision_data = solver.lcs::SolverInterface::get_host_collision_data();
				const auto& device_collision_data = solver.lcs::SolverInterface::get_device_collision_data();
				ImGui::Text("Broad VF/EE = %u / %u Narrow = %u , Triplet = %u",
					host_collision_data.broad_phase_collision_count[2],
					host_collision_data.broad_phase_collision_count[3],
					host_collision_data.narrow_phase_collision_count.front(),
					host_collision_data.narrow_phase_collision_count[offset_verts]);
				ImGui::Text("MaxCount = %zu / %zu ,  Narrow %zu , Triplet = %zu)",
					device_collision_data.broad_phase_list_vf.size() / 2,
					device_collision_data.broad_phase_list_ee.size() / 2,
					device_collision_data.narrow_phase_list.size(),
					device_collision_data.triplet_data.sa_cgA_contact_offdiag_triplet.size());
			}

			if (ImGui::CollapsingHeader("Data IO"))
			{
				std::string scene_directory = luisa::format("{}/SimulationState/0{}", LCSV_RESOURCE_PATH, lcs::get_scene_params().scene_id);

				if (ImGui::Button("Save mesh", ImVec2(-1, 0)))
				{
					fn_save_frame_to_obj(current_frame, surface_vertices, "");
				}
				ImGui::Checkbox("Output Each Frame", &lcs::get_scene_params().output_per_frame);
				if (ImGui::Button("Save State", ImVec2(-1, 0)))
				{
					std::string save_state_full_path = scene_directory + luisa::format("/frame_{}.state", current_frame);
					solver.lcs::SolverInterface::save_current_frame_state_file(save_state_full_path);
				}
				uint& state_frame = lcs::get_scene_params().load_state_frame;
				if (ImGui::Button(luisa::format("Load State in Frame {}", state_frame).c_str(), ImVec2(-1, 0)))
				{
					std::string save_state_full_path = scene_directory + luisa::format("/frame_{}.state", state_frame);
					solver.lcs::SolverInterface::load_state_from_state_file(save_state_full_path);
					lcs::get_scene_params().current_frame = state_frame;
					fn_update_rendering_vertices();
					fn_update_GUI_vertices();
				}
				ImGui::InputScalar("Load State Frame", ImGuiDataType_U32, &state_frame);
			}

			if (ImGui::CollapsingHeader("Highlight"))
			{
				static uint vis_element_id = -1u;
				ImGui::InputScalar("Highlignt Element ID", ImGuiDataType_U32, &vis_element_id);
				if (vis_element_id != -1u)
				{
					static const char* items[] = { "Vert", "Edge", "Face", "Pair" };
					static int		   current_item = 0;
					ImGui::Combo("Highlight Element Type", &current_item, items, IM_ARRAYSIZE(items));
					if (ImGui::Button("Highlignt Element", ImVec2(-1, 0)))
					{
						curve_vertices.clear();
						curve_edges.clear();
						curve_edge_colors.clear();
						const auto& mesh_data = solver.lcs::SolverInterface::get_host_mesh_data();
						if (current_item == 0)
						{
							const uint vid = vis_element_id;
							const auto p = fn_get_vertex_position(vid);
							curve_vertices.push_back(p);
							curve_edges.push_back({ 0, 0 });
							curve_edge_colors.push_back({ 1.0f, 0.0f, 0.0f });
						}
						else if (current_item == 1)
						{
							const uint eid = vis_element_id;
							const auto edge = mesh_data.sa_edges[eid];
							auto	   p1 = fn_get_vertex_position(edge.x);
							auto	   p2 = fn_get_vertex_position(edge.y);
							curve_vertices.push_back(p1);
							curve_vertices.push_back(p2);
							curve_edges.push_back({ 0, 1 });
							curve_edge_colors.push_back({ 1.0f, 0.0f, 0.0f });
						}
						else if (current_item == 2)
						{
							const uint fid = vis_element_id;
							const auto face = mesh_data.sa_faces[fid];
							auto	   p1 = fn_get_vertex_position(face.x);
							auto	   p2 = fn_get_vertex_position(face.y);
							auto	   p3 = fn_get_vertex_position(face.z);
							curve_vertices.push_back(p1);
							curve_vertices.push_back(p2);
							curve_vertices.push_back(p3);

							curve_edges.push_back({ 0, 1 });
							curve_edges.push_back({ 1, 2 });
							curve_edges.push_back({ 2, 0 });
							curve_edge_colors.push_back({ 1.0f, 0.0f, 0.0f });
							curve_edge_colors.push_back({ 1.0f, 0.0f, 0.0f });
							curve_edge_colors.push_back({ 1.0f, 0.0f, 0.0f });
						}
						else if (current_item == 3)
						{
							auto vis_target_point = [&](const lcs::CollisionPair::CollisionPairTemplate& info, uint lable)
							{
								using Vec3 = std::array<float, 3>;
								using Edge = std::array<uint, 2>;
								auto vec3_to_float3 = [](const Vec3& v)
								{ return luisa::float3{ v[0], v[1], v[2] }; };
								const auto			ids = info.get_indices();
								const luisa::float3 positions[4] = {
									vec3_to_float3(fn_get_vertex_position(ids.x)),
									vec3_to_float3(fn_get_vertex_position(ids.y)),
									vec3_to_float3(fn_get_vertex_position(ids.z)),
									vec3_to_float3(fn_get_vertex_position(ids.w)),
								};
								const auto	weights = luisa::abs(info.get_weight());
								const auto	type = info.get_collision_type();
								lcs::float3 p1;
								lcs::float3 p2;
								if (type == lcs::CollisionPair::type_vf())
								{
									p1 = positions[0];
									p2 = weights[1] * positions[1] + weights[2] * positions[2] + weights[3] * positions[3];
								}
								else if (type == lcs::CollisionPair::type_ee())
								{
									p1 = weights[0] * positions[0] + weights[1] * positions[1];
									p2 = weights[2] * positions[2] + weights[3] * positions[3];
								}
								// const uint lable = uint(info.get_area());
								curve_vertices.push_back(Vec3{ p1.x, p1.y, p1.z });
								curve_vertices.push_back(Vec3{ p2.x, p2.y, p2.z });
								curve_edges.push_back(Edge{ uint(curve_vertices.size() - 2), uint(curve_vertices.size() - 1) });
								curve_edge_colors.push_back(lcs::ColorUtils::camp10(lable));
							};

							const auto& untangling_data = solver.get_host_untangling_data();
							const auto& vf_ee_pairs = untangling_data.target_point_template_pairs;
							const uint	pair_idx = vis_element_id;
							vis_target_point(vf_ee_pairs[pair_idx], 0);
						}

						visualize_curves = polyscope::registerCurveNetwork(
							"Visualize Element", curve_vertices, curve_edges);
						visualize_curves->setRadius(5e-3f);
						visualize_curves->addEdgeColorQuantity("Highlight Color", curve_edge_colors)->setEnabled(true);
					}
				}
			}

			if (is_simulate_frame)
			{
				if (lcs::get_scene_params().output_per_frame && current_frame == 0)
				{
					// First frame
					fn_save_frame_to_obj(0, surface_vertices, "_init");
				}

				fn_single_step_with_ui();

				const float animation_fps = 60.0f;
				const uint	output_freq = (1.0f / animation_fps) / lcs::get_scene_params().implicit_dt;
				if (lcs::get_scene_params().output_per_frame && current_frame % output_freq == 0)
				{
					fn_save_frame_to_obj(current_frame, surface_vertices, "");
				}
				if (current_frame >= max_frame)
				{
					is_simulate_frame = false;
				}
			}
		};
		polyscope::show();
	}
#endif
	return 0;
}
