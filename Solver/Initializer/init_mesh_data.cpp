#include "Initializer/init_mesh_data.h"
#include "Core/affine_position.h"
#include "Core/float_nxn.h"
#include "Core/scalar.h"
#include "Energy/bending_energy.h"
#include "MeshOperation/mesh_reader.h"
#include "Initializer/initializer_utils.h"
#include "Utils/cpu_parallel.h"
#include <algorithm>
#include <cstdint>
#include <limits>
#include <numeric>
#include <unordered_map>
#include <unordered_set>

namespace lcs
{
	struct AABB
	{
		float3 packed_min;
		float3 packed_max;
		AABB   operator+(const AABB& input_aabb) const
		{
			AABB tmp;
			tmp.packed_min = lcs::min_vec(packed_min, input_aabb.packed_min);
			tmp.packed_max = lcs::max_vec(packed_max, input_aabb.packed_max);
			return tmp;
		}
		AABB()
			: packed_min(float3(Float_max))
			, packed_max(float3(-Float_max))
		{
		}
		AABB(const float3& pos)
			: packed_min(pos)
			, packed_max(pos)
		{
		}
	};

	namespace Initializer
	{
		// template<template<typename> typename BasicBuffer>
		void init_mesh_data(const std::vector<lcs::Initializer::WorldData>& world_data, lcs::MeshData<std::vector>* mesh_data)
		{
			const uint num_meshes = world_data.size();
			mesh_data->num_meshes = num_meshes;

			mesh_data->num_verts = 0;
			mesh_data->num_faces = 0;
			mesh_data->num_edges = 0;
			mesh_data->num_dihedral_edges = 0;
			mesh_data->num_tets = 0;

			mesh_data->prefix_num_verts.resize(1 + num_meshes, 0);
			mesh_data->prefix_num_faces.resize(1 + num_meshes, 0);
			mesh_data->prefix_num_edges.resize(1 + num_meshes, 0);
			mesh_data->prefix_num_dihedral_edges.resize(1 + num_meshes, 0);
			mesh_data->prefix_num_tets.resize(1 + num_meshes, 0);
			mesh_data->sorted_to_input_mesh_id.resize(num_meshes);
			mesh_data->input_to_sorted_mesh_id.resize(num_meshes);

			mesh_data->sa_rest_translate.resize(num_meshes);
			mesh_data->sa_rest_rotation.resize(num_meshes);
			mesh_data->sa_rest_scale.resize(num_meshes);

			mesh_data->fixed_verts_map.resize(num_meshes);

			// Constant scalar and init MeshData
			for (uint meshIdx = 0; meshIdx < num_meshes; meshIdx++)
			{
				auto& shell_info = world_data[meshIdx];
				auto& input_mesh = shell_info.input_mesh;

				uint registration_index = shell_info.get_registration_index();
				mesh_data->sorted_to_input_mesh_id[meshIdx] = registration_index;
				mesh_data->input_to_sorted_mesh_id[registration_index] = meshIdx;

				mesh_data->prefix_num_verts[meshIdx] = mesh_data->num_verts;
				mesh_data->prefix_num_faces[meshIdx] = mesh_data->num_faces;
				mesh_data->prefix_num_edges[meshIdx] = mesh_data->num_edges;
				mesh_data->prefix_num_dihedral_edges[meshIdx] = mesh_data->num_dihedral_edges;
				mesh_data->prefix_num_tets[meshIdx] = mesh_data->num_tets;

				const uint curr_num_verts = input_mesh.model_positions.size();
				const uint curr_num_faces = input_mesh.faces.size();
				const uint curr_num_edges = input_mesh.edges.size();
				const uint curr_num_dihedral_edges = input_mesh.dihedral_edges.size();
				const uint curr_num_tets = input_mesh.tetrahedrons.size();

				mesh_data->num_verts += curr_num_verts;
				mesh_data->num_faces += curr_num_faces;
				mesh_data->num_edges += curr_num_edges;
				mesh_data->num_dihedral_edges += curr_num_dihedral_edges;
				mesh_data->num_tets += curr_num_tets;
			}

			mesh_data->prefix_num_verts[num_meshes] = mesh_data->num_verts;
			mesh_data->prefix_num_faces[num_meshes] = mesh_data->num_faces;
			mesh_data->prefix_num_edges[num_meshes] = mesh_data->num_edges;
			mesh_data->prefix_num_dihedral_edges[num_meshes] = mesh_data->num_dihedral_edges;
			mesh_data->prefix_num_tets[num_meshes] = mesh_data->num_tets;

			uint num_verts = mesh_data->num_verts;
			uint num_faces = mesh_data->num_faces;
			uint num_edges = mesh_data->num_edges;
			uint num_dihedral_edges = mesh_data->num_dihedral_edges;
			uint num_tets = mesh_data->num_tets;

			LUISA_INFO("Mesh : (numVerts : {}) (numFaces : {})  (numEdges : {}) (numDihedralEdges : {}), (numTets : {})",
				num_verts,
				num_faces,
				num_edges,
				num_dihedral_edges,
				num_tets);

			// Read information
			{
				mesh_data->sa_rest_x.resize(num_verts);
				mesh_data->sa_model_x.resize(num_verts);
				mesh_data->sa_scaled_model_x.resize(num_verts);
				mesh_data->sa_material_x.resize(num_verts);
				mesh_data->sa_faces.resize(num_faces);
				mesh_data->sa_edges.resize(num_edges);
				mesh_data->sa_dihedral_edges.resize(num_dihedral_edges);
				mesh_data->sa_tetrahedrons.resize(num_tets);

				mesh_data->sa_rest_v.resize(num_verts);
				mesh_data->sa_is_fixed.resize(num_verts);
				mesh_data->sa_fixed_stiffness.resize(num_verts);
				mesh_data->sa_vert_mesh_type.resize(num_verts);
				mesh_data->sa_mesh_orientation.resize(num_meshes, 0u);

				mesh_data->sa_vert_mesh_id.resize(num_verts);
				mesh_data->sa_face_mesh_id.resize(num_faces);
				mesh_data->sa_edge_mesh_id.resize(num_edges);
				mesh_data->sa_dihedral_edge_mesh_id.resize(num_dihedral_edges);
				mesh_data->sa_tet_mesh_id.resize(num_tets);
				mesh_data->sa_global_vid_to_local_vid.resize(num_verts);

				uint prefix_num_verts = 0;
				uint prefix_num_faces = 0;
				uint prefix_num_edges = 0;
				uint prefix_num_dihedral_edges = 0;
				uint prefix_num_tets = 0;

				for (uint meshIdx = 0; meshIdx < num_meshes; meshIdx++)
				{
					auto&		curr_shell_info = world_data[meshIdx];
					const auto& curr_input_mesh = curr_shell_info.get_mesh();

					// Model info
					{
						mesh_data->sa_rest_translate[meshIdx] = curr_shell_info.translation;
						mesh_data->sa_rest_rotation[meshIdx] = curr_shell_info.rotation;
						mesh_data->sa_rest_scale[meshIdx] = curr_shell_info.scale;
						mesh_data->sa_mesh_orientation[meshIdx] = curr_shell_info.face_orientation_along_triangle_winding_mode;
					}

					const uint curr_num_verts = curr_input_mesh.model_positions.size();
					const uint curr_num_faces = curr_input_mesh.faces.size();
					const uint curr_num_edges = curr_input_mesh.edges.size();
					const uint curr_num_dihedral_edges = curr_input_mesh.dihedral_edges.size();
					const uint curr_num_tets = curr_input_mesh.tetrahedrons.size();

					// Read position with affine
					CpuParallel::parallel_for(
						0,
						curr_num_verts,
						[&](const uint vid)
						{
							std::array<float, 3> read_pos = curr_input_mesh.model_positions[vid];
							float3				 model_position = luisa::make_float3(read_pos[0], read_pos[1], read_pos[2]);
							float4x4			 model_matrix = lcs::make_model_matrix(
								curr_shell_info.translation, curr_shell_info.rotation, curr_shell_info.scale);
							float3 world_position = lcs::affine_position(model_matrix, model_position);
							mesh_data->sa_model_x[prefix_num_verts + vid] = model_position;
							mesh_data->sa_scaled_model_x[prefix_num_verts + vid] = curr_shell_info.scale * model_position;
							mesh_data->sa_rest_x[prefix_num_verts + vid] = world_position;
							mesh_data->sa_rest_v[prefix_num_verts + vid] = luisa::make_float3(0.0f);
							mesh_data->sa_vert_mesh_id[prefix_num_verts + vid] = meshIdx;
							mesh_data->sa_vert_mesh_type[prefix_num_verts + vid] = uint(curr_shell_info.material_type);
							mesh_data->sa_global_vid_to_local_vid[prefix_num_verts + vid] = vid;

							std::array<float, 2> read_uv = curr_input_mesh.has_uv ? curr_input_mesh.uv_positions[vid] : std::array<float, 2>{ 0.0f, 0.0f };
							mesh_data->sa_material_x[prefix_num_verts + vid] =
								luisa::make_float2(model_position.x, model_position.z); // Drop y for material coord
						});
					// Read triangle face
					CpuParallel::parallel_for(0,
						curr_num_faces,
						[&](const uint fid)
						{
							auto face = curr_input_mesh.faces[fid];
							mesh_data->sa_faces[prefix_num_faces + fid] =
								prefix_num_verts + luisa::make_uint3(face[0], face[1], face[2]);
							mesh_data->sa_face_mesh_id[prefix_num_faces + fid] = meshIdx;
						});

					// Read edge
					CpuParallel::parallel_for(0,
						curr_num_edges,
						[&](const uint eid)
						{
							auto edge = curr_input_mesh.edges[eid];
							mesh_data->sa_edges[prefix_num_edges + eid] =
								prefix_num_verts + luisa::make_uint2(edge[0], edge[1]);
							mesh_data->sa_edge_mesh_id[prefix_num_edges + eid] = meshIdx;
						});
					// Read bending edge
					CpuParallel::parallel_for(
						0,
						curr_num_dihedral_edges,
						[&](const uint eid)
						{
							auto bending_edge = curr_input_mesh.dihedral_edges[eid];
							mesh_data->sa_dihedral_edges[prefix_num_dihedral_edges + eid] =
								prefix_num_verts
								+ luisa::make_uint4(bending_edge[0], bending_edge[1], bending_edge[2], bending_edge[3]);
							mesh_data->sa_dihedral_edge_mesh_id[prefix_num_dihedral_edges + eid] = meshIdx;
						});

					// Read tetrahedrons
					CpuParallel::parallel_for(0,
						curr_num_tets,
						[&](const uint tid)
						{
							auto tet = curr_input_mesh.tetrahedrons[tid];
							mesh_data->sa_tetrahedrons[prefix_num_tets + tid] =
								prefix_num_verts
								+ luisa::make_uint4(tet[0], tet[1], tet[2], tet[3]);
							mesh_data->sa_tet_mesh_id[prefix_num_tets + tid] = meshIdx;
						});

					// Read fixed points
					mesh_data->fixed_verts_map[meshIdx].resize(curr_shell_info.fixed_point_indices.size());
					CpuParallel::single_thread_for(0,
						curr_shell_info.fixed_point_indices.size(),
						[&](const uint index)
						{
							const uint	local_vid = curr_shell_info.fixed_point_indices[index];
							const uint	global_vid = prefix_num_verts + local_vid;
							const float stiffness = curr_shell_info.fixed_point_stiffness[index];
							mesh_data->sa_is_fixed[global_vid] = true;
							mesh_data->sa_fixed_stiffness[global_vid] = stiffness;
							mesh_data->fixed_verts.push_back(global_vid);
							mesh_data->fixed_verts_map[meshIdx][index] = global_vid;
						});
					// Set fixed-points
					{
						AABB local_aabb = CpuParallel::parallel_for_and_reduce_sum<AABB>(
							0,
							curr_num_verts,
							[&](const uint vid)
							{
								auto   read_pos = mesh_data->sa_rest_x[prefix_num_verts + vid];
								float3 pos = luisa::make_float3(read_pos[0], read_pos[1], read_pos[2]);
								return AABB(pos);
							});
						auto pos_min = local_aabb.packed_min;
						auto pos_max = local_aabb.packed_max;

						float avg_spring_length =
							CpuParallel::parallel_for_and_reduce_sum<float>(
								0,
								curr_num_edges,
								[&](const uint eid)
								{
									auto edge = mesh_data->sa_edges[prefix_num_edges + eid];
									return length_vec(mesh_data->sa_rest_x[edge[0]] - mesh_data->sa_rest_x[edge[1]]);
								})
							/ float(curr_num_edges);

						LUISA_INFO("Mesh {:<2} : numVerts = {:<5}, numFaces = {:<5}, numEdges = {:<5}, numTets = {:5} avgEdgeLength = {:2.4f}, AABB range = {}",
							meshIdx,
							curr_num_verts,
							curr_num_faces,
							curr_num_edges,
							curr_num_tets,
							avg_spring_length,
							pos_max - pos_min);
					}

					prefix_num_verts += curr_num_verts;
					prefix_num_faces += curr_num_faces;
					prefix_num_edges += curr_num_edges;
					prefix_num_dihedral_edges += curr_num_dihedral_edges;
					prefix_num_tets += curr_num_tets;
				}
			}

			// Init topology adjacent list
			{
				mesh_data->vert_adj_faces.resize(num_verts);
				mesh_data->vert_adj_edges.resize(num_verts);
				mesh_data->vert_adj_dihedral_edges.resize(num_verts);
				mesh_data->vert_adj_verts.resize(num_verts);
				mesh_data->vert_adj_tets.resize(num_verts);

				// Vert adj faces
				for (uint eid = 0; eid < num_faces; eid++)
				{
					auto edge = mesh_data->sa_faces[eid];
					for (uint j = 0; j < 3; j++)
						mesh_data->vert_adj_faces[edge[j]].push_back(eid);
				}
				upload_2d_csr_from(mesh_data->sa_vert_adj_faces_csr, mesh_data->vert_adj_faces);

				// Vert adj edges
				for (uint eid = 0; eid < num_edges; eid++)
				{
					auto edge = mesh_data->sa_edges[eid];
					for (uint j = 0; j < 2; j++)
						mesh_data->vert_adj_edges[edge[j]].push_back(eid);
				}
				upload_2d_csr_from(mesh_data->sa_vert_adj_edges_csr, mesh_data->vert_adj_edges);

				// Vert adj bending-edges
				for (uint eid = 0; eid < num_dihedral_edges; eid++)
				{
					auto edge = mesh_data->sa_dihedral_edges[eid];
					for (uint j = 0; j < 4; j++)
						mesh_data->vert_adj_dihedral_edges[edge[j]].push_back(eid);
				}
				upload_2d_csr_from(mesh_data->sa_vert_adj_dihedral_edges_csr, mesh_data->vert_adj_dihedral_edges);

				// Vert adj tets
				for (uint tid = 0; tid < num_tets; tid++)
				{
					auto tet = mesh_data->sa_tetrahedrons[tid];
					for (uint j = 0; j < 4; j++)
						mesh_data->vert_adj_tets[tet[j]].push_back(tid);
				}
				upload_2d_csr_from(mesh_data->sa_vert_adj_tets_csr, mesh_data->vert_adj_tets);

				// Vert adj verts based on 1-order connection
				for (uint eid = 0; eid < num_edges; eid++)
				{
					auto edge = mesh_data->sa_edges[eid];
					for (uint j = 0; j < 2; j++)
					{
						const uint left = edge[j];
						const uint right = edge[1 - j];
						mesh_data->vert_adj_verts[left].push_back(right);
					}
				}
				upload_2d_csr_from(mesh_data->sa_vert_adj_verts_csr, mesh_data->vert_adj_verts);

				// Assign topology component labels within each registered object. The
				// labels remain host-side and are reused by PRP on both CPU and GPU.
				const uint invalid_component = std::numeric_limits<uint>::max();
				mesh_data->vert_topology_component_id.assign(num_verts, invalid_component);
				std::vector<uint> component_stack;
				for (uint object_id = 0u; object_id < num_meshes; ++object_id)
				{
					const uint vert_begin = mesh_data->prefix_num_verts[object_id];
					const uint vert_end = mesh_data->prefix_num_verts[object_id + 1u];
					uint	   component_id = 0u;
					for (uint seed_vid = vert_begin; seed_vid < vert_end; ++seed_vid)
					{
						if (mesh_data->vert_topology_component_id[seed_vid] != invalid_component)
							continue;
						if (component_id == invalid_component)
							LUISA_ERROR("Topology component count overflow in registered object {}.", object_id);
						component_stack.clear();
						component_stack.push_back(seed_vid);
						mesh_data->vert_topology_component_id[seed_vid] = component_id;
						while (!component_stack.empty())
						{
							const uint vid = component_stack.back();
							component_stack.pop_back();
							for (const uint adj_vid : mesh_data->vert_adj_verts[vid])
							{
								if (adj_vid < vert_begin || adj_vid >= vert_end)
									LUISA_ERROR(
										"Topology edge ({}, {}) crosses registered object {} vertex range [{}, {}).",
										vid,
										adj_vid,
										object_id,
										vert_begin,
										vert_end);
								if (mesh_data->vert_topology_component_id[adj_vid] != invalid_component)
									continue;
								mesh_data->vert_topology_component_id[adj_vid] = component_id;
								component_stack.push_back(adj_vid);
							}
						}
						++component_id;
					}
				}

				// Vert adj verts based on 1-order bending-connection
				auto insert_adj_vert = [](std::vector<std::vector<uint>>& adj_map, const uint& vid1, const uint& vid2)
				{
					if (vid1 == vid2)
						std::cerr << "redudant!";
					auto& inner_list = adj_map[vid1];
					auto  find_result = std::find(inner_list.begin(), inner_list.end(), vid2);
					if (find_result == inner_list.end())
					{
						inner_list.push_back(vid2);
					}
				};

				// Face adj edges
				mesh_data->face_adj_faces_ext.resize(num_faces);
				mesh_data->edge_adj_faces_ext.resize(num_edges);
				mesh_data->edge_adj_edges_ext.resize(num_edges);

				mesh_data->face_adj_edges.resize(num_faces);
				mesh_data->face_adj_faces.resize(num_faces);
				mesh_data->edge_adj_faces.resize(num_edges, luisa::make_uint2(-1u));
				mesh_data->edge_adj_edges.resize(num_edges, luisa::make_uint4(-1u));

				auto fn_vert_in_face = [](const uint& vid, const uint3& face)
				{
					return vid == face[0] || vid == face[1] || vid == face[2];
				};

				// Face adj edges
				CpuParallel::parallel_for(
					0,
					num_faces,
					[&](const uint fid)
					{
						// Find all adj edges by traversing verts adj edges
						std::unordered_map<uint64_t, uint> adj_edge_map;
						adj_edge_map.reserve(3);
						const auto face = mesh_data->sa_faces[fid];
						auto	   edge_key = [](const uint v0, const uint v1) -> uint64_t
						{
							const auto lo = min_scalar(v0, v1);
							const auto hi = std::max(v0, v1);
							return (uint64_t(lo) << 32u) | uint64_t(hi);
						};
						for (uint j = 0; j < 3; j++)
						{
							const uint	vid = face[j];
							const auto& vert_adj_edges = mesh_data->vert_adj_edges[vid];
							for (const uint& adj_eid : vert_adj_edges)
							{
								const auto adj_edge = mesh_data->sa_edges[adj_eid];
								if (fn_vert_in_face(adj_edge[0], face) && fn_vert_in_face(adj_edge[1], face))
								{
									adj_edge_map.try_emplace(edge_key(adj_edge[0], adj_edge[1]), adj_eid);
								}
							}
						}
						if (adj_edge_map.size() != 3)
							LUISA_ERROR("Face {} adj edge count {} != 3", fid, adj_edge_map.size());
						uint3 face_adj_edges;
						uint  idx = 0;
						for (const auto& [edge_pair, adj_eid] : adj_edge_map)
						{
							face_adj_edges[idx++] = adj_eid;
						}
						if (idx != 3)
							LUISA_ERROR("Face {} adj edge count {} != 3", fid, idx);

						// Sort adj edges by face vertex order
						uint2 edges[3] = { mesh_data->sa_edges[face_adj_edges[0]],
							mesh_data->sa_edges[face_adj_edges[1]],
							mesh_data->sa_edges[face_adj_edges[2]] };
						uint3 face_adj_edges_sorted = face_adj_edges;
						for (uint jj = 0; jj < 3; jj++)
						{
							uint desire_vid1 = face[jj];
							uint desire_vid2 = face[(jj + 1) % 3];
							for (uint kk = 0; kk < 3; kk++)
							{
								uint2 edge = edges[kk];
								if ((edge[0] == desire_vid1 && edge[1] == desire_vid2)
									|| (edge[1] == desire_vid1 && edge[0] == desire_vid2))
								{
									face_adj_edges_sorted[jj] = face_adj_edges[kk];
									break;
								}
								if (kk == 2)
									LUISA_ERROR("Can not find adjacent edge for face {}", fid);
							}
						}
						mesh_data->face_adj_edges[fid] = face_adj_edges_sorted;
					});

				// Edge adj faces
				std::vector<uint> edge_adj_face_count(num_edges, 0);
				for (uint fid = 0; fid < num_faces; fid++)
				{
					uint3 face_adj_edges = mesh_data->face_adj_edges[fid];
					for (uint j = 0; j < 3; j++)
					{
						uint adj_eid = face_adj_edges[j];
						mesh_data->edge_adj_faces_ext[adj_eid].push_back(fid);
						uint& offset = edge_adj_face_count[adj_eid];
						if (offset < 2)
						{
							mesh_data->edge_adj_faces[adj_eid][offset] = fid;
							offset += 1;
						}
					}
				}

				// Edge adj edges
				CpuParallel::parallel_for(0,
					num_edges,
					[&](const uint eid)
					{
						auto get_other_edges_in_face = [&](const uint adj_fid) -> uint2
						{
							uint3 face_edge = mesh_data->face_adj_edges[adj_fid];
							uint2 other_edges =
								face_edge[0] == eid	  ? face_edge.yz() //
								: face_edge[1] == eid ? face_edge.xz() //
								: face_edge[2] == eid ? face_edge.xy() //
													  : luisa::make_uint2(-1u);
							if (other_edges[0] == -1u)
							{
								LUISA_ERROR("Can not find adjacent edge");
							}
							return other_edges;
						};
						std::vector<uint> adj_edges_ext;
						{
							const auto& edge_adj_faces_ext = mesh_data->edge_adj_faces_ext[eid];
							for (const uint& adj_fid : edge_adj_faces_ext)
							{
								uint2 culled_edges = get_other_edges_in_face(adj_fid);
								adj_edges_ext.push_back(culled_edges[0]);
								adj_edges_ext.push_back(culled_edges[1]);
							}
							mesh_data->edge_adj_edges_ext[eid] = adj_edges_ext;
						}
						{
							uint4 adj_edges = luisa::make_uint4(-1u);
							for (uint jj = 0; jj < min_scalar(4ul, adj_edges_ext.size()); jj++)
							{
								adj_edges[jj] = adj_edges_ext[jj];
							}
							mesh_data->edge_adj_edges[eid] = adj_edges;
						}
					});

				// Face adj faces
				CpuParallel::parallel_for(0,
					num_faces,
					[&](const uint fid)
					{
						const uint3		  face_adj_edges = mesh_data->face_adj_edges[fid];
						std::vector<uint> adj_faces_ext;
						{
							for (uint j = 0; j < 3; j++)
							{
								uint		adj_eid = face_adj_edges[j];
								const auto& edge_adj_faces = mesh_data->edge_adj_faces_ext[adj_eid];
								for (const uint& adj_fid : edge_adj_faces)
									if (adj_fid != fid)
										adj_faces_ext.push_back(adj_fid);
							}
							mesh_data->face_adj_faces_ext[fid] = adj_faces_ext;
						}
						{
							uint3 face_adj_faces = luisa::make_uint3(-1u);
							for (uint jj = 0; jj < min_scalar(3ul, adj_faces_ext.size()); jj++)
							{
								face_adj_faces[jj] = adj_faces_ext[jj];
							}
							mesh_data->face_adj_faces[fid] = face_adj_faces;
						}
					});

				// LUISA_INFO("Edge adj faces:");
				// for (uint eid = 0; eid < num_edges; eid++)
				// 	LUISA_INFO("  Edge {} adj faces {} : {}", eid, mesh_data->edge_adj_faces[eid], mesh_data->edge_adj_faces_ext[eid]);
				// LUISA_INFO("Edge adj edges:");
				// for (uint eid = 0; eid < num_edges; eid++)
				// 	LUISA_INFO("  Edge {} adj edges {} : {}", eid, mesh_data->edge_adj_edges[eid], mesh_data->edge_adj_edges_ext[eid]);
				// LUISA_INFO("Face adj faces:");
				// for (uint fid = 0; fid < num_faces; fid++)
				// 	LUISA_INFO("  Face {} adj faces {} : {}", fid, mesh_data->face_adj_faces[fid], mesh_data->face_adj_faces_ext[fid]);

				upload_2d_csr_from(mesh_data->edge_adj_faces_csr, mesh_data->edge_adj_faces_ext);
				upload_2d_csr_from(mesh_data->edge_adj_edges_csr, mesh_data->edge_adj_edges_ext);
				upload_2d_csr_from(mesh_data->face_adj_faces_csr, mesh_data->face_adj_faces_ext);
			}

			// Compute rest area
			{
				mesh_data->sa_rest_vert_area.resize(num_verts);
				mesh_data->sa_rest_edge_area.resize(num_edges);
				mesh_data->sa_rest_face_area.resize(num_faces);
				mesh_data->sa_rest_tet_volume.resize(num_tets);
				mesh_data->sa_rest_vert_volume.resize(num_verts);
				mesh_data->sa_vert_thickness.resize(num_verts);
				mesh_data->sa_edge_thickness.resize(num_edges);
				mesh_data->sa_face_thickness.resize(num_faces);

				constexpr float min_area = 1e-7f;
				constexpr float min_volume = 1e-8f;

				std::atomic_uint num_small_area_faces = 0;
				std::atomic_uint num_small_volume_tets = 0;

				auto safe_area = [min_area, &num_small_area_faces](auto& area)
				{
					if (area < min_area)
					{
						// LUISA_WARNING("Small face area {}, set to {}", area, min_area);
						num_small_area_faces.fetch_add(1);
						area = min_area;
					}
				};
				auto safe_volume = [min_volume, &num_small_volume_tets](float& volume)
				{
					if (volume < min_volume)
					{
						// LUISA_WARNING("Small tet volume {}, set to {}", volume, min_volume);
						num_small_volume_tets.fetch_add(1);
						volume = min_volume;
					}
				};

				CpuParallel::parallel_for(0,
					num_faces,
					[&](const uint fid)
					{
						const uint3 face = mesh_data->sa_faces[fid];
						float		area = compute_face_area(mesh_data->sa_rest_x[face[0]],
							mesh_data->sa_rest_x[face[1]],
							mesh_data->sa_rest_x[face[2]]);
						safe_area(area);
						mesh_data->sa_rest_face_area[fid] = area;

						const uint mesh_idx = mesh_data->sa_face_mesh_id[fid];
						mesh_data->sa_face_thickness[fid] = world_data[mesh_idx].get_thickness();
					});
				CpuParallel::parallel_for(0,
					num_tets,
					[&](const uint tid)
					{
						const uint4 tet = mesh_data->sa_tetrahedrons[tid];
						float		volume = compute_tet_volume(mesh_data->sa_rest_x[tet[0]],
							mesh_data->sa_rest_x[tet[1]],
							mesh_data->sa_rest_x[tet[2]],
							mesh_data->sa_rest_x[tet[3]]);
						safe_volume(volume);
						mesh_data->sa_rest_tet_volume[tid] = volume;
					});

				CpuParallel::parallel_for(0,
					num_verts,
					[&](const uint vid)
					{
						const auto& adj_faces = mesh_data->vert_adj_faces[vid];
						double		area = 0.0;
						if (adj_faces.size() == 0)
						{
							LUISA_ERROR("Vertex {} has no adjacent face!", vid);
						}
						for (const uint& adj_fid : adj_faces)
							area += mesh_data->sa_rest_face_area[adj_fid] / 3.0;

						safe_area(area);
						mesh_data->sa_rest_vert_area[vid] = area;

						const uint	mesh_idx = mesh_data->sa_vert_mesh_id[vid];
						const auto& shell_info = world_data[mesh_idx];
						mesh_data->sa_vert_thickness[vid] = shell_info.get_thickness();

						const auto& adj_tets = mesh_data->vert_adj_tets[vid];
						if (shell_info.get_is_shell() || adj_tets.empty())
						{
							mesh_data->sa_rest_vert_volume[vid] =
								area * shell_info.get_thickness();
						}
						else
						{
							double volume = 0.0;
							for (const uint& adj_tid : adj_tets)
								volume += mesh_data->sa_rest_tet_volume[adj_tid] / 4.0;
							mesh_data->sa_rest_vert_volume[vid] = volume;
						}
					});
				CpuParallel::parallel_for(0,
					num_edges,
					[&](const uint eid)
					{
						const auto& adj_faces = mesh_data->edge_adj_faces_ext[eid];
						double		area = 0.0;
						for (const uint& adj_fid : adj_faces)
						{
							area += mesh_data->sa_rest_face_area[adj_fid] / 3.0;
						}
						safe_area(area);
						mesh_data->sa_rest_edge_area[eid] = area;

						const uint mesh_idx = mesh_data->sa_edge_mesh_id[eid];
						mesh_data->sa_edge_thickness[eid] = world_data[mesh_idx].get_thickness();
					});

				if (num_small_area_faces > 0)
					LUISA_WARNING("{} faces have small area < {}, set to {:.3e}", num_small_area_faces.load(), min_area, min_area);
				if (num_small_volume_tets > 0)
					LUISA_WARNING("{} tets have small volume < {}, set to {:.3e}", num_small_volume_tets.load(), min_volume, min_volume);

				// float sum_face_area = CpuParallel::parallel_reduce_sum(mesh_data->sa_rest_face_area);
				// float sum_edge_area = CpuParallel::parallel_reduce_sum(mesh_data->sa_rest_edge_area);
				// float sum_vert_area = CpuParallel::parallel_reduce_sum(mesh_data->sa_rest_vert_area);
				// // LUISA_INFO("Summary areas : face = {}, edge = {}, vert = {}", sum_face_area, sum_edge_area, sum_vert_area);
				// LUISA_INFO("Average areas : face = {}, edge = {}, vert = {}",
				//            sum_face_area / double(num_faces),
				//            sum_edge_area / double(num_edges),
				//            sum_vert_area / double(num_verts));
			}

			// Init mass info
			{
				mesh_data->sa_body_mass.resize(num_meshes);
				mesh_data->sa_rest_body_volume.resize(num_meshes, 0.0f);
				mesh_data->sa_rest_body_area.resize(num_meshes, 0.0f);
				for (uint meshIdx = 0; meshIdx < num_meshes; meshIdx++)
				{
					const auto& shell_info = world_data[meshIdx];

					float sum_volume = 0.0f;
					if (shell_info.get_is_shell()) // Shell volume = area * thickness
					{
						sum_volume = CpuParallel::parallel_for_and_reduce_sum<float>(
							0,
							mesh_data->prefix_num_faces[meshIdx + 1] - mesh_data->prefix_num_faces[meshIdx],
							[&](const uint fid)
							{
								float face_area =
									mesh_data->sa_rest_face_area[mesh_data->prefix_num_faces[meshIdx] + fid];
								return face_area * shell_info.get_thickness();
							});
					}
					else // For solid body, compute volume from tets or faces integration
					{
						if (shell_info.input_mesh.tetrahedrons.empty())
						{
							if (shell_info.material_type == Material::MaterialType::Tetrahedral)
							{
								LUISA_ERROR("Mesh {} is set as Tetrahedral type but has no tetrahedron elements!", meshIdx);
							}
							else // Use face integration
							{
								sum_volume = CpuParallel::parallel_for_and_reduce_sum<float>(
									0,
									mesh_data->prefix_num_faces[meshIdx + 1] - mesh_data->prefix_num_faces[meshIdx],
									[&](const uint fid)
									{
										uint3  face = mesh_data->sa_faces[mesh_data->prefix_num_faces[meshIdx] + fid];
										float3 v0 = mesh_data->sa_rest_x[face[0]];
										float3 v1 = mesh_data->sa_rest_x[face[1]];
										float3 v2 = mesh_data->sa_rest_x[face[2]];

										float3 e1 = v1 - v0;
										float3 e2 = v2 - v0;
										float3 N = cross(e1, e2);
										return luisa::dot(v0, N) / 6.0f;
									});
							}
						}
						else
						{
							sum_volume = CpuParallel::parallel_for_and_reduce_sum<float>(
								0,
								mesh_data->prefix_num_tets[meshIdx + 1] - mesh_data->prefix_num_tets[meshIdx],
								[&](const uint tid)
								{
									return mesh_data->sa_rest_tet_volume[mesh_data->prefix_num_tets[meshIdx] + tid];
								});
						}
					}
					mesh_data->sa_rest_body_volume[meshIdx] = sum_volume;

					float sum_surface_area = CpuParallel::parallel_for_and_reduce_sum<float>(
						0,
						mesh_data->prefix_num_faces[meshIdx + 1] - mesh_data->prefix_num_faces[meshIdx],
						[&](const uint fid)
						{ return mesh_data->sa_rest_face_area[mesh_data->prefix_num_faces[meshIdx] + fid]; });
					mesh_data->sa_rest_body_area[meshIdx] = sum_surface_area;

					const float input_mass = shell_info.get_mass();
					const float input_density = shell_info.get_density();
					mesh_data->sa_body_mass[meshIdx] = input_mass != 0.0f ? input_mass : sum_volume * input_density;

					LUISA_INFO("Mesh {}'s volume = {}{}, body mass = {}, avg vert mass = {}",
						meshIdx,
						sum_volume,
						shell_info.get_is_shell() ? luisa::format(", surface area = {}", sum_surface_area) : "",
						mesh_data->sa_body_mass[meshIdx],
						mesh_data->sa_body_mass[meshIdx]
							/ float(mesh_data->prefix_num_verts[meshIdx + 1] - mesh_data->prefix_num_verts[meshIdx]));
				}
				// {
				//     uint  prefix_num_faces = mesh_data->prefix_num_faces[meshIdx];
				//     uint  curr_num_faces   = mesh_data->prefix_num_faces[meshIdx + 1] - prefix_num_faces;
				//     float mesh_area        = CpuParallel::parallel_for_and_reduce_sum<float>(
				//         0,
				//         curr_num_faces,
				//         [&](const uint fid) { return mesh_data->sa_rest_face_area[prefix_num_faces + fid]; });
				//     body_areas[meshIdx]              = mesh_area;
				//     mesh_data->sa_body_mass[meshIdx] = shell_infos[meshIdx].mass != 0.0f ?
				//                                            shell_infos[meshIdx].mass :
				//                                            mesh_area * shell_infos[meshIdx].density;
				//     LUISA_INFO("Mesh {}'s area = {}, total mass = {}", meshIdx, mesh_area, mesh_data->sa_body_mass[meshIdx]);
				// }

				// Set vert mass
				mesh_data->sa_vert_mass.resize(num_verts);
				mesh_data->sa_vert_mass_inv.resize(num_verts);

				CpuParallel::parallel_for(0,
					num_verts,
					[&](const uint vid)
					{
						bool		is_fixed = mesh_data->sa_is_fixed[vid] != 0;
						const uint	mesh_id = mesh_data->sa_vert_mesh_id[vid];
						const float vert_volume = mesh_data->sa_rest_vert_volume[vid];
						const float body_volume = mesh_data->sa_rest_body_volume[mesh_id];
						const float weight = vert_volume / body_volume;
						//   const float vert_area = mesh_data->sa_rest_vert_area[vid];
						//   const float mesh_area = body_areas[mesh_id];
						//   const float weight    = vert_area / mesh_area;
						float mass = weight * mesh_data->sa_body_mass[mesh_id];
						if (mass < 1e-7f)
						{
							LUISA_WARNING("Vert {} in mesh {} has very small mass {}, volume = {}, weight = {}", vid, mesh_id, mass, vert_volume, weight);
						}
						mass = std::max(mass, 1e-7f); // Avoid zero mass for numerical stability
						mesh_data->sa_vert_mass[vid] = mass;
						mesh_data->sa_vert_mass_inv[vid] = is_fixed ? 0.0f : 1.0f / (mass);
					});
			}

			// // Init vert status
			// {
			//     mesh_data->sa_x_frame_outer.resize(num_verts);
			//     mesh_data->sa_v_frame_outer.resize(num_verts);
			//     CpuParallel::parallel_for(0,
			//                               num_verts,
			//                               [&](const uint vid)
			//                               {
			//                                   const float3 rest_x = mesh_data->sa_rest_x[vid];
			//                                   const float3 rest_v = mesh_data->sa_rest_v[vid];
			//                                   mesh_data->sa_x_frame_outer[vid] = rest_x;
			//                                   mesh_data->sa_v_frame_outer[vid] = rest_v;
			//                               });
			// }
		}

		void upload_mesh_buffers(luisa::compute::Device& device,
			luisa::compute::Stream&						 stream,
			lcs::MeshData<std::vector>*					 input_data,
			lcs::MeshData<luisa::compute::Buffer>*		 output_data)
		{
			output_data->num_meshes = input_data->num_meshes;
			output_data->num_verts = input_data->num_verts;
			output_data->num_faces = input_data->num_faces;
			output_data->num_edges = input_data->num_edges;
			output_data->num_dihedral_edges = input_data->num_dihedral_edges;
			output_data->num_tets = input_data->num_tets;

			stream << upload_buffer(device, output_data->sa_rest_translate, input_data->sa_rest_translate)
				   << upload_buffer(device, output_data->sa_rest_rotation, input_data->sa_rest_rotation)
				   << upload_buffer(device, output_data->sa_rest_scale, input_data->sa_rest_scale)
				   << upload_buffer(device, output_data->sa_model_x, input_data->sa_model_x)
				   << upload_buffer(device, output_data->sa_material_x, input_data->sa_material_x)
				   << upload_buffer(device, output_data->sa_scaled_model_x, input_data->sa_scaled_model_x)
				   << upload_buffer(device, output_data->sa_rest_x, input_data->sa_rest_x)
				   << upload_buffer(device, output_data->sa_rest_v, input_data->sa_rest_v)
				   << upload_buffer(device, output_data->sa_faces, input_data->sa_faces)
				   << upload_buffer(device, output_data->sa_edges, input_data->sa_edges)
				   << upload_buffer(device, output_data->sa_mesh_orientation, input_data->sa_mesh_orientation);

			if (input_data->num_dihedral_edges > 0)
				stream << upload_buffer(device, output_data->sa_dihedral_edges, input_data->sa_dihedral_edges)
					   << upload_buffer(device, output_data->sa_dihedral_edge_mesh_id, input_data->sa_dihedral_edge_mesh_id);

			if (!input_data->sa_tetrahedrons.empty())
				stream << upload_buffer(device, output_data->sa_tetrahedrons, input_data->sa_tetrahedrons)
					   << upload_buffer(device, output_data->sa_tet_mesh_id, input_data->sa_tet_mesh_id)
					   << upload_buffer(device, output_data->sa_rest_tet_volume, input_data->sa_rest_tet_volume);

			// TODO: We may not have face
			stream
				<< upload_buffer(device, output_data->sa_body_mass, input_data->sa_body_mass)
				<< upload_buffer(device, output_data->sa_vert_mass, input_data->sa_vert_mass)
				<< upload_buffer(device, output_data->sa_vert_mass_inv, input_data->sa_vert_mass_inv)
				<< upload_buffer(device, output_data->sa_is_fixed, input_data->sa_is_fixed)
				<< upload_buffer(device, output_data->sa_fixed_stiffness, input_data->sa_fixed_stiffness)
				<< upload_buffer(device, output_data->sa_vert_mesh_id, input_data->sa_vert_mesh_id)
				<< upload_buffer(device, output_data->sa_edge_mesh_id, input_data->sa_edge_mesh_id)
				<< upload_buffer(device, output_data->sa_face_mesh_id, input_data->sa_face_mesh_id)
				<< upload_buffer(device, output_data->sa_vert_mesh_type, input_data->sa_vert_mesh_type)
				<< upload_buffer(device, output_data->sa_global_vid_to_local_vid, input_data->sa_global_vid_to_local_vid)

				<< upload_buffer(device, output_data->sa_rest_body_area, input_data->sa_rest_body_area)
				<< upload_buffer(device, output_data->sa_rest_body_volume, input_data->sa_rest_body_volume)

				<< upload_buffer(device, output_data->sa_rest_vert_area, input_data->sa_rest_vert_area)
				<< upload_buffer(device, output_data->sa_rest_edge_area, input_data->sa_rest_edge_area)
				<< upload_buffer(device, output_data->sa_rest_face_area, input_data->sa_rest_face_area)

				<< upload_buffer(device, output_data->sa_rest_vert_volume, input_data->sa_rest_vert_volume)
				<< upload_buffer(device, output_data->sa_vert_thickness, input_data->sa_vert_thickness)
				<< upload_buffer(device, output_data->sa_edge_thickness, input_data->sa_edge_thickness)
				<< upload_buffer(device, output_data->sa_face_thickness, input_data->sa_face_thickness)

				// No std::vector<std::vector<uint>> vert_adj_verts info
				<< upload_buffer(device, output_data->sa_vert_adj_verts_csr, input_data->sa_vert_adj_verts_csr)
				<< upload_buffer(device, output_data->sa_vert_adj_faces_csr, input_data->sa_vert_adj_faces_csr)
				<< upload_buffer(device, output_data->sa_vert_adj_edges_csr, input_data->sa_vert_adj_edges_csr)
				<< upload_buffer(device, output_data->sa_vert_adj_dihedral_edges_csr, input_data->sa_vert_adj_dihedral_edges_csr)
				<< upload_buffer(device, output_data->sa_vert_adj_tets_csr, input_data->sa_vert_adj_tets_csr)
				<< upload_buffer(device, output_data->edge_adj_faces, input_data->edge_adj_faces)
				<< upload_buffer(device, output_data->edge_adj_edges, input_data->edge_adj_edges)
				<< upload_buffer(device, output_data->face_adj_edges, input_data->face_adj_edges)
				<< upload_buffer(device, output_data->face_adj_faces, input_data->face_adj_faces)
				<< upload_buffer(device, output_data->edge_adj_faces_csr, input_data->edge_adj_faces_csr)
				<< upload_buffer(device, output_data->edge_adj_edges_csr, input_data->edge_adj_edges_csr)
				<< upload_buffer(device, output_data->face_adj_faces_csr, input_data->face_adj_faces_csr)
				<< luisa::compute::synchronize();
		}

	} // namespace Initializer

} // namespace lcs
