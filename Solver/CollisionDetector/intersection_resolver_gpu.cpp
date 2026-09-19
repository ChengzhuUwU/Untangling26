#include "intersection_resolver_gpu.h"
#include "Core/float_n.h"
#include "SimulationCore/scene_params.h"
#include "SimulationCore/physical_material.h"
#include "Utils/color_utils.h"
#include "Utils/cpu_parallel.h"
#include "Utils/reduce_helper.h"
#include "luisa/core/basic_types.h"
#include "luisa/core/logging.h"
#include "luisa/core/mathematics.h"
#include "luisa/core/spin_mutex.h"
#include <CollisionDetector/intersection_resolver2.h>
#include <CollisionDetector/intersection_resolver_helper.h>
#include <CollisionDetector/cached_bvh.h>
#include <algorithm>
#include <bit>
#include <cstdint>
#include <Eigen/Sparse>
#include <Eigen/SparseCholesky>
#include <Eigen/IterativeLinearSolvers>
#include <array>
#include <atomic>
#include <cmath>
#include <iostream>
#include <limits>
#include <queue>

#include <set>
#include <unordered_map>
#include <unordered_set>
#include <mutex>
#include <numeric>
#include <string>
#include <vector>

namespace lcs
{
	// Unlimited target-hop locality, matching the established default on both paths.
	constexpr uint kPrpLocalityMaxHop = std::numeric_limits<uint>::max();

	// Temporary [DET] determinism-trace helpers shared with the CPU resolver
	// (bisecting CPU/GPU PRP evaluation divergence); gated on prp_debug.
	namespace det_trace_detail
	{
		inline void hash_bytes(uint64_t& h, const void* data, size_t n)
		{
			const auto* p = static_cast<const unsigned char*>(data);
			for (size_t i = 0; i < n; i++)
			{
				h ^= p[i];
				h *= 1099511628211ull;
			}
		}
		inline void hash_u64(uint64_t& h, uint64_t v)
		{
			hash_bytes(h, &v, 8);
		}
		template <typename T>
		void hash_pod_vector(uint64_t& h, const std::vector<T>& v)
		{
			hash_u64(h, static_cast<uint64_t>(v.size()));
			if constexpr (std::is_trivially_copyable_v<T>)
				if (!v.empty())
					hash_bytes(h, v.data(), v.size() * sizeof(T));
		}
		inline void trace(const char* stage, uint64_t h)
		{
			if (get_scene_params().prp_debug)
				LUISA_INFO("[DET] {} {:016x}", stage, h);
		}
	} // namespace det_trace_detail

	namespace
	{
		inline void prp_debug_reset()
		{
			if (!get_scene_params().prp_debug)
				return;
			auto& dbg = get_scene_params().prp_debug_info;
			dbg.bool_stats.clear();
			dbg.uint_stats.clear();
			dbg.float_stats.clear();
		}

		inline void prp_debug_set_uint(const std::string& key, const uint value)
		{
			if (!get_scene_params().prp_debug)
				return;
			get_scene_params().prp_debug_info.uint_stats[key] = value;
		}

		inline void prp_debug_set_float(const std::string& key, const float value)
		{
			if (!get_scene_params().prp_debug)
				return;
			get_scene_params().prp_debug_info.float_stats[key] = value;
		}

		inline void prp_debug_set_meta_keys(const uint num_verts,
			const uint								   num_edges,
			const uint								   num_faces,
			const uint								   num_pairs,
			const uint								   num_contours,
			const uint								   classified_contour_count,
			const uint								   num_loop_pairs)
		{
			if (get_scene_params().prp_debug || get_scene_params().collect_iteration_debug)
			{
				auto& dbg = get_scene_params().prp_debug_info;
				dbg.uint_stats["num_verts"] = num_verts;
				dbg.uint_stats["num_edges"] = num_edges;
				dbg.uint_stats["num_faces"] = num_faces;
				dbg.uint_stats["num_pairs"] = num_pairs;
				dbg.uint_stats["num_contours"] = num_contours;
				dbg.uint_stats["classified_contour_count"] = classified_contour_count;
				dbg.uint_stats["num_loop_pairs"] = num_loop_pairs;

				// IterationDebugRecord-compatible aliases.
				dbg.uint_stats["contour_count"] = num_contours;
				dbg.uint_stats["ef_pair_count"] = num_pairs;
				dbg.uint_stats["loop_vertex_count"] = num_loop_pairs;
				dbg.bool_stats["all_contours_classifiable_into_7_types"] = (classified_contour_count == num_contours);
			}
		}

	} // namespace

	// DevicePRPPrecomputeShaders is instance-owned by IntersectionResolver;
	// its kernels are compiled once per instance via the AsyncCompiler interface.

	void DevicePRPPrecomputeShaders::compile_normals(AsyncCompiler& compiler)
	{
		using namespace luisa::compute;

		compiler.compile<1>(fn_compute_face_normals,
			[](BufferVar<uint3>	  sa_faces,
				BufferVar<float3> sa_x,
				BufferVar<uint>	  sa_face_mesh_id,
				BufferVar<uint>	  sa_mesh_orientation,
				BufferVar<float3> face_normal)
			{
				const Uint	 fid = dispatch_x();
				const Uint3	 face = sa_faces.read(fid);
				const Float3 v0 = sa_x.read(face.x);
				const Float3 v1 = sa_x.read(face.y);
				const Float3 v2 = sa_x.read(face.z);
				Float3		 normal = cross(v1 - v0, v2 - v0);
				const Float	 normal_len2 = dot(normal, normal);
				$if(normal_len2 > 1e-20f)
				{
					normal *= rsqrt(normal_len2);
				}
				$else
				{
					normal = make_float3(0.0f);
				};

				const Uint mesh_id = sa_face_mesh_id.read(fid);
				Float	   orient = 1.0f;
				$if(sa_mesh_orientation.read(mesh_id) != 0u)
				{
					orient = -1.0f;
				};
				face_normal.write(fid, orient * normal);
			});

		compiler.compile<1>(fn_compute_edge_normals,
			[](BufferVar<uint>	  edge_adj_faces_csr,
				BufferVar<float>  sa_rest_face_area,
				BufferVar<float3> face_normal,
				BufferVar<float3> edge_normal)
			{
				const Uint eid = dispatch_x();
				const Uint start = edge_adj_faces_csr.read(eid);
				const Uint end = edge_adj_faces_csr.read(eid + 1u);
				const Uint count = end - start;
				Float3	   normal_sum = make_float3(0.0f);
				$for(ii, count)
				{
					const Uint fid = edge_adj_faces_csr.read(start + ii);
					normal_sum += sa_rest_face_area.read(fid) * face_normal.read(fid);
				};
				const Float normal_len2 = dot(normal_sum, normal_sum);
				$if(normal_len2 > 1e-20f)
				{
					normal_sum *= rsqrt(normal_len2);
				}
				$else
				{
					normal_sum = make_float3(0.0f);
				};
				edge_normal.write(eid, normal_sum);
			});

		compiler.compile<1>(fn_compute_vert_normals,
			[](BufferVar<uint>	  sa_vert_adj_faces_csr,
				BufferVar<float>  sa_rest_face_area,
				BufferVar<float3> face_normal,
				BufferVar<float3> vert_normal)
			{
				const Uint vid = dispatch_x();
				const Uint start = sa_vert_adj_faces_csr.read(vid);
				const Uint end = sa_vert_adj_faces_csr.read(vid + 1u);
				const Uint count = end - start;
				Float3	   normal_sum = make_float3(0.0f);
				$for(ii, count)
				{
					const Uint fid = sa_vert_adj_faces_csr.read(start + ii);
					normal_sum += sa_rest_face_area.read(fid) * face_normal.read(fid);
				};
				const Float normal_len2 = dot(normal_sum, normal_sum);
				$if(normal_len2 > 1e-20f)
				{
					normal_sum *= rsqrt(normal_len2);
				}
				$else
				{
					normal_sum = make_float3(0.0f);
				};
				vert_normal.write(vid, normal_sum);
			});
	}
	void DevicePRPPrecomputeShaders::compile_buffer_utils(AsyncCompiler& compiler)
	{
		using namespace luisa::compute;

		compiler.compile<1>(fn_reset_uint,
			[](BufferVar<uint> buffer)
			{
				const Uint pair_idx = dispatch_x();
				buffer.write(pair_idx, 0u);
			});

		compiler.compile<1>(fn_reset_uint_with_value,
			[](BufferVar<uint> buffer, Var<uint> value)
			{
				const Uint pair_idx = dispatch_x();
				buffer.write(pair_idx, value);
			});
		compiler.compile<1>(fn_copy_uint_buffer,
			[](BufferVar<uint> src, BufferVar<uint> dst, Var<uint> count)
			{
				const Uint idx = dispatch_x();
				$if(idx < count)
				{
					dst.write(idx, src.read(idx));
				};
			});

		// ─────────────────────────────────────────────

		compiler.compile<1>(fn_exclusive_prefix_sum,
			[](BufferVar<uint>	buffer_count,
				BufferVar<uint> buffer_prefix,
				BufferVar<uint> total_count,
				Var<uint>		offset)
			{
				const Uint					 vid = dispatch_x();
				Uint						 vert_count = buffer_count.read(vid);
				Uint						 block_sum = 0;
				Uint						 block_offset = ParallelIntrinsic::block_intrinsic_scan_exclusive<uint>(vid, vert_count, block_sum);
				luisa::compute::Shared<uint> block_prefix(1);
				$if(vid % 256 == 0)
				{
					block_prefix[0] = total_count->atomic(offset).fetch_add(block_sum);
				};
				luisa::compute::sync_block();
				const Uint global_index = block_prefix[0] + block_offset;
				buffer_prefix->write(vid, global_index);
			});
	}
	void DevicePRPPrecomputeShaders::compile_hit_compaction(AsyncCompiler& compiler)
	{
		using namespace luisa::compute;

		compiler.compile<1>(fn_prp_compact_valid_hits_with_task,
			[](BufferVar<uint2>	  in_hits,
				BufferVar<float4> in_attrs,
				BufferVar<uint>	  in_task_id,
				BufferVar<uint>	  hit_valid,
				BufferVar<uint2>  out_hits,
				BufferVar<float4> out_attrs,
				BufferVar<uint>	  out_task_id,
				BufferVar<uint>	  out_count,
				Var<uint>		  total_hits)
			{
				const Uint hit_idx = dispatch_x();
				$if(hit_idx >= total_hits)
				{
					$return();
				};
				$if(hit_valid.read(hit_idx) != 0u)
				{
					const Uint compact_idx = out_count.atomic(0u).fetch_add(1u);
					out_hits.write(compact_idx, in_hits.read(hit_idx));
					out_attrs.write(compact_idx, in_attrs.read(hit_idx));
					out_task_id.write(compact_idx, in_task_id.read(hit_idx));
				};
			});

		compiler.compile<1>(fn_prp_reset_device_combo_eval,
			[](BufferVar<PRPDeviceComboSummary> combo_summary,
				Var<uint>						task_count)
			{
				const Uint task_idx = dispatch_x();
				$if(task_idx >= task_count)
				{
					$return();
				};
				auto summary = combo_summary.read(task_idx);
				summary->counts0 = make_uint4(0u);
				summary->eval0 = make_float4(0.0f);
				summary->eval1 = make_float4(0.0f);
				summary->frontier0 = make_uint4(0xffffffffu);
				summary->frontier1 = make_uint4(0u);
				combo_summary.write(task_idx, summary);
			});

		compiler.compile<1>(fn_prp_accumulate_combo_summary,
			[](BufferVar<uint2>	  packed_hits,
				BufferVar<float4> packed_attrs,
				BufferVar<uint>	  packed_task_id,
				BufferVar<uint>	  hit_valid,
				BufferVar<float3> task_dirs,

				BufferVar<uint> task_boundary_hop_dense,

				BufferVar<uint3>				 sa_faces,
				BufferVar<uint2>				 sa_edges,
				BufferVar<float3>				 sa_x,
				BufferVar<float>				 sa_rest_vert_area,
				BufferVar<float>				 sa_rest_edge_area,
				BufferVar<float>				 sa_rest_face_area,
				BufferVar<float>				 sa_vert_mass,
				BufferVar<float3>				 vert_normal,
				BufferVar<float3>				 edge_normal,
				BufferVar<float3>				 face_normal,
				BufferVar<PRPDeviceComboSummary> combo_summary,
				Var<uint>						 num_verts,
				Var<uint>						 total_hits)
			{
				const Uint hit_idx = dispatch_x();
				$if(hit_idx >= total_hits)
				{
					$return();
				};
				$if(hit_valid.read(hit_idx) == 0u)
				{
					$return();
				};

				constexpr uint extract_mask = HitIndices::extract_mask;
				const Uint2	   ids = packed_hits.read(hit_idx);
				const Float4   attrs = packed_attrs.read(hit_idx);
				const Uint	   type = ids.x >> 30u;
				const Uint	   id1 = ids.x & extract_mask;
				const Uint	   id2 = ids.y & extract_mask;
				const Uint	   task_idx = packed_task_id.read(hit_idx);
				const Float3   dir = task_dirs.read(task_idx);
				const Float	   dist = attrs.z;
				const Float	   shifted_dist = dist + prp_response_depth_offset;

				auto dense_hop = [&](const Uint side, const Uint vid) -> Uint
				{
					return task_boundary_hop_dense.read((task_idx * 2u + side) * num_verts + vid);
				};
				auto safe_inverse_mass = [&](const Uint vid) -> Float
				{
					return 1.0f / max(sa_vert_mass.read(vid), 1e-8f);
				};

				Float  area = 0.0f;
				Float  alpha = 0.0f;
				Float  effective_inv_mass = 1.0f;
				Float3 char_n = make_float3(0.0f, 1.0f, 0.0f);
				$if(type == 0u)
				{
					const Uint	 vid = id1;
					const Uint	 fid = id2;
					const UInt3	 face = sa_faces.read(fid);
					const Float	 u = attrs.x;
					const Float	 v = attrs.y;
					const Float3 bary = make_float3(1.0f - u - v, u, v);
					area = sa_rest_vert_area.read(vid);
					alpha = abs(dot(vert_normal.read(vid), dir));
					effective_inv_mass = safe_inverse_mass(vid)
						+ bary.x * bary.x * safe_inverse_mass(face.x)
						+ bary.y * bary.y * safe_inverse_mass(face.y)
						+ bary.z * bary.z * safe_inverse_mass(face.z);
					const Float3 A = sa_x.read(face.x);
					const Float3 B = sa_x.read(face.y);
					const Float3 C = sa_x.read(face.z);
					char_n = cross(B - A, C - A);
				}
				$elif(type == 1u)
				{
					const Uint	 eid1 = id1;
					const Uint	 eid2 = id2;
					const UInt2	 edge1 = sa_edges.read(eid1);
					const UInt2	 edge2 = sa_edges.read(eid2);
					const Float	 t1 = attrs.x;
					const Float	 t2 = attrs.y;
					const Float2 bary1 = make_float2(1.0f - t1, t1);
					const Float2 bary2 = make_float2(1.0f - t2, t2);
					area = sa_rest_edge_area.read(eid1);
					alpha = abs(dot(edge_normal.read(eid1), dir));
					effective_inv_mass = bary1.x * bary1.x * safe_inverse_mass(edge1.x)
						+ bary1.y * bary1.y * safe_inverse_mass(edge1.y)
						+ bary2.x * bary2.x * safe_inverse_mass(edge2.x)
						+ bary2.y * bary2.y * safe_inverse_mass(edge2.y);
					const Float3 p0 = sa_x.read(edge1.x);
					const Float3 p1 = sa_x.read(edge1.y);
					const Float3 q0 = sa_x.read(edge2.x);
					const Float3 q1 = sa_x.read(edge2.y);
					char_n = cross(p1 - p0, q1 - q0);
					$if(dot(char_n, char_n) < 1e-16f)
					{
						const Float3 pt1 = p0 * bary1.x + p1 * bary1.y;
						const Float3 pt2 = q0 * bary2.x + q1 * bary2.y;
						char_n = pt2 - pt1;
					};
				}
				$else
				{
					const Uint	 fid = id1;
					const Uint	 vid = id2;
					const UInt3	 face = sa_faces.read(fid);
					const Float	 u = attrs.x;
					const Float	 v = attrs.y;
					const Float3 bary = make_float3(1.0f - u - v, u, v);
					area = sa_rest_face_area.read(fid);
					alpha = abs(dot(face_normal.read(fid), dir));
					effective_inv_mass = bary.x * bary.x * safe_inverse_mass(face.x)
						+ bary.y * bary.y * safe_inverse_mass(face.y)
						+ bary.z * bary.z * safe_inverse_mass(face.z)
						+ safe_inverse_mass(vid);
					const Float3 A = sa_x.read(face.x);
					const Float3 B = sa_x.read(face.y);
					const Float3 C = sa_x.read(face.z);
					char_n = cross(B - A, C - A);
				};

				const Float volume = area * alpha * dist;
				const Float objective_area = area;
				const Float displacement_area = area * shifted_dist * shifted_dist;
				const Float displacement_kappa = ite(effective_inv_mass > 1e-8f, shifted_dist * shifted_dist / effective_inv_mass, 0.0f);
				const Float displacement_cost = displacement_kappa;

				auto summary = combo_summary.atomic(task_idx);
				summary.counts0.x.fetch_add(1u);
				$if(type == 0u)
				{
					summary.counts0.y.fetch_add(1u);
				}
				$elif(type == 1u)
				{
					summary.counts0.z.fetch_add(1u);
				}
				$else
				{
					summary.counts0.w.fetch_add(1u);
				};
				summary.eval0.x.fetch_add(volume);
				summary.eval0.y.fetch_max(dist);
				summary.eval0.z.fetch_add(objective_area);
				summary.eval0.w.fetch_add(area);
				summary.eval1.x.fetch_add(displacement_area);
				summary.eval1.y.fetch_add(displacement_kappa);
				summary.eval1.z.fetch_add(displacement_cost);

				auto update_source_frontier = [&](const Uint vid)
				{
					const Uint invalid = 0xffffffffu;
					const Uint hop_to_curr = dense_hop(0u, vid);
					const Uint hop_to_oppo = dense_hop(1u, vid);
					$if(hop_to_curr != invalid)
					{
						summary.frontier0.x.fetch_min(hop_to_curr);
						summary.frontier1.x.fetch_max(hop_to_curr);
					};
					$if(hop_to_oppo != invalid)
					{
						summary.frontier0.z.fetch_min(hop_to_oppo);
						summary.frontier1.z.fetch_max(hop_to_oppo);
					};
				};
				auto update_target_frontier = [&](const Uint vid)
				{
					const Uint invalid = 0xffffffffu;
					const Uint hop_to_curr = dense_hop(1u, vid);
					const Uint hop_to_oppo = dense_hop(0u, vid);
					$if(hop_to_curr != invalid)
					{
						summary.frontier0.y.fetch_min(hop_to_curr);
						summary.frontier1.y.fetch_max(hop_to_curr);
					};
					$if(hop_to_oppo != invalid)
					{
						summary.frontier0.w.fetch_min(hop_to_oppo);
						summary.frontier1.w.fetch_max(hop_to_oppo);
					};
				};
				$if(type == 0u)
				{
					const UInt3 face = sa_faces.read(id2);
					update_source_frontier(id1);
					update_target_frontier(face.x);
					update_target_frontier(face.y);
					update_target_frontier(face.z);
				}
				$elif(type == 1u)
				{
					const UInt2 edge1 = sa_edges.read(id1);
					const UInt2 edge2 = sa_edges.read(id2);
					update_source_frontier(edge1.x);
					update_source_frontier(edge1.y);
					update_target_frontier(edge2.x);
					update_target_frontier(edge2.y);
				}
				$else
				{
					const UInt3 face = sa_faces.read(id1);
					update_source_frontier(face.x);
					update_source_frontier(face.y);
					update_source_frontier(face.z);
					update_target_frontier(id2);
				};

				const Float char_len2 = dot(char_n, char_n);
				$if(char_len2 > 1e-20f)
				{
					char_n = char_n * rsqrt(char_len2);
				}
				$else
				{
					char_n = make_float3(0.0f);
				};
			});
	}
	void DevicePRPPrecomputeShaders::compile_product_adjacency_and_union_find(AsyncCompiler& compiler,
		MeshData<luisa::compute::Buffer>*													 device_mesh_data)
	{
		using namespace luisa::compute;

		// Map a hit type/id to a linear vertex, edge, or face element index.
		auto fn_elem_index = [](Var<uint> type, Var<uint> id, Var<uint> N_verts, Var<uint> N_edges, Var<bool> is_src) -> Uint
		{
			// VF=0, EE=1, FV=2
			return ite(
				type == 0u,
				ite(is_src, id, N_verts + N_edges + id), // VF: src=vert, dst=face
				ite(type == 1u,
					N_verts + id,							// EE: src=edge_src, dst=edge_dst (both in edge range)
					ite(is_src, N_verts + N_edges + id, id) // FV: src=face, dst=vert
					));
		};

		auto list_contains = [](const BufferView<uint>& list, const Uint left, const Uint right) -> Bool
		{
			Bool find = false;
			Uint prefix = list->read(left);
			Uint suffix = list->read(left + 1u);
			$for(ii, suffix - prefix)
			{
				Uint elem = list->read(prefix + ii);
				$if(elem == right)
				{
					find = true;
					$break;
				};
			};
			return find;
		};
		auto verts_adjacent = [vert_adj_verts = device_mesh_data->sa_vert_adj_verts_csr.view(), list_contains](const Uint vid1, const Uint vid2) -> Bool
		{
			return list_contains(vert_adj_verts, vid1, vid2);
		};
		auto edge_contains_vert = [sa_edges = device_mesh_data->sa_edges.view()](const Uint eid, const Uint vid) -> Bool
		{
			Uint2 edge_verts = sa_edges->read(eid);
			return any(edge_verts == vid);
		};
		auto face_contains_vert = [sa_faces = device_mesh_data->sa_faces.view()](const Uint fid, const Uint vid) -> Bool
		{
			Uint3 face_verts = sa_faces->read(fid);
			return any(face_verts == vid);
		};
		auto face_contains_edge = [edge_adj_faces_ext = device_mesh_data->edge_adj_faces_csr.view(), list_contains](const Uint fid, const Uint eid) -> Bool
		{
			return list_contains(edge_adj_faces_ext, eid, fid);
		};
		auto edges_adjacent = [edge_adj_edges_ext = device_mesh_data->edge_adj_edges_csr.view(), list_contains](const Uint eid1, const Uint eid2) -> Bool
		{
			return list_contains(edge_adj_edges_ext, eid1, eid2);
		};
		auto hash_u32 = [](Uint x) -> Uint
		{
			x = x ^ (x >> 16u);
			x = x * 0x7feb352du;
			x = x ^ (x >> 15u);
			x = x * 0x846ca68bu;
			x = x ^ (x >> 16u);
			return x;
		};

		auto find_adjacent_hits = [fn_elem_index, hash_u32, face_contains_vert, face_contains_edge, verts_adjacent, edge_contains_vert, edges_adjacent,
									  edge_adj_verts = device_mesh_data->sa_edges.view(),
									  face_adj_verts = device_mesh_data->sa_faces.view(),
									  face_adj_edges = device_mesh_data->face_adj_edges.view(),
									  vert_adj_verts_csr = device_mesh_data->sa_vert_adj_verts_csr.view(),
									  vert_adj_faces_csr = device_mesh_data->sa_vert_adj_faces_csr.view(),
									  vert_adj_edges_csr = device_mesh_data->sa_vert_adj_edges_csr.view(),
									  edge_adj_edges_csr = device_mesh_data->edge_adj_edges_csr.view(),
									  edge_adj_faces_csr = device_mesh_data->edge_adj_faces_csr.view()](
									  const BufferVar<uint2>&	hits_info,
									  const BufferVar<float4>&	hit_attrs,
									  const Uint				i,
									  const Uint				N_verts,
									  const Uint				N_edges,
									  BufferVar<uint>&			count_src,
									  BufferVar<uint>&			prefix_src,
									  BufferVar<uint>&			csr_data_src,
									  BufferVar<uint>&			keyed_hash_keys,
									  const Uint				elem_key_stride,
									  const Uint				keyed_hash_capacity,
									  const Uint				use_keyed_rows,
									  const Uint				elem_row_offset,
									  const Uint				task_idx,
									  std::function<void(Uint)> try_pair_v2) -> void
		{
			const auto& left = hits_info.read(i);

			using namespace HitIndices;

			Uint L = get_pair_type(left);
			Uint L_id1 = left.x & extract_mask;
			Uint L_id2 = left.y & extract_mask;

			constexpr uint VF = uint(PairType::VF);
			constexpr uint FV = uint(PairType::FV);
			constexpr uint EE = uint(PairType::EE);

			auto traverse_adjacent = [&](Uint id, uint type, std::function<void(Uint, Uint2)> try_pair_func)
			{
				const Uint elem = fn_elem_index(type, id, N_verts, N_edges, true);
				Uint	   row = elem_row_offset + elem;
				Bool	   row_found = true;
				Bool	   stop_probe = false;
				$if(use_keyed_rows == 2u)
				{
					// Dense per-task rows: row = task * elem_key_stride + elem, O(1) direct addressing without hash probes. elem_key_stride carries N_elem here.
					row = task_idx * elem_key_stride + elem;
				}
				$elif(use_keyed_rows == 1u)
				{
					row_found = false;
					const Uint key = task_idx * elem_key_stride + elem;
					Uint	   slot = hash_u32(key) % keyed_hash_capacity;
					for (uint probe = 0u; probe < 64u; ++probe)
					{
						$if(!row_found & !stop_probe)
						{
							const Uint curr = keyed_hash_keys.read(slot);
							$if(curr == key)
							{
								row = slot;
								row_found = true;
							}
							$elif(curr == 0xffffffffu)
							{
								stop_probe = true;
							}
							$else
							{
								slot = (slot + 1u) % keyed_hash_capacity;
							};
						};
					}
				};
				$if(row_found)
				{
					const Uint count = count_src.read(row);
					const Uint prefix = prefix_src.read(row);
					$for(jj, count)
					{
						Uint j = csr_data_src.read(prefix + jj);
						$if(j != i)
						{
							Uint2 hit_info_j = hits_info.read(j);
							$if(get_pair_type(hit_info_j) == type)
							{
								try_pair_func(j, hit_info_j);
							};
						};
					};
				};
			};

			// VF
			$if(L == VF)
			{
				Uint vid = L_id1;
				Uint fid = L_id2;

				// Legacy VF-to-VF adjacency scan retained for reference.

				// VF adj VF: Access left.V adj verts (CSR) -> vert_contains_pairs
				{
					Uint prefix_vid = vert_adj_verts_csr->read(vid);
					Uint suffix_vid = vert_adj_verts_csr->read(vid + 1u);
					$for(ii, suffix_vid - prefix_vid)
					{
						Uint av = vert_adj_verts_csr->read(prefix_vid + ii);
						traverse_adjacent(av, VF,
							[&](Uint j, Uint2 hit_info_j)
							{
								$if(get_vf_fid(hit_info_j) == fid)
								{
									try_pair_v2(j);
								};
							});
					};
				}

				// Legacy VF-to-FV adjacency scan retained for reference.

				// VF adj FV: vert->adj faces CSR + face_contains_pairs CSR
				{
					Uint prefix_af = vert_adj_faces_csr->read(vid);
					Uint suffix_af = vert_adj_faces_csr->read(vid + 1u);
					$for(ai, suffix_af - prefix_af)
					{
						Uint af = vert_adj_faces_csr->read(prefix_af + ai);
						traverse_adjacent(af, FV,
							[&](Uint j, Uint2 hit_info_j)
							{
								$if(face_contains_vert(fid, get_fv_vid(hit_info_j)))
								{
									try_pair_v2(j);
								};
							});
					};
				}

				// Legacy VF-to-EE adjacency scan retained for reference.

				// VF adj EE: vert->adj edges CSR + edge_contains_pairs CSR
				{
					Uint prefix_ae = vert_adj_edges_csr->read(vid);
					Uint suffix_ae = vert_adj_edges_csr->read(vid + 1u);
					$for(ei, suffix_ae - prefix_ae)
					{
						Uint ae = vert_adj_edges_csr->read(prefix_ae + ei);
						traverse_adjacent(ae, EE,
							[&](Uint j, Uint2 hit_info_j)
							{
								$if(face_contains_edge(fid, get_ee_eid2(hit_info_j)))
								{
									try_pair_v2(j);
								};
							});
					};
				}
			}

			// FV
			$elif(L == FV)
			{
				Uint fid = L_id1;
				Uint vid = L_id2;

				// Legacy FV-to-VF adjacency scan retained for reference.

				// FV adj VF: face->adj verts CSR + vert_contains_pairs CSR
				{
					Uint3 face = face_adj_verts->read(fid);
					for (uint ii = 0; ii < 3; ii++)
					{
						Uint av = face[ii];
						traverse_adjacent(av, VF,
							[&](Uint j, Uint2 hit_info_j)
							{
								$if(face_contains_vert(get_vf_fid(hit_info_j), vid))
								{
									try_pair_v2(j);
								};
							});
					};
				}

				// Legacy FV-to-FV adjacency scan retained for reference.

				// FV adj FV: face_contains_pairs CSR
				{
					traverse_adjacent(fid, FV,
						[&](Uint j, Uint2 hit_info_j)
						{
							$if(verts_adjacent(vid, get_fv_vid(hit_info_j)))
							{
								try_pair_v2(j);
							};
						});
				}

				// Legacy FV-to-EE adjacency scan retained for reference.

				// FV adj EE: face->adj edges CSR + edge_contains_pairs CSR
				{
					Uint3 face_adj_edge = face_adj_edges->read(fid);
					for (uint ii = 0; ii < 3; ii++)
					{
						Uint ae = face_adj_edge[ii];
						traverse_adjacent(ae, EE,
							[&](Uint j, Uint2 hit_info_j)
							{
								$if(edge_contains_vert(get_ee_eid2(hit_info_j), vid))
								{
									try_pair_v2(j);
								};
							});
					};
				}
			}

			// EE
			$elif(L == EE)
			{
				Uint eid1 = L_id1;
				Uint eid2 = L_id2;

				// Legacy EE-to-VF adjacency scan retained for reference.

				// EE adj VF: edge->adj verts CSR + vert_contains_pairs CSR
				{
					Uint2 edge = edge_adj_verts->read(eid1);
					for (uint ii = 0; ii < 2; ii++)
					{
						Uint av = edge[ii];
						traverse_adjacent(av, VF,
							[&](Uint j, Uint2 hit_info_j)
							{
								$if(face_contains_edge(get_vf_fid(hit_info_j), eid2))
								{
									try_pair_v2(j);
								};
							});
					}
				}

				// Legacy EE-to-FV adjacency scan retained for reference.

				// EE adj FV: edge->adj faces CSR + face_contains_pairs CSR
				{
					Uint prefix_af = edge_adj_faces_csr->read(eid1);
					Uint suffix_af = edge_adj_faces_csr->read(eid1 + 1u);
					$for(ai, suffix_af - prefix_af)
					{
						Uint af = edge_adj_faces_csr->read(prefix_af + ai);
						traverse_adjacent(af, FV,
							[&](Uint j, Uint2 hit_info_j)
							{
								$if(edge_contains_vert(eid2, get_fv_vid(hit_info_j)))
								{
									try_pair_v2(j);
								};
							});
					};
				}

				// EE adj EE: Access left.E1's edge_contains_pairs and edge_adj_edges CSR
				{

					// Legacy EE-to-EE adjacency scan retained for reference.

					// Case 1: right.E1 == left.E1, right.E2 adj left.E2
					traverse_adjacent(eid1, EE,
						[&](Uint j, Uint2 hit_info_j)
						{
							$if(get_ee_eid1(hit_info_j) == eid1
								& edges_adjacent(eid2, get_ee_eid2(hit_info_j)))
							{
								try_pair_v2(j);
							};
						});

					// Legacy reverse EE adjacency scan retained for reference.

					// Case 2: right.E1 adj left.E1, right.E2 == left.E2
					Uint prefix_ae = edge_adj_edges_csr->read(eid1);
					Uint suffix_ae = edge_adj_edges_csr->read(eid1 + 1u);
					$for(ai, suffix_ae - prefix_ae)
					{
						Uint ae = edge_adj_edges_csr->read(prefix_ae + ai);
						traverse_adjacent(ae, EE,
							[&](Uint j, Uint2 hit_info_j)
							{
								$if(get_ee_eid2(hit_info_j) == eid2)
								{
									try_pair_v2(j);
								};
							});
					};
				}
			};
		};

		// ─────────────────────────────────────────────

		compiler.compile<1>(fn_batch_hook_cc_direct_keyed,
			[find_adjacent_hits](
				BufferVar<uint>	  count_src,
				BufferVar<uint>	  prefix_src,
				BufferVar<uint>	  csr_data_src,
				BufferVar<uint>	  keyed_hash_keys,
				BufferVar<uint2>  packed_hits,
				BufferVar<float4> packed_attrs,
				BufferVar<uint>	  packed_task_id,
				BufferVar<uint>	  parent,
				BufferVar<uint>	  changed,
				BufferVar<uint>	  converged,
				Uint N_verts, Uint N_edges, Uint elem_key_stride,
				Uint hash_capacity, Uint total_hits)
			{
				const Uint i = dispatch_x();
				$if(i >= total_hits | converged.read(0u) != 0u)
				{
					$return();
				};
				const Uint task_idx = packed_task_id.read(i);
				auto	   hook_pair = [&](const Uint j_adj)
				{
					Uint ru = i;
					$while(parent.read(ru) != ru)
					{
						ru = parent.read(ru);
					};
					Uint rv = j_adj;
					$while(parent.read(rv) != rv)
					{
						rv = parent.read(rv);
					};
					$if(ru != rv)
					{
						const Uint hi = max(ru, rv);
						const Uint lo = min(ru, rv);
						const Uint old_val = parent.atomic(hi).fetch_min(lo);
						$if(old_val > lo)
						{
							changed.atomic(0u).exchange(1u);
						};
					};
				};
				find_adjacent_hits(packed_hits, packed_attrs, i,
					N_verts, N_edges, count_src, prefix_src, csr_data_src,
					keyed_hash_keys, elem_key_stride, hash_capacity, 1u, 0u,
					task_idx, hook_pair);
			});
		compiler.compile<1>(fn_batch_hook_solid_interior_adjacency,
			[](
				BufferVar<uint2>  packed_hits,
				BufferVar<float4> packed_attrs,
				BufferVar<uint>	  packed_task_id,
				BufferVar<uint2>  task_hit_range,
				BufferVar<float3> task_dirs,
				BufferVar<uint>	  parent,
				BufferVar<uint>	  changed,
				BufferVar<uint>	  converged,
				BufferVar<uint3>  sa_faces,
				BufferVar<uint2>  sa_edges,
				BufferVar<uint>	  sa_face_mesh_id,
				BufferVar<uint>	  sa_vert_mesh_id,
				BufferVar<uint>	  sa_vert_mesh_type,
				BufferVar<float3> prp_face_normal,
				BufferVar<float3> prp_edge_normal,
				Uint			  total_hits,
				Uint			  all_solid,
				Float			  eps_n,
				Float			  eps_d)
			{
				const Uint i = dispatch_x();
				$if(i >= total_hits | converged.read(0u) != 0u)
				{
					$return();
				};
				const Uint	   task = packed_task_id.read(i);
				const Uint2	   hit_i = packed_hits.read(i);
				const Float4   attr_i = packed_attrs.read(i);
				constexpr uint extract_mask = HitIndices::extract_mask;
				const Uint	   type = hit_i.x >> 30u;
				const Uint	   id1 = hit_i.x & extract_mask;
				const Uint	   id2 = hit_i.y & extract_mask;
				const Float	   dist_i = attr_i.z;
				const Float3   dir = task_dirs.read(task);

				constexpr uint VF = uint(PairType::VF);
				constexpr uint EE = uint(PairType::EE);
				constexpr uint FV = uint(PairType::FV);

				constexpr uint MatRigid = uint(Material::MaterialType::Rigid);
				constexpr uint MatTet = uint(Material::MaterialType::Tetrahedral);

				Bool is_entry = false;
				Uint solid_mesh_id = 0xffffffffu;

				$if(type == VF)
				{
					const Uint fid_a = id2;
					const Uint mesh_a = sa_face_mesh_id.read(fid_a);
					const Uint v0 = sa_faces.read(fid_a).x;
					const Uint mat = sa_vert_mesh_type.read(v0);
					const Bool is_solid = (all_solid != 0u) | (mat == MatRigid | mat == MatTet);
					$if(is_solid)
					{
						const Float3 n_a = prp_face_normal.read(fid_a);
						const Float	 dot_a = dot(n_a, dir);
						$if(dot_a < -eps_n)
						{
							is_entry = true;
							solid_mesh_id = mesh_a;
						};
					};
				}
				$elif(type == EE)
				{
					const Uint eid2_a = id2;
					const Uint v0 = sa_edges.read(eid2_a).x;
					const Uint mesh_a = sa_vert_mesh_id.read(v0);
					const Uint mat = sa_vert_mesh_type.read(v0);
					const Bool is_solid = (all_solid != 0u) | (mat == MatRigid | mat == MatTet);
					$if(is_solid)
					{
						const Float3 n_a = prp_edge_normal.read(eid2_a);
						const Float	 dot_a = dot(n_a, dir);
						$if(dot_a < -eps_n)
						{
							is_entry = true;
							solid_mesh_id = mesh_a;
						};
					};
				}
				$elif(type == FV)
				{
					const Uint fid_a = id1;
					const Uint mesh_a = sa_face_mesh_id.read(fid_a);
					const Uint v0 = sa_faces.read(fid_a).x;
					const Uint mat = sa_vert_mesh_type.read(v0);
					const Bool is_solid = (all_solid != 0u) | (mat == MatRigid | mat == MatTet);
					$if(is_solid)
					{
						const Float3 n_a = prp_face_normal.read(fid_a);
						const Float	 dot_a = dot(n_a, dir);
						$if(dot_a > eps_n)
						{
							is_entry = true;
							solid_mesh_id = mesh_a;
						};
					};
				};

				$if(is_entry)
				{
					const Uint2 range = task_hit_range.read(task);
					const Uint	task_begin = range.x;
					const Uint	task_count = range.y;

					Uint  best_j = total_hits;
					Float min_dist_diff = 1e30f;

					$for(k, task_count)
					{
						const Uint j = task_begin + k;
						$if(j != i)
						{
							const Uint2 hit_j = packed_hits.read(j);
							const Uint	type_j = hit_j.x >> 30u;
							$if(type_j == type)
							{
								const Uint j_id1 = hit_j.x & extract_mask;
								const Uint j_id2 = hit_j.y & extract_mask;
								Bool	   shares_primitive = false;
								Uint	   j_mesh_id = 0xffffffffu;

								$if(type == VF)
								{
									shares_primitive = (j_id1 == id1);
									j_mesh_id = sa_face_mesh_id.read(j_id2);
								}
								$elif(type == EE)
								{
									shares_primitive = (j_id1 == id1);
									const Uint j_v0 = sa_edges.read(j_id2).x;
									j_mesh_id = sa_vert_mesh_id.read(j_v0);
								}
								$elif(type == FV)
								{
									shares_primitive = (j_id2 == id2);
									j_mesh_id = sa_face_mesh_id.read(j_id1);
								};

								$if(shares_primitive & j_mesh_id == solid_mesh_id)
								{
									const Float dist_j = packed_attrs.read(j).z;
									const Float dist_diff = dist_j - dist_i;
									$if(dist_diff > eps_d & dist_diff < min_dist_diff)
									{
										min_dist_diff = dist_diff;
										best_j = j;
									};
								};
							};
						};
					};

					$if(best_j < total_hits)
					{
						Bool		is_exit = false;
						const Uint2 hit_best = packed_hits.read(best_j);
						const Uint	best_id1 = hit_best.x & extract_mask;
						const Uint	best_id2 = hit_best.y & extract_mask;

						$if(type == VF)
						{
							const Float3 n_b = prp_face_normal.read(best_id2);
							is_exit = (dot(n_b, dir) > eps_n);
						}
						$elif(type == EE)
						{
							const Float3 n_b = prp_edge_normal.read(best_id2);
							is_exit = (dot(n_b, dir) > eps_n);
						}
						$elif(type == FV)
						{
							const Float3 n_b = prp_face_normal.read(best_id1);
							is_exit = (dot(n_b, dir) < -eps_n);
						};

						$if(is_exit)
						{
							Uint ru = i;
							$while(parent.read(ru) != ru)
							{
								ru = parent.read(ru);
							};
							Uint rv = best_j;
							$while(parent.read(rv) != rv)
							{
								rv = parent.read(rv);
							};
							$if(ru != rv)
							{
								const Uint hi = max(ru, rv);
								const Uint lo = min(ru, rv);
								const Uint old_val = parent.atomic(hi).fetch_min(lo);
								$if(old_val > lo)
								{
									changed.atomic(0u).exchange(1u);
								};
							};
						};
					};
				};
			});

		compiler.compile<1>(fn_batch_keyed_count_hits_per_elem,
			[fn_elem_index, hash_u32](
				BufferVar<uint2> hits,
				BufferVar<uint>	 packed_task_id,

				BufferVar<uint> keyed_hash_keys,
				BufferVar<uint> count_src,
				BufferVar<uint> touched_rows,
				BufferVar<uint> touched_count,
				BufferVar<uint> overflow,
				Var<uint>		N_verts,
				Var<uint>		N_edges,
				Var<uint>		elem_key_stride,
				Var<uint>		hash_capacity,
				Var<uint>		total_hits,
				Var<uint>		row_mode)
			{
				const Uint hit_idx = dispatch_x();
				$if(hit_idx >= total_hits)
				{
					$return();
				};

				const Uint2 hit = hits.read(hit_idx);
				const Uint	type = hit.x >> 30u;
				const Uint	src_id = hit.x & 0x3FFFFFFFu;
				const Uint	elem = fn_elem_index(type, src_id, N_verts, N_edges, true);
				const Uint	key = packed_task_id.read(hit_idx) * elem_key_stride + elem;
				$if(row_mode == 1u)
				{
					// Dense per-task rows: key is the row index, O(1) addressing.
					const Uint old_count = count_src.atomic(key).fetch_add(1u);
					$if(old_count == 0u)
					{
						const Uint touched_idx = touched_count.atomic(0u).fetch_add(1u);
						touched_rows.write(touched_idx, key);
					};
				}
				$else
				{
					const Uint empty = 0xffffffffu;
					Uint	   slot = hash_u32(key) % hash_capacity;
					Bool	   done = false;
					for (uint probe = 0u; probe < 64u; ++probe)
					{
						$if(!done)
						{
							const Uint old = keyed_hash_keys.atomic(slot).compare_exchange(empty, key);
							$if(old == empty | old == key)
							{
								$if(old == empty)
								{
									const Uint touched_idx = touched_count.atomic(0u).fetch_add(1u);
									touched_rows.write(touched_idx, slot);
								};
								count_src.atomic(slot).fetch_add(1u);
								done = true;
							}
							$else
							{
								slot = (slot + 1u) % hash_capacity;
							};
						};
					}
					$if(!done)
					{
						overflow.atomic(0u).exchange(1u);
					};
				};
			});

		compiler.compile<1>(fn_batch_keyed_fill_csr,
			[fn_elem_index, hash_u32](
				BufferVar<uint2> hits,
				BufferVar<uint>	 packed_task_id,

				BufferVar<uint> keyed_hash_keys,
				BufferVar<uint> prefix_src,
				BufferVar<uint> fill_counts,
				BufferVar<uint> csr_data_src,
				BufferVar<uint> overflow,
				Var<uint>		N_verts,
				Var<uint>		N_edges,
				Var<uint>		elem_key_stride,
				Var<uint>		hash_capacity,
				Var<uint>		total_hits,
				Var<uint>		row_mode)
			{
				const Uint hit_idx = dispatch_x();
				$if(hit_idx >= total_hits)
				{
					$return();
				};

				const Uint2 hit = hits.read(hit_idx);
				const Uint	type = hit.x >> 30u;
				const Uint	src_id = hit.x & 0x3FFFFFFFu;
				const Uint	elem = fn_elem_index(type, src_id, N_verts, N_edges, true);
				const Uint	key = packed_task_id.read(hit_idx) * elem_key_stride + elem;
				$if(row_mode == 1u)
				{
					// Dense per-task rows: key is the row index, O(1) addressing.
					const Uint out = prefix_src.read(key) + fill_counts.atomic(key).fetch_add(1u);
					csr_data_src.write(out, hit_idx);
				}
				$else
				{
					Uint slot = hash_u32(key) % hash_capacity;
					Bool done = false;
					Bool stop_probe = false;
					for (uint probe = 0u; probe < 64u; ++probe)
					{
						$if(!done & !stop_probe)
						{
							const Uint curr = keyed_hash_keys.read(slot);
							$if(curr == key)
							{
								const Uint out = prefix_src.read(slot) + fill_counts.atomic(slot).fetch_add(1u);
								csr_data_src.write(out, hit_idx);
								done = true;
							}
							$elif(curr == 0xffffffffu)
							{
								stop_probe = true;
							}
							$else
							{
								slot = (slot + 1u) % hash_capacity;
							};
						};
					}
					$if(!done)
					{
						overflow.atomic(0u).exchange(1u);
					};
				};
			});
		// ── end batched GPU culling kernels ──────────────────────────────────
	}
	void DevicePRPPrecomputeShaders::compile_cluster_culling(AsyncCompiler& compiler)
	{
		using namespace luisa::compute;

		compiler.compile<1>(fn_prp_init_cc,
			[](BufferVar<uint> parent, Uint count)
			{
				const Uint i = dispatch_x();
				$if(i < count)
				{
					parent.write(i, i);
				};
			});

		compiler.compile<1>(fn_prp_compress_cc,
			[](BufferVar<uint> parent, BufferVar<uint> converged, Uint count)
			{
				const Uint i = dispatch_x();
				$if(i < count)
				{
					$if(converged.read(0u) != 0u)
					{
						$return();
					};
					Uint p = parent.read(i);
					$while(p != parent.read(p))
					{
						p = parent.read(p);
					};
					parent.write(i, p);
				};
			});

		compiler.compile<1>(fn_prp_check_cc_converged,
			[](BufferVar<uint> changed, BufferVar<uint> converged)
			{
				$if(changed.read(0u) == 0u)
				{
					converged.write(0u, 1u);
				};
				changed.write(0u, 0u);
			});
		compiler.compile<1>(fn_prp_reset_cluster_selection,
			[](BufferVar<uint>	 root_task_idx,
				BufferVar<uint>	 root_hit_count,
				BufferVar<uint>	 root_kept_hit_count,
				BufferVar<uint>	 root_boundary_attached,
				BufferVar<uint>	 root_coverage_count,
				BufferVar<uint>	 root_canonical_key,
				BufferVar<uint>	 root_canonical_key_high,
				BufferVar<float> root_max_kept_tgt_geom,
				BufferVar<uint>	 root_dense_id,
				Uint			 total_hits)
			{
				const Uint i = dispatch_x();
				const Uint invalid_key = 0xffffffffu;
				$if(i < total_hits)
				{
					root_task_idx.write(i, 0xffffffffu);
					root_hit_count.write(i, 0u);
					root_kept_hit_count.write(i, 0u);
					root_boundary_attached.write(i, 0u);
					root_coverage_count.write(i, 0u);
					root_canonical_key.write(i, invalid_key);
					root_canonical_key_high.write(i, invalid_key);
					root_max_kept_tgt_geom.write(i, 0.0f);
					root_dense_id.write(i, 0xffffffffu);
				};
			});

		compiler.compile<1>(fn_prp_reset_cluster_task_selection,
			[](BufferVar<uint>	 selected_roots,
				BufferVar<uint>	 selected_boundary_counts,
				BufferVar<uint>	 selected_hit_counts,
				BufferVar<uint>	 selected_keys,
				BufferVar<uint>	 selected_key_highs,
				BufferVar<uint>	 task_component_count,
				BufferVar<uint>	 task_largest_component_hits,
				BufferVar<float> task_max_kept_tgt_geom,
				Uint			 task_count,
				Uint			 total_hits_sentinel)
			{
				const Uint task = dispatch_x();
				$if(task < task_count)
				{
					selected_roots.write(task, total_hits_sentinel);
					selected_boundary_counts.write(task, 0u);
					selected_hit_counts.write(task, 0u);
					selected_keys.write(task, 0xffffffffu);
					selected_key_highs.write(task, 0xffffffffu);
					task_component_count.write(task, 0u);
					task_largest_component_hits.write(task, 0u);
					task_max_kept_tgt_geom.write(task, 0.0f);
				};
			});

		compiler.compile<1>(fn_prp_accumulate_cluster_root_stats,
			[](BufferVar<uint2>	 packed_hits,
				BufferVar<uint>	 packed_task_id,
				BufferVar<uint>	 precluster_valid,
				BufferVar<uint>	 parent,
				BufferVar<uint3> faces,
				BufferVar<uint2> edges,
				BufferVar<uint>	 vert_boundary_flags_csr,
				BufferVar<uint>	 task_contour_idx,
				BufferVar<uint>	 task_boundary_hop_dense,
				BufferVar<float> task_boundary_geom_dense,
				BufferVar<uint>	 task_locality_max_hop,
				BufferVar<uint>	 root_task_idx,
				BufferVar<uint>	 root_hit_count,
				BufferVar<uint>	 root_kept_hit_count,
				BufferVar<uint>	 root_boundary_attached,
				BufferVar<uint>	 root_canonical_key_high,
				BufferVar<float> root_max_kept_tgt_geom,
				BufferVar<uint>	 root_error_flag,
				Uint			 total_hits,
				Uint			 num_verts,
				Uint			 use_coverage_locality)
			{
				const Uint hit_idx = dispatch_x();
				$if(hit_idx >= total_hits)
				{
					$return();
				};
				$if(precluster_valid.read(hit_idx) == 0u)
				{
					$return();
				};
				const Uint2 ids = packed_hits.read(hit_idx);
				const Uint	type = ids.x >> 30u;
				const Uint	id1 = ids.x & 0x3fffffffu;
				const Uint	id2 = ids.y & 0x3fffffffu;
				const Uint	task = packed_task_id.read(hit_idx);
				const Uint	contour_idx = task_contour_idx.read(task);
				const Uint	root = parent.read(hit_idx);
				$if(root >= total_hits)
				{
					root_error_flag.atomic(0u).exchange(1u);
					$return();
				};
				root_task_idx.write(root, task);
				root_hit_count.atomic(root).fetch_add(1u);
				const Uint canonical_key_high = (type << 30u) | (id1 >> 1u);
				root_canonical_key_high.atomic(root).fetch_min(canonical_key_high);

				auto dense_index = [&](const Uint mesh_side, const Uint vid) -> Uint
				{
					return ((task * 2u + mesh_side) * num_verts + vid);
				};
				auto vertex_has_boundary_flag = [&](const Uint vid, const Uint flag) -> Bool
				{
					Bool	   found = false;
					const Uint begin = vert_boundary_flags_csr.read(vid);
					const Uint end = vert_boundary_flags_csr.read(vid + 1u);
					$for(ii, end - begin)
					{
						found |= vert_boundary_flags_csr.read(begin + ii) == flag;
					};
					return found;
				};
				auto hit_is_boundary_attached = [&]() -> Bool
				{
					const Uint src_flag = 2u * contour_idx;
					const Uint dst_flag = src_flag + 1u;
					Bool	   attached = false;
					$if(type == 0u)
					{
						const UInt3 face = faces.read(id2);
						attached = vertex_has_boundary_flag(id1, src_flag)
							& vertex_has_boundary_flag(face.x, dst_flag)
							& vertex_has_boundary_flag(face.y, dst_flag)
							& vertex_has_boundary_flag(face.z, dst_flag);
					}
					$elif(type == 1u)
					{
						const UInt2 e0 = edges.read(id1);
						const UInt2 e1 = edges.read(id2);
						attached = vertex_has_boundary_flag(e0.x, src_flag)
							& vertex_has_boundary_flag(e0.y, src_flag)
							& vertex_has_boundary_flag(e1.x, dst_flag)
							& vertex_has_boundary_flag(e1.y, dst_flag);
					}
					$else
					{
						const UInt3 face = faces.read(id1);
						attached = vertex_has_boundary_flag(face.x, src_flag)
							& vertex_has_boundary_flag(face.y, src_flag)
							& vertex_has_boundary_flag(face.z, src_flag)
							& vertex_has_boundary_flag(id2, dst_flag);
					};
					return attached;
				};
				$if(hit_is_boundary_attached())
				{
					root_boundary_attached.atomic(root).exchange(1u);
				};

				auto min_target_hop = [&]() -> Uint
				{
					Uint best = 0xffffffffu;
					$if(type == 0u)
					{
						const UInt3 face = faces.read(id2);
						best = min(best, task_boundary_hop_dense.read(dense_index(1u, face.x)));
						best = min(best, task_boundary_hop_dense.read(dense_index(1u, face.y)));
						best = min(best, task_boundary_hop_dense.read(dense_index(1u, face.z)));
					}
					$elif(type == 1u)
					{
						const UInt2 e1 = edges.read(id2);
						best = min(best, task_boundary_hop_dense.read(dense_index(1u, e1.x)));
						best = min(best, task_boundary_hop_dense.read(dense_index(1u, e1.y)));
					}
					$else
					{
						best = min(best, task_boundary_hop_dense.read(dense_index(1u, id2)));
					};
					return best;
				};
				const Uint max_hop = task_locality_max_hop.read(task);
				const Bool kept_by_locality = (use_coverage_locality == 0u) | (max_hop == 0xffffffffu) | (min_target_hop() <= max_hop);
				$if(kept_by_locality)
				{
					root_kept_hit_count.atomic(root).fetch_add(1u);
					auto update_target_geom = [&](const Uint vid)
					{
						root_max_kept_tgt_geom.atomic(root).fetch_max(task_boundary_geom_dense.read(dense_index(1u, vid)));
					};
					$if(type == 0u)
					{
						const UInt3 face = faces.read(id2);
						update_target_geom(face.x);
						update_target_geom(face.y);
						update_target_geom(face.z);
					}
					$elif(type == 1u)
					{
						const UInt2 e1 = edges.read(id2);
						update_target_geom(e1.x);
						update_target_geom(e1.y);
					}
					$else
					{
						update_target_geom(id2);
					};
				};
			});

		compiler.compile<1>(fn_prp_assign_dense_cluster_roots,
			[](BufferVar<uint>	root_hit_count,
				BufferVar<uint> root_boundary_attached,
				BufferVar<uint> root_dense_id,
				BufferVar<uint> valid_root_count,
				BufferVar<uint> overflow_flag,
				Uint			total_hits,
				Uint			max_dense_roots)
			{
				const Uint root = dispatch_x();
				$if(root >= total_hits)
				{
					$return();
				};
				$if(root_hit_count.read(root) > 0u & root_boundary_attached.read(root) > 0u)
				{
					const Uint dense_id = valid_root_count.atomic(0u).fetch_add(1u);
					$if(dense_id < max_dense_roots)
					{
						root_dense_id.write(root, dense_id);
					}
					$else
					{
						overflow_flag.atomic(0u).exchange(1u);
						root_dense_id.write(root, 0xffffffffu);
					};
				}
				$else
				{
					root_dense_id.write(root, 0xffffffffu);
				};
			});

		compiler.compile<1>(fn_prp_fill_dense_cluster_coverage_mask,
			[](BufferVar<uint2>	 packed_hits,
				BufferVar<uint>	 packed_task_id,
				BufferVar<uint>	 precluster_valid,
				BufferVar<uint>	 parent,
				BufferVar<uint3> faces,
				BufferVar<uint2> edges,
				BufferVar<uint>	 task_boundary_hop_dense,
				BufferVar<uint>	 task_locality_max_hop,
				BufferVar<uint>	 root_dense_id,
				BufferVar<uint>	 dense_mask,
				Uint			 total_hits,
				Uint			 num_verts,
				Uint			 stride,
				Uint			 max_dense_roots,
				Uint			 use_coverage_locality)
			{
				const Uint hit_idx = dispatch_x();
				$if(hit_idx >= total_hits)
				{
					$return();
				};
				$if(precluster_valid.read(hit_idx) == 0u)
				{
					$return();
				};
				const Uint root = parent.read(hit_idx);
				$if(root >= total_hits)
				{
					$return();
				};
				const Uint dense_id = root_dense_id.read(root);
				$if(dense_id >= max_dense_roots)
				{
					$return();
				};

				const Uint	task = packed_task_id.read(hit_idx);
				const Uint2 ids = packed_hits.read(hit_idx);
				const Uint	type = ids.x >> 30u;
				const Uint	id1 = ids.x & 0x3fffffffu;
				const Uint	id2 = ids.y & 0x3fffffffu;

				auto dense_index = [&](const Uint mesh_side, const Uint vid) -> Uint
				{
					return ((task * 2u + mesh_side) * num_verts + vid);
				};
				auto min_target_hop = [&]() -> Uint
				{
					Uint best = 0xffffffffu;
					$if(type == 0u)
					{
						const UInt3 face = faces.read(id2);
						best = min(best, task_boundary_hop_dense.read(dense_index(1u, face.x)));
						best = min(best, task_boundary_hop_dense.read(dense_index(1u, face.y)));
						best = min(best, task_boundary_hop_dense.read(dense_index(1u, face.z)));
					}
					$elif(type == 1u)
					{
						const UInt2 e1 = edges.read(id2);
						best = min(best, task_boundary_hop_dense.read(dense_index(1u, e1.x)));
						best = min(best, task_boundary_hop_dense.read(dense_index(1u, e1.y)));
					}
					$else
					{
						best = min(best, task_boundary_hop_dense.read(dense_index(1u, id2)));
					};
					return best;
				};
				const Uint max_hop = task_locality_max_hop.read(task);
				const Bool kept_by_locality = (use_coverage_locality == 0u) | (max_hop == 0xffffffffu) | (min_target_hop() <= max_hop);
				$if(kept_by_locality)
				{
					auto mark_vertex = [&](const Uint side, const Uint vid)
					{
						const Uint global_vid = side * num_verts + vid;
						const Uint word_idx = dense_id * stride + (global_vid >> 5u);
						const Uint bit = 1u << (global_vid & 31u);
						dense_mask.atomic(word_idx).fetch_or(bit);
					};

					$if(type == 0u)
					{
						const UInt3 face = faces.read(id2);
						mark_vertex(0u, id1);
						mark_vertex(1u, face.x);
						mark_vertex(1u, face.y);
						mark_vertex(1u, face.z);
					}
					$elif(type == 1u)
					{
						const UInt2 e0 = edges.read(id1);
						const UInt2 e1 = edges.read(id2);
						mark_vertex(0u, e0.x);
						mark_vertex(0u, e0.y);
						mark_vertex(1u, e1.x);
						mark_vertex(1u, e1.y);
					}
					$else
					{
						const UInt3 face = faces.read(id1);
						mark_vertex(0u, face.x);
						mark_vertex(0u, face.y);
						mark_vertex(0u, face.z);
						mark_vertex(1u, id2);
					};
				};
			});

		compiler.compile<1>(fn_prp_count_cluster_ef_coverage,
			[](BufferVar<uint>	 root_task_idx,
				BufferVar<uint>	 root_hit_count,
				BufferVar<uint>	 root_boundary_attached,
				BufferVar<uint>	 root_dense_id,
				BufferVar<uint>	 task_ef_prefix,
				BufferVar<uint3> task_ef_edges,
				BufferVar<uint3> task_ef_faces,
				BufferVar<uint>	 dense_mask,
				BufferVar<uint>	 root_coverage_count,
				Uint			 num_verts,
				Uint			 stride,
				Uint			 max_dense_roots,
				Uint			 total_hits)
			{
				const Uint root = dispatch_x();
				$if(root >= total_hits)
				{
					$return();
				};
				$if(root_hit_count.read(root) == 0u | root_boundary_attached.read(root) == 0u)
				{
					$return();
				};
				const Uint dense_id = root_dense_id.read(root);
				$if(dense_id >= max_dense_roots)
				{
					$return();
				};
				auto contains_vertex = [&](const Uint side, const Uint vid) -> Bool
				{
					const Uint global_vid = side * num_verts + vid;
					const Uint word_idx = dense_id * stride + (global_vid >> 5u);
					const Uint bit = 1u << (global_vid & 31u);
					return (dense_mask.read(word_idx) & bit) != 0u;
				};
				const Uint task = root_task_idx.read(root);
				const Uint begin = task_ef_prefix.read(task);
				const Uint end = task_ef_prefix.read(task + 1u);
				Uint	   covered = 0u;
				$for(pi, begin, end)
				{
					const UInt3 ef = task_ef_edges.read(pi);
					const UInt2 edge = ef.xy();
					const UInt3 face = task_ef_faces.read(pi);
					const Bool	edge_ok = contains_vertex(ef.z, edge.x) | contains_vertex(ef.z, edge.y);
					const Bool	face_ok = contains_vertex(ef.z ^ 1u, face.x)
						| contains_vertex(ef.z ^ 1u, face.y) | contains_vertex(ef.z ^ 1u, face.z);
					covered += ite(edge_ok & face_ok, 1u, 0u);
				};
				root_coverage_count.write(root, covered);
			});

		compiler.compile<1>(fn_prp_accumulate_cluster_root_key_low,
			[](BufferVar<uint2> packed_hits,
				BufferVar<uint> precluster_valid,
				BufferVar<uint> parent,
				BufferVar<uint> root_canonical_key,
				BufferVar<uint> root_canonical_key_high,
				BufferVar<uint> root_error_flag,
				Uint			total_hits)
			{
				const Uint hit_idx = dispatch_x();
				$if(hit_idx >= total_hits)
				{
					$return();
				};
				$if(precluster_valid.read(hit_idx) == 0u)
				{
					$return();
				};
				const Uint2 ids = packed_hits.read(hit_idx);
				const Uint	type = ids.x >> 30u;
				const Uint	id1 = ids.x & 0x3fffffffu;
				const Uint	id2 = ids.y & 0x3fffffffu;
				const Uint	root = parent.read(hit_idx);
				$if(root >= total_hits)
				{
					root_error_flag.atomic(0u).exchange(1u);
					$return();
				};
				const Uint canonical_key_high = (type << 30u) | (id1 >> 1u);
				$if(root_canonical_key_high.read(root) == canonical_key_high)
				{
					root_canonical_key.atomic(root).fetch_min((id1 << 31u) | id2);
				};
			});

		compiler.compile<1>(fn_prp_select_cluster_roots_pass,
			[](BufferVar<uint>	root_task_idx,
				BufferVar<uint> root_hit_count,
				BufferVar<uint> root_boundary_attached,
				BufferVar<uint> root_coverage_count,
				BufferVar<uint> root_kept_hit_count,
				BufferVar<uint> root_canonical_key_low,
				BufferVar<uint> root_canonical_key_high,
				BufferVar<uint> task_best_cov,
				BufferVar<uint> task_best_kept,
				BufferVar<uint> task_best_key_high,
				BufferVar<uint> task_best_key_low,
				BufferVar<uint> task_best_root,
				BufferVar<uint> task_component_count,
				BufferVar<uint> task_largest_component_hits,
				Uint			pass_id,
				Uint			total_hits)
			{
				const Uint root = dispatch_x();
				$if(root >= total_hits)
				{
					$return();
				};
				const Uint hit_count = root_hit_count.read(root);
				const Uint task = root_task_idx.read(root);
				$if(hit_count == 0u | root_boundary_attached.read(root) == 0u)
				{
					$return();
				};
				const Uint cov = root_coverage_count.read(root);
				const Uint kept = root_kept_hit_count.read(root);
				$if(pass_id == 0u)
				{
					task_component_count.atomic(task).fetch_add(1u);
					task_largest_component_hits.atomic(task).fetch_max(hit_count);
					task_best_cov.atomic(task).fetch_max(cov);
				}
				$elif(pass_id == 1u)
				{
					$if(cov == task_best_cov.read(task))
					{
						task_best_kept.atomic(task).fetch_max(kept);
					};
				}
				$elif(pass_id == 2u)
				{
					$if(cov == task_best_cov.read(task) & kept == task_best_kept.read(task))
					{
						task_best_key_high.atomic(task).fetch_min(root_canonical_key_high.read(root));
					};
				}
				$elif(pass_id == 3u)
				{
					$if(cov == task_best_cov.read(task) & kept == task_best_kept.read(task)
						& root_canonical_key_high.read(root) == task_best_key_high.read(task))
					{
						task_best_key_low.atomic(task).fetch_min(root_canonical_key_low.read(root));
					};
				}
				$else
				{
					$if(cov == task_best_cov.read(task) & kept == task_best_kept.read(task)
						& root_canonical_key_high.read(root) == task_best_key_high.read(task)
						& root_canonical_key_low.read(root) == task_best_key_low.read(task))
					{
						task_best_root.atomic(task).fetch_min(root);
					};
				};
			});

		compiler.compile<1>(fn_prp_select_cluster_roots_finalize,
			[](BufferVar<uint>	 task_best_cov,
				BufferVar<uint>	 task_best_kept,
				BufferVar<uint>	 task_best_key_high,
				BufferVar<uint>	 task_best_key_low,
				BufferVar<uint>	 task_best_root,
				BufferVar<float> root_max_kept_tgt_geom,
				BufferVar<uint>	 selected_roots,
				BufferVar<uint>	 selected_boundary_counts,
				BufferVar<uint>	 selected_hit_counts,
				BufferVar<uint>	 selected_keys_low,
				BufferVar<uint>	 selected_keys_high,
				BufferVar<float> task_max_kept_tgt_geom,
				Uint			 num_tasks,
				Uint			 total_hits)
			{
				const Uint task = dispatch_x();
				$if(task >= num_tasks)
				{
					$return();
				};
				const Uint best_root = task_best_root.read(task);
				selected_roots.write(task, best_root);
				selected_boundary_counts.write(task, task_best_cov.read(task));
				selected_hit_counts.write(task, task_best_kept.read(task));
				selected_keys_high.write(task, task_best_key_high.read(task));
				selected_keys_low.write(task, task_best_key_low.read(task));
				$if(best_root < total_hits)
				{
					task_max_kept_tgt_geom.write(task, root_max_kept_tgt_geom.read(best_root));
				}
				$else
				{
					task_max_kept_tgt_geom.write(task, 0.0f);
				};
			});

		compiler.compile<1>(fn_prp_write_cluster_valid_flags,
			[](BufferVar<uint2>	 packed_hits,
				BufferVar<uint>	 packed_task_id,
				BufferVar<uint>	 parent,
				BufferVar<uint3> faces,
				BufferVar<uint2> edges,
				BufferVar<uint>	 task_boundary_hop_dense,
				BufferVar<uint>	 task_locality_max_hop,
				BufferVar<uint>	 selected_roots,
				BufferVar<uint>	 hit_valid,
				Uint			 total_hits,
				Uint			 num_verts,
				Uint			 keep_all_for_selected_task,
				Uint			 use_coverage_locality)
			{
				const Uint hit_idx = dispatch_x();
				$if(hit_idx >= total_hits)
				{
					$return();
				};
				$if(hit_valid.read(hit_idx) == 0u)
				{
					$return();
				};
				const Uint task = packed_task_id.read(hit_idx);
				const Uint selected_root = selected_roots.read(task);
				$if(selected_root >= total_hits)
				{
					hit_valid.write(hit_idx, 0u);
					$return();
				};
				$if(keep_all_for_selected_task != 0u)
				{
					hit_valid.write(hit_idx, 1u);
					$return();
				};
				$if(parent.read(hit_idx) != selected_root)
				{
					hit_valid.write(hit_idx, 0u);
					$return();
				};
				$if(use_coverage_locality != 0u)
				{
					const Uint2 ids = packed_hits.read(hit_idx);
					const Uint	type = ids.x >> 30u;
					const Uint	id2 = ids.y & 0x3fffffffu;
					auto		dense_index = [&](const Uint vid) -> Uint
					{
						return ((task * 2u + 1u) * num_verts + vid);
					};
					Uint best = 0xffffffffu;
					$if(type == 0u)
					{
						const UInt3 face = faces.read(id2);
						best = min(best, task_boundary_hop_dense.read(dense_index(face.x)));
						best = min(best, task_boundary_hop_dense.read(dense_index(face.y)));
						best = min(best, task_boundary_hop_dense.read(dense_index(face.z)));
					}
					$elif(type == 1u)
					{
						const UInt2 e1 = edges.read(id2);
						best = min(best, task_boundary_hop_dense.read(dense_index(e1.x)));
						best = min(best, task_boundary_hop_dense.read(dense_index(e1.y)));
					}
					$else
					{
						best = min(best, task_boundary_hop_dense.read(dense_index(id2)));
					};
					const Uint max_hop = task_locality_max_hop.read(task);
					$if(max_hop != 0xffffffffu & best > max_hop)
					{
						hit_valid.write(hit_idx, 0u);
						$return();
					};
				};
				hit_valid.write(hit_idx, 1u);
			});
	}
	void DevicePRPPrecomputeShaders::compile_batch_keyed_csr(AsyncCompiler& compiler)
	{
		using namespace luisa::compute;

		// ── Batched GPU culling kernels (all combos in one dispatch) ─────────

		compiler.compile<1>(fn_batch_gather_touched_counts,
			[](
				BufferVar<uint> count_src,
				BufferVar<uint> touched_rows,
				BufferVar<uint> touched_counts,
				BufferVar<uint> touched_count,
				Var<uint>		compact_capacity)
			{
				const Uint compact_idx = dispatch_x();
				$if(compact_idx >= compact_capacity)
				{
					$return();
				};
				const Uint count = touched_count.read(0u);
				Uint	   value = 0u;
				$if(compact_idx < count)
				{
					const Uint row = touched_rows.read(compact_idx);
					value = count_src.read(row);
				};
				touched_counts.write(compact_idx, value);
			});
		compiler.compile<1>(fn_batch_scatter_touched_prefix,
			[](
				BufferVar<uint> prefix_src,
				BufferVar<uint> touched_rows,
				BufferVar<uint> touched_prefix,
				BufferVar<uint> touched_count,
				Var<uint>		compact_capacity)
			{
				const Uint compact_idx = dispatch_x();
				$if(compact_idx >= compact_capacity)
				{
					$return();
				};
				$if(compact_idx < touched_count.read(0u))
				{
					const Uint row = touched_rows.read(compact_idx);
					prefix_src.write(row, touched_prefix.read(compact_idx));
				};
			});
		compiler.compile<1>(fn_batch_clear_touched_rows,
			[](
				BufferVar<uint> count_src,
				BufferVar<uint> touched_rows,
				BufferVar<uint> touched_count,
				Var<uint>		compact_capacity)
			{
				const Uint compact_idx = dispatch_x();
				$if(compact_idx >= compact_capacity)
				{
					$return();
				};
				$if(compact_idx < touched_count.read(0u))
				{
					count_src.write(touched_rows.read(compact_idx), 0u);
				};
			});
	}
	void DevicePRPPrecomputeShaders::compile_exact_filter_and_raycast(AsyncCompiler& compiler)
	{
		using namespace luisa::compute;

		luisa::compute::Callable fn_segment_triangle_intersection =
			[](const Float3& seg_start, const Float3& seg_end, const Float3& tri0, const Float3& tri1, const Float3& tri2)
		{
			Float4 result = make_float4(0.0f);
			Float3 bary = make_float3(0.0f);
			Float  t = 0.0f;
			$if(distance::LineIntersection(seg_start, seg_end, tri0, tri1, tri2, bary, t))
			{
				result = make_float4(bary.y, bary.z, t, 1.0f);
			};
			return result;
		};

		luisa::compute::Callable fn_swept_edge_edge_intersection = [](const Float3&	 p0,
																	   const Float3& p1,
																	   const Float3& q0,
																	   const Float3& q1,
																	   const Float3& swept_min,
																	   const Float3& swept_max,
																	   const Float3& direction,
																	   const Float	 max_ray_dist)
		{
			Float4		 result = make_float4(0.0f);
			const Float3 q_min = make_float3(min_scalar(q0.x, q1.x), min_scalar(q0.y, q1.y), min_scalar(q0.z, q1.z));
			const Float3 q_max = make_float3(max_scalar(q0.x, q1.x), max_scalar(q0.y, q1.y), max_scalar(q0.z, q1.z));
			const Bool	 overlap = (swept_min.x <= q_max.x & swept_max.x >= q_min.x) & (swept_min.y <= q_max.y & swept_max.y >= q_min.y) & (swept_min.z <= q_max.z & swept_max.z >= q_min.z);
			$if(overlap)
			{
				const Float3 e1 = p1 - p0;
				const Float3 e2 = q1 - q0;
				const Float3 r = q0 - p0;
				const Float3 n = cross(e1, e2);
				const Float	 det = -dot(direction, n);
				$if(abs(det) > 1e-10f)
				{
					const Float inv_det = 1.0f / det;
					const Float det_r_e1_e2 = dot(r, cross(e1, e2));
					const Float det_dir_r_e2 = dot(direction, cross(r, e2));
					const Float det_dir_e1_r = dot(direction, cross(e1, r));
					const Float s = -det_r_e1_e2 * inv_det;
					const Float t1 = -det_dir_r_e2 * inv_det;
					const Float t2 = det_dir_e1_r * inv_det;
					$if(s >= 0.0f & s <= max_ray_dist & t1 >= 0.0f & t1 <= 1.0f & t2 >= 0.0f & t2 <= 1.0f)
					{
						result = make_float4(s, t1, t2, 1.0f);
					};
				};
			};
			return result;
		};

		luisa::compute::Callable fn_swept_plane_aabb_intersect = [](const Float3&			  swept_min,
																	 const Float3&			  swept_max,
																	 const Float3&			  plane_normal,
																	 const Float&			  plane_d,
																	 const Bool&			  use_plane_test,
																	 const Float&			  proj_min,
																	 const Float&			  proj_max,
																	 const Float3&			  direction,
																	 const CompressedAABBVar& node)
		{
			const Float3 node_min = make_float3(node[0][0], node[0][1], node[0][2]);
			const Float3 node_max = make_float3(node[1][0], node[1][1], node[1][2]);

			Bool result = def<bool>(false);

			const Bool overlap = (swept_min.x <= node_max.x & swept_max.x >= node_min.x)
				& (swept_min.y <= node_max.y & swept_max.y >= node_min.y)
				& (swept_min.z <= node_max.z & swept_max.z >= node_min.z);

			$if(overlap)
			{
				const Float3 box_center = (node_min + node_max) * 0.5f;
				const Float3 box_extent = (node_max - node_min) * 0.5f;

				const Float center_proj = dot(box_center, direction);
				const Float proj_radius = box_extent.x * abs(direction.x)
					+ box_extent.y * abs(direction.y)
					+ box_extent.z * abs(direction.z);

				$if(center_proj + proj_radius >= proj_min - 1e-6f
					& center_proj - proj_radius <= proj_max + 1e-6f)
				{
					$if(!use_plane_test)
					{
						result = true;
					}
					$else
					{
						const Float center_dist = dot(plane_normal, box_center) - plane_d;
						const Float plane_radius = box_extent.x * abs(plane_normal.x)
							+ box_extent.y * abs(plane_normal.y)
							+ box_extent.z * abs(plane_normal.z);
						result = abs(center_dist) <= plane_radius + 1e-6f;
					};
				};
			};
			return result;
		};

		luisa::compute::Callable fn_ray_aabb_intersect = [](const Float3& orig, const Float3& dir_inv, const CompressedAABBVar& node, const Float& t_max)
		{
			const Float3 node_min = make_float3(node[0][0], node[0][1], node[0][2]);
			const Float3 node_max = make_float3(node[1][0], node[1][1], node[1][2]);

			Float t1x = (node_min.x - orig.x) * dir_inv.x;
			Float t2x = (node_max.x - orig.x) * dir_inv.x;
			Float tmin = ite(t1x < t2x, t1x, t2x);
			Float tmax = ite(t1x > t2x, t1x, t2x);

			Float t1y = (node_min.y - orig.y) * dir_inv.y;
			Float t2y = (node_max.y - orig.y) * dir_inv.y;
			tmin = ite(tmin > ite(t1y < t2y, t1y, t2y), tmin, ite(t1y < t2y, t1y, t2y));
			tmax = ite(tmax < ite(t1y > t2y, t1y, t2y), tmax, ite(t1y > t2y, t1y, t2y));

			Float t1z = (node_min.z - orig.z) * dir_inv.z;
			Float t2z = (node_max.z - orig.z) * dir_inv.z;
			tmin = ite(tmin > ite(t1z < t2z, t1z, t2z), tmin, ite(t1z < t2z, t1z, t2z));
			tmax = ite(tmax < ite(t1z > t2z, t1z, t2z), tmax, ite(t1z > t2z, t1z, t2z));

			return (tmax >= ite(tmin > def<float>(0.0f), tmin, def<float>(0.0f))) & (tmin <= t_max);
		};

		luisa::compute::Callable fn_is_in_range = [](const Uint primitive_id, const Uint range_start, const Uint range_count)
		{
			const Uint range_end = range_start + range_count;
			return primitive_id >= range_start & primitive_id < range_end;
		};
		luisa::compute::Callable fn_is_in_sorted_task_list = [](BufferVar<uint> task_prefix, BufferVar<uint> task_ids, const Uint task, const Uint primitive_id)
		{
			Uint lo = task_prefix.read(task);
			Uint hi = task_prefix.read(task + 1u);
			Bool found = false;
			$while(lo < hi)
			{
				const Uint mid = (lo + hi) >> 1u;
				const Uint curr = task_ids.read(mid);
				$if(curr < primitive_id)
				{
					lo = mid + 1u;
				}
				$else
				{
					$if(curr == primitive_id)
					{
						found = true;
						$break;
					};
					hi = mid;
				};
			};
			return found;
		};

		static constexpr uint STACK_SIZE = 64;

		compiler.compile<1>(fn_prp_filter_exact_candidate_indexed,
			[fn_is_in_sorted_task_list](
				BufferVar<uint2>  in_slice_hits,
				BufferVar<float4> in_slice_attrs,
				BufferVar<uint>	  task_coarse_hit_counts,
				BufferVar<uint>	  task_coarse_offsets,
				BufferVar<uint>	  task_target_face_prefix,
				BufferVar<uint>	  task_target_face_ids,
				BufferVar<uint>	  task_target_edge_prefix,
				BufferVar<uint>	  task_target_edge_ids,
				BufferVar<uint>	  task_source_face_prefix,
				BufferVar<uint>	  task_source_face_ids,
				BufferVar<uint>	  exact_offsets_or_counts,
				BufferVar<uint>	  exact_type_counts,
				BufferVar<uint>	  overflow_flag,
				BufferVar<uint2>  out_packed_hits,
				BufferVar<float4> out_packed_attrs,
				BufferVar<uint>	  out_packed_task_id,
				Var<uint>		  per_task_capacity,
				Var<uint>		  num_active_tasks,
				Var<uint>		  total_coarse_hits,
				Var<uint>		  total_exact_capacity,
				Var<uint>		  mode_write)
			{
				const Uint coarse_idx = dispatch_x();
				$if(coarse_idx >= total_coarse_hits | num_active_tasks == 0u)
				{
					$return();
				};
				Uint lo = 0u;
				Uint hi = num_active_tasks;
				$while(lo + 1u < hi)
				{
					const Uint mid = (lo + hi) >> 1u;
					$if(task_coarse_offsets.read(mid) <= coarse_idx)
					{
						lo = mid;
					}
					$else
					{
						hi = mid;
					};
				};
				const Uint task = lo;
				const Uint local = coarse_idx - task_coarse_offsets.read(task);
				$if(local >= task_coarse_hit_counts.read(task))
				{
					$return();
				};
				const Uint	   slice_idx = task * per_task_capacity + local;
				constexpr uint extract_mask = HitIndices::extract_mask;
				constexpr uint VF = uint(PairType::VF);
				constexpr uint EE = uint(PairType::EE);
				constexpr uint FV = uint(PairType::FV);
				const Uint2	   hit = in_slice_hits.read(slice_idx);
				const Uint	   type = hit.x >> 30u;
				const Uint	   id1 = hit.x & extract_mask;
				const Uint	   id2 = hit.y & extract_mask;
				Bool		   exact_valid = false;
				Uint		   exact_type = 3u;
				$if(type == VF)
				{
					exact_valid = fn_is_in_sorted_task_list(task_target_face_prefix, task_target_face_ids, task, id2);
					exact_type = 0u;
				}
				$else
				{
					$if(type == EE)
					{
						exact_valid = fn_is_in_sorted_task_list(task_target_edge_prefix, task_target_edge_ids, task, id2);
						exact_type = 1u;
					}
					$else
					{
						$if(type == FV)
						{
							exact_valid = fn_is_in_sorted_task_list(task_source_face_prefix, task_source_face_ids, task, id1);
							exact_type = 2u;
						};
					};
				};
				$if(exact_valid)
				{
					const Uint out_idx = exact_offsets_or_counts.atomic(task).fetch_add(1u);
					$if(mode_write == 0u)
					{
						exact_type_counts.atomic(exact_type).fetch_add(1u);
						exact_type_counts.atomic(3u).fetch_add(1u);
					}
					$else
					{
						$if(out_idx < total_exact_capacity)
						{
							out_packed_hits.write(out_idx, hit);
							out_packed_attrs.write(out_idx, in_slice_attrs.read(slice_idx));
							out_packed_task_id.write(out_idx, task);
						}
						$else
						{
							overflow_flag.atomic(0u).exchange(1u);
						};
					};
				};
			});

		compiler.compile<1>(fn_task_raycast_vf_scatter_uint2,
			[fn_segment_triangle_intersection, fn_ray_aabb_intersect, fn_is_in_range, fn_is_in_sorted_task_list](
				BufferVar<float3>		  sa_x,
				BufferVar<uint3>		  sa_faces,
				BufferVar<CompressedAABB> target_node_aabb,
				BufferVar<uint2>		  target_children,
				BufferVar<uint>			  target_object_idx,
				BufferVar<uint>			  target_is_healthy,
				BufferVar<uint>			  request_source_vert_ids,
				BufferVar<uint>			  request_task_id,
				BufferVar<uint2>		  task_target_face_range,
				BufferVar<uint>			  task_target_face_prefix,
				BufferVar<uint>			  task_target_face_ids,
				BufferVar<float3>		  task_dirs,
				Var<float>				  max_ray_dist,
				BufferVar<uint>			  task_offsets,
				BufferVar<uint2>		  out_indices,
				BufferVar<float4>		  out_attrs,
				BufferVar<uint>			  out_task_id,
				Var<uint>				  per_task_capacity,
				Var<uint>				  request_count)
			{
				const Uint request_idx = dispatch_x();
				$if(request_idx >= request_count)
				{
					$return();
				};
				$if(target_is_healthy.read(0u) == 0u)
				{
					$return();
				};
				const Uint	task = request_task_id.read(request_idx);
				const Uint2 target_range = task_target_face_range.read(task);
				$if(target_range.y == 0u)
				{
					$return();
				};
				const Float3		  direction = task_dirs.read(task);
				const Uint			  vid = request_source_vert_ids.read(request_idx);
				const Float3		  orig = sa_x.read(vid);
				const Float3		  end = orig + direction * max_ray_dist;
				const Float3		  direction_inv = 1.0f / direction;
				ArrayUInt<STACK_SIZE> stack;
				Int					  stack_ptr = 0;
				stack[stack_ptr] = 0u;
				stack_ptr += 1;
				Uint iter = 0u;
				$while(stack_ptr > 0)
				{
					stack_ptr -= 1;
					const Uint node = stack[stack_ptr];
					const auto node_aabb = target_node_aabb.read(node);
					$if(fn_ray_aabb_intersect(orig, direction_inv, node_aabb, max_ray_dist))
					{
						const Uint fid = target_object_idx.read(node);
						$if(fid != 0xffffffffu)
						{
							$if(fn_is_in_range(fid, target_range.x, target_range.y)
								& fn_is_in_sorted_task_list(task_target_face_prefix, task_target_face_ids, task, fid))
							{
								const UInt3 face = sa_faces.read(fid);
								$if(!any(face == vid))
								{
									const Float3 A = sa_x.read(face.x);
									const Float3 B = sa_x.read(face.y);
									const Float3 C = sa_x.read(face.z);
									const Float4 hit_info = fn_segment_triangle_intersection(orig, end, A, B, C);
									// Reject slightly-behind hits (time < 0) to match CPU out_time >= 0 check.
									$if(hit_info.w > 0.5f & hit_info.z >= 0.0f)
									{
										const Uint slot_start = task * per_task_capacity;
										const Uint local_idx = task_offsets.atomic(task).fetch_add(1u);
										$if(local_idx < per_task_capacity)
										{
											const Uint	   idx = slot_start + local_idx;
											constexpr uint mask_vf = uint(PairType::VF) << 30u;
											out_indices.write(idx, make_uint2(vid | mask_vf, fid));
											out_attrs.write(idx, make_float4(hit_info.x, hit_info.y, hit_info.z * max_ray_dist, 0.0f));
											out_task_id.write(idx, task);
										};
									};
								};
							};
						}
						$else
						{
							const UInt2 child = target_children.read(node);
							$if(stack_ptr < Int(STACK_SIZE))
							{
								stack[stack_ptr] = child.x;
								stack_ptr += 1;
							}
							$else
							{
								// device_log("PRP task VF traversal stack overflow at node {}, child {}, task {}", node, child.y, task);
								device_assert(false, "PRP task VF scatter traversal stack overflow.");
							};
							$if(stack_ptr < Int(STACK_SIZE))
							{
								stack[stack_ptr] = child.y;
								stack_ptr += 1;
							}
							$else
							{
								// device_log("PRP task VF traversal stack overflow at node {}, child {}, task {}", node, child.y, task);
								device_assert(false, "PRP task VF scatter traversal stack overflow.");
							};
						};
					};
					iter += 1u;
					$if(iter > 200000u)
					{
						device_assert(false, "PRP task VF scatter traversal exceeded iteration budget.");
						$break;
					};
				};
			});

		compiler.compile<1>(fn_task_raycast_ee_scatter_uint2,
			[fn_swept_edge_edge_intersection, fn_swept_plane_aabb_intersect, fn_is_in_range, fn_is_in_sorted_task_list](
				BufferVar<float3>		  sa_x,
				BufferVar<uint2>		  sa_edges,
				BufferVar<CompressedAABB> target_node_aabb,
				BufferVar<uint2>		  target_children,
				BufferVar<uint>			  target_object_idx,
				BufferVar<uint>			  target_is_healthy,
				BufferVar<uint>			  request_source_edge_ids,
				BufferVar<float>		  request_ray_max_dist,
				BufferVar<uint>			  request_task_id,
				BufferVar<uint2>		  task_target_edge_range,
				BufferVar<float>		  task_target_edge_proj_min,
				BufferVar<uint>			  task_target_edge_prefix,
				BufferVar<uint>			  task_target_edge_ids,
				BufferVar<float3>		  task_dirs,
				Var<float>				  max_ray_dist,
				BufferVar<uint>			  task_offsets,
				BufferVar<uint2>		  out_indices,
				BufferVar<float4>		  out_attrs,
				BufferVar<uint>			  out_task_id,
				Var<uint>				  per_task_capacity,
				Var<uint>				  request_count)
			{
				const Uint request_idx = dispatch_x();
				$if(request_idx >= request_count)
				{
					$return();
				};
				$if(target_is_healthy.read(0u) == 0u)
				{
					$return();
				};
				const Uint	task = request_task_id.read(request_idx);
				const Uint2 target_range = task_target_edge_range.read(task);
				const Float target_proj_min = task_target_edge_proj_min.read(task);
				$if(target_range.y == 0u)
				{
					$return();
				};
				const Float3 direction = task_dirs.read(task);
				Float		 ray_max_dist = request_ray_max_dist.read(request_idx);
				ray_max_dist = ite(ray_max_dist < def<float>(0.0f), def<float>(0.0f), ray_max_dist);
				ray_max_dist = ite(ray_max_dist > max_ray_dist, max_ray_dist, ray_max_dist);
				$if(ray_max_dist <= 0.0f)
				{
					$return();
				};
				const Uint	 eid1 = request_source_edge_ids.read(request_idx);
				const UInt2	 edge1 = sa_edges.read(eid1);
				const Float3 p0 = sa_x.read(edge1.x);
				const Float3 p1 = sa_x.read(edge1.y);
				const Float3 p0_end = p0 + direction * ray_max_dist;
				const Float3 p1_end = p1 + direction * ray_max_dist;
				const Float3 swept_min = make_float3(
					min_scalar(min_scalar(p0.x, p1.x), min_scalar(p0_end.x, p1_end.x)),
					min_scalar(min_scalar(p0.y, p1.y), min_scalar(p0_end.y, p1_end.y)),
					min_scalar(min_scalar(p0.z, p1.z), min_scalar(p0_end.z, p1_end.z)));
				const Float3 swept_max = make_float3(
					max_scalar(max_scalar(p0.x, p1.x), max_scalar(p0_end.x, p1_end.x)),
					max_scalar(max_scalar(p0.y, p1.y), max_scalar(p0_end.y, p1_end.y)),
					max_scalar(max_scalar(p0.z, p1.z), max_scalar(p0_end.z, p1_end.z)));
				const Float3 e1 = p1 - p0;
				const Float3 plane_normal_raw = cross(e1, direction);
				const Float	 plane_n_len2 = dot(plane_normal_raw, plane_normal_raw);
				const Bool	 use_plane_test = plane_n_len2 >= 1e-20f;
				Float3		 plane_normal = make_float3(0.0f);
				Float		 plane_d = 0.0f;
				$if(use_plane_test)
				{
					plane_normal = normalize(plane_normal_raw);
					plane_d = dot(plane_normal, p0);
				};
				const Float			  edge_proj0 = dot(p0, direction);
				const Float			  edge_proj1 = dot(p1, direction);
				const Float			  proj_min = max_scalar(min_scalar(edge_proj0, edge_proj1), target_proj_min - 1e-5f);
				const Float			  proj_max = max_scalar(edge_proj0, edge_proj1) + ray_max_dist * dot(direction, direction);
				ArrayUInt<STACK_SIZE> stack;
				Int					  stack_ptr = 0;
				stack[stack_ptr] = 0u;
				stack_ptr += 1;
				Uint iter = 0u;
				$while(stack_ptr > 0)
				{
					stack_ptr -= 1;
					const Uint node = stack[stack_ptr];
					const auto node_aabb = target_node_aabb.read(node);
					$if(fn_swept_plane_aabb_intersect(swept_min, swept_max, plane_normal, plane_d, use_plane_test, proj_min, proj_max, direction, node_aabb))
					{
						const Uint eid2 = target_object_idx.read(node);
						$if(eid2 != 0xffffffffu)
						{
							$if(fn_is_in_range(eid2, target_range.x, target_range.y)
								& fn_is_in_sorted_task_list(task_target_edge_prefix, task_target_edge_ids, task, eid2))
							{
								const UInt2 edge2 = sa_edges.read(eid2);
								$if(edge1.x != edge2.x & edge1.x != edge2.y & edge1.y != edge2.x & edge1.y != edge2.y)
								{
									const Float3 q0 = sa_x.read(edge2.x);
									const Float3 q1 = sa_x.read(edge2.y);
									const Float4 hit_info = fn_swept_edge_edge_intersection(p0, p1, q0, q1, swept_min, swept_max, direction, ray_max_dist);
									$if(hit_info.w > 0.5f)
									{
										const Uint slot_start = task * per_task_capacity;
										const Uint local_idx = task_offsets.atomic(task).fetch_add(1u);
										$if(local_idx < per_task_capacity)
										{
											const Uint	   idx = slot_start + local_idx;
											constexpr uint mask_ee = uint(PairType::EE) << 30u;
											out_indices.write(idx, make_uint2(eid1 | mask_ee, eid2));
											// EE: hit_info = (s, t1, t2, 1). Store (t1, t2, s, 0) -> (bary.x, bary.y, dist, _)
											out_attrs.write(idx, make_float4(hit_info.y, hit_info.z, hit_info.x, 0.0f));
											out_task_id.write(idx, task);
										};
									};
								};
							};
						}
						$else
						{
							const UInt2 child = target_children.read(node);
							$if(stack_ptr < Int(STACK_SIZE))
							{
								stack[stack_ptr] = child.x;
								stack_ptr += 1;
							}
							$else
							{
								// device_log("PRP task EE traversal stack overflow at node {}, child {}, task {}", node, child.y, task);
								device_assert(false, "PRP task VF scatter traversal stack overflow.");
							};
							$if(stack_ptr < Int(STACK_SIZE))
							{
								stack[stack_ptr] = child.y;
								stack_ptr += 1;
							}
							$else
							{
								// device_log("PRP task EE traversal stack overflow at node {}, child {}, task {}", node, child.y, task);
								device_assert(false, "PRP task VF scatter traversal stack overflow.");
							};
						};
					};
					iter += 1u;
					$if(iter > 200000u)
					{
						device_assert(false, "PRP task EE scatter traversal exceeded iteration budget.");
						$break;
					};
				};
			});

		compiler.compile<1>(fn_task_raycast_fv_scatter_uint2,
			[fn_segment_triangle_intersection, fn_ray_aabb_intersect, fn_is_in_range, fn_is_in_sorted_task_list](
				BufferVar<float3>		  sa_x,
				BufferVar<uint3>		  sa_faces,
				BufferVar<CompressedAABB> target_node_aabb,
				BufferVar<uint2>		  target_children,
				BufferVar<uint>			  target_object_idx,
				BufferVar<uint>			  target_is_healthy,
				BufferVar<uint>			  request_dst_vert_ids,
				BufferVar<uint>			  request_task_id,
				BufferVar<uint2>		  task_source_face_range,
				BufferVar<uint>			  task_source_face_prefix,
				BufferVar<uint>			  task_source_face_ids,
				BufferVar<float3>		  task_dirs,
				Var<float>				  max_ray_dist,
				BufferVar<uint>			  task_offsets,
				BufferVar<uint2>		  out_indices,
				BufferVar<float4>		  out_attrs,
				BufferVar<uint>			  out_task_id,
				Var<uint>				  per_task_capacity,
				Var<uint>				  request_count)
			{
				const Uint request_idx = dispatch_x();
				$if(request_idx >= request_count)
				{
					$return();
				};
				$if(target_is_healthy.read(0u) == 0u)
				{
					$return();
				};
				const Uint	task = request_task_id.read(request_idx);
				const Uint2 source_range = task_source_face_range.read(task);
				$if(source_range.y == 0u)
				{
					$return();
				};
				const Float3		  direction_neg = -task_dirs.read(task);
				const Uint			  vid = request_dst_vert_ids.read(request_idx);
				const Float3		  orig = sa_x.read(vid);
				const Float3		  end = orig + direction_neg * max_ray_dist;
				const Float3		  direction_inv = 1.0f / direction_neg;
				ArrayUInt<STACK_SIZE> stack;
				Int					  stack_ptr = 0;
				stack[stack_ptr] = 0u;
				stack_ptr += 1;
				Uint iter = 0u;
				$while(stack_ptr > 0)
				{
					stack_ptr -= 1;
					const Uint node = stack[stack_ptr];
					const auto node_aabb = target_node_aabb.read(node);
					$if(fn_ray_aabb_intersect(orig, direction_inv, node_aabb, max_ray_dist))
					{
						const Uint fid = target_object_idx.read(node);
						$if(fid != 0xffffffffu)
						{
							$if(fn_is_in_range(fid, source_range.x, source_range.y)
								& fn_is_in_sorted_task_list(task_source_face_prefix, task_source_face_ids, task, fid))
							{
								const UInt3 face = sa_faces.read(fid);
								$if(!any(face == vid))
								{
									const Float3 A = sa_x.read(face.x);
									const Float3 B = sa_x.read(face.y);
									const Float3 C = sa_x.read(face.z);
									const Float4 hit_info = fn_segment_triangle_intersection(orig, end, A, B, C);
									// Reject slightly-behind hits (time < 0) to match CPU out_time >= 0 check.
									$if(hit_info.w > 0.5f & hit_info.z >= 0.0f)
									{
										const Uint slot_start = task * per_task_capacity;
										const Uint local_idx = task_offsets.atomic(task).fetch_add(1u);
										$if(local_idx < per_task_capacity)
										{
											const Uint	   idx = slot_start + local_idx;
											constexpr uint mask_fv = uint(PairType::FV) << 30u;
											out_indices.write(idx, make_uint2(fid | mask_fv, vid));
											out_attrs.write(idx, make_float4(hit_info.x, hit_info.y, hit_info.z * max_ray_dist, 0.0f));
											out_task_id.write(idx, task);
										};
									};
								};
							};
						}
						$else
						{
							const UInt2 child = target_children.read(node);
							$if(stack_ptr < Int(STACK_SIZE))
							{
								stack[stack_ptr] = child.x;
								stack_ptr += 1;
							};
							$if(stack_ptr < Int(STACK_SIZE))
							{
								stack[stack_ptr] = child.y;
								stack_ptr += 1;
							};
						};
					};
					iter += 1u;
					$if(iter > 200000u)
					{
						device_assert(false, "PRP task FV scatter traversal exceeded iteration budget.");
						$break;
					};
				};
			});
	}

	void DevicePRPPrecomputeShaders::compile(AsyncCompiler& compiler, MeshData<luisa::compute::Buffer>* device_mesh_data)
	{
		if (compiled)
			return;

		compile_normals(compiler);
		compile_buffer_utils(compiler);
		compile_hit_compaction(compiler);
		compile_product_adjacency_and_union_find(compiler, device_mesh_data);
		compile_cluster_culling(compiler);
		compile_batch_keyed_csr(compiler);
		compile_exact_filter_and_raycast(compiler);

		compiled = true;
	}

	void device_compute_prp_normals(luisa::compute::Device& device,
		luisa::compute::Stream&								stream,
		UntanglingData<luisa::compute::Buffer>*				device_untangling_data,
		MeshData<luisa::compute::Buffer>*					device_mesh_data,
		SimulationData<luisa::compute::Buffer>*				device_sim_data,
		const uint											num_faces,
		const uint											num_edges,
		const uint											num_verts,
		DevicePRPPrecomputeShaders&							prp_shaders)
	{
		(void)device;
		if (num_faces != 0u)
		{
			stream << prp_shaders
						  .fn_compute_face_normals(device_mesh_data->sa_faces,
							  device_sim_data->sa_x,
							  device_mesh_data->sa_face_mesh_id,
							  device_mesh_data->sa_mesh_orientation,
							  device_untangling_data->prp_face_normal)
						  .dispatch(num_faces);
		}
		if (num_edges != 0u)
		{
			stream << prp_shaders
						  .fn_compute_edge_normals(device_mesh_data->edge_adj_faces_csr,
							  device_mesh_data->sa_rest_face_area,
							  device_untangling_data->prp_face_normal,
							  device_untangling_data->prp_edge_normal)
						  .dispatch(num_edges);
		}
		if (num_verts != 0u)
		{
			stream << prp_shaders
						  .fn_compute_vert_normals(device_mesh_data->sa_vert_adj_faces_csr,
							  device_mesh_data->sa_rest_face_area,
							  device_untangling_data->prp_face_normal,
							  device_untangling_data->prp_vert_normal)
						  .dispatch(num_verts);
		}
	}

	namespace
	{
		// Initial per-task coarse-hit storage estimate for the batched GPU raycast retry loop.
		uint gpu_prp_estimate_initial_per_task_capacity(
			const std::vector<uint>& vf_request_task_id,
			const std::vector<uint>& ee_request_task_id,
			const std::vector<uint>& fv_request_task_id,
			const uint				 num_contours_active)
		{
			std::vector<uint> request_counts(num_contours_active, 0u);
			auto			  add_request_counts = [&](const std::vector<uint>& request_task_ids)
			{
				for (const uint task_idx : request_task_ids)
				{
					if (task_idx >= num_contours_active)
						LUISA_ERROR("GPU PRP request task id {} out of range for {} active tasks.", task_idx, num_contours_active);
					request_counts[task_idx]++;
				}
			};
			add_request_counts(vf_request_task_id);
			add_request_counts(ee_request_task_id);
			add_request_counts(fv_request_task_id);
			const uint64_t total_request_count = static_cast<uint64_t>(vf_request_task_id.size()) + static_cast<uint64_t>(ee_request_task_id.size()) + static_cast<uint64_t>(fv_request_task_id.size());
			const uint	   max_request_count = request_counts.empty() ? 0u : *std::max_element(request_counts.begin(), request_counts.end());
			const uint	   avg_request_count = static_cast<uint>((total_request_count + num_contours_active - 1u) / std::max<uint>(num_contours_active, 1u));
			const bool	   dense_micro_batch = num_contours_active <= 8u;
			const bool	   large_batch = num_contours_active >= 24u;
			const uint	   min_initial_capacity = large_batch ? 8192u : (dense_micro_batch ? 4096u : 1024u);
			const uint	   capacity_multiplier = large_batch ? 16u : (dense_micro_batch ? 8u : 4u);
			const uint	   avg_capacity_multiplier = large_batch ? 8u : (dense_micro_batch ? 4u : 2u);
			const uint	   request_based_capacity = std::max<uint>({ min_initial_capacity,
				capacity_multiplier * std::max(max_request_count, avg_request_count),
				avg_capacity_multiplier * avg_request_count });
			return std::min<uint>(request_based_capacity, 1u << 20u);
		}

		// Reconstruct a host HitInfo from a packed GPU hit record.
		RayCasting::HitInfo gpu_prp_reconstruct_hit_info(
			const MeshData<std::vector>* host_mesh_data,
			const uint2 ids, const float4 attrs, const float3 dir)
		{
			const auto type = get_pair_type(ids);
			const uint id1 = ids.x & HitIndices::extract_mask;
			const uint id2 = ids.y & HitIndices::extract_mask;
			if (type == PairType::VF)
			{
				const uint3 face = host_mesh_data->sa_faces[id2];
				return RayCasting::HitInfo::make_vf_hit(0, 0, id1, id2, face, luisa::make_float2(1.0f - attrs.x - attrs.y, attrs.x), dir, attrs.z);
			}
			if (type == PairType::EE)
			{
				const uint2 edge1 = host_mesh_data->sa_edges[id1];
				const uint2 edge2 = host_mesh_data->sa_edges[id2];
				return RayCasting::HitInfo::make_ee_hit(0, 0, id1, id2, edge1, edge2, 1.0f - attrs.x, 1.0f - attrs.y, dir, attrs.z);
			}
			const uint3 face = host_mesh_data->sa_faces[id1];
			return RayCasting::HitInfo::make_fv_hit(0, 0, id2, id1, face, luisa::make_float2(1.0f - attrs.x - attrs.y, attrs.x), dir, attrs.z);
		}
	} // namespace

	void device_resolve_intersections_PRP(luisa::compute::Device& device,
		luisa::compute::Stream&									  stream,
		CollisionData<luisa::compute::Buffer>*					  device_collision_data,
		CollisionData<std::vector>*								  host_collision_data,
		UntanglingData<luisa::compute::Buffer>*					  device_untangling_data,
		UntanglingData<std::vector>*							  host_untangling_data,
		MeshData<luisa::compute::Buffer>*						  device_mesh_data,
		MeshData<std::vector>*									  host_mesh_data,
		SimulationData<luisa::compute::Buffer>*					  device_sim_data,
		SimulationData<std::vector>*							  host_sim_data,
		LbvhData<luisa::compute::Buffer>*						  device_lbvh_face_data,
		LbvhData<luisa::compute::Buffer>*						  device_lbvh_edge_data,
		DevicePRPPrecomputeShaders&								  prp_shaders)
	{
		prp_debug_reset();
		auto& iteration_stats = get_scene_params().prp_debug_info.uint_stats;
		iteration_stats["prp_ray_retry_count"] = 0u;
		iteration_stats["prp_ray_max_task_hits"] = 0u;
		iteration_stats["prp_ray_max_task_capacity"] = 0u;
		PRP_PROFILE_RESET();
		PRP_PROFILE_SCOPE("device_resolve_intersections_PRP.total");

		if (device_lbvh_face_data == nullptr || device_lbvh_edge_data == nullptr)
		{
			LUISA_ERROR("Device PRP traversal requires LBVH data pointers.");
			return;
		}

		const uint num_verts = host_mesh_data->num_verts;
		const uint num_edges = host_mesh_data->num_edges;
		const uint num_faces = host_mesh_data->num_faces;
		const uint num_pairs = host_collision_data->narrow_phase_collision_count[1];
		uint	   dynamic_resize_count = 0u;
		auto	   dynamic_resize = [&](auto& buffer, const uint required_size, const std::string_view name)
		{
			const auto old_size = buffer.size();
			Initializer::dynamic_resize_template(device, buffer, required_size, name);
			if (buffer.size() != old_size)
				++dynamic_resize_count;
		};

		auto&		ef_list = host_collision_data->narrow_phase_list_ef;
		const auto& ef_indices = host_collision_data->narrow_phase_list_ef_indices;
		const auto& contours = host_untangling_data->intersection_contours;

		// EF resize
		{
			dynamic_resize(device_untangling_data->ef_pair_mesh_index, num_pairs, "ef_pair_mesh_index");
			dynamic_resize(device_untangling_data->ef_pair_contour_index, num_pairs, "ef_pair_contour_index");
			dynamic_resize(device_untangling_data->ef_pair_length, num_pairs, "ef_pair_length");
			dynamic_resize(device_untangling_data->ef_pair_pos_3D, num_pairs, "ef_pair_pos_3D");
		}

		// Contour construction
		{
			PRP_PROFILE_SCOPE("build_intersection_contours");
			intersection_contour_construction_from_ext_adjacent(host_untangling_data, host_collision_data);
			if (host_untangling_data->num_contours.empty())
				host_untangling_data->num_contours.resize(1);
			host_untangling_data->num_contours.front() = host_untangling_data->intersection_contours.size();

			if (num_pairs != 0u)
			{
				stream << device_untangling_data->ef_pair_mesh_index.view(0, num_pairs).copy_from(host_untangling_data->ef_pair_mesh_index.data())
					   << device_untangling_data->ef_pair_contour_index.view(0, num_pairs).copy_from(host_untangling_data->ef_pair_contour_index.data())
					   << device_untangling_data->ef_pair_length.view(0, num_pairs).copy_from(host_untangling_data->ef_pair_length.data())
					   << device_untangling_data->ef_pair_pos_3D.view(0, num_pairs).copy_from(host_untangling_data->ef_pair_pos_3D.data());
			}
		}

		const uint num_contours = host_untangling_data->intersection_contours.size();
		{
			uint64_t h = 1469598103934665603ull;
			for (const auto& contour : contours)
				det_trace_detail::hash_pod_vector(h, contour);
			det_trace_detail::hash_pod_vector(h, host_untangling_data->ef_pair_mesh_index);
			det_trace_detail::trace("P2_contours", h);
		}
		uint classified_contour_count = 0;
		uint num_loop_pairs = host_untangling_data->loop_pairs_vertex.size();

		// Contour classification
		{
			PRP_PROFILE_SCOPE("classify_contours");
			const auto& ef_pair_mesh_index = host_untangling_data->ef_pair_mesh_index;

			std::vector<luisa::ubyte> pair_is_boundary(num_pairs, false);
			CpuParallel::parallel_for(0, num_pairs,
				[&](const uint pair_idx)
				{
					const uint eid = ef_indices[pair_idx].x;
					pair_is_boundary[pair_idx] = host_mesh_data->edge_adj_faces_ext[eid].size() <= 1 ? 1u : 0u;
				});

			auto& intersection_contours_info = host_untangling_data->intersection_contours_info;
			intersection_contours_info.resize(num_contours);
			for (uint contour_idx = 0; contour_idx < num_contours; contour_idx++)
			{
				const auto contour_type = classify_contour_type(ef_list, ef_pair_mesh_index, host_untangling_data->ef_pair_adj_pairs_ext,
					pair_is_boundary, intersection_contours_info[contour_idx].loop_pairs_is_boundary, contours[contour_idx], contour_idx);
				auto&	   contour_info = intersection_contours_info[contour_idx];
				contour_info.contour_type = contour_type;
				// A role is closed exactly when it owns at least one EF and none of its EF graph vertices has degree one.
				for (uint role = 0u; role < 2u; ++role)
				{
					bool has_role_pair = false;
					bool every_pair_has_nonunit_degree = true;
					for (const uint pair_idx : contours[contour_idx])
					{
						if (ef_pair_mesh_index[pair_idx] != role)
							continue;
						has_role_pair = true;
						// A loop-vertex EF encodes a contour cell, not an open endpoint; its degree-one storage form therefore must not make the role open.
						if (!ef_list[pair_idx].get_is_loop_vertex()
							&& host_untangling_data->ef_pair_adj_pairs_ext[pair_idx].size() == 1u)
							every_pair_has_nonunit_degree = false;
					}
					contour_info.mesh_is_closed[role] = has_role_pair && every_pair_has_nonunit_degree;
				}
			}
			classified_contour_count = std::count_if(intersection_contours_info.begin(), intersection_contours_info.end(),
				[](const auto& info)
				{ return info.contour_type != ContourType::Undifined; });
		}

		// Contour CSR and range construction
		std::vector<std::array<uint, 2>> list_object_ids(num_contours);
		std::vector<std::array<uint, 2>> list_topology_component_ids(num_contours,
			{ std::numeric_limits<uint>::max(), std::numeric_limits<uint>::max() });
		{
			PRP_PROFILE_SCOPE("build_intersection_contours_csr");
			for (uint contour_idx = 0u; contour_idx < num_contours; ++contour_idx)
			{
				if (contours[contour_idx].empty())
					LUISA_ERROR("Contour {} is empty!", contour_idx);
				const uint first_pair_idx = contours[contour_idx].front();
				const auto identity = prp_get_contour_side_identity(
					ef_list[first_pair_idx],
					host_untangling_data->ef_pair_mesh_index[first_pair_idx],
					host_mesh_data);
				list_object_ids[contour_idx] = identity.object_ids;
				list_topology_component_ids[contour_idx] = identity.topology_component_ids;
			}
		}

		prp_debug_set_meta_keys(num_verts, num_edges, num_faces, num_pairs, num_contours, classified_contour_count, num_loop_pairs);

		// IterationDebugRecord-compatible aliases (mirrors CPU path in intersection_resolver2.cpp).
		if (get_scene_params().prp_debug)
		{
			auto& dbg = get_scene_params().prp_debug_info;
			dbg.uint_stats["contour_count"] = num_contours;
			dbg.uint_stats["ef_pair_count"] = num_pairs;
			dbg.uint_stats["loop_vertex_count"] = num_loop_pairs;
			dbg.bool_stats["all_contours_classifiable_into_7_types"] = (classified_contour_count == num_contours);
		}

		// Compute normals
		std::vector<float3> host_vert_normal;
		std::vector<float3> host_edge_normal;
		std::vector<float3> host_face_normal;
		{
			PRP_PROFILE_SCOPE("utils.compute_normals");
			device_compute_prp_normals(device, stream, device_untangling_data, device_mesh_data, device_sim_data, num_faces, num_edges, num_verts, prp_shaders);
			host_vert_normal.resize(num_verts);
			host_edge_normal.resize(num_edges);
			host_face_normal.resize(num_faces);
			if (num_verts > 0)
				stream << device_untangling_data->prp_vert_normal.copy_to(host_vert_normal.data());
			if (num_edges > 0)
				stream << device_untangling_data->prp_edge_normal.copy_to(host_edge_normal.data());
			if (num_faces > 0)
				stream << device_untangling_data->prp_face_normal.copy_to(host_face_normal.data());
			stream << luisa::compute::synchronize();
		}

		// Vertex-contour adjacency and vertex-in-boundary flags
		std::vector<std::vector<uint>> vert_in_boundary_flag(num_verts);
		{
			PRP_PROFILE_SCOPE("build_vert_adj_boundaries");
			std::vector<uint> host_csr;
			{
				for (uint pair_idx = 0u; pair_idx < num_pairs; ++pair_idx)
				{
					const auto& pair = ef_list[pair_idx];
					if (pair.get_is_loop_vertex())
						continue;
					const uint	mesh_idx = host_untangling_data->ef_pair_mesh_index[pair_idx];
					const uint	contour_idx = host_untangling_data->ef_pair_contour_index[pair_idx];
					const uint2 edge = pair.get_edge();
					const uint3 face = pair.get_face();
					const uint	edge_side_flag = 2u * contour_idx + mesh_idx;
					const uint	face_side_flag = edge_side_flag ^ 1u;
					vert_in_boundary_flag[edge.x].push_back(edge_side_flag);
					vert_in_boundary_flag[edge.y].push_back(edge_side_flag);
					vert_in_boundary_flag[face.x].push_back(face_side_flag);
					vert_in_boundary_flag[face.y].push_back(face_side_flag);
					vert_in_boundary_flag[face.z].push_back(face_side_flag);
				}
				CpuParallel::parallel_for(0u, num_verts,
					[&](uint vid)
					{
						auto& flags = vert_in_boundary_flag[vid];
						std::sort(flags.begin(), flags.end());
						flags.erase(std::unique(flags.begin(), flags.end()), flags.end());
					});

				Initializer::upload_2d_csr_from(host_csr, vert_in_boundary_flag);
				const uint total_adj_entries = static_cast<uint>(host_csr.size());
				dynamic_resize(device_untangling_data->prp_vert_adj_contours_csr, total_adj_entries, "prp_vert_adj_contours_csr");
				stream << device_untangling_data->prp_vert_adj_contours_csr.view(0, total_adj_entries).copy_from(host_csr.data());
			}
		}

		const auto&		  sa_rest_face_area = host_mesh_data->sa_rest_face_area;
		const auto&		  sa_rest_edge_area = host_mesh_data->sa_rest_edge_area;
		const auto&		  sa_rest_vert_area = host_mesh_data->sa_rest_vert_area;
		const auto&		  sa_edges = host_mesh_data->sa_edges;
		const auto&		  sa_faces = host_mesh_data->sa_faces;
		std::vector<uint> sa_verts(num_verts, 0u);
		std::iota(sa_verts.begin(), sa_verts.end(), 0u);

		// Materialized PRP results shared by discrete screening and optional post-screening direction refinement.
		std::vector<PRPMinContourHitInfo> list_min_hit_info(num_contours);
		std::vector<double>				  list_raycast_times(num_contours, 0.0);
		std::vector<uint>				  final_contour_pair_count(num_contours, 0u);
		for (uint contour_idx = 0u; contour_idx < num_contours; ++contour_idx)
			final_contour_pair_count[contour_idx] = static_cast<uint>(contours[contour_idx].size());
		uint				 total_mismatch_count = 0u;
		uint				 pass_stage_count = 2u; // preprocessing + boundary/support preparation completed
		bool				 timeout_detected = false;
		uint				 timeout_contour_count = 0u;
		constexpr float		 contour_timeout_ms_threshold = 3000.0f;
		RayCasting::BVHCache face_bvh_cache;
		RayCasting::BVHCache edge_bvh_cache;
		RayCasting::BVHCache reverse_face_bvh_cache;

		std::vector<uint> active_contours;
		active_contours.reserve(num_contours);
		for (uint contour_idx = 0u; contour_idx < num_contours; ++contour_idx)
		{
			if (get_scene_params().should_contour_skip(contour_idx))
				continue;
			active_contours.push_back(contour_idx);
		}

		// All GPU-specific fields were previously commented out, so this type is layout-identical to PrpStandardContourRuntime.
		using GpuPrpStandardContourRuntime = PrpStandardContourRuntime;

		auto find_contours_hits_batched = [&]() -> bool
		{
			PRP_PROFILE_SCOPE("find_contours_hits_batched");

			LUISA_INFO("  GPU PRP contour evaluation: global batched source elements across contours.");

			auto upload_buffer = [&](auto& buffer, const auto& values, const char* name)
			{
				dynamic_resize(buffer, std::max<uint>(static_cast<uint>(values.size()), 1u), name);
				if (!values.empty())
					stream << buffer.view(0u, values.size()).copy_from(values.data());
			};

			std::vector<uint>  vf_request_src_ids;
			std::vector<uint>  vf_request_task_id;
			std::vector<uint>  ee_request_src_ids;
			std::vector<float> ee_request_ray_max_dist;
			std::vector<uint>  ee_request_task_id;
			std::vector<uint>  fv_request_dst_ids;
			std::vector<uint>  fv_request_task_id;
			struct GpuCandidateUploadCache
			{
				bool				  valid = false;
				std::vector<uint64_t> signature;
			};
			GpuCandidateUploadCache candidate_upload_cache;
			// Cache per-level boundary tables and device uploads; they are independent of combo direction.
			struct GpuStaticLevelUploadCache
			{
				bool				  valid = false;
				std::vector<uint64_t> signature;
			};
			GpuStaticLevelUploadCache static_level_upload_cache;

			auto update_runtime_candidates = [&](const std::vector<GpuPrpStandardContourRuntime>& runtimes,
												 const std::vector<uint>&						  runtime_indices,
												 const uint										  combo_idx,
												 const bool										  use_partial)
			{
				ee_request_ray_max_dist.clear();

				auto build_signature = [&]()
				{
					std::vector<uint64_t> signature;
					signature.reserve(2u + runtime_indices.size() * 32u);
					signature.push_back(use_partial ? 1ull : 0ull);
					signature.push_back(static_cast<uint64_t>(runtime_indices.size()));
					auto append_vector_signature = [&](const std::vector<uint>& values)
					{
						signature.push_back(static_cast<uint64_t>(values.size()));
						signature.push_back(values.empty() ? std::numeric_limits<uint>::max() : values.front());
						signature.push_back(values.empty() ? std::numeric_limits<uint>::max() : values.back());
					};
					for (const uint contour_idx : runtime_indices)
					{
						const auto& runtime = runtimes[contour_idx];
						const auto& active_verts = use_partial ? runtime.partial_candidate_verts : runtime.mesh_candidate_verts;
						const auto& active_edges = use_partial ? runtime.partial_candidate_edges : runtime.mesh_candidate_edges;
						const auto& active_faces = use_partial ? runtime.partial_candidate_faces : runtime.mesh_candidate_faces;
						signature.push_back(contour_idx);
						signature.push_back(runtime.contour_idx);
						signature.push_back(runtime.curr_k_src);
						signature.push_back(runtime.curr_k_dst);
						append_vector_signature(active_verts[0]);
						append_vector_signature(active_edges[0]);
						append_vector_signature(active_faces[0]);
						append_vector_signature(active_verts[1]);
						append_vector_signature(active_edges[1]);
						append_vector_signature(active_faces[1]);
					}
					return signature;
				};

				const auto signature = build_signature();
				const bool rebuild_static_upload = !candidate_upload_cache.valid || signature != candidate_upload_cache.signature;
				if (rebuild_static_upload)
				{
					PRP_PROFILE_SCOPE("evaluate_current_runtime_directions.update_runtime_candidates_pack_upload.static");
					vf_request_src_ids.clear();
					vf_request_task_id.clear();
					ee_request_src_ids.clear();
					ee_request_task_id.clear();
					fv_request_dst_ids.clear();
					fv_request_task_id.clear();

					std::vector<uint>  task_contour_idx;
					std::vector<uint2> task_target_face_range;
					std::vector<uint2> task_target_edge_range;
					std::vector<uint2> task_source_face_range;
					std::vector<uint>  task_src_face_prefix(1u, 0u);
					std::vector<uint>  task_dst_face_prefix(1u, 0u);
					std::vector<uint>  task_dst_edge_prefix(1u, 0u);
					std::vector<uint>  task_src_face_ids;
					std::vector<uint>  task_dst_face_ids;
					std::vector<uint>  task_dst_edge_ids;

					uint task_idx = 0;
					for (const uint contour_idx : runtime_indices)
					{
						const auto& runtime = runtimes[contour_idx];
						const auto& active_verts = use_partial ? runtime.partial_candidate_verts : runtime.mesh_candidate_verts;
						const auto& active_edges = use_partial ? runtime.partial_candidate_edges : runtime.mesh_candidate_edges;
						const auto& active_faces = use_partial ? runtime.partial_candidate_faces : runtime.mesh_candidate_faces;

						task_contour_idx.push_back(runtime.contour_idx);
						task_target_face_range.push_back(prp_compute_dense_span(active_faces[1], runtime.mesh_candidate_face_range[1]));
						task_target_edge_range.push_back(prp_compute_dense_span(active_edges[1], runtime.mesh_candidate_edge_range[1]));
						task_source_face_range.push_back(prp_compute_dense_span(active_faces[0], runtime.mesh_candidate_face_range[0]));

						task_src_face_ids.insert(task_src_face_ids.end(), active_faces[0].begin(), active_faces[0].end());
						task_dst_face_ids.insert(task_dst_face_ids.end(), active_faces[1].begin(), active_faces[1].end());
						task_dst_edge_ids.insert(task_dst_edge_ids.end(), active_edges[1].begin(), active_edges[1].end());
						task_src_face_prefix.push_back(static_cast<uint>(task_src_face_ids.size()));
						task_dst_face_prefix.push_back(static_cast<uint>(task_dst_face_ids.size()));
						task_dst_edge_prefix.push_back(static_cast<uint>(task_dst_edge_ids.size()));

						for (const uint vid : active_verts[0])
						{
							vf_request_src_ids.push_back(vid);
							vf_request_task_id.push_back(task_idx);
						}
						for (const uint eid : active_edges[0])
						{
							ee_request_src_ids.push_back(eid);
							ee_request_task_id.push_back(task_idx);
						}
						for (const uint vid : active_verts[1])
						{
							fv_request_dst_ids.push_back(vid);
							fv_request_task_id.push_back(task_idx);
						}
						task_idx += 1;
					}

					upload_buffer(device_untangling_data->prp_batch_task_contour_idx, task_contour_idx, "prp_batch_task_contour_idx");
					upload_buffer(device_untangling_data->prp_batch_task_target_face_range, task_target_face_range, "prp_batch_task_target_face_range");
					upload_buffer(device_untangling_data->prp_batch_task_target_edge_range, task_target_edge_range, "prp_batch_task_target_edge_range");
					upload_buffer(device_untangling_data->prp_batch_task_source_face_range, task_source_face_range, "prp_batch_task_source_face_range");
					upload_buffer(device_untangling_data->prp_batch_task_src_face_prefix, task_src_face_prefix, "prp_batch_task_src_face_prefix");
					upload_buffer(device_untangling_data->prp_batch_task_src_face_ids, task_src_face_ids, "prp_batch_task_src_face_ids");
					upload_buffer(device_untangling_data->prp_batch_task_dst_face_prefix, task_dst_face_prefix, "prp_batch_task_dst_face_prefix");
					upload_buffer(device_untangling_data->prp_batch_task_dst_face_ids, task_dst_face_ids, "prp_batch_task_dst_face_ids");
					upload_buffer(device_untangling_data->prp_batch_task_dst_edge_prefix, task_dst_edge_prefix, "prp_batch_task_dst_edge_prefix");
					upload_buffer(device_untangling_data->prp_batch_task_dst_edge_ids, task_dst_edge_ids, "prp_batch_task_dst_edge_ids");
					upload_buffer(device_untangling_data->prp_batch_vf_request_src_ids, vf_request_src_ids, "prp_batch_vf_request_src_ids");
					upload_buffer(device_untangling_data->prp_batch_vf_request_task_id, vf_request_task_id, "prp_batch_vf_request_task_id");
					upload_buffer(device_untangling_data->prp_batch_ee_request_src_ids, ee_request_src_ids, "prp_batch_ee_request_src_ids");
					upload_buffer(device_untangling_data->prp_batch_ee_request_task_id, ee_request_task_id, "prp_batch_ee_request_task_id");
					upload_buffer(device_untangling_data->prp_batch_fv_request_dst_ids, fv_request_dst_ids, "prp_batch_fv_request_dst_ids");
					upload_buffer(device_untangling_data->prp_batch_fv_request_task_id, fv_request_task_id, "prp_batch_fv_request_task_id");

					candidate_upload_cache.valid = true;
					candidate_upload_cache.signature = signature;
				}

				std::vector<float> task_target_edge_proj_min;
				task_target_edge_proj_min.reserve(runtime_indices.size());
				ee_request_ray_max_dist.reserve(ee_request_src_ids.size());
				{
					PRP_PROFILE_SCOPE("evaluate_current_runtime_directions.update_runtime_candidates_pack_upload.dynamic");
					for (const uint contour_idx : runtime_indices)
					{
						const auto&	 runtime = runtimes[contour_idx];
						const float3 dir = runtime.contour_dirs[combo_idx];
						const float	 dir_len2 = luisa::dot(dir, dir);
						const auto&	 active_edges = use_partial ? runtime.partial_candidate_edges : runtime.mesh_candidate_edges;
						const float2 target_edge_projection = prp_compute_edge_projection_bounds(active_edges[1], dir, *host_mesh_data, *host_sim_data);
						task_target_edge_proj_min.push_back(target_edge_projection.x);
						for (const uint eid : active_edges[0])
							ee_request_ray_max_dist.push_back(prp_compute_ee_ray_max_dist(eid, target_edge_projection, dir, dir_len2, *host_mesh_data, *host_sim_data));
						;
					}
					upload_buffer(device_untangling_data->prp_batch_task_target_edge_proj_min, task_target_edge_proj_min, "prp_batch_task_target_edge_proj_min");
					upload_buffer(device_untangling_data->prp_batch_ee_request_ray_max_dist, ee_request_ray_max_dist, "prp_batch_ee_request_ray_max_dist");
				}
			};

			auto build_standard_runtime = [&](const uint contour_idx, const std::vector<uint>& contour, GpuPrpStandardContourRuntime& runtime,
											  const std::vector<uint8_t>* pair_mesh_flip = nullptr)
			{
				if (contour.empty())
				{
					runtime.contour_idx = contour_idx;
					runtime.contour.clear();
					return;
				}
				if (contour_idx >= list_object_ids.size()
					|| contour_idx >= list_topology_component_ids.size())
					LUISA_ERROR("Contour {} identity is missing from GPU PRP setup.", contour_idx);
				prp_initialize_runtime_identity_and_geometry(
					runtime,
					contour_idx,
					contour,
					host_collision_data,
					host_untangling_data,
					ef_indices,
					host_mesh_data,
					host_sim_data,
					pair_mesh_flip);

				runtime.edge_contains_baries.resize(num_edges);
				runtime.face_contains_baries.resize(num_faces);
				{
					for (uint pi = 0u; pi < contour.size(); ++pi)
					{
						const uint	pair_idx = contour[pi];
						const uint	boundary_contour_idx = host_untangling_data->ef_pair_contour_index[pair_idx];
						const uint2 ef = ef_indices[pair_idx];
						const float bary = ef_list[pair_idx].get_edge_bary().x;
						runtime.edge_contains_baries[ef.x].push_back({ boundary_contour_idx, bary });
						runtime.face_contains_baries[ef.y].push_back({ boundary_contour_idx, ef_list[pair_idx].get_face_bary() });
					}
					CpuParallel::parallel_for(0u, num_edges,
						[&](const uint eid)
						{
							auto& pairs = runtime.edge_contains_baries[eid];
							std::sort(pairs.begin(), pairs.end());
						});
				}
			};

			auto evaluate_batched_combos = [&](std::vector<GpuPrpStandardContourRuntime>& runtimes,
											   const std::vector<uint>&					  runtime_indices,
											   const uint								  combo_idx,
											   const bool								  use_partial) -> std::vector<PRPMinContourHitInfo>
			{
				PRP_PROFILE_SCOPE("evaluate_batched_combos");
				std::vector<PRPMinContourHitInfo> contour_results(num_contours);
				if (runtime_indices.empty())
					return contour_results;

				struct TaskInfo
				{
					uint contour_idx;
				};
				std::vector<TaskInfo> contour_tasks;
				std::vector<float3>	  contour_dirs;
				for (const uint contour_idx : runtime_indices)
				{
					const auto& runtime = runtimes[contour_idx];
					if (!std::binary_search(runtime.active_combos.begin(), runtime.active_combos.end(), combo_idx))
						LUISA_ERROR("Combo idx {} is not active for contour {}", combo_idx, contour_idx);
					if (use_partial && runtime.combo_converged[combo_idx] != 0u)
						LUISA_ERROR("Combo idx {} is already converged for contour {}", combo_idx, contour_idx);
					contour_tasks.push_back({ contour_idx });
					contour_dirs.push_back(runtime.contour_dirs[combo_idx]);
				}
				upload_buffer(device_untangling_data->prp_batch_combo_dirs, contour_dirs, "prp_batch_combo_dirs");

				const uint num_contours_active = static_cast<uint>(contour_tasks.size());
				if (num_contours_active == 0u)
					return contour_results;

				auto& shaders = prp_shaders;

				const uint num_active_tasks_256 = (std::max(num_contours_active, 1u) + 255u) & ~255u;
				dynamic_resize(device_untangling_data->prp_total_counts, std::max<uint>(2u * num_active_tasks_256 + 8u + 5u * num_active_tasks_256, 100u), "prp_total_counts");
				dynamic_resize(device_untangling_data->prp_runtime_count, num_active_tasks_256, "prp_runtime_count");
				dynamic_resize(device_untangling_data->prp_runtime_prefix, num_active_tasks_256, "prp_runtime_prefix");
				auto task_count_buffer = device_untangling_data->prp_runtime_count.view(0u, num_active_tasks_256);
				auto task_count_host_view = device_untangling_data->prp_runtime_count.view(0u, num_contours_active);
				auto task_packed_offsets_buffer = device_untangling_data->prp_runtime_prefix.view(0u, num_active_tasks_256);

				const float		  kMaxRayDist = 1.0f;
				auto&			  raw_slice_buf = device_untangling_data->prp_batch_uint2_indices;
				auto&			  raw_slice_attrs_buf = device_untangling_data->prp_batch_hit_attrs;
				auto&			  raw_slice_task_id_buf = device_untangling_data->prp_batch_hit_combo_id;
				auto&			  raw_packed_buf = device_untangling_data->prp_runtime_uint2_indices;
				auto&			  raw_packed_attrs_buf = device_untangling_data->prp_runtime_hit_attrs;
				auto&			  raw_packed_task_id_buf = device_untangling_data->prp_batch_compact_task_id;
				std::vector<uint> contour_raw_hit_counts(num_contours_active, 0u);
				std::vector<uint> contour_offsets(num_contours_active, 0u);
				// Defer verification readback checks to the next unavoidable cluster-selection synchronize.
				uint exact_scatter_overflow = 0u;

				// Keyed-CSR and union-find verification copies share the cluster-selection synchronize.
				uint direct_keyed_overflow = 0u;
				uint direct_keyed_fill_overflow = 0u;
				uint direct_keyed_total_entries = 0u;
				uint cc_converged_flag = 0u;
				// Captured for the deferred verification check after the selection synchronize; the original const lives inside the keyed-CSR scope.
				uint direct_keyed_hash_capacity_captured = 0u;

				static constexpr uint kMaxRaycastCapacity = 1u << 20u;
				static constexpr uint kMaxRaycastRetryAttempts = 8u;
				uint				  per_task_capacity = gpu_prp_estimate_initial_per_task_capacity(vf_request_task_id, ee_request_task_id, fv_request_task_id, num_contours_active);
				uint				  total_coarse_hits = 0u;
				std::array<uint, 4u>  exact_type_counts = { 0u, 0u, 0u, 0u };
				uint				  total_raw_hits = 0u;
				std::vector<uint>	  contour_coarse_hit_counts(num_contours_active, 0u);
				bool				  raycast_ready = false;
				for (uint raycast_attempt = 0u; raycast_attempt < kMaxRaycastRetryAttempts; ++raycast_attempt)
				{
					PRP_PROFILE_SCOPE(fmt::format("evaluate_batched_combos.perform_raycasting_single_traversal_{}", raycast_attempt));
					const uint total_capacity = num_contours_active * per_task_capacity;
					dynamic_resize(raw_slice_buf, std::max(total_capacity, 1u), "prp_batch_uint2_indices");
					dynamic_resize(raw_slice_attrs_buf, std::max(total_capacity, 1u), "prp_batch_hit_attrs");
					dynamic_resize(raw_slice_task_id_buf, std::max(total_capacity, 1u), "prp_batch_hit_combo_id");
					// Derive per-task slice bases on device so hit counters need no slice-offset upload or host round trip.
					stream << shaders.fn_reset_uint(task_count_buffer).dispatch(num_active_tasks_256);

					if (!vf_request_src_ids.empty())
					{
						PRP_PROFILE_SCOPE("evaluate_batched_combos.perform_raycasting.raycast_VF");
						stream << shaders.fn_task_raycast_vf_scatter_uint2(device_sim_data->sa_x, device_mesh_data->sa_faces, device_lbvh_face_data->sa_node_aabb_v2, device_lbvh_face_data->sa_children, device_lbvh_face_data->sa_object_idx, device_lbvh_face_data->sa_is_healthy, device_untangling_data->prp_batch_vf_request_src_ids, device_untangling_data->prp_batch_vf_request_task_id, device_untangling_data->prp_batch_task_target_face_range, device_untangling_data->prp_batch_task_dst_face_prefix, device_untangling_data->prp_batch_task_dst_face_ids, device_untangling_data->prp_batch_combo_dirs, kMaxRayDist, task_count_buffer, raw_slice_buf, raw_slice_attrs_buf, raw_slice_task_id_buf, per_task_capacity, static_cast<uint>(vf_request_src_ids.size())).dispatch(vf_request_src_ids.size());
					}
					if (!ee_request_src_ids.empty())
					{
						PRP_PROFILE_SCOPE("evaluate_batched_combos.perform_raycasting.raycast_EE");
						stream << shaders.fn_task_raycast_ee_scatter_uint2(device_sim_data->sa_x, device_mesh_data->sa_edges, device_lbvh_edge_data->sa_node_aabb_v2, device_lbvh_edge_data->sa_children, device_lbvh_edge_data->sa_object_idx, device_lbvh_edge_data->sa_is_healthy, device_untangling_data->prp_batch_ee_request_src_ids, device_untangling_data->prp_batch_ee_request_ray_max_dist, device_untangling_data->prp_batch_ee_request_task_id, device_untangling_data->prp_batch_task_target_edge_range, device_untangling_data->prp_batch_task_target_edge_proj_min, device_untangling_data->prp_batch_task_dst_edge_prefix, device_untangling_data->prp_batch_task_dst_edge_ids, device_untangling_data->prp_batch_combo_dirs, kMaxRayDist, task_count_buffer, raw_slice_buf, raw_slice_attrs_buf, raw_slice_task_id_buf, per_task_capacity, static_cast<uint>(ee_request_src_ids.size())).dispatch(ee_request_src_ids.size());
					}
					if (!fv_request_dst_ids.empty())
					{
						PRP_PROFILE_SCOPE("evaluate_batched_combos.perform_raycasting.raycast_FV");
						stream << shaders.fn_task_raycast_fv_scatter_uint2(device_sim_data->sa_x, device_mesh_data->sa_faces, device_lbvh_face_data->sa_node_aabb_v2, device_lbvh_face_data->sa_children, device_lbvh_face_data->sa_object_idx, device_lbvh_face_data->sa_is_healthy, device_untangling_data->prp_batch_fv_request_dst_ids, device_untangling_data->prp_batch_fv_request_task_id, device_untangling_data->prp_batch_task_source_face_range, device_untangling_data->prp_batch_task_src_face_prefix, device_untangling_data->prp_batch_task_src_face_ids, device_untangling_data->prp_batch_combo_dirs, kMaxRayDist, task_count_buffer, raw_slice_buf, raw_slice_attrs_buf, raw_slice_task_id_buf, per_task_capacity, static_cast<uint>(fv_request_dst_ids.size())).dispatch(fv_request_dst_ids.size());
					}

					// Chain the coarse compact prefix-sum and the exact-candidate count pass on device; read everything back in a single synchronize.
					stream << shaders.fn_reset_uint(task_packed_offsets_buffer).dispatch(num_active_tasks_256);
					{
						auto total_coarse_count_buffer = device_untangling_data->prp_total_counts.view(2u * num_active_tasks_256 + 6u, 1u);
						stream << shaders.fn_reset_uint(total_coarse_count_buffer).dispatch(1u)
							   << shaders.fn_exclusive_prefix_sum(task_count_buffer, task_packed_offsets_buffer, total_coarse_count_buffer, 0u).dispatch(num_active_tasks_256)
							   << total_coarse_count_buffer.copy_to(&total_coarse_hits);
					}

					auto exact_count_buffer = device_untangling_data->prp_total_counts.view(0u, num_active_tasks_256);
					auto exact_offset_buffer = device_untangling_data->prp_total_counts.view(num_active_tasks_256, num_active_tasks_256);
					auto exact_type_counts_buffer = device_untangling_data->prp_total_counts.view(2u * num_active_tasks_256, 4u);
					auto total_exact_count_buffer = device_untangling_data->prp_total_counts.view(2u * num_active_tasks_256 + 4u, 1u);
					auto exact_overflow_buffer = device_untangling_data->prp_total_counts.view(2u * num_active_tasks_256 + 5u, 1u);
					{
						PRP_PROFILE_SCOPE("evaluate_batched_combos.perform_raycasting.indexed_exact_count");
						stream << shaders.fn_reset_uint(exact_count_buffer).dispatch(num_active_tasks_256)
							   << shaders.fn_reset_uint(exact_type_counts_buffer).dispatch(4u)
							   << shaders.fn_reset_uint(exact_overflow_buffer).dispatch(1u)
							   << shaders.fn_prp_filter_exact_candidate_indexed(raw_slice_buf, raw_slice_attrs_buf, task_count_buffer, task_packed_offsets_buffer, device_untangling_data->prp_batch_task_dst_face_prefix, device_untangling_data->prp_batch_task_dst_face_ids, device_untangling_data->prp_batch_task_dst_edge_prefix, device_untangling_data->prp_batch_task_dst_edge_ids, device_untangling_data->prp_batch_task_src_face_prefix, device_untangling_data->prp_batch_task_src_face_ids, exact_count_buffer, exact_type_counts_buffer, exact_overflow_buffer, raw_packed_buf, raw_packed_attrs_buf, raw_packed_task_id_buf, per_task_capacity, num_contours_active, total_capacity, 0u, 0u).dispatch(total_capacity)
							   << shaders.fn_reset_uint(total_exact_count_buffer).dispatch(1u)
							   << shaders.fn_exclusive_prefix_sum(exact_count_buffer, exact_offset_buffer, total_exact_count_buffer, 0u).dispatch(num_active_tasks_256);
						stream << task_count_host_view.copy_to(contour_coarse_hit_counts.data())
							   << exact_count_buffer.subview(0u, num_contours_active).copy_to(contour_raw_hit_counts.data())
							   << exact_offset_buffer.subview(0u, num_contours_active).copy_to(contour_offsets.data())
							   << exact_type_counts_buffer.copy_to(exact_type_counts.data())
							   << total_exact_count_buffer.copy_to(&total_raw_hits)
							   << luisa::compute::synchronize();
					}

					bool capacity_overflow = false;
					for (uint task_idx = 0u; task_idx < num_contours_active; ++task_idx)
						capacity_overflow = capacity_overflow || contour_coarse_hit_counts[task_idx] > per_task_capacity;
					if (capacity_overflow)
					{
						get_scene_params().prp_debug_info.uint_stats["prp_ray_retry_count"]++;
						const uint previous_capacity = per_task_capacity;
						if (previous_capacity >= kMaxRaycastCapacity)
						{
							for (uint task_idx = 0u; task_idx < num_contours_active; ++task_idx)
							{
								if (contour_coarse_hit_counts[task_idx] <= previous_capacity)
									continue;
								const auto& runtime = runtimes[contour_tasks[task_idx].contour_idx];
								LUISA_WARNING("GPU PRP ray capacity: contour {}, combo {}, k {}/{}, diameter {}/{}, hits {}, stored candidate-valid hits {}, candidate V {}/{}, E {}/{}, F {}/{}.",
									runtime.contour_idx, combo_idx, runtime.curr_k_src, runtime.curr_k_dst,
									runtime.mesh_diameter_src, runtime.mesh_diameter_dst,
									contour_coarse_hit_counts[task_idx], contour_raw_hit_counts[task_idx],
									runtime.partial_candidate_verts[0].size(), runtime.partial_candidate_verts[1].size(),
									runtime.partial_candidate_edges[0].size(), runtime.partial_candidate_edges[1].size(),
									runtime.partial_candidate_faces[0].size(), runtime.partial_candidate_faces[1].size());
							}
							LUISA_ERROR("GPU PRP raycast capacity overflow at max per-task capacity {} for {} active tasks.", previous_capacity, num_contours_active);
						}
						per_task_capacity = static_cast<uint>(std::min<uint64_t>(static_cast<uint64_t>(kMaxRaycastCapacity), std::max<uint64_t>(static_cast<uint64_t>(previous_capacity) * 2ull, static_cast<uint64_t>(previous_capacity) + 1024ull)));
						if (per_task_capacity <= previous_capacity)
							LUISA_ERROR("GPU PRP raycast capacity retry stalled at {} for {} active tasks.", previous_capacity, num_contours_active);
						continue;
					}
					if (exact_type_counts[3] != total_raw_hits)
						LUISA_ERROR("GPU indexed exact filter total mismatch: type total {} vs prefix total {}.", exact_type_counts[3], total_raw_hits);
					if (total_raw_hits != total_coarse_hits)
						LUISA_ERROR("GPU PRP stored hits outside the exact candidate domain: {} stored, {} valid.", total_coarse_hits, total_raw_hits);
					auto& ray_stats = get_scene_params().prp_debug_info.uint_stats;
					for (const uint hit_count : contour_coarse_hit_counts)
						ray_stats["prp_ray_max_task_hits"] = std::max(ray_stats["prp_ray_max_task_hits"], hit_count);
					ray_stats["prp_ray_max_task_capacity"] = std::max(ray_stats["prp_ray_max_task_capacity"], per_task_capacity);
					raycast_ready = true;
					break;
				}
				if (!raycast_ready)
					LUISA_ERROR("GPU PRP raycast capacity did not converge after {} retries for {} active tasks.", kMaxRaycastRetryAttempts, num_contours_active);

				if (total_raw_hits == 0u)
				{
					// Empty directions still carry their evaluation range into the scheduler.
					for (const auto& task : contour_tasks)
					{
						const auto& runtime = runtimes[task.contour_idx];
						auto&		info = contour_results[task.contour_idx];
						info.contour_idx = task.contour_idx;
						info.combo_idx = combo_idx;
						info.separation_direction = runtime.contour_dirs[combo_idx];
						info.evaluated_kring_src = use_partial ? runtime.curr_k_src : std::numeric_limits<uint>::max();
						info.evaluated_kring_dst = use_partial ? runtime.curr_k_dst : std::numeric_limits<uint>::max();
						info.full_mesh_evaluated = !use_partial
							|| (runtime.curr_k_src > runtime.mesh_diameter_src && runtime.curr_k_dst > runtime.mesh_diameter_dst);
					}
					return contour_results;
				}

				dynamic_resize(raw_packed_buf, total_raw_hits, "prp_runtime_uint2_indices");
				dynamic_resize(raw_packed_attrs_buf, total_raw_hits, "prp_runtime_hit_attrs");
				dynamic_resize(raw_packed_task_id_buf, total_raw_hits, "prp_batch_compact_task_id");
				{
					PRP_PROFILE_SCOPE("evaluate_batched_combos.perform_raycasting.indexed_exact_scatter");
					auto exact_count_buffer = device_untangling_data->prp_total_counts.view(0u, num_active_tasks_256);
					auto exact_offset_buffer = device_untangling_data->prp_total_counts.view(num_active_tasks_256, num_active_tasks_256);
					auto exact_type_counts_buffer = device_untangling_data->prp_total_counts.view(2u * num_active_tasks_256, 4u);
					auto exact_overflow_buffer = device_untangling_data->prp_total_counts.view(2u * num_active_tasks_256 + 5u, 1u);
					stream << shaders.fn_copy_uint_buffer(exact_offset_buffer, exact_count_buffer, num_contours_active).dispatch(num_active_tasks_256)
						   << shaders.fn_reset_uint(exact_overflow_buffer).dispatch(1u)
						   << shaders.fn_prp_filter_exact_candidate_indexed(raw_slice_buf, raw_slice_attrs_buf, task_count_buffer, task_packed_offsets_buffer, device_untangling_data->prp_batch_task_dst_face_prefix, device_untangling_data->prp_batch_task_dst_face_ids, device_untangling_data->prp_batch_task_dst_edge_prefix, device_untangling_data->prp_batch_task_dst_edge_ids, device_untangling_data->prp_batch_task_src_face_prefix, device_untangling_data->prp_batch_task_src_face_ids, exact_count_buffer, exact_type_counts_buffer, exact_overflow_buffer, raw_packed_buf, raw_packed_attrs_buf, raw_packed_task_id_buf, per_task_capacity, num_contours_active, total_coarse_hits, total_raw_hits, 1u).dispatch(total_coarse_hits)
						   << exact_overflow_buffer.copy_to(&exact_scatter_overflow);
				}

				{
					std::vector<uint64_t> signature;
					signature.push_back(kPrpLocalityMaxHop);
					for (const uint runtime_idx : runtime_indices)
						signature.push_back(runtime_idx);
					if (!static_level_upload_cache.valid || signature != static_level_upload_cache.signature)
					{
						std::vector<uint>  task_ef_prefix(1u, 0u);
						std::vector<uint3> task_ef_edges;
						std::vector<uint3> task_ef_faces;
						std::vector<uint>  task_boundary_hop_dense;
						std::vector<float> task_boundary_geom_dense;
						std::vector<uint>  task_locality_max_hop(num_contours_active, kPrpLocalityMaxHop);
						const uint64_t	   dense_size = static_cast<uint64_t>(num_contours_active) * 2ull * num_verts;
						if (dense_size >= std::numeric_limits<uint>::max())
							LUISA_ERROR("GPU PRP boundary table exceeds uint range: tasks={}, verts={}", num_contours_active, num_verts);
						task_boundary_hop_dense.reserve(dense_size);
						task_boundary_geom_dense.reserve(dense_size);
						for (const auto& task : contour_tasks)
						{
							const auto& runtime = runtimes[task.contour_idx];
							for (const uint pair_idx : runtime.contour)
							{
								// Loop-vertex EF records intentionally collapse both primitives.
								const auto& pair = ef_list[pair_idx];
								const uint2 edge = pair.get_edge();
								task_ef_edges.push_back(luisa::make_uint3(edge.x, edge.y, host_untangling_data->ef_pair_mesh_index[pair_idx]));
								task_ef_faces.push_back(pair.get_face());
							}
							if (task_ef_edges.size() >= std::numeric_limits<uint>::max())
								LUISA_ERROR("GPU PRP task EF records exceed uint range.");
							task_ef_prefix.push_back(static_cast<uint>(task_ef_edges.size()));
							for (uint side = 0u; side < 2u; ++side)
							{
								if (runtime.vert_to_boundary_hop_dist[side].size() != num_verts
									|| runtime.vert_to_boundary_geom_dist[side].size() != num_verts)
									LUISA_ERROR("GPU PRP requires dense boundary tables for contour {} side {}.", runtime.contour_idx, side);
								task_boundary_hop_dense.insert(task_boundary_hop_dense.end(), runtime.vert_to_boundary_hop_dist[side].begin(), runtime.vert_to_boundary_hop_dist[side].end());
								task_boundary_geom_dense.insert(task_boundary_geom_dense.end(), runtime.vert_to_boundary_geom_dist[side].begin(), runtime.vert_to_boundary_geom_dist[side].end());
							}
						}
						upload_buffer(device_untangling_data->prp_batch_task_ef_prefix, task_ef_prefix, "prp_batch_task_ef_prefix");
						upload_buffer(device_untangling_data->prp_batch_task_ef_edges, task_ef_edges, "prp_batch_task_ef_edges");
						upload_buffer(device_untangling_data->prp_batch_task_ef_faces, task_ef_faces, "prp_batch_task_ef_faces");
						upload_buffer(device_untangling_data->prp_batch_task_boundary_hop_dense, task_boundary_hop_dense, "prp_batch_task_boundary_hop_dense");
						upload_buffer(device_untangling_data->prp_batch_task_boundary_geom_dense, task_boundary_geom_dense, "prp_batch_task_boundary_geom_dense");
						upload_buffer(device_untangling_data->prp_batch_task_locality_max_hop, task_locality_max_hop, "prp_batch_task_locality_max_hop");
						static_level_upload_cache.valid = true;
						static_level_upload_cache.signature = std::move(signature);
					}
				}

				dynamic_resize(device_untangling_data->prp_runtime_flags, total_raw_hits, "prp_runtime_flags");
				stream << shaders.fn_reset_uint_with_value(device_untangling_data->prp_runtime_flags, 1u).dispatch(total_raw_hits);

				const uint N_elem = num_verts + num_edges + num_faces;

				const uint total_hits_256 = (std::max(total_raw_hits, 1u) + 255u) & ~255u;
				const uint max_per_task = *std::max_element(contour_raw_hit_counts.begin(), contour_raw_hit_counts.end());

				// Build primitive lookup and connected components for Cluster Culling.

				{
					PRP_PROFILE_SCOPE("evaluate_batched_combos.build_csr_for_cluster_culling");

					dynamic_resize(device_untangling_data->prp_csr_count_dst, total_hits_256, "prp_csr_touched_rows");
					dynamic_resize(device_untangling_data->prp_csr_prefix_dst, total_hits_256, "prp_csr_touched_counts");
					dynamic_resize(device_untangling_data->prp_csr_data_dst, total_hits_256, "prp_csr_touched_prefix");
					dynamic_resize(device_untangling_data->prp_runtime_cc_parent, total_raw_hits, "prp_runtime_cc_parent");
					dynamic_resize(device_untangling_data->prp_runtime_cc_changed, 1u, "prp_runtime_cc_changed");
					auto touched_count_buffer = device_untangling_data->prp_total_counts.view(num_contours_active + 1u, 1u);

					std::vector<uint2> task_hit_ranges(num_contours_active);
					for (uint task_idx = 0u; task_idx < num_contours_active; ++task_idx)
					{
						task_hit_ranges[task_idx] = luisa::make_uint2(contour_offsets[task_idx], contour_raw_hit_counts[task_idx]);
					}
					upload_buffer(device_untangling_data->prp_batch_task_hit_range, task_hit_ranges, "prp_batch_task_hit_range");

					stream << shaders.fn_prp_init_cc(device_untangling_data->prp_runtime_cc_parent, total_raw_hits).dispatch(total_raw_hits);

					PRP_PROFILE_SCOPE("evaluate_batched_combos.build_csr_for_cluster_culling.direct_union");
					const uint64_t keyed_key_space = static_cast<uint64_t>(num_contours_active) * static_cast<uint64_t>(N_elem);
					if (keyed_key_space >= static_cast<uint64_t>(std::numeric_limits<uint>::max()))
						LUISA_ERROR("GPU PRP keyed primitive CSR key space exceeds uint sentinel range: tasks={}, elems={}", num_contours_active, N_elem);
					uint64_t keyed_hash_capacity64 = std::max<uint64_t>(1024ull, static_cast<uint64_t>(std::max(total_raw_hits, 1u)) * 2ull);
					if (keyed_hash_capacity64 > static_cast<uint64_t>(std::numeric_limits<uint>::max()) - 255ull)
						LUISA_ERROR("GPU PRP keyed hash capacity exceeds uint range: raw_hits={}", total_raw_hits);
					keyed_hash_capacity64 = (keyed_hash_capacity64 + 255ull) & ~255ull;
					const uint keyed_hash_capacity = static_cast<uint>(keyed_hash_capacity64);
					direct_keyed_hash_capacity_captured = keyed_hash_capacity;

					dynamic_resize(device_untangling_data->prp_runtime_anchor_hash_keys, keyed_hash_capacity, "prp_keyed_csr_hash_keys");
					dynamic_resize(device_untangling_data->prp_csr_count_src, keyed_hash_capacity, "prp_keyed_csr_count_src");
					dynamic_resize(device_untangling_data->prp_csr_prefix_src, keyed_hash_capacity, "prp_keyed_csr_prefix_src");
					dynamic_resize(device_untangling_data->prp_csr_data_src, std::max(total_raw_hits, 1u), "prp_keyed_csr_data_src");
					dynamic_resize(device_untangling_data->prp_runtime_pair_adj_count, keyed_hash_capacity, "prp_keyed_csr_fill_counts");

					auto keyed_overflow_buffer = device_untangling_data->prp_total_counts.view(num_contours_active + 4u, 1u);
					auto count_total_keyed_csr_buffer = device_untangling_data->prp_total_counts.view(num_contours_active + 5u, 1u);
					auto keyed_fill_overflow_buffer = device_untangling_data->prp_total_counts.view(num_contours_active + 6u, 1u);
					{
						PRP_PROFILE_SCOPE("evaluate_batched_combos.build_csr_for_cluster_culling.keyed_primitive_csr");
						stream << shaders.fn_reset_uint_with_value(device_untangling_data->prp_runtime_anchor_hash_keys, 0xffffffffu).dispatch(keyed_hash_capacity)
							   << shaders.fn_reset_uint(device_untangling_data->prp_csr_count_src).dispatch(keyed_hash_capacity)
							   << shaders.fn_reset_uint(touched_count_buffer).dispatch(1u)
							   << shaders.fn_reset_uint(keyed_overflow_buffer).dispatch(1u)
							   << shaders.fn_reset_uint(keyed_fill_overflow_buffer).dispatch(1u)
							   << shaders.fn_batch_keyed_count_hits_per_elem(
											 raw_packed_buf,
											 raw_packed_task_id_buf,

											 device_untangling_data->prp_runtime_anchor_hash_keys,
											 device_untangling_data->prp_csr_count_src,
											 device_untangling_data->prp_csr_count_dst,
											 touched_count_buffer,
											 keyed_overflow_buffer,
											 num_verts,
											 num_edges,
											 N_elem,
											 keyed_hash_capacity,
											 total_raw_hits,
											 0u)
									  .dispatch(total_hits_256)
							   << shaders.fn_batch_gather_touched_counts(
											 device_untangling_data->prp_csr_count_src,
											 device_untangling_data->prp_csr_count_dst,
											 device_untangling_data->prp_csr_prefix_dst,
											 touched_count_buffer,
											 total_hits_256)
									  .dispatch(total_hits_256)
							   << shaders.fn_reset_uint(count_total_keyed_csr_buffer).dispatch(1u)
							   << shaders.fn_exclusive_prefix_sum(
											 device_untangling_data->prp_csr_prefix_dst,
											 device_untangling_data->prp_csr_data_dst,
											 count_total_keyed_csr_buffer,
											 0)
									  .dispatch(total_hits_256)
							   << shaders.fn_batch_scatter_touched_prefix(
											 device_untangling_data->prp_csr_prefix_src,
											 device_untangling_data->prp_csr_count_dst,
											 device_untangling_data->prp_csr_data_dst,
											 touched_count_buffer,
											 total_hits_256)
									  .dispatch(total_hits_256);

						stream << shaders.fn_batch_clear_touched_rows(
											 device_untangling_data->prp_runtime_pair_adj_count,
											 device_untangling_data->prp_csr_count_dst,
											 touched_count_buffer,
											 total_hits_256)
									  .dispatch(total_hits_256)
							   << shaders.fn_batch_keyed_fill_csr(
											 raw_packed_buf,
											 raw_packed_task_id_buf,

											 device_untangling_data->prp_runtime_anchor_hash_keys,
											 device_untangling_data->prp_csr_prefix_src,
											 device_untangling_data->prp_runtime_pair_adj_count,
											 device_untangling_data->prp_csr_data_src,
											 keyed_fill_overflow_buffer,
											 num_verts,
											 num_edges,
											 N_elem,
											 keyed_hash_capacity,
											 total_raw_hits,
											 0u)
									  .dispatch(total_hits_256);
					}
					// Defer keyed-CSR verification copies; undercounted probes can only reduce in-bounds writes.

					const uint max_cc_iters = std::max(20u, static_cast<uint>(std::ceil(std::log2(std::max(max_per_task, 1u)))) + 2u);
					// A sticky device convergence flag replaces per-iteration union-find readbacks.
					stream << shaders.fn_reset_uint(device_untangling_data->prp_runtime_cc_changed).dispatch(1u)
						   << shaders.fn_reset_uint(device_untangling_data->prp_runtime_cc_converged).dispatch(1u);
					for (uint iter = 0u; iter < max_cc_iters; ++iter)
					{
						stream << shaders.fn_batch_hook_cc_direct_keyed(
											 device_untangling_data->prp_csr_count_src,
											 device_untangling_data->prp_csr_prefix_src,
											 device_untangling_data->prp_csr_data_src,
											 device_untangling_data->prp_runtime_anchor_hash_keys,
											 raw_packed_buf,
											 raw_packed_attrs_buf,
											 raw_packed_task_id_buf,
											 device_untangling_data->prp_runtime_cc_parent,
											 device_untangling_data->prp_runtime_cc_changed,
											 device_untangling_data->prp_runtime_cc_converged,
											 num_verts,
											 num_edges,
											 N_elem,
											 keyed_hash_capacity,
											 total_raw_hits)
									  .dispatch(total_hits_256);
						if (get_scene_params().PRP_solid_interior_adjacency)
						{
							stream << shaders.fn_batch_hook_solid_interior_adjacency(
												 raw_packed_buf,
												 raw_packed_attrs_buf,
												 raw_packed_task_id_buf,
												 device_untangling_data->prp_batch_task_hit_range,
												 device_untangling_data->prp_batch_combo_dirs,
												 device_untangling_data->prp_runtime_cc_parent,
												 device_untangling_data->prp_runtime_cc_changed,
												 device_untangling_data->prp_runtime_cc_converged,
												 device_mesh_data->sa_faces,
												 device_mesh_data->sa_edges,
												 device_mesh_data->sa_face_mesh_id,
												 device_mesh_data->sa_vert_mesh_id,
												 device_mesh_data->sa_vert_mesh_type,
												 device_untangling_data->prp_face_normal,
												 device_untangling_data->prp_edge_normal,
												 total_raw_hits,
												 get_scene_params().PRP_solid_interior_adjacency_all_meshes ? 1u : 0u,
												 get_scene_params().PRP_solid_interior_adjacency_normal_eps,
												 get_scene_params().PRP_solid_interior_adjacency_dist_eps)
										  .dispatch(total_hits_256);
						}
						stream << shaders.fn_prp_compress_cc(device_untangling_data->prp_runtime_cc_parent, device_untangling_data->prp_runtime_cc_converged, total_raw_hits).dispatch(total_raw_hits)
							   << shaders.fn_prp_check_cc_converged(device_untangling_data->prp_runtime_cc_changed, device_untangling_data->prp_runtime_cc_converged).dispatch(1u);
					}
				}

				// Enqueue device combo summaries beside cluster selection and share its synchronize.
				std::vector<PRPDeviceComboSummary> host_device_combo_summaries;
				const bool						   enable_device_combo_summary = get_scene_params().prp_debug;
				auto							   dispatch_device_summary_eval = [&]()
				{
					stream << shaders.fn_prp_reset_device_combo_eval(
										 device_untangling_data->prp_batch_combo_summary,
										 num_contours_active)
								  .dispatch(num_contours_active)
						   << shaders.fn_prp_accumulate_combo_summary(
										 raw_packed_buf,
										 raw_packed_attrs_buf,
										 raw_packed_task_id_buf,
										 device_untangling_data->prp_runtime_flags,
										 device_untangling_data->prp_batch_combo_dirs,
										 device_untangling_data->prp_batch_task_boundary_hop_dense,
										 device_mesh_data->sa_faces,
										 device_mesh_data->sa_edges,
										 device_sim_data->sa_x,
										 device_mesh_data->sa_rest_vert_area,
										 device_mesh_data->sa_rest_edge_area,
										 device_mesh_data->sa_rest_face_area,
										 device_mesh_data->sa_vert_mass,
										 device_untangling_data->prp_vert_normal,
										 device_untangling_data->prp_edge_normal,
										 device_untangling_data->prp_face_normal,
										 device_untangling_data->prp_batch_combo_summary,
										 num_verts,
										 total_raw_hits)
								  .dispatch(total_raw_hits);
				};
				if (enable_device_combo_summary)
				{
					dynamic_resize(device_untangling_data->prp_batch_combo_summary, num_contours_active, "prp_batch_combo_summary");
					host_device_combo_summaries.resize(num_contours_active);
				}

				// Cluster Culling
				{
					PRP_PROFILE_SCOPE("evaluate_batched_combos.cluster_culling");
					const uint max_cc_iters = std::max(20u, static_cast<uint>(std::ceil(std::log2(std::max(max_per_task, 1u)))) + 2u);
					uint	   changed = 1u;

					{
						PRP_PROFILE_SCOPE("evaluate_batched_combos.cluster_culling.device_selection");
						const uint	   coverage_stride = (2u * num_verts + 31u) / 32u;
						const uint	   max_dense_roots = std::max(2048u, num_contours_active * 64u);
						const uint64_t total_dense_mask_elements = static_cast<uint64_t>(max_dense_roots) * static_cast<uint64_t>(coverage_stride);
						if (total_dense_mask_elements > std::numeric_limits<uint>::max())
							LUISA_ERROR("GPU PRP coverage dense mask exceeds uint range: roots={}, stride={}", max_dense_roots, coverage_stride);
						dynamic_resize(device_untangling_data->prp_runtime_root_task_idx, total_raw_hits, "prp_runtime_root_task_idx");
						dynamic_resize(device_untangling_data->prp_runtime_root_hit_count, total_raw_hits, "prp_runtime_root_hit_count");
						dynamic_resize(device_untangling_data->prp_runtime_root_kept_hit_count, total_raw_hits, "prp_runtime_root_kept_hit_count");
						dynamic_resize(device_untangling_data->prp_runtime_root_boundary_attached, total_raw_hits, "prp_runtime_root_boundary_attached");
						dynamic_resize(device_untangling_data->prp_runtime_root_coverage_count, total_raw_hits, "prp_runtime_root_coverage_count");
						dynamic_resize(device_untangling_data->prp_runtime_root_canonical_key, total_raw_hits, "prp_runtime_root_canonical_key");
						dynamic_resize(device_untangling_data->prp_runtime_root_canonical_key_high, total_raw_hits, "prp_runtime_root_canonical_key_high");
						dynamic_resize(device_untangling_data->prp_runtime_root_max_kept_tgt_geom, total_raw_hits, "prp_runtime_root_max_kept_tgt_geom");
						dynamic_resize(device_untangling_data->prp_runtime_root_dense_id, total_raw_hits, "prp_runtime_root_dense_id");
						dynamic_resize(device_untangling_data->prp_runtime_valid_root_count, 1u, "prp_runtime_valid_root_count");
						dynamic_resize(device_untangling_data->prp_runtime_coverage_dense_mask, static_cast<uint>(total_dense_mask_elements), "prp_runtime_coverage_dense_mask");
						dynamic_resize(device_untangling_data->prp_batch_selected_roots, num_contours_active, "prp_batch_selected_roots");
						dynamic_resize(device_untangling_data->prp_batch_selected_boundary_counts, num_contours_active, "prp_batch_selected_boundary_counts");
						dynamic_resize(device_untangling_data->prp_batch_selected_hit_counts, num_contours_active, "prp_batch_selected_hit_counts");
						dynamic_resize(device_untangling_data->prp_batch_selected_keys, num_contours_active, "prp_batch_selected_keys");
						dynamic_resize(device_untangling_data->prp_batch_selected_key_highs, num_contours_active, "prp_batch_selected_key_highs");
						dynamic_resize(device_untangling_data->prp_batch_task_component_count, num_contours_active, "prp_batch_task_component_count");
						dynamic_resize(device_untangling_data->prp_batch_task_largest_component_hits, num_contours_active, "prp_batch_task_largest_component_hits");
						dynamic_resize(device_untangling_data->prp_batch_task_max_kept_tgt_geom, num_contours_active, "prp_batch_task_max_kept_tgt_geom");

						stream << shaders.fn_prp_compress_cc(device_untangling_data->prp_runtime_cc_parent, device_untangling_data->prp_runtime_cc_converged, total_raw_hits).dispatch(total_raw_hits)
							   << shaders.fn_prp_reset_cluster_selection(
											 device_untangling_data->prp_runtime_root_task_idx,
											 device_untangling_data->prp_runtime_root_hit_count,
											 device_untangling_data->prp_runtime_root_kept_hit_count,
											 device_untangling_data->prp_runtime_root_boundary_attached,
											 device_untangling_data->prp_runtime_root_coverage_count,
											 device_untangling_data->prp_runtime_root_canonical_key,
											 device_untangling_data->prp_runtime_root_canonical_key_high,
											 device_untangling_data->prp_runtime_root_max_kept_tgt_geom,
											 device_untangling_data->prp_runtime_root_dense_id,
											 total_raw_hits)
									  .dispatch(total_raw_hits)
							   << shaders.fn_prp_reset_cluster_task_selection(
											 device_untangling_data->prp_batch_selected_roots,
											 device_untangling_data->prp_batch_selected_boundary_counts,
											 device_untangling_data->prp_batch_selected_hit_counts,
											 device_untangling_data->prp_batch_selected_keys,
											 device_untangling_data->prp_batch_selected_key_highs,
											 device_untangling_data->prp_batch_task_component_count,
											 device_untangling_data->prp_batch_task_largest_component_hits,
											 device_untangling_data->prp_batch_task_max_kept_tgt_geom,
											 num_contours_active,
											 total_raw_hits)
									  .dispatch(num_active_tasks_256)
							   << shaders.fn_reset_uint(device_untangling_data->prp_runtime_cc_changed).dispatch(1u)
							   << shaders.fn_reset_uint(device_untangling_data->prp_runtime_valid_root_count).dispatch(1u)
							   << shaders.fn_reset_uint(device_untangling_data->prp_runtime_coverage_dense_mask).dispatch(static_cast<uint>(total_dense_mask_elements))
							   << shaders.fn_prp_accumulate_cluster_root_stats(
											 raw_packed_buf,
											 raw_packed_task_id_buf,
											 device_untangling_data->prp_runtime_flags,
											 device_untangling_data->prp_runtime_cc_parent,
											 device_mesh_data->sa_faces,
											 device_mesh_data->sa_edges,
											 device_untangling_data->prp_vert_adj_contours_csr,
											 device_untangling_data->prp_batch_task_contour_idx,
											 device_untangling_data->prp_batch_task_boundary_hop_dense,
											 device_untangling_data->prp_batch_task_boundary_geom_dense,
											 device_untangling_data->prp_batch_task_locality_max_hop,
											 device_untangling_data->prp_runtime_root_task_idx,
											 device_untangling_data->prp_runtime_root_hit_count,
											 device_untangling_data->prp_runtime_root_kept_hit_count,
											 device_untangling_data->prp_runtime_root_boundary_attached,
											 device_untangling_data->prp_runtime_root_canonical_key_high,
											 device_untangling_data->prp_runtime_root_max_kept_tgt_geom,
											 device_untangling_data->prp_runtime_cc_changed,
											 total_raw_hits,
											 num_verts,
											 1u)
									  .dispatch(total_raw_hits)
							   << shaders.fn_prp_assign_dense_cluster_roots(
											 device_untangling_data->prp_runtime_root_hit_count,
											 device_untangling_data->prp_runtime_root_boundary_attached,
											 device_untangling_data->prp_runtime_root_dense_id,
											 device_untangling_data->prp_runtime_valid_root_count,
											 device_untangling_data->prp_runtime_cc_changed,
											 total_raw_hits,
											 max_dense_roots)
									  .dispatch(total_raw_hits)
							   << shaders.fn_prp_fill_dense_cluster_coverage_mask(
											 raw_packed_buf,
											 raw_packed_task_id_buf,
											 device_untangling_data->prp_runtime_flags,
											 device_untangling_data->prp_runtime_cc_parent,
											 device_mesh_data->sa_faces,
											 device_mesh_data->sa_edges,
											 device_untangling_data->prp_batch_task_boundary_hop_dense,
											 device_untangling_data->prp_batch_task_locality_max_hop,
											 device_untangling_data->prp_runtime_root_dense_id,
											 device_untangling_data->prp_runtime_coverage_dense_mask,
											 total_raw_hits,
											 num_verts,
											 coverage_stride,
											 max_dense_roots,
											 1u)
									  .dispatch(total_raw_hits)
							   << shaders.fn_prp_count_cluster_ef_coverage(
											 device_untangling_data->prp_runtime_root_task_idx,
											 device_untangling_data->prp_runtime_root_hit_count,
											 device_untangling_data->prp_runtime_root_boundary_attached,
											 device_untangling_data->prp_runtime_root_dense_id,
											 device_untangling_data->prp_batch_task_ef_prefix,
											 device_untangling_data->prp_batch_task_ef_edges,
											 device_untangling_data->prp_batch_task_ef_faces,
											 device_untangling_data->prp_runtime_coverage_dense_mask,
											 device_untangling_data->prp_runtime_root_coverage_count,
											 num_verts,
											 coverage_stride,
											 max_dense_roots,
											 total_raw_hits)
									  .dispatch(total_raw_hits)
							   << shaders.fn_prp_accumulate_cluster_root_key_low(
											 raw_packed_buf,
											 device_untangling_data->prp_runtime_flags,
											 device_untangling_data->prp_runtime_cc_parent,
											 device_untangling_data->prp_runtime_root_canonical_key,
											 device_untangling_data->prp_runtime_root_canonical_key_high,
											 device_untangling_data->prp_runtime_cc_changed,
											 total_raw_hits)
									  .dispatch(total_raw_hits)
							   // Five parallel narrowing passes replace the serial per-task root scan.
							   << shaders.fn_reset_uint(device_untangling_data->prp_total_counts.view(2u * num_active_tasks_256 + 7u + 0u * num_contours_active, num_contours_active)).dispatch(num_contours_active)
							   << shaders.fn_reset_uint(device_untangling_data->prp_total_counts.view(2u * num_active_tasks_256 + 7u + 1u * num_contours_active, num_contours_active)).dispatch(num_contours_active)
							   << shaders.fn_reset_uint_with_value(device_untangling_data->prp_total_counts.view(2u * num_active_tasks_256 + 7u + 2u * num_contours_active, num_contours_active), 0xffffffffu).dispatch(num_contours_active)
							   << shaders.fn_reset_uint_with_value(device_untangling_data->prp_total_counts.view(2u * num_active_tasks_256 + 7u + 3u * num_contours_active, num_contours_active), 0xffffffffu).dispatch(num_contours_active)
							   << shaders.fn_reset_uint_with_value(device_untangling_data->prp_total_counts.view(2u * num_active_tasks_256 + 7u + 4u * num_contours_active, num_contours_active), total_raw_hits).dispatch(num_contours_active)
							   << shaders.fn_prp_select_cluster_roots_pass(
											 device_untangling_data->prp_runtime_root_task_idx,
											 device_untangling_data->prp_runtime_root_hit_count,
											 device_untangling_data->prp_runtime_root_boundary_attached,
											 device_untangling_data->prp_runtime_root_coverage_count,
											 device_untangling_data->prp_runtime_root_kept_hit_count,
											 device_untangling_data->prp_runtime_root_canonical_key,
											 device_untangling_data->prp_runtime_root_canonical_key_high,
											 device_untangling_data->prp_total_counts.view(2u * num_active_tasks_256 + 7u + 0u * num_contours_active, num_contours_active),
											 device_untangling_data->prp_total_counts.view(2u * num_active_tasks_256 + 7u + 1u * num_contours_active, num_contours_active),
											 device_untangling_data->prp_total_counts.view(2u * num_active_tasks_256 + 7u + 2u * num_contours_active, num_contours_active),
											 device_untangling_data->prp_total_counts.view(2u * num_active_tasks_256 + 7u + 3u * num_contours_active, num_contours_active),
											 device_untangling_data->prp_total_counts.view(2u * num_active_tasks_256 + 7u + 4u * num_contours_active, num_contours_active),
											 device_untangling_data->prp_batch_task_component_count,
											 device_untangling_data->prp_batch_task_largest_component_hits,
											 0u,
											 total_raw_hits)
									  .dispatch(total_raw_hits)
							   << shaders.fn_prp_select_cluster_roots_pass(
											 device_untangling_data->prp_runtime_root_task_idx,
											 device_untangling_data->prp_runtime_root_hit_count,
											 device_untangling_data->prp_runtime_root_boundary_attached,
											 device_untangling_data->prp_runtime_root_coverage_count,
											 device_untangling_data->prp_runtime_root_kept_hit_count,
											 device_untangling_data->prp_runtime_root_canonical_key,
											 device_untangling_data->prp_runtime_root_canonical_key_high,
											 device_untangling_data->prp_total_counts.view(2u * num_active_tasks_256 + 7u + 0u * num_contours_active, num_contours_active),
											 device_untangling_data->prp_total_counts.view(2u * num_active_tasks_256 + 7u + 1u * num_contours_active, num_contours_active),
											 device_untangling_data->prp_total_counts.view(2u * num_active_tasks_256 + 7u + 2u * num_contours_active, num_contours_active),
											 device_untangling_data->prp_total_counts.view(2u * num_active_tasks_256 + 7u + 3u * num_contours_active, num_contours_active),
											 device_untangling_data->prp_total_counts.view(2u * num_active_tasks_256 + 7u + 4u * num_contours_active, num_contours_active),
											 device_untangling_data->prp_batch_task_component_count,
											 device_untangling_data->prp_batch_task_largest_component_hits,
											 1u,
											 total_raw_hits)
									  .dispatch(total_raw_hits)
							   << shaders.fn_prp_select_cluster_roots_pass(
											 device_untangling_data->prp_runtime_root_task_idx,
											 device_untangling_data->prp_runtime_root_hit_count,
											 device_untangling_data->prp_runtime_root_boundary_attached,
											 device_untangling_data->prp_runtime_root_coverage_count,
											 device_untangling_data->prp_runtime_root_kept_hit_count,
											 device_untangling_data->prp_runtime_root_canonical_key,
											 device_untangling_data->prp_runtime_root_canonical_key_high,
											 device_untangling_data->prp_total_counts.view(2u * num_active_tasks_256 + 7u + 0u * num_contours_active, num_contours_active),
											 device_untangling_data->prp_total_counts.view(2u * num_active_tasks_256 + 7u + 1u * num_contours_active, num_contours_active),
											 device_untangling_data->prp_total_counts.view(2u * num_active_tasks_256 + 7u + 2u * num_contours_active, num_contours_active),
											 device_untangling_data->prp_total_counts.view(2u * num_active_tasks_256 + 7u + 3u * num_contours_active, num_contours_active),
											 device_untangling_data->prp_total_counts.view(2u * num_active_tasks_256 + 7u + 4u * num_contours_active, num_contours_active),
											 device_untangling_data->prp_batch_task_component_count,
											 device_untangling_data->prp_batch_task_largest_component_hits,
											 2u,
											 total_raw_hits)
									  .dispatch(total_raw_hits)
							   << shaders.fn_prp_select_cluster_roots_pass(
											 device_untangling_data->prp_runtime_root_task_idx,
											 device_untangling_data->prp_runtime_root_hit_count,
											 device_untangling_data->prp_runtime_root_boundary_attached,
											 device_untangling_data->prp_runtime_root_coverage_count,
											 device_untangling_data->prp_runtime_root_kept_hit_count,
											 device_untangling_data->prp_runtime_root_canonical_key,
											 device_untangling_data->prp_runtime_root_canonical_key_high,
											 device_untangling_data->prp_total_counts.view(2u * num_active_tasks_256 + 7u + 0u * num_contours_active, num_contours_active),
											 device_untangling_data->prp_total_counts.view(2u * num_active_tasks_256 + 7u + 1u * num_contours_active, num_contours_active),
											 device_untangling_data->prp_total_counts.view(2u * num_active_tasks_256 + 7u + 2u * num_contours_active, num_contours_active),
											 device_untangling_data->prp_total_counts.view(2u * num_active_tasks_256 + 7u + 3u * num_contours_active, num_contours_active),
											 device_untangling_data->prp_total_counts.view(2u * num_active_tasks_256 + 7u + 4u * num_contours_active, num_contours_active),
											 device_untangling_data->prp_batch_task_component_count,
											 device_untangling_data->prp_batch_task_largest_component_hits,
											 3u,
											 total_raw_hits)
									  .dispatch(total_raw_hits)
							   << shaders.fn_prp_select_cluster_roots_pass(
											 device_untangling_data->prp_runtime_root_task_idx,
											 device_untangling_data->prp_runtime_root_hit_count,
											 device_untangling_data->prp_runtime_root_boundary_attached,
											 device_untangling_data->prp_runtime_root_coverage_count,
											 device_untangling_data->prp_runtime_root_kept_hit_count,
											 device_untangling_data->prp_runtime_root_canonical_key,
											 device_untangling_data->prp_runtime_root_canonical_key_high,
											 device_untangling_data->prp_total_counts.view(2u * num_active_tasks_256 + 7u + 0u * num_contours_active, num_contours_active),
											 device_untangling_data->prp_total_counts.view(2u * num_active_tasks_256 + 7u + 1u * num_contours_active, num_contours_active),
											 device_untangling_data->prp_total_counts.view(2u * num_active_tasks_256 + 7u + 2u * num_contours_active, num_contours_active),
											 device_untangling_data->prp_total_counts.view(2u * num_active_tasks_256 + 7u + 3u * num_contours_active, num_contours_active),
											 device_untangling_data->prp_total_counts.view(2u * num_active_tasks_256 + 7u + 4u * num_contours_active, num_contours_active),
											 device_untangling_data->prp_batch_task_component_count,
											 device_untangling_data->prp_batch_task_largest_component_hits,
											 4u,
											 total_raw_hits)
									  .dispatch(total_raw_hits)
							   << shaders.fn_prp_select_cluster_roots_finalize(
											 device_untangling_data->prp_total_counts.view(2u * num_active_tasks_256 + 7u + 0u * num_contours_active, num_contours_active),
											 device_untangling_data->prp_total_counts.view(2u * num_active_tasks_256 + 7u + 1u * num_contours_active, num_contours_active),
											 device_untangling_data->prp_total_counts.view(2u * num_active_tasks_256 + 7u + 2u * num_contours_active, num_contours_active),
											 device_untangling_data->prp_total_counts.view(2u * num_active_tasks_256 + 7u + 3u * num_contours_active, num_contours_active),
											 device_untangling_data->prp_total_counts.view(2u * num_active_tasks_256 + 7u + 4u * num_contours_active, num_contours_active),
											 device_untangling_data->prp_runtime_root_max_kept_tgt_geom,
											 device_untangling_data->prp_batch_selected_roots,
											 device_untangling_data->prp_batch_selected_boundary_counts,
											 device_untangling_data->prp_batch_selected_hit_counts,
											 device_untangling_data->prp_batch_selected_keys,
											 device_untangling_data->prp_batch_selected_key_highs,
											 device_untangling_data->prp_batch_task_max_kept_tgt_geom,
											 num_contours_active,
											 total_raw_hits)
									  .dispatch(num_active_tasks_256)
							   << shaders.fn_prp_write_cluster_valid_flags(
											 raw_packed_buf,
											 raw_packed_task_id_buf,
											 device_untangling_data->prp_runtime_cc_parent,
											 device_mesh_data->sa_faces,
											 device_mesh_data->sa_edges,
											 device_untangling_data->prp_batch_task_boundary_hop_dense,
											 device_untangling_data->prp_batch_task_locality_max_hop,
											 device_untangling_data->prp_batch_selected_roots,
											 device_untangling_data->prp_runtime_flags,
											 total_raw_hits,
											 num_verts,
											 0u,
											 1u)
									  .dispatch(total_raw_hits);
						// Chain combo summaries and derivative passes onto the cluster-selection synchronize.
						uint64_t shadow_valid_hits = 0u;
						uint64_t shadow_vf_hits = 0u;
						uint64_t shadow_ee_hits = 0u;
						uint64_t shadow_fv_hits = 0u;
						float	 shadow_max_depth = 0.0f;
						if (enable_device_combo_summary)
						{
							PRP_PROFILE_SCOPE("evaluate_batched_combos.device_summary");
							dispatch_device_summary_eval();
							stream << device_untangling_data->prp_batch_combo_summary.view(0u, num_contours_active).copy_to(host_device_combo_summaries.data());
						}
						uint host_valid_root_count = 0u;
						stream << device_untangling_data->prp_total_counts.view(num_contours_active + 4u, 1u).copy_to(&direct_keyed_overflow)
							   << device_untangling_data->prp_total_counts.view(num_contours_active + 5u, 1u).copy_to(&direct_keyed_total_entries)
							   << device_untangling_data->prp_total_counts.view(num_contours_active + 6u, 1u).copy_to(&direct_keyed_fill_overflow)
							   << device_untangling_data->prp_runtime_valid_root_count.view(0u, 1u).copy_to(&host_valid_root_count)
							   << device_untangling_data->prp_runtime_cc_changed.view(0u, 1u).copy_to(&changed)
							   << device_untangling_data->prp_runtime_cc_converged.view(0u, 1u).copy_to(&cc_converged_flag)
							   << luisa::compute::synchronize();
						if (changed != 0u)
							LUISA_ERROR("GPU PRP device cluster selection failed: coverage dense mask or valid root count overflow or invalid component parent (hits={}, valid_roots={}, max_dense_roots={})", total_raw_hits, host_valid_root_count, max_dense_roots);
						if (cc_converged_flag == 0u)
							LUISA_ERROR("Cluster culling failed to converge after {} iterations.", max_cc_iters);
						if (exact_scatter_overflow != 0u)
							LUISA_ERROR("GPU indexed exact filter wrote past compact hit capacity {} for {} active contours.", total_raw_hits, num_contours_active);

						if (enable_device_combo_summary)
						{
							for (const auto& summary : host_device_combo_summaries)
							{
								shadow_valid_hits += summary.counts0.x;
								shadow_vf_hits += summary.counts0.y;
								shadow_ee_hits += summary.counts0.z;
								shadow_fv_hits += summary.counts0.w;
								shadow_max_depth = std::max(shadow_max_depth, summary.eval0.y);
							}
							if (get_scene_params().prp_debug)
							{
								prp_debug_set_uint("prp.gpu.shadow_eval.valid_hit_count", static_cast<uint>(shadow_valid_hits));
								prp_debug_set_uint("prp.gpu.shadow_eval.vf_hit_count", static_cast<uint>(shadow_vf_hits));
								prp_debug_set_uint("prp.gpu.shadow_eval.ee_hit_count", static_cast<uint>(shadow_ee_hits));
								prp_debug_set_uint("prp.gpu.shadow_eval.fv_hit_count", static_cast<uint>(shadow_fv_hits));
								prp_debug_set_float("prp.gpu.shadow_eval.max_penetration_depth", shadow_max_depth);
							}
						}
					}
				}
				// Direct keyed-CSR verification readbacks share the cluster-selection synchronize.
				{
					if (direct_keyed_overflow != 0u)
						LUISA_ERROR("GPU PRP keyed primitive CSR hash overflow: hits={}, hash_capacity={}", total_raw_hits, direct_keyed_hash_capacity_captured);
					if (direct_keyed_total_entries > total_raw_hits)
						LUISA_ERROR("GPU PRP keyed primitive CSR count exceeded raw hit count: entries={}, raw_hits={}", direct_keyed_total_entries, total_raw_hits);
					if (direct_keyed_fill_overflow != 0u)
						LUISA_ERROR("GPU PRP keyed primitive CSR fill lookup failed: hits={}, hash_capacity={}", total_raw_hits, direct_keyed_hash_capacity_captured);
				}

				// Host evaluation
				{
					PRP_PROFILE_SCOPE("evaluate_batched_combos.host_evaluation");
					uint  total_valid_hits = 0u;
					auto  valid_hit_count_buffer = device_untangling_data->prp_total_counts.view(num_contours_active + 3u, 1u);
					auto& compact_valid_hits_buf = raw_slice_buf;
					auto& compact_valid_attrs_buf = raw_slice_attrs_buf;
					auto& compact_valid_task_id_buf = raw_slice_task_id_buf;
					{
						PRP_PROFILE_SCOPE("evaluate_batched_combos.host_evaluation.compact_valid_hits");
						dynamic_resize(compact_valid_hits_buf, std::max(total_raw_hits, 1u), "prp_batch_uint2_indices");
						dynamic_resize(compact_valid_attrs_buf, std::max(total_raw_hits, 1u), "prp_batch_hit_attrs");
						dynamic_resize(compact_valid_task_id_buf, std::max(total_raw_hits, 1u), "prp_batch_hit_combo_id");
						// Read the compact count first so host vectors can be sized before payload copies.
						stream << shaders.fn_reset_uint(valid_hit_count_buffer).dispatch(1u)
							   << shaders.fn_prp_compact_valid_hits_with_task(
											 raw_packed_buf,
											 raw_packed_attrs_buf,
											 raw_packed_task_id_buf,
											 device_untangling_data->prp_runtime_flags,
											 compact_valid_hits_buf,
											 compact_valid_attrs_buf,
											 compact_valid_task_id_buf,
											 valid_hit_count_buffer,
											 total_raw_hits)
									  .dispatch(total_raw_hits)
							   << valid_hit_count_buffer.copy_to(&total_valid_hits)
							   << luisa::compute::synchronize();
					}

					std::vector<uint2>	host_valid_hits(total_valid_hits);
					std::vector<float4> host_valid_attrs(total_valid_hits);
					std::vector<uint>	host_task_ids(total_valid_hits);
					if (total_valid_hits > 0u)
					{
						PRP_PROFILE_SCOPE("evaluate_batched_combos.host_evaluation.readback_valid_buffers");
						stream << compact_valid_hits_buf.view(0u, total_valid_hits).copy_to(host_valid_hits.data())
							   << compact_valid_attrs_buf.view(0u, total_valid_hits).copy_to(host_valid_attrs.data())
							   << compact_valid_task_id_buf.view(0u, total_valid_hits).copy_to(host_task_ids.data())
							   << luisa::compute::synchronize();
					}

					if (get_scene_params().prp_debug)
					{
						auto& uint_stats = get_scene_params().prp_debug_info.uint_stats;
						auto  add_uint_stat = [&](const std::string& key, const uint64_t value)
						{
							const uint clamped = value > static_cast<uint64_t>(std::numeric_limits<uint>::max())
								? std::numeric_limits<uint>::max()
								: static_cast<uint>(value);
							auto&	   dst = uint_stats[key];
							dst = dst > std::numeric_limits<uint>::max() - clamped ? std::numeric_limits<uint>::max() : dst + clamped;
						};
						add_uint_stat("prp.gpu.host_eval.raw_hit_count", total_raw_hits);
						add_uint_stat("prp.gpu.host_eval.valid_hit_count", total_valid_hits);
					}

					std::vector<std::vector<uint2>>	 valid_hits_by_task(num_contours_active);
					std::vector<std::vector<float4>> valid_attrs_by_task(num_contours_active);
					{
						PRP_PROFILE_SCOPE("evaluate_batched_combos.host_evaluation.group_valid_by_task");
						std::vector<uint> valid_counts_by_task(num_contours_active, 0u);
						for (uint hit_idx = 0u; hit_idx < total_valid_hits; ++hit_idx)
						{
							const uint task_id = host_task_ids[hit_idx];
							if (task_id >= num_contours_active)
								LUISA_ERROR("Compacted PRP hit has invalid task id {} for {} active contours.", task_id, num_contours_active);
							valid_counts_by_task[task_id]++;
						}
						for (uint task_idx = 0u; task_idx < num_contours_active; ++task_idx)
						{
							valid_hits_by_task[task_idx].reserve(valid_counts_by_task[task_idx]);
							valid_attrs_by_task[task_idx].reserve(valid_counts_by_task[task_idx]);
						}
						for (uint hit_idx = 0u; hit_idx < total_valid_hits; ++hit_idx)
						{
							const uint task_id = host_task_ids[hit_idx];
							valid_hits_by_task[task_id].push_back(host_valid_hits[hit_idx]);
							valid_attrs_by_task[task_id].push_back(host_valid_attrs[hit_idx]);
						}
					}

					CpuParallel::parallel_for(
						0u, num_contours_active,
						[&](const uint task_idx)
						{
							const uint contour_idx = contour_tasks[task_idx].contour_idx;
							auto&	   runtime = runtimes[contour_idx];
							auto&	   info = contour_results[contour_idx];
							info.separation_direction = runtime.contour_dirs[combo_idx];
							info.contour_idx = runtime.contour_idx;
							info.combo_idx = combo_idx;
							info.unculled_hit_count = contour_raw_hit_counts[task_idx];
							info.evaluated_kring_src = use_partial ? runtime.curr_k_src : std::numeric_limits<uint>::max();
							info.evaluated_kring_dst = use_partial ? runtime.curr_k_dst : std::numeric_limits<uint>::max();
							auto& valid_hits = valid_hits_by_task[task_idx];
							auto& valid_attrs = valid_attrs_by_task[task_idx];
							if (valid_hits.empty())
								return;

							std::vector<RayCasting::HitInfo> hits_info(valid_hits.size());
							{
								PRP_PROFILE_SCOPE("evaluate_batched_combos.host_evaluation.reconstruct_hitinfo");
								CpuParallel::parallel_for(0u, static_cast<uint>(valid_hits.size()),
									[&](const uint i)
									{
										hits_info[i] = gpu_prp_reconstruct_hit_info(host_mesh_data, valid_hits[i], valid_attrs[i], info.separation_direction);
									});
							}

							if (hits_info.empty())
							{
								info.contour_idx = runtime.contour_idx;
								info.combo_idx = combo_idx;
								info.separation_direction = runtime.contour_dirs[combo_idx];
								info.unculled_hit_count = contour_raw_hit_counts[task_idx];
								return;
							}

							{
								PRP_PROFILE_SCOPE("evaluate_batched_combos.host_evaluation.ensure_sparse_hop");
								prp_ensure_sparse_boundary_hops_for_hits(runtime, host_mesh_data, hits_info);
							}
							prp_fill_combo_eval_info(info, runtime, hits_info,
								host_untangling_data, host_collision_data, host_mesh_data,
								sa_rest_face_area, sa_rest_edge_area, sa_rest_vert_area,
								host_face_normal, host_edge_normal, host_vert_normal);
							{
								PRP_PROFILE_SCOPE("evaluate_batched_combos.host_evaluation.store_hit_list");
								info.hit_list = std::move(hits_info);
							}
						},
						1u);
				}

				return contour_results;
			};

			// Forward to the shared implementation so CPU and GPU debug stats stay identical.
			auto export_runtime_combo_debug = [&](const GpuPrpStandardContourRuntime& runtime)
			{
				if (!get_scene_params().prp_debug)
					return;
				lcs::export_runtime_combo_debug(runtime, get_scene_params().prp_debug_info);
			};

			// Evaluate current combo directions for a set of target contours
			auto evaluate_current_runtime_directions = [&](std::vector<GpuPrpStandardContourRuntime>& runtimes,
														   const std::vector<uint>&					  target_contours,
														   const bool								  print_combo_summary)
			{
				PRP_PROFILE_SCOPE("evaluate_current_runtime_directions");
				// Keyed-CSR batch size controls scratch memory; 32 fits 12 GB devices.
				constexpr uint max_combo_runtime_batch_size = 32u;
				lcs::evaluate_current_runtime_directions_template(
					runtimes, target_contours, print_combo_summary,
					[&](const std::vector<uint>& contour_indices)
					{
						PRP_PROFILE_SCOPE("evaluate_current_runtime_directions.build_candidates");
						lcs::prp_build_candidates_for_contours(runtimes, contour_indices, host_mesh_data, host_sim_data);
					},
					[&](const std::vector<uint>& contour_indices,
						const uint				 combo_idx,
						const bool				 use_adaptive) -> std::vector<PRPMinContourHitInfo>
					{
						std::vector<PRPMinContourHitInfo> merged(num_contours);
						for (uint batch_begin = 0u; batch_begin < contour_indices.size(); batch_begin += max_combo_runtime_batch_size)
						{
							const uint		  batch_end = std::min<uint>(batch_begin + max_combo_runtime_batch_size, static_cast<uint>(contour_indices.size()));
							std::vector<uint> batch(contour_indices.begin() + batch_begin, contour_indices.begin() + batch_end);
							{
								PRP_PROFILE_SCOPE("evaluate_current_runtime_directions.update_runtime_candidates_pack_upload");
								update_runtime_candidates(runtimes, batch, combo_idx, use_adaptive);
							}
							luisa::Clock combo_clock;
							combo_clock.tic();
							auto		 batch_results = evaluate_batched_combos(runtimes, batch, combo_idx, use_adaptive);
							const double combo_ms = combo_clock.toc();
							for (const uint contour_idx : batch)
							{
								auto& rt = runtimes[contour_idx];
								list_raycast_times[rt.contour_idx] += combo_ms;
								merged[contour_idx] = std::move(batch_results[contour_idx]);
							}
						}
						return merged;
					});
			};

			PRPContourMergeAdjacency pre_finalize_merge_adjacency(num_contours);
			std::vector<float>		 pre_finalize_penetration_area(num_contours, 0.0f);

			// Forward combo selection to the shared CPU predicate and tie-break order.
			auto select_runtime_best_results = [&](std::vector<GpuPrpStandardContourRuntime>& runtimes,
												   const std::vector<uint>&					  target_contours)
			{
				std::vector<SceneParams::PRPDebugInfo> dummy_debug_infos; // unused by GPU path
				lcs::select_runtime_best_results(list_min_hit_info, dummy_debug_infos, runtimes, target_contours, true);
				{
					PRP_PROFILE_SCOPE("select_runtime_best_results.capture_pre_finalize_contour_merge");
					prp_capture_pre_finalize_contour_merge_data(
						pre_finalize_merge_adjacency,
						pre_finalize_penetration_area,
						list_min_hit_info,
						target_contours,
						vert_in_boundary_flag,
						list_object_ids);
				}
				lcs::prp_finalize_selected_best_hit_infos(
					list_min_hit_info,
					runtimes,
					target_contours,
					host_untangling_data,
					host_collision_data,
					host_mesh_data,
					sa_rest_face_area,
					sa_rest_edge_area,
					sa_rest_vert_area,
					host_face_normal,
					host_edge_normal,
					host_vert_normal);
				if (get_scene_params().prp_debug)
				{
					for (const uint runtime_idx : target_contours)
					{
						const auto& runtime = runtimes[runtime_idx];
						export_runtime_combo_debug(runtime);
					}
				}
				{
					PRP_PROFILE_SCOPE("select_runtime_best_results.release_unselected_hit_lists");
					for (const uint runtime_idx : target_contours)
					{
						auto&	   runtime = runtimes[runtime_idx];
						const auto best_it = find_best_result_in_combo_results(runtime.combo_results);
						if (best_it == runtime.combo_results.end())
							continue;
						const uint best_combo_idx = best_it->combo_idx;
						for (uint combo_idx = 0u; combo_idx < runtime.combo_results.size(); ++combo_idx)
						{
							if (combo_idx == best_combo_idx)
								continue;
							std::vector<RayCasting::HitInfo>().swap(runtime.combo_results[combo_idx].hit_list);
						}
					}
				}
			};

			auto run_batched_runtime_evaluation = [&](std::vector<GpuPrpStandardContourRuntime>& runtimes,
													  const std::vector<uint>&					 target_contours)
			{
				if (target_contours.empty())
					return;

				evaluate_current_runtime_directions(runtimes, target_contours, true);
				prp_optimize_runtime_directions(
					runtimes,
					target_contours,
					host_mesh_data,
					host_sim_data,
					get_scene_params().PRP_direction_optimization_iterations,
					[&](std::vector<GpuPrpStandardContourRuntime>& trial_runtimes,
						const std::vector<uint>&				   trial_contours)
					{
						evaluate_current_runtime_directions(trial_runtimes, trial_contours, false);
					});
				select_runtime_best_results(runtimes, target_contours);
			};

			std::vector<GpuPrpStandardContourRuntime> runtimes(num_contours);
			{
				PRP_PROFILE_SCOPE("build_standard_runtime");
				CpuParallel::parallel_for(0u, static_cast<uint>(active_contours.size()),
					[&](const uint i)
					{
						const uint contour_idx = active_contours[i];
						build_standard_runtime(contour_idx, contours[contour_idx], runtimes[contour_idx]);
					});
			}

			list_raycast_times.assign(num_contours, 0.0);
			run_batched_runtime_evaluation(runtimes, active_contours);
			pass_stage_count = std::max(pass_stage_count, 4u);

			{
				PRP_PROFILE_SCOPE("suppress_adjacent_contours");
				const auto suppressed = prp_select_pre_finalize_contours_to_suppress(
					pre_finalize_merge_adjacency,
					pre_finalize_penetration_area,
					active_contours);

				for (const uint cid : suppressed)
				{
					final_contour_pair_count[cid] = 0u;
					list_raycast_times[cid] = 0.0;
					prp_clear_suppressed_contour_response(list_min_hit_info[cid], cid);
					LUISA_INFO("  Suppressed contour: {}", cid);
				}
				if (!suppressed.empty())
					pass_stage_count = std::max(pass_stage_count, 5u);
			}
			return true;
		};

		luisa::Clock total_raycast_clock;
		total_raycast_clock.tic();
		const bool	used_global_standard_batch = find_contours_hits_batched();
		const float total_raycast_time = total_raycast_clock.toc();

		// Legacy per-contour raycast diagnostics retained for reference.
		LUISA_INFO("End of ray-casting for all contours.");
		std::vector<uint>		 contour_ray_counts;
		std::vector<std::string> formatted_objectives;
		std::vector<std::string> formatted_costs;
		for (uint contour_idx : active_contours)
		{
			contour_ray_counts.push_back(list_min_hit_info[contour_idx].hit_count);
			formatted_objectives.push_back(fmt::format("{:.2e}", prp_response_objective(list_min_hit_info[contour_idx])));
			formatted_costs.push_back(fmt::format("{:.2e}", list_min_hit_info[contour_idx].displacement_cost));
		}
		LUISA_INFO("  Active contours = {}", active_contours);
		LUISA_INFO("  => Ray Count = {}", contour_ray_counts);
		LUISA_INFO("  => Response Objectives = {}", fmt::join(formatted_objectives, ", "));
		LUISA_INFO("  => Displacement Costs = {}", fmt::join(formatted_costs, ", "));

		LUISA_INFO("  GPU Total ray-casting time for all contours: {:.2f} ms", total_raycast_time);

		if (get_scene_params().prp_debug)
		{
			prp_debug_set_uint("prp.regress.pass_stage_count", pass_stage_count);
			for (uint contour_idx = 0; contour_idx < num_contours; contour_idx++)
			{
				if (get_scene_params().should_contour_skip(contour_idx))
					continue;
				prp_write_contour_eval_debug(
					contour_idx,
					list_min_hit_info[contour_idx],
					static_cast<float>(list_raycast_times[contour_idx]),
					final_contour_pair_count[contour_idx],
					list_object_ids[contour_idx],
					list_topology_component_ids[contour_idx],
					host_untangling_data->intersection_contours_info[contour_idx].contour_type);
			}
		}

		host_untangling_data->target_point_template_pairs.clear();
		host_untangling_data->target_point_template_pairs_indices.clear();

		// Construct constraints based on ray-cast results
		{
			{
				uint64_t h = 1469598103934665603ull;
				for (const auto& info : list_min_hit_info)
				{
					det_trace_detail::hash_u64(h, info.contour_idx);
					det_trace_detail::hash_u64(h, info.combo_idx);
					det_trace_detail::hash_bytes(h, &info.separation_direction, sizeof(float3));
					det_trace_detail::hash_bytes(h, &info.penetration_depth, sizeof(float));
					det_trace_detail::hash_bytes(h, &info.penetration_volume, sizeof(float));
					det_trace_detail::hash_bytes(h, &info.penetration_objective, sizeof(float));
					det_trace_detail::hash_bytes(h, &info.penetration_area, sizeof(float));
					det_trace_detail::hash_bytes(h, &info.displacement_cost, sizeof(float));
					det_trace_detail::hash_u64(h, info.hit_count);
					det_trace_detail::hash_u64(h, info.unculled_hit_count);
					det_trace_detail::hash_u64(h, info.evaluated_kring_src);
					det_trace_detail::hash_u64(h, info.evaluated_kring_dst);
					det_trace_detail::hash_u64(h, info.response_eligible);
					det_trace_detail::hash_u64(h, info.no_response_terminal);
				}
				det_trace_detail::trace("P4a_eval_scalars", h);
				uint64_t h2 = 1469598103934665603ull;
				for (const auto& info : list_min_hit_info)
					det_trace_detail::hash_pod_vector(h2, info.hit_list);
				det_trace_detail::trace("P4b_eval_hitlists", h2);
				{
					// Canonical (order-insensitive) hit multiset hash: distinguishes
					// hit ORDER divergence from hit VALUE divergence.
					uint64_t h3 = 1469598103934665603ull;
					for (const auto& info : list_min_hit_info)
					{
						auto sorted = info.hit_list;
						std::sort(sorted.begin(), sorted.end(),
							[](const auto& l, const auto& r)
							{ return std::memcmp(&l, &r, sizeof(l)) < 0; });
						det_trace_detail::hash_pod_vector(h3, sorted);
					}
					det_trace_detail::trace("P4c_canonical_hitsets", h3);
				}
			}
			PRP_PROFILE_SCOPE("construct_constraints");
			make_response_from_min_hitinfo(
				host_untangling_data->target_point_template_pairs,
				host_untangling_data->target_point_template_pairs_indices,
				host_mesh_data,
				host_face_normal,
				host_edge_normal,
				host_vert_normal,
				list_min_hit_info);

			{
				uint64_t h = 1469598103934665603ull;
				det_trace_detail::hash_pod_vector(h, host_untangling_data->target_point_template_pairs);
				det_trace_detail::hash_pod_vector(h, host_untangling_data->target_point_template_pairs_indices);
				det_trace_detail::trace("P5_response_pairs", h);
			}

			upload_response_pairs_to_gpu(device, stream, host_untangling_data, host_collision_data, device_collision_data);
		}

		host_apply_untangling_constraint(
			host_untangling_data->target_point_template_pairs,
			host_untangling_data->target_point_template_pairs_indices,
			host_collision_data->narrow_phase_list_ef_indices,
			host_mesh_data,
			host_untangling_data);

		PRP_PROFILE_EXPORT_JSON("prp_profile_latest_gpu");
	}

} // namespace lcs
