#pragma once

#include "Core/scalar.h"
#include "SimulationCore/base_mesh.h"
#include "SimulationCore/simulation_data.h"
#include "SimulationCore/collision_data.h"
#include "SimulationCore/simulation_type.h"
#include "CollisionDetector/intersection_resolver2.h"
#include "CollisionDetector/lbvh.h"
#include <vector>
#include <string>
#include <string_view>
#include <luisa/luisa-compute.h>
#include <Utils/async_compiler.h>

namespace lcs
{
	class DevicePRPPrecomputeShaders
	{
	public:
		luisa::compute::Shader<1,
			luisa::compute::Buffer<uint3>,
			luisa::compute::Buffer<float3>,
			luisa::compute::Buffer<uint>,
			luisa::compute::Buffer<uint>,
			luisa::compute::Buffer<float3>>
			fn_compute_face_normals;

		luisa::compute::Shader<1,
			luisa::compute::Buffer<uint>,
			luisa::compute::Buffer<float>,
			luisa::compute::Buffer<float3>,
			luisa::compute::Buffer<float3>>
			fn_compute_edge_normals;

		luisa::compute::Shader<1,
			luisa::compute::Buffer<uint>,
			luisa::compute::Buffer<float>,
			luisa::compute::Buffer<float3>,
			luisa::compute::Buffer<float3>>
			fn_compute_vert_normals;

		luisa::compute::Shader<1,
			luisa::compute::Buffer<uint>>
			fn_reset_uint;

		luisa::compute::Shader<1,
			luisa::compute::Buffer<uint>, uint>
			fn_reset_uint_with_value;
		luisa::compute::Shader<1,
			luisa::compute::Buffer<uint>,
			luisa::compute::Buffer<uint>,
			uint>
			fn_copy_uint_buffer;

		luisa::compute::Shader<1,
			luisa::compute::Buffer<uint>,
			luisa::compute::Buffer<uint>,
			luisa::compute::Buffer<uint>,
			uint>
			fn_exclusive_prefix_sum;

		// ── Batch raycasting kernels (all combos in one dispatch) ───────
		luisa::compute::Shader<1,
			luisa::compute::Buffer<float3>,			// sa_x
			luisa::compute::Buffer<uint3>,			// sa_faces
			luisa::compute::Buffer<CompressedAABB>, // target_node_aabb
			luisa::compute::Buffer<uint2>,			// target_children
			luisa::compute::Buffer<uint>,			// target_object_idx
			luisa::compute::Buffer<uint>,			// target_is_healthy
			luisa::compute::Buffer<uint>,			// request_source_vert_ids
			luisa::compute::Buffer<uint>,			// request_task_id
			luisa::compute::Buffer<uint2>,			// task_target_face_range
			luisa::compute::Buffer<uint>,			// task_target_face_prefix
			luisa::compute::Buffer<uint>,			// task_target_face_ids
			luisa::compute::Buffer<float3>,			// task_dirs
			float,									// max_ray_dist
			luisa::compute::Buffer<uint>,			// task_offsets
			luisa::compute::Buffer<uint2>,			// out_indices
			luisa::compute::Buffer<luisa::float4>,	// out_attrs (bary.x, bary.y, dist, _)
			luisa::compute::Buffer<uint>,			// out_task_id
			uint,									// per_task_capacity
			uint>									// request_count
			fn_task_raycast_vf_scatter_uint2;

		luisa::compute::Shader<1,
			luisa::compute::Buffer<float3>,			// sa_x
			luisa::compute::Buffer<uint2>,			// sa_edges
			luisa::compute::Buffer<CompressedAABB>, // target_node_aabb
			luisa::compute::Buffer<uint2>,			// target_children
			luisa::compute::Buffer<uint>,			// target_object_idx
			luisa::compute::Buffer<uint>,			// target_is_healthy
			luisa::compute::Buffer<uint>,			// request_source_edge_ids
			luisa::compute::Buffer<float>,			// request_ray_max_dist
			luisa::compute::Buffer<uint>,			// request_task_id
			luisa::compute::Buffer<uint2>,			// task_target_edge_range
			luisa::compute::Buffer<float>,			// task_target_edge_proj_min
			luisa::compute::Buffer<uint>,			// task_target_edge_prefix
			luisa::compute::Buffer<uint>,			// task_target_edge_ids
			luisa::compute::Buffer<float3>,			// task_dirs
			float,									// max_ray_dist
			luisa::compute::Buffer<uint>,			// task_offsets
			luisa::compute::Buffer<uint2>,			// out_indices
			luisa::compute::Buffer<luisa::float4>,	// out_attrs (s, t1, t2, _)
			luisa::compute::Buffer<uint>,			// out_task_id
			uint,									// per_task_capacity
			uint>									// request_count
			fn_task_raycast_ee_scatter_uint2;

		luisa::compute::Shader<1,
			luisa::compute::Buffer<float3>,			// sa_x
			luisa::compute::Buffer<uint3>,			// sa_faces
			luisa::compute::Buffer<CompressedAABB>, // target_node_aabb
			luisa::compute::Buffer<uint2>,			// target_children
			luisa::compute::Buffer<uint>,			// target_object_idx
			luisa::compute::Buffer<uint>,			// target_is_healthy
			luisa::compute::Buffer<uint>,			// request_dst_vert_ids
			luisa::compute::Buffer<uint>,			// request_task_id
			luisa::compute::Buffer<uint2>,			// task_source_face_range
			luisa::compute::Buffer<uint>,			// task_source_face_prefix
			luisa::compute::Buffer<uint>,			// task_source_face_ids
			luisa::compute::Buffer<float3>,			// task_dirs
			float,									// max_ray_dist
			luisa::compute::Buffer<uint>,			// task_offsets
			luisa::compute::Buffer<uint2>,			// out_indices
			luisa::compute::Buffer<luisa::float4>,	// out_attrs (bary.x, bary.y, dist, _)
			luisa::compute::Buffer<uint>,			// out_task_id
			uint,									// per_task_capacity
			uint>									// request_count
			fn_task_raycast_fv_scatter_uint2;

		luisa::compute::Shader<1,
			luisa::compute::Buffer<uint2>,		   // in_hits
			luisa::compute::Buffer<luisa::float4>, // in_attrs
			luisa::compute::Buffer<uint>,		   // in_task_id
			luisa::compute::Buffer<uint>,		   // hit_valid
			luisa::compute::Buffer<uint2>,		   // out_hits
			luisa::compute::Buffer<luisa::float4>, // out_attrs
			luisa::compute::Buffer<uint>,		   // out_task_id
			luisa::compute::Buffer<uint>,		   // out_count
			uint>								   // total_hits
			fn_prp_compact_valid_hits_with_task;

		luisa::compute::Shader<1,
			luisa::compute::Buffer<PRPDeviceComboSummary>,
			uint>
			fn_prp_reset_device_combo_eval;

		luisa::compute::Shader<1,
			luisa::compute::Buffer<uint2>,		   // packed hits
			luisa::compute::Buffer<luisa::float4>, // packed attrs
			luisa::compute::Buffer<uint>,		   // packed task id
			luisa::compute::Buffer<uint>,		   // hit valid
			luisa::compute::Buffer<float3>,		   // task dirs

			luisa::compute::Buffer<uint>, // task boundary hop dense

			luisa::compute::Buffer<uint3>,	// faces
			luisa::compute::Buffer<uint2>,	// edges
			luisa::compute::Buffer<float3>, // positions
			luisa::compute::Buffer<float>,	// rest vert area
			luisa::compute::Buffer<float>,	// rest edge area
			luisa::compute::Buffer<float>,	// rest face area
			luisa::compute::Buffer<float>,	// vert mass
			luisa::compute::Buffer<float3>, // vert normals
			luisa::compute::Buffer<float3>, // edge normals
			luisa::compute::Buffer<float3>, // face normals
			luisa::compute::Buffer<PRPDeviceComboSummary>,
			uint, // num verts
			uint> // total hits
			fn_prp_accumulate_combo_summary;

		luisa::compute::Shader<1,
			luisa::compute::Buffer<uint2>,		   // in_slice_hits
			luisa::compute::Buffer<luisa::float4>, // in_slice_attrs
			luisa::compute::Buffer<uint>,		   // task_coarse_hit_counts
			luisa::compute::Buffer<uint>,		   // task_coarse_offsets
			luisa::compute::Buffer<uint>,		   // task_target_face_prefix
			luisa::compute::Buffer<uint>,		   // task_target_face_ids
			luisa::compute::Buffer<uint>,		   // task_target_edge_prefix
			luisa::compute::Buffer<uint>,		   // task_target_edge_ids
			luisa::compute::Buffer<uint>,		   // task_source_face_prefix
			luisa::compute::Buffer<uint>,		   // task_source_face_ids
			luisa::compute::Buffer<uint>,		   // exact_offsets_or_counts
			luisa::compute::Buffer<uint>,		   // exact_type_counts
			luisa::compute::Buffer<uint>,		   // overflow_flag
			luisa::compute::Buffer<uint2>,		   // out_packed_hits
			luisa::compute::Buffer<luisa::float4>, // out_packed_attrs
			luisa::compute::Buffer<uint>,		   // out_packed_task_id
			uint,								   // per_task_capacity
			uint,								   // num_active_tasks
			uint,								   // total_coarse_hits
			uint,								   // total_exact_capacity
			uint>								   // mode_write
			fn_prp_filter_exact_candidate_indexed;

		luisa::compute::Shader<1,
			luisa::compute::Buffer<uint>, // parent
			uint>						  // count
			fn_prp_init_cc;

		luisa::compute::Shader<1,
			luisa::compute::Buffer<uint>, // parent
			luisa::compute::Buffer<uint>, // converged (1-slot sticky early-out)
			uint>						  // count
			fn_prp_compress_cc;

		luisa::compute::Shader<1,
			luisa::compute::Buffer<uint>, // changed (1-slot atomic)
			luisa::compute::Buffer<uint>> // converged (1-slot sticky)
			fn_prp_check_cc_converged;

		luisa::compute::Shader<1,
			luisa::compute::Buffer<uint>,  // root_task_idx
			luisa::compute::Buffer<uint>,  // root_hit_count
			luisa::compute::Buffer<uint>,  // root_kept_hit_count
			luisa::compute::Buffer<uint>,  // root_boundary_attached
			luisa::compute::Buffer<uint>,  // root_coverage_count
			luisa::compute::Buffer<uint>,  // root_canonical_key_low
			luisa::compute::Buffer<uint>,  // root_canonical_key_high
			luisa::compute::Buffer<float>, // root_max_kept_tgt_geom
			luisa::compute::Buffer<uint>,  // root_dense_id
			uint>						   // total_hits
			fn_prp_reset_cluster_selection;

		luisa::compute::Shader<1,
			luisa::compute::Buffer<uint>,  // selected_roots
			luisa::compute::Buffer<uint>,  // selected_boundary_counts
			luisa::compute::Buffer<uint>,  // selected_hit_counts
			luisa::compute::Buffer<uint>,  // selected_keys_low
			luisa::compute::Buffer<uint>,  // selected_keys_high
			luisa::compute::Buffer<uint>,  // task_component_count
			luisa::compute::Buffer<uint>,  // task_largest_component_hits
			luisa::compute::Buffer<float>, // task_max_kept_tgt_geom
			uint,						   // task_count
			uint>						   // total_hits_sentinel
			fn_prp_reset_cluster_task_selection;

		luisa::compute::Shader<1,
			luisa::compute::Buffer<uint2>, // packed_hits
			luisa::compute::Buffer<uint>,  // packed_task_id
			luisa::compute::Buffer<uint>,  // precluster_valid
			luisa::compute::Buffer<uint>,  // parent
			luisa::compute::Buffer<uint3>, // faces
			luisa::compute::Buffer<uint2>, // edges
			luisa::compute::Buffer<uint>,  // vert_boundary_flags_csr
			luisa::compute::Buffer<uint>,  // task_contour_idx
			luisa::compute::Buffer<uint>,  // task_boundary_hop_dense
			luisa::compute::Buffer<float>, // task_boundary_geom_dense
			luisa::compute::Buffer<uint>,  // task_locality_max_hop
			luisa::compute::Buffer<uint>,  // root_task_idx
			luisa::compute::Buffer<uint>,  // root_hit_count
			luisa::compute::Buffer<uint>,  // root_kept_hit_count
			luisa::compute::Buffer<uint>,  // root_boundary_attached
			luisa::compute::Buffer<uint>,  // root_canonical_key_high
			luisa::compute::Buffer<float>, // root_max_kept_tgt_geom
			luisa::compute::Buffer<uint>,  // root_error_flag
			uint,						   // total_hits
			uint,						   // num_verts
			uint>						   // use_coverage_locality
			fn_prp_accumulate_cluster_root_stats;

		luisa::compute::Shader<1,
			luisa::compute::Buffer<uint>, // root_hit_count
			luisa::compute::Buffer<uint>, // root_boundary_attached
			luisa::compute::Buffer<uint>, // root_dense_id
			luisa::compute::Buffer<uint>, // valid_root_count
			luisa::compute::Buffer<uint>, // overflow_flag
			uint,						  // total_hits
			uint>						  // max_dense_roots
			fn_prp_assign_dense_cluster_roots;

		luisa::compute::Shader<1,
			luisa::compute::Buffer<uint2>, // packed_hits
			luisa::compute::Buffer<uint>,  // packed_task_id
			luisa::compute::Buffer<uint>,  // precluster_valid
			luisa::compute::Buffer<uint>,  // parent
			luisa::compute::Buffer<uint3>, // faces
			luisa::compute::Buffer<uint2>, // edges
			luisa::compute::Buffer<uint>,  // task_boundary_hop_dense
			luisa::compute::Buffer<uint>,  // task_locality_max_hop
			luisa::compute::Buffer<uint>,  // root_dense_id
			luisa::compute::Buffer<uint>,  // dense_mask
			uint,						   // total_hits
			uint,						   // num_verts
			uint,						   // stride
			uint,						   // max_dense_roots
			uint>						   // use_coverage_locality
			fn_prp_fill_dense_cluster_coverage_mask;

		luisa::compute::Shader<1,
			luisa::compute::Buffer<uint>,  // root_task_idx
			luisa::compute::Buffer<uint>,  // root_hit_count
			luisa::compute::Buffer<uint>,  // root_boundary_attached
			luisa::compute::Buffer<uint>,  // root_dense_id
			luisa::compute::Buffer<uint>,  // task_ef_prefix
			luisa::compute::Buffer<uint3>, // task_ef_edges (vertex, vertex, role)
			luisa::compute::Buffer<uint3>, // task_ef_faces (stored EF vertices)
			luisa::compute::Buffer<uint>,  // dense_mask
			luisa::compute::Buffer<uint>,  // root_coverage_count
			uint,						   // num_verts
			uint,						   // stride
			uint,						   // max_dense_roots
			uint>						   // total_hits
			fn_prp_count_cluster_ef_coverage;

		luisa::compute::Shader<1,
			luisa::compute::Buffer<uint2>, // packed_hits
			luisa::compute::Buffer<uint>,  // precluster_valid
			luisa::compute::Buffer<uint>,  // parent
			luisa::compute::Buffer<uint>,  // root_canonical_key_low
			luisa::compute::Buffer<uint>,  // root_canonical_key_high
			luisa::compute::Buffer<uint>,  // root_error_flag
			uint>						   // total_hits
			fn_prp_accumulate_cluster_root_key_low;

		// Parallel per-task winner selection replaces the former serial root scan.
		luisa::compute::Shader<1,
			luisa::compute::Buffer<uint>, // root_task_idx
			luisa::compute::Buffer<uint>, // root_hit_count
			luisa::compute::Buffer<uint>, // root_boundary_attached
			luisa::compute::Buffer<uint>, // root_coverage_count
			luisa::compute::Buffer<uint>, // root_kept_hit_count
			luisa::compute::Buffer<uint>, // root_canonical_key_low
			luisa::compute::Buffer<uint>, // root_canonical_key_high
			luisa::compute::Buffer<uint>, // task_best_cov
			luisa::compute::Buffer<uint>, // task_best_kept
			luisa::compute::Buffer<uint>, // task_best_key_high
			luisa::compute::Buffer<uint>, // task_best_key_low
			luisa::compute::Buffer<uint>, // task_best_root
			luisa::compute::Buffer<uint>, // task_component_count
			luisa::compute::Buffer<uint>, // task_largest_component_hits
			uint,						  // pass_id
			uint>						  // total_hits
			fn_prp_select_cluster_roots_pass;

		luisa::compute::Shader<1,
			luisa::compute::Buffer<uint>,  // task_best_cov
			luisa::compute::Buffer<uint>,  // task_best_kept
			luisa::compute::Buffer<uint>,  // task_best_key_high
			luisa::compute::Buffer<uint>,  // task_best_key_low
			luisa::compute::Buffer<uint>,  // task_best_root
			luisa::compute::Buffer<float>, // root_max_kept_tgt_geom
			luisa::compute::Buffer<uint>,  // selected_roots
			luisa::compute::Buffer<uint>,  // selected_boundary_counts
			luisa::compute::Buffer<uint>,  // selected_hit_counts
			luisa::compute::Buffer<uint>,  // selected_keys_low
			luisa::compute::Buffer<uint>,  // selected_keys_high
			luisa::compute::Buffer<float>, // task_max_kept_tgt_geom
			uint,						   // num_tasks
			uint>						   // total_hits
			fn_prp_select_cluster_roots_finalize;

		luisa::compute::Shader<1,
			luisa::compute::Buffer<uint2>, // packed_hits
			luisa::compute::Buffer<uint>,  // packed_task_id
			luisa::compute::Buffer<uint>,  // parent
			luisa::compute::Buffer<uint3>, // faces
			luisa::compute::Buffer<uint2>, // edges
			luisa::compute::Buffer<uint>,  // task_boundary_hop_dense
			luisa::compute::Buffer<uint>,  // task_locality_max_hop
			luisa::compute::Buffer<uint>,  // selected_roots
			luisa::compute::Buffer<uint>,  // hit_valid in/out
			uint,						   // total_hits
			uint,						   // num_verts
			uint,						   // keep_all_for_selected_task
			uint>						   // use_coverage_locality
			fn_prp_write_cluster_valid_flags;

		// ── Batched GPU culling shaders (all combos in one dispatch) ─────────

		luisa::compute::Shader<1,
			luisa::compute::Buffer<uint>, // count_src
			luisa::compute::Buffer<uint>, // touched_rows
			luisa::compute::Buffer<uint>, // touched_counts
			luisa::compute::Buffer<uint>, // touched_count
			uint>						  // compact_capacity
			fn_batch_gather_touched_counts;

		luisa::compute::Shader<1,
			luisa::compute::Buffer<uint>, // prefix_src
			luisa::compute::Buffer<uint>, // touched_rows
			luisa::compute::Buffer<uint>, // touched_prefix
			luisa::compute::Buffer<uint>, // touched_count
			uint>						  // compact_capacity
			fn_batch_scatter_touched_prefix;

		luisa::compute::Shader<1,
			luisa::compute::Buffer<uint>, // count_src
			luisa::compute::Buffer<uint>, // touched_rows
			luisa::compute::Buffer<uint>, // touched_count
			uint>						  // compact_capacity
			fn_batch_clear_touched_rows;

		luisa::compute::Shader<1,
			luisa::compute::Buffer<uint2>, // packed_hits
			luisa::compute::Buffer<uint>,  // packed_task_id

			luisa::compute::Buffer<uint>, // keyed_hash_keys
			luisa::compute::Buffer<uint>, // count_src
			luisa::compute::Buffer<uint>, // touched_rows
			luisa::compute::Buffer<uint>, // touched_count
			luisa::compute::Buffer<uint>, // overflow
			uint,						  // N_verts
			uint,						  // N_edges
			uint,						  // elem_key_stride
			uint,						  // hash_capacity
			uint,						  // total_hits
			uint>						  // row_mode (0=keyed hash, 1=dense task rows)
			fn_batch_keyed_count_hits_per_elem;

		luisa::compute::Shader<1,
			luisa::compute::Buffer<uint2>, // packed_hits
			luisa::compute::Buffer<uint>,  // packed_task_id

			luisa::compute::Buffer<uint>, // keyed_hash_keys
			luisa::compute::Buffer<uint>, // prefix_src
			luisa::compute::Buffer<uint>, // fill_counts
			luisa::compute::Buffer<uint>, // csr_data_src
			luisa::compute::Buffer<uint>, // overflow
			uint,						  // N_verts
			uint,						  // N_edges
			uint,						  // elem_key_stride
			uint,						  // hash_capacity
			uint,						  // total_hits
			uint>						  // row_mode (0=keyed hash, 1=dense task rows)
			fn_batch_keyed_fill_csr;

		luisa::compute::Shader<1,
			luisa::compute::Buffer<uint>,	// count_src
			luisa::compute::Buffer<uint>,	// prefix_src
			luisa::compute::Buffer<uint>,	// csr_data_src
			luisa::compute::Buffer<uint>,	// keyed_hash_keys
			luisa::compute::Buffer<uint2>,	// packed_hits
			luisa::compute::Buffer<float4>, // packed_attrs
			luisa::compute::Buffer<uint>,	// packed_task_id
			luisa::compute::Buffer<uint>,	// parent
			luisa::compute::Buffer<uint>,	// changed
			luisa::compute::Buffer<uint>,	// converged
			uint, uint, uint, uint, uint>
			fn_batch_hook_cc_direct_keyed;

		luisa::compute::Shader<1,
			luisa::compute::Buffer<uint2>,	// packed_hits
			luisa::compute::Buffer<float4>, // packed_attrs
			luisa::compute::Buffer<uint>,	// packed_task_id
			luisa::compute::Buffer<uint2>,	// task_hit_range
			luisa::compute::Buffer<float3>, // task_dirs
			luisa::compute::Buffer<uint>,	// parent
			luisa::compute::Buffer<uint>,	// changed
			luisa::compute::Buffer<uint>,	// converged
			luisa::compute::Buffer<uint3>,	// sa_faces
			luisa::compute::Buffer<uint2>,	// sa_edges
			luisa::compute::Buffer<uint>,	// sa_face_mesh_id
			luisa::compute::Buffer<uint>,	// sa_vert_mesh_id
			luisa::compute::Buffer<uint>,	// sa_vert_mesh_type
			luisa::compute::Buffer<float3>, // prp_face_normal
			luisa::compute::Buffer<float3>, // prp_edge_normal
			uint,							// total_hits
			uint,							// all_solid
			float,							// eps_n
			float>							// eps_d
			fn_batch_hook_solid_interior_adjacency;

		void reset()
		{
			*this = DevicePRPPrecomputeShaders{};
		}

		void compile(AsyncCompiler& compiler, MeshData<luisa::compute::Buffer>* device_mesh_data);

	private:
		bool compiled = false;

		void compile_normals(AsyncCompiler& compiler);
		void compile_buffer_utils(AsyncCompiler& compiler);
		void compile_hit_compaction(AsyncCompiler& compiler);
		void compile_product_adjacency_and_union_find(AsyncCompiler& compiler, MeshData<luisa::compute::Buffer>* device_mesh_data);
		void compile_cluster_culling(AsyncCompiler& compiler);
		void compile_batch_keyed_csr(AsyncCompiler& compiler);
		void compile_exact_filter_and_raycast(AsyncCompiler& compiler);
	};

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
		DevicePRPPrecomputeShaders&								  prp_shaders);

	void device_compute_prp_normals(luisa::compute::Device& device,
		luisa::compute::Stream&								stream,
		UntanglingData<luisa::compute::Buffer>*				device_untangling_data,
		MeshData<luisa::compute::Buffer>*					device_mesh_data,
		SimulationData<luisa::compute::Buffer>*				device_sim_data,
		const uint											num_faces,
		const uint											num_edges,
		const uint											num_verts,
		DevicePRPPrecomputeShaders&							prp_shaders);

} // namespace lcs
