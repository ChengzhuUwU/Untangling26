#pragma once

#include "CollisionDetector/intersection_resolver_helper.h"
#include "CollisionDetector/lbvh.h"
#include "Core/scalar.h"
#include "SimulationCore/base_mesh.h"
#include "SimulationCore/scene_params.h"
#include "SimulationCore/simulation_data.h"
#include "SimulationCore/collision_data.h"
#include "SimulationCore/simulation_type.h"
#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>
#include <queue>
#include <span>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <string>
#include <string_view>
#include <luisa/luisa-compute.h>
#include <Utils/async_compiler.h>

namespace lcs
{
	enum struct ContourType
	{
		Undifined = 0,
		Closed = 1, // Circle to EF pair
		Eight = 2,	// Circle to Loop-vertex
		LL = 3,		// Loop-vertex to Loop-vertex
		BLI = 4,	// Boundary to Loop-vertex (B-L in contour, B-L-I in space)
		Cross = 5,	// Boundary to Loop-vertex - Boundary (Open version of Eight)
		BIBI = 6,	// B in mesh1 to B in mesh2
		BBII = 7,	// B in mesh1 to B in mesh1
	};
	struct ContourInfo
	{
		uint					  num_pairs;
		ContourType				  contour_type;
		std::vector<uint>		  loop_pairs;
		std::vector<uint>		  boundary_pairs;
		std::array<bool, 2>		  mesh_is_closed;
		std::vector<luisa::ubyte> loop_pairs_is_boundary;
	};

	inline constexpr std::string_view contour_type_to_string(ContourType t) noexcept
	{
		switch (t)
		{
			case ContourType::Closed:
				return "Closed";
			case ContourType::Eight:
				return "Eight ";
			case ContourType::LL:
				return "LL    ";
			case ContourType::BLI:
				return "BLI   ";
			case ContourType::Cross:
				return "Cross ";
			case ContourType::BIBI:
				return "BIBI  ";
			case ContourType::BBII:
				return "BBII  ";
			case ContourType::Undifined:
				return "Undifined";
		}
		return "Undifined";
	}

	struct PRPHitLite
	{
		uint4  info0; // [hit_type, src_mesh_idx, src_id, dst_id]
		uint4  src_verts;
		uint4  dst_verts;
		float4 info1; // [bary.x, bary.y, dist, contour_penetration_depth]
		float4 info2; // [dir.x, dir.y, dir.z, 0]
		uint2  extra; // [contour_idx, reserved]
	};

	struct PRPDeviceComboSummary
	{
		uint4  counts0;	  // [valid_hits, vf, ee, fv]
		float4 eval0;	  // [volume, max_depth, objective_area_sum, area]
		float4 eval1;	  // [displacement_area, displacement_kappa, displacement_cost, reserved]
		uint4  frontier0; // [min_frontier_src, min_frontier_dst, min_hop_to_dst, min_hop_to_src]
		uint4  frontier1; // [max_frontier_src, max_frontier_dst, max_hop_to_dst, max_hop_to_src]
	};

} // namespace lcs

LUISA_STRUCT(lcs::PRPHitLite, info0, src_verts, dst_verts, info1, info2, extra){};
LUISA_STRUCT(lcs::PRPDeviceComboSummary, counts0, eval0, eval1, frontier0, frontier1){};

namespace lcs
{
	inline constexpr float prp_response_depth_offset = 3e-3f;

	template <template <typename...> typename BufferType>
	struct UntanglingData : SimulationType
	{
		BufferType<uint2> ef_pair_adj_pairs;
		BufferType<uint>  ef_pair_contour_index;
		BufferType<uint>  ef_pair_mesh_index;
		BufferType<uint>  ef_pair_region_index;
		BufferType<float> ef_pair_length;

		BufferType<uint>								  num_contours;
		std::vector<std::vector<uint>>					  intersection_contours;
		std::vector<ContourInfo>						  intersection_contours_info;
		std::vector<CollisionPair::CollisionPairTemplate> target_point_template_pairs;
		std::vector<uint4>								  target_point_template_pairs_indices;
		std::vector<std::set<uint>>						  region_contains_verts;
		std::vector<std::set<uint>>						  region_contains_verts_1order;
		BufferType<float4>								  ef_pair_pos_2D;
		BufferType<float3>								  ef_pair_pos_3D;
		std::vector<uint2>								  loop_pairs_vertex;
		std::vector<std::vector<uint>>					  ef_pair_adj_pairs_ext;
		std::vector<float3>								  extended_positions;
		std::vector<uint2>								  extended_edges;
		std::vector<uint2>								  ef_pair_semented_indices;
		std::vector<std::vector<uint>>					  ef_pair_adj_pairs_mesh_flip_weight;

		BufferType<uint> vert_region_indices_csr;
		BufferType<uint> edge_region_indices_csr;
		BufferType<uint> face_region_indices_csr;

		// GPU PRP buffers
		BufferType<float3>				  prp_face_normal;
		BufferType<float3>				  prp_edge_normal;
		BufferType<float3>				  prp_vert_normal;
		BufferType<uint>				  prp_vert_adj_contours_csr;
		BufferType<uint>				  prp_total_counts;
		BufferType<uint>				  prp_contour_pairs_csr;
		BufferType<uint>				  prp_runtime_count;
		BufferType<uint>				  prp_runtime_prefix;
		BufferType<uint>				  prp_runtime_flags;
		BufferType<uint>				  prp_runtime_pair_adj_count;
		BufferType<uint2>				  prp_runtime_uint2_indices;
		BufferType<luisa::float4>		  prp_runtime_hit_attrs;
		BufferType<uint2>				  prp_batch_uint2_indices;
		BufferType<luisa::float4>		  prp_batch_hit_attrs;
		BufferType<float3>				  prp_batch_combo_dirs;
		BufferType<uint>				  prp_batch_hit_combo_id;
		BufferType<uint>				  prp_batch_selected_roots;
		BufferType<uint>				  prp_batch_vf_request_src_ids;
		BufferType<uint>				  prp_batch_vf_request_task_id;
		BufferType<uint>				  prp_batch_ee_request_src_ids;
		BufferType<float>				  prp_batch_ee_request_ray_max_dist;
		BufferType<uint>				  prp_batch_ee_request_task_id;
		BufferType<uint>				  prp_batch_fv_request_dst_ids;
		BufferType<uint>				  prp_batch_fv_request_task_id;
		BufferType<uint>				  prp_batch_task_contour_idx;
		BufferType<uint2>				  prp_batch_task_target_face_range;
		BufferType<uint2>				  prp_batch_task_target_edge_range;
		BufferType<float>				  prp_batch_task_target_edge_proj_min;
		BufferType<uint2>				  prp_batch_task_source_face_range;
		BufferType<uint>				  prp_batch_task_src_face_prefix;
		BufferType<uint>				  prp_batch_task_src_face_ids;
		BufferType<uint>				  prp_batch_task_dst_face_prefix;
		BufferType<uint>				  prp_batch_task_dst_face_ids;
		BufferType<uint>				  prp_batch_task_dst_edge_prefix;
		BufferType<uint>				  prp_batch_task_dst_edge_ids;
		BufferType<uint>				  prp_batch_task_boundary_hop_dense;
		BufferType<float>				  prp_batch_task_boundary_geom_dense;
		BufferType<uint>				  prp_batch_task_locality_max_hop;
		BufferType<uint2>				  prp_batch_task_hit_range;
		BufferType<uint>				  prp_batch_compact_task_id;
		BufferType<PRPDeviceComboSummary> prp_batch_combo_summary;
		BufferType<uint>				  prp_csr_count_src;
		BufferType<uint>				  prp_csr_count_dst;
		BufferType<uint>				  prp_csr_prefix_src;
		BufferType<uint>				  prp_csr_prefix_dst;
		BufferType<uint>				  prp_csr_data_src;
		BufferType<uint>				  prp_csr_data_dst;
		BufferType<uint>				  prp_runtime_cc_parent;
		BufferType<uint>				  prp_runtime_cc_changed;
		BufferType<uint>				  prp_runtime_cc_converged;
		BufferType<uint>				  prp_runtime_root_task_idx;
		BufferType<uint>				  prp_runtime_root_hit_count;
		BufferType<uint>				  prp_runtime_root_kept_hit_count;
		BufferType<uint>				  prp_runtime_root_boundary_attached;
		BufferType<uint>				  prp_runtime_root_coverage_count;
		BufferType<uint>				  prp_runtime_root_canonical_key;
		BufferType<uint>				  prp_runtime_root_canonical_key_high;
		BufferType<float>				  prp_runtime_root_max_kept_tgt_geom;
		BufferType<uint>				  prp_runtime_anchor_hash_keys;
		BufferType<uint>				  prp_runtime_coverage_dense_mask;
		BufferType<uint>				  prp_runtime_root_dense_id;
		BufferType<uint>				  prp_runtime_valid_root_count;
		BufferType<uint>				  prp_batch_task_ef_prefix;
		BufferType<uint3>				  prp_batch_task_ef_edges;
		BufferType<uint3>				  prp_batch_task_ef_faces;
		BufferType<uint>				  prp_batch_selected_boundary_counts;
		BufferType<uint>				  prp_batch_selected_hit_counts;
		BufferType<uint>				  prp_batch_selected_keys;
		BufferType<uint>				  prp_batch_selected_key_highs;
		BufferType<uint>				  prp_batch_task_component_count;
		BufferType<uint>				  prp_batch_task_largest_component_hits;
		BufferType<float>				  prp_batch_task_max_kept_tgt_geom;

		std::vector<uint> vert_mutex;

		inline void resize_collision_data_list(luisa::compute::Device& device,
			const uint												   num_verts,
			const uint												   num_faces,
			const uint												   num_edges,
			const uint												   num_dofs,
			const bool												   allocate_contact_list,
			const bool												   allocate_triplet)
		{
			constexpr uint csr_scale = 5u;

			const uint max_intersections = 1 * num_edges;
			const uint max_contour_side_capacity = std::max(2u, max_intersections * 2u);
			const uint max_hits = max_intersections * 4u;
			lcs::Initializer::resize_buffer(device, this->ef_pair_adj_pairs, max_intersections);
			lcs::Initializer::resize_buffer(device, this->ef_pair_contour_index, max_intersections);
			lcs::Initializer::resize_buffer(device, this->ef_pair_mesh_index, max_intersections);
			lcs::Initializer::resize_buffer(device, this->ef_pair_region_index, max_intersections);
			lcs::Initializer::resize_buffer(device, this->ef_pair_length, max_intersections);
			lcs::Initializer::resize_buffer(device, this->ef_pair_pos_2D, max_intersections);
			lcs::Initializer::resize_buffer(device, this->ef_pair_pos_3D, max_intersections);
			lcs::Initializer::resize_buffer(device, this->prp_face_normal, num_faces);
			lcs::Initializer::resize_buffer(device, this->prp_edge_normal, num_edges);
			lcs::Initializer::resize_buffer(device, this->prp_vert_normal, num_verts);
			lcs::Initializer::resize_buffer(device, this->prp_vert_adj_contours_csr, max_intersections * csr_scale);
			lcs::Initializer::resize_buffer(device, this->prp_total_counts, 100);
			lcs::Initializer::resize_buffer(device, this->prp_contour_pairs_csr, max_intersections);
			lcs::Initializer::resize_buffer(device, this->prp_runtime_count, std::max({ num_verts, num_edges, num_faces, 2u }));
			lcs::Initializer::resize_buffer(device, this->prp_runtime_prefix, std::max({ num_verts, num_edges, num_faces, 2u }));
			lcs::Initializer::resize_buffer(device, this->prp_runtime_flags, max_hits);
			lcs::Initializer::resize_buffer(device, this->prp_runtime_pair_adj_count, (max_hits + 255u) & ~255u);
			lcs::Initializer::resize_buffer(device, this->prp_runtime_uint2_indices, max_hits);
			lcs::Initializer::resize_buffer(device, this->prp_batch_uint2_indices, max_hits);
			lcs::Initializer::resize_buffer(device, this->prp_batch_hit_attrs, max_hits);
			lcs::Initializer::resize_buffer(device, this->prp_runtime_hit_attrs, max_hits);
			lcs::Initializer::resize_buffer(device, this->prp_batch_combo_dirs, max_contour_side_capacity);
			lcs::Initializer::resize_buffer(device, this->prp_batch_hit_combo_id, max_hits);
			lcs::Initializer::resize_buffer(device, this->prp_batch_selected_roots, max_contour_side_capacity);
			lcs::Initializer::resize_buffer(device, this->prp_batch_vf_request_src_ids, max_contour_side_capacity * 4u);
			lcs::Initializer::resize_buffer(device, this->prp_batch_ee_request_src_ids, max_contour_side_capacity * 4u);
			lcs::Initializer::resize_buffer(device, this->prp_batch_ee_request_ray_max_dist, max_contour_side_capacity * 4u);
			lcs::Initializer::resize_buffer(device, this->prp_batch_fv_request_dst_ids, max_contour_side_capacity * 4u);
			lcs::Initializer::resize_buffer(device, this->prp_batch_vf_request_task_id, max_contour_side_capacity * 4u);
			lcs::Initializer::resize_buffer(device, this->prp_batch_ee_request_task_id, max_contour_side_capacity * 4u);
			lcs::Initializer::resize_buffer(device, this->prp_batch_fv_request_task_id, max_contour_side_capacity * 4u);
			lcs::Initializer::resize_buffer(device, this->prp_batch_task_contour_idx, max_contour_side_capacity);
			lcs::Initializer::resize_buffer(device, this->prp_batch_task_target_face_range, max_contour_side_capacity);
			lcs::Initializer::resize_buffer(device, this->prp_batch_task_target_edge_range, max_contour_side_capacity);
			lcs::Initializer::resize_buffer(device, this->prp_batch_task_target_edge_proj_min, max_contour_side_capacity);
			lcs::Initializer::resize_buffer(device, this->prp_batch_task_source_face_range, max_contour_side_capacity);
			lcs::Initializer::resize_buffer(device, this->prp_batch_task_src_face_prefix, max_contour_side_capacity + 1u);
			lcs::Initializer::resize_buffer(device, this->prp_batch_task_src_face_ids, max_hits);
			lcs::Initializer::resize_buffer(device, this->prp_batch_task_dst_face_prefix, max_contour_side_capacity + 1u);
			lcs::Initializer::resize_buffer(device, this->prp_batch_task_dst_face_ids, max_hits);
			lcs::Initializer::resize_buffer(device, this->prp_batch_task_dst_edge_prefix, max_contour_side_capacity + 1u);
			lcs::Initializer::resize_buffer(device, this->prp_batch_task_dst_edge_ids, max_hits);
			lcs::Initializer::resize_buffer(device, this->prp_batch_task_boundary_hop_dense, 1u);
			lcs::Initializer::resize_buffer(device, this->prp_batch_task_boundary_geom_dense, 1u);
			lcs::Initializer::resize_buffer(device, this->prp_batch_task_locality_max_hop, max_contour_side_capacity);
			lcs::Initializer::resize_buffer(device, this->prp_batch_task_hit_range, max_contour_side_capacity);
			lcs::Initializer::resize_buffer(device, this->prp_batch_compact_task_id, max_hits);
			lcs::Initializer::resize_buffer(device, this->prp_batch_combo_summary, max_contour_side_capacity);
			const uint N_elem = num_verts + num_edges + num_faces;
			const uint N_elem_256 = (N_elem + 255u) & ~255u;
			lcs::Initializer::resize_buffer(device, this->prp_csr_count_src, N_elem_256);
			lcs::Initializer::resize_buffer(device, this->prp_csr_count_dst, N_elem_256);
			lcs::Initializer::resize_buffer(device, this->prp_csr_prefix_src, N_elem_256);
			lcs::Initializer::resize_buffer(device, this->prp_csr_prefix_dst, N_elem_256);
			lcs::Initializer::resize_buffer(device, this->prp_csr_data_src, max_hits);
			lcs::Initializer::resize_buffer(device, this->prp_csr_data_dst, max_hits);
			lcs::Initializer::resize_buffer(device, this->prp_runtime_cc_parent, max_hits);
			lcs::Initializer::resize_buffer(device, this->prp_runtime_cc_changed, 1);
			lcs::Initializer::resize_buffer(device, this->prp_runtime_cc_converged, 1);
			lcs::Initializer::resize_buffer(device, this->prp_runtime_root_task_idx, max_hits);
			lcs::Initializer::resize_buffer(device, this->prp_runtime_root_hit_count, max_hits);
			lcs::Initializer::resize_buffer(device, this->prp_runtime_root_kept_hit_count, max_hits);
			lcs::Initializer::resize_buffer(device, this->prp_runtime_root_boundary_attached, max_hits);
			lcs::Initializer::resize_buffer(device, this->prp_runtime_root_coverage_count, max_hits);
			lcs::Initializer::resize_buffer(device, this->prp_runtime_root_canonical_key, max_hits);
			lcs::Initializer::resize_buffer(device, this->prp_runtime_root_canonical_key_high, max_hits);
			lcs::Initializer::resize_buffer(device, this->prp_runtime_root_max_kept_tgt_geom, max_hits);
			lcs::Initializer::resize_buffer(device, this->prp_runtime_anchor_hash_keys, 1u);
			lcs::Initializer::resize_buffer(device, this->prp_runtime_coverage_dense_mask, 1u);
			lcs::Initializer::resize_buffer(device, this->prp_runtime_root_dense_id, max_hits);
			lcs::Initializer::resize_buffer(device, this->prp_runtime_valid_root_count, 1u);
			lcs::Initializer::resize_buffer(device, this->prp_batch_task_ef_prefix, max_contour_side_capacity + 1u);
			lcs::Initializer::resize_buffer(device, this->prp_batch_task_ef_edges, 1u);
			lcs::Initializer::resize_buffer(device, this->prp_batch_task_ef_faces, 1u);

			lcs::Initializer::resize_buffer(device, this->prp_batch_selected_boundary_counts, max_contour_side_capacity);
			lcs::Initializer::resize_buffer(device, this->prp_batch_selected_hit_counts, max_contour_side_capacity);
			lcs::Initializer::resize_buffer(device, this->prp_batch_selected_keys, max_contour_side_capacity);
			lcs::Initializer::resize_buffer(device, this->prp_batch_selected_key_highs, max_contour_side_capacity);
			lcs::Initializer::resize_buffer(device, this->prp_batch_task_component_count, max_contour_side_capacity);
			lcs::Initializer::resize_buffer(device, this->prp_batch_task_largest_component_hits, max_contour_side_capacity);
			lcs::Initializer::resize_buffer(device, this->prp_batch_task_max_kept_tgt_geom, max_contour_side_capacity);
			lcs::Initializer::resize_buffer(device, this->num_contours, 1);
			lcs::Initializer::resize_buffer(device, this->vert_region_indices_csr, num_verts * 2 + 1);
			lcs::Initializer::resize_buffer(device, this->edge_region_indices_csr, num_edges * 2 + 1);
			lcs::Initializer::resize_buffer(device, this->face_region_indices_csr, num_faces * 2 + 1);
		}
	};

} // namespace lcs

namespace lcs
{
} // namespace lcs

namespace lcs
{
	void get_intersection_curve_data(std::vector<std::array<float, 3>>& output_positions,
		std::vector<std::array<uint, 2>>&								output_edges,
		std::vector<std::array<float, 3>>&								output_colors,
		UntanglingData<std::vector>*									host_untangling_data,
		CollisionData<std::vector>*										host_collision_data,
		SimulationData<std::vector>*									host_sim_data);

	void intersection_contour_construction_from_ext_adjacent(UntanglingData<std::vector>* host_untangling_data,
		CollisionData<std::vector>*														  host_collision_data);

	ContourType classify_contour_type(const std::vector<CollisionPair::EfPair>& ef_list,
		const std::vector<uint>&												ef_pair_mesh_index,
		const std::vector<std::vector<uint>>&									ef_pair_adj_pairs_ext,
		const std::vector<luisa::ubyte>&										pair_is_boundary,
		std::vector<luisa::ubyte>&												contour_boundary_pairs_is_boundary,
		const std::vector<uint>&												contour,
		const uint																contour_idx);

	bool is_hit_adj_contour(const RayCasting::HitInfo& hit,
		const std::vector<std::vector<uint>>&		   vert_in_boundary_flag,
		const uint									   contour_idx);

	std::vector<uint> get_adj_contours(const RayCasting::HitInfo& hit,
		const std::vector<std::vector<uint>>&					  vert_in_boundary_flag,
		const uint												  contour_idx);

	struct PRPMinContourHitInfo;
	using PRPContourMergeAdjacency = std::vector<std::vector<uint>>;

	void prp_capture_pre_finalize_contour_merge_data(
		PRPContourMergeAdjacency&				 merge_adjacency,
		std::vector<float>&						 pre_finalize_penetration_area,
		const std::vector<PRPMinContourHitInfo>& selected_hit_infos,
		const std::vector<uint>&				 target_contours,
		const std::vector<std::vector<uint>>&	 vert_in_boundary_flag,
		const std::vector<std::array<uint, 2>>&	 contour_object_ids);

	std::vector<uint> prp_select_pre_finalize_contours_to_suppress(
		const PRPContourMergeAdjacency& merge_adjacency,
		const std::vector<float>&		pre_finalize_penetration_area,
		const std::vector<uint>&		active_contours);

	void prp_clear_suppressed_contour_response(
		PRPMinContourHitInfo& hit_info,
		uint				  contour_idx);

	struct PrpStandardContourRuntime;

	std::vector<float3> get_raycasting_direction(
		const std::vector<float3>&				contour_points_3D,
		const std::vector<float>&				contour_points_weight,
		const std::array<std::vector<uint>, 2>& boundary_verts,
		SimulationData<std::vector>*			host_sim_data,
		MeshData<std::vector>*					host_mesh_data);

	void calculate_hop_dist_and_geom_dist_in_mesh(
		std::array<std::vector<uint>, 2>&		vert_to_boundary_hop_dist,
		std::array<std::vector<float>, 2>&		vert_to_boundary_geom_dist,
		std::array<uint, 2>&					mesh_max_hop_dist,
		std::array<float, 2>&					contour_min_geom_dist_to_another_contour,
		std::array<float, 2>&					contour_max_geom_dist_to_another_contour,
		const MeshData<std::vector>*			host_mesh_data,
		const std::array<std::vector<uint>, 2>& boundary_verts,
		const uint								contour_idx,
		const std::array<uint, 2>&				object_ids);
	float3 get_hit_characteristic_normal(const RayCasting::HitInfo& hit, const float3* sa_x);

	void host_resolve_intersections_PRP(luisa::compute::Device& device,
		luisa::compute::Stream&									stream,
		CollisionData<luisa::compute::Buffer>*					device_collision_data,
		CollisionData<std::vector>*								host_collision_data,
		UntanglingData<std::vector>*							host_untangling_data,
		MeshData<std::vector>*									host_mesh_data,
		SimulationData<std::vector>*							host_sim_data);

	struct PRPContourSideIdentity
	{
		std::array<uint, 2> object_ids = {
			std::numeric_limits<uint>::max(), std::numeric_limits<uint>::max()
		};
		std::array<uint, 2> topology_component_ids = {
			std::numeric_limits<uint>::max(), std::numeric_limits<uint>::max()
		};
	};

	PRPContourSideIdentity prp_get_contour_side_identity(
		const CollisionPair::EfPair& first_pair,
		uint						 edge_side,
		const MeshData<std::vector>* host_mesh_data);

	[[nodiscard]] inline bool prp_contour_roles_share_topology(
		const std::array<uint, 2>& object_ids,
		const std::array<uint, 2>& topology_component_ids) noexcept
	{
		const uint invalid = std::numeric_limits<uint>::max();
		return object_ids[0] == object_ids[1]
			&& topology_component_ids[0] != invalid
			&& topology_component_ids[0] == topology_component_ids[1];
	}

	[[nodiscard]] inline bool prp_intrinsic_contour_side_restriction_applies(
		const bool				   enabled,
		const std::array<uint, 2>& object_ids,
		const std::array<uint, 2>& topology_component_ids,
		const bool				   use_deformed_boundary_distance = false) noexcept
	{
		if (!enabled)
			return false;
		if (use_deformed_boundary_distance)
		{
			const uint invalid = std::numeric_limits<uint>::max();
			return object_ids[0] != invalid && object_ids[1] != invalid;
		}
		return prp_contour_roles_share_topology(object_ids, topology_component_ids);
	}

	[[nodiscard]] inline float compute_normalized_phi(
		const float distance_to_own_contour,
		const float distance_to_opposite_contour,
		const float eps = 1e-6f) noexcept
	{
		const float unreachable = std::numeric_limits<float>::max() * 0.5f;
		const bool	g0_valid = std::isfinite(distance_to_own_contour) && distance_to_own_contour < unreachable;
		const bool	g1_valid = std::isfinite(distance_to_opposite_contour) && distance_to_opposite_contour < unreachable;
		if (!g0_valid && !g1_valid)
			return 0.0f;
		if (!g0_valid)
			return 1.0f;
		if (!g1_valid)
			return -1.0f;
		return (distance_to_own_contour - distance_to_opposite_contour) / (distance_to_own_contour + distance_to_opposite_contour + eps);
	}

	[[nodiscard]] inline bool prp_intrinsic_contour_side_owns_vertex(
		const float distance_to_own_contour,
		const float distance_to_opposite_contour,
		const float tau = 0.25f) noexcept
	{
		const float unreachable = std::numeric_limits<float>::max() * 0.5f;
		if (!std::isfinite(distance_to_own_contour)
			|| distance_to_own_contour >= unreachable)
			return false;
		if (!std::isfinite(distance_to_opposite_contour)
			|| distance_to_opposite_contour >= unreachable)
			return true;
		const float phi = compute_normalized_phi(distance_to_own_contour, distance_to_opposite_contour);
		return phi <= tau;
	}

	// Option C: each enabled metric must independently satisfy the same ownership window.
	[[nodiscard]] inline bool prp_dual_metric_contour_side_owns_vertex(
		const float rest_own, const float rest_opposite,
		const float current_own, const float current_opposite,
		const float tau = 0.25f,
		const bool use_rest = true, const bool use_current = true) noexcept
	{
		return (!use_rest || prp_intrinsic_contour_side_owns_vertex(rest_own, rest_opposite, tau))
			&& (!use_current || prp_intrinsic_contour_side_owns_vertex(current_own, current_opposite, tau));
	}

	// Historical normalized-coordinate blend. Incomplete distance pairs abstain;
	// retain the September 14 fallback policy for comparison with that experiment.
	[[nodiscard]] inline float prp_intrinsic_contour_side_blended_phi(
		const float rest_own, const float rest_opposite,
		const float current_own, const float current_opposite,
		const float blend_weight, const float eps = 1e-6f) noexcept
	{
		const auto reachable = [](const float distance) noexcept
		{
			return std::isfinite(distance) && distance < std::numeric_limits<float>::max() * 0.5f;
		};
		if (!std::isfinite(blend_weight) || (!reachable(rest_own) && !reachable(current_own)))
			return std::numeric_limits<float>::infinity();
		const float phi_rest = compute_normalized_phi(rest_own, rest_opposite, eps);
		const float phi_current = compute_normalized_phi(current_own, current_opposite, eps);
		if (!(reachable(rest_own) && reachable(rest_opposite)))
			return phi_current;
		if (!(reachable(current_own) && reachable(current_opposite)))
			return phi_rest;
		const float weight = std::min(std::max(blend_weight, 0.0f), 1.0f);
		return weight * phi_rest + (1.0f - weight) * phi_current;
	}

	[[nodiscard]] inline bool prp_intrinsic_contour_side_owns_vertex_blended(
		const float rest_own, const float rest_opposite,
		const float current_own, const float current_opposite,
		const float tau, const float blend_weight) noexcept
	{
		return prp_intrinsic_contour_side_blended_phi(
				   rest_own, rest_opposite, current_own, current_opposite, blend_weight)
			<= tau;
	}

	struct PRPMinContourHitInfo
	{
		uint   contour_idx = 0u;
		uint   combo_idx = 0u;
		float3 separation_direction = luisa::make_float3(0.0f);
		float  penetration_depth = 0.0f;
		float  penetration_volume = 0.0f;
		float  penetration_objective = 0.0f;
		float  penetration_area = 0.0f;
		float  displacement_cost = 0.0f;
		float  displacement_cost_area = 0.0f;
		float  displacement_cost_kappa = 0.0f;
		uint   hit_count = 0u;
		uint   unculled_hit_count = 0u;
		uint   max_frontier_hop_src = 0u;
		uint   max_frontier_hop_dst = 0u;
		uint   max_hop_to_dst_boundary = 0u;
		uint   max_hop_to_src_boundary = 0u;
		uint   evaluated_kring_src = std::numeric_limits<uint>::max();
		uint   evaluated_kring_dst = std::numeric_limits<uint>::max();
		// Effective adaptive-k stopping result for the selected Cluster Culling response.
		bool							 selected_frontier_converged = true;
		bool							 full_mesh_evaluated = false;
		bool							 self_collision_flipped = false;
		float							 contour_coverage = 0.0f;
		uint							 covered_anchor_count = 0u;
		uint							 boundary_anchor_count = 0u;
		float							 normalized_penetration_objective = std::numeric_limits<float>::infinity();
		float							 normalized_displacement_cost = std::numeric_limits<float>::infinity();
		bool							 response_eligible = false;
		bool							 no_response_terminal = false;
		uint							 expansion_level_count = 0u;
		uint							 direction_optimization_iteration_count = 0u;
		uint							 direction_optimization_accepted_count = 0u;
		std::vector<RayCasting::HitInfo> hit_list;
	};

	struct PrpStandardContourRuntime;

	struct PRPPenetrationEvaluation
	{
		float volume = 0.0f;
		float depth = 0.0f;
		float objective_area = 0.0f;
		float objective = 0.0f;
		float area = 0.0f;
		float displacement_cost = 0.0f;
		float displacement_cost_area = 0.0f;
		float displacement_cost_kappa = 0.0f;
		uint  invalid_hit_count = 0u;
		float invalid_area = 0.0f;
	};

	PRPPenetrationEvaluation prp_eval_penetration_v2(
		const std::vector<RayCasting::HitInfo>& hits_info,
		const std::vector<float>&				sa_rest_face_area,
		const std::vector<float>&				sa_rest_edge_area,
		const std::vector<float>&				sa_rest_vert_area,
		const std::vector<float>&				sa_vert_mass,
		const std::vector<float3>&				face_normal,
		const std::vector<float3>&				edge_normal,
		const std::vector<float3>&				vert_normal,
		bool									need_flip,
		const PrpStandardContourRuntime&		runtime);

	inline bool prp_does_contour_need_antiflip(const std::vector<luisa::ubyte>& loop_pairs_is_boundary) noexcept
	{
		return loop_pairs_is_boundary.empty();
	}

	// Shared EE ray-distance helpers used by both the CPU batched path and the GPU request packing.
	[[nodiscard]] inline float2 prp_compute_edge_projection_bounds(
		const std::vector<uint>&		   edge_ids,
		const float3&					   dir,
		const MeshData<std::vector>&	   host_mesh_data,
		const SimulationData<std::vector>& host_sim_data) noexcept
	{
		if (edge_ids.empty())
			return luisa::make_float2(0.0f);
		float min_proj = std::numeric_limits<float>::max();
		float max_proj = -std::numeric_limits<float>::max();
		for (const uint eid : edge_ids)
		{
			const uint2 edge = host_mesh_data.sa_edges[eid];
			const float proj0 = luisa::dot(host_sim_data.sa_x[edge.x], dir);
			const float proj1 = luisa::dot(host_sim_data.sa_x[edge.y], dir);
			min_proj = std::min(min_proj, std::min(proj0, proj1));
			max_proj = std::max(max_proj, std::max(proj0, proj1));
		}
		return luisa::make_float2(min_proj, max_proj);
	}

	[[nodiscard]] inline float prp_compute_ee_ray_max_dist(
		const uint						   eid,
		const float2					   target_edge_proj,
		const float3&					   dir,
		const float						   dir_len2,
		const MeshData<std::vector>&	   host_mesh_data,
		const SimulationData<std::vector>& host_sim_data) noexcept
	{
		if (dir_len2 <= 1e-12f)
			return 1.0f;
		const uint2 edge = host_mesh_data.sa_edges[eid];
		const float src_min_proj = std::min(luisa::dot(host_sim_data.sa_x[edge.x], dir), luisa::dot(host_sim_data.sa_x[edge.y], dir));
		return std::clamp((target_edge_proj.y - src_min_proj) / dir_len2 + 1e-5f, 0.0f, 1.0f);
	}

	struct PrpStandardContourRuntime
	{
		uint				contour_idx = 0u;
		std::vector<uint>	contour;
		std::array<uint, 2> object_ids = { 0u, 0u };
		std::array<uint, 2> topology_component_ids = {
			std::numeric_limits<uint>::max(), std::numeric_limits<uint>::max()
		};
		std::array<std::vector<uint>, 2> mesh_candidate_verts;
		std::array<std::vector<uint>, 2> mesh_candidate_edges;
		std::array<std::vector<uint>, 2> mesh_candidate_faces;
		std::array<uint2, 2>			 mesh_candidate_vert_range = { luisa::make_uint2(0u), luisa::make_uint2(0u) };
		std::array<uint2, 2>			 mesh_candidate_edge_range = { luisa::make_uint2(0u), luisa::make_uint2(0u) };
		std::array<uint2, 2>			 mesh_candidate_face_range = { luisa::make_uint2(0u), luisa::make_uint2(0u) };

		std::array<std::vector<std::pair<uint, uint>>, 2>  sparse_boundary_hop_dist;
		std::array<std::vector<std::pair<uint, float>>, 2> sparse_boundary_geom_dist;
		std::array<std::vector<std::pair<uint, uint>>, 2>  sparse_boundary_nearest_vert;
		std::array<std::vector<uint>, 2>				   sparse_boundary_hop_frontier;
		std::array<uint, 2>								   sparse_boundary_hop_max_discovered_hop = { 0u, 0u };
		std::array<bool, 2>								   sparse_boundary_hop_exhausted = { false, false };

		std::vector<std::vector<std::pair<uint, float>>>  edge_contains_baries; // contour_idx, edge_1 bary
		std::vector<std::vector<std::pair<uint, float3>>> face_contains_baries;
		bool											  need_flip = false;

		std::array<std::vector<uint>, 2>			   boundary_verts;
		std::array<std::vector<uint>, 2>			   partial_candidate_verts;
		std::array<std::vector<uint>, 2>			   partial_candidate_edges;
		std::array<std::vector<uint>, 2>			   partial_candidate_faces;
		bool										   intrinsic_contour_side_candidate_restriction_applied = false;
		std::array<uint, 2>							   intrinsic_candidate_vertex_count_before = { 0u, 0u };
		std::array<uint, 2>							   intrinsic_candidate_edge_count_before = { 0u, 0u };
		std::array<uint, 2>							   intrinsic_candidate_face_count_before = { 0u, 0u };
		std::array<std::vector<uint>, 2>			   vert_to_boundary_hop_dist;
		std::array<std::vector<float>, 2>			   vert_to_boundary_geom_dist;
		std::array<std::unordered_map<uint, float>, 2> deformed_boundary_distance_cache;
		std::array<std::unordered_set<uint>, 2>		   deformed_boundary_distance_shortened_vertices;
		std::array<float, 2>						   deformed_boundary_distance_min = {
			std::numeric_limits<float>::max(), std::numeric_limits<float>::max()
		};
		std::array<float, 2> deformed_boundary_distance_max = { 0.0f, 0.0f };
		std::array<uint, 2>	 mesh_max_hop_dist = { 0u, 0u };
		std::array<float, 2> contour_min_geom_dist_to_another_contour = { 0.0f, 0.0f };
		std::array<float, 2> contour_max_geom_dist_to_another_contour = { 0.0f, 0.0f };
		std::vector<float3>	 contour_dirs;
		std::vector<uint>	 active_combos;
		// Terminal evaluation state, independent of geometric frontier detachment.
		std::vector<luisa::ubyte>		  combo_converged;
		std::vector<luisa::ubyte>		  combo_objective_pruned;
		std::vector<PRPMinContourHitInfo> combo_results;
		std::vector<luisa::ubyte>		  combo_eval_present;
		bool							  no_response_terminal = false;
		RayCasting::CachedBVH*			  face_bvh = nullptr;
		RayCasting::CachedBVH*			  edge_bvh = nullptr;
		RayCasting::CachedBVH*			  reverse_face_bvh = nullptr;
		uint							  boundary_vert_count_src = 0u;
		uint							  boundary_vert_count_dst = 0u;
		uint							  total_candidate_count = 0u;
		uint							  init_k_src = std::numeric_limits<uint>::max();
		uint							  init_k_dst = std::numeric_limits<uint>::max();
		uint							  curr_k_src = std::numeric_limits<uint>::max();
		uint							  curr_k_dst = std::numeric_limits<uint>::max();
		uint							  mesh_diameter_src = 0u;
		uint							  mesh_diameter_dst = 0u;
		bool							  finished = false;

		[[nodiscard]] uint boundary_hop_distance(const uint boundary_mesh_idx, const uint vid) const noexcept
		{
			constexpr uint hop_inf = std::numeric_limits<uint>::max();

			const auto& dense = vert_to_boundary_hop_dist[boundary_mesh_idx];
			if (vid < dense.size())
				return dense[vid];

			const auto& sparse = sparse_boundary_hop_dist[boundary_mesh_idx];
			const auto	it = std::lower_bound(sparse.begin(), sparse.end(), std::make_pair(vid, 0u),
				[](const auto& a, const auto& b)
				{ return a.first < b.first; });
			return (it != sparse.end() && it->first == vid) ? it->second : hop_inf;
		}

		[[nodiscard]] float boundary_geom_distance(const uint boundary_mesh_idx, const uint vid) const noexcept
		{
			const float geom_inf = std::numeric_limits<float>::infinity();
			const auto& dense = vert_to_boundary_geom_dist[boundary_mesh_idx];
			if (vid < dense.size())
				return dense[vid];

			const auto& sparse = sparse_boundary_geom_dist[boundary_mesh_idx];
			const auto	it = std::lower_bound(sparse.begin(), sparse.end(), std::make_pair(vid, 0.0f),
				[](const auto& a, const auto& b)
				{ return a.first < b.first; });
			return (it != sparse.end() && it->first == vid) ? it->second : geom_inf;
		}
	};

	// Shared PRP runtime preparation used by both host and device resolvers.
	void prp_initialize_runtime_candidate_ranges(
		PrpStandardContourRuntime&	 runtime,
		const MeshData<std::vector>* host_mesh_data);

	void prp_initialize_runtime_boundary_primitives(
		PrpStandardContourRuntime&		   runtime,
		const std::vector<uint>&		   contour,
		const CollisionData<std::vector>*  host_collision_data,
		const UntanglingData<std::vector>* host_untangling_data,
		const std::vector<uint2>&		   ef_indices,
		const std::vector<uint8_t>*		   pair_mesh_flip = nullptr);

	void prp_initialize_runtime_combo_state(PrpStandardContourRuntime& runtime);

	void prp_initialize_runtime_adaptive_state(PrpStandardContourRuntime& runtime);

	void prp_build_runtime_contour_points(
		const std::vector<uint>&		   contour,
		const UntanglingData<std::vector>* host_untangling_data,
		std::vector<float3>&			   contour_points_3D,
		std::vector<float>&				   contour_points_weight);

	// Identity + geometry initialization shared by the CPU batched path and the GPU
	// batched path. Fills the runtime from the contour EF list, computes boundary
	// distances/directions/combo state, and optionally records contour identities.
	void prp_initialize_runtime_identity_and_geometry(
		PrpStandardContourRuntime&		  runtime,
		const uint						  contour_idx,
		const std::vector<uint>&		  contour,
		const CollisionData<std::vector>* host_collision_data,
		UntanglingData<std::vector>*	  host_untangling_data,
		const std::vector<uint2>&		  ef_indices,
		MeshData<std::vector>*			  host_mesh_data,
		SimulationData<std::vector>*	  host_sim_data,
		const std::vector<uint8_t>*		  pair_mesh_flip = nullptr,
		std::vector<std::array<uint, 2>>* contour_object_ids_out = nullptr,
		std::vector<std::array<uint, 2>>* contour_topology_component_ids_out = nullptr);

	void prp_build_candidates_for_contours(
		std::vector<PrpStandardContourRuntime>& runtimes,
		const std::vector<uint>&				contour_indices,
		const MeshData<std::vector>*			host_mesh_data,
		const SimulationData<std::vector>*		host_sim_data);

	void prp_compute_host_normals(
		const MeshData<std::vector>*	   host_mesh_data,
		const SimulationData<std::vector>* host_sim_data,
		std::vector<float3>&			   face_normal,
		std::vector<float3>&			   edge_normal,
		std::vector<float3>&			   vert_normal);

	void prp_build_vertex_boundary_flags(
		const CollisionData<std::vector>*  host_collision_data,
		const UntanglingData<std::vector>* host_untangling_data,
		const MeshData<std::vector>*	   host_mesh_data,
		std::vector<std::vector<uint>>&	   vert_in_boundary_flag);

	template <typename Request>
	std::vector<Request>& prp_find_or_insert_bvh_requests(
		std::vector<std::pair<const RayCasting::CachedBVH*, std::vector<Request>>>& buckets,
		const RayCasting::CachedBVH*												bvh)
	{
		for (auto& [key, requests] : buckets)
			if (key == bvh)
				return requests;
		buckets.emplace_back(bvh, std::vector<Request>{});
		return buckets.back().second;
	}

	struct MinMaxHopInfo
	{
		static std::array<uint, 8> zero() { return {
			std::numeric_limits<uint>::max(), std::numeric_limits<uint>::max(), std::numeric_limits<uint>::max(), std::numeric_limits<uint>::max(),
			0, 0, 0, 0
		}; }
		std::array<uint, 8>		   data = zero();

		uint& min_frontier_hop_src() { return data[0]; }
		uint& min_frontier_hop_dst() { return data[1]; }
		uint& min_hop_to_dst_boundary() { return data[2]; }
		uint& min_hop_to_src_boundary() { return data[3]; }
		uint& max_frontier_hop_src() { return data[4]; }
		uint& max_frontier_hop_dst() { return data[5]; }
		uint& max_hop_to_dst_boundary() { return data[6]; }
		uint& max_hop_to_src_boundary() { return data[7]; }

		const uint& min_frontier_hop_src() const { return data[0]; }
		const uint& min_frontier_hop_dst() const { return data[1]; }
		const uint& min_hop_to_dst_boundary() const { return data[2]; }
		const uint& min_hop_to_src_boundary() const { return data[3]; }
		const uint& max_frontier_hop_src() const { return data[4]; }
		const uint& max_frontier_hop_dst() const { return data[5]; }
		const uint& max_hop_to_dst_boundary() const { return data[6]; }
		const uint& max_hop_to_src_boundary() const { return data[7]; }
	};
	MinMaxHopInfo compute_hit_min_max_frontier_hop(const PrpStandardContourRuntime& runtime, const std::vector<RayCasting::HitInfo>& hits);

	inline bool prp_has_usable_response(const PRPMinContourHitInfo& info) noexcept
	{
		return !info.hit_list.empty()
			&& info.covered_anchor_count > 0u
			&& std::isfinite(info.penetration_objective)
			&& std::isfinite(info.contour_coverage)
			&& info.contour_coverage > 0.0f;
	}

	inline bool prp_has_full_contour_coverage(const PRPMinContourHitInfo& info) noexcept
	{
		return info.boundary_anchor_count > 0u
			&& info.covered_anchor_count == info.boundary_anchor_count;
	}

	inline bool prp_has_valid_hit_info(const PRPMinContourHitInfo& info) noexcept
	{
		return info.response_eligible && prp_has_usable_response(info);
	}

	inline float prp_penetration_objective_divided_by_squared_coverage(
		const float penetration_objective,
		const float contour_coverage) noexcept
	{
		if (!std::isfinite(penetration_objective)
			|| !std::isfinite(contour_coverage)
			|| contour_coverage <= 0.0f)
			return std::numeric_limits<float>::infinity();
		const double coverage = static_cast<double>(contour_coverage);
		const double objective = static_cast<double>(penetration_objective)
			/ (coverage * coverage);
		return std::isfinite(objective)
				&& objective <= static_cast<double>(std::numeric_limits<float>::max())
			? static_cast<float>(objective)
			: std::numeric_limits<float>::infinity();
	}

	inline float prp_response_objective(const PRPMinContourHitInfo& info) noexcept
	{
		return prp_penetration_objective_divided_by_squared_coverage(info.penetration_objective, info.contour_coverage);
	}

	inline bool prp_response_is_better(
		const PRPMinContourHitInfo& candidate,
		const PRPMinContourHitInfo& best) noexcept
	{
		auto scalar_comparison = [](const float lhs, const float rhs) noexcept
		{
			const bool lhs_finite = std::isfinite(lhs);
			const bool rhs_finite = std::isfinite(rhs);
			if (lhs_finite != rhs_finite)
				return lhs_finite ? -1 : 1;
			if (!lhs_finite)
				return 0;
			const float tolerance = 1e-7f
				+ 1e-5f * std::max(std::abs(lhs), std::abs(rhs));
			if (lhs + tolerance < rhs)
				return -1;
			if (rhs + tolerance < lhs)
				return 1;
			return 0;
		};
		if (const int comparison = scalar_comparison(prp_response_objective(candidate), prp_response_objective(best));
			comparison != 0)
			return comparison < 0;
		if (const int comparison = scalar_comparison(candidate.penetration_depth, best.penetration_depth);
			comparison != 0)
			return comparison < 0;
		if (const int comparison = scalar_comparison(candidate.displacement_cost, best.displacement_cost);
			comparison != 0)
			return comparison < 0;
		if (candidate.hit_count != best.hit_count)
			return candidate.hit_count < best.hit_count;
		return candidate.combo_idx < best.combo_idx;
	}

	struct PRPSelectionResult
	{
		uint selected_idx = std::numeric_limits<uint>::max();

		[[nodiscard]] bool has_response() const noexcept
		{
			return selected_idx != std::numeric_limits<uint>::max();
		}
	};

	inline PRPSelectionResult prp_select_response(
		const std::vector<PRPMinContourHitInfo>& combo_results) noexcept
	{
		PRPSelectionResult result;

		for (uint combo_idx = 0u; combo_idx < combo_results.size(); ++combo_idx)
		{
			const auto& candidate = combo_results[combo_idx];
			if (!prp_has_valid_hit_info(candidate))
				continue;
			if (!result.has_response()
				|| prp_response_is_better(
					candidate, combo_results[result.selected_idx]))
				result.selected_idx = combo_idx;
		}
		return result;
	}

	inline uint2 prp_compute_dense_span(const std::vector<uint>& ids, const uint2 default_range) noexcept
	{
		if (ids.empty())
			return luisa::make_uint2(default_range.x, 0u);

		uint min_id = std::numeric_limits<uint>::max();
		uint max_id = 0u;
		for (const uint id : ids)
		{
			min_id = std::min(min_id, id);
			max_id = std::max(max_id, id);
		}
		return luisa::make_uint2(min_id, max_id - min_id + 1u);
	}

	// Select the best combo result using the canonical validity/objective comparator.
	std::vector<PRPMinContourHitInfo>::const_iterator
	find_best_result_in_combo_results(const std::vector<PRPMinContourHitInfo>& combo_results);

	// Refine the best discrete direction on the unit sphere via trial PRP evaluations.
	void prp_optimize_runtime_directions(
		std::vector<PrpStandardContourRuntime>&														  runtimes,
		const std::vector<uint>&																	  target_contours,
		MeshData<std::vector>*																		  host_mesh_data,
		SimulationData<std::vector>*																  host_sim_data,
		uint																						  max_iterations,
		const std::function<void(std::vector<PrpStandardContourRuntime>&, const std::vector<uint>&)>& evaluate_trial_directions);

	// Export per-combo debug info; CPU and GPU share this format.
	void export_runtime_combo_debug(const PrpStandardContourRuntime& runtime,
		SceneParams::PRPDebugInfo&									 debug_sink);

	// Export the per-contour evaluation summary (prp.cN.eval.*, .meta.*, .opt.* keys)
	// shared by the CPU and GPU resolvers.
	void prp_write_contour_eval_debug(
		const uint					contour_idx,
		const PRPMinContourHitInfo& min_hit_info,
		const float					raycast_ms,
		const uint					final_pair_count,
		const std::array<uint, 2>&	object_ids,
		const std::array<uint, 2>&	topology_component_ids,
		const ContourType			contour_type);

	// Select best runtime results and write them to list_min_hit_info.
	void select_runtime_best_results(
		std::vector<PRPMinContourHitInfo>&			  list_min_hit_info,
		std::vector<SceneParams::PRPDebugInfo>&		  contour_debug_infos,
		const std::vector<PrpStandardContourRuntime>& runtimes,
		const std::vector<uint>&					  target_contours,
		bool										  accumulate_stats = false);

	// Shared post-evaluation: run the shared host penetration evaluation on a selected
	// hit list and write depth/volume/objective/cost, EF-anchor coverage, and frontier
	// hops into `info`. Used identically by the CPU batched path and the GPU host
	// evaluation, so both backends report the same combo metrics.
	void prp_fill_combo_eval_info(
		PRPMinContourHitInfo&					info,
		const PrpStandardContourRuntime&		runtime,
		const std::vector<RayCasting::HitInfo>& hits_info,
		const UntanglingData<std::vector>*		host_untangling_data,
		const CollisionData<std::vector>*		host_collision_data,
		const MeshData<std::vector>*			host_mesh_data,
		const std::vector<float>&				sa_rest_face_area,
		const std::vector<float>&				sa_rest_edge_area,
		const std::vector<float>&				sa_rest_vert_area,
		const std::vector<float3>&				face_normal,
		const std::vector<float3>&				edge_normal,
		const std::vector<float3>&				vert_normal);

	void prp_finalize_selected_best_hit_infos(
		std::vector<PRPMinContourHitInfo>&			  list_min_hit_info,
		const std::vector<PrpStandardContourRuntime>& runtimes,
		const std::vector<uint>&					  target_contours,
		UntanglingData<std::vector>*				  host_untangling_data,
		CollisionData<std::vector>*					  host_collision_data,
		MeshData<std::vector>*						  host_mesh_data,
		const std::vector<float>&					  sa_rest_face_area,
		const std::vector<float>&					  sa_rest_edge_area,
		const std::vector<float>&					  sa_rest_vert_area,
		const std::vector<float3>&					  face_normal,
		const std::vector<float3>&					  edge_normal,
		const std::vector<float3>&					  vert_normal);

	inline void prp_expand_sparse_boundary_hop_one_level(
		PrpStandardContourRuntime&	 runtime,
		const MeshData<std::vector>* host_mesh_data,
		const uint					 mesh_idx)
	{
		if (runtime.sparse_boundary_hop_exhausted[mesh_idx])
			return;

		auto& frontier = runtime.sparse_boundary_hop_frontier[mesh_idx];
		if (frontier.empty())
		{
			runtime.sparse_boundary_hop_exhausted[mesh_idx] = true;
			return;
		}

		const uint object_id = runtime.object_ids[mesh_idx];
		const uint prefix_vid = host_mesh_data->prefix_num_verts[object_id];
		const uint suffix_vid = host_mesh_data->prefix_num_verts[object_id + 1u];
		auto&	   hop_vec = runtime.sparse_boundary_hop_dist[mesh_idx];
		auto&	   nearest_vec = runtime.sparse_boundary_nearest_vert[mesh_idx];
		const uint next_hop = runtime.sparse_boundary_hop_max_discovered_hop[mesh_idx] + 1u;
		auto	   query_nearest = [&](const uint vid) noexcept -> uint
		{
			const auto it = std::lower_bound(nearest_vec.begin(), nearest_vec.end(), std::make_pair(vid, 0u),
				[](const auto& a, const auto& b)
				{ return a.first < b.first; });
			return (it != nearest_vec.end() && it->first == vid) ? it->second : vid;
		};

		// Collect and deduplicate all adjacent verts from the current frontier
		std::vector<std::pair<uint, uint>> candidates;
		candidates.reserve(frontier.size() * 4u);
		for (const uint frontier_vid : frontier)
		{
			const uint nearest_boundary_vid = query_nearest(frontier_vid);
			for (const uint adj_vid : host_mesh_data->vert_adj_verts[frontier_vid])
				if (adj_vid >= prefix_vid && adj_vid < suffix_vid)
					candidates.emplace_back(adj_vid, nearest_boundary_vid);
		}
		std::sort(candidates.begin(), candidates.end());
		candidates.erase(std::unique(candidates.begin(), candidates.end(),
							 [](const auto& a, const auto& b)
							 { return a.first == b.first; }),
			candidates.end());

		// Keep only verts not already in hop_vec; preserve sorted order
		std::vector<uint>				   next_frontier;
		std::vector<std::pair<uint, uint>> new_pairs;
		std::vector<std::pair<uint, uint>> new_nearest_pairs;
		for (const auto& [vid, nearest_boundary_vid] : candidates)
		{
			const auto it = std::lower_bound(hop_vec.begin(), hop_vec.end(), std::make_pair(vid, 0u),
				[](const auto& a, const auto& b)
				{ return a.first < b.first; });
			if (it == hop_vec.end() || it->first != vid)
			{
				next_frontier.push_back(vid);
				new_pairs.emplace_back(vid, next_hop);
				new_nearest_pairs.emplace_back(vid, nearest_boundary_vid);
			}
		}

		if (next_frontier.empty())
		{
			frontier.clear();
			runtime.sparse_boundary_hop_exhausted[mesh_idx] = true;
			return;
		}

		// new_pairs is already sorted (candidates was sorted); merge into hop_vec
		std::vector<std::pair<uint, uint>> merged;
		merged.reserve(hop_vec.size() + new_pairs.size());
		std::merge(hop_vec.begin(), hop_vec.end(), new_pairs.begin(), new_pairs.end(),
			std::back_inserter(merged),
			[](const auto& a, const auto& b)
			{ return a.first < b.first; });
		hop_vec = std::move(merged);

		std::vector<std::pair<uint, uint>> merged_nearest;
		merged_nearest.reserve(nearest_vec.size() + new_nearest_pairs.size());
		std::merge(nearest_vec.begin(), nearest_vec.end(), new_nearest_pairs.begin(), new_nearest_pairs.end(),
			std::back_inserter(merged_nearest),
			[](const auto& a, const auto& b)
			{ return a.first < b.first; });
		nearest_vec = std::move(merged_nearest);

		frontier = std::move(next_frontier);
		runtime.sparse_boundary_hop_max_discovered_hop[mesh_idx] = next_hop;
		runtime.mesh_max_hop_dist[mesh_idx] = std::max(runtime.mesh_max_hop_dist[mesh_idx], next_hop);
	}

	inline bool prp_sparse_boundary_hop_query_pending(
		const std::vector<std::pair<uint, uint>>& hop_vec,
		const std::vector<uint>&				  query_verts,
		const uint								  prefix_vid,
		const uint								  suffix_vid)
	{
		for (const uint vid : query_verts)
		{
			if (vid < prefix_vid || vid >= suffix_vid)
				continue;
			const auto it = std::lower_bound(hop_vec.begin(), hop_vec.end(), std::make_pair(vid, 0u),
				[](const auto& a, const auto& b)
				{ return a.first < b.first; });
			if (it == hop_vec.end() || it->first != vid)
				return true;
		}
		return false;
	}

	inline void prp_expand_sparse_boundary_hop_until_known(
		PrpStandardContourRuntime&	 runtime,
		const MeshData<std::vector>* host_mesh_data,
		const uint					 mesh_idx,
		const std::vector<uint>&	 query_verts)
	{
		const uint object_id = runtime.object_ids[mesh_idx];
		const uint prefix_vid = host_mesh_data->prefix_num_verts[object_id];
		const uint suffix_vid = host_mesh_data->prefix_num_verts[object_id + 1u];
		while (!runtime.sparse_boundary_hop_exhausted[mesh_idx]
			&& prp_sparse_boundary_hop_query_pending(runtime.sparse_boundary_hop_dist[mesh_idx], query_verts, prefix_vid, suffix_vid))
		{
			prp_expand_sparse_boundary_hop_one_level(runtime, host_mesh_data, mesh_idx);
		}
	}

	inline void prp_ensure_sparse_boundary_hops_for_hits(
		PrpStandardContourRuntime&				runtime,
		const MeshData<std::vector>*			host_mesh_data,
		const std::vector<RayCasting::HitInfo>& hits)
	{
		if (!runtime.vert_to_boundary_hop_dist[0].empty()
			&& !runtime.vert_to_boundary_hop_dist[1].empty())
			return;

		const bool						 is_self_collision = runtime.object_ids[0] == runtime.object_ids[1];
		std::array<std::vector<uint>, 2> query_verts;
		query_verts[0].reserve(hits.size() * 8u);
		query_verts[1].reserve(hits.size() * 8u);
		for (const auto& hit : hits)
		{
			const auto source_verts = hit.get_source_verts();
			const auto target_verts = hit.get_target_verts();
			query_verts[0].insert(query_verts[0].end(), source_verts.begin(), source_verts.end());
			query_verts[1].insert(query_verts[1].end(), target_verts.begin(), target_verts.end());
			if (is_self_collision)
			{
				query_verts[0].insert(query_verts[0].end(), target_verts.begin(), target_verts.end());
				query_verts[1].insert(query_verts[1].end(), source_verts.begin(), source_verts.end());
			}
		}
		for (auto& verts : query_verts)
		{
			std::sort(verts.begin(), verts.end());
			verts.erase(std::unique(verts.begin(), verts.end()), verts.end());
		}

		prp_expand_sparse_boundary_hop_until_known(runtime, host_mesh_data, 0u, query_verts[0]);
		prp_expand_sparse_boundary_hop_until_known(runtime, host_mesh_data, 1u, query_verts[1]);
	}

	inline void prp_reset_sparse_runtime_eval_state(PrpStandardContourRuntime& runtime)
	{
		for (uint mesh_idx = 0u; mesh_idx < 2u; ++mesh_idx)
		{
			runtime.partial_candidate_verts[mesh_idx].clear();
			runtime.partial_candidate_edges[mesh_idx].clear();
			runtime.partial_candidate_faces[mesh_idx].clear();
		}
		runtime.total_candidate_count = runtime.boundary_vert_count_src + runtime.boundary_vert_count_dst;
	}

	// Build partial candidates within k-ring distances from each contour boundary.
	void build_standard_candidates_at_level(
		PrpStandardContourRuntime&		   runtime,
		const uint						   k_src,
		const uint						   k_dst,
		const MeshData<std::vector>*	   host_mesh_data,
		const SimulationData<std::vector>* host_sim_data);

	inline uint prp_compute_adaptive_kring_initial_level(
		const uint total_boundary_vert_count,
		const uint mesh_diameter) noexcept
	{
		constexpr uint k_init = 3u;
		if (mesh_diameter == 0u)
			return k_init;

		const uint k_init_adaptive = static_cast<uint>(std::ceil(std::sqrt(static_cast<float>(total_boundary_vert_count))));
		return std::max(k_init, std::min(k_init_adaptive, mesh_diameter));
	}

	// Evaluate active PRP direction combinations with adaptive k-ring growth.
	void evaluate_current_runtime_directions_template(
		std::vector<PrpStandardContourRuntime>&														  runtimes,
		const std::vector<uint>&																	  target_contours,
		bool																						  print_combo_summary,
		const std::function<void(const std::vector<uint>&)>&										  build_candidates_fn,
		const std::function<std::vector<PRPMinContourHitInfo>(const std::vector<uint>&, uint, bool)>& eval_combo_fn);

	template <typename HasValidFn, typename SetBoolFn, typename SetUIntFn, typename SetFloatFn, typename SetFloat3Fn>
	inline void prp_export_combo_stage_debug(
		const uint					contour_idx,
		const uint					combo_idx,
		const std::string_view		stage,
		const PRPMinContourHitInfo& info,
		const bool					present,
		HasValidFn&&				has_valid_hit_info,
		SetBoolFn&&					set_bool,
		SetUIntFn&&					set_uint,
		SetFloatFn&&				set_float,
		SetFloat3Fn&&				set_float3)
	{
		const std::string base_key = "prp.c" + std::to_string(contour_idx)
			+ ".combo" + std::to_string(combo_idx)
			+ "." + std::string(stage) + ".";
		const bool has_valid_hits = present && has_valid_hit_info(info);
		set_bool(base_key + "present", present);
		set_bool(base_key + "has_valid_hits", has_valid_hits);
		set_uint(base_key + "present", present ? 1u : 0u);
		set_uint(base_key + "has_valid_hits", has_valid_hits ? 1u : 0u);
		if (!present)
			return;
		set_uint(base_key + "hit_count", info.hit_count);
		set_uint(base_key + "unculled_hit_count", info.unculled_hit_count);
		set_uint(base_key + "opt_iter_count", info.direction_optimization_iteration_count);
		set_uint(base_key + "opt_accepted_count", info.direction_optimization_accepted_count);
		set_bool(base_key + "self_collision_flipped", info.self_collision_flipped);
		set_uint(base_key + "self_collision_flipped", info.self_collision_flipped ? 1u : 0u);
		set_float(base_key + "penetration_depth", info.penetration_depth);
		set_float(base_key + "penetration_volume", info.penetration_volume);
		set_float(base_key + "penetration_objective", info.penetration_objective);
		set_float(base_key + "response_objective", info.penetration_objective);
		set_float(base_key + "displacement_cost", info.displacement_cost);
		set_uint(base_key + "covered_anchor_count", info.covered_anchor_count);
		set_uint(base_key + "boundary_anchor_count", info.boundary_anchor_count);
		set_bool(base_key + "response_eligible", info.response_eligible);
		set_bool(base_key + "frontier_converged", info.selected_frontier_converged);
		set_bool(base_key + "full_mesh_evaluated", info.full_mesh_evaluated);
		set_bool(base_key + "full_contour_coverage", prp_has_full_contour_coverage(info));
		set_uint(base_key + "expansion_level_count", info.expansion_level_count);
		set_uint(base_key + "evaluated_kring_src", info.evaluated_kring_src);
		set_uint(base_key + "evaluated_kring_dst", info.evaluated_kring_dst);
		set_uint(base_key + "max_frontier_hop_src", info.max_frontier_hop_src);
		set_uint(base_key + "max_frontier_hop_dst", info.max_frontier_hop_dst);
		set_float3(base_key + "best_direction", info.separation_direction);
	}

	void make_response_from_min_hitinfo(
		std::vector<CollisionPair::CollisionPairTemplate>& response_collision_pairs,
		std::vector<uint4>&								   response_collision_indices,
		MeshData<std::vector>*							   host_mesh_data,
		const std::vector<float3>&						   face_normal,
		const std::vector<float3>&						   edge_normal,
		const std::vector<float3>&						   vert_normal,
		const std::vector<PRPMinContourHitInfo>&		   list_min_hit_info);

	void upload_response_pairs_to_gpu(
		luisa::compute::Device&				   device,
		luisa::compute::Stream&				   stream,
		UntanglingData<std::vector>*		   host_untangling_data,
		CollisionData<std::vector>*			   host_collision_data,
		CollisionData<luisa::compute::Buffer>* device_collision_data);

	void host_apply_untangling_constraint(
		const std::vector<CollisionPair::CollisionPairTemplate>& target_point_template_pairs,
		const std::vector<uint4>&								 target_point_template_pairs_indices,
		const std::vector<uint2>&								 ef_pair_indices,
		MeshData<std::vector>*									 host_mesh_data,
		UntanglingData<std::vector>*							 host_untangling_data);

} // namespace lcs
