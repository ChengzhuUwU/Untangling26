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
#include <array>
#include <cmath>
#include <iterator>
#include <limits>
#include <map>
#include <numeric>
#include <queue>

#include <set>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <mutex>
#include <vector>

namespace lcs
{
	// Unlimited target-hop locality, matching the established default on both paths.
	constexpr uint kPrpLocalityMaxHop = std::numeric_limits<uint>::max();

	namespace
	{
		constexpr float kPrpSolutionProjectionThreshold = 0.1f;

		struct PrpHitSolutionSignature
		{
			float  src_q = 0.0f;
			float  dst_q = 0.0f;
			int8_t src_sign = 0;
			int8_t dst_sign = 0;
			uint   key = 4u; // (0, 0)
		};

		[[nodiscard]] int8_t prp_solution_projection_sign(const float q, const float threshold) noexcept
		{
			if (!std::isfinite(q) || std::abs(q) <= threshold)
				return 0;
			return q > 0.0f ? int8_t{ 1 } : int8_t{ -1 };
		}

		[[nodiscard]] uint prp_solution_sign_code(const int8_t sign) noexcept
		{
			return sign < 0 ? 0u : (sign > 0 ? 2u : 1u);
		}

		[[nodiscard]] PrpHitSolutionSignature prp_classify_hit_solution(
			const RayCasting::HitInfo& hit,
			const std::vector<float3>& face_normal,
			const std::vector<float3>& edge_normal,
			const std::vector<float3>& vert_normal,
			const float				   threshold)
		{
			float3 n_src = luisa::make_float3(0.0f);
			float3 n_dst = luisa::make_float3(0.0f);
			if (hit.is_vf())
			{
				n_src = vert_normal[hit.get_vid()];
				n_dst = face_normal[hit.get_fid()];
			}
			else if (hit.is_ee())
			{
				n_src = edge_normal[hit.get_eid1()];
				n_dst = edge_normal[hit.get_eid2()];
			}
			else if (hit.is_fv())
			{
				n_src = face_normal[hit.get_fid()];
				n_dst = vert_normal[hit.get_vid()];
			}
			else
			{
				LUISA_ERROR("Unknown PRP hit type while classifying RayCasting Solution.");
			}

			const float3 r = hit.get_direction();
			const float	 r_len = luisa::length(r);
			const float	 src_len = luisa::length(n_src);
			const float	 dst_len = luisa::length(n_dst);

			PrpHitSolutionSignature result;
			result.src_q = r_len > 1e-12f && src_len > 1e-12f
				? luisa::dot(n_src / src_len, r / r_len)
				: 0.0f;
			result.dst_q = r_len > 1e-12f && dst_len > 1e-12f
				? luisa::dot(n_dst / dst_len, r / r_len)
				: 0.0f;
			result.src_sign = prp_solution_projection_sign(result.src_q, threshold);
			result.dst_sign = prp_solution_projection_sign(result.dst_q, threshold);
			result.key = 3u * prp_solution_sign_code(result.src_sign) + prp_solution_sign_code(result.dst_sign);
			return result;
		}

		// Pack the source-mesh bit separately from the visual solution label; other modes keep the 3x3 sign-key coloring.
		constexpr uint kPrpResponseVisualLabelBits = 8u;
		constexpr uint kPrpResponseVisualLabelCount = 1u << kPrpResponseVisualLabelBits;
		constexpr uint kPrpResponseVisualLabelMask = kPrpResponseVisualLabelCount - 1u;

		[[nodiscard]] constexpr uint prp_pack_response_region_label(
			const uint contour_idx,
			const uint mesh_idx,
			const uint visual_solution_label) noexcept
		{
			return 2u
				* (kPrpResponseVisualLabelCount * contour_idx
					+ (visual_solution_label & kPrpResponseVisualLabelMask))
				+ (mesh_idx & 1u);
		}

		[[nodiscard]] constexpr uint prp_unpack_response_visual_region(const uint packed_region_idx) noexcept
		{
			return packed_region_idx >> 1u;
		}

		[[nodiscard]] constexpr uint prp_unpack_response_semantic_region(const uint packed_region_idx) noexcept
		{
			const uint mesh_idx = packed_region_idx & 1u;
			const uint contour_idx = (packed_region_idx >> 1u) / kPrpResponseVisualLabelCount;
			return 2u * contour_idx + mesh_idx;
		}

		static_assert(prp_unpack_response_visual_region(
						  prp_pack_response_region_label(3u, 1u, 5u))
			== 3u * kPrpResponseVisualLabelCount + 5u);
		static_assert(prp_unpack_response_semantic_region(
						  prp_pack_response_region_label(3u, 1u, 5u))
			== 2u * 3u + 1u);
	} // namespace

	void get_intersection_curve_data(std::vector<std::array<float, 3>>& output_positions,
		std::vector<std::array<uint, 2>>&								output_edges,
		std::vector<std::array<float, 3>>&								output_colors,
		UntanglingData<std::vector>*									host_untangling_data,
		CollisionData<std::vector>*										host_collision_data,
		SimulationData<std::vector>*									host_sim_data)
	{
		using Edge = std::array<uint, 2>;
		using Vec3 = std::array<float, 3>;
		using Pair = lcs::CollisionPair::CollisionPairTemplate;

		const uint num_ef = host_collision_data->narrow_phase_collision_count[1];
		output_positions.clear();
		output_edges.clear();
		output_colors.clear();

		auto vis_target_point = [&](const Pair& info, const uint label)
		{
			const auto	 ids = info.get_indices();
			const float	 area = info.get_area();
			const float3 positions[4] = {
				host_sim_data->sa_x[ids.x],
				host_sim_data->sa_x[ids.y],
				host_sim_data->sa_x[ids.z],
				host_sim_data->sa_x[ids.w],
			};
			const float thickness = host_sim_data->sa_contact_active_verts_offset[ids[0]]
				+ host_sim_data->sa_contact_active_verts_offset[ids[2]];
			const float4 weights = luisa::abs(info.get_weight());
			const auto	 type = info.get_collision_type();
			float3		 p1;
			float3		 p2;
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
			float dist = length(p2 - p1);

			output_positions.push_back(Vec3{ p1.x, p1.y, p1.z });
			output_positions.push_back(Vec3{ p2.x, p2.y, p2.z });
			output_edges.push_back(Edge{ static_cast<uint>(output_positions.size() - 2), static_cast<uint>(output_positions.size() - 1) });

			auto color = dist < thickness + 1e-6f ? Vec3{ 0.0f, 0.0f, 0.0f } : lcs::ColorUtils::highlight10(label + 1);
			output_colors.push_back({ color[0], color[1], color[2] });
		};

		const auto collision_list = std::span(host_untangling_data->target_point_template_pairs);
		for (uint i = 0; i < collision_list.size(); i++)
		{
			const auto& info = collision_list[i];
			const uint	packed_region_idx = host_untangling_data->target_point_template_pairs_indices[i][2];
			// A visual region is a unique (contour, selected Solution) pair. The source-mesh bit is intentionally ignored for color.
			const uint label = prp_unpack_response_visual_region(packed_region_idx);
			vis_target_point(info, label);
		}

		const uint prefix = output_positions.size();

		// Intersection Contour
		if (num_ef > 0)
		{
			// output_positions.resize(num_ef);
			for (uint pair_idx = 0; pair_idx < num_ef; pair_idx++)
			{
				const float3 intersection_point = host_untangling_data->ef_pair_pos_3D[pair_idx];
				output_positions.push_back(Vec3{ intersection_point.x, intersection_point.y, intersection_point.z });
			}
			// Intersection contour
			const bool contour_aware = host_untangling_data->ef_pair_contour_index.empty() ? false : true;
			for (uint pair_idx = 0; pair_idx < num_ef; pair_idx++)
			{
				const auto pair = host_collision_data->narrow_phase_list_ef[pair_idx];
				const bool is_loop_vertex = pair.get_is_loop_vertex();
				const uint contour_idx = contour_aware ? host_untangling_data->ef_pair_contour_index[pair_idx] : 0;

				if (get_scene_params().should_contour_skip(contour_idx))
					continue;

				const auto& pair_adj_pairs = host_untangling_data->ef_pair_adj_pairs_ext[pair_idx];
				if (pair_adj_pairs.empty())
				{
					output_edges.push_back(Edge{ prefix + pair_idx, prefix + pair_idx });
					Vec3 color = contour_aware ? ColorUtils::highlight10(contour_idx) : Vec3{ 0.8f, 0.8f, 0.8f };
					output_colors.push_back(color);
					LUISA_INFO("Pair idx {}: Contour idx = {}, is_loop_vertex = {}, EF = {} / {}, no adjacent pair", pair_idx, contour_idx, is_loop_vertex,
						pair.get_edge(), pair.get_face());
					continue;
				}
				// LUISA_INFO("Pair idx {}: Contour idx = {}, is_loop_vertex = {}, adj_count = {}", pair_idx, contour_idx, is_loop_vertex, pair_adj_pairs.size());
				for (const auto& adj_pair_idx : pair_adj_pairs)
				{
					if (adj_pair_idx >= num_ef)
					{
						LUISA_ERROR("Adj pair idx {} out of range {}", adj_pair_idx, num_ef);
					}
					output_edges.push_back(Edge{ prefix + pair_idx, prefix + adj_pair_idx });
					// Vec3 color = is_loop_vertex ? Vec3{0.0, 0.0, 1.0} : ColorUtils::highlight10(contour_idx);
					Vec3 color = contour_aware ? ColorUtils::highlight10(contour_idx) : Vec3{ 0.8f, 0.8f, 0.8f };
					output_colors.push_back(color);
				}
			}
		}
	}

	// Build intersection contours from EF pairs and extended adjacency; populate contour indices, mesh indices, and flip weights.
	void intersection_contour_construction_from_ext_adjacent(UntanglingData<std::vector>* host_untangling_data,
		CollisionData<std::vector>*														  host_collision_data)
	{
		const uint	num_pairs = host_collision_data->narrow_phase_collision_count[1];
		const auto& ef_list = host_collision_data->narrow_phase_list_ef;
		const auto& ef_indices = host_collision_data->narrow_phase_list_ef_indices;

		auto& contours = host_untangling_data->intersection_contours;
		auto& ef_pair_contour_index = host_untangling_data->ef_pair_contour_index;
		auto& ef_pair_mesh_index = host_untangling_data->ef_pair_mesh_index;

		contours.clear();
		ef_pair_contour_index.clear();
		ef_pair_mesh_index.clear();

		{
			std::vector<std::vector<uint>> ef_pair_adj_pairs_mesh_flip_weight(num_pairs);
			for (uint pair_idx = 0; pair_idx < num_pairs; pair_idx++)
			{
				const auto& adj_pairs = host_untangling_data->ef_pair_adj_pairs_ext[pair_idx];
				auto&		weights = ef_pair_adj_pairs_mesh_flip_weight[pair_idx];
				weights.resize(adj_pairs.size(), 0);
				const auto	pair = ef_list[pair_idx];
				const uint3 curr_face = pair.get_face();
				for (uint adj_idx = 0; adj_idx < adj_pairs.size(); adj_idx++)
				{
					const uint	adj_pair_idx = adj_pairs[adj_idx];
					const auto	adj_pair = ef_list[adj_pair_idx];
					const uint3 adj_face = adj_pair.get_face();
					const bool	same_mesh = luisa::all(curr_face == adj_face);
					weights[adj_idx] = same_mesh ? 0u : 1u;
				}
			}
			host_untangling_data->ef_pair_adj_pairs_mesh_flip_weight = std::move(ef_pair_adj_pairs_mesh_flip_weight);
		}

		{
			contours.clear();

			contours = contour_flood_filling_template(host_untangling_data->ef_pair_adj_pairs_ext,
				host_untangling_data->ef_pair_adj_pairs_mesh_flip_weight,
				host_untangling_data->ef_pair_mesh_index);
		}

		const uint num_contours = contours.size();

		ef_pair_contour_index.resize(num_pairs, -1u);
		for (uint contour_idx = 0; contour_idx < num_contours; contour_idx++)
		{
			const auto& contour = contours[contour_idx];
			for (const uint pair_idx : contour)
			{
				ef_pair_contour_index[pair_idx] = contour_idx;
			}
		}
	}

	ContourType classify_contour_type(const std::vector<CollisionPair::EfPair>& ef_list,
		const std::vector<uint>&												ef_pair_mesh_index,
		const std::vector<std::vector<uint>>&									ef_pair_adj_pairs_ext,
		const std::vector<luisa::ubyte>&										pair_is_boundary,
		std::vector<luisa::ubyte>&												contour_boundary_pairs_is_boundary,
		const std::vector<uint>&												contour,
		const uint																contour_idx)
	{
		std::vector<uint> contour_boundary_pairs;
		std::vector<uint> contour_loop_pairs;
		contour_boundary_pairs.reserve(contour.size());
		contour_boundary_pairs_is_boundary.reserve(contour.size());
		contour_loop_pairs.reserve(contour.size());
		bool is_closed = true;
		for (const uint pair_idx : contour)
		{
			if (ef_list[pair_idx].get_is_loop_vertex())
			{
				contour_loop_pairs.push_back(pair_idx);
				contour_boundary_pairs_is_boundary.push_back(pair_is_boundary[pair_idx] != 0);
			}
			if (pair_is_boundary[pair_idx] != 0)
			{
				contour_boundary_pairs.push_back(pair_idx);
				is_closed = false;
			}
			if (!ef_list[pair_idx].get_is_loop_vertex()
				&& ef_pair_adj_pairs_ext[pair_idx].size() == 1)
			{
				is_closed = false;
			}
		}

		if (contour_boundary_pairs.empty())
		{
			if (contour_loop_pairs.empty() && is_closed)
				return ContourType::Closed;
			if (contour_loop_pairs.size() == 1 && is_closed)
				return ContourType::Eight;
			if (contour_loop_pairs.size() == 2 && is_closed)
				return ContourType::LL;
			return ContourType::Undifined;
		}

		if (contour_loop_pairs.empty())
		{
			if (contour_boundary_pairs.size() == 2)
			{
				const uint front = contour_boundary_pairs.front();
				const uint back = contour_boundary_pairs.back();
				return ef_pair_mesh_index[front] == ef_pair_mesh_index[back] ? ContourType::BBII : ContourType::BIBI;
			}
			return ContourType::Undifined;
		}

		if (contour_loop_pairs.size() == 1)
		{
			if (contour_boundary_pairs.size() == 1)
				return ContourType::BLI;
			if (contour_boundary_pairs.size() == 2)
				return ContourType::Cross;
			return ContourType::Undifined;
		}
		return ContourType::Undifined;
	};

	// The region should connected to the invalid boundary verts, and have the corrorsponding mesh lable
	bool is_hit_adj_contour(const RayCasting::HitInfo& hit,
		const std::vector<std::vector<uint>>&		   vert_in_boundary_flag,
		const uint									   contour_idx)
	{
		const auto& src_verts = hit.src_verts;
		const auto& dst_verts = hit.dst_verts;
		const uint	src_mesh_idx = hit.src_mesh_idx;
		const uint	dst_mesh_idx = src_mesh_idx ^ 1u;

		auto has_boundary_flag = [&](const uint vid, const uint mesh_idx)
		{
			// Phase D-2: sorted boundary flags make binary_search reduce adjacency probes from O(deg_v) to O(log deg_v).
			const auto& flags = vert_in_boundary_flag[vid];
			return std::binary_search(flags.begin(), flags.end(), 2 * contour_idx + mesh_idx);
		};

		if (hit.is_vf())
		{
			return has_boundary_flag(src_verts[0], src_mesh_idx)
				&& has_boundary_flag(dst_verts[0], dst_mesh_idx)
				&& has_boundary_flag(dst_verts[1], dst_mesh_idx)
				&& has_boundary_flag(dst_verts[2], dst_mesh_idx);
		}
		if (hit.is_ee())
		{
			return has_boundary_flag(src_verts[0], src_mesh_idx)
				&& has_boundary_flag(src_verts[1], src_mesh_idx)
				&& has_boundary_flag(dst_verts[0], dst_mesh_idx)
				&& has_boundary_flag(dst_verts[1], dst_mesh_idx);
		}
		return has_boundary_flag(src_verts[0], src_mesh_idx)
			&& has_boundary_flag(src_verts[1], src_mesh_idx)
			&& has_boundary_flag(src_verts[2], src_mesh_idx)
			&& has_boundary_flag(dst_verts[0], dst_mesh_idx);
	}
	std::vector<uint> get_adj_contours(const RayCasting::HitInfo& hit,
		const std::vector<std::vector<uint>>&					  vert_in_boundary_flag,
		const uint												  contour_idx)
	{
		// Get source and target vertices
		auto source_verts = hit.get_source_verts();
		auto target_verts = hit.get_target_verts();

		bool source_in_boundary = false;
		bool target_in_boundary = false;

		std::vector<uint> source_contains_flags = vert_in_boundary_flag[source_verts.front()];
		std::vector<uint> target_contains_flags = vert_in_boundary_flag[target_verts.front()];
		for (const uint vid : source_verts)
		{
			source_contains_flags = SetOperation::get_elements_in_both_A_and_B(vert_in_boundary_flag[vid], source_contains_flags);
		}
		for (const uint vid : target_verts)
		{
			target_contains_flags = SetOperation::get_elements_in_both_A_and_B(vert_in_boundary_flag[vid], target_contains_flags);
		}
		source_contains_flags = SetOperation::A_remove_B(source_contains_flags, { 2 * contour_idx + 0, 2 * contour_idx + 1 });
		target_contains_flags = SetOperation::A_remove_B(target_contains_flags, { 2 * contour_idx + 0, 2 * contour_idx + 1 });

		std::set<uint> other_contours;
		for (const uint src_flag : source_contains_flags)
		{
			for (const uint dst_flag : target_contains_flags)
			{
				if (src_flag == (dst_flag ^ 1))
				{
					const uint adj_contour_idx = src_flag / 2;
					other_contours.insert(adj_contour_idx);
				}
			}
		}

		return std::vector<uint>(other_contours.begin(), other_contours.end());
	}

	void prp_capture_pre_finalize_contour_merge_data(
		PRPContourMergeAdjacency&				 merge_adjacency,
		std::vector<float>&						 pre_finalize_penetration_area,
		const std::vector<PRPMinContourHitInfo>& selected_hit_infos,
		const std::vector<uint>&				 target_contours,
		const std::vector<std::vector<uint>>&	 vert_in_boundary_flag,
		const std::vector<std::array<uint, 2>>&	 contour_object_ids)
	{
		const uint num_contours = static_cast<uint>(selected_hit_infos.size());
		if (merge_adjacency.size() != num_contours
			|| pre_finalize_penetration_area.size() != num_contours
			|| contour_object_ids.size() != num_contours)
			LUISA_ERROR(
				"Pre-finalize contour merge storage mismatch: selected {}, adjacency {}, area {}, object ids {}.",
				num_contours,
				merge_adjacency.size(),
				pre_finalize_penetration_area.size(),
				contour_object_ids.size());

		auto same_object_pair = [&](const uint a, const uint b) noexcept
		{
			const auto& oa = contour_object_ids[a];
			const auto& ob = contour_object_ids[b];
			const uint	sentinel = std::numeric_limits<uint>::max();
			if (oa[0] == sentinel || oa[1] == sentinel || ob[0] == sentinel || ob[1] == sentinel)
				return false;
			return std::min(oa[0], oa[1]) == std::min(ob[0], ob[1])
				&& std::max(oa[0], oa[1]) == std::max(ob[0], ob[1]);
		};

		for (const uint contour_idx : target_contours)
		{
			if (contour_idx >= num_contours)
				LUISA_ERROR("Pre-finalize contour merge target {} exceeds contour count {}.", contour_idx, num_contours);

			const auto& info = selected_hit_infos[contour_idx];
			pre_finalize_penetration_area[contour_idx] = info.penetration_area;
			std::set<uint> adjacent_contours;
			for (const auto& hit : info.hit_list)
			{
				for (const uint candidate : get_adj_contours(hit, vert_in_boundary_flag, contour_idx))
				{
					if (candidate >= num_contours)
						LUISA_ERROR(
							"Contour {} pre-finalize Hit references adjacent contour {} outside {} contours.",
							contour_idx,
							candidate,
							num_contours);
					if (candidate != contour_idx && same_object_pair(contour_idx, candidate))
						adjacent_contours.insert(candidate);
				}
			}
			merge_adjacency[contour_idx].assign(adjacent_contours.begin(), adjacent_contours.end());
			if (!merge_adjacency[contour_idx].empty())
				LUISA_WARNING(
					"  => Contour {}: Found adjacent contours from pre-finalize ray-cast hits: {}",
					contour_idx,
					merge_adjacency[contour_idx]);
		}
	}

	std::vector<uint> prp_select_pre_finalize_contours_to_suppress(
		const PRPContourMergeAdjacency& merge_adjacency,
		const std::vector<float>&		pre_finalize_penetration_area,
		const std::vector<uint>&		active_contours)
	{
		const uint num_contours = static_cast<uint>(merge_adjacency.size());
		if (pre_finalize_penetration_area.size() != num_contours)
			LUISA_ERROR(
				"Pre-finalize contour merge area count {} does not match adjacency count {}.",
				pre_finalize_penetration_area.size(),
				num_contours);

		std::vector<luisa::ubyte> active(num_contours, 0u);
		for (const uint contour_idx : active_contours)
		{
			if (contour_idx >= num_contours)
				LUISA_ERROR("Active contour {} exceeds pre-finalize merge graph size {}.", contour_idx, num_contours);
			if (active[contour_idx] != 0u)
				LUISA_ERROR("Active contour {} is duplicated in pre-finalize merge selection.", contour_idx);
			active[contour_idx] = 1u;
		}

		std::vector<uint> sorted_conflicting_contours;
		for (const uint contour_idx : active_contours)
		{
			if (merge_adjacency[contour_idx].empty())
				continue;
			if (!std::isfinite(pre_finalize_penetration_area[contour_idx]))
				LUISA_ERROR(
					"Contour {} has non-finite pre-finalize penetration area {} during merge selection.",
					contour_idx,
					pre_finalize_penetration_area[contour_idx]);
			sorted_conflicting_contours.push_back(contour_idx);
		}
		std::sort(sorted_conflicting_contours.begin(), sorted_conflicting_contours.end(),
			[&](const uint a, const uint b)
			{
				const float area_a = pre_finalize_penetration_area[a];
				const float area_b = pre_finalize_penetration_area[b];
				if (area_a != area_b)
					return area_a > area_b;
				return a < b;
			});

		std::vector<luisa::ubyte> suppressed(num_contours, 0u);
		for (const uint owner : sorted_conflicting_contours)
		{
			if (suppressed[owner] != 0u)
				continue;
			for (const uint adjacent : merge_adjacency[owner])
			{
				if (adjacent >= num_contours)
					LUISA_ERROR(
						"Contour {} merge adjacency {} exceeds graph size {}.",
						owner,
						adjacent,
						num_contours);
				if (adjacent != owner && active[adjacent] != 0u)
					suppressed[adjacent] = 1u;
			}
		}

		std::vector<uint> result;
		for (uint contour_idx = 0u; contour_idx < num_contours; ++contour_idx)
		{
			if (suppressed[contour_idx] != 0u)
				result.push_back(contour_idx);
		}
		return result;
	}

	void prp_clear_suppressed_contour_response(
		PRPMinContourHitInfo& hit_info,
		const uint			  contour_idx)
	{
		PRPMinContourHitInfo cleared_info{};
		cleared_info.contour_idx = contour_idx;
		hit_info = std::move(cleared_info);
	}

	namespace
	{

		inline bool prp_is_finite_float3(const float3& v) noexcept
		{
			return std::isfinite(v.x) && std::isfinite(v.y) && std::isfinite(v.z);
		}

		inline bool prp_is_valid_direction(const float3& v) noexcept
		{
			return prp_is_finite_float3(v) && length_squared_vec(v) > 1e-12f;
		}

		inline float3 prp_safe_normalize(const float3& v, const float3& fallback) noexcept
		{
			if (prp_is_valid_direction(v))
				return luisa::normalize(v);
			if (prp_is_valid_direction(fallback))
				return luisa::normalize(fallback);
			return luisa::make_float3(0.0f, 1.0f, 0.0f);
		}

	} // namespace

	void calculate_hop_dist_and_geom_dist_in_mesh(
		std::array<std::vector<uint>, 2>&		vert_to_boundary_hop_dist,
		std::array<std::vector<float>, 2>&		vert_to_boundary_geom_dist,
		std::array<uint, 2>&					mesh_max_hop_dist,
		std::array<float, 2>&					contour_min_geom_dist_to_another_contour,
		std::array<float, 2>&					contour_max_geom_dist_to_another_contour,
		const MeshData<std::vector>*			host_mesh_data,
		const std::array<std::vector<uint>, 2>& boundary_verts,
		const uint								contour_idx,
		const std::array<uint, 2>&				object_ids)
	{
		const uint	hop_inf = std::numeric_limits<uint>::max();
		const float geo_inf = std::numeric_limits<float>::max();
		mesh_max_hop_dist[0] = 0u;
		mesh_max_hop_dist[1] = 0u;
		contour_min_geom_dist_to_another_contour[0] = std::numeric_limits<float>::max();
		contour_min_geom_dist_to_another_contour[1] = std::numeric_limits<float>::max();
		contour_max_geom_dist_to_another_contour[0] = 0.0f;
		contour_max_geom_dist_to_another_contour[1] = 0.0f;

		const uint num_verts = host_mesh_data->num_verts;
		for (uint mesh_idx = 0u; mesh_idx < 2u; ++mesh_idx)
		{
			const uint object_id = object_ids[mesh_idx];
			const uint prefix_vid = host_mesh_data->prefix_num_verts[object_id];
			const uint suffix_vid = host_mesh_data->prefix_num_verts[object_id + 1u];
			vert_to_boundary_hop_dist[mesh_idx].assign(num_verts, hop_inf);
			vert_to_boundary_geom_dist[mesh_idx].assign(num_verts, geo_inf);

			using HopNode = std::pair<uint, uint>;
			using GeoNode = std::pair<float, uint>;
			std::priority_queue<HopNode, std::vector<HopNode>, std::greater<HopNode>> hop_pq;
			std::priority_queue<GeoNode, std::vector<GeoNode>, std::greater<GeoNode>> geo_pq;

			for (const uint vid : boundary_verts[mesh_idx])
			{
				if (vid < prefix_vid || vid >= suffix_vid)
					LUISA_ERROR("Boundary vertex {} of mesh {} has non-zero initial hop distance", vid, mesh_idx);
				vert_to_boundary_hop_dist[mesh_idx][vid] = 0u;
				vert_to_boundary_geom_dist[mesh_idx][vid] = 0.0f;
				hop_pq.emplace(0u, vid);
				geo_pq.emplace(0.0f, vid);
			}
			while (!hop_pq.empty())
			{
				auto [hop_dist, vid] = hop_pq.top();
				hop_pq.pop();
				if (hop_dist > vert_to_boundary_hop_dist[mesh_idx][vid])
					continue;
				for (const uint adj_vid : host_mesh_data->vert_adj_verts[vid])
				{
					if (adj_vid < prefix_vid || adj_vid >= suffix_vid)
						continue;
					const uint next_hop = hop_dist + 1u;
					if (next_hop < vert_to_boundary_hop_dist[mesh_idx][adj_vid])
					{
						vert_to_boundary_hop_dist[mesh_idx][adj_vid] = next_hop;
						hop_pq.emplace(next_hop, adj_vid);
					}
				}
			}
			while (!geo_pq.empty())
			{
				auto [curr_dist, vid] = geo_pq.top();
				geo_pq.pop();
				if (curr_dist > vert_to_boundary_geom_dist[mesh_idx][vid])
					continue;
				for (const uint adj_vid : host_mesh_data->vert_adj_verts[vid])
				{
					if (adj_vid < prefix_vid || adj_vid >= suffix_vid)
						continue;
					const float edge_len = luisa::distance(host_mesh_data->sa_rest_x[vid], host_mesh_data->sa_rest_x[adj_vid]);
					const float next_dist = curr_dist + edge_len;
					if (next_dist < vert_to_boundary_geom_dist[mesh_idx][adj_vid])
					{
						vert_to_boundary_geom_dist[mesh_idx][adj_vid] = next_dist;
						geo_pq.emplace(next_dist, adj_vid);
					}
				}
			}
		}
		for (uint mesh_idx = 0u; mesh_idx < 2u; ++mesh_idx)
		{
			const uint object_id = object_ids[mesh_idx];
			const uint prefix_vid = host_mesh_data->prefix_num_verts[object_id];
			const uint suffix_vid = host_mesh_data->prefix_num_verts[object_id + 1u];
			for (uint vid = prefix_vid; vid < suffix_vid; ++vid)
			{
				const uint d = vert_to_boundary_hop_dist[mesh_idx][vid];
				if (d != hop_inf)
					mesh_max_hop_dist[mesh_idx] = std::max(mesh_max_hop_dist[mesh_idx], d);
			}
			for (uint vid : boundary_verts[mesh_idx])
			{
				const float g = vert_to_boundary_geom_dist[1 - mesh_idx][vid];
				if (g != geo_inf)
				{
					contour_max_geom_dist_to_another_contour[mesh_idx] = std::max(contour_max_geom_dist_to_another_contour[mesh_idx], g);
					contour_min_geom_dist_to_another_contour[mesh_idx] = std::min(contour_min_geom_dist_to_another_contour[mesh_idx], g);
				}
			}
			if (contour_min_geom_dist_to_another_contour[mesh_idx] == geo_inf)
				contour_min_geom_dist_to_another_contour[mesh_idx] = 0.0f;
		}
	}

	template <typename Func>
	inline void visit_prp_hit_source_vertices(const RayCasting::HitInfo& hit, Func&& visit)
	{
		if (hit.is_vf())
		{
			visit(hit.src_verts[0]);
		}
		else if (hit.is_ee())
		{
			visit(hit.src_verts[0]);
			visit(hit.src_verts[1]);
		}
		else if (hit.is_fv())
		{
			visit(hit.src_verts[0]);
			visit(hit.src_verts[1]);
			visit(hit.src_verts[2]);
		}
	}

	template <typename Func>
	inline void visit_prp_hit_target_vertices(const RayCasting::HitInfo& hit, Func&& visit)
	{
		if (hit.is_vf())
		{
			visit(hit.dst_verts[0]);
			visit(hit.dst_verts[1]);
			visit(hit.dst_verts[2]);
		}
		else if (hit.is_ee())
		{
			visit(hit.dst_verts[0]);
			visit(hit.dst_verts[1]);
		}
		else if (hit.is_fv())
		{
			visit(hit.dst_verts[0]);
		}
	}

	inline void update_prp_hit_source_frontier_hop(const PrpStandardContourRuntime& runtime, const uint src_vid, MinMaxHopInfo& local_min_max_hop)
	{
		const uint hop_to_curr_contour = runtime.boundary_hop_distance(0u, src_vid);
		const uint hop_to_oppo_contour = runtime.boundary_hop_distance(1u, src_vid);
		if (hop_to_curr_contour != std::numeric_limits<uint>::max())
		{
			local_min_max_hop.min_frontier_hop_src() = std::min(local_min_max_hop.min_frontier_hop_src(), hop_to_curr_contour);
			local_min_max_hop.max_frontier_hop_src() = std::max(local_min_max_hop.max_frontier_hop_src(), hop_to_curr_contour);
		}
		if (hop_to_oppo_contour != std::numeric_limits<uint>::max())
		{
			local_min_max_hop.min_hop_to_dst_boundary() = std::min(local_min_max_hop.min_hop_to_dst_boundary(), hop_to_oppo_contour);
			local_min_max_hop.max_hop_to_dst_boundary() = std::max(local_min_max_hop.max_hop_to_dst_boundary(), hop_to_oppo_contour);
		}
	}

	inline void update_prp_hit_target_frontier_hop(const PrpStandardContourRuntime& runtime, const uint dst_vid, MinMaxHopInfo& local_min_max_hop)
	{
		const uint hop_to_curr_contour = runtime.boundary_hop_distance(1u, dst_vid);
		const uint hop_to_oppo_contour = runtime.boundary_hop_distance(0u, dst_vid);
		if (hop_to_curr_contour != std::numeric_limits<uint>::max())
		{
			local_min_max_hop.min_frontier_hop_dst() = std::min(local_min_max_hop.min_frontier_hop_dst(), hop_to_curr_contour);
			local_min_max_hop.max_frontier_hop_dst() = std::max(local_min_max_hop.max_frontier_hop_dst(), hop_to_curr_contour);
		}
		if (hop_to_oppo_contour != std::numeric_limits<uint>::max())
		{
			local_min_max_hop.min_hop_to_src_boundary() = std::min(local_min_max_hop.min_hop_to_src_boundary(), hop_to_oppo_contour);
			local_min_max_hop.max_hop_to_src_boundary() = std::max(local_min_max_hop.max_hop_to_src_boundary(), hop_to_oppo_contour);
		}
	}

	MinMaxHopInfo compute_hit_min_max_frontier_hop(const PrpStandardContourRuntime& runtime, const std::vector<RayCasting::HitInfo>& hits)
	{
		PRP_PROFILE_SCOPE("compute_standard_hit_max_frontier_hop");
		MinMaxHopInfo min_max_hop;
		// Legacy serial frontier-hop reduction retained for reference.
		min_max_hop = CpuParallel::parallel_for_and_reduce(
			0u,
			static_cast<uint>(hits.size()),
			[&](const uint hit_idx) -> MinMaxHopInfo
			{
				const auto&	  hit = hits[hit_idx];
				MinMaxHopInfo local_min_max_hop;
				visit_prp_hit_source_vertices(hit,
					[&](const uint src_vid)
					{ update_prp_hit_source_frontier_hop(runtime, src_vid, local_min_max_hop); });
				visit_prp_hit_target_vertices(hit,
					[&](const uint dst_vid)
					{ update_prp_hit_target_frontier_hop(runtime, dst_vid, local_min_max_hop); });
				return local_min_max_hop;
			},
			[](const MinMaxHopInfo& a, const MinMaxHopInfo& b) -> MinMaxHopInfo
			{
				MinMaxHopInfo result;
				result.min_frontier_hop_src() = std::min(a.min_frontier_hop_src(), b.min_frontier_hop_src());
				result.min_frontier_hop_dst() = std::min(a.min_frontier_hop_dst(), b.min_frontier_hop_dst());
				result.min_hop_to_dst_boundary() = std::min(a.min_hop_to_dst_boundary(), b.min_hop_to_dst_boundary());
				result.min_hop_to_src_boundary() = std::min(a.min_hop_to_src_boundary(), b.min_hop_to_src_boundary());
				result.max_frontier_hop_src() = std::max(a.max_frontier_hop_src(), b.max_frontier_hop_src());
				result.max_frontier_hop_dst() = std::max(a.max_frontier_hop_dst(), b.max_frontier_hop_dst());
				result.max_hop_to_dst_boundary() = std::max(a.max_hop_to_dst_boundary(), b.max_hop_to_dst_boundary());
				result.max_hop_to_src_boundary() = std::max(a.max_hop_to_src_boundary(), b.max_hop_to_src_boundary());
				return result;
			},
			MinMaxHopInfo());
		// Legacy frontier-hop diagnostic logging retained for reference.
		return min_max_hop;
	};
	PRPContourSideIdentity prp_get_contour_side_identity(
		const CollisionPair::EfPair& first_pair,
		const uint					 edge_side,
		const MeshData<std::vector>* host_mesh_data)
	{
		if (edge_side >= 2u)
			LUISA_ERROR("EF edge side {} is invalid while resolving contour identity.", edge_side);
		if (host_mesh_data == nullptr)
			LUISA_ERROR("Mesh data is null while resolving contour identity.");
		if (host_mesh_data->sa_vert_mesh_id.size() != host_mesh_data->num_verts
			|| host_mesh_data->vert_topology_component_id.size() != host_mesh_data->num_verts)
			LUISA_ERROR(
				"Contour identity requires {} object and topology labels, got {} and {}.",
				host_mesh_data->num_verts,
				host_mesh_data->sa_vert_mesh_id.size(),
				host_mesh_data->vert_topology_component_id.size());

		PRPContourSideIdentity identity;
		auto				   assign_primitive_identity = [&](const uint side, const std::initializer_list<uint> vertices, const char* primitive_name)
		{
			const uint first_vid = *vertices.begin();
			if (first_vid >= host_mesh_data->num_verts)
				LUISA_ERROR("{} vertex {} exceeds mesh vertex count {}.", primitive_name, first_vid, host_mesh_data->num_verts);
			const uint object_id = host_mesh_data->sa_vert_mesh_id[first_vid];
			const uint component_id = host_mesh_data->vert_topology_component_id[first_vid];
			if (component_id == std::numeric_limits<uint>::max())
				LUISA_ERROR("{} vertex {} has no topology component label.", primitive_name, first_vid);
			for (const uint vid : vertices)
			{
				if (vid >= host_mesh_data->num_verts)
					LUISA_ERROR("{} vertex {} exceeds mesh vertex count {}.", primitive_name, vid, host_mesh_data->num_verts);
				if (host_mesh_data->sa_vert_mesh_id[vid] != object_id
					|| host_mesh_data->vert_topology_component_id[vid] != component_id)
					LUISA_ERROR(
						"{} vertices do not share one registered object and topology component.",
						primitive_name);
			}
			identity.object_ids[side] = object_id;
			identity.topology_component_ids[side] = component_id;
		};

		const uint2 edge = first_pair.get_edge();
		const uint3 face = first_pair.get_face();
		assign_primitive_identity(edge_side, { edge.x, edge.y }, "EF edge");
		assign_primitive_identity(edge_side ^ 1u, { face.x, face.y, face.z }, "EF face");
		return identity;
	}

	void export_runtime_combo_debug(const PrpStandardContourRuntime& runtime, SceneParams::PRPDebugInfo& debug_sink)
	{
		if (!get_scene_params().prp_debug)
			return;
		const std::string contour_key = "prp.c" + std::to_string(runtime.contour_idx) + ".";
		debug_sink.uint_stats[contour_key + "combo_count"] = static_cast<uint>(runtime.contour_dirs.size());
		const std::string intrinsic_key = contour_key + "intrinsic_contour_side_candidates.";
		debug_sink.bool_stats[intrinsic_key + "use_rest_geodesic_distance"] = get_scene_params().PRP_use_rest_geodesic_distance_for_intrinsic_candidates;
		debug_sink.bool_stats[intrinsic_key + "use_blended_coordinates"] = get_scene_params().PRP_use_blended_intrinsic_coordinates;
		debug_sink.float_stats[intrinsic_key + "blend_weight"] = get_scene_params().PRP_intrinsic_blend_weight;
		debug_sink.bool_stats[intrinsic_key + "enabled"] = get_scene_params().PRP_use_intrinsic_contour_side_candidates;
		debug_sink.bool_stats[intrinsic_key + "applied"] = runtime.intrinsic_contour_side_candidate_restriction_applied;
		debug_sink.uint_stats[intrinsic_key + "enabled"] = get_scene_params().PRP_use_intrinsic_contour_side_candidates ? 1u : 0u;
		debug_sink.uint_stats[intrinsic_key + "applied"] = runtime.intrinsic_contour_side_candidate_restriction_applied ? 1u : 0u;
		const bool use_deformed_boundary_distance = get_scene_params().PRP_use_deformed_boundary_distance_for_intrinsic_candidates;
		debug_sink.bool_stats[intrinsic_key + "use_deformed_boundary_distance"] = use_deformed_boundary_distance;
		debug_sink.uint_stats[intrinsic_key + "use_deformed_boundary_distance"] = use_deformed_boundary_distance ? 1u : 0u;
		debug_sink.float_stats[intrinsic_key + "min_boundary_separation"] = std::min(runtime.contour_min_geom_dist_to_another_contour[0], runtime.contour_min_geom_dist_to_another_contour[1]);
		debug_sink.float_stats[intrinsic_key + "max_boundary_separation"] = std::max(runtime.contour_max_geom_dist_to_another_contour[0], runtime.contour_max_geom_dist_to_another_contour[1]);
		for (uint role = 0u; role < 2u; ++role)
		{
			const std::string intrinsic_role_key = intrinsic_key + "role" + std::to_string(role) + ".";
			debug_sink.uint_stats[intrinsic_role_key + "vertex_count_before"] = runtime.intrinsic_candidate_vertex_count_before[role];
			debug_sink.uint_stats[intrinsic_role_key + "edge_count_before"] = runtime.intrinsic_candidate_edge_count_before[role];
			debug_sink.uint_stats[intrinsic_role_key + "face_count_before"] = runtime.intrinsic_candidate_face_count_before[role];
			debug_sink.uint_stats[intrinsic_role_key + "vertex_count_after"] = static_cast<uint>(runtime.partial_candidate_verts[role].size());
			debug_sink.uint_stats[intrinsic_role_key + "edge_count_after"] = static_cast<uint>(runtime.partial_candidate_edges[role].size());
			debug_sink.uint_stats[intrinsic_role_key + "face_count_after"] = static_cast<uint>(runtime.partial_candidate_faces[role].size());
			const uint deformed_query_count = static_cast<uint>(runtime.deformed_boundary_distance_cache[role].size());
			const uint deformed_shortened_count = static_cast<uint>(runtime.deformed_boundary_distance_shortened_vertices[role].size());
			debug_sink.uint_stats[intrinsic_role_key + "deformed_distance_query_count"] = deformed_query_count;
			debug_sink.uint_stats[intrinsic_role_key + "deformed_distance_shortened_count"] = deformed_shortened_count;
			debug_sink.float_stats[intrinsic_role_key + "deformed_distance_shortened_ratio"] =
				deformed_query_count > 0u
				? static_cast<float>(deformed_shortened_count) / static_cast<float>(deformed_query_count)
				: 0.0f;
			debug_sink.float_stats[intrinsic_role_key + "deformed_distance_min"] = deformed_query_count > 0u ? runtime.deformed_boundary_distance_min[role] : 0.0f;
			debug_sink.float_stats[intrinsic_role_key + "deformed_distance_max"] = deformed_query_count > 0u ? runtime.deformed_boundary_distance_max[role] : 0.0f;
		}
		for (uint combo_idx = 0u; combo_idx < runtime.contour_dirs.size(); ++combo_idx)
		{
			const std::string combo_key = "prp.c" + std::to_string(runtime.contour_idx) + ".combo" + std::to_string(combo_idx) + ".";
			const auto&		  combo_result = runtime.combo_results[combo_idx];
			const uint		  opt_iter_count = combo_result.direction_optimization_iteration_count;
			const uint		  opt_accepted_count = combo_result.direction_optimization_accepted_count;
			const bool		  opt_selected = opt_iter_count > 0u;
			debug_sink.bool_stats[combo_key + "opt_selected"] = opt_selected;
			debug_sink.uint_stats[combo_key + "opt_selected"] = opt_selected ? 1u : 0u;
			debug_sink.uint_stats[combo_key + "opt_iter_count"] = opt_iter_count;
			debug_sink.uint_stats[combo_key + "opt_accepted_count"] = opt_accepted_count;
			const bool objective_pruned = combo_idx < runtime.combo_objective_pruned.size()
				&& runtime.combo_objective_pruned[combo_idx] != 0u;
			debug_sink.bool_stats[combo_key + "objective_pruned"] = objective_pruned;
			debug_sink.uint_stats[combo_key + "objective_pruned"] = objective_pruned ? 1u : 0u;
			prp_export_combo_stage_debug(
				runtime.contour_idx,
				combo_idx,
				"eval",
				combo_result,
				runtime.combo_eval_present[combo_idx] != 0u,
				prp_has_valid_hit_info,
				[&](const std::string& key, const bool value)
				{
					debug_sink.bool_stats[key] = value;
				},
				[&](const std::string& key, const uint value)
				{
					debug_sink.uint_stats[key] = value;
				},
				[&](const std::string& key, const float value)
				{
					debug_sink.float_stats[key] = value;
				},
				[&](const std::string& key_prefix, const float3& value)
				{
					debug_sink.float_stats[key_prefix + ".x"] = value.x;
					debug_sink.float_stats[key_prefix + ".y"] = value.y;
					debug_sink.float_stats[key_prefix + ".z"] = value.z;
				});
			prp_export_combo_stage_debug(
				runtime.contour_idx,
				combo_idx,
				"opt_eval",
				combo_result,
				opt_accepted_count > 0u,
				prp_has_valid_hit_info,
				[&](const std::string& key, const bool value)
				{
					debug_sink.bool_stats[key] = value;
				},
				[&](const std::string& key, const uint value)
				{
					debug_sink.uint_stats[key] = value;
				},
				[&](const std::string& key, const float value)
				{
					debug_sink.float_stats[key] = value;
				},
				[&](const std::string& key_prefix, const float3& value)
				{
					debug_sink.float_stats[key_prefix + ".x"] = value.x;
					debug_sink.float_stats[key_prefix + ".y"] = value.y;
					debug_sink.float_stats[key_prefix + ".z"] = value.z;
				});
		}
	};

	std::vector<PRPMinContourHitInfo>::const_iterator find_best_result_in_combo_results(
		const std::vector<PRPMinContourHitInfo>& combo_results)
	{
		const auto selection = prp_select_response(combo_results);
		return selection.has_response()
			? combo_results.begin() + selection.selected_idx
			: combo_results.end();
	}
	void select_runtime_best_results(
		std::vector<PRPMinContourHitInfo>&			  list_min_hit_info,
		std::vector<SceneParams::PRPDebugInfo>&		  contour_debug_infos,
		const std::vector<PrpStandardContourRuntime>& runtimes,
		const std::vector<uint>&					  target_contours,
		const bool									  accumulate_stats)
	{
		uint response_contour_count = 0u;
		uint no_response_contour_count = 0u;
		for (const uint runtime_idx : target_contours)
		{
			const auto&			 runtime = runtimes[runtime_idx];
			PRPMinContourHitInfo best_result;
			best_result.contour_idx = runtime_idx;

			const auto selected = find_best_result_in_combo_results(runtime.combo_results);
			if (selected == runtime.combo_results.end())
			{
				if (!runtime.no_response_terminal)
					LUISA_ERROR(
						"PRP contour {} has no selectable response without an explicit no-response terminal state.",
						runtime.contour_idx);
				best_result.no_response_terminal = true;
				no_response_contour_count++;
				list_min_hit_info[runtime_idx] = std::move(best_result);
				if (get_scene_params().prp_debug && runtime_idx < contour_debug_infos.size())
					export_runtime_combo_debug(runtime, contour_debug_infos[runtime_idx]);
				continue;
			}
			best_result = *selected;
			response_contour_count++;

			if (best_result.self_collision_flipped)
			{
				LUISA_WARNING(
					"Contour {}, Optimal combo {} is self-collision flipped; skipping response injection for this contour this iteration.",
					runtime_idx,
					best_result.combo_idx);
				PRPMinContourHitInfo skipped;
				skipped.contour_idx = runtime_idx;
				skipped.self_collision_flipped = true;
				list_min_hit_info[runtime_idx] = std::move(skipped);
				if (get_scene_params().prp_debug && runtime_idx < contour_debug_infos.size())
					export_runtime_combo_debug(runtime, contour_debug_infos[runtime_idx]);
				continue;
			}
			list_min_hit_info[runtime_idx] = best_result;
			if (get_scene_params().prp_debug && runtime_idx < contour_debug_infos.size())
				export_runtime_combo_debug(runtime, contour_debug_infos[runtime_idx]);
		}

		auto& stats = get_scene_params().prp_debug_info.uint_stats;
		auto  write_stat = [&](const char* key, const uint value)
		{
			if (accumulate_stats)
				stats[key] += value;
			else
				stats[key] = value;
		};
		write_stat("prp_response_contour_count", response_contour_count);
		write_stat("prp_no_response_contour_count", no_response_contour_count);
	}
	void evaluate_current_runtime_directions_template(
		std::vector<PrpStandardContourRuntime>&														  runtimes,
		const std::vector<uint>&																	  target_contours,
		const bool																					  print_combo_summary,
		const std::function<void(const std::vector<uint>&)>&										  build_candidates_fn,
		const std::function<std::vector<PRPMinContourHitInfo>(const std::vector<uint>&, uint, bool)>& eval_combo_fn)
	{
		if (target_contours.empty())
			return;

		uint max_combo_count = 0u;
		for (const uint contour_idx : target_contours)
		{
			auto& runtime = runtimes.at(contour_idx);
			prp_reset_sparse_runtime_eval_state(runtime);
			runtime.curr_k_src = runtime.init_k_src;
			runtime.curr_k_dst = runtime.init_k_dst;
			runtime.finished = false;
			runtime.no_response_terminal = false;
			max_combo_count = std::max(max_combo_count, static_cast<uint>(runtime.contour_dirs.size()));
			for (const uint combo_idx : runtime.active_combos)
			{
				if (combo_idx >= runtime.contour_dirs.size()
					|| combo_idx >= runtime.combo_converged.size()
					|| combo_idx >= runtime.combo_objective_pruned.size()
					|| combo_idx >= runtime.combo_results.size()
					|| combo_idx >= runtime.combo_eval_present.size())
					LUISA_ERROR("Invalid PRP combo {} for contour {}.", combo_idx, contour_idx);
				runtime.combo_converged[combo_idx] = 0u;
				runtime.combo_objective_pruned[combo_idx] = 0u;
				runtime.combo_results[combo_idx] = PRPMinContourHitInfo{};
				runtime.combo_eval_present[combo_idx] = 0u;
			}
		}

		auto grow_kring = [](const uint current, const uint diameter)
		{
			if (current >= diameter)
				return current;
			return static_cast<uint>(std::min<uint64_t>(diameter,
				std::max<uint64_t>(1u, static_cast<uint64_t>(current) * 2u)));
		};
		std::vector<uint> remaining = target_contours;
		// Even if the roles alternate growth, doubling a uint radius is finite.
		constexpr uint max_levels = 2u * std::numeric_limits<uint>::digits + 1u;
		for (uint level = 0u; !remaining.empty(); ++level)
		{
			if (level >= max_levels)
				LUISA_ERROR("PRP expansion exceeded {} levels for contours {}.", max_levels, remaining);
			build_candidates_fn(remaining);
			for (uint combo_idx = 0u; combo_idx < max_combo_count; ++combo_idx)
			{
				std::vector<uint> active;
				for (const uint contour_idx : remaining)
				{
					const auto& runtime = runtimes[contour_idx];
					if (std::binary_search(runtime.active_combos.begin(), runtime.active_combos.end(), combo_idx)
						&& runtime.combo_converged[combo_idx] == 0u
						&& runtime.combo_objective_pruned[combo_idx] == 0u)
						active.push_back(contour_idx);
				}
				if (active.empty())
					continue;
				auto results = eval_combo_fn(active, combo_idx, true);
				for (const uint contour_idx : active)
				{
					auto& runtime = runtimes[contour_idx];
					auto& info = results.at(contour_idx);
					info.contour_idx = contour_idx;
					info.combo_idx = combo_idx;
					info.expansion_level_count = level + 1u;
					info.full_mesh_evaluated = runtime.curr_k_src >= runtime.mesh_diameter_src
						&& runtime.curr_k_dst >= runtime.mesh_diameter_dst;
					const bool usable = prp_has_usable_response(info);
					const bool src_complete = info.max_frontier_hop_src < runtime.curr_k_src
						|| runtime.curr_k_src >= runtime.mesh_diameter_src;
					const bool dst_complete = info.max_frontier_hop_dst < runtime.curr_k_dst
						|| runtime.curr_k_dst >= runtime.mesh_diameter_dst;
					const bool detached = src_complete && dst_complete;
					info.response_eligible = usable && detached;
					runtime.combo_converged[combo_idx] = info.response_eligible || info.full_mesh_evaluated ? 1u : 0u;
					runtime.combo_eval_present[combo_idx] = 1u;
					if (get_scene_params().prp_debug)
					{
						auto&	   stats = get_scene_params().prp_debug_info.uint_stats;
						const auto key = fmt::format("prp.c{}.combo{}.range{}.", contour_idx, combo_idx, level);
						stats[key + "k_src"] = runtime.curr_k_src;
						stats[key + "k_dst"] = runtime.curr_k_dst;
						stats[key + "raw_hits"] = info.unculled_hit_count;
						stats[key + "kept_hits"] = info.hit_count;
						stats[key + "covered_anchors"] = info.covered_anchor_count;
						stats[key + "anchors"] = info.boundary_anchor_count;
					}
					runtime.combo_results[combo_idx] = std::move(info);
				}
			}

			std::vector<uint> next;
			for (const uint contour_idx : remaining)
			{
				auto& runtime = runtimes[contour_idx];
				bool  has_response = false;
				float best_objective = std::numeric_limits<float>::infinity();
				for (const uint ci : runtime.active_combos)
				{
					const auto& candidate = runtime.combo_results[ci];
					if (prp_has_valid_hit_info(candidate))
					{
						has_response = true;
						best_objective = std::min(best_objective, prp_response_objective(candidate));
					}
				}
				// Preserve Option C's existing objective-based direction pruning.
				if (has_response)
				{
					for (const uint ci : runtime.active_combos)
					{
						if (runtime.combo_converged[ci] == 0u
							&& runtime.combo_eval_present[ci] != 0u
							&& prp_response_objective(runtime.combo_results[ci]) >= best_objective)
							runtime.combo_objective_pruned[ci] = 1u;
					}
				}
				bool pending = false;
				bool grow_src = false;
				bool grow_dst = false;
				for (const uint ci : runtime.active_combos)
				{
					if (runtime.combo_eval_present[ci] == 0u)
						LUISA_ERROR("PRP contour {} combo {} was not evaluated.", contour_idx, ci);
					if (runtime.combo_converged[ci] != 0u || runtime.combo_objective_pruned[ci] != 0u)
						continue;
					pending = true;
					const auto& info = runtime.combo_results[ci];
					// Unusable support cannot establish detachment; search both remaining roles.
					grow_src |= runtime.curr_k_src < runtime.mesh_diameter_src
						&& (!prp_has_usable_response(info) || info.max_frontier_hop_src >= runtime.curr_k_src);
					grow_dst |= runtime.curr_k_dst < runtime.mesh_diameter_dst
						&& (!prp_has_usable_response(info) || info.max_frontier_hop_dst >= runtime.curr_k_dst);
				}
				if (!pending)
				{
					runtime.finished = true;
					runtime.no_response_terminal = !has_response;
					continue;
				}
				const uint next_src = grow_src ? grow_kring(runtime.curr_k_src, runtime.mesh_diameter_src) : runtime.curr_k_src;
				const uint next_dst = grow_dst ? grow_kring(runtime.curr_k_dst, runtime.mesh_diameter_dst) : runtime.curr_k_dst;
				if (next_src <= runtime.curr_k_src && next_dst <= runtime.curr_k_dst)
					LUISA_ERROR("PRP expansion stalled for contour {} at k {}/{}.", contour_idx, runtime.curr_k_src, runtime.curr_k_dst);
				runtime.curr_k_src = next_src;
				runtime.curr_k_dst = next_dst;
				next.push_back(contour_idx);
			}
			remaining.swap(next);
		}

		if (get_scene_params().prp_debug && print_combo_summary)
		{
			for (const uint contour_idx : target_contours)
			{
				const auto& runtime = runtimes[contour_idx];
				const auto	best = find_best_result_in_combo_results(runtime.combo_results);
				if (best == runtime.combo_results.end())
					LUISA_INFO("Contour {}: no usable Cluster Culling response.", contour_idx);
				else
					LUISA_INFO("Contour {}: combo {}, k {}/{}, hits {}, coverage {}/{}, objective {}.",
						contour_idx, best->combo_idx, best->evaluated_kring_src, best->evaluated_kring_dst,
						best->hit_count, best->covered_anchor_count, best->boundary_anchor_count,
						prp_response_objective(*best));
			}
		}
	}

	float3 get_hit_characteristic_normal(const RayCasting::HitInfo& hit, const float3* sa_x)
	{
		if (hit.is_vf())
		{
			const float3 A = sa_x[hit.dst_verts[0]], B = sa_x[hit.dst_verts[1]], C = sa_x[hit.dst_verts[2]];
			return luisa::cross(B - A, C - A);
		}
		if (hit.is_fv())
		{
			const float3 A = sa_x[hit.src_verts[0]], B = sa_x[hit.src_verts[1]], C = sa_x[hit.src_verts[2]];
			return luisa::cross(B - A, C - A);
		}
		if (hit.is_ee())
		{
			const float3 p0 = sa_x[hit.src_verts[0]], p1 = sa_x[hit.src_verts[1]], q0 = sa_x[hit.dst_verts[0]], q1 = sa_x[hit.dst_verts[1]];
			float3		 c = luisa::cross(p1 - p0, q1 - q0);
			if (luisa::length(c) < 1e-8f)
			{
				float3 pt1 = p0 * hit.bary[0] + p1 * (1.0f - hit.bary[0]), pt2 = q0 * hit.bary[1] + q1 * (1.0f - hit.bary[1]);
				c = pt2 - pt1;
			}
			return c;
		}
		return luisa::make_float3(0, 1, 0);
	}

	namespace
	{
		struct PRPDirectionOptimizationStep
		{
			float3 direction = luisa::make_float3(0.0f);
			float  predicted_reduction = 0.0f;
			float  gradient_norm = 0.0f;
			bool   valid = false;
		};

		PRPDirectionOptimizationStep prp_compute_direction_optimization_step(
			MeshData<std::vector>*					host_mesh_data,
			SimulationData<std::vector>*			host_sim_data,
			const std::vector<RayCasting::HitInfo>& hits,
			const float3&							input_direction,
			const float								trust_radius)
		{
			if (hits.empty() || !prp_is_valid_direction(input_direction))
				return {};

			const float3 r = luisa::normalize(input_direction);
			float3		 grad = luisa::make_float3(0.0f);
			float3x3	 hess = luisa::make_float3x3(0.0f);
			const float	 displacement_offset = prp_response_depth_offset;
			uint		 contributing_hits = 0u;

			for (const auto& hit : hits)
			{
				const float3 raw_normal = get_hit_characteristic_normal(hit, host_sim_data->sa_x.data());
				const float	 normal_length = luisa::length(raw_normal);
				if (!std::isfinite(normal_length) || normal_length < 1e-8f)
					continue;
				const float3 n = raw_normal / normal_length;
				const float	 n_dot_r = luisa::dot(n, r);
				if (!std::isfinite(n_dot_r) || std::abs(n_dot_r) < 1e-6f)
					continue;

				const float effective_inv_mass = compute_effective_inverse_mass_weight(hit, host_mesh_data->sa_vert_mass);
				if (!std::isfinite(effective_inv_mass) || effective_inv_mass <= 1e-8f)
					continue;

				const float	 weight = 1.0f / effective_inv_mass;
				const float	 depth = hit.dist;
				const float3 depth_derivative = -depth * n / n_dot_r;
				grad += 2.0f * weight * (depth + displacement_offset) * depth_derivative;
				const float hessian_scale = 2.0f * weight
					/ (n_dot_r * n_dot_r)
					* (3.0f * depth * depth + 2.0f * depth * displacement_offset);
				hess = hess + hessian_scale * outer_product(n, n);
				contributing_hits++;
			}

			if (contributing_hits == 0u)
				return {};

			const float3 tangent_grad = grad - r * luisa::dot(r, grad);
			const float3 reference_axis = std::abs(r.y) < 0.9f
				? luisa::make_float3(0.0f, 1.0f, 0.0f)
				: luisa::make_float3(1.0f, 0.0f, 0.0f);
			const float3 e1 = luisa::normalize(luisa::cross(reference_axis, r));
			const float3 e2 = luisa::normalize(luisa::cross(r, e1));
			const float	 g0 = luisa::dot(e1, tangent_grad);
			const float	 g1 = luisa::dot(e2, tangent_grad);
			const float	 gradient_norm = std::sqrt(g0 * g0 + g1 * g1);
			if (!std::isfinite(gradient_norm) || gradient_norm < 1e-6f)
				return {};

			const float h00 = luisa::dot(e1, hess * e1);
			const float h01 = luisa::dot(e1, hess * e2);
			const float h10 = luisa::dot(e2, hess * e1);
			const float h11 = luisa::dot(e2, hess * e2);
			const float det = h00 * h11 - h01 * h10;
			float		delta0 = 0.0f;
			float		delta1 = 0.0f;
			if (std::isfinite(det) && det > 1e-10f && h00 > 0.0f)
			{
				delta0 = -(h11 * g0 - h01 * g1) / det;
				delta1 = -(h00 * g1 - h10 * g0) / det;
			}
			else
			{
				const float gg = g0 * g0 + g1 * g1;
				const float g_h_g = g0 * (h00 * g0 + h01 * g1)
					+ g1 * (h10 * g0 + h11 * g1);
				const float alpha = g_h_g > 1e-10f
					? gg / g_h_g
					: trust_radius / (gradient_norm + 1e-20f);
				delta0 = -alpha * g0;
				delta1 = -alpha * g1;
			}

			const float step_norm = std::sqrt(delta0 * delta0 + delta1 * delta1);
			if (!std::isfinite(step_norm) || step_norm < 1e-7f)
				return {};
			if (step_norm > trust_radius)
			{
				const float scale = trust_radius / step_norm;
				delta0 *= scale;
				delta1 *= scale;
			}

			const float3 trial_direction = r + delta0 * e1 + delta1 * e2;
			if (!prp_is_valid_direction(trial_direction))
				return {};
			return {
				luisa::normalize(trial_direction),
				std::max(0.0f, -(g0 * delta0 + g1 * delta1)),
				gradient_norm,
				true
			};
		}
	} // namespace

	void prp_optimize_runtime_directions(
		std::vector<PrpStandardContourRuntime>&														  runtimes,
		const std::vector<uint>&																	  target_contours,
		MeshData<std::vector>*																		  host_mesh_data,
		SimulationData<std::vector>*																  host_sim_data,
		const uint																					  max_iterations,
		const std::function<void(std::vector<PrpStandardContourRuntime>&, const std::vector<uint>&)>& evaluate_trial_directions)
	{
		if (max_iterations == 0u || target_contours.empty())
			return;

		struct RuntimeOptimizationState
		{
			uint  combo_idx = std::numeric_limits<uint>::max();
			uint  attempted = 0u;
			uint  accepted = 0u;
			float trust_radius = 0.3f;
			bool  done = false;
		};
		std::vector<RuntimeOptimizationState> states(runtimes.size());
		for (const uint runtime_idx : target_contours)
		{
			const auto best = find_best_result_in_combo_results(runtimes[runtime_idx].combo_results);
			if (best == runtimes[runtime_idx].combo_results.end()
				|| best->hit_list.empty())
			{
				states[runtime_idx].done = true;
				continue;
			}
			states[runtime_idx].combo_idx = best->combo_idx;
		}

		for (uint iteration = 0u; iteration < max_iterations; ++iteration)
		{
			std::vector<PRPMinContourHitInfo> accepted_results(runtimes.size());
			std::vector<float3>				  accepted_directions(runtimes.size(), luisa::make_float3(0.0f));
			std::vector<std::vector<uint>>	  active_combo_backups(runtimes.size());
			std::vector<float>				  predicted_reductions(runtimes.size(), 0.0f);
			std::vector<float>				  gradient_norms(runtimes.size(), 0.0f);
			std::vector<uint>				  trial_contours;
			trial_contours.reserve(target_contours.size());

			for (const uint runtime_idx : target_contours)
			{
				auto& state = states[runtime_idx];
				auto& runtime = runtimes[runtime_idx];
				if (state.done || state.combo_idx >= runtime.combo_results.size())
					continue;
				const auto& accepted = runtime.combo_results[state.combo_idx];
				const auto	step = prp_compute_direction_optimization_step(
					host_mesh_data,
					host_sim_data,
					accepted.hit_list,
					accepted.separation_direction,
					state.trust_radius);
				if (!step.valid)
				{
					state.done = true;
					continue;
				}

				state.attempted++;
				accepted_results[runtime_idx] = accepted;
				accepted_directions[runtime_idx] = runtime.contour_dirs[state.combo_idx];
				active_combo_backups[runtime_idx] = runtime.active_combos;
				predicted_reductions[runtime_idx] = step.predicted_reduction;
				gradient_norms[runtime_idx] = step.gradient_norm;
				runtime.contour_dirs[state.combo_idx] = step.direction;
				runtime.active_combos = { state.combo_idx };
				trial_contours.push_back(runtime_idx);
			}

			if (trial_contours.empty())
				break;
			evaluate_trial_directions(runtimes, trial_contours);

			for (const uint runtime_idx : trial_contours)
			{
				auto&	   state = states[runtime_idx];
				auto&	   runtime = runtimes[runtime_idx];
				const uint combo_idx = state.combo_idx;
				runtime.active_combos = std::move(active_combo_backups[runtime_idx]);
				const auto								accepted = std::move(accepted_results[runtime_idx]);
				auto&									trial = runtime.combo_results[combo_idx];
				const std::vector<PRPMinContourHitInfo> acceptance_candidates = {
					accepted, trial
				};
				const auto acceptance = prp_select_response(acceptance_candidates);
				const bool accept_trial = acceptance.has_response()
					&& acceptance.selected_idx == 1u;
				if (accept_trial)
				{
					state.accepted++;
					trial.direction_optimization_iteration_count = state.attempted;
					trial.direction_optimization_accepted_count = state.accepted;
					runtime.contour_dirs[combo_idx] = trial.separation_direction;
					const float actual_reduction = prp_response_objective(accepted)
						- prp_response_objective(trial);
					if (predicted_reductions[runtime_idx] > 1e-20f
						&& actual_reduction / predicted_reductions[runtime_idx] > 0.75f)
						state.trust_radius = std::min(1.0f, state.trust_radius * 2.0f);
				}
				else
				{
					runtime.combo_results[combo_idx] = accepted;
					runtime.combo_results[combo_idx].direction_optimization_iteration_count = state.attempted;
					runtime.combo_results[combo_idx].direction_optimization_accepted_count = state.accepted;
					runtime.contour_dirs[combo_idx] = accepted_directions[runtime_idx];
					state.trust_radius = std::max(0.01f, state.trust_radius * 0.5f);
					if (state.trust_radius <= 0.01f
						|| gradient_norms[runtime_idx] < 1e-6f)
						state.done = true;
				}
			}
		}
	}

	std::vector<float3> get_raycasting_direction(
		const std::vector<float3>&				contour_points_3D,
		const std::vector<float>&				contour_points_weight,
		const std::array<std::vector<uint>, 2>& boundary_verts,
		SimulationData<std::vector>*			host_sim_data,
		MeshData<std::vector>*					host_mesh_data)
	{
		auto add_dynamic_seed = [](std::vector<float3>& directions, const float3& raw_dir)
		{
			if (!prp_is_valid_direction(raw_dir))
				return;
			directions.push_back(prp_safe_normalize(raw_dir, luisa::make_float3(0.0f, 1.0f, 0.0f)));
		};
		auto add_dynamic_signed = [&](std::vector<float3>& directions, const float3& dir)
		{
			add_dynamic_seed(directions, dir);
			add_dynamic_seed(directions, -dir);
		};

		std::vector<float3>				  contour_directions;
		const auto&						  input_verts = boundary_verts;
		const float3					  contour_center = get_weighted_mean(contour_points_3D, luisa::make_float3(0.0f), contour_points_weight);
		std::array<std::vector<float>, 2> mesh_vert_weights;
		std::array<float3, 2>			  mesh_centers = {
			luisa::make_float3(0.0f), luisa::make_float3(0.0f)
		};
		float3x3 Sw = luisa::make_float3x3(0.0f);

		for (uint mesh_idx = 0; mesh_idx < 2; mesh_idx++)
		{
			const auto& vert_set = input_verts[mesh_idx];
			auto&		vert_weights = mesh_vert_weights[mesh_idx];
			vert_weights.reserve(vert_set.size());
			for (const uint vid : vert_set)
			{
				const float3 vert_pos = host_sim_data->sa_x[vid];
				const float	 weight = std::max(host_mesh_data->sa_rest_vert_area[vid], 1e-8f);
				vert_weights.push_back(weight);
				mesh_centers[mesh_idx] += vert_pos * weight;
				const float3 centered = vert_pos - contour_center;
				Sw = Sw + weight * outer_product(centered, centered);
			}
			const float sum_weights = std::accumulate(vert_weights.begin(), vert_weights.end(), 0.0f);
			mesh_centers[mesh_idx] = sum_weights > 1e-8f
				? mesh_centers[mesh_idx] / sum_weights
				: contour_center;
		}

		const auto contour_proxy_normal = compute_inner_lda_direction(contour_points_3D, contour_points_weight);
		const auto eigvecs = compute_all_eigenvectors(Sw);

		for (const auto& eigvec : std::span(eigvecs).subspan(0, 3))
			add_dynamic_signed(contour_directions, eigvec);

		for (uint mesh_idx = 0; mesh_idx < 2; mesh_idx++)
		{
			float3 dir = contour_center - mesh_centers[mesh_idx];
			if (!prp_is_valid_direction(dir))
				dir = eigvecs[mesh_idx + 1];
			add_dynamic_seed(contour_directions, dir);
		}

		float3 global_centroid = luisa::make_float3(0.0f);
		float  total_mass = 0.0f;
		for (uint v = 0; v < host_mesh_data->num_verts; ++v)
		{
			const float w = std::max(host_mesh_data->sa_rest_vert_area[v], 1e-8f);
			global_centroid += host_sim_data->sa_x[v] * w;
			total_mass += w;
		}
		if (total_mass > 1e-8f)
		{
			global_centroid /= total_mass;
			const float3 radial_outward = contour_center - global_centroid;
			if (prp_is_valid_direction(radial_outward))
				add_dynamic_signed(contour_directions, radial_outward);
		}

		if (contour_directions.empty())
			add_dynamic_signed(contour_directions, contour_proxy_normal);
		return contour_directions;
	}

	void build_standard_candidates_at_level(
		PrpStandardContourRuntime&		   runtime,
		const uint						   k_src,
		const uint						   k_dst,
		const MeshData<std::vector>*	   host_mesh_data,
		const SimulationData<std::vector>* host_sim_data)
	{
		auto build_sparse_candidates_for_side = [&](const uint mesh_idx, const uint k)
		{
			while (!runtime.sparse_boundary_hop_exhausted[mesh_idx]
				&& runtime.sparse_boundary_hop_max_discovered_hop[mesh_idx] < k)
			{
				prp_expand_sparse_boundary_hop_one_level(runtime, host_mesh_data, mesh_idx);
			}

			auto& verts = runtime.partial_candidate_verts[mesh_idx];
			auto& edges = runtime.partial_candidate_edges[mesh_idx];
			auto& faces = runtime.partial_candidate_faces[mesh_idx];
			verts.clear();
			edges.clear();
			faces.clear();

			const uint2 vert_range = runtime.mesh_candidate_vert_range[mesh_idx];
			const uint2 edge_range = runtime.mesh_candidate_edge_range[mesh_idx];
			const uint2 face_range = runtime.mesh_candidate_face_range[mesh_idx];
			auto		fn_vert_is_in_range = [&](const uint vid) -> bool
			{
				// return vid >= vert_range.x && vid < vert_end;
				return std::binary_search(runtime.mesh_candidate_verts[mesh_idx].begin(), runtime.mesh_candidate_verts[mesh_idx].end(), vid);
			};
			auto fn_edge_is_in_range = [&](const uint eid) -> bool
			{
				return std::binary_search(runtime.mesh_candidate_edges[mesh_idx].begin(), runtime.mesh_candidate_edges[mesh_idx].end(), eid);
			};
			auto fn_face_is_in_range = [&](const uint fid) -> bool
			{
				return std::binary_search(runtime.mesh_candidate_faces[mesh_idx].begin(), runtime.mesh_candidate_faces[mesh_idx].end(), fid);
			};

			for (const auto& [vid, hop] : runtime.sparse_boundary_hop_dist[mesh_idx])
				if (hop <= k && fn_vert_is_in_range(vid))
					verts.push_back(vid);

			for (const uint vid : verts)
			{
				for (const uint eid : host_mesh_data->vert_adj_edges[vid])
				{
					if (!fn_edge_is_in_range(eid))
						continue;
					const uint2 edge = host_mesh_data->sa_edges[eid];
					if (runtime.boundary_hop_distance(mesh_idx, edge.x) <= k && runtime.boundary_hop_distance(mesh_idx, edge.y) <= k)
						edges.push_back(eid);
				}
				for (const uint fid : host_mesh_data->vert_adj_faces[vid])
				{
					if (!fn_face_is_in_range(fid))
						continue;
					const uint3 face = host_mesh_data->sa_faces[fid];
					if (runtime.boundary_hop_distance(mesh_idx, face.x) <= k
						&& runtime.boundary_hop_distance(mesh_idx, face.y) <= k
						&& runtime.boundary_hop_distance(mesh_idx, face.z) <= k)
					{
						faces.push_back(fid);
					}
				}
			}
			std::sort(edges.begin(), edges.end());
			edges.erase(std::unique(edges.begin(), edges.end()), edges.end());
			std::sort(faces.begin(), faces.end());
			faces.erase(std::unique(faces.begin(), faces.end()), faces.end());
		};

		for (uint mesh_idx = 0u; mesh_idx < 2u; ++mesh_idx)
		{
			const uint	k = (mesh_idx == 0u) ? k_src : k_dst;
			const auto& dist = runtime.vert_to_boundary_hop_dist[mesh_idx];
			runtime.partial_candidate_verts[mesh_idx].clear();
			runtime.partial_candidate_edges[mesh_idx].clear();
			runtime.partial_candidate_faces[mesh_idx].clear();

			if (runtime.mesh_candidate_verts[mesh_idx].empty() && !runtime.sparse_boundary_hop_dist[mesh_idx].empty())
			{
				build_sparse_candidates_for_side(mesh_idx, k);
				continue;
			}

			for (const uint vid : runtime.mesh_candidate_verts[mesh_idx])
				if (dist[vid] <= k)
					runtime.partial_candidate_verts[mesh_idx].push_back(vid);

			for (const uint eid : runtime.mesh_candidate_edges[mesh_idx])
			{
				const uint2 edge = host_mesh_data->sa_edges[eid];
				if (dist[edge.x] <= k && dist[edge.y] <= k)
					runtime.partial_candidate_edges[mesh_idx].push_back(eid);
			}

			for (const uint fid : runtime.mesh_candidate_faces[mesh_idx])
			{
				const uint3 face = host_mesh_data->sa_faces[fid];
				if (dist[face.x] <= k && dist[face.y] <= k && dist[face.z] <= k)
					runtime.partial_candidate_faces[mesh_idx].push_back(fid);
			}
		}

		runtime.intrinsic_contour_side_candidate_restriction_applied =
			prp_intrinsic_contour_side_restriction_applies(
				get_scene_params().PRP_use_intrinsic_contour_side_candidates
					&& (get_scene_params().PRP_use_rest_geodesic_distance_for_intrinsic_candidates
						|| get_scene_params().PRP_use_deformed_boundary_distance_for_intrinsic_candidates),
				runtime.object_ids,
				runtime.topology_component_ids,
				get_scene_params().PRP_use_deformed_boundary_distance_for_intrinsic_candidates);
		for (uint role = 0u; role < 2u; ++role)
		{
			runtime.intrinsic_candidate_vertex_count_before[role] = static_cast<uint>(runtime.partial_candidate_verts[role].size());
			runtime.intrinsic_candidate_edge_count_before[role] = static_cast<uint>(runtime.partial_candidate_edges[role].size());
			runtime.intrinsic_candidate_face_count_before[role] = static_cast<uint>(runtime.partial_candidate_faces[role].size());
		}
		if (runtime.intrinsic_contour_side_candidate_restriction_applied)
		{
			const bool use_rest_distance = get_scene_params().PRP_use_rest_geodesic_distance_for_intrinsic_candidates;
			const bool use_deformed_boundary_distance = get_scene_params().PRP_use_deformed_boundary_distance_for_intrinsic_candidates;
			if (use_deformed_boundary_distance)
			{
				if (host_sim_data == nullptr)
					LUISA_ERROR("Deformed boundary distance requires host simulation positions.");
				if (runtime.boundary_verts[0].empty() || runtime.boundary_verts[1].empty())
					LUISA_ERROR(
						"Contour {} cannot compute deformed boundary distance with empty side anchors ({}, {}).",
						runtime.contour_idx,
						runtime.boundary_verts[0].size(),
						runtime.boundary_verts[1].size());
			}

			auto deformed_boundary_distance = [&](const uint boundary_role, const uint vid) -> float
			{
				auto& cache = runtime.deformed_boundary_distance_cache[boundary_role];
				if (const auto it = cache.find(vid); it != cache.end())
					return it->second;
				if (vid >= host_sim_data->sa_x.size())
					LUISA_ERROR(
						"Contour {} candidate vertex {} exceeds current-position array size {}.",
						runtime.contour_idx, vid, host_sim_data->sa_x.size());
				const float3 x = host_sim_data->sa_x[vid];
				if (!prp_is_finite_float3(x))
					LUISA_ERROR("Contour {} candidate vertex {} has a non-finite current position.", runtime.contour_idx, vid);

				float min_distance = std::numeric_limits<float>::max();
				for (const uint boundary_vid : runtime.boundary_verts[boundary_role])
				{
					if (boundary_vid >= host_sim_data->sa_x.size())
						LUISA_ERROR(
							"Contour {} boundary vertex {} exceeds current-position array size {}.",
							runtime.contour_idx, boundary_vid, host_sim_data->sa_x.size());
					const float3 boundary_x = host_sim_data->sa_x[boundary_vid];
					if (!prp_is_finite_float3(boundary_x))
						LUISA_ERROR("Contour {} boundary vertex {} has a non-finite current position.", runtime.contour_idx, boundary_vid);
					min_distance = std::min(min_distance, luisa::distance(x, boundary_x));
				}
				if (!std::isfinite(min_distance))
					LUISA_ERROR(
						"Contour {} role {} produced a non-finite deformed boundary distance for vertex {}.",
						runtime.contour_idx, boundary_role, vid);
				cache.emplace(vid, min_distance);
				runtime.deformed_boundary_distance_min[boundary_role] = std::min(runtime.deformed_boundary_distance_min[boundary_role], min_distance);
				runtime.deformed_boundary_distance_max[boundary_role] = std::max(runtime.deformed_boundary_distance_max[boundary_role], min_distance);
				return min_distance;
			};

			for (uint role = 0u; role < 2u; ++role)
			{
				auto current_side_distance = [&](const uint boundary_role, const uint vid) -> float
				{
					const float rest_distance = runtime.boundary_geom_distance(boundary_role, vid);
					const float current_distance = deformed_boundary_distance(boundary_role, vid);
					// Retain the historical distance-comparison diagnostic, without fusing metrics.
					if (current_distance < rest_distance)
						runtime.deformed_boundary_distance_shortened_vertices[boundary_role].insert(vid);
					return current_distance;
				};
				auto owns_vertex = [&](const uint vid)
				{
					const float rest_own = runtime.boundary_geom_distance(role, vid);
					const float rest_opposite = runtime.boundary_geom_distance(role ^ 1u, vid);
					const float tau = get_scene_params().PRP_intrinsic_tau;
					if (get_scene_params().PRP_use_blended_intrinsic_coordinates
						&& use_rest_distance && use_deformed_boundary_distance)
						return prp_intrinsic_contour_side_owns_vertex_blended(
							rest_own, rest_opposite,
							current_side_distance(role, vid), current_side_distance(role ^ 1u, vid),
							tau, get_scene_params().PRP_intrinsic_blend_weight);
					if (use_rest_distance && !prp_intrinsic_contour_side_owns_vertex(rest_own, rest_opposite, tau))
						return false;
					if (!use_deformed_boundary_distance)
						return true;
					return prp_dual_metric_contour_side_owns_vertex(
						rest_own, rest_opposite,
						current_side_distance(role, vid), current_side_distance(role ^ 1u, vid),
						tau, use_rest_distance, true);
				};

				auto& verts = runtime.partial_candidate_verts[role];
				verts.erase(std::remove_if(verts.begin(), verts.end(),
								[&](const uint vid)
								{ return !owns_vertex(vid); }),
					verts.end());

				auto& edges = runtime.partial_candidate_edges[role];
				edges.erase(std::remove_if(edges.begin(), edges.end(),
								[&](const uint eid)
								{
									const uint2 edge = host_mesh_data->sa_edges[eid];
									return !owns_vertex(edge.x) && !owns_vertex(edge.y);
								}),
					edges.end());

				auto& faces = runtime.partial_candidate_faces[role];
				faces.erase(std::remove_if(faces.begin(), faces.end(),
								[&](const uint fid)
								{
									const uint3 face = host_mesh_data->sa_faces[fid];
									return !owns_vertex(face.x)
										&& !owns_vertex(face.y)
										&& !owns_vertex(face.z);
								}),
					faces.end());
			}
		}
		runtime.total_candidate_count = static_cast<uint>(
			runtime.partial_candidate_verts[0].size() + runtime.partial_candidate_verts[1].size()
			+ runtime.partial_candidate_edges[0].size() + runtime.partial_candidate_edges[1].size()
			+ runtime.partial_candidate_faces[0].size() + runtime.partial_candidate_faces[1].size());
	}

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
		const std::vector<float3>&				vert_normal)
	{
		const auto eval_cost = prp_eval_penetration_v2(hits_info,
			sa_rest_face_area, sa_rest_edge_area, sa_rest_vert_area,
			host_mesh_data->sa_vert_mass,
			face_normal, edge_normal, vert_normal, runtime.need_flip, runtime);
		info.penetration_depth = eval_cost.depth;
		info.penetration_volume = eval_cost.volume;
		info.penetration_area = eval_cost.area;
		info.penetration_objective = eval_cost.objective;
		info.displacement_cost = eval_cost.displacement_cost;
		info.displacement_cost_area = eval_cost.displacement_cost_area;
		info.displacement_cost_kappa = eval_cost.displacement_cost_kappa;
		info.hit_count = static_cast<uint>(hits_info.size());

		// Finalized EF-pair coverage via per-mesh hit support-vertex sets.
		const auto& contour = runtime.contour;
		const uint	ef_count = static_cast<uint>(contour.size());
		uint		covered_ef_count = 0u;
		if (ef_count > 0u)
		{
			const uint								 num_verts = host_mesh_data->num_verts;
			std::array<std::vector<luisa::ubyte>, 2> support_set = {
				std::vector<luisa::ubyte>(num_verts, 0u), std::vector<luisa::ubyte>(num_verts, 0u)
			};
			for (const auto& h : hits_info)
			{
				const uint sm = h.src_mesh_idx;
				visit_prp_hit_source_vertices(h, [&](const uint vid)
					{ if (vid < num_verts) support_set[sm][vid] = 1u; });
				visit_prp_hit_target_vertices(h, [&](const uint vid)
					{ if (vid < num_verts) support_set[sm ^ 1u][vid] = 1u; });
			}
			for (uint pi = 0u; pi < ef_count; ++pi)
			{
				const uint	pair_idx = contour[pi];
				const uint	mesh_idx = host_untangling_data->ef_pair_mesh_index[pair_idx];
				const auto& ef_pair = host_collision_data->narrow_phase_list_ef[pair_idx];
				const uint2 edge = ef_pair.get_edge();
				const uint3 face = ef_pair.get_face();
				// An EF pair is covered when its edge (mesh_idx) and opposite face (mesh_idx^1) vertices appear in the support sets.
				const auto& e_set = support_set[mesh_idx];
				const auto& f_set = support_set[mesh_idx ^ 1u];
				const bool	edge_ok = e_set[edge.x] || e_set[edge.y];
				const bool	face_ok = f_set[face.x] || f_set[face.y] || f_set[face.z];
				if (edge_ok && face_ok)
					++covered_ef_count;
			}
		}
		info.boundary_anchor_count = ef_count;
		info.covered_anchor_count = covered_ef_count;
		info.contour_coverage = ef_count > 0u
			? static_cast<float>(covered_ef_count) / static_cast<float>(ef_count)
			: 0.0f;
		if (covered_ef_count > 0u)
			info.normalized_displacement_cost = info.displacement_cost
				/ static_cast<float>(covered_ef_count);
		info.normalized_penetration_objective = prp_response_objective(info);

		const MinMaxHopInfo min_max_info = compute_hit_min_max_frontier_hop(runtime, hits_info);
		info.max_frontier_hop_src = min_max_info.max_frontier_hop_src();
		info.max_frontier_hop_dst = min_max_info.max_frontier_hop_dst();
		info.selected_frontier_converged = info.max_frontier_hop_src < info.evaluated_kring_src && info.max_frontier_hop_dst < info.evaluated_kring_dst;
	}

	PRPPenetrationEvaluation prp_eval_penetration_v2(
		const std::vector<RayCasting::HitInfo>& hits_info,
		const std::vector<float>&				sa_rest_face_area,
		const std::vector<float>&				sa_rest_edge_area,
		const std::vector<float>&				sa_rest_vert_area,
		const std::vector<float>&				sa_vert_mass,
		const std::vector<float3>&				face_normal,
		const std::vector<float3>&				edge_normal,
		const std::vector<float3>&				vert_normal,
		const bool								need_flip,
		const PrpStandardContourRuntime&		runtime)
	{
		PRPPenetrationEvaluation eval{};
		if (hits_info.empty())
			return eval;

		constexpr float prp_d_hat = 3e-3f;
		eval = CpuParallel::parallel_for_and_reduce(
			0u, static_cast<uint>(hits_info.size()),
			[&](const uint idx) -> PRPPenetrationEvaluation
			{
				const auto& hit = hits_info[idx];

				float area = 0.0f;
				float alpha = 0.0f;
				if (hit.is_vf())
				{
					area = sa_rest_vert_area[hit.get_vid()];
					alpha = luisa::abs(luisa::dot(vert_normal[hit.get_vid()], hit.get_direction()));
				}
				else if (hit.is_ee())
				{
					area = sa_rest_edge_area[hit.get_eid1()];
					alpha = luisa::abs(luisa::dot(edge_normal[hit.get_eid1()], hit.get_direction()));
				}
				else if (hit.is_fv())
				{
					area = sa_rest_face_area[hit.get_fid()];
					alpha = luisa::abs(luisa::dot(face_normal[hit.get_fid()], hit.get_direction()));
				}
				const float dist = hit.dist;
				const float shifted_dist = dist + prp_d_hat;
				const float effective_inv_mass = compute_effective_inverse_mass_weight(hit, sa_vert_mass);

				return PRPPenetrationEvaluation{
					.volume = area * alpha * dist,
					.depth = dist,
					.objective_area = area,
					.area = area,
					.displacement_cost_area = area * shifted_dist * shifted_dist,
					.displacement_cost_kappa = effective_inv_mass > 1e-8f ? (shifted_dist * shifted_dist / effective_inv_mass) : 0.0f
				};
			},
			[](const PRPPenetrationEvaluation& a, const PRPPenetrationEvaluation& b)
			{
				PRPPenetrationEvaluation c;
				c.volume = a.volume + b.volume;
				c.depth = std::max(a.depth, b.depth);
				c.objective_area = a.objective_area + b.objective_area;
				c.area = a.area + b.area;
				c.displacement_cost_area = a.displacement_cost_area + b.displacement_cost_area;
				c.displacement_cost_kappa = a.displacement_cost_kappa + b.displacement_cost_kappa;
				c.invalid_hit_count = a.invalid_hit_count + b.invalid_hit_count;
				c.invalid_area = a.invalid_area + b.invalid_area;
				return c;
			},
			PRPPenetrationEvaluation{});

		eval.objective = eval.objective_area * eval.depth * eval.depth;
		eval.displacement_cost = eval.displacement_cost_kappa;
		return eval;
	}

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
		const std::vector<float3>&					  vert_normal)
	{
		for (const uint contour_idx : target_contours)
		{
			if (contour_idx >= list_min_hit_info.size() || contour_idx >= runtimes.size())
				continue;

			auto& info = list_min_hit_info[contour_idx];
			if (info.no_response_terminal)
			{
				if (!info.hit_list.empty())
					LUISA_ERROR(
						"PRP no-response contour {} unexpectedly contains {} materialized Hits.",
						contour_idx,
						info.hit_list.size());
				continue;
			}
			const uint	pre_finalize_hit_count = static_cast<uint>(info.hit_list.size());
			const float pre_finalize_penetration_depth = info.penetration_depth;
			const float pre_finalize_penetration_objective = info.penetration_objective;
			const float pre_finalize_displacement_cost = info.displacement_cost;
			const uint	pre_finalize_covered_anchor_count = info.covered_anchor_count;
			const float pre_finalize_contour_coverage = info.contour_coverage;
			const auto& runtime = runtimes[contour_idx];
			// Evaluate the selected support with the shared host arithmetic before assembly.
			const auto finalized_eval = prp_eval_penetration_v2(
				info.hit_list,
				sa_rest_face_area,
				sa_rest_edge_area,
				sa_rest_vert_area,
				host_mesh_data->sa_vert_mass,
				face_normal,
				edge_normal,
				vert_normal,
				runtimes[contour_idx].need_flip,
				runtimes[contour_idx]);
			info.penetration_depth = finalized_eval.depth;
			info.penetration_volume = finalized_eval.volume;
			info.penetration_area = finalized_eval.area;
			info.penetration_objective = finalized_eval.objective;
			info.displacement_cost = finalized_eval.displacement_cost;
			info.displacement_cost_area = finalized_eval.displacement_cost_area;
			info.displacement_cost_kappa = finalized_eval.displacement_cost_kappa;
			info.hit_count = static_cast<uint>(info.hit_list.size());

			const uint								 num_verts = host_mesh_data->num_verts;
			std::array<std::vector<luisa::ubyte>, 2> support = {
				std::vector<luisa::ubyte>(num_verts, 0u),
				std::vector<luisa::ubyte>(num_verts, 0u)
			};
			for (const auto& hit : info.hit_list)
			{
				if (hit.src_mesh_idx >= 2u)
					LUISA_ERROR("Finalized PRP Hit has invalid source mesh role {}.", hit.src_mesh_idx);
				visit_prp_hit_source_vertices(hit,
					[&](const uint vid)
					{
						if (vid >= num_verts)
							LUISA_ERROR("Finalized PRP source vertex {} exceeds vertex count {}.", vid, num_verts);
						support[hit.src_mesh_idx][vid] = 1u;
					});
				visit_prp_hit_target_vertices(hit,
					[&](const uint vid)
					{
						if (vid >= num_verts)
							LUISA_ERROR("Finalized PRP target vertex {} exceeds vertex count {}.", vid, num_verts);
						support[hit.src_mesh_idx ^ 1u][vid] = 1u;
					});
			}
			uint finalized_covered_anchor_count = 0u;
			for (const uint pair_idx : runtime.contour)
			{
				if (pair_idx >= host_untangling_data->ef_pair_mesh_index.size()
					|| pair_idx >= host_collision_data->narrow_phase_list_ef.size())
					LUISA_ERROR("Finalized PRP contour pair {} exceeds EF storage.", pair_idx);
				const uint mesh_idx = host_untangling_data->ef_pair_mesh_index[pair_idx];
				if (mesh_idx >= 2u)
					LUISA_ERROR("Finalized PRP EF pair {} has invalid mesh role {}.", pair_idx, mesh_idx);
				const auto& ef_pair = host_collision_data->narrow_phase_list_ef[pair_idx];
				const uint2 edge = ef_pair.get_edge();
				const uint3 face = ef_pair.get_face();
				if (edge.x >= num_verts || edge.y >= num_verts
					|| face.x >= num_verts || face.y >= num_verts || face.z >= num_verts)
					LUISA_ERROR("Finalized PRP EF pair {} references a vertex outside {} vertices.", pair_idx, num_verts);
				const bool edge_covered = support[mesh_idx][edge.x] || support[mesh_idx][edge.y];
				const bool face_covered = support[mesh_idx ^ 1u][face.x]
					|| support[mesh_idx ^ 1u][face.y]
					|| support[mesh_idx ^ 1u][face.z];
				finalized_covered_anchor_count += edge_covered && face_covered ? 1u : 0u;
			}
			info.boundary_anchor_count = static_cast<uint>(runtime.contour.size());
			info.covered_anchor_count = finalized_covered_anchor_count;
			info.contour_coverage = info.boundary_anchor_count > 0u
				? static_cast<float>(finalized_covered_anchor_count)
					/ static_cast<float>(info.boundary_anchor_count)
				: 0.0f;
			info.normalized_penetration_objective = prp_penetration_objective_divided_by_squared_coverage(info.penetration_objective, info.contour_coverage);
			info.normalized_displacement_cost = finalized_covered_anchor_count > 0u
				? info.displacement_cost / static_cast<float>(finalized_covered_anchor_count)
				: std::numeric_limits<float>::infinity();
			if (get_scene_params().prp_debug)
			{
				LUISA_INFO(
					"  [Finalize response evaluation] Contour {}: hits {} -> {}, coverage {}/{} ({:.6f}) -> {}/{} ({:.6f}), depth {:.6e} -> {:.6e}, objective {:.6e} -> {:.6e}, cost {:.6e} -> {:.6e}.",
					contour_idx,
					pre_finalize_hit_count,
					info.hit_count,
					pre_finalize_covered_anchor_count,
					info.boundary_anchor_count,
					pre_finalize_contour_coverage,
					info.covered_anchor_count,
					info.boundary_anchor_count,
					info.contour_coverage,
					pre_finalize_penetration_depth,
					info.penetration_depth,
					pre_finalize_penetration_objective,
					info.penetration_objective,
					pre_finalize_displacement_cost,
					info.displacement_cost);
				const std::string finalize_key = "prp.c" + std::to_string(contour_idx) + ".finalize.";
				auto&			  debug = get_scene_params().prp_debug_info;
				debug.uint_stats[finalize_key + "pre_hit_count"] = pre_finalize_hit_count;
				debug.uint_stats[finalize_key + "post_hit_count"] = info.hit_count;
				debug.uint_stats[finalize_key + "pre_covered_anchor_count"] = pre_finalize_covered_anchor_count;
				debug.uint_stats[finalize_key + "post_covered_anchor_count"] = info.covered_anchor_count;
				debug.float_stats[finalize_key + "pre_contour_coverage"] = pre_finalize_contour_coverage;
				debug.float_stats[finalize_key + "post_contour_coverage"] = info.contour_coverage;
				debug.float_stats[finalize_key + "pre_penetration_depth"] = pre_finalize_penetration_depth;
				debug.float_stats[finalize_key + "post_penetration_depth"] = info.penetration_depth;
				debug.float_stats[finalize_key + "pre_penetration_objective"] = pre_finalize_penetration_objective;
				debug.float_stats[finalize_key + "post_penetration_objective"] = info.penetration_objective;
				debug.float_stats[finalize_key + "pre_displacement_cost"] = pre_finalize_displacement_cost;
				debug.float_stats[finalize_key + "post_displacement_cost"] = info.displacement_cost;
			}
		}
	}

	void make_response_from_min_hitinfo(
		std::vector<CollisionPair::CollisionPairTemplate>& response_collision_pairs,
		std::vector<uint4>&								   response_collision_indices,
		MeshData<std::vector>*							   host_mesh_data,
		const std::vector<float3>&						   face_normal,
		const std::vector<float3>&						   edge_normal,
		const std::vector<float3>&						   vert_normal,
		const std::vector<PRPMinContourHitInfo>&		   list_min_hit_info)
	{
		const float		global_depth_threshold = get_scene_params().untangling_response_depth;
		const float		response_stiffness = get_scene_params().stiffness_untangling;
		constexpr float response_area_floor = 3e-6f;

		for (uint contour_idx = 0; contour_idx < list_min_hit_info.size(); contour_idx++)
		{
			const auto&										  min_hit_info = list_min_hit_info[contour_idx];
			const auto&										  min_hit_list = min_hit_info.hit_list;
			const float										  depth_threshold = global_depth_threshold;
			std::vector<CollisionPair::CollisionPairTemplate> local_pairs(min_hit_list.size());
			std::vector<uint4>								  local_pairs_indices(min_hit_list.size());
			CpuParallel::parallel_for(
				0,
				min_hit_list.size(),
				[&](const uint index)
				{
					const auto&							 hit = min_hit_list[index];
					CollisionPair::CollisionPairTemplate collision_pair;

					const uint mesh_idx = hit.src_mesh_idx;
					const auto solution = prp_classify_hit_solution(
						hit,
						face_normal,
						edge_normal,
						vert_normal,
						kPrpSolutionProjectionThreshold);
					const uint visual_solution_label = solution.key;
					LUISA_ASSERT(
						visual_solution_label < kPrpResponseVisualLabelCount,
						"PRP visual Solution label {} exceeds the packed region capacity {}.",
						visual_solution_label,
						kPrpResponseVisualLabelCount);
					const uint region_idx = prp_pack_response_region_label(contour_idx, mesh_idx, visual_solution_label);
					float3	   normal = luisa::make_float3(hit.direction[0], hit.direction[1], hit.direction[2]);

					const float penetration_depth = min_hit_info.penetration_depth;
					float		C = std::max(-penetration_depth - 3e-3f, -depth_threshold);

					if (hit.is_vf())
					{
						const uint	vid = hit.get_vid();
						const uint	fid = hit.get_fid();
						const float vert_area = host_mesh_data->sa_rest_vert_area[vid];
						const float face_area = host_mesh_data->sa_rest_face_area[fid];
						const float area = std::max(0.5f * (vert_area + face_area), response_area_floor);
						const float stiff = area * response_stiffness;
						float		k1 = stiff * C;
						float		k2 = stiff;
						if (luisa::isnan(k1) || luisa::isnan(k2) || luisa::isinf(k1) || luisa::isinf(k2))
						{
							LUISA_ERROR("Invalid stiffness computed for VF pair: vid {}, fid {}, area {}, C {}, k1 {}, k2 {}",
								vid, fid, area, C, k1, k2);
							k1 = 0.0f;
							k2 = 0.0f;
						}

						uint4  indices = hit.get_vf_indices();
						float3 face_bary = hit.get_face_bary();
						collision_pair.make_vf_pair(indices, normal, k1, k2, area, face_bary);
						local_pairs_indices[index] = luisa::make_uint4(vid, fid, region_idx, 0);
					}
					else if (hit.is_fv())
					{
						const uint	fid = hit.get_fid();
						const uint	vid = hit.get_vid();
						const float face_area = host_mesh_data->sa_rest_face_area[fid];
						const float vert_area = host_mesh_data->sa_rest_vert_area[vid];
						const float area = std::max(0.5f * (face_area + vert_area), response_area_floor);
						const float stiff = area * response_stiffness;
						float		k1 = stiff * C;
						float		k2 = stiff;

						uint4 fv_indices = hit.get_fv_indices();
						uint4 indices = luisa::make_uint4(fv_indices.w, fv_indices.x, fv_indices.y, fv_indices.z);
						normal = -normal;

						float3 face_bary = hit.get_face_bary();
						collision_pair.make_vf_pair(indices, normal, k1, k2, area, face_bary);
						local_pairs_indices[index] = luisa::make_uint4(vid, fid, region_idx, 1);
					}
					else if (hit.is_ee())
					{
						uint		eid1 = hit.get_eid1();
						uint		eid2 = hit.get_eid2();
						const float edge1_area = host_mesh_data->sa_rest_edge_area[eid1];
						const float edge2_area = host_mesh_data->sa_rest_edge_area[eid2];
						const float area = std::max(0.5f * (edge1_area + edge2_area), response_area_floor);
						const float stiff = area * response_stiffness;
						float		k1 = stiff * C;
						float		k2 = stiff;

						uint4  indices = hit.get_ee_indices();
						float2 edge1_bary = hit.get_edge1_bary();
						float2 edge2_bary = hit.get_edge2_bary();

						collision_pair.make_ee_pair(indices, normal, k1, k2, area, edge1_bary, edge2_bary);
						local_pairs_indices[index] = luisa::make_uint4(eid1, eid2, region_idx, 2);
					}

					local_pairs[index] = collision_pair;
				},
				32);
			response_collision_pairs.insert(response_collision_pairs.end(), local_pairs.begin(), local_pairs.end());
			response_collision_indices.insert(response_collision_indices.end(),
				local_pairs_indices.begin(),
				local_pairs_indices.end());
		}
	};

	void upload_response_pairs_to_gpu(
		luisa::compute::Device&				   device,
		luisa::compute::Stream&				   stream,
		UntanglingData<std::vector>*		   host_untangling_data,
		CollisionData<std::vector>*			   host_collision_data,
		CollisionData<luisa::compute::Buffer>* device_collision_data)
	{
		auto& dst_list = device_collision_data->narrow_phase_list;
		auto& dst_list_indices = device_collision_data->narrow_phase_list_indices;
		auto& src_list = host_untangling_data->target_point_template_pairs;
		auto& src_list_indices = host_untangling_data->target_point_template_pairs_indices;
		auto& host_list = host_collision_data->narrow_phase_list;
		auto& host_list_indices = host_collision_data->narrow_phase_list_indices;

		const uint num_penalty_pairs = host_collision_data->narrow_phase_collision_count[0];
		const uint num_untangling_pairs = static_cast<uint>(src_list.size());
		if (num_untangling_pairs == 0u)
		{
			return;
		}
		if (src_list_indices.size() != num_untangling_pairs)
		{
			LUISA_ERROR("PRP response pair/index count mismatch: {} pairs vs {} indices.",
				num_untangling_pairs,
				src_list_indices.size());
		}

		const uint total_pairs = num_penalty_pairs + num_untangling_pairs;
		if (host_list.size() < total_pairs)
		{
			host_list.resize(total_pairs);
		}
		if (host_list_indices.size() < total_pairs)
		{
			host_list_indices.resize(total_pairs);
		}

		const bool need_resize_list = dst_list.size() < total_pairs;
		const bool need_resize_indices = dst_list_indices.size() < total_pairs;
		if (need_resize_list || need_resize_indices)
		{
			if (num_penalty_pairs != 0)
			{
				stream << dst_list.view(0, num_penalty_pairs).copy_to(host_list.data())
					   << dst_list_indices.view(0, num_penalty_pairs).copy_to(host_list_indices.data())
					   << luisa::compute::synchronize();
			}
			Initializer::dynamic_resize_template(device, device_collision_data->narrow_phase_list, total_pairs, "narrow_phase_list");
			Initializer::dynamic_resize_template(device, device_collision_data->narrow_phase_list_indices, total_pairs, "narrow_phase_list_indices");
			if (num_penalty_pairs != 0)
			{
				stream << device_collision_data->narrow_phase_list.view(0, num_penalty_pairs).copy_from(host_list.data())
					   << device_collision_data->narrow_phase_list_indices.view(0, num_penalty_pairs).copy_from(host_list_indices.data());
			}
		}

		CpuParallel::parallel_for(
			0u,
			num_untangling_pairs,
			[&](const uint pair_idx)
			{
				host_list[num_penalty_pairs + pair_idx] = src_list[pair_idx];
				const uint4 src_index = src_list_indices[pair_idx];
				host_list_indices[num_penalty_pairs + pair_idx] = luisa::make_uint2(src_index.x, src_index.y);
			});

		host_collision_data->narrow_phase_collision_count[0] = total_pairs;

		stream << device_collision_data->narrow_phase_collision_count.view(0, 1).copy_from(host_collision_data->narrow_phase_collision_count.data())
			   << device_collision_data->narrow_phase_list.view(num_penalty_pairs, num_untangling_pairs).copy_from(host_list.data() + num_penalty_pairs)
			   << device_collision_data->narrow_phase_list_indices.view(num_penalty_pairs, num_untangling_pairs).copy_from(host_list_indices.data() + num_penalty_pairs);
	}

	void host_apply_untangling_constraint(
		const std::vector<CollisionPair::CollisionPairTemplate>& target_point_template_pairs,
		const std::vector<uint4>&								 target_point_template_pairs_indices,
		const std::vector<uint2>&								 ef_pair_indices,
		MeshData<std::vector>*									 host_mesh_data,
		UntanglingData<std::vector>*							 host_untangling_data)
	{
		const auto& response_collision_pairs = target_point_template_pairs;
		// x: src_id, y: dst_id, z: packed contour/Solution/source-mesh label, w: type
		const auto& response_collision_indices = target_point_template_pairs_indices;
		if (response_collision_indices.size() != response_collision_pairs.size())
		{
			LUISA_ERROR("PRP response pair/index count mismatch before CCD cull upload: {} pairs vs {} indices.",
				response_collision_pairs.size(),
				response_collision_indices.size());
		}

		// Upload
		{
			PRP_PROFILE_SCOPE("upload_vertex_region_indices");
			const uint	num_verts = host_mesh_data->num_verts;
			const auto& ef_pair_mesh_index = host_untangling_data->ef_pair_mesh_index;
			const auto& ef_pair_contour_index = host_untangling_data->ef_pair_contour_index;
			const uint	num_ef = static_cast<uint>(ef_pair_mesh_index.size());

			if (ef_pair_indices.size() < num_ef || ef_pair_contour_index.size() < num_ef)
			{
				LUISA_ERROR("EF region upload expects {} pair indices and contour indices, got {} EF indices and {} contour indices.",
					num_ef,
					ef_pair_indices.size(),
					ef_pair_contour_index.size());
			}
			const uint					 num_edges = host_mesh_data->num_edges;
			const uint					 num_faces = host_mesh_data->num_faces;
			ConcurrentNestedVector<uint> concurrent_vert_region_indices(num_verts);
			ConcurrentNestedVector<uint> concurrent_edge_region_indices(num_edges);
			ConcurrentNestedVector<uint> concurrent_face_region_indices(num_faces);

			CpuParallel::parallel_for(
				0,
				response_collision_pairs.size(),
				[&](const uint pair_idx)
				{
					const auto& pair = response_collision_pairs[pair_idx];
					const auto	verts = pair.get_indices();
					const uint	region_idx = prp_unpack_response_semantic_region(response_collision_indices[pair_idx].z);
					const uint	type = response_collision_indices[pair_idx].w; // 0 for VF pair, 1 for FV pair, 2 for EE pair
					uint4		region_info =
						type == 0	? luisa::make_uint4(region_idx, region_idx ^ 1, region_idx ^ 1, region_idx ^ 1) // VF
						: type == 1 ? luisa::make_uint4(region_idx ^ 1, region_idx, region_idx, region_idx)			// FV
									: luisa::make_uint4(region_idx, region_idx, region_idx ^ 1, region_idx ^ 1);	// EE
					for (uint ii = 0; ii < 4; ii++)
					{
						const uint vid = verts[ii];
						const uint r_idx = region_info[ii];
						concurrent_vert_region_indices.push_back(vid, r_idx);
					}
				});

			CpuParallel::parallel_for(
				0,
				num_ef,
				[&](const uint pair_idx)
				{
					const uint2 ef = ef_pair_indices[pair_idx];
					const uint	mesh_idx = ef_pair_mesh_index[pair_idx];
					const uint	contour_idx = ef_pair_contour_index[pair_idx];
					const uint	edge_region_idx = 2u * contour_idx + mesh_idx;
					const uint	face_region_idx = edge_region_idx ^ 1u;
					const uint2 edge = host_mesh_data->sa_edges[ef.x];
					const uint3 face = host_mesh_data->sa_faces[ef.y];

					concurrent_vert_region_indices.push_back(edge.x, edge_region_idx);
					concurrent_vert_region_indices.push_back(edge.y, edge_region_idx);
					concurrent_vert_region_indices.push_back(face.x, face_region_idx);
					concurrent_vert_region_indices.push_back(face.y, face_region_idx);
					concurrent_vert_region_indices.push_back(face.z, face_region_idx);
				});

			auto make_unique_and_sort = [](ConcurrentNestedVector<uint>& concurrent_vec)
			{
				CpuParallel::parallel_for(
					0, concurrent_vec.size(),
					[&](const uint idx)
					{
						auto& vec = concurrent_vec.unsafe_row(idx);
						std::sort(vec.begin(), vec.end());
						vec.erase(std::unique(vec.begin(), vec.end()), vec.end());
					},
					32);
			};
			make_unique_and_sort(concurrent_vert_region_indices);
			// TODO: Use intersection set instead of union set?
			CpuParallel::parallel_for(
				0, num_faces,
				[&](const uint fid)
				{
					const uint3	   face = host_mesh_data->sa_faces[fid];
					std::set<uint> face_regions;
					for (uint ii = 0; ii < 3; ii++)
					{
						const uint	vid = face[ii];
						const auto& vert_regions = concurrent_vert_region_indices.unsafe_row(vid);
						face_regions.insert(vert_regions.begin(), vert_regions.end());
					}
					auto& vec = concurrent_face_region_indices.unsafe_row(fid);
					vec.assign(face_regions.begin(), face_regions.end());
				},
				32);
			CpuParallel::parallel_for(
				0, num_edges,
				[&](const uint eid)
				{
					const uint2	   edge = host_mesh_data->sa_edges[eid];
					std::set<uint> edge_regions;
					for (uint ii = 0; ii < 2; ii++)
					{
						const uint	vid = edge[ii];
						const auto& vert_regions = concurrent_vert_region_indices.unsafe_row(vid);
						edge_regions.insert(vert_regions.begin(), vert_regions.end());
					}
					auto& vec = concurrent_edge_region_indices.unsafe_row(eid);
					vec.assign(edge_regions.begin(), edge_regions.end());
				},
				32);

			Initializer::upload_2d_csr_from(host_untangling_data->vert_region_indices_csr, concurrent_vert_region_indices.unsafe_data());
			Initializer::upload_2d_csr_from(host_untangling_data->edge_region_indices_csr, concurrent_edge_region_indices.unsafe_data());
			Initializer::upload_2d_csr_from(host_untangling_data->face_region_indices_csr, concurrent_face_region_indices.unsafe_data());
		}
	}

	void prp_initialize_runtime_candidate_ranges(
		PrpStandardContourRuntime&	 runtime,
		const MeshData<std::vector>* host_mesh_data)
	{
		if (host_mesh_data == nullptr)
			LUISA_ERROR("Mesh data is null while initializing PRP candidate ranges.");
		for (uint mesh_idx = 0u; mesh_idx < 2u; ++mesh_idx)
		{
			const uint object_id = runtime.object_ids[mesh_idx];
			const uint prefix_vid = host_mesh_data->prefix_num_verts[object_id];
			const uint suffix_vid = host_mesh_data->prefix_num_verts[object_id + 1u];
			const uint prefix_eid = host_mesh_data->prefix_num_edges[object_id];
			const uint suffix_eid = host_mesh_data->prefix_num_edges[object_id + 1u];
			const uint prefix_fid = host_mesh_data->prefix_num_faces[object_id];
			const uint suffix_fid = host_mesh_data->prefix_num_faces[object_id + 1u];
			runtime.mesh_candidate_vert_range[mesh_idx] = luisa::make_uint2(prefix_vid, suffix_vid - prefix_vid);
			runtime.mesh_candidate_edge_range[mesh_idx] = luisa::make_uint2(prefix_eid, suffix_eid - prefix_eid);
			runtime.mesh_candidate_face_range[mesh_idx] = luisa::make_uint2(prefix_fid, suffix_fid - prefix_fid);

			runtime.mesh_candidate_verts[mesh_idx].resize(suffix_vid - prefix_vid);
			runtime.mesh_candidate_edges[mesh_idx].resize(suffix_eid - prefix_eid);
			runtime.mesh_candidate_faces[mesh_idx].resize(suffix_fid - prefix_fid);
			std::iota(runtime.mesh_candidate_verts[mesh_idx].begin(), runtime.mesh_candidate_verts[mesh_idx].end(), prefix_vid);
			std::iota(runtime.mesh_candidate_edges[mesh_idx].begin(), runtime.mesh_candidate_edges[mesh_idx].end(), prefix_eid);
			std::iota(runtime.mesh_candidate_faces[mesh_idx].begin(), runtime.mesh_candidate_faces[mesh_idx].end(), prefix_fid);
		}
	}

	void prp_initialize_runtime_boundary_primitives(
		PrpStandardContourRuntime&		   runtime,
		const std::vector<uint>&		   contour,
		const CollisionData<std::vector>*  host_collision_data,
		const UntanglingData<std::vector>* host_untangling_data,
		const std::vector<uint2>&		   ef_indices,
		const std::vector<uint8_t>*		   pair_mesh_flip)
	{
		std::array<std::set<uint>, 2> boundary_verts_set;
		for (uint pi = 0u; pi < contour.size(); ++pi)
		{
			const uint	pair_idx = contour[pi];
			const uint	native_mesh_idx = host_untangling_data->ef_pair_mesh_index[pair_idx];
			const uint	flip = pair_mesh_flip != nullptr && pi < pair_mesh_flip->size()
				? static_cast<uint>((*pair_mesh_flip)[pi])
				: 0u;
			const uint	mesh_idx = native_mesh_idx ^ flip;
			const auto& ef_pair = host_collision_data->narrow_phase_list_ef[pair_idx];
			const uint2 edge = ef_pair.get_edge();
			const uint3 face = ef_pair.get_face();
			boundary_verts_set[mesh_idx].insert(edge.x);
			boundary_verts_set[mesh_idx].insert(edge.y);
			boundary_verts_set[mesh_idx ^ 1u].insert(face.x);
			boundary_verts_set[mesh_idx ^ 1u].insert(face.y);
			boundary_verts_set[mesh_idx ^ 1u].insert(face.z);
		}
		for (uint mesh_idx = 0u; mesh_idx < 2u; ++mesh_idx)
		{
			runtime.boundary_verts[mesh_idx] = { boundary_verts_set[mesh_idx].begin(), boundary_verts_set[mesh_idx].end() };
		}
	}

	void prp_initialize_runtime_combo_state(PrpStandardContourRuntime& runtime)
	{
		runtime.active_combos.resize(runtime.contour_dirs.size());
		std::iota(runtime.active_combos.begin(), runtime.active_combos.end(), 0u);
		runtime.combo_converged.assign(runtime.contour_dirs.size(), 0u);
		runtime.combo_objective_pruned.assign(runtime.contour_dirs.size(), 0u);
		runtime.combo_results.assign(runtime.contour_dirs.size(), PRPMinContourHitInfo{});
		runtime.combo_eval_present.assign(runtime.contour_dirs.size(), 0u);
	}

	void prp_initialize_runtime_adaptive_state(PrpStandardContourRuntime& runtime)
	{
		runtime.boundary_vert_count_src = static_cast<uint>(runtime.boundary_verts[0].size());
		runtime.boundary_vert_count_dst = static_cast<uint>(runtime.boundary_verts[1].size());
		runtime.total_candidate_count = static_cast<uint>(
			runtime.mesh_candidate_verts[0].size() + runtime.mesh_candidate_verts[1].size()
			+ runtime.mesh_candidate_edges[0].size() + runtime.mesh_candidate_edges[1].size()
			+ runtime.mesh_candidate_faces[0].size() + runtime.mesh_candidate_faces[1].size());
		runtime.mesh_diameter_src = runtime.mesh_max_hop_dist[0];
		runtime.mesh_diameter_dst = runtime.mesh_max_hop_dist[1];
		runtime.curr_k_src = prp_compute_adaptive_kring_initial_level(runtime.boundary_vert_count_src, runtime.mesh_diameter_src);
		runtime.curr_k_dst = prp_compute_adaptive_kring_initial_level(runtime.boundary_vert_count_dst, runtime.mesh_diameter_dst);
		runtime.init_k_src = runtime.curr_k_src;
		runtime.init_k_dst = runtime.curr_k_dst;
	}

	void prp_build_runtime_contour_points(
		const std::vector<uint>&		   contour,
		const UntanglingData<std::vector>* host_untangling_data,
		std::vector<float3>&			   contour_points_3D,
		std::vector<float>&				   contour_points_weight)
	{
		contour_points_3D.resize(contour.size());
		contour_points_weight.resize(contour.size());
		for (uint i = 0u; i < contour.size(); ++i)
		{
			const uint pair_idx = contour[i];
			contour_points_3D[i] = host_untangling_data->ef_pair_pos_3D[pair_idx];
			contour_points_weight[i] = host_untangling_data->ef_pair_length[pair_idx];
		}
	}

	void prp_initialize_runtime_identity_and_geometry(
		PrpStandardContourRuntime&		  runtime,
		const uint						  contour_idx,
		const std::vector<uint>&		  contour,
		const CollisionData<std::vector>* host_collision_data,
		UntanglingData<std::vector>*	  host_untangling_data,
		const std::vector<uint2>&		  ef_indices,
		MeshData<std::vector>*			  host_mesh_data,
		SimulationData<std::vector>*	  host_sim_data,
		const std::vector<uint8_t>*		  pair_mesh_flip,
		std::vector<std::array<uint, 2>>* contour_object_ids_out,
		std::vector<std::array<uint, 2>>* contour_topology_component_ids_out)
	{
		runtime.contour_idx = contour_idx;
		runtime.contour = contour;
		if (contour.empty())
			return;
		const uint first_pair_idx = contour.front();
		const auto identity = prp_get_contour_side_identity(
			host_collision_data->narrow_phase_list_ef[first_pair_idx],
			host_untangling_data->ef_pair_mesh_index[first_pair_idx],
			host_mesh_data);
		runtime.object_ids = identity.object_ids;
		runtime.topology_component_ids = identity.topology_component_ids;
		if (contour_object_ids_out != nullptr)
			(*contour_object_ids_out)[contour_idx] = runtime.object_ids;
		if (contour_topology_component_ids_out != nullptr)
			(*contour_topology_component_ids_out)[contour_idx] = runtime.topology_component_ids;

		prp_initialize_runtime_candidate_ranges(runtime, host_mesh_data);
		prp_initialize_runtime_boundary_primitives(runtime, contour, host_collision_data, host_untangling_data, ef_indices, pair_mesh_flip);
		calculate_hop_dist_and_geom_dist_in_mesh(
			runtime.vert_to_boundary_hop_dist,
			runtime.vert_to_boundary_geom_dist,
			runtime.mesh_max_hop_dist,
			runtime.contour_min_geom_dist_to_another_contour,
			runtime.contour_max_geom_dist_to_another_contour,
			host_mesh_data,
			runtime.boundary_verts,
			runtime.contour_idx,
			runtime.object_ids);

		const auto& loop_pairs_is_boundary = host_untangling_data->intersection_contours_info[contour_idx].loop_pairs_is_boundary;
		runtime.need_flip = !prp_does_contour_need_antiflip(loop_pairs_is_boundary);
		std::vector<float3> contour_points_3D;
		std::vector<float>	contour_points_weight;
		prp_build_runtime_contour_points(contour, host_untangling_data, contour_points_3D, contour_points_weight);
		runtime.contour_dirs = get_raycasting_direction(
			contour_points_3D,
			contour_points_weight,
			runtime.boundary_verts,
			host_sim_data,
			host_mesh_data);
		prp_initialize_runtime_combo_state(runtime);
		prp_initialize_runtime_adaptive_state(runtime);
	}

	void prp_build_candidates_for_contours(
		std::vector<PrpStandardContourRuntime>& runtimes,
		const std::vector<uint>&				contour_indices,
		const MeshData<std::vector>*			host_mesh_data,
		const SimulationData<std::vector>*		host_sim_data)
	{
		CpuParallel::parallel_for(
			0u,
			static_cast<uint>(contour_indices.size()),
			[&](const uint index)
			{
				const uint contour_idx = contour_indices[index];
				auto&	   runtime = runtimes.at(contour_idx);
				build_standard_candidates_at_level(runtime, runtime.curr_k_src, runtime.curr_k_dst, host_mesh_data, host_sim_data);
			},
			1u);
	}

	void prp_compute_host_normals(
		const MeshData<std::vector>*	   host_mesh_data,
		const SimulationData<std::vector>* host_sim_data,
		std::vector<float3>&			   face_normal,
		std::vector<float3>&			   edge_normal,
		std::vector<float3>&			   vert_normal)
	{
		const auto safe_normalize = [](const float3 normal)
		{
			const float length_squared = luisa::dot(normal, normal);
			return length_squared > 1e-20f ? normal / std::sqrt(length_squared) : luisa::make_float3(0.0f);
		};
		const auto& face_area = host_mesh_data->sa_rest_face_area;
		CpuParallel::parallel_for(0u, host_mesh_data->num_faces,
			[&](const uint fid)
			{
				const auto	 face = host_mesh_data->sa_faces[fid];
				const float3 normal = safe_normalize(luisa::cross(
					host_sim_data->sa_x[face.y] - host_sim_data->sa_x[face.x],
					host_sim_data->sa_x[face.z] - host_sim_data->sa_x[face.x]));
				const uint	 mesh_id = host_mesh_data->sa_face_mesh_id[fid];
				face_normal[fid] = (host_mesh_data->sa_mesh_orientation[mesh_id] == 0u ? 1.0f : -1.0f) * normal;
			});
		CpuParallel::parallel_for(0u, host_mesh_data->num_edges,
			[&](const uint eid)
			{
				float3 normal_sum = luisa::make_float3(0.0f);
				for (const uint fid : host_mesh_data->edge_adj_faces_ext[eid])
					normal_sum += face_area[fid] * face_normal[fid];
				edge_normal[eid] = safe_normalize(normal_sum);
			});
		CpuParallel::parallel_for(0u, host_mesh_data->num_verts,
			[&](const uint vid)
			{
				float3 normal_sum = luisa::make_float3(0.0f);
				for (const uint fid : host_mesh_data->vert_adj_faces[vid])
					normal_sum += face_area[fid] * face_normal[fid];
				vert_normal[vid] = safe_normalize(normal_sum);
			});
	}

	void prp_build_vertex_boundary_flags(
		const CollisionData<std::vector>*  host_collision_data,
		const UntanglingData<std::vector>* host_untangling_data,
		const MeshData<std::vector>*	   host_mesh_data,
		std::vector<std::vector<uint>>&	   vert_in_boundary_flag)
	{
		const uint					 num_pairs = host_collision_data->narrow_phase_collision_count[1];
		ConcurrentNestedVector<uint> flags(host_mesh_data->num_verts);
		CpuParallel::parallel_for(0u, num_pairs,
			[&](const uint pair_idx)
			{
				const auto& pair = host_collision_data->narrow_phase_list_ef[pair_idx];
				if (pair.get_is_loop_vertex())
					return;
				const uint	mesh_idx = host_untangling_data->ef_pair_mesh_index[pair_idx];
				const uint	contour_idx = host_untangling_data->ef_pair_contour_index[pair_idx];
				const uint	edge_side_flag = 2u * contour_idx + mesh_idx;
				const uint	face_side_flag = edge_side_flag ^ 1u;
				const uint2 edge = pair.get_edge();
				const uint3 face = pair.get_face();
				flags.push_back(edge.x, edge_side_flag);
				flags.push_back(edge.y, edge_side_flag);
				flags.push_back(face.x, face_side_flag);
				flags.push_back(face.y, face_side_flag);
				flags.push_back(face.z, face_side_flag);
			});
		vert_in_boundary_flag = std::move(flags.unsafe_data());
		CpuParallel::parallel_for(0u, host_mesh_data->num_verts,
			[&](const uint vid)
			{
				auto& row = vert_in_boundary_flag[vid];
				std::sort(row.begin(), row.end());
				row.erase(std::unique(row.begin(), row.end()), row.end());
			});
	}

	void prp_write_contour_eval_debug(
		const uint					contour_idx,
		const PRPMinContourHitInfo& min_hit_info,
		const float					raycast_ms,
		const uint					final_pair_count,
		const std::array<uint, 2>&	object_ids,
		const std::array<uint, 2>&	topology_component_ids,
		const ContourType			contour_type)
	{
		auto&			  debug = get_scene_params().prp_debug_info;
		const std::string contour_key = "prp.c" + std::to_string(contour_idx) + ".eval.";
		debug.uint_stats[contour_key + "best_combo"] = min_hit_info.combo_idx;
		debug.uint_stats[contour_key + "hit_count"] = min_hit_info.hit_count;
		debug.uint_stats[contour_key + "unculled_hit_count"] = min_hit_info.unculled_hit_count;
		debug.float_stats[contour_key + "penetration_depth"] = min_hit_info.penetration_depth;
		debug.float_stats[contour_key + "penetration_volume"] = min_hit_info.penetration_volume;
		debug.float_stats[contour_key + "penetration_objective"] = min_hit_info.penetration_objective;
		debug.float_stats[contour_key + "response_objective"] = min_hit_info.penetration_objective;
		debug.float_stats[contour_key + "coverage"] = min_hit_info.contour_coverage;
		debug.uint_stats[contour_key + "covered_anchor_count"] = min_hit_info.covered_anchor_count;
		debug.uint_stats[contour_key + "boundary_anchor_count"] = min_hit_info.boundary_anchor_count;
		debug.float_stats[contour_key + "raycast_ms"] = raycast_ms;
		debug.float_stats[contour_key + "displacement_cost"] = min_hit_info.displacement_cost;

		const std::string meta_key = "prp.c" + std::to_string(contour_idx) + ".meta.";
		debug.uint_stats[meta_key + "contour_ef_pair_count"] = final_pair_count;
		debug.uint_stats[meta_key + "object_id0"] = object_ids[0];
		debug.uint_stats[meta_key + "object_id1"] = object_ids[1];
		debug.bool_stats[meta_key + "is_self_intersection"] = prp_contour_roles_share_topology(object_ids, topology_component_ids);
		debug.uint_stats["prp.c" + std::to_string(contour_idx) + ".contour_type"] = static_cast<uint>(contour_type);

		debug.float_stats["prp.c" + std::to_string(contour_idx) + ".opt.best_direction.x"] = min_hit_info.separation_direction.x;
		debug.float_stats["prp.c" + std::to_string(contour_idx) + ".opt.best_direction.y"] = min_hit_info.separation_direction.y;
		debug.float_stats["prp.c" + std::to_string(contour_idx) + ".opt.best_direction.z"] = min_hit_info.separation_direction.z;
	}

	// Temporary [DET] determinism-trace helpers (bisecting synthetic run-to-run
	// divergence); gated on prp_debug and slated for removal after diagnosis.
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

	void host_resolve_intersections_PRP(luisa::compute::Device& device,
		luisa::compute::Stream&									stream,
		CollisionData<luisa::compute::Buffer>*					device_collision_data,
		CollisionData<std::vector>*								host_collision_data,
		UntanglingData<std::vector>*							host_untangling_data,
		MeshData<std::vector>*									host_mesh_data,
		SimulationData<std::vector>*							host_sim_data)
	{
		// LUISA_INFO("Resolving intersections (Intersection Resolver 2)...");
		PRP_PROFILE_RESET();
		{
			PRP_PROFILE_SCOPE("total");
			if (get_scene_params().prp_debug)
			{
				auto& dbg = get_scene_params().prp_debug_info;
				dbg.bool_stats.clear();
				dbg.uint_stats.clear();
				dbg.float_stats.clear();
			}
			uint num_pairs = host_collision_data->narrow_phase_collision_count[1];

			auto& ef_list = host_collision_data->narrow_phase_list_ef;
			auto& ef_indices = host_collision_data->narrow_phase_list_ef_indices;

			const uint num_verts = host_mesh_data->num_verts;
			const uint num_edges = host_mesh_data->num_edges;
			const uint num_faces = host_mesh_data->num_faces;

			// Init normals
			const std::vector<float>& sa_rest_face_area = host_mesh_data->sa_rest_face_area;
			const std::vector<float>& sa_rest_edge_area = host_mesh_data->sa_rest_edge_area;
			const std::vector<float>& sa_rest_vert_area = host_mesh_data->sa_rest_vert_area;
			std::vector<float3>		  face_normal(num_faces);
			std::vector<float3>		  edge_normal(num_edges);
			std::vector<float3>		  vert_normal(num_verts);
			{
				PRP_PROFILE_SCOPE("utils.compute_normals");
				prp_compute_host_normals(host_mesh_data, host_sim_data, face_normal, edge_normal, vert_normal);
			}

			{
				PRP_PROFILE_SCOPE("build_intersection_contours");
				intersection_contour_construction_from_ext_adjacent(host_untangling_data, host_collision_data);
			}

			{
				uint64_t h = 1469598103934665603ull;
				for (const auto& contour : host_untangling_data->intersection_contours)
					det_trace_detail::hash_pod_vector(h, contour);
				det_trace_detail::hash_pod_vector(h, host_untangling_data->ef_pair_mesh_index);
				det_trace_detail::trace("P2_contours", h);
			}

			const auto& contours = host_untangling_data->intersection_contours;
			const uint	num_contours = contours.size();
			const auto& ef_pair_mesh_index = host_untangling_data->ef_pair_mesh_index;

			const uint num_loop_pairs = host_untangling_data->loop_pairs_vertex.size();
			auto&	   intersection_contours_info = host_untangling_data->intersection_contours_info;
			intersection_contours_info.resize(num_contours);

			{
				PRP_PROFILE_SCOPE("classify_contours");
				std::vector<luisa::ubyte> pair_is_boundary(num_pairs, false);
				CpuParallel::parallel_for(0, num_pairs,
					[&](const uint pair_idx)
					{
						const uint eid = ef_indices[pair_idx].x;
						pair_is_boundary[pair_idx] = host_mesh_data->edge_adj_faces_ext[eid].size() <= 1 ? 1u : 0u;
					});
				for (uint contour_idx = 0; contour_idx < num_contours; contour_idx++)
				{
					const auto contour_type = classify_contour_type(
						ef_list, ef_pair_mesh_index, host_untangling_data->ef_pair_adj_pairs_ext,
						pair_is_boundary, intersection_contours_info[contour_idx].loop_pairs_is_boundary, contours[contour_idx], contour_idx);
					auto& contour_info = intersection_contours_info[contour_idx];
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
			}

			uint classified_contour_count = std::count_if(intersection_contours_info.begin(), intersection_contours_info.end(),
				[](const auto& info)
				{ return info.contour_type != ContourType::Undifined; });

			{
				uint64_t h = 1469598103934665603ull;
				det_trace_detail::hash_u64(h, classified_contour_count);
				for (const auto& info : intersection_contours_info)
				{
					det_trace_detail::hash_u64(h, static_cast<uint64_t>(info.contour_type));
					det_trace_detail::hash_u64(h, info.mesh_is_closed[0]);
					det_trace_detail::hash_u64(h, info.mesh_is_closed[1]);
					det_trace_detail::hash_pod_vector(h, info.loop_pairs_is_boundary);
				}
				det_trace_detail::trace("P3_classify", h);
			}

			LUISA_INFO("  #EF pairs = {}, #contours = {}, #classified_contours = {}, #loop_pairs = {}",
				num_pairs, num_contours, classified_contour_count, num_loop_pairs);

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

			auto& response_collision_pairs = host_untangling_data->target_point_template_pairs;
			auto& response_collision_indices = host_untangling_data->target_point_template_pairs_indices;
			response_collision_pairs.clear();
			response_collision_indices.clear();

			RayCasting::BVHCache						face_bvh_cache;
			RayCasting::BVHCache						edge_bvh_cache;
			RayCasting::BVHCache						reverse_face_bvh_cache;
			std::unordered_map<uint, std::vector<uint>> full_body_face_candidates_cache;
			std::unordered_map<uint, std::vector<uint>> full_body_edge_candidates_cache;
			std::mutex									full_body_face_candidates_mutex;
			std::mutex									full_body_edge_candidates_mutex;

			using PrpDebugInfo = SceneParams::PRPDebugInfo;
			auto merge_prp_debug_info = [](PrpDebugInfo& dst, const PrpDebugInfo& src)
			{
				for (const auto& [key, value] : src.bool_stats)
					dst.bool_stats[key] = value;
				for (const auto& [key, value] : src.uint_stats)
					dst.uint_stats[key] = value;
				for (const auto& [key, value] : src.float_stats)
					dst.float_stats[key] = value;
			};

			// The region should connected to the invalid boundary verts, and have the corrorsponding mesh lable
			std::vector<std::vector<uint>> vert_in_boundary_flag(num_verts);
			{
				PRP_PROFILE_SCOPE("utils.compute_vert_in_boundary_flag");
				prp_build_vertex_boundary_flags(host_collision_data, host_untangling_data, host_mesh_data, vert_in_boundary_flag);
			}

			const auto& contours_for_raycasting = contours;
			const uint	num_contours_total = static_cast<uint>(contours_for_raycasting.size());

			std::vector<PRPMinContourHitInfo> list_min_hit_info(num_contours_total);
			std::vector<double>				  contour_raycast_times(num_contours_total, 0.0);
			std::vector<uint>				  final_contour_pair_count(num_contours_total, 0u);
			for (uint contour_idx = 0u; contour_idx < num_contours_total; ++contour_idx)
				final_contour_pair_count[contour_idx] = static_cast<uint>(contours_for_raycasting[contour_idx].size());
			std::vector<std::array<uint, 2>> contour_object_ids(num_contours_total,
				{ std::numeric_limits<uint>::max(), std::numeric_limits<uint>::max() });
			std::vector<std::array<uint, 2>> contour_topology_component_ids(num_contours_total,
				{ std::numeric_limits<uint>::max(), std::numeric_limits<uint>::max() });
			std::vector<PrpDebugInfo>		 contour_debug_infos(get_scene_params().prp_debug ? num_contours_total : 0u);
			auto							 write_contour_eval_debug = [&](const uint contour_idx)
			{
				if (!get_scene_params().prp_debug || get_scene_params().should_contour_skip(contour_idx))
					return;
				if (contour_idx < contour_debug_infos.size())
					merge_prp_debug_info(get_scene_params().prp_debug_info, contour_debug_infos[contour_idx]);
				prp_write_contour_eval_debug(
					contour_idx,
					list_min_hit_info[contour_idx],
					static_cast<float>(contour_raycast_times[contour_idx]),
					final_contour_pair_count[contour_idx],
					contour_object_ids[contour_idx],
					contour_topology_component_ids[contour_idx],
					host_untangling_data->intersection_contours_info[contour_idx].contour_type);
			};

			auto find_contours_hits_batched = [&]() -> bool
			{
				PRP_PROFILE_SCOPE("find_contours_hits_batched");

				// Phase D-1: reuse adjacency buckets across combo evaluations to avoid repeated outer-vector construction.
				std::vector<std::vector<uint>> shared_vert_contains_pairs;
				std::vector<std::vector<uint>> shared_edge_contains_pairs;
				std::vector<std::vector<uint>> shared_face_contains_pairs;
				std::vector<std::vector<uint>> shared_fv_dst_vert_contains_pairs;
				{
					PRP_PROFILE_SCOPE("init_shared_adjacency_buckets");
					shared_vert_contains_pairs.resize(num_verts);
					shared_edge_contains_pairs.resize(num_edges);
					shared_face_contains_pairs.resize(num_faces);
					shared_fv_dst_vert_contains_pairs.resize(num_verts);
				}

				auto get_full_body_candidates = [&](auto& cache, std::mutex& mutex, const auto& prefix_counts, const uint object_id) -> const std::vector<uint>&
				{
					std::lock_guard<std::mutex> lock(mutex);
					auto						it = cache.find(object_id);
					if (it != cache.end())
						return it->second;

					const uint prefix = prefix_counts[object_id];
					const uint suffix = prefix_counts[object_id + 1u];
					auto&	   candidates = cache[object_id];
					candidates.resize(suffix - prefix);
					std::iota(candidates.begin(), candidates.end(), prefix);
					return candidates;
				};

				auto build_standard_runtime = [&](const uint contour_idx, const std::vector<uint>& contour, PrpStandardContourRuntime& runtime,
												  const std::vector<uint8_t>* pair_mesh_flip = nullptr)
				{
					runtime.contour_idx = contour_idx;
					runtime.contour = contour;
					if (contour.empty())
						return;
					prp_initialize_runtime_identity_and_geometry(
						runtime, contour_idx, contour, host_collision_data, host_untangling_data,
						ef_indices, host_mesh_data, host_sim_data, pair_mesh_flip,
						&contour_object_ids, &contour_topology_component_ids);
					// CPU-only cached BVH attachment for the three ray query groups.
					const uint	target_object_id = runtime.object_ids[1];
					const uint	source_object_id = runtime.object_ids[0];
					const auto& target_full_body_faces = get_full_body_candidates(full_body_face_candidates_cache, full_body_face_candidates_mutex, host_mesh_data->prefix_num_faces, target_object_id);
					const auto& target_full_body_edges = get_full_body_candidates(full_body_edge_candidates_cache, full_body_edge_candidates_mutex, host_mesh_data->prefix_num_edges, target_object_id);
					const auto& source_full_body_faces = get_full_body_candidates(full_body_face_candidates_cache, full_body_face_candidates_mutex, host_mesh_data->prefix_num_faces, source_object_id);
					const uint	candidate_face_hash = RayCasting::BVHCache::vector_hash({ 0u, target_object_id, 0u });
					runtime.face_bvh = face_bvh_cache.get_or_build(target_full_body_faces, host_sim_data->sa_x, host_mesh_data->sa_faces, candidate_face_hash);
					const uint candidate_edge_hash = RayCasting::BVHCache::vector_hash({ 1u, target_object_id, 0u });
					runtime.edge_bvh = edge_bvh_cache.get_or_build(target_full_body_edges, host_sim_data->sa_x, host_mesh_data->sa_edges, candidate_edge_hash);
					const uint source_face_hash = RayCasting::BVHCache::vector_hash({ 3u, source_object_id, 0u });
					runtime.reverse_face_bvh = reverse_face_bvh_cache.get_or_build(source_full_body_faces, host_sim_data->sa_x, host_mesh_data->sa_faces, source_face_hash);
				};

				// Evaluate all contours with the same combo in a batched manner
				auto evaluate_batched_contours_of_single_combo = [&](std::vector<PrpStandardContourRuntime>& runtimes,
																	 const std::vector<uint>&				 runtime_indices,
																	 const uint								 combo_idx,
																	 const bool								 use_partial) -> std::vector<PRPMinContourHitInfo>
				{
					PRP_PROFILE_SCOPE("evaluate_batched_contours_of_single_combo");
					luisa::Clock combo_clock;
					struct RawHitCounts
					{
						uint vf = 0u;
						uint ee = 0u;
						uint fv = 0u;
					};

					std::vector<PRPMinContourHitInfo> contour_min_results;

					// Size scratch storage by the active batch to avoid num_contours_total × primitive CSR blowup.
					const uint num_contours_process = static_cast<uint>(runtime_indices.size());

					std::vector<std::vector<RayCasting::HitInfo>> raw_hits_by_slot;
					std::vector<RawHitCounts>					  raw_hit_counts_by_slot;
					{
						PRP_PROFILE_SCOPE("evaluate_batched_contours_of_single_combo.init_scratch");
						contour_min_results.resize(num_contours_total);
						raw_hits_by_slot.resize(num_contours_process);
						raw_hit_counts_by_slot.resize(num_contours_process);
					}

					// Group candidate requests by BVH pointer with a small linear table; only 1–2 keys are typical.
					std::vector<std::pair<const RayCasting::CachedBVH*, std::vector<RayCasting::BatchedVertRayRequest>>> vf_requests_by_bvh;
					std::vector<std::pair<const RayCasting::CachedBVH*, std::vector<RayCasting::BatchedEdgeRayRequest>>> ee_requests_by_bvh;
					std::vector<std::pair<const RayCasting::CachedBVH*, std::vector<RayCasting::BatchedVertRayRequest>>> fv_requests_by_bvh;
					{
						PRP_PROFILE_SCOPE("evaluate_batched_contours_of_single_combo.prepare_requests");

						// Pre-reserve shared BVH request vectors once per batch to avoid repeated reallocations.
						{
							std::unordered_map<const RayCasting::CachedBVH*, uint> vf_total, ee_total, fv_total;
							const uint											   n_contours = static_cast<uint>(runtime_indices.size());
							for (uint local_idx = 0u; local_idx < n_contours; ++local_idx)
							{
								const auto& runtime = runtimes[runtime_indices[local_idx]];
								const auto& src_verts = use_partial ? runtime.partial_candidate_verts[0] : runtime.mesh_candidate_verts[0];
								const auto& src_edges = use_partial ? runtime.partial_candidate_edges[0] : runtime.mesh_candidate_edges[0];
								const auto& dst_verts = use_partial ? runtime.partial_candidate_verts[1] : runtime.mesh_candidate_verts[1];
								const auto& target_faces = use_partial ? runtime.partial_candidate_faces[1] : runtime.mesh_candidate_faces[1];
								const auto& target_edges = use_partial ? runtime.partial_candidate_edges[1] : runtime.mesh_candidate_edges[1];
								const auto& source_faces = use_partial ? runtime.partial_candidate_faces[0] : runtime.mesh_candidate_faces[0];
								if (!target_faces.empty())
									vf_total[runtime.face_bvh] += static_cast<uint>(src_verts.size());
								if (!target_edges.empty())
									ee_total[runtime.edge_bvh] += static_cast<uint>(src_edges.size());
								if (!source_faces.empty())
									fv_total[runtime.reverse_face_bvh] += static_cast<uint>(dst_verts.size());
							}
							for (const auto& [bvh, n] : vf_total)
								prp_find_or_insert_bvh_requests(vf_requests_by_bvh, bvh).reserve(n);
							for (const auto& [bvh, n] : ee_total)
								prp_find_or_insert_bvh_requests(ee_requests_by_bvh, bvh).reserve(n);
							for (const auto& [bvh, n] : fv_total)
								prp_find_or_insert_bvh_requests(fv_requests_by_bvh, bvh).reserve(n);
						}

						for (uint local_idx = 0u; local_idx < num_contours_process; ++local_idx)
						{
							const uint	 contour_idx = runtime_indices[local_idx];
							const auto&	 runtime = runtimes[contour_idx];
							const float3 dir = runtime.contour_dirs[combo_idx];
							const float	 dir_len2 = luisa::dot(dir, dir);
							const auto&	 src_verts = use_partial ? runtime.partial_candidate_verts[0] : runtime.mesh_candidate_verts[0];
							const auto&	 src_edges = use_partial ? runtime.partial_candidate_edges[0] : runtime.mesh_candidate_edges[0];
							const auto&	 dst_verts = use_partial ? runtime.partial_candidate_verts[1] : runtime.mesh_candidate_verts[1];
							const auto&	 target_faces = use_partial ? runtime.partial_candidate_faces[1] : runtime.mesh_candidate_faces[1];
							const auto&	 target_edges = use_partial ? runtime.partial_candidate_edges[1] : runtime.mesh_candidate_edges[1];
							const auto&	 source_faces = use_partial ? runtime.partial_candidate_faces[0] : runtime.mesh_candidate_faces[0];

							const uint2				 vf_target_range = prp_compute_dense_span(target_faces, runtime.mesh_candidate_face_range[1]);
							const uint2				 ee_target_range = prp_compute_dense_span(target_edges, runtime.mesh_candidate_edge_range[1]);
							const uint2				 fv_source_range = prp_compute_dense_span(source_faces, runtime.mesh_candidate_face_range[0]);
							const std::vector<uint>* vf_target_hop_dist = use_partial ? &runtime.vert_to_boundary_hop_dist[1] : nullptr;
							const std::vector<uint>* ee_target_hop_dist = use_partial ? &runtime.vert_to_boundary_hop_dist[1] : nullptr;
							const std::vector<uint>* fv_target_hop_dist = use_partial ? &runtime.vert_to_boundary_hop_dist[0] : nullptr;
							const uint				 vf_target_max_hop = use_partial ? runtime.curr_k_dst : std::numeric_limits<uint>::max();
							const uint				 ee_target_max_hop = use_partial ? runtime.curr_k_dst : std::numeric_limits<uint>::max();
							const uint				 fv_target_max_hop = use_partial ? runtime.curr_k_src : std::numeric_limits<uint>::max();
							const float2			 target_edge_projection = prp_compute_edge_projection_bounds(target_edges, dir, *host_mesh_data, *host_sim_data);

							if (!target_faces.empty())
							{
								auto& vf_requests = prp_find_or_insert_bvh_requests(vf_requests_by_bvh, runtime.face_bvh);
								for (const uint vid : src_verts)
								{
									vf_requests.push_back({ local_idx, vid, dir, RayCasting::BatchedVertRayOutputType::VF, vf_target_range, 1.0f, vf_target_hop_dist, vf_target_max_hop, &target_faces });
								}
							}

							if (!target_edges.empty())
							{
								auto& ee_requests = prp_find_or_insert_bvh_requests(ee_requests_by_bvh, runtime.edge_bvh);
								for (const uint eid : src_edges)
								{
									const float ee_ray_max_dist = prp_compute_ee_ray_max_dist(eid, target_edge_projection, dir, dir_len2, *host_mesh_data, *host_sim_data);
									ee_requests.push_back({ local_idx, eid, dir, ee_target_range, ee_ray_max_dist, target_edge_projection.x, ee_target_hop_dist, ee_target_max_hop, &target_edges });
								}
							}

							if (!source_faces.empty())
							{
								auto& fv_requests = prp_find_or_insert_bvh_requests(fv_requests_by_bvh, runtime.reverse_face_bvh);
								for (const uint vid : dst_verts)
								{
									fv_requests.push_back({ local_idx, vid, dir, RayCasting::BatchedVertRayOutputType::FV, fv_source_range, 1.0f, fv_target_hop_dist, fv_target_max_hop, &source_faces });
								}
							}
						}
					}

					// Perform raycasting (grouped by local slot; group_count == batch_size)
					{
						PRP_PROFILE_SCOPE("raycasting");
						{
							PRP_PROFILE_SCOPE("raycasting.raycast_VF");
							for (auto& [bvh, requests] : vf_requests_by_bvh)
							{
								auto grouped_hits = bvh->ray_cast_from_verts_batched(requests, host_mesh_data->sa_faces, num_contours_process, 1.0f);
								for (uint local_idx = 0u; local_idx < num_contours_process; ++local_idx)
								{
									auto& dst = raw_hits_by_slot[local_idx];
									auto& src = grouped_hits[local_idx];
									raw_hit_counts_by_slot[local_idx].vf += static_cast<uint>(src.size());
									dst.insert(dst.end(), src.begin(), src.end());
								}
							}
						}
						{
							PRP_PROFILE_SCOPE("raycasting.raycast_EE");
							for (auto& [bvh, requests] : ee_requests_by_bvh)
							{
								auto grouped_hits = bvh->ray_cast_from_edges_batched(
									requests, host_mesh_data->sa_edges, host_mesh_data->sa_edges, num_contours_process, 1.0f);
								for (uint local_idx = 0u; local_idx < num_contours_process; ++local_idx)
								{
									auto& dst = raw_hits_by_slot[local_idx];
									auto& src = grouped_hits[local_idx];
									raw_hit_counts_by_slot[local_idx].ee += static_cast<uint>(src.size());
									dst.insert(dst.end(), src.begin(), src.end());
								}
							}
						}
						{
							PRP_PROFILE_SCOPE("raycasting.raycast_FV");
							for (auto& [bvh, requests] : fv_requests_by_bvh)
							{
								auto grouped_hits = bvh->ray_cast_from_verts_batched(requests, host_mesh_data->sa_faces, num_contours_process, 1.0f);
								for (uint local_idx = 0u; local_idx < num_contours_process; ++local_idx)
								{
									auto& dst = raw_hits_by_slot[local_idx];
									auto& src = grouped_hits[local_idx];
									raw_hit_counts_by_slot[local_idx].fv += static_cast<uint>(src.size());
									dst.insert(dst.end(), src.begin(), src.end());
								}
							}
						}
					}

					// Build contour hit adjacency (offsets/counts indexed by local slot)
					std::vector<uint>				 slot_offsets(num_contours_process, 0u);
					std::vector<uint>				 slot_hits_count(num_contours_process, 0u);
					uint							 packed_hit_count = 0u;
					std::vector<RayCasting::HitInfo> packed_hits;
					std::vector<uint>				 packed_hit_contour_idx;
					std::vector<luisa::ubyte>		 packed_component_is_boundary_adj;
					std::vector<uint>				 packed_hit_component_id;
					{
						PRP_PROFILE_SCOPE("evaluate_batched_contours_of_single_combo.pack_hits");
						for (uint local_idx = 0u; local_idx < num_contours_process; ++local_idx)
						{
							slot_offsets[local_idx] = packed_hit_count;
							slot_hits_count[local_idx] = static_cast<uint>(raw_hits_by_slot[local_idx].size());
							packed_hit_count += slot_hits_count[local_idx];
						}

						// Pack hits contiguously for cache-friendly adjacency; retain global contour indices for filtering.
						packed_hits.resize(packed_hit_count);
						packed_hit_contour_idx.assign(packed_hit_count, 0u);
						packed_component_is_boundary_adj.assign(packed_hit_count, false);
						packed_hit_component_id.assign(packed_hit_count, -1u);
						for (uint local_idx = 0u; local_idx < num_contours_process; ++local_idx)
						{
							const uint	offset = slot_offsets[local_idx];
							const auto& contour_hits = raw_hits_by_slot[local_idx];
							if (contour_hits.empty())
								continue;
							const uint global_contour_idx = runtime_indices[local_idx];
							std::copy(contour_hits.begin(), contour_hits.end(), packed_hits.begin() + offset);
							std::fill(packed_hit_contour_idx.begin() + offset, packed_hit_contour_idx.begin() + offset + contour_hits.size(), global_contour_idx);
						}
					}

					// Mark boundary adjacent hits
					{
						PRP_PROFILE_SCOPE("evaluate_batched_contours_of_single_combo.mark_boundary_adjacency");
						CpuParallel::parallel_for(
							0u, packed_hit_count,
							[&](const uint hit_idx)
							{
								const uint contour_idx = packed_hit_contour_idx[hit_idx];
								packed_component_is_boundary_adj[hit_idx] = is_hit_adj_contour(packed_hits[hit_idx], vert_in_boundary_flag, contour_idx);
							},
							64u);
					}

					// Build adjacency for each hit
					std::vector<std::vector<uint>> pair_adj_pairs(packed_hit_count);
					if (packed_hit_count > 1u)
					{
						PRP_PROFILE_SCOPE("evaluate_batched_contours_of_single_combo.build_hit_adjacency");
						// Per-slot CSR keeps each adjacency row limited to hits from the current contour and clears only touched rows for reuse.
						std::vector<std::vector<uint>>& vert_contains_pairs = shared_vert_contains_pairs;
						std::vector<std::vector<uint>>& edge_contains_pairs = shared_edge_contains_pairs;
						std::vector<std::vector<uint>>& face_contains_pairs = shared_face_contains_pairs;
						std::vector<std::vector<uint>>& fv_dst_vert_contains_pairs = shared_fv_dst_vert_contains_pairs;

						std::vector<uint> touched_v;
						std::vector<uint> touched_e;
						std::vector<uint> touched_f;
						std::vector<uint> touched_fv_v;

						for (uint slot = 0u; slot < num_contours_process; ++slot)
						{
							const uint base = slot_offsets[slot];
							const uint n = slot_hits_count[slot];
							if (n == 0u)
								continue;

							touched_v.clear();
							touched_e.clear();
							touched_f.clear();
							touched_fv_v.clear();

							// Sequential populate (cheap vs queries) — keeps a per-slot touched list so we can reset in O(touched) instead of O(num_primitives).
							for (uint k = 0u; k < n; ++k)
							{
								const uint	hit_idx = base + k;
								const auto& hit = packed_hits[hit_idx];
								if (hit.is_vf())
								{
									const uint vid = hit.get_vid();
									if (vert_contains_pairs[vid].empty())
										touched_v.push_back(vid);
									vert_contains_pairs[vid].push_back(hit_idx);
								}
								else if (hit.is_ee())
								{
									const uint eid1 = hit.get_eid1();
									const uint eid2 = hit.get_eid2();
									if (edge_contains_pairs[eid1].empty())
										touched_e.push_back(eid1);
									edge_contains_pairs[eid1].push_back(hit_idx);
								}
								else if (hit.is_fv())
								{
									const uint fid = hit.get_fid();
									if (face_contains_pairs[fid].empty())
										touched_f.push_back(fid);
									face_contains_pairs[fid].push_back(hit_idx);

									const uint dst_vid = hit.get_vid();
									if (dst_vid < fv_dst_vert_contains_pairs.size())
									{
										if (fv_dst_vert_contains_pairs[dst_vid].empty())
											touched_fv_v.push_back(dst_vid);
										fv_dst_vert_contains_pairs[dst_vid].push_back(hit_idx);
									}
								}
							}

							CpuParallel::parallel_for(
								0u, n,
								[&](const uint k)
								{
									const uint hit_idx = base + k;
									auto	   adj_hits = find_adjacent_hits(
										packed_hits,
										hit_idx,
										host_mesh_data,
										vert_contains_pairs,
										edge_contains_pairs,
										face_contains_pairs,
										1u, 0u, nullptr, 0u);

									pair_adj_pairs[hit_idx] = std::move(adj_hits);
								},
								32u);

							// Volumetric solid interior hit adjacency for solid bodies (rigid/affine, tetrahedral)
							if (get_scene_params().PRP_solid_interior_adjacency)
							{
								uint solid_interior_connections = 0u;
								auto connect_mutual = [&](const uint h_a, const uint h_b)
								{
									if (h_a == h_b)
										return;
									if (std::find(pair_adj_pairs[h_a].begin(), pair_adj_pairs[h_a].end(), h_b) == pair_adj_pairs[h_a].end())
									{
										pair_adj_pairs[h_a].push_back(h_b);
										solid_interior_connections++;
									}
									if (std::find(pair_adj_pairs[h_b].begin(), pair_adj_pairs[h_b].end(), h_a) == pair_adj_pairs[h_b].end())
									{
										pair_adj_pairs[h_b].push_back(h_a);
										solid_interior_connections++;
									}
								};

								const bool	all_solid = get_scene_params().PRP_solid_interior_adjacency_all_meshes;
								const float eps_n = get_scene_params().PRP_solid_interior_adjacency_normal_eps;
								const float eps_d = get_scene_params().PRP_solid_interior_adjacency_dist_eps;

								// 1. VF hits sharing same source vertex vid traversing solid target mesh
								for (const uint vid : touched_v)
								{
									const auto&		  candidates = vert_contains_pairs[vid];
									std::vector<uint> vf_hits;
									vf_hits.reserve(candidates.size());
									for (const uint idx : candidates)
									{
										if (idx >= base && idx < base + n && packed_hits[idx].is_vf())
											vf_hits.push_back(idx);
									}
									if (vf_hits.size() >= 2u)
									{
										std::sort(vf_hits.begin(), vf_hits.end(), [&](const uint a, const uint b)
											{ return packed_hits[a].dist < packed_hits[b].dist; });
										for (size_t p = 0; p + 1 < vf_hits.size(); ++p)
										{
											const uint h_a = vf_hits[p];
											const uint h_b = vf_hits[p + 1];
											const uint fid_a = packed_hits[h_a].get_fid();
											const uint fid_b = packed_hits[h_b].get_fid();
											if (host_mesh_data->sa_face_mesh_id[fid_a] == host_mesh_data->sa_face_mesh_id[fid_b])
											{
												const uint v0 = host_mesh_data->sa_faces[fid_a].x;
												const uint mat = host_mesh_data->sa_vert_mesh_type[v0];
												const bool is_solid = all_solid || (mat == uint(Material::MaterialType::Rigid) || mat == uint(Material::MaterialType::Tetrahedral));
												if (is_solid)
												{
													const float dist_a = packed_hits[h_a].dist;
													const float dist_b = packed_hits[h_b].dist;
													if (dist_b - dist_a > eps_d)
													{
														const float3 r = packed_hits[h_a].get_direction();
														const float	 dot_a = luisa::dot(face_normal[fid_a], r);
														const float	 dot_b = luisa::dot(face_normal[fid_b], r);
														if (dot_a < -eps_n && dot_b > eps_n)
														{
															connect_mutual(h_a, h_b);
														}
													}
												}
											}
										}
									}
								}

								// 2. FV hits sharing same destination vertex vid traversing solid source mesh
								for (const uint dst_vid : touched_fv_v)
								{
									const auto&		  candidates = fv_dst_vert_contains_pairs[dst_vid];
									std::vector<uint> fv_hits;
									fv_hits.reserve(candidates.size());
									for (const uint idx : candidates)
									{
										if (idx >= base && idx < base + n && packed_hits[idx].is_fv())
											fv_hits.push_back(idx);
									}
									if (fv_hits.size() >= 2u)
									{
										std::sort(fv_hits.begin(), fv_hits.end(), [&](const uint a, const uint b)
											{ return packed_hits[a].dist < packed_hits[b].dist; });
										for (size_t p = 0; p + 1 < fv_hits.size(); ++p)
										{
											const uint h_a = fv_hits[p];
											const uint h_b = fv_hits[p + 1];
											const uint fid_a = packed_hits[h_a].get_fid();
											const uint fid_b = packed_hits[h_b].get_fid();
											if (host_mesh_data->sa_face_mesh_id[fid_a] == host_mesh_data->sa_face_mesh_id[fid_b])
											{
												const uint v0 = host_mesh_data->sa_faces[fid_a].x;
												const uint mat = host_mesh_data->sa_vert_mesh_type[v0];
												const bool is_solid = all_solid || (mat == uint(Material::MaterialType::Rigid) || mat == uint(Material::MaterialType::Tetrahedral));
												if (is_solid)
												{
													const float dist_a = packed_hits[h_a].dist;
													const float dist_b = packed_hits[h_b].dist;
													if (dist_b - dist_a > eps_d)
													{
														const float3 r = packed_hits[h_a].get_direction();
														const float	 dot_a = luisa::dot(face_normal[fid_a], r);
														const float	 dot_b = luisa::dot(face_normal[fid_b], r);
														if (dot_a > eps_n && dot_b < -eps_n)
														{
															connect_mutual(h_a, h_b);
														}
													}
												}
											}
										}
									}
								}

								// 3. EE hits sharing same source edge eid1 traversing solid target mesh
								for (const uint eid1 : touched_e)
								{
									const auto&		  candidates = edge_contains_pairs[eid1];
									std::vector<uint> ee_hits;
									ee_hits.reserve(candidates.size());
									for (const uint idx : candidates)
									{
										if (idx >= base && idx < base + n && packed_hits[idx].is_ee())
											ee_hits.push_back(idx);
									}
									if (ee_hits.size() >= 2u)
									{
										std::sort(ee_hits.begin(), ee_hits.end(), [&](const uint a, const uint b)
											{ return packed_hits[a].dist < packed_hits[b].dist; });
										for (size_t p = 0; p + 1 < ee_hits.size(); ++p)
										{
											const uint h_a = ee_hits[p];
											const uint h_b = ee_hits[p + 1];
											const uint eid2_a = packed_hits[h_a].get_eid2();
											const uint eid2_b = packed_hits[h_b].get_eid2();
											const uint v0_a = host_mesh_data->sa_edges[eid2_a].x;
											const uint v0_b = host_mesh_data->sa_edges[eid2_b].x;
											if (host_mesh_data->sa_vert_mesh_id[v0_a] == host_mesh_data->sa_vert_mesh_id[v0_b])
											{
												const uint mat = host_mesh_data->sa_vert_mesh_type[v0_a];
												const bool is_solid = all_solid || (mat == uint(Material::MaterialType::Rigid) || mat == uint(Material::MaterialType::Tetrahedral));
												if (is_solid)
												{
													const float dist_a = packed_hits[h_a].dist;
													const float dist_b = packed_hits[h_b].dist;
													if (dist_b - dist_a > eps_d)
													{
														const float3 r = packed_hits[h_a].get_direction();
														const float	 dot_a = luisa::dot(edge_normal[eid2_a], r);
														const float	 dot_b = luisa::dot(edge_normal[eid2_b], r);
														if (dot_a < -eps_n && dot_b > eps_n)
														{
															connect_mutual(h_a, h_b);
														}
													}
												}
											}
										}
									}
								}
								if (get_scene_params().prp_debug && solid_interior_connections > 0u)
								{
									LUISA_INFO("PRP Solid Interior Adjacency: established {} volumetric connections across solid interior.", solid_interior_connections);
								}
							}

							// Adjacency must be symmetric; report the first asymmetric edge instead of repairing it here.
							for (uint k = 0u; k < n; ++k)
							{
								const uint src = base + k;
								for (const uint dst : pair_adj_pairs[src])
								{
									if (dst < base || dst >= base + n)
									{
										LUISA_ERROR("PRP hit adjacency escaped its slot: slot={}, local_src={}, src_global={}, dst_global={}, slot=[{}, {}).",
											slot, k, src, dst, base, base + n);
									}
									const auto& rev = pair_adj_pairs[dst];
									if (std::find(rev.begin(), rev.end(), src) == rev.end())
									{
										const auto& a = packed_hits[src];
										const auto& b = packed_hits[dst];
										auto		type_code = [](const RayCasting::HitInfo& h) -> uint
										{
											return h.is_vf() ? 0u : (h.is_ee() ? 1u : 2u);
										};
										LUISA_ERROR("PRP hit adjacency is not dual: slot={}, contour={}, src_global={}, dst_global={}, "
													"src(type={}, src_id={}, dst_id={}), dst(type={}, src_id={}, dst_id={}).",
											slot, packed_hit_contour_idx[src], src, dst,
											type_code(a), a.src_id, a.dst_id,
											type_code(b), b.src_id, b.dst_id);
									}
								}
							}

							for (uint vid : touched_v)
								vert_contains_pairs[vid].clear();
							for (uint eid : touched_e)
								edge_contains_pairs[eid].clear();
							for (uint fid : touched_f)
								face_contains_pairs[fid].clear();
							for (uint vid : touched_fv_v)
								fv_dst_vert_contains_pairs[vid].clear();
						}
					}

					// Parallel cluster culling
					std::vector<luisa::ubyte> packed_hit_valid(packed_hit_count, 0u);
					{
						PRP_PROFILE_SCOPE("parallel_cluster_culling");

						CpuParallel::parallel_for(
							0u, num_contours_process,
							[&](const uint local_idx)
							{
								const uint offset = slot_offsets[local_idx];
								const uint hit_count = slot_hits_count[local_idx];
								if (hit_count == 0u)
									return;
								const uint contour_idx = runtime_indices[local_idx];
								auto&	   runtime = runtimes[contour_idx];

								std::vector<luisa::ubyte> component_is_boundary_adj(hit_count, false);
								for (uint local_idx = 0u; local_idx < hit_count; ++local_idx)
									component_is_boundary_adj[local_idx] = packed_component_is_boundary_adj[offset + local_idx];

								std::vector<luisa::ubyte>	   hit_visited(hit_count, false);
								std::vector<std::vector<uint>> candidate_output_hists;

								for (uint seed_local_idx = 0u; seed_local_idx < hit_count; ++seed_local_idx)
								{
									if (hit_visited[seed_local_idx])
										continue;

									std::vector<uint> output_hits;
									std::vector<uint> to_visit;
									output_hits.clear();
									to_visit.clear();
									to_visit.push_back(seed_local_idx);
									hit_visited[seed_local_idx] = true;

									bool hit_has_boundary_adj = false;
									while (!to_visit.empty())
									{
										const uint local_hit_idx = to_visit.back();
										to_visit.pop_back();
										output_hits.push_back(local_hit_idx);
										hit_has_boundary_adj = hit_has_boundary_adj || component_is_boundary_adj[local_hit_idx];
										for (const uint global_adj_idx : pair_adj_pairs[offset + local_hit_idx])
										{
											const uint local_adj_idx = global_adj_idx - offset;
											if (local_adj_idx < hit_count && !hit_visited[local_adj_idx])
											{
												hit_visited[local_adj_idx] = true;
												to_visit.push_back(local_adj_idx);
											}
										}
									}
									if (!hit_has_boundary_adj)
									{
										continue;
									}
									else
									{
										candidate_output_hists.push_back(output_hits);
										const uint component_idx = static_cast<uint>(candidate_output_hists.size()) - 1u;
										for (const uint local_hit_idx : output_hits)
											packed_hit_component_id[offset + local_hit_idx] = component_idx;
										// break;
									}
								}

								if (!candidate_output_hists.empty())
								{
									// Coverage + locality selection (Phase B). EF-pair coverage with per-mesh source/target vertex sets.

									const auto& contour_ef_pairs = runtime.contour;
									const uint	ef_count = static_cast<uint>(contour_ef_pairs.size());
									const uint	num_verts = host_mesh_data->num_verts;

									const uint max_hop = kPrpLocalityMaxHop;

									auto min_target_hop = [&](const RayCasting::HitInfo& h) -> uint
									{
										const uint dst_mesh = h.src_mesh_idx ^ 1u;
										uint	   best = std::numeric_limits<uint>::max();
										visit_prp_hit_target_vertices(h, [&](const uint vid)
											{ best = std::min(best, runtime.boundary_hop_distance(dst_mesh, vid)); });
										return best;
									};

									auto component_anchor_coverage = [&](const std::vector<uint>& comp_hits, uint& covered_ef, uint& kept_count)
									{
										kept_count = 0u;
										std::array<std::vector<luisa::ubyte>, 2> src_set = { std::vector<luisa::ubyte>(num_verts, 0), std::vector<luisa::ubyte>(num_verts, 0) };
										std::array<std::vector<luisa::ubyte>, 2> dst_set = { std::vector<luisa::ubyte>(num_verts, 0), std::vector<luisa::ubyte>(num_verts, 0) };
										for (const uint local_hit_idx : comp_hits)
										{
											const auto& h = packed_hits[offset + local_hit_idx];
											if (max_hop != std::numeric_limits<uint>::max() && min_target_hop(h) > max_hop)
												continue;
											kept_count++;
											const uint sm = h.src_mesh_idx;
											visit_prp_hit_source_vertices(h, [&](const uint vid)
												{ if (vid < num_verts) src_set[sm][vid] = 1; });
											visit_prp_hit_target_vertices(h, [&](const uint vid)
												{ if (vid < num_verts) dst_set[sm ^ 1u][vid] = 1; });
										}
										covered_ef = 0u;
										for (uint pi = 0u; pi < ef_count; ++pi)
										{
											const uint	pair_idx = contour_ef_pairs[pi];
											const uint	mesh_idx = host_untangling_data->ef_pair_mesh_index[pair_idx];
											const auto& ef_pair = host_collision_data->narrow_phase_list_ef[pair_idx];
											const uint2 edge = ef_pair.get_edge();
											const uint3 face = ef_pair.get_face();
											const auto& e_set_s = src_set[mesh_idx];
											const auto& e_set_d = dst_set[mesh_idx];
											const auto& f_set_s = src_set[mesh_idx ^ 1u];
											const auto& f_set_d = dst_set[mesh_idx ^ 1u];
											bool		edge_ok = e_set_s[edge.x] || e_set_s[edge.y] || e_set_d[edge.x] || e_set_d[edge.y];
											bool		face_ok = f_set_s[face.x] || f_set_s[face.y] || f_set_s[face.z]
												|| f_set_d[face.x] || f_set_d[face.y] || f_set_d[face.z];
											if (edge_ok && face_ok)
												++covered_ef;
										}
									};

									auto component_canonical_key = [&](const std::vector<uint>& comp_hits) -> uint64_t
									{
										uint64_t key = std::numeric_limits<uint64_t>::max();
										for (const uint local_hit_idx : comp_hits)
										{
											const auto&	   h = packed_hits[offset + local_hit_idx];
											const uint	   type = h.is_vf() ? 0u : (h.is_ee() ? 1u : 2u);
											const uint64_t hit_key =
												(static_cast<uint64_t>(type) << 62u)
												| (static_cast<uint64_t>(h.src_id & 0x3fffffffu) << 31u)
												| static_cast<uint64_t>(h.dst_id & 0x7fffffffu);
											key = std::min(key, hit_key);
										}
										return key;
									};

									uint	 best_comp_idx = 0u;
									uint	 best_cov = 0u;
									uint	 best_hits = 0u;
									uint64_t best_key = std::numeric_limits<uint64_t>::max();
									bool	 best_found = false;
									for (uint ci = 0u; ci < candidate_output_hists.size(); ++ci) // For each candidate (Cluster)
									{
										const auto& comp_hits = candidate_output_hists[ci];
										uint		covered_ef = 0u;
										uint		kept_hit_count = 0u;
										component_anchor_coverage(comp_hits, covered_ef, kept_hit_count);
										const uint	   cov = covered_ef;
										const uint	   nhit = kept_hit_count;
										const uint64_t nkey = component_canonical_key(comp_hits);
										// Prefer maximal thinned coverage, then larger hit count, then the smallest source-vertex key.
										const bool better = !best_found
											|| (cov > best_cov)
											|| (cov == best_cov && nhit > best_hits)
											|| (cov == best_cov && nhit == best_hits && nkey < best_key);
										if (better)
										{
											best_comp_idx = ci;
											best_cov = cov;
											best_hits = nhit;
											best_key = nkey;
											best_found = true;
										}
									}

									// Locality-thin the selected component by target-to-contour hop distance.
									const auto& best_comp = candidate_output_hists[best_comp_idx];
									for (const uint local_hit_idx : best_comp)
									{
										const uint	global_idx = offset + local_hit_idx;
										const auto& h = packed_hits[global_idx];
										if (max_hop == std::numeric_limits<uint>::max() || min_target_hop(h) <= max_hop)
											packed_hit_valid[global_idx] = true;
									}
								}
							},
							1u);
					}

					// Assemble contour results
					{
						PRP_PROFILE_SCOPE("evaluate_batched_contours_of_single_combo.assemble_contour_results");
						for (uint local_idx = 0u; local_idx < num_contours_process; ++local_idx)
						{
							const uint contour_idx = runtime_indices[local_idx];
							auto&	   runtime = runtimes[contour_idx];
							auto&	   info = contour_min_results[contour_idx];

							info.contour_idx = contour_idx;
							info.combo_idx = combo_idx;
							info.separation_direction = runtime.contour_dirs[combo_idx];
							info.evaluated_kring_src = use_partial ? runtime.curr_k_src : std::numeric_limits<uint>::max();
							info.evaluated_kring_dst = use_partial ? runtime.curr_k_dst : std::numeric_limits<uint>::max();
							info.full_mesh_evaluated = !use_partial
								|| (runtime.curr_k_src > runtime.mesh_diameter_src
									&& runtime.curr_k_dst > runtime.mesh_diameter_dst);
							info.unculled_hit_count = raw_hit_counts_by_slot[local_idx].vf + raw_hit_counts_by_slot[local_idx].ee + raw_hit_counts_by_slot[local_idx].fv;

							const uint offset = slot_offsets[local_idx];
							const uint count = slot_hits_count[local_idx];
							if (count == 0u)
								continue;
							std::vector<RayCasting::HitInfo> hits_info;
							{
								PRP_PROFILE_SCOPE("assemble_contour_results.build_hit_list");
								hits_info.reserve(count);
								for (uint packed_local_idx = 0u; packed_local_idx < count; ++packed_local_idx)
								{
									const uint global_hit_idx = offset + packed_local_idx;
									if (packed_hit_valid[global_hit_idx])
										hits_info.push_back(packed_hits[global_hit_idx]);
								}
							}
							if (hits_info.empty())
								continue;

							const auto* metric_hits_info = &hits_info;
							prp_fill_combo_eval_info(info, runtime, *metric_hits_info,
								host_untangling_data, host_collision_data, host_mesh_data,
								sa_rest_face_area, sa_rest_edge_area, sa_rest_vert_area,
								face_normal, edge_normal, vert_normal);

							if (get_scene_params().prp_debug)
							{
								uint					 component_count = 0u;
								uint					 boundary_attached_component_count = 0u;
								std::unordered_set<uint> seen_components;
								for (uint pi = 0u; pi < count; ++pi)
								{
									const uint gi = offset + pi;
									const uint cid = packed_hit_component_id[gi];
									if (cid != -1u && seen_components.find(cid) == seen_components.end())
									{
										seen_components.insert(cid);
										component_count++;
										if (packed_component_is_boundary_adj[gi])
											boundary_attached_component_count++;
									}
								}

								if (get_scene_params().prp_debug)
								{
									auto&			  debug = get_scene_params().prp_debug_info;
									const std::string prefix = "prp.c" + std::to_string(contour_idx) + ".";
									debug.uint_stats[prefix + "raw_hit_count"] = count;
									debug.uint_stats[prefix + "valid_hit_count"] = static_cast<uint>(metric_hits_info->size());
									debug.uint_stats[prefix + "component_count"] = component_count;
									debug.uint_stats[prefix + "boundary_attached_component_count"] = boundary_attached_component_count;
									debug.float_stats[prefix + "coverage"] = info.contour_coverage;
									debug.uint_stats[prefix + "covered_anchor_count"] = info.covered_anchor_count;
									debug.uint_stats[prefix + "boundary_anchor_count"] = info.boundary_anchor_count;
								}
							}

							info.hit_list = std::move(hits_info);
						}
					}

					const double combo_ms = combo_clock.toc();
					for (const uint runtime_idx : runtime_indices)
						contour_raycast_times[runtime_idx] += combo_ms / float(runtime_indices.size());

					return contour_min_results;
				};

				auto evaluate_current_runtime_directions = [&](std::vector<PrpStandardContourRuntime>& runtimes,
															   const std::vector<uint>&				   target_contours,
															   const bool							   print_combo_summary)
				{
					lcs::evaluate_current_runtime_directions_template(
						runtimes, target_contours, print_combo_summary,
						[&](const std::vector<uint>& contour_indices)
						{
							lcs::prp_build_candidates_for_contours(runtimes, contour_indices, host_mesh_data, host_sim_data);
						},
						[&](const std::vector<uint>& contour_indices,
							const uint				 combo_idx,
							const bool				 use_adaptive) -> std::vector<PRPMinContourHitInfo>
						{
							return evaluate_batched_contours_of_single_combo(runtimes, contour_indices, combo_idx, use_adaptive);
						});
				};

				PRPContourMergeAdjacency pre_finalize_merge_adjacency(num_contours_total);
				std::vector<float>		 pre_finalize_penetration_area(num_contours_total, 0.0f);
				uint					 max_kring_expansion_src = 0u, max_kring_expansion_dst = 0u;
				uint					 unfinished_contours = 0u;

				auto run_batched_runtime_evaluation = [&](std::vector<PrpStandardContourRuntime>& contour_runtime_results,
														  const std::vector<uint>&				  target_contours,
														  const bool							  accumulate_stats)
				{
					if (target_contours.empty())
						return;

					// Accumulate before batch runtimes are released; direction trials do not count twice.
					auto accumulate_kring_summary = [&]()
					{
						constexpr uint ku = std::numeric_limits<uint>::max();
						uint		   max_exp_s = 0u, max_exp_d = 0u, n_unfinished = 0u;
						for (const uint cid : target_contours)
						{
							const auto& rt = contour_runtime_results[cid];
							if (rt.curr_k_src != ku && rt.init_k_src != ku)
								max_exp_s = std::max(max_exp_s,
									rt.curr_k_src >= rt.init_k_src ? rt.curr_k_src - rt.init_k_src : 0u);
							if (rt.curr_k_dst != ku && rt.init_k_dst != ku)
								max_exp_d = std::max(max_exp_d,
									rt.curr_k_dst >= rt.init_k_dst ? rt.curr_k_dst - rt.init_k_dst : 0u);
							if (!rt.finished)
								++n_unfinished;
						}
						max_kring_expansion_src = std::max(max_kring_expansion_src, max_exp_s);
						max_kring_expansion_dst = std::max(max_kring_expansion_dst, max_exp_d);
						unfinished_contours += n_unfinished;
					};

					PRP_PROFILE_SCOPE("batched_direction_standard");
					evaluate_current_runtime_directions(contour_runtime_results, target_contours, true);
					accumulate_kring_summary();
					prp_optimize_runtime_directions(
						contour_runtime_results,
						target_contours,
						host_mesh_data,
						host_sim_data,
						get_scene_params().PRP_direction_optimization_iterations,
						[&](std::vector<PrpStandardContourRuntime>& trial_runtimes,
							const std::vector<uint>&				trial_contours)
						{
							evaluate_current_runtime_directions(trial_runtimes, trial_contours, false);
						});
					{
						PRP_PROFILE_SCOPE("batched_direction_standard.select_runtime_best_results");
						select_runtime_best_results(
							list_min_hit_info,
							contour_debug_infos,
							contour_runtime_results,
							target_contours,
							accumulate_stats);
					}
					{
						PRP_PROFILE_SCOPE("batched_direction_standard.capture_pre_finalize_contour_merge");
						// Record which contours the selected support reaches before merging responses.
						prp_capture_pre_finalize_contour_merge_data(
							pre_finalize_merge_adjacency,
							pre_finalize_penetration_area,
							list_min_hit_info,
							target_contours,
							vert_in_boundary_flag,
							contour_object_ids);
					}
					{
						PRP_PROFILE_SCOPE("batched_direction_standard.finalize_selected_best_hit_infos");
						prp_finalize_selected_best_hit_infos(
							list_min_hit_info,
							contour_runtime_results,
							target_contours,
							host_untangling_data,
							host_collision_data,
							host_mesh_data,
							sa_rest_face_area,
							sa_rest_edge_area,
							sa_rest_vert_area,
							face_normal,
							edge_normal,
							vert_normal);
					}
				};

				std::vector<PrpStandardContourRuntime> contour_runtime_results(num_contours_total);
				std::vector<uint>					   active_contours;
				active_contours.reserve(num_contours_total);
				for (uint contour_idx = 0u; contour_idx < num_contours_total; ++contour_idx)
				{
					if (get_scene_params().should_contour_skip(contour_idx))
						continue;
					active_contours.push_back(contour_idx);
				}
				const uint configured_batch_size = get_scene_params().PRP_cpu_contour_batch_size;
				if (configured_batch_size == 0u)
					LUISA_ERROR("PRP_cpu_contour_batch_size must be positive.");
				const size_t contour_batch_size = static_cast<size_t>(configured_batch_size);
				const size_t contour_batch_count = active_contours.empty()
					? 0u
					: 1u + (active_contours.size() - 1u) / contour_batch_size;
				LUISA_INFO("  CPU PRP contour evaluation: {} contours, {} batches (up to {} contours/batch).",
					active_contours.size(), contour_batch_count, contour_batch_size);
				for (size_t batch_begin = 0u, batch_idx = 0u;
					batch_begin < active_contours.size();
					batch_begin += std::min(contour_batch_size, active_contours.size() - batch_begin), ++batch_idx)
				{
					const size_t	  batch_count = std::min(contour_batch_size, active_contours.size() - batch_begin);
					std::vector<uint> contour_batch(
						active_contours.begin() + static_cast<std::ptrdiff_t>(batch_begin),
						active_contours.begin() + static_cast<std::ptrdiff_t>(batch_begin + batch_count));
					{
						PRP_PROFILE_SCOPE("build_standard_runtime");
						CpuParallel::parallel_for(
							0u, static_cast<uint>(contour_batch.size()),
							[&](const uint idx)
							{
								const uint contour_idx = contour_batch[idx];
								auto&	   runtime = contour_runtime_results[contour_idx];
								build_standard_runtime(contour_idx, contours_for_raycasting[contour_idx], runtime);
							},
							1u);
					}

					run_batched_runtime_evaluation(contour_runtime_results, contour_batch, batch_idx != 0u);

					// Release per-contour mesh, adjacency, and hit storage before constructing the next batch.
					for (const uint contour_idx : contour_batch)
						contour_runtime_results[contour_idx] = PrpStandardContourRuntime{};
				}
				LUISA_INFO("  CPU PRP batches complete: {}/{}, {} contours; [Adaptive Krings] max expansion src {}/dst {}, {} unfinished.",
					contour_batch_count, contour_batch_count, active_contours.size(),
					max_kring_expansion_src, max_kring_expansion_dst, unfinished_contours);

				{
					PRP_PROFILE_SCOPE("suppress_adjacent_contours");
					const auto suppressed = prp_select_pre_finalize_contours_to_suppress(
						pre_finalize_merge_adjacency,
						pre_finalize_penetration_area,
						active_contours);

					for (const uint cid : suppressed)
					{
						final_contour_pair_count[cid] = 0u;
						contour_raycast_times[cid] = 0.0;
						// Preserve suppressed-contour debug stats for CPU/GPU parity; suppression only clears the response.
						prp_clear_suppressed_contour_response(list_min_hit_info[cid], cid);
					}
					LUISA_INFO("  Suppressed contours: {}", suppressed);
				}

				return true;
			};

			luisa::Clock total_clock;
			total_clock.tic();
			{
				const bool used_global_standard_batch = find_contours_hits_batched();
			}
			const auto total_cost = total_clock.toc();
			if (get_scene_params().prp_debug)
				for (uint contour_idx = 0; contour_idx < num_contours_total; contour_idx++)
					write_contour_eval_debug(contour_idx);

			LUISA_INFO("End of ray-casting for all contours.");
			std::vector<uint>		 contour_indices;
			std::vector<uint>		 contour_ray_counts;
			std::vector<std::string> formatted_objectives;
			std::vector<std::string> formatted_costs;
			std::vector<std::string> formatted_depths;
			for (uint contour_idx = 0; contour_idx < num_contours_total; contour_idx++)
			{
				if (get_scene_params().should_contour_skip(contour_idx))
					continue;
				contour_indices.push_back(contour_idx);
				contour_ray_counts.push_back(list_min_hit_info[contour_idx].hit_count);
				formatted_objectives.push_back(fmt::format("{:.2e}", prp_response_objective(list_min_hit_info[contour_idx])));
				formatted_costs.push_back(fmt::format("{:.2e}", list_min_hit_info[contour_idx].displacement_cost));
				formatted_depths.push_back(fmt::format("{:.2e}", list_min_hit_info[contour_idx].penetration_depth));
			}
			LUISA_INFO("  Active contours = {}", contour_indices);
			LUISA_INFO("  => Ray Count = {}", contour_ray_counts);
			LUISA_INFO("  => Response Objectives = {}", fmt::join(formatted_objectives, ", "));
			LUISA_INFO("  => Displacement Costs = {}", fmt::join(formatted_costs, ", "));
			LUISA_INFO("  => Penetration Depths = {}", fmt::join(formatted_depths, ", "));

			LUISA_INFO("  CPU Total ray-casting time for all contours: {:.2f} ms", total_cost);

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
					{
						const char* names[] = { "contour_idx", "combo_idx", "sep_dir", "depth", "volume", "objective", "area", "cost", "hit_count", "unculled", "kring_src", "kring_dst", "eligible", "terminal" };
						uint64_t	fh[14];
						for (int k = 0; k < 14; k++)
							fh[k] = 1469598103934665603ull;
						for (const auto& info : list_min_hit_info)
						{
							det_trace_detail::hash_u64(fh[0], info.contour_idx);
							det_trace_detail::hash_u64(fh[1], info.combo_idx);
							det_trace_detail::hash_bytes(fh[2], &info.separation_direction, sizeof(float3));
							det_trace_detail::hash_bytes(fh[3], &info.penetration_depth, sizeof(float));
							det_trace_detail::hash_bytes(fh[4], &info.penetration_volume, sizeof(float));
							det_trace_detail::hash_bytes(fh[5], &info.penetration_objective, sizeof(float));
							det_trace_detail::hash_bytes(fh[6], &info.penetration_area, sizeof(float));
							det_trace_detail::hash_bytes(fh[7], &info.displacement_cost, sizeof(float));
							det_trace_detail::hash_u64(fh[8], info.hit_count);
							det_trace_detail::hash_u64(fh[9], info.unculled_hit_count);
							det_trace_detail::hash_u64(fh[10], info.evaluated_kring_src);
							det_trace_detail::hash_u64(fh[11], info.evaluated_kring_dst);
							det_trace_detail::hash_u64(fh[12], info.response_eligible);
							det_trace_detail::hash_u64(fh[13], info.no_response_terminal);
						}
						if (get_scene_params().prp_debug)
							for (int k = 0; k < 14; k++)
								LUISA_INFO("[DET] P4f_{} {:016x}", names[k], fh[k]);
					}
					uint64_t h2 = 1469598103934665603ull;
					for (const auto& info : list_min_hit_info)
						det_trace_detail::hash_pod_vector(h2, info.hit_list);
					det_trace_detail::trace("P4b_eval_hitlists", h2);
					{
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
					face_normal,
					edge_normal,
					vert_normal,
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
		}
		PRP_PROFILE_EXPORT_JSON("prp_profile_latest_cpu");
	}

} // namespace lcs
