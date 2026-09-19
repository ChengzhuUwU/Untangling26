#include "cached_bvh.h"
#include "CollisionDetector/distance.hpp"
#include "Utils/cpu_parallel.h"
#include <cmath>
#include <limits>

namespace lcs
{
	namespace RayCasting
	{
		CachedBVH::BVHNode::BVHNode()
			: bmin(0.0f)
			, bmax(0.0f)
			, left(-1)
			, right(-1)
			, start(-1)
			, count(0)
		{
		}

		void CachedBVH::compute_swept_aabb(const float3& a, const float3& b,
			const float3& direction, const float max_distance, float3& out_min, float3& out_max)
		{
			const float3 a_end = a + direction * max_distance;
			const float3 b_end = b + direction * max_distance;
			out_min = luisa::min(luisa::min(a, b), luisa::min(a_end, b_end));
			out_max = luisa::max(luisa::max(a, b), luisa::max(a_end, b_end));
		}

		// Build BVH for faces (uint3)
		void CachedBVH::build(const std::vector<uint>& candidate_faces,
			const std::vector<float3>&				   sa_x,
			const std::vector<uint3>&				   sa_faces)
		{
			ordered_element_indices.assign(candidate_faces.begin(), candidate_faces.end());
			sa_x_ptr = &sa_x;
			primitive_type = PrimitiveType::Face;

			const int elem_count = static_cast<int>(ordered_element_indices.size());
			nodes.clear();
			nodes.reserve(elem_count * 2 + 1);

			// Compute centroids keyed by global face id
			uint max_id = 0u;
			for (uint id : ordered_element_indices)
				max_id = std::max(max_id, id);
			std::vector<float3> centroids(max_id + 1);
			for (int i = 0; i < elem_count; ++i)
			{
				uint   fid = ordered_element_indices[i];
				uint3  f = sa_faces[fid];
				float3 v0 = sa_x[f.x];
				float3 v1 = sa_x[f.y];
				float3 v2 = sa_x[f.z];
				centroids[fid] = (v0 + v1 + v2) / 3.0f;
			}

			auto compute_bounds = [&](int start, int end, float3& out_min, float3& out_max)
			{
				uint   fid0 = ordered_element_indices[start];
				uint3  f0 = sa_faces[fid0];
				float3 v0 = sa_x[f0.x];
				float3 v1 = sa_x[f0.y];
				float3 v2 = sa_x[f0.z];
				out_min = luisa::min(v0, luisa::min(v1, v2));
				out_max = luisa::max(v0, luisa::max(v1, v2));
				for (int i = start + 1; i < end; ++i)
				{
					uint   fid = ordered_element_indices[i];
					uint3  f = sa_faces[fid];
					float3 a = sa_x[f.x];
					float3 b = sa_x[f.y];
					float3 c = sa_x[f.z];
					out_min = luisa::min(out_min, luisa::min(a, luisa::min(b, c)));
					out_max = luisa::max(out_max, luisa::max(a, luisa::max(b, c)));
				}
			};

			build_internal(centroids, compute_bounds, elem_count);
		}

		// Build BVH for edges (uint2)
		void CachedBVH::build(const std::vector<uint>& candidate_edges,
			const std::vector<float3>&				   sa_x,
			const std::vector<uint2>&				   sa_edges)
		{
			ordered_element_indices.assign(candidate_edges.begin(), candidate_edges.end());
			sa_x_ptr = &sa_x;
			primitive_type = PrimitiveType::Edge;

			const int elem_count = static_cast<int>(ordered_element_indices.size());
			nodes.clear();
			nodes.reserve(elem_count * 2 + 1);

			// Compute centroids keyed by global edge id
			uint max_id = 0u;
			for (uint id : ordered_element_indices)
				max_id = std::max(max_id, id);
			std::vector<float3> centroids(max_id + 1);
			for (int i = 0; i < elem_count; ++i)
			{
				uint   eid = ordered_element_indices[i];
				uint2  e = sa_edges[eid];
				float3 v0 = sa_x[e.x];
				float3 v1 = sa_x[e.y];
				centroids[eid] = (v0 + v1) * 0.5f;
			}

			auto compute_bounds = [&](int start, int end, float3& out_min, float3& out_max)
			{
				uint   eid0 = ordered_element_indices[start];
				uint2  e0 = sa_edges[eid0];
				float3 v0 = sa_x[e0.x];
				float3 v1 = sa_x[e0.y];
				out_min = luisa::min(v0, v1);
				out_max = luisa::max(v0, v1);
				for (int i = start + 1; i < end; ++i)
				{
					uint   eid = ordered_element_indices[i];
					uint2  e = sa_edges[eid];
					float3 a = sa_x[e.x];
					float3 b = sa_x[e.y];
					out_min = luisa::min(out_min, luisa::min(a, b));
					out_max = luisa::max(out_max, luisa::max(a, b));
				}
			};

			build_internal(centroids, compute_bounds, elem_count);
		}

		std::vector<std::vector<HitInfo>> CachedBVH::ray_cast_from_verts_batched(const std::vector<BatchedVertRayRequest>& requests,
			const std::vector<uint3>&																					   sa_faces,
			const uint																									   group_count,
			const float																									   kMaxRayDist,
			BatchedRaycastStats*																						   stats) const
		{
			std::vector<std::vector<HitInfo>> out_hits(group_count);
			if (nodes.empty() || requests.empty() || primitive_type != PrimitiveType::Face)
				return out_hits;
			if (sa_x_ptr == nullptr)
				LUISA_ERROR("Batched vertex raycast BVH has null vertex-position storage.");

			const int  root = 0;
			const uint num_vertices = static_cast<uint>(sa_x_ptr->size());
			const uint num_faces = static_cast<uint>(sa_faces.size());
			auto	   ray_aabb_intersect = [&](const float3& orig, const float3& dir_inv, const BVHNode& n, float t_max) -> bool
			{
				float3 t1, t2;
				t1.x = (n.bmin.x - orig.x) * dir_inv.x;
				t2.x = (n.bmax.x - orig.x) * dir_inv.x;
				float tmin = fminf(t1.x, t2.x);
				float tmax = fmaxf(t1.x, t2.x);

				t1.y = (n.bmin.y - orig.y) * dir_inv.y;
				t2.y = (n.bmax.y - orig.y) * dir_inv.y;
				tmin = fmaxf(tmin, fminf(t1.y, t2.y));
				tmax = fminf(tmax, fmaxf(t1.y, t2.y));

				t1.z = (n.bmin.z - orig.z) * dir_inv.z;
				t2.z = (n.bmax.z - orig.z) * dir_inv.z;
				tmin = fmaxf(tmin, fminf(t1.z, t2.z));
				tmax = fminf(tmax, fmaxf(t1.z, t2.z));

				return tmax >= fmaxf(tmin, 0.0f) && tmin <= t_max;
			};
			auto target_in_dense_range = [](const uint target_id, const uint2 target_range) -> bool
			{
				if (target_range.y == 0u)
					return false;
				if (target_range.y != std::numeric_limits<uint>::max())
				{
					const uint target_end = target_range.x + target_range.y;
					if (target_id < target_range.x || target_id >= target_end)
						return false;
				}
				return true;
			};
			auto target_in_exact_set = [](const uint target_id, const std::vector<uint>* target_ids) -> bool
			{
				return target_ids == nullptr || std::binary_search(target_ids->begin(), target_ids->end(), target_id);
			};
			std::vector<std::vector<HitInfo>> request_hits(requests.size());
			std::vector<BatchedRaycastStats>  request_stats(stats == nullptr ? 0u : requests.size());
			CpuParallel::parallel_for(
				0u,
				static_cast<uint>(requests.size()),
				[&](const uint index)
				{
					const auto& request = requests[index];
					auto&		curr_output = request_hits[index];
					auto*		curr_stats = stats == nullptr ? nullptr : &request_stats[index];
					const uint	vid = request.source_id;
					if (request.group_idx >= group_count)
						LUISA_ERROR("Batched vertex ray request {} has group_idx {} out of range {}.", index, request.group_idx, group_count);
					if (vid >= num_vertices)
						LUISA_ERROR("Batched vertex ray request {} has source vertex {} out of range {}.", index, vid, num_vertices);
					const float3 contour_dir = request.contour_direction;
					const float3 cast_dir = request.output_type == BatchedVertRayOutputType::VF ? contour_dir : -contour_dir;
					const float	 ray_max_dist = std::clamp(request.ray_max_dist, 0.0f, kMaxRayDist);
					if (ray_max_dist <= 0.0f || request.target_range.y == 0u)
						return;
					const float3 orig = (*sa_x_ptr)[vid];
					const float3 end = orig + cast_dir * ray_max_dist;
					const float3 dir_inv = 1.0f / cast_dir;

					std::vector<int> stack_nodes;
					stack_nodes.reserve(64);
					stack_nodes.push_back(root);
					while (!stack_nodes.empty())
					{
						const int nid = stack_nodes.back();
						stack_nodes.pop_back();
						if (curr_stats != nullptr)
							++curr_stats->visited_node_count;
						const BVHNode& node = nodes[nid];
						if (!ray_aabb_intersect(orig, dir_inv, node, ray_max_dist))
							continue;
						if (node.left == -1 && node.right == -1)
						{
							for (int i = node.start; i < node.start + node.count; ++i)
							{
								const uint fid = ordered_element_indices[i];
								if (curr_stats != nullptr)
									++curr_stats->tested_leaf_primitive_count;
								if (!target_in_dense_range(fid, request.target_range))
								{
									if (curr_stats != nullptr)
										++curr_stats->target_filter_reject_count;
									continue;
								}
								if (!target_in_exact_set(fid, request.target_primitive_ids))
								{
									if (curr_stats != nullptr)
										++curr_stats->target_filter_reject_count;
									continue;
								}
								if (fid >= num_faces)
									LUISA_ERROR("Batched vertex ray request {} reached target face {} out of range {}.", index, fid, num_faces);
								const uint3 face = sa_faces[fid];
								if (face.x >= num_vertices || face.y >= num_vertices || face.z >= num_vertices)
									LUISA_ERROR("Batched vertex ray request {} reached face {} with vertex ids ({}, {}, {}) out of range {}.",
										index, fid, face.x, face.y, face.z, num_vertices);
								if (request.target_vertex_hop_dist != nullptr && request.target_max_hop != std::numeric_limits<uint>::max())
								{
									const auto& target_hop_dist = *request.target_vertex_hop_dist;
									if (face.x >= target_hop_dist.size() || face.y >= target_hop_dist.size() || face.z >= target_hop_dist.size())
										LUISA_ERROR("Batched vertex ray request {} reached face {} with vertex ids ({}, {}, {}) out of hop-distance range {}.",
											index, fid, face.x, face.y, face.z, target_hop_dist.size());
									if (target_hop_dist[face.x] > request.target_max_hop
										|| target_hop_dist[face.y] > request.target_max_hop
										|| target_hop_dist[face.z] > request.target_max_hop)
									{
										if (curr_stats != nullptr)
											++curr_stats->target_hop_filter_reject_count;
										continue;
									}
								}
								if (luisa::any(vid == face))
									continue;
								const float3 A = (*sa_x_ptr)[face.x];
								const float3 B = (*sa_x_ptr)[face.y];
								const float3 C = (*sa_x_ptr)[face.z];
								float3		 out_bary;
								float		 out_time;
								if (distance::LineIntersection(orig, end, A, B, C, out_bary, out_time)
									&& out_time >= 0.0f && out_time <= 1.0f)
								{
									const float hit_dist = out_time * ray_max_dist;
									if (request.output_type == BatchedVertRayOutputType::VF)
									{
										curr_output.push_back(HitInfo::make_vf_hit(
											request.group_idx, 0u, vid, fid, face, luisa::make_float2(out_bary.x, out_bary.y), contour_dir, hit_dist));
									}
									else if (request.output_type == BatchedVertRayOutputType::FV)
									{
										curr_output.push_back(HitInfo::make_fv_hit(
											request.group_idx, 0u, vid, fid, face, luisa::make_float2(out_bary.x, out_bary.y), contour_dir, hit_dist));
									}
									if (curr_stats != nullptr)
										++curr_stats->produced_hit_count;
								}
							}
						}
						else
						{
							if (node.left != -1)
								stack_nodes.push_back(node.left);
							if (node.right != -1)
								stack_nodes.push_back(node.right);
						}
					}
				},
				32u);

			if (stats != nullptr)
			{
				stats->request_count += requests.size();
				for (const auto& local_stats : request_stats)
				{
					stats->visited_node_count += local_stats.visited_node_count;
					stats->tested_leaf_primitive_count += local_stats.tested_leaf_primitive_count;
					stats->target_filter_reject_count += local_stats.target_filter_reject_count;
					stats->target_hop_filter_reject_count += local_stats.target_hop_filter_reject_count;
					stats->produced_hit_count += local_stats.produced_hit_count;
				}
			}

			for (size_t request_idx = 0u; request_idx < requests.size(); ++request_idx)
			{
				const auto group_idx = requests[request_idx].group_idx;
				if (group_idx >= group_count)
					LUISA_ERROR("Batched vertex ray request {} has group_idx {} out of range {}.", request_idx, group_idx, group_count);
				auto& dst = out_hits[group_idx];
				auto& src = request_hits[request_idx];
				dst.insert(dst.end(), src.begin(), src.end());
			}

			return out_hits;
		}

		std::vector<std::vector<HitInfo>> CachedBVH::ray_cast_from_edges_batched(const std::vector<BatchedEdgeRayRequest>& requests,
			const std::vector<uint2>&																					   sa_left_edges,
			const std::vector<uint2>&																					   sa_right_edges,
			const uint																									   group_count,
			const float																									   kMaxRayDist,
			BatchedRaycastStats*																						   stats) const
		{
			std::vector<std::vector<HitInfo>> out_hits(group_count);
			if (nodes.empty() || requests.empty() || primitive_type != PrimitiveType::Edge)
				return out_hits;
			if (sa_x_ptr == nullptr)
				LUISA_ERROR("Batched edge raycast BVH has null vertex-position storage.");

			const int  root = 0;
			const uint num_vertices = static_cast<uint>(sa_x_ptr->size());
			const uint num_left_edges = static_cast<uint>(sa_left_edges.size());
			const uint num_right_edges = static_cast<uint>(sa_right_edges.size());
			auto	   aabb_intersect = [&](const float3& amin, const float3& amax, const BVHNode& n) -> bool
			{
				return (amin.x <= n.bmax.x && amax.x >= n.bmin.x)
					&& (amin.y <= n.bmax.y && amax.y >= n.bmin.y)
					&& (amin.z <= n.bmax.z && amax.z >= n.bmin.z);
			};
			auto target_in_dense_range = [](const uint target_id, const uint2 target_range) -> bool
			{
				if (target_range.y == 0u)
					return false;
				if (target_range.y != std::numeric_limits<uint>::max())
				{
					const uint target_end = target_range.x + target_range.y;
					if (target_id < target_range.x || target_id >= target_end)
						return false;
				}
				return true;
			};
			auto target_in_exact_set = [](const uint target_id, const std::vector<uint>* target_ids) -> bool
			{
				return target_ids == nullptr || std::binary_search(target_ids->begin(), target_ids->end(), target_id);
			};
			std::vector<std::vector<HitInfo>> request_hits(requests.size());
			std::vector<BatchedRaycastStats>  request_stats(stats == nullptr ? 0u : requests.size());
			CpuParallel::parallel_for(
				0u,
				static_cast<uint>(requests.size()),
				[&](const uint index)
				{
					const auto& request = requests[index];
					auto&		curr_output = request_hits[index];
					auto*		curr_stats = stats == nullptr ? nullptr : &request_stats[index];
					const uint	eid1 = request.source_id;
					if (request.group_idx >= group_count)
						LUISA_ERROR("Batched edge ray request {} has group_idx {} out of range {}.", index, request.group_idx, group_count);
					if (eid1 >= num_left_edges)
						LUISA_ERROR("Batched edge ray request {} has source edge {} out of range {}.", index, eid1, num_left_edges);
					const uint2 edge1 = sa_left_edges[eid1];
					if (edge1.x >= num_vertices || edge1.y >= num_vertices)
						LUISA_ERROR("Batched edge ray request {} reached source edge {} with vertex ids ({}, {}) out of range {}.",
							index, eid1, edge1.x, edge1.y, num_vertices);
					const float3 dir = request.direction;
					const float	 ray_max_dist = std::clamp(request.ray_max_dist, 0.0f, kMaxRayDist);
					if (ray_max_dist <= 0.0f || request.target_range.y == 0u)
						return;

					const float3 p0 = (*sa_x_ptr)[edge1.x];
					const float3 p1 = (*sa_x_ptr)[edge1.y];
					const float3 e1 = p1 - p0;
					float3		 swept_min, swept_max;
					compute_swept_aabb(p0, p1, dir, ray_max_dist, swept_min, swept_max);

					float3		plane_normal = luisa::cross(e1, dir);
					const float normal_len = luisa::length(plane_normal);
					const bool	use_plane_test = normal_len >= 1e-10f;
					float		plane_d = 0.0f;
					if (use_plane_test)
					{
						plane_normal /= normal_len;
						plane_d = luisa::dot(plane_normal, p0);
					}

					const float dir_len2 = luisa::dot(dir, dir);
					const float edge_proj0 = luisa::dot(p0, dir);
					const float edge_proj1 = luisa::dot(p1, dir);
					// Phase C-1: combine the source projection bound with the caller-supplied target floor.
					const float src_proj_min = std::min(edge_proj0, edge_proj1);
					const float proj_min = std::max(src_proj_min, request.target_proj_min - 1e-5f);
					const float proj_max = std::max(edge_proj0, edge_proj1) + ray_max_dist * dir_len2;

					auto swept_plane_aabb_intersect = [&](const BVHNode& node) -> bool
					{
						if (!aabb_intersect(swept_min, swept_max, node))
							return false;

						const float3 box_center = (node.bmin + node.bmax) * 0.5f;
						const float3 box_extent = (node.bmax - node.bmin) * 0.5f;
						const float	 center_proj = luisa::dot(box_center, dir);
						const float	 proj_radius = box_extent.x * std::abs(dir.x)
							+ box_extent.y * std::abs(dir.y)
							+ box_extent.z * std::abs(dir.z);
						if (center_proj + proj_radius < proj_min - 1e-6f
							|| center_proj - proj_radius > proj_max + 1e-6f)
							return false;

						if (!use_plane_test)
							return true;

						const float center_dist = luisa::dot(plane_normal, box_center) - plane_d;
						const float plane_radius = box_extent.x * std::abs(plane_normal.x)
							+ box_extent.y * std::abs(plane_normal.y)
							+ box_extent.z * std::abs(plane_normal.z);
						return std::abs(center_dist) <= plane_radius + 1e-6f;
					};

					std::vector<int> stack;
					stack.reserve(64);
					stack.push_back(root);
					while (!stack.empty())
					{
						const int nid = stack.back();
						stack.pop_back();
						if (curr_stats != nullptr)
							++curr_stats->visited_node_count;
						const BVHNode& node = nodes[nid];
						if (!swept_plane_aabb_intersect(node))
							continue;
						if (node.left == -1 && node.right == -1)
						{
							for (int i = node.start; i < node.start + node.count; ++i)
							{
								const uint eid2 = ordered_element_indices[i];
								if (curr_stats != nullptr)
									++curr_stats->tested_leaf_primitive_count;
								if (!target_in_dense_range(eid2, request.target_range))
								{
									if (curr_stats != nullptr)
										++curr_stats->target_filter_reject_count;
									continue;
								}
								if (!target_in_exact_set(eid2, request.target_primitive_ids))
								{
									if (curr_stats != nullptr)
										++curr_stats->target_filter_reject_count;
									continue;
								}
								if (eid2 >= num_right_edges)
									LUISA_ERROR("Batched edge ray request {} reached target edge {} out of range {}.", index, eid2, num_right_edges);
								const uint2 edge2 = sa_right_edges[eid2];
								if (edge2.x >= num_vertices || edge2.y >= num_vertices)
									LUISA_ERROR("Batched edge ray request {} reached target edge {} with vertex ids ({}, {}) out of range {}.",
										index, eid2, edge2.x, edge2.y, num_vertices);
								if (request.target_vertex_hop_dist != nullptr && request.target_max_hop != std::numeric_limits<uint>::max())
								{
									const auto& target_hop_dist = *request.target_vertex_hop_dist;
									if (edge2.x >= target_hop_dist.size() || edge2.y >= target_hop_dist.size())
										LUISA_ERROR("Batched edge ray request {} reached target edge {} with vertex ids ({}, {}) out of hop-distance range {}.",
											index, eid2, edge2.x, edge2.y, target_hop_dist.size());
									if (target_hop_dist[edge2.x] > request.target_max_hop || target_hop_dist[edge2.y] > request.target_max_hop)
									{
										if (curr_stats != nullptr)
											++curr_stats->target_hop_filter_reject_count;
										continue;
									}
								}
								if (edge1.x == edge2.x || edge1.x == edge2.y || edge1.y == edge2.x || edge1.y == edge2.y)
									continue;

								const float3 q0 = (*sa_x_ptr)[edge2.x];
								const float3 q1 = (*sa_x_ptr)[edge2.y];
								if (use_plane_test)
								{
									const float plane_dist0 = luisa::dot(plane_normal, q0) - plane_d;
									const float plane_dist1 = luisa::dot(plane_normal, q1) - plane_d;
									if ((plane_dist0 > 1e-6f && plane_dist1 > 1e-6f)
										|| (plane_dist0 < -1e-6f && plane_dist1 < -1e-6f))
										continue;
								}

								const float3 e2 = q1 - q0;
								const float3 r = q0 - p0;
								const float3 n = luisa::cross(e1, e2);
								const float	 det = -luisa::dot(dir, n);
								if (std::abs(det) < 1e-10f)
									continue;

								const float inv_det = 1.0f / det;
								const float det_r_e1_e2 = luisa::dot(r, luisa::cross(e1, e2));
								const float det_dir_r_e2 = luisa::dot(dir, luisa::cross(r, e2));
								const float det_dir_e1_r = luisa::dot(dir, luisa::cross(e1, r));
								const float s = -det_r_e1_e2 * inv_det;
								const float t1 = -det_dir_r_e2 * inv_det;
								const float t2 = det_dir_e1_r * inv_det;
								if (s >= 0.0f && s <= ray_max_dist && t1 >= 0.0f && t1 <= 1.0f && t2 >= 0.0f && t2 <= 1.0f)
								{
									curr_output.push_back(HitInfo::make_ee_hit(
										request.group_idx, 0u, eid1, eid2, edge1, edge2, 1.0f - t1, 1.0f - t2, dir, s));
									if (curr_stats != nullptr)
										++curr_stats->produced_hit_count;
								}
							}
						}
						else
						{
							if (node.left != -1)
								stack.push_back(node.left);
							if (node.right != -1)
								stack.push_back(node.right);
						}
					}
				},
				32u);

			if (stats != nullptr)
			{
				stats->request_count += requests.size();
				for (const auto& local_stats : request_stats)
				{
					stats->visited_node_count += local_stats.visited_node_count;
					stats->tested_leaf_primitive_count += local_stats.tested_leaf_primitive_count;
					stats->target_filter_reject_count += local_stats.target_filter_reject_count;
					stats->target_hop_filter_reject_count += local_stats.target_hop_filter_reject_count;
					stats->produced_hit_count += local_stats.produced_hit_count;
				}
			}

			for (size_t request_idx = 0u; request_idx < requests.size(); ++request_idx)
			{
				const auto group_idx = requests[request_idx].group_idx;
				if (group_idx >= group_count)
					LUISA_ERROR("Batched edge ray request {} has group_idx {} out of range {}.", request_idx, group_idx, group_count);
				auto& dst = out_hits[group_idx];
				auto& src = request_hits[request_idx];
				dst.insert(dst.end(), src.begin(), src.end());
			}

			return out_hits;
		}

		size_t BVHCache::vector_hash(const std::vector<uint>& values)
		{
			size_t hash = values.size();
			for (const uint value : values)
				hash ^= static_cast<size_t>(value) + 0x9e3779b97f4a7c15ULL + (hash << 6) + (hash >> 2);
			return hash;
		}

		CachedBVH* BVHCache::get_or_build(const std::vector<uint>& candidate_faces,
			const std::vector<float3>& sa_x, const std::vector<uint3>& sa_faces,
			const uint candidate_hash)
		{
			const size_t				key = candidate_hash != -1u ? candidate_hash : vector_hash(candidate_faces);
			std::lock_guard<std::mutex> lock(mutex);
			if (auto it = cache.find(key); it != cache.end())
				return &it->second;
			CachedBVH bvh;
			bvh.build(candidate_faces, sa_x, sa_faces);
			auto [it, inserted] = cache.emplace(key, std::move(bvh));
			return &it->second;
		}

		CachedBVH* BVHCache::get_or_build(const std::vector<uint>& candidate_edges,
			const std::vector<float3>& sa_x, const std::vector<uint2>& sa_edges,
			const uint candidate_hash)
		{
			const size_t				key = candidate_hash != -1u ? candidate_hash : vector_hash(candidate_edges);
			std::lock_guard<std::mutex> lock(mutex);
			if (auto it = cache.find(key); it != cache.end())
				return &it->second;
			CachedBVH bvh;
			bvh.build(candidate_edges, sa_x, sa_edges);
			auto [it, inserted] = cache.emplace(key, std::move(bvh));
			return &it->second;
		}

	}; // namespace RayCasting

} // namespace lcs
