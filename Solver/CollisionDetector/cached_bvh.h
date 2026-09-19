#pragma once

#include "CollisionDetector/intersection_resolver_helper.h"

#include <algorithm>
#include <mutex>
#include <unordered_map>

namespace lcs::RayCasting
{
	// CPU BVH used by batched PRP ray casting for face and edge primitives.
	struct CachedBVH
	{
		enum class PrimitiveType
		{
			None,
			Face,
			Edge
		};

		struct BVHNode
		{
			float3 bmin;
			float3 bmax;
			int	   left;
			int	   right;
			int	   start;
			int	   count;

			BVHNode();
		};

		static void compute_swept_aabb(const float3& a, const float3& b,
			const float3& direction, float max_distance, float3& out_min, float3& out_max);

		std::vector<BVHNode>	   nodes;
		std::vector<uint>		   ordered_element_indices;
		const std::vector<float3>* sa_x_ptr = nullptr;
		PrimitiveType			   primitive_type = PrimitiveType::None;

		void build(const std::vector<uint>& candidate_faces,
			const std::vector<float3>& sa_x, const std::vector<uint3>& sa_faces);
		void build(const std::vector<uint>& candidate_edges,
			const std::vector<float3>& sa_x, const std::vector<uint2>& sa_edges);

		std::vector<std::vector<HitInfo>> ray_cast_from_verts_batched(
			const std::vector<BatchedVertRayRequest>& requests,
			const std::vector<uint3>& sa_faces, uint group_count, float max_ray_distance,
			BatchedRaycastStats* stats = nullptr) const;

		std::vector<std::vector<HitInfo>> ray_cast_from_edges_batched(
			const std::vector<BatchedEdgeRayRequest>& requests,
			const std::vector<uint2>&				  sa_left_edges,
			const std::vector<uint2>& sa_right_edges, uint group_count,
			float max_ray_distance, BatchedRaycastStats* stats = nullptr) const;

	private:
		template <typename ComputeBoundsFn>
		int build_node(int start, int end, std::vector<float3>& centroids,
			ComputeBoundsFn& compute_bounds);

		template <typename ComputeBoundsFn>
		void build_internal(std::vector<float3>& centroids,
			ComputeBoundsFn compute_bounds, int element_count);
	};

	// Thread-safe cache keyed by candidate primitive signatures.
	struct BVHCache
	{
		std::unordered_map<size_t, CachedBVH> cache;
		std::mutex							  mutex;

		static size_t vector_hash(const std::vector<uint>& values);

		CachedBVH* get_or_build(const std::vector<uint>& candidate_faces,
			const std::vector<float3>& sa_x, const std::vector<uint3>& sa_faces,
			uint candidate_hash);
		CachedBVH* get_or_build(const std::vector<uint>& candidate_edges,
			const std::vector<float3>& sa_x, const std::vector<uint2>& sa_edges,
			uint candidate_hash);
	};
} // namespace lcs::RayCasting

#include "cached_bvh.hpp"
