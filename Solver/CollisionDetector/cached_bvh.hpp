#pragma once

namespace lcs::RayCasting
{
	template <typename ComputeBoundsFn>
	int CachedBVH::build_node(const int start, const int end,
		std::vector<float3>& centroids, ComputeBoundsFn& compute_bounds)
	{
		BVHNode node;
		compute_bounds(start, end, node.bmin, node.bmax);
		const int node_index = static_cast<int>(nodes.size());
		nodes.push_back(node);

		const int	  count = end - start;
		constexpr int leaf_size = 8;
		if (count <= leaf_size)
		{
			nodes[node_index].start = start;
			nodes[node_index].count = count;
			return node_index;
		}

		const float3 extent = node.bmax - node.bmin;
		int			 axis = 0;
		if (extent.y > extent.x && extent.y >= extent.z)
			axis = 1;
		else if (extent.z > extent.x && extent.z > extent.y)
			axis = 2;

		const int middle = (start + end) / 2;
		std::nth_element(ordered_element_indices.begin() + start,
			ordered_element_indices.begin() + middle,
			ordered_element_indices.begin() + end,
			[&centroids, axis](const uint a, const uint b)
			{
				if (axis == 0)
					return centroids[a].x < centroids[b].x;
				if (axis == 1)
					return centroids[a].y < centroids[b].y;
				return centroids[a].z < centroids[b].z;
			});

		nodes[node_index].left = build_node(start, middle, centroids, compute_bounds);
		nodes[node_index].right = build_node(middle, end, centroids, compute_bounds);
		return node_index;
	}

	template <typename ComputeBoundsFn>
	void CachedBVH::build_internal(std::vector<float3>& centroids,
		ComputeBoundsFn compute_bounds, const int element_count)
	{
		if (!ordered_element_indices.empty())
			build_node(0, element_count, centroids, compute_bounds);
	}
} // namespace lcs::RayCasting
