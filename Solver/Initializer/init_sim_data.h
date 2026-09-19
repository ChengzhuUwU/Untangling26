#pragma once

#include "Initializer/init_mesh_data.h"
#include "SimulationCore/base_mesh.h"
#include "SimulationCore/simulation_data.h"

namespace lcs::Initializer
{
	[[nodiscard]] inline bool bending_edge_has_distinct_vertices(const uint4 vertices) noexcept
	{
		return vertices[0] != vertices[1]
			&& vertices[0] != vertices[2]
			&& vertices[0] != vertices[3]
			&& vertices[1] != vertices[2]
			&& vertices[1] != vertices[3]
			&& vertices[2] != vertices[3];
	}

	[[nodiscard]] inline bool bending_edge_has_valid_geometry(const uint4 edge, const std::vector<luisa::float3>& rest_x) noexcept
	{
		if (!bending_edge_has_distinct_vertices(edge))
			return false;
		if (edge[0] >= rest_x.size() || edge[1] >= rest_x.size() || edge[2] >= rest_x.size() || edge[3] >= rest_x.size())
			return false;
		const auto& x0 = rest_x[edge[0]];
		const auto& x1 = rest_x[edge[1]];
		const auto& x2 = rest_x[edge[2]];
		const auto& x3 = rest_x[edge[3]];
		const auto	e0 = x1 - x0;
		const auto	e1 = x2 - x0;
		const auto	e2 = x3 - x0;
		const auto	n1 = luisa::cross(e0, e1);
		const auto	n2 = luisa::cross(e2, e0);
		const float e0_sqnm = luisa::dot(e0, e0);
		const float n1_sqnm = luisa::dot(n1, n1);
		const float n2_sqnm = luisa::dot(n2, n2);
		return (e0_sqnm >= 1e-12f && n1_sqnm >= 1e-20f && n2_sqnm >= 1e-20f);
	}

	void init_sim_data(const std::vector<lcs::Initializer::WorldData>& shell_infos,
		lcs::MeshData<std::vector>*									   mesh_data,
		lcs::SimulationData<std::vector>*							   sim_data);
	void upload_sim_buffers(luisa::compute::Device&	 device,
		luisa::compute::Stream&						 stream,
		lcs::SimulationData<std::vector>*			 input_data,
		lcs::SimulationData<luisa::compute::Buffer>* output_data);

	void resize_pcg_data(luisa::compute::Device&	 device,
		luisa::compute::Stream&						 stream,
		lcs::MeshData<std::vector>*					 mesh_data,
		lcs::SimulationData<std::vector>*			 host_data,
		lcs::SimulationData<luisa::compute::Buffer>* device_data);

	void init_colored_data(lcs::SimulationData<std::vector>* sim_data);
	void upload_colored_data(luisa::compute::Device& device,
		luisa::compute::Stream&						 stream,
		lcs::SimulationData<std::vector>*			 input_data,
		lcs::SimulationData<luisa::compute::Buffer>* output_data);

} // namespace lcs::Initializer
