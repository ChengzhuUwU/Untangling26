#pragma once

#include "CollisionDetector/lbvh.h"
#include "CollisionDetector/intersection_resolver_gpu.h"
#include "CollisionDetector/intersection_resolver2.h"
#include "Core/scalar.h"
#include "SimulationCore/base_mesh.h"
#include "SimulationCore/simulation_data.h"
#include "SimulationCore/collision_data.h"
#include "SimulationCore/simulation_type.h"
#include <vector>
#include <luisa/luisa-compute.h>
#include <Utils/async_compiler.h>

namespace lcs
{

} // namespace lcs

namespace lcs
{

	class IntersectionResolver
	{
		template <typename T>
		using Buffer = luisa::compute::Buffer<T>;
		template <typename T>
		using BufferView = luisa::compute::BufferView<T>;
		using Stream = luisa::compute::Stream;
		using Device = luisa::compute::Device;

	private:
		void compile_detection(AsyncCompiler& compiler);
		void compile_resolve(AsyncCompiler& compiler);

	public:
		void compile(AsyncCompiler& compiler);

	public:
		void download_intersection_list(Stream& stream);
		void ef_dcd_intersection_query(Stream& stream,
			const Buffer<float3>&			   sa_x,
			const Buffer<uint2>&			   sa_edges,
			const Buffer<uint3>&			   sa_faces,
			const Buffer<float3>&			   sa_rest_x,
			const Buffer<float3>&			   sa_x_step_start,
			const Buffer<float>&			   sa_rest_edge_area,
			const Buffer<float>&			   sa_rest_face_area,
			const Buffer<VertexProperty>&	   sa_x_property,
			const Buffer<float>&			   d_hat,
			const Buffer<float>&			   thickness,
			const float						   kappa);
		void host_resolve(Device& device, Stream& stream);
		void host_clear(Device& device, Stream& stream);

	private:
		void host_preprocess_ef_pairs(Device& device, Stream& stream);
		void host_identify_contour_type(Device& device, Stream& stream);
		void host_identify_region(Device& device, Stream& stream);
		void host_flood_filling(Device& device, Stream& stream);
		void host_resolve_ICM(Device& device, Stream& stream);
		void host_resolve_GIA(Device& device, Stream& stream);
		void host_resolve_PRP(Device& device, Stream& stream);
		void device_resolve_PRP(Device& device, Stream& stream);

	public:
		void host_assemble_ef_response(Stream& stream, std::vector<float3>& sa_cgB, std::vector<float3x3>& sa_cgA_diag);
		void device_assemble_ef_response(Stream& stream, Buffer<float3>& sa_cgB, Buffer<float3x3>& sa_cgA_diag);
		void host_spmv(Stream& stream, const std::vector<float3>& input_ptr, std::vector<float3>& output_ptr);
		void host_spmv_ICM(Stream& stream, const std::vector<float3>& input_ptr, std::vector<float3>& output_ptr);
		void device_spmv(Stream& stream, const Buffer<float3>& input_ptr, Buffer<float3>& output_ptr);
		void device_spmv_ICM(Stream& stream, const Buffer<float3>& input_ptr, Buffer<float3>& output_ptr);

	public:
		void filter_broad_phase_pairs(Stream& stream);
		void filter_invalid_pairs(Stream& stream);
		void reset_prp_shaders() { prp_shaders.reset(); }
		void get_intersection_curves(
			std::vector<std::array<float, 3>>& output_positions,
			std::vector<std::array<uint, 2>>&  output_edges,
			std::vector<std::array<float, 3>>& output_colors);

	private:
		DevicePRPPrecomputeShaders				prp_shaders;
		CollisionData<luisa::compute::Buffer>*	device_collision_data;
		CollisionData<std::vector>*				host_collision_data;
		UntanglingData<luisa::compute::Buffer>* device_untangling_data;
		UntanglingData<std::vector>*			host_untangling_data;
		MeshData<luisa::compute::Buffer>*		device_mesh_data;
		MeshData<std::vector>*					host_mesh_data;
		SimulationData<luisa::compute::Buffer>* device_sim_data;
		SimulationData<std::vector>*			host_sim_data;
		LbvhData<luisa::compute::Buffer>*		lbvh_data_face = nullptr;
		LbvhData<luisa::compute::Buffer>*		lbvh_data_edge = nullptr;

		// GIA host-side working state (populated per step in GIA mode)
		float												 gia_cached_avg_edge_len = 0.0f;
		uint												 gia_cached_num_edges = 0u;
		float												 gia_last_perturb_eps = 0.0f;
		bool												 gia_has_active_ef_response = false;
		std::vector<int>									 gia_face_orient;
		std::vector<float3>									 gia_face_normals;
		std::vector<uint2>									 gia_invalid_region_pair;
		std::vector<std::array<std::vector<uint>, 2>>		 gia_contour_pierced_faces;

	public:
		void set_collision_data(CollisionData<luisa::compute::Buffer>* collision_data,
			CollisionData<std::vector>*								   host_collision_data,
			UntanglingData<luisa::compute::Buffer>*					   untangling_data,
			UntanglingData<std::vector>*							   host_untangling_data,
			MeshData<luisa::compute::Buffer>*						   mesh_data,
			MeshData<std::vector>*									   host_mesh_data,
			SimulationData<luisa::compute::Buffer>*					   sim_data,
			SimulationData<std::vector>*							   host_sim_data)
		{
			this->device_collision_data = collision_data;
			this->host_collision_data = host_collision_data;
			this->device_untangling_data = untangling_data;
			this->host_untangling_data = host_untangling_data;
			this->device_mesh_data = mesh_data;
			this->host_mesh_data = host_mesh_data;
			this->device_sim_data = sim_data;
			this->host_sim_data = host_sim_data;
		}

		void set_lbvh_data(LbvhData<luisa::compute::Buffer>* face_data, LbvhData<luisa::compute::Buffer>* edge_data)
		{
			this->lbvh_data_face = face_data;
			this->lbvh_data_edge = edge_data;
		}

		void atomic_add(auto& target_array, const auto& value, const uint index)
		{
			auto mutex_view = std::span(reinterpret_cast<luisa::spin_mutex*>(host_untangling_data->vert_mutex.data()), host_mesh_data->num_verts);
			mutex_view[index].lock();
			target_array[index] = target_array[index] + value;
			mutex_view[index].unlock();
		}

	private:
		using CDBG = lcs::CollisionData<luisa::compute::Buffer>; // CollisionData Binding Group

		CollisionData<luisa::compute::Buffer>& get_collision_data() { return *device_collision_data; }

	private:
		luisa::compute::Shader<1, Buffer<uint>> fn_reset_uint;
		luisa::compute::Shader<1,
			Buffer<uint>,
			Buffer<uint>,
			Buffer<CollisionPair::EfPair>,
			Buffer<uint2>,
			Buffer<float3>,
			Buffer<uint2>,
			Buffer<uint3>,
			Buffer<float3>,
			Buffer<float3>,
			Buffer<float>,
			Buffer<float>,
			Buffer<VertexProperty>,
			Buffer<uint3>,
			Buffer<float>,
			Buffer<float>,
			unsigned int,
			float>
			fn_narrow_phase_ef_dcd_query;

		luisa::compute::Shader<1,
			lcs::CollisionData<luisa::compute::Buffer>,
			Buffer<float3>,
			Buffer<uint2>,
			Buffer<uint3>,
			Buffer<uint>,
			Buffer<uint>,
			Buffer<float>,
			unsigned int,
			unsigned int>
			fn_add_ef_from_proximity_pairs;

		luisa::compute::Shader<1,
			lcs::CollisionData<luisa::compute::Buffer>,
			//  lcs::UntanglingData<luisa::compute::Buffer>,
			luisa::compute::Buffer<luisa::Vector<float, 3>>,
			luisa::compute::Buffer<float>,
			luisa::compute::Buffer<float>>
																			  fn_identify_invalid_verts_from_proximity_pairs;
		luisa::compute::Shader<1, lcs::CollisionData<luisa::compute::Buffer>> fn_filter_invalid_proximity_pairs_from_invalid_verts;
		luisa::compute::Shader<1,
			luisa::compute::Buffer<CollisionPair::CollisionPairTemplate>,
			luisa::compute::Buffer<uint2>,
			luisa::compute::Buffer<uint>,
			luisa::compute::Buffer<uint>,
			luisa::compute::Buffer<uint>>
			fn_filter_invalid_proximity_pairs_from_region_info;
		luisa::compute::Shader<1,
			luisa::compute::Buffer<uint>,
			luisa::compute::Buffer<uint>,
			luisa::compute::Buffer<uint>,
			luisa::compute::Buffer<uint>,
			luisa::compute::Buffer<uint>,
			luisa::compute::Buffer<uint3>,
			uint>
			fn_filter_broadphase_vf_pairs;

		luisa::compute::Shader<1,
			luisa::compute::Buffer<uint>,
			luisa::compute::Buffer<uint>,
			luisa::compute::Buffer<uint>,
			luisa::compute::Buffer<uint>,
			luisa::compute::Buffer<uint>,
			luisa::compute::Buffer<uint2>,
			uint>
			fn_filter_broadphase_ee_pairs;

		luisa::compute::Shader<1,
			lcs::CollisionData<luisa::compute::Buffer>,
			Buffer<float3>,
			Buffer<float3>>
			fn_spmv_ICM;

		luisa::compute::Shader<1,
			lcs::CollisionData<luisa::compute::Buffer>,
			Buffer<float3>,
			Buffer<float3x3>>
			fn_assemble_ICM;
	};

} // namespace lcs
