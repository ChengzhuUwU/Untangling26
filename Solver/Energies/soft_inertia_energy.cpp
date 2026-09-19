#include "soft_inertia_energy.h"
#include "SimulationCore/base_mesh.h"
#include "SimulationCore/scene_params.h"
#include "Utils/cpu_parallel.h"
#include "Utils/reduce_helper.h"

using namespace luisa::compute;

namespace lcs
{
	SoftInertiaEnergy::SoftInertiaEnergy(BufferView<float3> sa_q_tilde_view,
		BufferView<uint>									sa_q_is_fixed_view,
		BufferView<float>									sa_system_energy_view) noexcept
		: _sa_q_tilde_view(sa_q_tilde_view)
		, _sa_q_is_fixed_view(sa_q_is_fixed_view)
		, _sa_system_energy_view(sa_system_energy_view)
	{
	}

	void SoftInertiaEnergy::compile(AsyncCompiler& compiler)
	{
		luisa::compute::ShaderOption default_option = { .enable_debug_info = false };
		compiler.compile<1>(
			_shader,
			[sa_q_tilde = _sa_q_tilde_view,
				sa_q_is_fixed = _sa_q_is_fixed_view,
				sa_system_energy = _sa_system_energy_view](
				Var<Constitutions::SoftInertia<luisa::compute::Buffer>> constraint, Var<BufferView<float3>> sa_q, Float substep_dt, Bool use_quasi_static_mode, UInt qs_inertia_mode)
			{
				auto& soft_inertia_indices = constraint.constraint_indices;
				auto& sa_vert_mass = constraint.sa_soft_vert_mass;
				auto& sa_stiffness_dirichlet = constraint.sa_stiffness_dirichlet;

				const Uint index = dispatch_id().x;
				const Uint x_vid = soft_inertia_indices.read(index);
				const Bool is_fixed = sa_q_is_fixed->read(index) != 0u;
				const Bool energy_gradient_active = !use_quasi_static_mode | is_fixed | (qs_inertia_mode == static_cast<uint>(QuasiStaticInertiaMode::Full));

				Float energy = 0.0f;
				$if(energy_gradient_active)
				{
					Float3		x_new = sa_q->read(x_vid);
					Float3		x_tilde = sa_q_tilde->read(index);
					Float		mass = sa_vert_mass->read(index);
					const Float squared_inv_dt = 1.0f / (substep_dt * substep_dt);
					energy = squared_inv_dt * length_squared_vec(x_new - x_tilde) * mass / (2.0f);
					{
						Float stiffness_dirichlet = sa_stiffness_dirichlet->read(index);
						energy = stiffness_dirichlet * energy;
					};
				};

				energy = ParallelIntrinsic::block_intrinsic_reduce(index, energy, ParallelIntrinsic::warp_reduce_op_sum<float>);
				$if(index % 256 == 0)
				{
					sa_system_energy->atomic(offset_inertia).fetch_add(energy);
				};
			},
			default_option);

		// gradient/hessian evaluate shader (soft inertia)
		compiler.compile<1>(
			_eval_shader,
			[sa_q_tilde = _sa_q_tilde_view,
				sa_q_is_fixed = _sa_q_is_fixed_view](Var<Constitutions::SoftInertia<luisa::compute::Buffer>> contraint,
				Var<BufferView<float3>>																		 sa_q,
				Float																						 substep_dt,
				Bool																						 use_quasi_static_mode,
				Bool																						 use_static_mode,
				UInt																						 qs_inertia_mode)
			{
				auto& sa_vert_mass = contraint.sa_soft_vert_mass;
				auto& sa_stiffness_dirichlet = contraint.sa_stiffness_dirichlet;
				auto& output_gradient = contraint.constraint_gradients;
				auto& output_hessian = contraint.constraint_hessians;

				const UInt	index = dispatch_id().x;
				const Bool	is_fixed = sa_q_is_fixed->read(index) != 0u;
				const Bool	energy_gradient_active = !use_quasi_static_mode | is_fixed | (qs_inertia_mode == static_cast<uint>(QuasiStaticInertiaMode::Full));
				const Bool	hessian_active = !use_static_mode & (!use_quasi_static_mode | is_fixed | (qs_inertia_mode != static_cast<uint>(QuasiStaticInertiaMode::Off)));
				const Float h = substep_dt;
				const Float h_2_inv = 1.0f / (h * h);

				// The evaluation path receives the compact generalized-coordinate
				// buffer. Soft DOFs occupy [0, num_verts_soft), whereas
				// constraint_indices stores the corresponding mesh-vertex index for
				// position-space energy evaluation and adjacency construction.
				Float3 x_k = sa_q->read(index);
				Float3 x_tilde = sa_q_tilde->read(index);
				Float  mass = sa_vert_mass->read(index);

				Float3	 gradient = Zero3;
				Float3x3 hessian = make_float3x3(0.0f);
				$if(energy_gradient_active)
				{
					gradient = mass * h_2_inv * (x_k - x_tilde);
				};
				$if(hessian_active)
				{
					hessian = make_float3x3(1.0f) * mass * h_2_inv;
				};

				{
					const Float stiffness_dirichlet = sa_stiffness_dirichlet->read(index);

					gradient = stiffness_dirichlet * gradient;
					hessian = stiffness_dirichlet * hessian;
				};

				output_gradient->write(index, gradient);
				output_hessian->write(index, hessian);
			},
			default_option);
	}

	void SoftInertiaEnergy::device_compute_energy(luisa::compute::Stream& stream)
	{
		// This class does not know which constitution to dispatch; caller should use the stored _shader directly.
		// Left intentionally empty — caller will dispatch using shader member exposed via friend or directly if needed.
	}

	void SoftInertiaEnergy::device_compute_energy(luisa::compute::Stream& stream,
		const Constitutions::SoftInertia<luisa::compute::Buffer>&		  constraint,
		const luisa::compute::Buffer<float3>&							  sa_q,
		float															  substep_dt,
		size_t															  dispatch_count)
	{
		stream << _shader(constraint,
			sa_q.view(),
			substep_dt,
			get_scene_params().use_quasi_static_mode,
			static_cast<uint>(get_scene_params().qs_inertia_mode))
					  .dispatch(dispatch_count);
	}

	void SoftInertiaEnergy::device_evaluate(luisa::compute::Stream& stream,
		const Constitutions::SoftInertia<luisa::compute::Buffer>&	constraint,
		const luisa::compute::Buffer<float3>&						sa_q,
		float														substep_dt,
		size_t														dispatch_count)
	{
		stream << _eval_shader(constraint,
			sa_q.view(),
			substep_dt,
			get_scene_params().use_quasi_static_mode,
			get_scene_params().use_static_mode,
			static_cast<uint>(get_scene_params().qs_inertia_mode))
					  .dispatch(dispatch_count);
	}

	double SoftInertiaEnergy::host_evaluate(const std::vector<float>& host_energy)
	{
		return host_energy[offset_inertia];
	}

	void SoftInertiaEnergy::host_evaluate(lcs::SimulationData<std::vector>& host_sim_data,
		lcs::MeshData<std::vector>&											host_mesh_data)
	{
		auto& inertia_data = host_sim_data.get_soft_inertia_data();
		if (inertia_data.is_valid())
		{
			CpuParallel::parallel_for(0,
				host_sim_data.num_verts_soft,
				[sa_q = std::span(host_sim_data.sa_q),
					sa_x_tilde = std::span(host_sim_data.sa_q_tilde),
					sa_q_is_fixed = std::span(host_sim_data.sa_q_is_fixed),
					sa_vert_mass = std::span(inertia_data.sa_soft_vert_mass),
					sa_stiffness_dirichlet = std::span(inertia_data.sa_stiffness_dirichlet),
					output_gradient = std::span(inertia_data.constraint_gradients),
					output_hessian = std::span(inertia_data.constraint_hessians),
					substep_dt = get_scene_params().get_substep_dt(),
					use_quasi_static_mode = get_scene_params().use_quasi_static_mode,
					use_static_mode = get_scene_params().use_static_mode,
					qs_inertia_mode = get_scene_params().qs_inertia_mode](const uint index)
				{
					const float h = substep_dt;
					const float h_2_inv = 1.f / (h * h);
					const bool	is_fixed = sa_q_is_fixed[index] != 0u;
					const bool	energy_gradient_active = !use_quasi_static_mode || is_fixed || qs_inertia_mode == QuasiStaticInertiaMode::Full;
					const bool	hessian_active = !use_static_mode && (!use_quasi_static_mode || is_fixed || qs_inertia_mode != QuasiStaticInertiaMode::Off);

					float3 x_k = sa_q[index];
					float3 x_tilde = sa_x_tilde[index];

					float	 mass = sa_vert_mass[index];
					float3	 gradient = Zero3;
					float3x3 hessian = luisa::make_float3x3(0.0f);
					if (energy_gradient_active)
					{
						gradient = mass * h_2_inv * (x_k - x_tilde);
					}
					if (hessian_active)
					{
						hessian = mass * h_2_inv * luisa::float3x3::eye(1.0f);
					}

					{
						const float stiffness_dirichlet = sa_stiffness_dirichlet[index];
						gradient = stiffness_dirichlet * gradient;
						hessian = stiffness_dirichlet * hessian;
					}
					{
						output_gradient[index] = gradient;
						output_hessian[index] = hessian;
					}
				});
		}
	}

} // namespace lcs
