#include "abd_inertia_energy.h"
#include "SimulationCore/base_mesh.h"
#include "SimulationCore/scene_params.h"
#include "Utils/cpu_parallel.h"
#include "Utils/reduce_helper.h"

using namespace luisa::compute;

namespace lcs
{
	AbdInertiaEnergy::AbdInertiaEnergy(BufferView<float3> sa_q_tilde,
		BufferView<uint>								  sa_q_is_fixed,
		BufferView<float>								  sa_system_energy) noexcept
		: _sa_q_tilde(sa_q_tilde)
		, _sa_q_is_fixed(sa_q_is_fixed)
		, _sa_system_energy(sa_system_energy)
	{
	}

	void AbdInertiaEnergy::compile(AsyncCompiler& compiler)
	{
		luisa::compute::ShaderOption default_option = { .enable_debug_info = false };
		compiler.compile<1>(
			_shader,
			[sa_q_tilde = _sa_q_tilde,
				sa_q_is_fixed = _sa_q_is_fixed,
				sa_system_energy = _sa_system_energy](
				Var<Constitutions::AbdInertia<luisa::compute::Buffer>> constraint, Var<BufferView<float3>> sa_q, Float substep_dt, Bool use_quasi_static_mode, UInt qs_inertia_mode)
			{
				auto& sa_affine_bodies = constraint.constraint_indices;
				auto& sa_vert_mass = constraint.sa_affine_bodies_mass_matrix;
				auto& sa_stiffness_dirichlet = constraint.sa_stiffness_dirichlet;

				const Uint	body_idx = dispatch_id().x;
				const Uint4 affine_body = sa_affine_bodies->read(body_idx);
				const Bool	is_fixed = sa_q_is_fixed->read(affine_body[0]) != 0u;
				const Bool	energy_gradient_active = !use_quasi_static_mode | is_fixed | (qs_inertia_mode == static_cast<uint>(QuasiStaticInertiaMode::Full));

				Float energy = 0.0f;
				$if(energy_gradient_active)
				{
					const Float h = substep_dt;
					const Float squared_inv_dt = 1.0f / (h * h);
					Float		stiffness_dirichlet = sa_stiffness_dirichlet->read(body_idx);

					auto   mass_matrix = sa_vert_mass->read(body_idx);
					Float3 delta[4] = {
						sa_q.read(affine_body[0]) - sa_q_tilde->read(affine_body[0]),
						sa_q.read(affine_body[1]) - sa_q_tilde->read(affine_body[1]),
						sa_q.read(affine_body[2]) - sa_q_tilde->read(affine_body[2]),
						sa_q.read(affine_body[3]) - sa_q_tilde->read(affine_body[3]),
					};

					for (uint ii = 0; ii < 4; ii++)
					{
						for (uint jj = 0; jj < 4; jj++)
						{
							Float mass = mass_matrix[ii][jj];
							energy += squared_inv_dt * dot(delta[ii], delta[jj]) * mass / (2.0f);
						}
					}

					energy *= stiffness_dirichlet;
				};

				energy = ParallelIntrinsic::block_intrinsic_reduce(
					body_idx, energy, ParallelIntrinsic::warp_reduce_op_sum<float>);
				$if(body_idx % 256 == 0)
				{
					sa_system_energy->atomic(offset_abd_inertia).fetch_add(energy);
				};
			},
			default_option);

		// gradient/hessian evaluate shader for ABD inertia
		compiler.compile<1>(
			_eval_shader,
			[sa_q_tilde = _sa_q_tilde,
				sa_q_is_fixed = _sa_q_is_fixed](Var<Constitutions::AbdInertia<luisa::compute::Buffer>> constraint,
				Var<BufferView<float3>>																   sa_q,
				Float																				   substep_dt,
				Bool																				   use_quasi_static_mode,
				Bool																				   use_static_mode,
				UInt																				   qs_inertia_mode)
			{
				auto& sa_affine_bodies = constraint.constraint_indices;
				auto& sa_vert_mass = constraint.sa_affine_bodies_mass_matrix;
				auto& sa_stiffness_dirichlet = constraint.sa_stiffness_dirichlet;

				const UInt	body_idx = dispatch_id().x;
				const UInt4 affine_body = sa_affine_bodies->read(body_idx);
				const Bool	is_fixed = sa_q_is_fixed->read(affine_body[0]) != 0u;
				const Bool	energy_gradient_active = !use_quasi_static_mode | is_fixed | (qs_inertia_mode == static_cast<uint>(QuasiStaticInertiaMode::Full));
				const Bool	hessian_active = !use_static_mode & (!use_quasi_static_mode | is_fixed | (qs_inertia_mode != static_cast<uint>(QuasiStaticInertiaMode::Off)));

				const Float h = substep_dt;
				const Float h_2_inv = 1.0f / (h * h);

				Float3 delta[4] = { sa_q->read(affine_body[0]) - sa_q_tilde->read(affine_body[0]),
					sa_q->read(affine_body[1]) - sa_q_tilde->read(affine_body[1]),
					sa_q->read(affine_body[2]) - sa_q_tilde->read(affine_body[2]),
					sa_q->read(affine_body[3]) - sa_q_tilde->read(affine_body[3]) };

				Float4x4 mass_matrix = sa_vert_mass->read(body_idx);

				const Float stiffness = sa_stiffness_dirichlet->read(body_idx) * h_2_inv;
				Float		hessian_stiffness = 0.0f;
				Float3		gradient[4] = { Zero3, Zero3, Zero3, Zero3 };
				$if(energy_gradient_active)
				{
					for (uint ii = 0; ii < 4; ii++)
					{
						for (uint jj = 0; jj < 4; jj++)
						{
							gradient[ii] += mass_matrix[ii][jj] * delta[jj];
						}
					}
				};
				$if(hessian_active)
				{
					hessian_stiffness = stiffness;
				};

				auto& abd_gradients = constraint.constraint_gradients;
				auto& abd_hessians = constraint.constraint_hessians;

				abd_gradients->write(4 * body_idx + 0, stiffness * gradient[0]);
				abd_gradients->write(4 * body_idx + 1, stiffness * gradient[1]);
				abd_gradients->write(4 * body_idx + 2, stiffness * gradient[2]);
				abd_gradients->write(4 * body_idx + 3, stiffness * gradient[3]);

				for (uint ii = 0; ii < 4; ii++)
				{
					for (uint jj = 0; jj < 4; jj++)
					{
						abd_hessians->write(body_idx * 16 + ii * 4 + jj,
							hessian_stiffness * mass_matrix[ii][jj] * make_float3x3(1.0f));
					}
				}
			},
			default_option);
	}

	void AbdInertiaEnergy::device_compute_energy(luisa::compute::Stream& stream)
	{
		// Caller should use the typed overload below to dispatch with the appropriate buffers and counts.
	}

	void AbdInertiaEnergy::device_compute_energy(luisa::compute::Stream& stream,
		const Constitutions::AbdInertia<luisa::compute::Buffer>&		 constraint,
		const luisa::compute::Buffer<float3>&							 sa_q,
		float															 substep_dt,
		size_t															 dispatch_count)
	{
		stream << _shader(constraint,
			sa_q.view(),
			substep_dt,
			get_scene_params().use_quasi_static_mode,
			static_cast<uint>(get_scene_params().qs_inertia_mode))
					  .dispatch(dispatch_count);
	}

	void AbdInertiaEnergy::device_evaluate(luisa::compute::Stream& stream,
		const Constitutions::AbdInertia<luisa::compute::Buffer>&   constraint,
		const luisa::compute::Buffer<float3>&					   sa_q,
		float													   substep_dt,
		size_t													   dispatch_count)
	{
		stream << _eval_shader(constraint,
			sa_q.view(),
			substep_dt,
			get_scene_params().use_quasi_static_mode,
			get_scene_params().use_static_mode,
			static_cast<uint>(get_scene_params().qs_inertia_mode))
					  .dispatch(dispatch_count);
		// std::vector<float3>	  host_gradients(constraint.constraint_indices.size() * 4);
		// std::vector<float3x3> host_hessians(constraint.constraint_indices.size() * 16);
		// stream << constraint.constraint_gradients.copy_to(host_gradients.data());
		// stream << constraint.constraint_hessians.copy_to(host_hessians.data());
		// stream << synchronize();
		// for (uint i = 0; i < host_hessians.size(); i++)
		// {
		// 	LUISA_INFO("Hessian {} (Body {}, ii = {}, ii = {}) {}", i, i / 16, (i % 16) / 4, i % 4, host_hessians[i]);
		// }
	}

	double AbdInertiaEnergy::host_evaluate(const std::vector<float>& host_energy)
	{
		return host_energy[offset_abd_inertia];
	}

	void AbdInertiaEnergy::host_evaluate(lcs::SimulationData<std::vector>& host_sim_data,
		lcs::MeshData<std::vector>& /*host_mesh_data*/)
	{
		auto& abd_data = host_sim_data.get_abd_inertia_data();

		if (abd_data.is_valid())
		{
			CpuParallel::parallel_for(
				0,
				abd_data.get_num_indices(),
				[abd_gradients = std::span(abd_data.constraint_gradients),
					abd_hessians = std::span(abd_data.constraint_hessians),
					abd_indices = std::span(abd_data.constraint_indices),
					abd_mass_matrix = std::span(abd_data.sa_affine_bodies_mass_matrix),
					abd_stiffness_dirichlet = std::span(abd_data.sa_stiffness_dirichlet),
					abd_q = std::span(host_sim_data.sa_q),
					abd_q_tilde = std::span(host_sim_data.sa_q_tilde),
					sa_q_is_fixed = std::span(host_sim_data.sa_q_is_fixed),
					use_quasi_static_mode = get_scene_params().use_quasi_static_mode,
					use_static_mode = get_scene_params().use_static_mode,
					qs_inertia_mode = get_scene_params().qs_inertia_mode](const uint body_idx)
				{
					const float substep_dt = get_scene_params().get_substep_dt();
					const float h = substep_dt;
					const float h_2_inv = 1.f / (h * h);

					const uint4 indices = abd_indices[body_idx];

					float3	 delta_q[4] = { abd_q[indices[0]] - abd_q_tilde[indices[0]],
						abd_q[indices[1]] - abd_q_tilde[indices[1]],
						abd_q[indices[2]] - abd_q_tilde[indices[2]],
						abd_q[indices[3]] - abd_q_tilde[indices[3]] };
					float4x4 mass_matrix = abd_mass_matrix[body_idx];
					float3	 gradient[4] = { Zero3, Zero3, Zero3, Zero3 };

					const bool	is_fixed = sa_q_is_fixed[indices[0]] != 0u;
					const bool	energy_gradient_active = !use_quasi_static_mode || is_fixed || qs_inertia_mode == QuasiStaticInertiaMode::Full;
					const bool	hessian_active = !use_static_mode && (!use_quasi_static_mode || is_fixed || qs_inertia_mode != QuasiStaticInertiaMode::Off);
					const float stiffness = abd_stiffness_dirichlet[body_idx] * h_2_inv;
					const float hessian_stiffness = hessian_active ? stiffness : 0.0f;

					if (energy_gradient_active)
					{
						for (uint ii = 0; ii < 4; ii++)
						{
							for (uint jj = 0; jj < 4; jj++)
							{
								gradient[ii] += mass_matrix[ii][jj] * delta_q[jj];
							}
						}
					}

					abd_gradients[4 * body_idx + 0] = stiffness * gradient[0];
					abd_gradients[4 * body_idx + 1] = stiffness * gradient[1];
					abd_gradients[4 * body_idx + 2] = stiffness * gradient[2];
					abd_gradients[4 * body_idx + 3] = stiffness * gradient[3];

					for (uint ii = 0; ii < 4; ii++)
					{
						for (uint jj = 0; jj < 4; jj++)
						{
							abd_hessians[body_idx * 16 + ii * 4 + jj] =
								float3x3::eye(hessian_stiffness * mass_matrix[ii][jj]);
						}
					}
				},
				32);
		}
		// for (uint i = 0; i < abd_data.constraint_hessians.size(); i++)
		// {
		// 	LUISA_INFO("Hessian {} (Body {}, ii = {}, ii = {}) {}", i, i / 16, (i % 16) / 4, i % 4, abd_data.constraint_hessians[i]);
		// }
	}

} // namespace lcs
