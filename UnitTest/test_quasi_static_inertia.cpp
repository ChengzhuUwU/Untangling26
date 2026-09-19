// Unit tests for quasi-static / dynamic inertia semantic consistency
// (Solver/Energies/soft_inertia_energy.cpp, abd_inertia_energy.cpp).
//
// The host evaluation path must implement the configured quasi-static policy:
//
//   dynamic          : E_i = s_i * m_i / (2 h^2) * ||q_i - q~_i||^2 for every DOF
//                      (s_i = dirichlet stiffness scale, m_i = mass)
//   quasi-static off  : free DOFs have E = g = H = 0
//   quasi-static H-only (default): free DOFs have E = g = 0 and
//                                  H = s_i * m_i / h^2
//   quasi-static full : free DOFs retain E, g, and H
//   quasi-static fixed: the Dirichlet/prediction anchor survives ->
//                       E = s_i * m_i / (2 h^2) * ||q_i - q~_i||^2
//
// Consistency is verified two ways per mode:
//  - analytic gradient/Hessian values of the quadratic above, and
//  - central finite differences of the energy and of the gradient.
// Soft vertices (free and fixed) and ABD bodies (free and fixed) are covered.

#include "Energies/abd_inertia_energy.h"
#include "Energies/soft_inertia_energy.h"
#include "SimulationCore/scene_params.h"
#include "SimulationCore/simulation_data.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <memory>
#include <vector>

using namespace lcs;
using luisa::float3;
using luisa::float3x3;

namespace
{
	int g_failures = 0;

	void require(bool condition, const char* message)
	{
		if (condition)
		{
			std::cout << "[PASS] " << message << std::endl;
		}
		else
		{
			g_failures++;
			std::cout << "[FAIL] " << message << std::endl;
		}
	}

	float3 make_v(float x, float y, float z)
	{
		return luisa::make_float3(x, y, z);
	}

	float dot3(const float3& a, const float3& b)
	{
		return luisa::dot(a, b);
	}

	struct InertiaFixture
	{
		// 4 soft vertices (0,1 free; 2,3 fixed) + 2 ABD bodies
		// (body 0 free at DOFs 4..7; body 1 fixed at DOFs 8..11).
		static constexpr uint num_soft = 4u;
		static constexpr uint num_bodies = 2u;
		static constexpr uint num_dof = num_soft + 4u * num_bodies;

		SimulationData<std::vector> sim_data;
		MeshData<std::vector>		mesh_data;
		std::vector<float>			soft_mass{ 2.0f, 1.5f, 1.0f, 0.7f };
		std::vector<float>			soft_stiffness{ 1.0f, 1.0f, 1e6f, 1e6f };
		std::vector<uint>			soft_is_fixed{ 0u, 0u, 1u, 1u };
		float						abd_stiffness_free = 1.0f;
		float						abd_stiffness_fixed = 1e6f;

		InertiaFixture()
		{
			sim_data.num_verts_soft = num_soft;
			sim_data.num_affine_bodies = num_bodies;
			sim_data.num_dof = num_dof;
			sim_data.sa_q.resize(num_dof);
			sim_data.sa_q_tilde.resize(num_dof);
			sim_data.sa_q_is_fixed.resize(num_dof);

			auto& inertia = sim_data.get_soft_inertia_data();
			inertia.constraint_indices.resize(num_soft);
			inertia.sa_soft_vert_mass.resize(num_soft);
			inertia.sa_stiffness_dirichlet.resize(num_soft);
			inertia.constraint_gradients.resize(num_soft);
			inertia.constraint_hessians.resize(num_soft);

			auto& abd = sim_data.get_abd_inertia_data();
			abd.constraint_indices.resize(num_bodies);
			abd.sa_stiffness_dirichlet.resize(num_bodies);
			abd.sa_affine_bodies_mass_matrix.resize(num_bodies);
			abd.constraint_gradients.resize(4u * num_bodies);
			abd.constraint_hessians.resize(16u * num_bodies);

			for (uint i = 0; i < num_soft; i++)
			{
				inertia.constraint_indices[i] = i; // identity mapping (see DOF layout note)
				inertia.sa_soft_vert_mass[i] = soft_mass[i];
				inertia.sa_stiffness_dirichlet[i] = soft_stiffness[i];
				sim_data.sa_q_is_fixed[i] = soft_is_fixed[i];
			}

			// body 0 free, body 1 fixed
			for (uint b = 0; b < num_bodies; b++)
			{
				const uint base = num_soft + 4u * b;
				abd.constraint_indices[b] = luisa::make_uint4(base + 0u, base + 1u, base + 2u, base + 3u);
				abd.sa_stiffness_dirichlet[b] = b == 0u ? abd_stiffness_free : abd_stiffness_fixed;
				for (uint k = 0; k < 4u; k++)
					sim_data.sa_q_is_fixed[base + k] = b == 0u ? 0u : 1u;

				// Compressed 4x4 mass matrix: translation mass 3, small terms
				// coupling translation and linear rows so the gradient is not
				// diagonal in body space.
				float4x4 m{};
				for (uint r = 0; r < 4u; r++)
					for (uint c = 0; c < 4u; c++)
						m[r][c] = (r == c) ? (r == 0u ? 3.0f : 0.5f) : (r + c == 3u ? 0.05f : 0.0f);
				abd.sa_affine_bodies_mass_matrix[b] = m;
			}

			// Distinct positions and predictions per DOF.
			for (uint i = 0; i < num_dof; i++)
			{
				const float f = static_cast<float>(i);
				sim_data.sa_q[i] = make_v(0.1f * f + 0.3f, -0.2f * f - 0.1f, 0.15f * f + 0.05f);
				sim_data.sa_q_tilde[i] = make_v(-0.05f * f + 0.02f, 0.08f * f - 0.4f, -0.11f * f + 0.2f);
			}
		}

		void evaluate()
		{
			SoftInertiaEnergy soft{ luisa::compute::BufferView<float3>{},
				luisa::compute::BufferView<uint>{},
				luisa::compute::BufferView<float>{} };
			soft.host_evaluate(sim_data, mesh_data);

			AbdInertiaEnergy abd{ luisa::compute::BufferView<float3>{},
				luisa::compute::BufferView<uint>{},
				luisa::compute::BufferView<float>{} };
			abd.host_evaluate(sim_data, mesh_data);
		}

		// Energy of the soft quadratic at an arbitrary q (the mathematical
		// definition the host path must express).
		double soft_energy_at(uint i, const std::vector<float3>& q) const
		{
			const bool active = !use_quasi_static || sim_data.sa_q_is_fixed[i] != 0u || qs_inertia_mode == QuasiStaticInertiaMode::Full;
			if (!active)
				return 0.0;
			const double h2 = static_cast<double>(substep_dt) * static_cast<double>(substep_dt);
			const float3 d = q[i] - sim_data.sa_q_tilde[i];
			const double norm2 = static_cast<double>(dot3(d, d));
			return static_cast<double>(soft_stiffness[i]) * static_cast<double>(soft_mass[i]) * norm2 / (2.0 * h2);
		}

		// Energy of ABD body b at an arbitrary q.
		double abd_energy_at(uint b, const std::vector<float3>& q) const
		{
			const uint base = num_soft + 4u * b;
			const bool active = !use_quasi_static || sim_data.sa_q_is_fixed[base] != 0u || qs_inertia_mode == QuasiStaticInertiaMode::Full;
			if (!active)
				return 0.0;
			const double   h2 = static_cast<double>(substep_dt) * static_cast<double>(substep_dt);
			const auto&	   abd_data = sim_data.get_abd_inertia_data();
			const float4x4 m = abd_data.sa_affine_bodies_mass_matrix[b];
			double		   energy = 0.0;
			for (uint r = 0; r < 4u; r++)
				for (uint c = 0; c < 4u; c++)
				{
					const float3 dr = q[base + r] - sim_data.sa_q_tilde[base + r];
					const float3 dc = q[base + c] - sim_data.sa_q_tilde[base + c];
					energy += static_cast<double>(m[r][c]) * static_cast<double>(dot3(dr, dc));
				}
			const double s = b == 0u ? static_cast<double>(abd_stiffness_free) : static_cast<double>(abd_stiffness_fixed);
			return s * energy / (2.0 * h2);
		}

		bool				   use_quasi_static = false;
		QuasiStaticInertiaMode qs_inertia_mode = QuasiStaticInertiaMode::HessianOnly;
		float				   substep_dt = 1.0f / 60.0f;
	};

	template <typename T>
	bool approx(T a, T b, T tol)
	{
		const T diff = a > b ? a - b : b - a;
		return diff <= tol;
	}

	bool near_vec(const float3& a, const float3& b, float tol)
	{
		return approx(a.x, b.x, tol) && approx(a.y, b.y, tol) && approx(a.z, b.z, tol);
	}

	bool near_matrix(const float3x3& a, const float3x3& b, float tol)
	{
		for (uint c = 0; c < 3u; c++)
			for (uint r = 0; r < 3u; r++)
				if (!approx(a[c][r], b[c][r], tol))
					return false;
		return true;
	}
} // namespace

int main()
{
	// CpuParallel dispatches through the luisa fiber pool; the RAII
	// scheduler below plays the role SolverInterface plays in production.
	luisa::fiber::scheduler fiber_scheduler;

	auto scene_params = std::make_shared<lcs::SceneParams>();
	lcs::set_scene_params_ptr(scene_params);
	scene_params->implicit_dt = 1.0f / 60.0f;

	struct ModeCase
	{
		bool				   use_quasi_static;
		QuasiStaticInertiaMode qs_inertia_mode;
		const char*			   name;
	};
	const ModeCase mode_cases[] = {
		{ false, QuasiStaticInertiaMode::HessianOnly, "dynamic" },
		{ true, QuasiStaticInertiaMode::Off, "quasi-static/off" },
		{ true, QuasiStaticInertiaMode::HessianOnly, "quasi-static/hessian-only" },
		{ true, QuasiStaticInertiaMode::Full, "quasi-static/full" },
	};

	for (const auto& test_case : mode_cases)
	{
		InertiaFixture fx;
		fx.use_quasi_static = test_case.use_quasi_static;
		fx.qs_inertia_mode = test_case.qs_inertia_mode;
		scene_params->use_quasi_static_mode = fx.use_quasi_static;
		scene_params->use_static_mode = false;
		scene_params->qs_inertia_mode = fx.qs_inertia_mode;
		fx.evaluate();

		const char* mode = test_case.name;
		const float h2_inv = 1.0f / (fx.substep_dt * fx.substep_dt);

		// ---------- soft vertices: analytic gradient / Hessian ----------
		bool soft_analytic = true;
		for (uint i = 0; i < InertiaFixture::num_soft; i++)
		{
			auto&		   inertia = fx.sim_data.get_soft_inertia_data();
			const bool	   is_fixed = fx.sim_data.sa_q_is_fixed[i] != 0u;
			const bool	   energy_gradient_active = !fx.use_quasi_static || is_fixed || fx.qs_inertia_mode == QuasiStaticInertiaMode::Full;
			const bool	   hessian_active = !fx.use_quasi_static || is_fixed || fx.qs_inertia_mode != QuasiStaticInertiaMode::Off;
			const float	   gradient_scale = energy_gradient_active ? fx.soft_stiffness[i] : 0.0f;
			const float	   hessian_scale = hessian_active ? fx.soft_stiffness[i] : 0.0f;
			const float3   expected_g = gradient_scale * fx.soft_mass[i] * h2_inv * (fx.sim_data.sa_q[i] - fx.sim_data.sa_q_tilde[i]);
			const float3x3 expected_h = hessian_scale * fx.soft_mass[i] * h2_inv * float3x3::eye(1.0f);
			if (!near_vec(inertia.constraint_gradients[i], expected_g, 1e-3f * (1.0f + std::abs(expected_g.x))) || !near_matrix(inertia.constraint_hessians[i], expected_h, 1e-3f * expected_h[0][0]))
			{
				soft_analytic = false;
				std::cout << "        soft vertex " << i << " mismatch in " << mode << " mode" << std::endl;
			}
		}
		require(soft_analytic, "soft inertia gradient/Hessian match the selected dynamic/quasi-static policy");

		// ---------- soft vertices: finite differences ----------
		bool soft_fd = true;
		{
			const float eps = 1e-4f;
			for (uint i = 0; i < InertiaFixture::num_soft; i++)
			{
				auto& inertia = fx.sim_data.get_soft_inertia_data();
				// FD of energy -> gradient
				for (uint c = 0; c < 3u; c++)
				{
					auto bump = [&](float delta)
					{
						std::vector<float3> q = fx.sim_data.sa_q;
						if (c == 0u)
							q[i].x += delta;
						else if (c == 1u)
							q[i].y += delta;
						else
							q[i].z += delta;
						return fx.soft_energy_at(i, q);
					};
					const double fd = (bump(eps) - bump(-eps)) / (2.0 * eps);
					const float	 target = c == 0u ? inertia.constraint_gradients[i].x : c == 1u ? inertia.constraint_gradients[i].y
																								: inertia.constraint_gradients[i].z;
					if (!approx(static_cast<float>(fd), target, 1e-2f * (1.0f + std::abs(target))))
					{
						soft_fd = false;
						std::cout << "        soft energy FD mismatch vertex " << i << " component " << c
								  << " (" << fd << " vs " << target << ") in " << mode << " mode" << std::endl;
					}
				}
				// FD of gradient -> Hessian diagonal
				const bool hessian_only_regularizer = fx.use_quasi_static
					&& fx.sim_data.sa_q_is_fixed[i] == 0u
					&& fx.qs_inertia_mode == QuasiStaticInertiaMode::HessianOnly;
				if (hessian_only_regularizer)
					continue;
				for (uint c = 0; c < 3u; c++)
				{
					auto bump_gradient = [&](float delta)
					{
						std::vector<float3> q = fx.sim_data.sa_q;
						if (c == 0u)
							q[i].x += delta;
						else if (c == 1u)
							q[i].y += delta;
						else
							q[i].z += delta;
						std::vector<float3> saved = fx.sim_data.sa_q;
						fx.sim_data.sa_q = q;
						fx.evaluate();
						auto&		 inertia_data = fx.sim_data.get_soft_inertia_data();
						const float3 g = inertia_data.constraint_gradients[i];
						fx.sim_data.sa_q = saved;
						fx.evaluate();
						return g;
					};
					const float3 fd = (bump_gradient(eps) - bump_gradient(-eps)) / (2.f * eps);
					const float3 target_row = c == 0u ? make_v(inertia.constraint_hessians[i][0][0], inertia.constraint_hessians[i][0][1], inertia.constraint_hessians[i][0][2]) : c == 1u ? make_v(inertia.constraint_hessians[i][1][0], inertia.constraint_hessians[i][1][1], inertia.constraint_hessians[i][1][2])
																																														   : make_v(inertia.constraint_hessians[i][2][0], inertia.constraint_hessians[i][2][1], inertia.constraint_hessians[i][2][2]);
					const float	 row_max = std::max(std::abs(target_row.x), std::max(std::abs(target_row.y), std::abs(target_row.z)));
					if (!near_vec(fd, target_row, 1e-2f * (1.0f + row_max)))
					{
						soft_fd = false;
						std::cout << "        soft gradient FD mismatch vertex " << i << " component " << c
								  << " in " << mode << " mode" << std::endl;
					}
				}
			}
		}
		require(soft_fd, "soft inertia energy/gradient finite differences hold; derivative-backed Hessians match");

		// ---------- ABD bodies: analytic gradient / Hessian ----------
		bool abd_analytic = true;
		{
			auto& abd = fx.sim_data.get_abd_inertia_data();
			for (uint b = 0; b < InertiaFixture::num_bodies; b++)
			{
				const uint	   base = InertiaFixture::num_soft + 4u * b;
				const bool	   is_fixed = fx.sim_data.sa_q_is_fixed[base] != 0u;
				const bool	   energy_gradient_active = !fx.use_quasi_static || is_fixed || fx.qs_inertia_mode == QuasiStaticInertiaMode::Full;
				const bool	   hessian_active = !fx.use_quasi_static || is_fixed || fx.qs_inertia_mode != QuasiStaticInertiaMode::Off;
				const float	   base_scale = (b == 0u ? fx.abd_stiffness_free : fx.abd_stiffness_fixed) * h2_inv;
				const float	   gradient_scale = energy_gradient_active ? base_scale : 0.0f;
				const float	   hessian_scale = hessian_active ? base_scale : 0.0f;
				const float4x4 m = abd.sa_affine_bodies_mass_matrix[b];
				float3		   delta[4];
				for (uint k = 0; k < 4u; k++)
					delta[k] = fx.sim_data.sa_q[base + k] - fx.sim_data.sa_q_tilde[base + k];

				for (uint r = 0; r < 4u; r++)
				{
					float3 expected_g = make_v(0.f, 0.f, 0.f);
					for (uint c = 0; c < 4u; c++)
						expected_g = expected_g + m[r][c] * delta[c];
					expected_g = gradient_scale * expected_g;
					if (!near_vec(abd.constraint_gradients[4u * b + r], expected_g, 1e-3f * (1.0f + std::abs(expected_g.x))))
						abd_analytic = false;

					for (uint c = 0; c < 4u; c++)
					{
						const float3x3 expected_h = hessian_scale * m[r][c] * float3x3::eye(1.0f);
						if (!near_matrix(abd.constraint_hessians[16u * b + 4u * r + c], expected_h, 1e-3f * (1.0f + std::abs(expected_h[0][0]))))
							abd_analytic = false;
					}
				}
			}
		}
		require(abd_analytic, "ABD inertia gradient/Hessian match the selected dynamic/quasi-static policy");

		// ---------- ABD finite differences ----------
		bool abd_fd = true;
		{
			const float eps = 1e-4f;
			for (uint b = 0; b < InertiaFixture::num_bodies; b++)
			{
				const uint base = InertiaFixture::num_soft + 4u * b;
				for (uint k = 0; k < 4u; k++)
				{
					auto bump = [&](float delta)
					{
						std::vector<float3> q = fx.sim_data.sa_q;
						q[base + k].x += delta;
						q[base + k].y += 0.5f * delta;
						return fx.abd_energy_at(b, q);
					};
					const double fd = (bump(eps) - bump(-eps)) / (2.0 * eps);
					const float3 target = fx.sim_data.get_abd_inertia_data().constraint_gradients[4u * b + k];
					const float	 directional = target.x + 0.5f * target.y;
					if (!approx(static_cast<float>(fd), directional, 1e-2f * (1.0f + std::abs(directional))))
					{
						abd_fd = false;
						std::cout << "        ABD energy FD mismatch body " << b << " dof " << k
								  << " (" << fd << " vs " << directional << ") in " << mode << " mode" << std::endl;
					}
				}
			}
		}
		require(abd_fd, "ABD inertia energy is finite-difference consistent with its gradient in every mode");
	}

	// ---------- explicit quasi-static invariants ----------
	{
		InertiaFixture fx;
		fx.use_quasi_static = true;
		fx.qs_inertia_mode = QuasiStaticInertiaMode::HessianOnly;
		scene_params->use_quasi_static_mode = true;
		scene_params->use_static_mode = false;
		scene_params->qs_inertia_mode = QuasiStaticInertiaMode::HessianOnly;
		fx.evaluate();
		const float h2_inv = 1.0f / (fx.substep_dt * fx.substep_dt);

		auto& inertia = fx.sim_data.get_soft_inertia_data();
		bool  free_soft_hessian_only = true;
		for (const uint i : { 0u, 1u })
		{
			const float3x3 expected_h = fx.soft_mass[i] * h2_inv * float3x3::eye(1.0f);
			if (fx.soft_energy_at(i, fx.sim_data.sa_q) != 0.0
				|| !near_vec(inertia.constraint_gradients[i], make_v(0.f, 0.f, 0.f), 0.0f)
				|| !near_matrix(inertia.constraint_hessians[i], expected_h, 1e-3f * expected_h[0][0]))
				free_soft_hessian_only = false;
		}
		require(free_soft_hessian_only, "quasi-static free soft vertices have zero energy/gradient and retain m/h^2 Hessian");

		auto&		   abd = fx.sim_data.get_abd_inertia_data();
		bool		   abd_free_hessian_only = fx.abd_energy_at(0u, fx.sim_data.sa_q) == 0.0;
		const float4x4 free_mass_matrix = abd.sa_affine_bodies_mass_matrix[0];
		for (uint k = 0; k < 4u; k++)
		{
			if (!near_vec(abd.constraint_gradients[k], make_v(0.f, 0.f, 0.f), 0.0f))
				abd_free_hessian_only = false;
			for (uint c = 0; c < 4u; c++)
			{
				const float3x3 expected_h = float3x3::eye(h2_inv * free_mass_matrix[k][c]);
				if (!near_matrix(abd.constraint_hessians[4u * k + c], expected_h, 1e-3f * (1.0f + std::abs(expected_h[0][0]))))
					abd_free_hessian_only = false;
			}
		}
		require(abd_free_hessian_only, "quasi-static free ABD body has zero energy/gradient and retains M/h^2 Hessian");

		bool fixed_anchor_kept = true;
		for (const uint i : { 2u, 3u })
		{
			if (inertia.constraint_hessians[i][0][0] <= 0.0f)
				fixed_anchor_kept = false;
		}
		for (uint k = 0; k < 4u; k++)
		{
			const uint idx = 16u + 5u * k; // body 1 (fixed), diagonal block (k,k)
			if (abd.constraint_hessians[idx][0][0] <= 0.0f)
				fixed_anchor_kept = false;
		}
		require(fixed_anchor_kept, "quasi-static fixed DOFs keep a positive Dirichlet anchor");
	}

	// ---------- explicit off/full ablation invariants ----------
	{
		InertiaFixture fx;
		fx.use_quasi_static = true;
		scene_params->use_quasi_static_mode = true;
		scene_params->use_static_mode = false;

		fx.qs_inertia_mode = QuasiStaticInertiaMode::Off;
		scene_params->qs_inertia_mode = fx.qs_inertia_mode;
		fx.evaluate();
		auto& soft = fx.sim_data.get_soft_inertia_data();
		auto& abd = fx.sim_data.get_abd_inertia_data();
		require(near_matrix(soft.constraint_hessians[0], float3x3::eye(0.0f), 0.0f)
				&& near_matrix(abd.constraint_hessians[0], float3x3::eye(0.0f), 0.0f),
			"quasi-static inertia off removes the free-DOF Hessian");

		fx.qs_inertia_mode = QuasiStaticInertiaMode::Full;
		scene_params->qs_inertia_mode = fx.qs_inertia_mode;
		fx.evaluate();
		require(fx.soft_energy_at(0u, fx.sim_data.sa_q) > 0.0
				&& !near_vec(soft.constraint_gradients[0], make_v(0.f, 0.f, 0.f), 0.0f)
				&& soft.constraint_hessians[0][0][0] > 0.0f,
			"quasi-static inertia full restores free-DOF energy, gradient, and Hessian");
	}

	// Pure static mode does not assemble inertia curvature.
	{
		InertiaFixture fx;
		scene_params->use_quasi_static_mode = false;
		scene_params->use_static_mode = true;
		scene_params->qs_inertia_mode = QuasiStaticInertiaMode::HessianOnly;
		fx.evaluate();
		auto& soft = fx.sim_data.get_soft_inertia_data();
		auto& abd = fx.sim_data.get_abd_inertia_data();
		require(near_matrix(soft.constraint_hessians[0], float3x3::eye(0.0f), 0.0f)
				&& near_matrix(abd.constraint_hessians[0], float3x3::eye(0.0f), 0.0f),
			"static mode omits inertia Hessians");
	}

	std::cout << (g_failures == 0 ? "ALL INERTIA TESTS PASSED" : "INERTIA TEST FAILURES PRESENT") << std::endl;
	return g_failures == 0 ? 0 : 1;
}
