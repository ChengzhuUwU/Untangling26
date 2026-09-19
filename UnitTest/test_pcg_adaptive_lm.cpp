// Unit tests for the adaptive Levenberg-Marquardt PCG solve semantics
// (Solver/LinearSolver/precond_cg.cpp::host_solve_core).
//
// Covered behavior:
//  1. healthy SPD systems converge at lambda = 0 with no retries;
//  2. indefinite / under-constrained systems escalate lambda and accept the
//     first successful candidate;
//  3. requested lambda (> 0) runs exactly one candidate and never escalates;
//  4. finite non-convergence returns the last finite direction with a
//     converged=false report, both after ladder exhaustion and in strict mode;
//  5. zero RHS converges immediately with a zero solution and no 0/0;
//  6. pcg_iter_count == 0 is a defined, explicitly non-converged outcome;
//  7. NaN/Inf inputs and invalid factors fail without returning a direction;
//  8. sa_cgA_diag is never modified by any attempt;
//  9. retry attempts are fully reset (repeat runs are identical);
// 10. tiny systems (num_dof < 256, including num_dof == 1) work.

#include "LinearSolver/precond_cg.h"
#include "SimulationCore/scene_params.h"
#include "Utils/cpu_parallel.h"

#include <cmath>
#include <cstdio>
#include <functional>
#include <iostream>
#include <memory>
#include <string>
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

	float3 diag3(float x, float y, float z)
	{
		return luisa::make_float3(x, y, z);
	}

	float3x3 diag3_matrix(float x, float y, float z)
	{
		return luisa::make_float3x3(diag3(x, 0.f, 0.f), diag3(0.f, y, 0.f), diag3(0.f, 0.f, z));
	}

	struct Fixture
	{
		SimulationData<std::vector> sim_data;

		explicit Fixture(uint num_dof)
		{
			sim_data.num_dof = num_dof;
			sim_data.sa_cgX.resize(num_dof);
			sim_data.sa_cgB.resize(num_dof);
			sim_data.sa_cgA_diag.resize(num_dof);
			sim_data.sa_cgMinv.resize(num_dof);
			sim_data.sa_cgP.resize(num_dof);
			sim_data.sa_cgQ.resize(num_dof);
			sim_data.sa_cgR.resize(num_dof);
			sim_data.sa_cgZ.resize(num_dof);
		}

		void fill_diagonal_system(const std::vector<float3>& diag, const std::vector<float3>& rhs)
		{
			CpuParallel::parallel_for(0, static_cast<uint>(diag.size()),
				[&](const uint i)
				{
					sim_data.sa_cgA_diag[i] = diag3_matrix(diag[i].x, diag[i].y, diag[i].z);
					sim_data.sa_cgB[i] = rhs[i];
				});
		}

		PcgSolveReport solve(std::function<void(const std::vector<float3>&, std::vector<float3>&)> spmv,
			std::optional<float> regularization_override = std::nullopt)
		{
			ConjugateGradientSolver solver;
			solver.set_data(nullptr, nullptr, &sim_data, nullptr);
			return solver.host_solve_core(std::move(spmv), regularization_override);
		}

		// q_i = D_i p_i, reading the live (possibly pollutable) diagonal.
		std::function<void(const std::vector<float3>&, std::vector<float3>&)> diagonal_spmv()
		{
			return [this](const std::vector<float3>& p, std::vector<float3>& q)
			{
				CpuParallel::parallel_for(0, static_cast<uint>(p.size()),
					[&](const uint i)
					{ q[i] = sim_data.sa_cgA_diag[i] * p[i]; });
			};
		}

		// q_i = D_i p_i - coupling * (p_{i-1} + p_{i+1}): a coupled SPD system
		// whose operator is NOT expressible from the diagonal alone.
		std::function<void(const std::vector<float3>&, std::vector<float3>&)> tridiagonal_spmv(float coupling)
		{
			return [this, coupling](const std::vector<float3>& p, std::vector<float3>& q)
			{
				const uint n = static_cast<uint>(p.size());
				CpuParallel::parallel_for(0, n,
					[&](const uint i)
					{
						float3 value = sim_data.sa_cgA_diag[i] * p[i];
						if (i > 0u)
							value = value - coupling * p[i - 1u];
						if (i + 1u < n)
							value = value - coupling * p[i + 1u];
						q[i] = value;
					});
			};
		}

		// (A + lambda S) with S_i = max(trace(D_i)/3, 0) * I rebuilt from the
		// current (assumed untouched) stored diagonal, plus the same coupling.
		std::function<void(const std::vector<float3>&, std::vector<float3>&)> tridiagonal_spmv_regularized(float coupling, float lambda)
		{
			return [this, coupling, lambda](const std::vector<float3>& p, std::vector<float3>& q)
			{
				const uint n = static_cast<uint>(p.size());
				CpuParallel::parallel_for(0, n,
					[&](const uint i)
					{
						const float3x3 d = sim_data.sa_cgA_diag[i];
						const float	   scale = (d[0][0] + d[1][1] + d[2][2]) * (1.0f / 3.0f);
						float3		   value = sim_data.sa_cgA_diag[i] * p[i] + lambda * (scale > 0.f ? scale : 0.f) * p[i];
						if (i > 0u)
							value = value - coupling * p[i - 1u];
						if (i + 1u < n)
							value = value - coupling * p[i + 1u];
						q[i] = value;
					});
			};
		}

		std::vector<float> capture_diagonal() const
		{
			std::vector<float> out;
			out.reserve(sim_data.sa_cgA_diag.size() * 3u);
			for (const auto& block : sim_data.sa_cgA_diag)
			{
				out.push_back(block[0][0]);
				out.push_back(block[1][1]);
				out.push_back(block[2][2]);
			}
			return out;
		}
	};

	// Indefinite system used by tests 2/3/9: DOF 1 has a slightly negative x
	// diagonal. With lambda = 0 the Jacobi inverse zeroes that row, so the x
	// component of the solution is unreachable and PCG cannot converge. The
	// candidate lambda = 1e-2 is the first whose lambda * trace(D)/3
	// augmentation makes that diagonal positive.
	struct IndefiniteSystem
	{
		std::vector<float3> diag{ diag3(1.f, 1.f, 1.f), diag3(-1e-3f, 1.f, 1.f) };
		std::vector<float3> rhs{ diag3(1.f, 1.f, 1.f), diag3(1.f, 1.f, 1.f) };
	};
} // namespace

// Case names let the runner split LUISA_ERROR-aborting non-finite scenarios
// (which terminate the process by design) into their own invocations:
//   all cases by default, or a single one via argv[1].
int main(int argc, char** argv)
{
	// CpuParallel dispatches through the luisa fiber pool; the RAII
	// scheduler below plays the role SolverInterface plays in production.
	luisa::fiber::scheduler fiber_scheduler;

	auto scene_params = std::make_shared<lcs::SceneParams>();
	lcs::set_scene_params_ptr(scene_params);
	// Initial solves are unregularized; the Newton delta gate opts into LM.
	require(scene_params->pcg_lm_adaptive, "adaptive LM is enabled by default");
	const std::string only_case = argc > 1 ? std::string(argv[1]) : std::string();
	// A no-argument run skips only the LUISA_ERROR-aborting non-finite case.
	const auto wants = [&only_case](const char* name)
	{
		if (only_case.empty())
			return std::string(name) != "case_nonfinite";
		return only_case == name;
	};

	// ------------------------------------------------------------------
	// 1. Healthy SPD system: lambda = 0 accepted, no retries, no escalation.
	// ------------------------------------------------------------------
	if (wants("case_spd"))
	{
		Fixture				fx(10);
		std::vector<float3> diag(10);
		std::vector<float3> rhs(10);
		for (uint i = 0; i < 10; i++)
		{
			const float d = 0.5f + 0.25f * static_cast<float>(i);
			diag[i] = diag3(d, d * 1.5f, d * 2.f);
			rhs[i] = diag3(1.f + static_cast<float>(i), -2.f + static_cast<float>(i), 3.f);
		}
		fx.fill_diagonal_system(diag, rhs);
		scene_params->pcg_iter_count = 100;

		const auto report = fx.solve(fx.diagonal_spmv());
		require(report.converged, "SPD system converges");
		require(report.regularization_factor == 0.0f, "SPD system uses lambda = 0");
		require(report.regularization_retries == 0u, "SPD system performs no retries");
		require(!report.adaptive_retry_used, "SPD system does not report adaptive retry");
		require(report.relative_residual <= 1e-4f, "SPD system reaches the residual tolerance");
		require(report.iterations > 0u, "SPD system takes at least one PCG step");
	}

	// ------------------------------------------------------------------
	// 2. Indefinite block: single strict candidate, finite non-converged
	//    return (the Newton delta gate owns escalation, not the solve).
	// ------------------------------------------------------------------
	const IndefiniteSystem system;
	if (wants("case_escalation"))
	{
		Fixture fx(2);
		fx.fill_diagonal_system(system.diag, system.rhs);
		const auto diag_backup = fx.capture_diagonal();
		scene_params->pcg_iter_count = 50;

		const auto report = fx.solve(fx.diagonal_spmv());
		require(!report.converged, "indefinite system does not report success under the strict single candidate");
		require(report.regularization_factor == 0.0f, "strict solve never substitutes the requested lambda");
		require(report.regularization_retries == 0u, "no in-solve retry ladder exists");
		require(!report.adaptive_retry_used, "escalation is owned by the Newton delta gate, never the solve");
		require(!report.failure_reason.empty(), "non-convergence reports its cause");
		require(std::isfinite(fx.sim_data.sa_cgX[0].x), "strict failure still returns a finite direction");
		require(diag_backup == fx.capture_diagonal(), "the stored diagonal stays unmodified");
	}

	// ------------------------------------------------------------------
	// 2b. One-shot forced regularization (Newton delta gate path): the forced
	//     lambda is consumed exactly once and reported unchanged.
	// ------------------------------------------------------------------
	if (wants("case_two_step"))
	{
		Fixture fx(2);
		fx.fill_diagonal_system(system.diag, system.rhs);
		scene_params->pcg_iter_count = 50;
		scene_params->pcg_lm_escalation_factor = 0.1f;
		ConjugateGradientSolver solver;
		solver.set_data(nullptr, nullptr, &fx.sim_data, nullptr);
		const auto report = solver.host_solve_core(fx.diagonal_spmv(), 0.1f);
		require(report.converged, "forced lambda = 0.1 converges on the indefinite system");
		require(std::abs(report.regularization_factor - 0.1f) < 1e-9f,
			"the forced one-shot lambda is reported unchanged");
		require(report.regularization_retries == 0u, "a forced solve runs a single candidate");
		const auto next_report = solver.host_solve_core(fx.diagonal_spmv());
		require(next_report.regularization_factor == 0.0f,
			"a call-scoped override cannot leak into the next solve");
	}

	// ------------------------------------------------------------------
	// 3. Internally requested lambda: one candidate, no escalation, still converges here.
	// ------------------------------------------------------------------
	if (wants("case_fixed_lambda"))
	{
		Fixture fx(2);
		fx.fill_diagonal_system(system.diag, system.rhs);
		scene_params->pcg_iter_count = 50;

		const auto report = fx.solve(fx.diagonal_spmv(), 0.1f);
		require(report.converged, "requested lambda = 0.1 converges on the indefinite system");
		require(std::abs(report.regularization_factor - 1e-1f) < 1e-9f, "requested lambda is reported unchanged");
		require(report.regularization_retries == 0u, "requested lambda never escalates");
		require(!report.adaptive_retry_used, "requested lambda reports no adaptive retry");
	}

	// ------------------------------------------------------------------
	// 9. Retry reset: a second adaptive run must be identical to the first.
	// ------------------------------------------------------------------
	if (wants("case_retry_reset"))
	{
		Fixture fx(2);
		scene_params->pcg_iter_count = 50;
		auto run = [&]()
		{
			fx.fill_diagonal_system(system.diag, system.rhs);
			return fx.solve(fx.diagonal_spmv());
		};

		const auto first = run();
		const auto second = run();
		require(second.converged == first.converged && second.regularization_retries == first.regularization_retries && second.iterations == first.iterations && std::abs(second.regularization_factor - first.regularization_factor) < 1e-12f,
			"repeated solves are identical (retry state is fully reset)");
	}

	// ------------------------------------------------------------------
	// 3b. Explicitly disabled adaptive LM: factor 0 is a strict
	//     unregularized single-candidate solve - no retries even on failure.
	// ------------------------------------------------------------------
	{
		Fixture				fx(10);
		std::vector<float3> diag(10);
		std::vector<float3> rhs(10);
		for (uint i = 0; i < 10; i++)
		{
			const float d = 1.f + static_cast<float>(i);
			diag[i] = diag3(d, d, d);
			rhs[i] = diag3(1.f, 1.f, 1.f);
		}
		fx.fill_diagonal_system(diag, rhs);
		scene_params->pcg_lm_adaptive = false;
		scene_params->pcg_iter_count = 100;
		const auto report = fx.solve(fx.diagonal_spmv());
		require(report.converged && report.regularization_factor == 0.0f && report.regularization_retries == 0u,
			"strict default (LM off) solves a healthy SPD system at lambda = 0 with no ladder");

		// Same strict mode on the indefinite system is covered below; it now
		// returns the finite non-converged direction instead of aborting.
		scene_params->pcg_lm_adaptive = true;
	}

	// ------------------------------------------------------------------
	// 4. Unsalvageable block: the single strict candidate fails fast with a
	//    non-converged report (no in-solve ladder to exhaust).
	// ------------------------------------------------------------------
	if (wants("case_all_fail"))
	{
		Fixture					  fx(1);
		const std::vector<float3> diag = { diag3(-1.f, -1.f, -1.f) };
		const std::vector<float3> rhs = { diag3(1.f, 1.f, 1.f) };
		fx.fill_diagonal_system(diag, rhs);
		scene_params->pcg_iter_count = 50;

		// trace(D)/3 < 0 clamps to zero, so no regularization can repair this
		// block; the solve must return an explicit non-converged report.
		const auto report = fx.solve(fx.diagonal_spmv());
		require(!report.converged, "unsalvageable negative block does not report success");
		require(report.regularization_retries == 0u, "no in-solve retry ladder exists to exhaust");
		require(!report.failure_reason.empty(), "unsalvageable block reports a failure reason");
		require(report.iterations == 0u, "unsalvageable block takes no PCG steps");
		require(std::isfinite(fx.sim_data.sa_cgX[0].x) && std::isfinite(fx.sim_data.sa_cgX[0].y) && std::isfinite(fx.sim_data.sa_cgX[0].z),
			"a failed strict solve still returns a finite direction");
	}

	// ------------------------------------------------------------------
	// 4b. Strict default (LM off) on a system lambda=0 cannot solve: single
	//     candidate, finite non-converged return, no escalation.
	// ------------------------------------------------------------------
	if (wants("case_strict_zero_singular"))
	{
		Fixture fx(2);
		fx.fill_diagonal_system(system.diag, system.rhs);
		scene_params->pcg_lm_adaptive = false;
		scene_params->pcg_iter_count = 50;
		const auto report = fx.solve(fx.diagonal_spmv());
		require(!report.converged, "strict lambda=0 on an unsolvable system does not report success");
		require(report.regularization_retries == 0u, "strict lambda=0 failure does not enter the ladder");
		require(!report.failure_reason.empty(), "strict lambda=0 failure reports its cause");
	}

	// ------------------------------------------------------------------
	// 4c. Negative curvature returns X from before the rejected step.
	// ------------------------------------------------------------------
	if (wants("case_negative_curvature"))
	{
		Fixture fx(2);
		fx.fill_diagonal_system(std::vector<float3>(2, diag3(1.f, 1.f, 1.f)),
			std::vector<float3>{ diag3(1.f, 2.f, 3.f), diag3(-1.f, 1.f, 0.5f) });
		scene_params->pcg_lm_adaptive = false;
		scene_params->pcg_iter_count = 10u;
		const auto negative_spmv = [](const std::vector<float3>& p, std::vector<float3>& q)
		{
			CpuParallel::parallel_for(0, static_cast<uint>(p.size()),
				[&](const uint i)
				{ q[i] = -p[i]; });
		};
		const auto report = fx.solve(negative_spmv);
		require(!report.converged && report.failure_reason == "non-positive pT(A+lambda S)p",
			"negative curvature returns a distinct non-converged report");
		require(report.iterations == 0u && report.infinity_norm == 0.0f,
			"negative curvature returns X from before the rejected step");
	}

	// ------------------------------------------------------------------
	// 4d. Iteration-budget exhaustion reports the plateau residual and keeps
	//     the last finite, non-zero iterate.
	// ------------------------------------------------------------------
	if (wants("case_plateau"))
	{
		const uint			n = 10u;
		Fixture				fx(n);
		std::vector<float3> rhs(n);
		for (uint i = 0u; i < n; i++)
			rhs[i] = diag3(1.f + static_cast<float>(i), -0.5f * static_cast<float>(i), 0.25f);
		fx.fill_diagonal_system(std::vector<float3>(n, diag3(3.f, 3.f, 3.f)), rhs);
		scene_params->pcg_lm_adaptive = false;
		scene_params->pcg_iter_count = 1u;
		const auto report = fx.solve(fx.tridiagonal_spmv(1.0f));
		require(!report.converged && report.failure_reason.starts_with("plateau: relative residual "),
			"iteration exhaustion reports a plateau residual");
		require(report.iterations == 1u && report.relative_residual > 1e-6f,
			"plateau report preserves the completed-step count and residual");
		require(std::isfinite(report.infinity_norm) && report.infinity_norm > 0.0f,
			"plateau returns the last finite non-zero iterate");
	}

	// ------------------------------------------------------------------
	// 5. Zero RHS: immediate convergence, zero solution, no NaN.
	// ------------------------------------------------------------------
	if (wants("case_zero_rhs"))
	{
		Fixture				fx(4);
		std::vector<float3> diag(4);
		for (uint i = 0; i < 4; i++)
			diag[i] = diag3(1.f + static_cast<float>(i), 2.f, 3.f);
		fx.fill_diagonal_system(diag, std::vector<float3>(4, diag3(0.f, 0.f, 0.f)));
		scene_params->pcg_iter_count = 50;

		const auto report = fx.solve(fx.diagonal_spmv());
		require(report.converged, "zero RHS converges immediately");
		require(report.relative_residual == 0.0f, "zero RHS reports zero residual (no 0/0)");
		bool zero_solution = true;
		for (const auto& x : fx.sim_data.sa_cgX)
			if (!std::isfinite(x.x) || !std::isfinite(x.y) || !std::isfinite(x.z) || x.x != 0.f || x.y != 0.f || x.z != 0.f)
				zero_solution = false;
		require(zero_solution, "zero RHS yields a finite zero solution");
	}

	// ------------------------------------------------------------------
	// 6. pcg_iter_count == 0: defined outcome, explicitly non-converged.
	// ------------------------------------------------------------------
	if (wants("case_zero_iter"))
	{
		Fixture fx(3);
		fx.fill_diagonal_system(std::vector<float3>(3, diag3(1.f, 1.f, 1.f)),
			std::vector<float3>{ diag3(1.f, 2.f, 3.f), diag3(1.f, 2.f, 3.f), diag3(1.f, 2.f, 3.f) });
		scene_params->pcg_iter_count = 0u;

		const auto report = fx.solve(fx.diagonal_spmv());
		require(!report.converged, "zero iteration budget is not reported as success");
		require(!report.failure_reason.empty(), "zero iteration budget reports its reason");
		require(std::abs(report.relative_residual - 1.0f) < 1e-9f, "zero iteration budget reports the unsolved residual");
	}

	// ------------------------------------------------------------------
	// 7. NaN / Inf right-hand side and invalid factors: loud failure.
	// ------------------------------------------------------------------
	if (wants("case_nonfinite"))
	{
		Fixture fx(2);
		scene_params->pcg_iter_count = 50;

		fx.fill_diagonal_system(std::vector<float3>(2, diag3(1.f, 1.f, 1.f)),
			std::vector<float3>{ diag3(1.f, 1.f, 1.f), diag3(std::nanf(""), 1.f, 1.f) });
		auto report = fx.solve(fx.diagonal_spmv());
		require(!report.converged, "NaN right-hand side does not report success");
		require(report.failure_reason.find("non-finite") != std::string::npos,
			"NaN right-hand side reports the non-finite cause");

		fx.fill_diagonal_system(std::vector<float3>(2, diag3(1.f, 1.f, 1.f)),
			std::vector<float3>{ diag3(1.f, 1.f, 1.f), diag3(1.f, 1.f, 1.f) });

		report = fx.solve(fx.diagonal_spmv(), -0.5f);
		require(!report.converged && !report.failure_reason.empty(), "negative factor fails loudly");

		report = fx.solve(fx.diagonal_spmv(), std::nanf(""));
		require(!report.converged && !report.failure_reason.empty(), "NaN factor fails loudly");
	}

	// ------------------------------------------------------------------
	// 10. Tiny systems, including the num_dof == 1 reduction corner.
	// ------------------------------------------------------------------
	if (wants("case_tiny"))
	{
		scene_params->pcg_iter_count = 200;
		bool all_converged = true;
		for (const uint n : { 1u, 2u, 3u, 5u, 255u, 256u, 257u })
		{
			Fixture				fx(n);
			std::vector<float3> diag(n);
			std::vector<float3> rhs(n);
			for (uint i = 0; i < n; i++)
			{
				const float d = 1.f + 0.1f * static_cast<float>(i % 7);
				diag[i] = diag3(d, d, d);
				rhs[i] = diag3(1.f, -1.f, 0.5f);
			}
			fx.fill_diagonal_system(diag, rhs);
			const auto report = fx.solve(fx.diagonal_spmv());
			if (!report.converged || report.regularization_factor != 0.0f)
			{
				all_converged = false;
				std::cout << "        tiny system n=" << n << " failed" << std::endl;
			}
		}
		require(all_converged, "tiny systems (1, 2, 3, 5, 255, 256, 257 DOF) all converge at lambda = 0");
	}

	// ------------------------------------------------------------------
	// 11. Coupled SPD system with independent residual verification, at
	//     lambda = 0 and with a requested lambda (operator and preconditioner
	//     must use the same regularized matrix; the stored diagonal stays
	//     original).
	// ------------------------------------------------------------------
	if (wants("case_coupled"))
	{
		const uint			n = 50;
		Fixture				fx(n);
		std::vector<float3> diag(n);
		std::vector<float3> rhs(n);
		for (uint i = 0; i < n; i++)
		{
			diag[i] = diag3(3.f, 3.f, 3.f); // scale = trace/3 = 3
			rhs[i] = diag3(std::sin(static_cast<float>(i)), std::cos(0.5f * static_cast<float>(i)), 1.f);
		}
		fx.fill_diagonal_system(diag, rhs);
		auto spmv = fx.tridiagonal_spmv(1.f);

		scene_params->pcg_iter_count = 500;
		const auto plain = fx.solve(spmv);
		require(plain.converged && plain.regularization_factor == 0.0f, "coupled SPD system converges at lambda = 0");

		auto residual_of = [&](const std::function<void(const std::vector<float3>&, std::vector<float3>&)>& op)
		{
			std::vector<float3> ax(n, diag3(0.f, 0.f, 0.f));
			op(fx.sim_data.sa_cgX, ax);
			double num = 0.0;
			double den = 0.0;
			for (uint i = 0; i < n; i++)
			{
				const float3 r = ax[i] - fx.sim_data.sa_cgB[i];
				num += static_cast<double>(luisa::dot(r, r));
				den += static_cast<double>(luisa::dot(fx.sim_data.sa_cgB[i], fx.sim_data.sa_cgB[i]));
			}
			return std::sqrt(num / den);
		};
		require(residual_of(spmv) < 1e-3, "lambda = 0 solution satisfies A x = b independently");

		// Internally requested lambda: the solution must satisfy the REGULARIZED system
		// (A + lambda S) x = b, with S built from the untouched diagonal.
		const auto regularized = fx.solve(spmv, 0.05f);
		require(regularized.converged && std::abs(regularized.regularization_factor - 0.05f) < 1e-9f,
			"coupled system converges with requested lambda = 0.05");

		require(residual_of(fx.tridiagonal_spmv_regularized(1.f, 0.05f)) < 1e-3,
			"regularized solution satisfies the regularized system built from the unmodified diagonal");
	}

	std::cout << (g_failures == 0 ? "ALL PCG TESTS PASSED" : "PCG TEST FAILURES PRESENT") << std::endl;
	return g_failures == 0 ? 0 : 1;
}
