#include "precond_cg.h"
#include "Core/lc_to_eigen.h"
#include "Core/scalar.h"
#include "SimulationCore/scene_params.h"
#include "Utils/cpu_parallel.h"
#include "Utils/reduce_helper.h"
#include "luisa/core/logging.h"
#include <array>
#include <atomic>
#include <cmath>
#include <string>

namespace lcs
{

	constexpr float pcg_epsilon = 1e-10f;
	constexpr float kPcgRelativeTolerance = 1e-6f; // ||r|| / ||b||, shared by CPU and GPU.
	// PCG always evaluates one regularization candidate; any adaptive re-solve is requested later
	// by the Newton delta gate rather than by a residual ladder inside PCG.

	inline bool is_finite_matrix(const float3x3& value)
	{
		for (uint column = 0; column < 3; column++)
			for (uint row = 0; row < 3; row++)
				if (!std::isfinite(value[column][row]))
					return false;
		return true;
	}
	inline bool is_finite_vector(const float3& value)
	{
		return std::isfinite(value.x) && std::isfinite(value.y) && std::isfinite(value.z);
	}
	inline bool are_finite_vectors(const std::vector<float3>& values)
	{
		for (const auto& value : values)
			if (!is_finite_vector(value))
				return false;
		return true;
	}

	inline void warn_pcg_nonconvergence(const char* backend, const PcgSolveReport& report)
	{
		LUISA_WARNING("{} PCG did not converge; returning the last finite direction: steps = {}, relative residual = {:.9g}, |x|_inf = {:.3e}, lambda = {:.1e} ({} retries), reason = {}",
			backend,
			report.iterations,
			report.relative_residual,
			report.infinity_norm,
			report.regularization_factor,
			report.regularization_retries,
			report.failure_reason.empty() ? "unspecified" : report.failure_reason);
	}

	inline void publish_pcg_report(const char* backend, const PcgSolveReport& report)
	{
		auto& scene_params = get_scene_params();
		if (scene_params.collect_iteration_debug)
		{
			auto& uint_stats = scene_params.prp_debug_info.uint_stats;
			uint_stats["frame"] = scene_params.current_frame;
			uint_stats["nonlinear_iter"] = scene_params.current_nonlinear_iter;
			uint_stats["pcg_regularization_retries"] = report.regularization_retries;
			uint_stats["pcg_iterations"] = report.iterations;
			// Cumulative counters let experiment drivers histogram the selected
			// lambda without parsing per-iteration logs.
			uint_stats["pcg_solve_total_count"] += 1u;
			uint_stats["pcg_solve_total_retries"] += report.regularization_retries;
			if (report.regularization_retries > 0u)
				uint_stats["pcg_solve_upgraded_count"] += 1u;
			scene_params.prp_debug_info.float_stats["pcg_regularization_factor"] = report.regularization_factor;
			scene_params.prp_debug_info.float_stats["pcg_relative_residual"] = report.relative_residual;
			scene_params.prp_debug_info.float_stats["pcg_infinity_norm"] = report.infinity_norm;
			scene_params.prp_debug_info.bool_stats["pcg_converged"] = report.converged;
			scene_params.prp_debug_info.bool_stats["pcg_adaptive_retry_used"] = report.adaptive_retry_used;
		}
		if (scene_params.print_pcg_info && report.converged)
		{
			LUISA_INFO("  In newton iter {:2}, {} PCG steps = {:3}, error = {:7.6f}, |x|_inf = {:.3e}, lambda = {:.1e} ({} retries{})",
				scene_params.current_nonlinear_iter,
				backend,
				report.iterations,
				report.relative_residual,
				report.infinity_norm,
				report.regularization_factor,
				report.regularization_retries,
				report.adaptive_retry_used ? ", adaptive" : "");
		}
	}

	inline float pcg_regularization_scale(const float3x3& diagA)
	{
		return max_scalar((diagA[0][0] + diagA[1][1] + diagA[2][2]) * (1.0f / 3.0f), 0.0f);
	}
	inline float3x3 make_pcg_jacobi_inverse(const float3x3& diagA)
	{
		const float diag_x = diagA[0][0];
		const float diag_y = diagA[1][1];
		const float diag_z = diagA[2][2];
		const float scale = max_scalar(max_scalar(abs_scalar(diag_x), abs_scalar(diag_y)), abs_scalar(diag_z));
		if (scale < pcg_epsilon)
		{
			return luisa::make_float3x3(0.0f);
		}
		const float min_diag = max_scalar(1e-6f, 1e-6f * scale);
		const float inv_x = diag_x > min_diag ? 1.0f / diag_x : 0.0f;
		const float inv_y = diag_y > min_diag ? 1.0f / diag_y : 0.0f;
		const float inv_z = diag_z > min_diag ? 1.0f / diag_z : 0.0f;
		return luisa::make_float3x3(
			luisa::make_float3(inv_x, 0.0f, 0.0f),
			luisa::make_float3(0.0f, inv_y, 0.0f),
			luisa::make_float3(0.0f, 0.0f, inv_z));
	}

	void ConjugateGradientSolver::compile(AsyncCompiler& compiler)
	{
		using namespace luisa::compute;

		luisa::compute::ShaderOption default_option(compiler.default_option());

		compiler.compile<1>(fn_reset_float3,
			[](Var<luisa::compute::BufferView<float3>> buffer)
			{ buffer->write(dispatch_id().x, luisa::compute::make_float3(0.0f)); });
		compiler.compile<1>(fn_reset_float,
			[](Var<luisa::compute::BufferView<float>> buffer)
			{ buffer->write(dispatch_id().x, Float(0.0f)); });
		compiler.compile<1>(fn_reset_uint,
			[](Var<luisa::compute::BufferView<uint>> buffer)
			{ buffer->write(dispatch_id().x, Uint(0u)); });

		auto fn_get_dispatch_block = [sa_num_dof = sim_data->sa_num_dof.view()]()
		{
			const Uint idx = 0;
			return get_dispatch_block(sa_num_dof->read(idx), 256);
		};

		compiler.compile<1>(
			fn_pcg_init,
			[sa_cgX = sim_data->sa_cgX.view(),
				sa_cgB = sim_data->sa_cgB.view(),
				sa_cgQ = sim_data->sa_cgQ.view(),
				sa_cgR = sim_data->sa_cgR.view(),
				sa_cgP = sim_data->sa_cgP.view(),
				sa_cgZ = sim_data->sa_cgZ.view()]()
			{
				const UInt	 vid = dispatch_id().x;
				const Float3 zero = make_float3(0.0f);
				sa_cgX->write(vid, zero);
				sa_cgR->write(vid, sa_cgB->read(vid));
				sa_cgP->write(vid, zero);
				sa_cgQ->write(vid, zero);
				sa_cgZ->write(vid, zero);
			},
			default_option);

		compiler.compile<1>(fn_dot_pq,
			[sa_cgP = sim_data->sa_cgP.view(),
				sa_cgQ = sim_data->sa_cgQ.view(),
				sa_block_result = sim_data->sa_block_result.view()]()
			{
				const UInt vid = dispatch_id().x;

				Float dot_pq = 0.0f;
				{
					Float3 p = sa_cgP->read(vid);
					Float3 q = sa_cgQ->read(vid);
					dot_pq = dot_vec(p, q);
				};

				dot_pq = ParallelIntrinsic::block_intrinsic_reduce(
					vid, dot_pq, ParallelIntrinsic::warp_reduce_op_sum<float>);

				$if(vid % 256 == 0)
				{
					sa_block_result->write(vid / 256, dot_pq);
				};
			});

		compiler.compile<1>(fn_dot_pq_second_pass,
			[sa_block_result = sim_data->sa_block_result.view(),
				sa_convergence = sim_data->sa_convergence.view(),
				fn_get_dispatch_block]()
			{
				const UInt vid = dispatch_id().x;
				Float	   dot_pq = 0.0f;
				$for(blockIdx, vid, fn_get_dispatch_block(), 256u)
				{
					dot_pq += sa_block_result->read(blockIdx);
				};

				dot_pq = ParallelIntrinsic::block_intrinsic_reduce(
					vid, dot_pq, ParallelIntrinsic::warp_reduce_op_sum<float>);

				$if(vid == 0)
				{
					sa_convergence->write(2, dot_pq);
				};
			});

		compiler.compile<1>(
			fn_pcg_update_p,
			[sa_cgP = sim_data->sa_cgP.view(),
				sa_cgZ = sim_data->sa_cgZ.view()](Float beta)
			{
				const UInt	 vid = dispatch_id().x;
				const Float3 p = sa_cgP->read(vid);
				sa_cgP->write(vid, sa_cgZ->read(vid) + beta * p);
			},
			default_option);

		compiler.compile<1>(
			fn_pcg_step,
			[sa_cgX = sim_data->sa_cgX.view(),
				sa_cgR = sim_data->sa_cgR.view(),
				sa_cgP = sim_data->sa_cgP.view(),
				sa_cgQ = sim_data->sa_cgQ.view()](Float alpha)
			{
				const UInt vid = dispatch_id().x;
				sa_cgX->write(vid, sa_cgX->read(vid) + alpha * sa_cgP->read(vid));
				sa_cgR->write(vid, sa_cgR->read(vid) - alpha * sa_cgQ->read(vid));
			},
			default_option);

		compiler.compile<1>(
			fn_pcg_add_regularization,
			[sa_cgA_diag = sim_data->sa_cgA_diag.view(),
				sa_cgP = sim_data->sa_cgP.view(),
				sa_cgQ = sim_data->sa_cgQ.view()](Float regularization_factor)
			{
				const UInt vid = dispatch_id().x;
				$if(regularization_factor > 0.0f)
				{
					const Float3x3 diagA = sa_cgA_diag->read(vid);
					const Float	   scale = max_scalar(
						(diagA[0][0] + diagA[1][1] + diagA[2][2]) * (1.0f / 3.0f),
						def<float>(0.0f));
					sa_cgQ->write(vid,
						sa_cgQ->read(vid) + regularization_factor * scale * sa_cgP->read(vid));
				};
			},
			default_option);

		compiler.compile<1>(
			fn_pcg_make_preconditioner,
			[sa_cgA_diag = sim_data->sa_cgA_diag.view(),
				sa_cgMinv = sim_data->sa_cgMinv.view(),
				sa_convergence = sim_data->sa_convergence.view()](Float regularization_factor)
			{
				const UInt	   vid = dispatch_id().x;
				const Float3x3 diagA = sa_cgA_diag->read(vid);
				Bool		   preconditioner_invalid = false;
				for (uint column = 0u; column < 3u; column++)
				{
					for (uint row = 0u; row < 3u; row++)
					{
						preconditioner_invalid = preconditioner_invalid
							| luisa::compute::isnan(diagA[column][row])
							| luisa::compute::isinf(diagA[column][row]);
					}
				}
				const Float regularization_scale = max_scalar(
					(diagA[0][0] + diagA[1][1] + diagA[2][2]) * (1.0f / 3.0f),
					def<float>(0.0f));
				const Float augmentation = regularization_factor * regularization_scale;
				const Float diag_x = diagA[0][0] + augmentation;
				const Float diag_y = diagA[1][1] + augmentation;
				const Float diag_z = diagA[2][2] + augmentation;
				const Float scale = max_scalar(max_scalar(abs_scalar(diag_x), abs_scalar(diag_y)), abs_scalar(diag_z));
				const Float min_diag = max_scalar(def<float>(1e-6f), 1e-6f * scale);
				const Float inv_x = ite((scale >= pcg_epsilon) & (diag_x > min_diag), 1.0f / diag_x, 0.0f);
				const Float inv_y = ite((scale >= pcg_epsilon) & (diag_y > min_diag), 1.0f / diag_y, 0.0f);
				const Float inv_z = ite((scale >= pcg_epsilon) & (diag_z > min_diag), 1.0f / diag_z, 0.0f);
				preconditioner_invalid = preconditioner_invalid
					| luisa::compute::isnan(inv_x) | luisa::compute::isinf(inv_x)
					| luisa::compute::isnan(inv_y) | luisa::compute::isinf(inv_y)
					| luisa::compute::isnan(inv_z) | luisa::compute::isinf(inv_z);
				$if(preconditioner_invalid)
				{
					sa_convergence->atomic(3u).exchange(1.0f);
				};
				Float3x3 inv_M = make_float3x3(
					make_float3(inv_x, 0.0f, 0.0f),
					make_float3(0.0f, inv_y, 0.0f),
					make_float3(0.0f, 0.0f, inv_z));
				sa_cgMinv->write(vid, inv_M);
			},
			default_option);

		compiler.compile<1>(
			fn_pcg_apply_preconditioner,
			[sa_cgR = sim_data->sa_cgR.view(),
				sa_cgZ = sim_data->sa_cgZ.view(),
				sa_cgMinv = sim_data->sa_cgMinv.view(),
				sa_block_result = sim_data->sa_block_result.view()]()
			{
				const UInt	   vid = dispatch_id().x;
				const Float3   r = sa_cgR->read(vid);
				const Float3x3 inv_M = sa_cgMinv->read(vid);
				Float3		   z = inv_M * r;
				sa_cgZ->write(vid, z);

				Float  dot_rz = dot_vec(r, z);
				Float  dot_rr = dot_vec(r, r);
				Float2 dot_rr_rz = makeFloat2(dot_rr, dot_rz);
				dot_rr_rz = ParallelIntrinsic::block_intrinsic_reduce(
					vid, dot_rr_rz, ParallelIntrinsic::warp_reduce_op_sum<float2>);
				$if(vid % 256 == 0)
				{
					const Uint blockIdx = vid / 256;
					sa_block_result->write(2 * blockIdx + 0, dot_rr_rz[0]);
					sa_block_result->write(2 * blockIdx + 1, dot_rr_rz[1]);
				};
			},
			default_option);

		compiler.compile<1>(
			fn_pcg_apply_preconditioner_second_pass,
			[sa_block_result = sim_data->sa_block_result.view(),
				sa_convergence = sim_data->sa_convergence.view(),
				fn_get_dispatch_block]()
			{
				const UInt vid = dispatch_id().x;

				Float dot_rr = 0.0f;
				Float dot_rz = 0.0f;
				$for(blockIdx, vid, fn_get_dispatch_block(), 256u)
				{
					dot_rr += sa_block_result->read(2 * blockIdx + 0);
					dot_rz += sa_block_result->read(2 * blockIdx + 1);
				};
				Float2 dot_rr_rz = make_float2(dot_rr, dot_rz);

				dot_rr_rz = ParallelIntrinsic::block_intrinsic_reduce(
					vid, dot_rr_rz, ParallelIntrinsic::warp_reduce_op_sum<float2>);

				$if(vid == 0)
				{
					sa_convergence->write(0, dot_rr_rz[0]);
					sa_convergence->write(1, dot_rr_rz[1]);
				};
			});
	}

	static inline float fast_dot(const std::vector<float3>& left_ptr, const std::vector<float3>& right_ptr)
	{
		return CpuParallel::parallel_for_and_reduce_sum<float>(
			0, left_ptr.size(), [&](const uint vid)
			{ return luisa::dot(left_ptr[vid], right_ptr[vid]); });
	};
	static inline float fast_norm(const std::vector<float3>& ptr)
	{
		float tmp = CpuParallel::parallel_for_and_reduce_sum<float>(
			0, ptr.size(), [&](const uint vid)
			{ return luisa::dot(ptr[vid], ptr[vid]); });
		return sqrt(tmp);
	};
	static inline float fast_infinity_norm(const std::vector<float3>& ptr) // Min value in array
	{
		return CpuParallel::parallel_for_and_reduce(
			0,
			ptr.size(),
			[&](const uint vid)
			{ return luisa::length(ptr[vid]); },
			[](const float left, const float right)
			{ return max_scalar(left, right); },
			-1e9f);
	};

	PcgSolveReport ConjugateGradientSolver::host_solve_core(
		std::function<void(const std::vector<float3>&, std::vector<float3>&)> func_spmv,
		std::optional<float>												  regularization_override)
	{
		auto& sa_cgX = host_sim_data->sa_cgX;
		auto& sa_cgB = host_sim_data->sa_cgB;
		auto& sa_cgA_diag = host_sim_data->sa_cgA_diag;
		auto& sa_cgMinv = host_sim_data->sa_cgMinv;
		auto& sa_cgP = host_sim_data->sa_cgP;
		auto& sa_cgQ = host_sim_data->sa_cgQ;
		auto& sa_cgR = host_sim_data->sa_cgR;
		auto& sa_cgZ = host_sim_data->sa_cgZ;

		const uint	num_verts = static_cast<uint>(sa_cgX.size());
		const float configured_factor = regularization_override.value_or(0.0f);
		if (!std::isfinite(configured_factor) || configured_factor < 0.0f)
		{
			LUISA_ERROR("Invalid PCG regularization factor {}. The internal override must be finite and non-negative.", configured_factor);
			PcgSolveReport failed;
			failed.failure_reason = "invalid regularization factor";
			return failed;
		}

		auto initialize_zero_guess = [&]()
		{
			CpuParallel::parallel_for(0, num_verts,
				[&](const uint vid)
				{
					sa_cgX[vid] = Zero3;
					sa_cgR[vid] = sa_cgB[vid];
					sa_cgP[vid] = Zero3;
					sa_cgQ[vid] = Zero3;
					sa_cgZ[vid] = Zero3;
				});
		};

		if (num_verts == 0u)
		{
			PcgSolveReport empty;
			empty.converged = true;
			empty.regularization_factor = configured_factor > 0.0f ? configured_factor : 0.0f;
			return empty;
		}
		if (!are_finite_vectors(sa_cgB))
		{
			LUISA_ERROR("CPU PCG right-hand side contains NaN or Inf.");
			PcgSolveReport failed;
			failed.regularization_factor = configured_factor > 0.0f ? configured_factor : 0.0f;
			failed.failure_reason = "non-finite right-hand side";
			return failed;
		}

		const float rhs_squared_norm = fast_dot(sa_cgB, sa_cgB);
		if (!std::isfinite(rhs_squared_norm) || rhs_squared_norm < 0.0f)
		{
			LUISA_ERROR("CPU PCG right-hand-side norm is invalid: {}.", rhs_squared_norm);
			PcgSolveReport failed;
			failed.regularization_factor = configured_factor > 0.0f ? configured_factor : 0.0f;
			failed.failure_reason = "non-finite right-hand-side norm";
			return failed;
		}

		initialize_zero_guess();
		if (rhs_squared_norm == 0.0f)
		{
			PcgSolveReport zero_rhs;
			zero_rhs.regularization_factor = configured_factor > 0.0f ? configured_factor : 0.0f;
			zero_rhs.converged = true;
			return zero_rhs;
		}
		if (get_scene_params().pcg_iter_count == 0u)
		{
			PcgSolveReport skipped;
			skipped.regularization_factor = configured_factor > 0.0f ? configured_factor : 0.0f;
			skipped.relative_residual = 1.0f;
			skipped.failure_reason = "pcg_iter_count is zero; linear solve skipped";
			if (get_scene_params().prp_debug)
				warn_pcg_nonconvergence("CPU", skipped);
			return skipped;
		}

		// Escalation policy (2026-08-31, author decision): the PCG solve is a
		// SINGLE candidate. Finite non-convergence returns the last iterate to
		// the Newton loop (D2 semantics); the Newton delta gate alone decides
		// whether to re-solve with pcg_lm_escalation_factor. The historical
		// in-solve retry ladder is retired - a residual plateau must not
		// trigger regularization by itself.
		const size_t   candidate_count = 1u;
		const float	   initial_norm = std::sqrt(rhs_squared_norm);
		std::string	   last_failure = "no regularization candidate was attempted";
		PcgSolveReport last_attempt_report;

		for (size_t attempt = 0; attempt < candidate_count; attempt++)
		{
			const float regularization_factor = configured_factor;
			initialize_zero_guess();

			std::atomic_bool preconditioner_is_finite{ true };
			CpuParallel::parallel_for(0, num_verts,
				[&](const uint vid)
				{
					const float3x3 diagA = sa_cgA_diag[vid];
					if (!is_finite_matrix(diagA))
					{
						preconditioner_is_finite.store(false, std::memory_order_relaxed);
						sa_cgMinv[vid] = luisa::make_float3x3(0.0f);
						return;
					}
					const float	   augmentation = regularization_factor * pcg_regularization_scale(diagA);
					const float3x3 regularized_diag = diagA + make_eye3x3(augmentation);
					const float3x3 inverse = make_pcg_jacobi_inverse(regularized_diag);
					if (!is_finite_matrix(inverse))
						preconditioner_is_finite.store(false, std::memory_order_relaxed);
					sa_cgMinv[vid] = inverse;
				});

			std::string failure_reason;
			bool		converged = false;
			uint		completed_steps = 0u;
			float		relative_residual = 1.0f;
			float		previous_rz = 0.0f;

			if (!preconditioner_is_finite.load(std::memory_order_relaxed))
			{
				PcgSolveReport failed;
				failed.regularization_factor = regularization_factor;
				failed.relative_residual = relative_residual;
				failed.regularization_retries = static_cast<uint>(attempt);
				failed.adaptive_retry_used = false;
				failed.failure_reason = "non-finite Jacobi preconditioner";
				LUISA_ERROR("CPU PCG cannot construct a finite Jacobi preconditioner at lambda={:.1e}; the Hessian diagonal contains NaN or Inf.", regularization_factor);
				return failed;
			}
			else
			{
				for (uint iter = 0; iter < get_scene_params().pcg_iter_count; iter++)
				{
					get_scene_params().current_pcg_it = iter;
					CpuParallel::parallel_for(0, num_verts,
						[&](const uint vid)
						{ sa_cgZ[vid] = sa_cgMinv[vid] * sa_cgR[vid]; });

					const float rr = fast_dot(sa_cgR, sa_cgR);
					const float rz = fast_dot(sa_cgR, sa_cgZ);
					if (!std::isfinite(rr) || rr < 0.0f || !std::isfinite(rz))
					{
						PcgSolveReport failed;
						failed.regularization_factor = regularization_factor;
						failed.relative_residual = relative_residual;
						failed.regularization_retries = static_cast<uint>(attempt);
						failed.iterations = completed_steps;
						failed.adaptive_retry_used = false;
						failed.failure_reason = "non-finite or invalid PCG residual scalar";
						LUISA_ERROR("CPU PCG residual became invalid at lambda={:.1e}, step {} (rTr={}, rTz={}).", regularization_factor, completed_steps, rr, rz);
						return failed;
					}
					relative_residual = std::sqrt(rr) / initial_norm;
					if (!std::isfinite(relative_residual))
					{
						PcgSolveReport failed;
						failed.regularization_factor = regularization_factor;
						failed.relative_residual = relative_residual;
						failed.regularization_retries = static_cast<uint>(attempt);
						failed.iterations = completed_steps;
						failed.adaptive_retry_used = false;
						failed.failure_reason = "non-finite relative residual";
						LUISA_ERROR("CPU PCG relative residual became NaN or Inf at lambda={:.1e}, step {}.", regularization_factor, completed_steps);
						return failed;
					}
					if (relative_residual <= kPcgRelativeTolerance)
					{
						converged = true;
						break;
					}
					if (!(rz > 0.0f))
					{
						failure_reason = "non-positive rTz";
						break;
					}

					const float beta = iter == 0u ? 0.0f : rz / previous_rz;
					if (!std::isfinite(beta))
					{
						PcgSolveReport failed;
						failed.regularization_factor = regularization_factor;
						failed.relative_residual = relative_residual;
						failed.regularization_retries = static_cast<uint>(attempt);
						failed.iterations = completed_steps;
						failed.adaptive_retry_used = false;
						failed.failure_reason = "non-finite beta";
						LUISA_ERROR("CPU PCG beta became NaN or Inf at lambda={:.1e}, step {}.", regularization_factor, completed_steps);
						return failed;
					}
					CpuParallel::parallel_for(0, num_verts,
						[&](const uint vid)
						{ sa_cgP[vid] = sa_cgZ[vid] + beta * sa_cgP[vid]; });

					func_spmv(sa_cgP, sa_cgQ);
					if (regularization_factor > 0.0f)
					{
						CpuParallel::parallel_for(0, num_verts,
							[&](const uint vid)
							{
								const float scale = pcg_regularization_scale(sa_cgA_diag[vid]);
								sa_cgQ[vid] += regularization_factor * scale * sa_cgP[vid];
							});
					}

					const float pAregp = fast_dot(sa_cgP, sa_cgQ);
					if (!std::isfinite(pAregp))
					{
						PcgSolveReport failed;
						failed.regularization_factor = regularization_factor;
						failed.relative_residual = relative_residual;
						failed.regularization_retries = static_cast<uint>(attempt);
						failed.iterations = completed_steps;
						failed.adaptive_retry_used = false;
						failed.failure_reason = "non-finite pT(A+lambda S)p";
						LUISA_ERROR("CPU PCG pT(A+lambda S)p became NaN or Inf at lambda={:.1e}, step {}.", regularization_factor, completed_steps);
						return failed;
					}
					if (!(pAregp > 0.0f))
					{
						failure_reason = "non-positive pT(A+lambda S)p";
						break;
					}
					const float alpha = rz / pAregp;
					if (!std::isfinite(alpha))
					{
						PcgSolveReport failed;
						failed.regularization_factor = regularization_factor;
						failed.relative_residual = relative_residual;
						failed.regularization_retries = static_cast<uint>(attempt);
						failed.iterations = completed_steps;
						failed.adaptive_retry_used = false;
						failed.failure_reason = "non-finite alpha";
						LUISA_ERROR("CPU PCG alpha became NaN or Inf at lambda={:.1e}, step {}.", regularization_factor, completed_steps);
						return failed;
					}

					CpuParallel::parallel_for(0, num_verts,
						[&](const uint vid)
						{
							sa_cgX[vid] += alpha * sa_cgP[vid];
							sa_cgR[vid] -= alpha * sa_cgQ[vid];
						});
					completed_steps++;

					const float updated_rr = fast_dot(sa_cgR, sa_cgR);
					if (!std::isfinite(updated_rr) || updated_rr < 0.0f)
					{
						PcgSolveReport failed;
						failed.regularization_factor = regularization_factor;
						failed.relative_residual = relative_residual;
						failed.regularization_retries = static_cast<uint>(attempt);
						failed.iterations = completed_steps;
						failed.adaptive_retry_used = false;
						failed.failure_reason = "non-finite or invalid updated residual";
						LUISA_ERROR("CPU PCG updated residual became invalid at lambda={:.1e}, step {} (rTr={}).", regularization_factor, completed_steps, updated_rr);
						return failed;
					}
					relative_residual = std::sqrt(updated_rr) / initial_norm;
					if (!std::isfinite(relative_residual))
					{
						PcgSolveReport failed;
						failed.regularization_factor = regularization_factor;
						failed.relative_residual = relative_residual;
						failed.regularization_retries = static_cast<uint>(attempt);
						failed.iterations = completed_steps;
						failed.adaptive_retry_used = false;
						failed.failure_reason = "non-finite updated relative residual";
						LUISA_ERROR("CPU PCG updated relative residual became NaN or Inf at lambda={:.1e}, step {}.", regularization_factor, completed_steps);
						return failed;
					}
					if (relative_residual <= kPcgRelativeTolerance)
					{
						converged = true;
						break;
					}
					previous_rz = rz;
				}
			}

			if (!converged && failure_reason.empty())
				failure_reason = fmt::format("plateau: relative residual {:.9g} after {} iterations", relative_residual, completed_steps);
			if (!are_finite_vectors(sa_cgX))
			{
				PcgSolveReport failed;
				failed.regularization_factor = regularization_factor;
				failed.relative_residual = relative_residual;
				failed.regularization_retries = static_cast<uint>(attempt);
				failed.iterations = completed_steps;
				failed.adaptive_retry_used = false;
				failed.failure_reason = "non-finite PCG solution";
				LUISA_ERROR("CPU PCG solution contains NaN or Inf at lambda={:.1e}, step {}.", regularization_factor, completed_steps);
				return failed;
			}

			if (converged)
			{
				PcgSolveReport report;
				report.regularization_factor = regularization_factor;
				report.relative_residual = relative_residual;
				report.infinity_norm = fast_infinity_norm(sa_cgX);
				report.regularization_retries = static_cast<uint>(attempt);
				report.iterations = completed_steps;
				report.converged = true;
				report.adaptive_retry_used = false;
				return report;
			}

			last_failure = failure_reason;
			last_attempt_report.regularization_factor = regularization_factor;
			last_attempt_report.relative_residual = relative_residual;
			last_attempt_report.infinity_norm = fast_infinity_norm(sa_cgX);
			last_attempt_report.regularization_retries = static_cast<uint>(attempt);
			last_attempt_report.iterations = completed_steps;
			last_attempt_report.converged = false;
			last_attempt_report.adaptive_retry_used = false;
			last_attempt_report.failure_reason = failure_reason;
		}

		last_attempt_report.adaptive_retry_used = false;
		last_attempt_report.failure_reason = last_failure;
		warn_pcg_nonconvergence("CPU", last_attempt_report);
		return last_attempt_report;
	}

	void ConjugateGradientSolver::host_solve(luisa::compute::Stream&		  stream,
		std::function<void(const std::vector<float3>&, std::vector<float3>&)> func_spmv,
		std::function<double()>												  func_compute_energy,
		std::optional<float>												  regularization_override)
	{
		(void)stream;
		(void)func_compute_energy;
		last_report_ = host_solve_core(std::move(func_spmv), regularization_override);
		publish_pcg_report("CPU", last_report_);
	}

	void ConjugateGradientSolver::device_solve(
		luisa::compute::Stream&																		stream,
		std::function<void(const luisa::compute::Buffer<float3>&, luisa::compute::Buffer<float3>&)> func_spmv,
		std::function<double()>																		func_compute_energy,
		std::optional<float>																		regularization_override)
	{
		(void)func_compute_energy;

		auto&		host_cgX = host_sim_data->sa_cgX;
		const uint	num_verts = static_cast<uint>(host_cgX.size());
		const float configured_factor = regularization_override.value_or(0.0f);
		if (!std::isfinite(configured_factor) || configured_factor < 0.0f)
		{
			LUISA_ERROR("Invalid PCG regularization factor {}. The internal override must be finite and non-negative.", configured_factor);
			return;
		}
		if (num_verts == 0u)
		{
			PcgSolveReport empty;
			empty.converged = true;
			empty.regularization_factor = configured_factor > 0.0f ? configured_factor : 0.0f;
			publish_pcg_report("GPU", empty);
			return;
		}

		const uint block_count = get_dispatch_block(num_verts, 256u);
		const uint reduction_threads = min_scalar(block_count, 256u);
		// Escalation policy: single candidate; see the CPU core comment. The
		// Newton delta gate alone decides escalation.
		const size_t		 candidate_count = 1u;
		std::array<float, 4> reductions{}; // [0] = rTr, [1] = rTz, [2] = pT(A+lambda S)p, [3] = invalid preconditioner
		std::string			 last_failure = "no regularization candidate was attempted";
		PcgSolveReport		 last_attempt_report;

		for (size_t attempt = 0; attempt < candidate_count; attempt++)
		{
			const float regularization_factor = configured_factor;
			reductions.fill(0.0f);

			stream << fn_reset_float(sim_data->sa_convergence.view(3, 1)).dispatch(1)
				   << fn_pcg_init().dispatch(num_verts)
				   << fn_pcg_make_preconditioner(regularization_factor).dispatch(num_verts)
				   << fn_pcg_apply_preconditioner().dispatch(num_verts)
				   << fn_pcg_apply_preconditioner_second_pass().dispatch(reduction_threads)
				   << sim_data->sa_convergence.view(0, 4).copy_to(reductions.data())
				   << luisa::compute::synchronize();

			std::string failure_reason;
			bool		converged = false;
			uint		completed_steps = 0u;
			float		rr = reductions[0];
			float		rz = reductions[1];
			float		relative_residual = 1.0f;
			float		previous_rz = 0.0f;

			if (reductions[3] != 0.0f)
			{
				LUISA_ERROR("GPU PCG cannot construct a finite Jacobi preconditioner at lambda={:.1e}; the Hessian diagonal contains NaN or Inf.", regularization_factor);
				return;
			}
			else if (!std::isfinite(rr) || rr < 0.0f || !std::isfinite(rz))
			{
				LUISA_ERROR("GPU PCG initial residual is invalid at lambda={:.1e} (rTr={}, rTz={}); the right-hand side or Hessian diagonal contains NaN or Inf.", regularization_factor, rr, rz);
				return;
			}
			else if (rr == 0.0f)
			{
				relative_residual = 0.0f;
				converged = true;
			}
			else if (get_scene_params().pcg_iter_count == 0u)
			{
				failure_reason = "pcg_iter_count is zero; linear solve skipped";
			}
			else
			{
				const float initial_norm = std::sqrt(rr);
				for (uint iter = 0; iter < get_scene_params().pcg_iter_count; iter++)
				{
					get_scene_params().current_pcg_it = iter;
					relative_residual = std::sqrt(rr) / initial_norm;
					if (!std::isfinite(relative_residual))
					{
						LUISA_ERROR("GPU PCG relative residual became NaN or Inf at lambda={:.1e}, step {}.", regularization_factor, completed_steps);
						return;
					}
					if (relative_residual <= kPcgRelativeTolerance)
					{
						converged = true;
						break;
					}
					if (!(rz > 0.0f))
					{
						failure_reason = "non-positive rTz";
						break;
					}

					const float beta = iter == 0u ? 0.0f : rz / previous_rz;
					if (!std::isfinite(beta))
					{
						LUISA_ERROR("GPU PCG beta became NaN or Inf at lambda={:.1e}, step {}.", regularization_factor, completed_steps);
						return;
					}

					stream << fn_pcg_update_p(beta).dispatch(num_verts);
					func_spmv(sim_data->sa_cgP, sim_data->sa_cgQ);
					stream << fn_pcg_add_regularization(regularization_factor).dispatch(num_verts)
						   << fn_dot_pq().dispatch(num_verts)
						   << fn_dot_pq_second_pass().dispatch(reduction_threads)
						   << sim_data->sa_convergence.view(2, 1).copy_to(&reductions[2])
						   << luisa::compute::synchronize();

					const float pAregp = reductions[2];
					if (!std::isfinite(pAregp))
					{
						LUISA_ERROR("GPU PCG pT(A+lambda S)p became NaN or Inf at lambda={:.1e}, step {}.", regularization_factor, completed_steps);
						return;
					}
					if (!(pAregp > 0.0f))
					{
						failure_reason = "non-positive pT(A+lambda S)p";
						break;
					}
					const float alpha = rz / pAregp;
					if (!std::isfinite(alpha))
					{
						LUISA_ERROR("GPU PCG alpha became NaN or Inf at lambda={:.1e}, step {}.", regularization_factor, completed_steps);
						return;
					}

					const float rz_for_step = rz;
					stream << fn_pcg_step(alpha).dispatch(num_verts)
						   << fn_pcg_apply_preconditioner().dispatch(num_verts)
						   << fn_pcg_apply_preconditioner_second_pass().dispatch(reduction_threads)
						   << sim_data->sa_convergence.view(0, 2).copy_to(reductions.data())
						   << luisa::compute::synchronize();
					completed_steps++;
					previous_rz = rz_for_step;
					rr = reductions[0];
					rz = reductions[1];

					if (!std::isfinite(rr) || rr < 0.0f || !std::isfinite(rz))
					{
						LUISA_ERROR("GPU PCG updated residual became invalid at lambda={:.1e}, step {} (rTr={}, rTz={}).", regularization_factor, completed_steps, rr, rz);
						return;
					}
					relative_residual = std::sqrt(rr) / initial_norm;
					if (!std::isfinite(relative_residual))
					{
						LUISA_ERROR("GPU PCG updated relative residual became NaN or Inf at lambda={:.1e}, step {}.", regularization_factor, completed_steps);
						return;
					}
					if (relative_residual <= kPcgRelativeTolerance)
					{
						converged = true;
						break;
					}
				}
			}

			if (!converged && failure_reason.empty())
				failure_reason = fmt::format("plateau: relative residual {:.9g} after {} iterations", relative_residual, completed_steps);
			stream << sim_data->sa_cgX.copy_to(host_cgX.data())
				   << luisa::compute::synchronize();
			if (!are_finite_vectors(host_cgX))
			{
				LUISA_ERROR("GPU PCG solution contains NaN or Inf at lambda={:.1e}, step {}.", regularization_factor, completed_steps);
				return;
			}

			if (converged)
			{
				PcgSolveReport report;
				report.regularization_factor = regularization_factor;
				report.relative_residual = relative_residual;
				report.infinity_norm = fast_infinity_norm(host_cgX);
				report.regularization_retries = static_cast<uint>(attempt);
				report.iterations = completed_steps;
				report.converged = true;
				report.adaptive_retry_used = false;
				last_report_ = report;
				publish_pcg_report("GPU", report);
				return;
			}

			last_failure = failure_reason;
			last_attempt_report.regularization_factor = regularization_factor;
			last_attempt_report.relative_residual = relative_residual;
			last_attempt_report.infinity_norm = fast_infinity_norm(host_cgX);
			last_attempt_report.regularization_retries = static_cast<uint>(attempt);
			last_attempt_report.iterations = completed_steps;
			last_attempt_report.converged = false;
			last_attempt_report.adaptive_retry_used = false;
			last_attempt_report.failure_reason = failure_reason;
			if (get_scene_params().pcg_iter_count == 0u)
			{
				warn_pcg_nonconvergence("GPU", last_attempt_report);
				last_report_ = last_attempt_report;
				publish_pcg_report("GPU", last_attempt_report);
				return;
			}
		}

		last_attempt_report.adaptive_retry_used = false;
		last_attempt_report.failure_reason = last_failure;
		warn_pcg_nonconvergence("GPU", last_attempt_report);
		last_report_ = last_attempt_report;
		publish_pcg_report("GPU", last_attempt_report);
	}

	void ConjugateGradientSolver::eigen_solve(const Eigen::SparseMatrix<float>& eigen_cgA,
		Eigen::VectorXf&														eigen_cgX,
		const Eigen::VectorXf&													eigen_cgB,
		std::function<double()>													func_compute_energy)
	{
		std::vector<float3>& host_cgX = host_sim_data->sa_cgX;

		const uint num_verts = host_cgX.size();

		auto eigen_iter_solve = [&]()
		{
			// Solve cgA * dx = cg_b_vec for dx using Conjugate Gradient
			Eigen::ConjugateGradient<Eigen::SparseMatrix<float>, Eigen::Lower> solver; // Eigen::IncompleteCholesky<float>

			// solver.setMaxIterations(128);
			solver.setTolerance(1e-2f);
			solver.compute(eigen_cgA);

			// 计算Jacobi预条件子的对角线逆
			// Eigen::VectorXf eigen_cgR = eigen_cgB - eigen_cgA * eigen_cgX;
			// Eigen::VectorXf eigen_cgM_inv(eigen_cgR.rows());
			// for (int i = 0; i < eigen_cgR.rows(); ++i) {
			//     float diag = eigen_cgA.coeff(i, i);
			//     eigen_cgM_inv[i] = (std::abs(diag) > 1e-12f) ? (1.0f / diag) : 0.0f;
			// }
			// Eigen::VectorXf eigen_cgZ = eigen_cgR.cwiseProduct(eigen_cgM_inv);
			// Eigen::VectorXf eigen_cgQ = eigen_cgA * eigen_cgZ;
			// LUISA_INFO("initB = {}, initR = {}, initM = {}, initZ = {}, initQ = {}",
			//     eigen_cgB.norm(), eigen_cgR.norm(), eigen_cgM_inv.norm(), eigen_cgZ.norm(), eigen_cgQ.norm());

			solver._solve_impl(eigen_cgB, eigen_cgX);
			if (solver.info() != Eigen::Success)
			{
				LUISA_ERROR("Eigen: Solve failed in {} iterations", solver.iterations());
			}
			else
			{
				CpuParallel::parallel_for(0,
					num_verts,
					[&](const uint vid)
					{ host_cgX[vid] = eigen3_to_float3(eigen_cgX.segment<3>(3 * vid)); });

				if (get_scene_params().print_pcg_info)
					LUISA_INFO("  In newton iter {:2}, Eigen-PCG iters = {}, error = {:.3e}, max_element(p) = {:.3e}",
						get_scene_params().current_nonlinear_iter,
						solver.iterations(),
						solver.error(),
						fast_infinity_norm(host_cgX)); // from normR_0 -> normR
			}
		};
		auto eigen_decompose_solve = [&]()
		{
			// Solve cgA * dx = cg_b_vec for dx using SimplicialLDLT decomposition
			Eigen::SimplicialLDLT<Eigen::SparseMatrix<float>> solver;
			solver.compute(eigen_cgA);
			if (solver.info() != Eigen::Success)
			{
				LUISA_ERROR("Eigen: SimplicialLDLT decomposition failed!");
				return;
			}
			solver._solve_impl(eigen_cgB, eigen_cgX);
			if (solver.info() != Eigen::Success)
			{
				LUISA_ERROR("Eigen: SimplicialLDLT solve failed!");
				return;
			}
			else
			{
				float error = (eigen_cgB - eigen_cgA * eigen_cgX).norm();
				CpuParallel::parallel_for(0,
					num_verts,
					[&](const uint vid)
					{ host_cgX[vid] = eigen3_to_float3(eigen_cgX.segment<3>(3 * vid)); });
				if (get_scene_params().print_pcg_info)
					LUISA_INFO("  In newton iter {:2}, Eigen-Decompose : error = {:.3e}, max_element(p) = {:.3e}",
						get_scene_params().current_nonlinear_iter,
						error,
						fast_infinity_norm(host_cgX)); // from normR_0 -> normR
			}
		};

		eigen_iter_solve();
	}

} // namespace lcs
