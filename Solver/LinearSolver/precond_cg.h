#pragma once

#include "SimulationCore/base_mesh.h"
#include "SimulationCore/simulation_data.h"
#include <optional>
#include <string>
#include <vector>
#include <Eigen/Sparse>
#include <Eigen/Eigenvalues>
#include <Utils/async_compiler.h>

namespace lcs
{

	// Provenance record of one Newton linear solve. Returned by host_solve_core
	// for CPU solves and filled by device_solve for GPU solves; published into
	// SceneParams::prp_debug_info by the callers.
	struct PcgSolveReport
	{
		float		regularization_factor = 0.0f; // lambda that produced the returned solution
		float		relative_residual = 0.0f;	  // final ||r|| / ||b||
		float		infinity_norm = 0.0f;		  // max ||x_i|| over blocks
		uint		regularization_retries = 0u;  // candidates rejected before the accepted one
		uint		iterations = 0u;			  // PCG steps taken by the accepted candidate
		bool		converged = false;			  // false whenever the solve was skipped or failed
		bool		adaptive_retry_used = false;  // true iff factor==0 and at least one candidate failed
		std::string failure_reason;				  // last failure cause; empty when converged
	};

	// struct PcgData
	// {
	// };

	class ConjugateGradientSolver
	{

	public:
		constexpr static bool use_eigen = false;
		constexpr static bool use_upper_triangle = false;

	public:
		void set_data(MeshData<std::vector>*		host_mesh_data,
			MeshData<luisa::compute::Buffer>*		mesh_data,
			SimulationData<std::vector>*			host_sim_data,
			SimulationData<luisa::compute::Buffer>* sim_data)
		{
			this->host_mesh_data = host_mesh_data;
			this->mesh_data = mesh_data;
			this->host_sim_data = host_sim_data;
			this->sim_data = sim_data;
		}
		void compile(AsyncCompiler& compiler);

	public:
		// The optional regularization override applies only to this call (Newton delta gate).
		// Return host cgX
		void host_solve(luisa::compute::Stream&									  stream,
			std::function<void(const std::vector<float3>&, std::vector<float3>&)> func_spmv,
			std::function<double()>												  func_compute_energy,
			std::optional<float> regularization_override = std::nullopt);
		// Return device cgX and host cgX
		void device_solve(luisa::compute::Stream&														stream,
			std::function<void(const luisa::compute::Buffer<float3>&, luisa::compute::Buffer<float3>&)> func_spmv,
			std::function<double()>																		func_compute_energy,
			std::optional<float> regularization_override = std::nullopt);
		// Provenance of the most recent host/device solve. Always populated,
		// independent of the collect_iteration_debug switch, so the Newton
		// delta gate can read it on every iteration.
		[[nodiscard]] const PcgSolveReport& last_solve_report() const noexcept { return last_report_; }
		void								eigen_solve(const Eigen::SparseMatrix<float>& eigen_cgA,
			Eigen::VectorXf&							   eigen_cgX,
			const Eigen::VectorXf&						   eigen_cgB,
			std::function<double()>						   func_compute_energy);

	public:
		// Stream-free CPU solve so unit tests can exercise the retry/failure
		// semantics without creating a device. Finite non-convergence returns
		// the last iterate with converged=false; non-finite state LUISA_ERRORs.
		PcgSolveReport host_solve_core(
			std::function<void(const std::vector<float3>&, std::vector<float3>&)> func_spmv,
			std::optional<float> regularization_override = std::nullopt);

	private:
		luisa::compute::Shader<1, luisa::compute::BufferView<float3>> fn_reset_float3;
		luisa::compute::Shader<1, luisa::compute::BufferView<float>>  fn_reset_float;
		luisa::compute::Shader<1, luisa::compute::BufferView<uint>>	  fn_reset_uint;

		luisa::compute::Shader<1> fn_pcg_init;

		luisa::compute::Shader<1>		 fn_dot_pq;
		luisa::compute::Shader<1>		 fn_dot_pq_second_pass;
		luisa::compute::Shader<1, float> fn_pcg_update_p;
		luisa::compute::Shader<1, float> fn_pcg_step;

		luisa::compute::Shader<1, float> fn_pcg_add_regularization;
		luisa::compute::Shader<1, float> fn_pcg_make_preconditioner;
		luisa::compute::Shader<1>		 fn_pcg_apply_preconditioner;
		luisa::compute::Shader<1>		 fn_pcg_apply_preconditioner_second_pass;

	private:
		MeshData<std::vector>*					host_mesh_data;
		MeshData<luisa::compute::Buffer>*		mesh_data;
		SimulationData<std::vector>*			host_sim_data;
		SimulationData<luisa::compute::Buffer>* sim_data;
		PcgSolveReport							last_report_;
	};

} // namespace lcs
