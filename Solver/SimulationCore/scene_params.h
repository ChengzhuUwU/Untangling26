#pragma once

#include "Core/float_n.h"
#include <memory>
#include <vector>

namespace lcs
{

	enum SolverType
	{
		SolverTypeNewton,
		SolverTypeXPBD,
		SolverTypeVBD,
	};

	enum class QuasiStaticInertiaMode : uint
	{
		Off = 0u,
		HessianOnly = 1u,
		Full = 2u,
	};

	struct SceneParams
	{

		bool prp_debug = false;
		struct PRPDebugInfo
		{
			std::unordered_map<std::string, bool>  bool_stats;
			std::unordered_map<std::string, uint>  uint_stats;
			std::unordered_map<std::string, float> float_stats;
		} prp_debug_info;

		// Global Config
		bool use_gpu = true;
		bool fix_scene = false;
		bool use_energy_linesearch = false;
		bool use_ccd_linesearch = true;
		bool use_global_ccd = true;
		bool use_quasi_static_mode = false;
		bool use_static_mode = false;

		QuasiStaticInertiaMode qs_inertia_mode = QuasiStaticInertiaMode::HessianOnly;

		bool print_system_energy = false;
		bool print_pcg_info = true;
		bool print_collision_info = true;

		// Collision
		bool use_floor = true;
		bool use_self_collision = true;

		// Untangling
		bool ignore_near_zero_dist_pairs = true;
		bool use_untangling = false;
		bool use_gpu_untangling = false; // GPU batched contour-evaluation path; forced off for ICM/GIA baselines.
		bool use_untangling_ICM = false;
		bool use_untangling_GIA = false;
		bool use_untangling_PRP = false;

		bool accumulate_ICM_correction = true;
		bool GIA_use_EF_response = true;

		uint  PRP_direction_optimization_iterations = 0u;
		float untangling_response_depth = 3e-3f;
		uint  untangling_process_contours_count = 256;
		uint  PRP_cpu_contour_batch_size = 32u;

		bool  PRP_use_intrinsic_contour_side_candidates = true;
		float PRP_intrinsic_tau = 0.25f;
		bool  PRP_use_blended_intrinsic_coordinates = false; // false: independent AND constraints
		float PRP_intrinsic_blend_weight = 0.5f;			 // rest-coordinate weight, unrelated to PCG/LM
		bool  PRP_use_rest_geodesic_distance_for_intrinsic_candidates = true;
		bool  PRP_use_deformed_boundary_distance_for_intrinsic_candidates = true;

		bool  PRP_solid_interior_adjacency = true;
		bool  PRP_solid_interior_adjacency_all_meshes = false;
		float PRP_solid_interior_adjacency_normal_eps = 1e-4f;
		float PRP_solid_interior_adjacency_dist_eps = 1e-5f;

		// PCG regularization
		bool  pcg_lm_adaptive = true;
		float pcg_lm_escalation_factor = 0.1f;

		// Deterministic solve mode (phase 1, CPU physics path): forces fixed-order
		// sequential assembly, collision gradient/Hessian assembly on the host, and
		// single-threaded SpMV reduce-by-key so repeated runs are bitwise identical.
		bool consistent_solve = false;

		// Animation
		bool output_per_frame = false;
		bool output_per_iteration = false;
		uint scene_id = 0;
		uint load_state_frame = 0;

		// Iteration info
		uint num_substep = 1;
		uint nonlinear_iter_count = 1;
		uint pcg_iter_count = 100;

		uint current_frame = 0;
		uint current_nonlinear_iter = 0;
		uint current_pcg_it = 0;
		uint current_substep = 0;

		bool collect_iteration_debug = true;

		uint collision_detection_frequece = 1;

		uint contact_energy_type = 1; // 0 for quadratic, 1 for log-barrier

		float implicit_dt = 1.f / 60.f;
		float explicit_dt = 1E-4;
		float dt = implicit_dt;
		float dt_inv = 1.0f / dt;
		float dt_2_inv = dt_inv * dt_inv;

		// Stiffness
		float stiffness_bending_ui = 1.0f;
		float stiffness_collision = 1e8;
		float stiffness_untangling = stiffness_collision;
		float stiffness_dirichlet = 1e9;

		// Damping
		float damping_rate = 2.0f;

		// Thickness & Friction
		float d_hat = 1e-3f;

		lcs::float3 gravity{ 0, -9.8f, 0 };
		lcs::float3 floor{ 0, 0, 0 };

		SceneParams() {}

		void update_dt(const float input_dt)
		{
			dt = input_dt;
			dt_inv = 1.0f / dt;
			dt_2_inv = dt_inv * dt_inv;
		}
		float get_substep_dt() { return implicit_dt / float(num_substep); }
		float get_bending_stiffness_scaling() { return stiffness_bending_ui; }
		bool  should_contour_skip(const uint contour_idx)
		{
			return contour_idx >= untangling_process_contours_count;
		}

		void validate_untangling_configuration();
	};

	void						 set_scene_params_ptr(const std::shared_ptr<SceneParams>& scene_params_ptr);
	std::shared_ptr<SceneParams> get_scene_params_ptr();
	SceneParams&				 get_scene_params();
	// std::vector<SceneParams>& get_scene_params_array();

} // namespace lcs
