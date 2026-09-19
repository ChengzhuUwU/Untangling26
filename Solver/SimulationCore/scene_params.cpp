#include <cmath>
#include <memory>
#include <stdexcept>
#include <luisa/core/logging.h>
#include "scene_params.h"

namespace lcs
{

	static std::weak_ptr<SceneParams> g_scene_params_ptr;

	void set_scene_params_ptr(const std::shared_ptr<SceneParams>& scene_params_ptr)
	{
		g_scene_params_ptr = scene_params_ptr;
	}

	std::shared_ptr<SceneParams> get_scene_params_ptr()
	{
		return g_scene_params_ptr.lock();
	}

	SceneParams& get_scene_params()
	{
		auto ptr = g_scene_params_ptr.lock();
		if (!ptr)
			throw std::runtime_error(
				"SceneParams is not initialized. Create a SolverInterface/NewtonSolver instance first.");
		return *ptr;
	}

	void SceneParams::validate_untangling_configuration()
	{
		if (PRP_use_deformed_boundary_distance_for_intrinsic_candidates
			&& !PRP_use_intrinsic_contour_side_candidates)
		{
			LUISA_ERROR("Deformed boundary distance requires PRP_use_intrinsic_contour_side_candidates=true.");
		}
		if (!std::isfinite(PRP_intrinsic_tau) || PRP_intrinsic_tau < 0.0f || PRP_intrinsic_tau > 1.0f)
		{
			LUISA_ERROR("PRP_intrinsic_tau must be finite and lie in [0, 1], got {}.", PRP_intrinsic_tau);
		}
		if (use_untangling)
		{
			if (!use_untangling_ICM && !use_untangling_GIA && !use_untangling_PRP)
			{
				use_untangling_PRP = true;
			}
			const uint active_methods = (use_untangling_PRP ? 1u : 0u) + (use_untangling_ICM ? 1u : 0u) + (use_untangling_GIA ? 1u : 0u);
			if (active_methods > 1u)
			{
				LUISA_ERROR("Invalid untangling configuration: multiple methods enabled (PRP={}, ICM={}, GIA={}). Only one method may be enabled at a time.",
					use_untangling_PRP, use_untangling_ICM, use_untangling_GIA);
			}
			if (use_gpu_untangling && (use_untangling_ICM || use_untangling_GIA))
			{
				LUISA_ERROR("Invalid untangling configuration: use_gpu_untangling is only supported for PRP (ICM={}, GIA={}).",
					use_untangling_ICM, use_untangling_GIA);
			}
		}
	}

} // namespace lcs
