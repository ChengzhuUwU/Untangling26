#include "CollisionDetector/intersection_resolver2.h"
#include "Initializer/init_sim_data.h"
#include <luisa/core/fiber.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <numeric>
#include <vector>

namespace
{

	using lcs::PRPMinContourHitInfo;

	[[noreturn]] void fail(const char* message)
	{
		std::cerr << "test_prp_selection: " << message << '\n';
		std::exit(EXIT_FAILURE);
	}

	void require(const bool condition, const char* message)
	{
		if (!condition)
			fail(message);
	}

	void require_direction_near(
		const luisa::float3& actual,
		const luisa::float3& expected,
		const char*			 message)
	{
		constexpr float tolerance = 1e-6f;
		const float		error = std::max({ std::abs(actual.x - expected.x),
			std::abs(actual.y - expected.y),
			std::abs(actual.z - expected.z) });
		require(error <= tolerance, message);
	}

	PRPMinContourHitInfo make_response(
		const uint	combo_idx,
		const float penetration_objective,
		const float coverage,
		const bool	eligible)
	{
		PRPMinContourHitInfo info;
		info.combo_idx = combo_idx;
		info.hit_list.emplace_back();
		info.hit_count = 1u;
		info.covered_anchor_count = 1u;
		info.boundary_anchor_count = 1u;
		info.contour_coverage = coverage;
		info.penetration_objective = penetration_objective;
		info.penetration_depth = 1.0f;
		info.displacement_cost = 1.0f;
		info.response_eligible = eligible;
		return info;
	}

} // namespace

int main()
{
	auto scene_params = std::make_shared<lcs::SceneParams>();
	lcs::set_scene_params_ptr(scene_params);
	require(scene_params->PRP_direction_optimization_iterations == 0u,
		"direction optimization must remain disabled by default");

	{
		require(lcs::prp_contour_roles_share_topology({ 3u, 3u }, { 5u, 5u }),
			"self-collision roles must share an object and topology component");
		require(!lcs::prp_contour_roles_share_topology({ 3u, 3u }, { 5u, 8u }),
			"disconnected components must have distinct role identities");
		require(!lcs::prp_contour_roles_share_topology({ 3u, 7u }, { 5u, 5u }),
			"different objects must have distinct role identities");
		require(!lcs::prp_contour_roles_share_topology(
					{ 3u, 3u },
					{ std::numeric_limits<uint>::max(), std::numeric_limits<uint>::max() }),
			"unavailable component identity must not match");
	}

	{
		require(lcs::prp_intrinsic_contour_side_restriction_applies(
					true, { 3u, 3u }, { 5u, 5u }),
			"intrinsic contour-side candidates must apply to an enabled self-collision contour");
		require(!lcs::prp_intrinsic_contour_side_restriction_applies(
					false, { 3u, 3u }, { 5u, 5u }),
			"intrinsic contour-side candidates must remain opt-in");
		require(!lcs::prp_intrinsic_contour_side_restriction_applies(
					true, { 3u, 7u }, { 5u, 5u }),
			"intrinsic contour-side candidates must not mix different objects");
		require(!lcs::prp_intrinsic_contour_side_restriction_applies(
					true, { 3u, 3u }, { 5u, 8u }),
			"intrinsic contour-side candidates must not compare disconnected topology components");

		require(lcs::prp_intrinsic_contour_side_restriction_applies(
					true, { 3u, 7u }, { 5u, 8u }, true),
			"deformed boundary distance must permit inter-body contour coordinates");
		require(lcs::prp_intrinsic_contour_side_restriction_applies(
					true, { 3u, 3u }, { 5u, 8u }, true),
			"deformed boundary distance must permit disconnected contour coordinates");
		require(!lcs::prp_intrinsic_contour_side_restriction_applies(
					true,
					{ std::numeric_limits<uint>::max(), 7u },
					{ 5u, 8u },
					true),
			"deformed boundary distance must reject an invalid object identity");

		require(lcs::prp_intrinsic_contour_side_owns_vertex(0.25f, 1.0f),
			"a vertex closer to its own contour preimage must remain in that role");
		require(!lcs::prp_intrinsic_contour_side_owns_vertex(1.0f, 0.25f),
			"a vertex beyond the intrinsic bisector must leave that role");
		require(lcs::prp_intrinsic_contour_side_owns_vertex(0.5f, 0.5f),
			"the intrinsic bisector must belong to both closed candidate covers");
		require(lcs::prp_intrinsic_contour_side_owns_vertex(
					0.5f, std::numeric_limits<float>::max()),
			"a finite own-side path must win when the opposite contour is unreachable");
		require(!lcs::prp_intrinsic_contour_side_owns_vertex(
					std::numeric_limits<float>::max(), 0.5f),
			"an unreachable own-side path must not enter that role");

		require(!lcs::prp_dual_metric_contour_side_owns_vertex(10.0f, 1.0f, 0.1f, 0.2f),
			"spatial proximity must not override a failed rest-geodesic constraint");
		require(!lcs::prp_dual_metric_contour_side_owns_vertex(1.0f, 10.0f, 0.9f, 0.1f),
			"rest proximity must not override a failed current-space constraint");
		require(lcs::prp_dual_metric_contour_side_owns_vertex(2.0f, 3.0f, 0.3f, 0.2f),
			"both metrics may accept a vertex inside their overlap windows");
		require(!lcs::prp_dual_metric_contour_side_owns_vertex(2.0f, 3.0f, 0.3f, 0.2f, 0.0f),
			"the configured tau must apply independently to both metrics");
		require(lcs::prp_dual_metric_contour_side_owns_vertex(5.0f, 3.0f, 0.5f, 0.3f),
			"the closed tau=0.25 boundary must be accepted by both metrics");
		require(lcs::prp_dual_metric_contour_side_owns_vertex(
			0.5f, std::numeric_limits<float>::max(), 0.1f, 0.2f),
			"inter-body candidates still require a passing current-space constraint");
		require(!lcs::prp_dual_metric_contour_side_owns_vertex(
			0.5f, std::numeric_limits<float>::max(), 0.9f, 0.1f),
			"an unreachable opposite rest anchor must not bypass the current-space constraint");
		require(!lcs::prp_dual_metric_contour_side_owns_vertex(
			std::numeric_limits<float>::max(), 0.5f, 0.1f, 0.2f),
			"current-space proximity must not rescue an unreachable own rest anchor");
		// A passing coordinate may compensate for the other only in blend mode.
		require(lcs::prp_intrinsic_contour_side_owns_vertex_blended(8.0f, 2.0f, 4.0f, 6.0f, 0.25f, 0.5f),
			"equal-weight blend must admit phi_rest=.6, phi_current=-.2");
		require(!lcs::prp_dual_metric_contour_side_owns_vertex(8.0f, 2.0f, 4.0f, 6.0f),
			"AND must still reject the same compensating pair");
		require(!lcs::prp_intrinsic_contour_side_owns_vertex_blended(8.0f, 2.0f, 4.0f, 6.0f, 0.25f, 0.75f),
			"increasing the rest weight must reject this pair at w=.75");
		require(lcs::prp_intrinsic_contour_side_owns_vertex_blended(800.0f, 200.0f, 0.4f, 0.6f, 0.25f, 0.5f),
			"blend must combine normalized coordinates, not raw distances");
		require(lcs::prp_intrinsic_contour_side_owns_vertex_blended(8.0f, 2.0f, 4.0f, 6.0f, 0.25f, 0.0f),
			"w=0 must use current ownership for complete distance pairs");
		require(!lcs::prp_intrinsic_contour_side_owns_vertex_blended(8.0f, 2.0f, 4.0f, 6.0f, 0.25f, 1.0f),
			"w=1 must use rest ownership for complete distance pairs");
		require(!lcs::prp_intrinsic_contour_side_owns_vertex_blended(1.0f, 9.0f, 9.0f, 1.0f, 0.25f, 0.0f),
			"the current endpoint must retain its rejection");
		require(lcs::prp_intrinsic_contour_side_owns_vertex_blended(1.0f, 9.0f, 9.0f, 1.0f, 0.25f, 1.0f),
			"the rest endpoint must retain its acceptance");
		require(lcs::prp_intrinsic_contour_side_owns_vertex_blended(5.0f, 3.0f, 5.0f, 3.0f, 0.25f, 0.5f),
			"blend must accept the closed ownership-window boundary");
		const float unreachable = std::numeric_limits<float>::max();
		require(!lcs::prp_intrinsic_contour_side_owns_vertex_blended(unreachable, 1.0f, unreachable, 1.0f, 1.0f, 0.5f),
			"no reachable own anchor must reject even at tau=1");
		require(!lcs::prp_intrinsic_contour_side_owns_vertex_blended(0.5f, unreachable, 0.9f, 0.1f, 0.25f, 1.0f),
			"incomplete rest pairs must abstain under the historical fallback policy");
		require(lcs::prp_intrinsic_contour_side_owns_vertex_blended(1.0f, 9.0f, unreachable, 0.1f, 0.25f, 0.0f),
			"incomplete current pairs must fall back to defined rest coordinates");
		for (const float weight : { 0.0f, 0.25f, 0.5f, 0.75f, 1.0f })
			require(std::abs(lcs::prp_intrinsic_contour_side_blended_phi(8.0f, 2.0f, 4.0f, 6.0f, weight)
				- (0.8f * weight - 0.2f)) < 1e-6f, "all requested weights must interpolate the normalized coordinates");

		for (const float tau : { 0.0f, 0.25f, 1.0f })
		{
			require(lcs::prp_dual_metric_contour_side_owns_vertex(1.0f, 10.0f, 0.9f, 0.1f, tau, true, false),
				"rest-only must ignore a failing current constraint");
			require(lcs::prp_dual_metric_contour_side_owns_vertex(10.0f, 1.0f, 0.1f, 0.2f, tau, false, true),
				"current-only must ignore a failing rest constraint");
			require(lcs::prp_dual_metric_contour_side_owns_vertex(10.0f, 1.0f, 0.1f, 0.2f, tau, true, false) == (tau == 1.0f),
				"rest-only must enforce its selected tau");
			require(lcs::prp_dual_metric_contour_side_owns_vertex(1.0f, 10.0f, 0.9f, 0.1f, tau, false, true) == (tau == 1.0f),
				"current-only must enforce its selected tau");
		}
	}

	{
		lcs::SimulationData<std::vector> sim_data;
		lcs::MeshData<std::vector>		 mesh_data;
		sim_data.sa_x = {
			luisa::make_float3(-3.0f, -2.0f, -1.0f),
			luisa::make_float3(-3.0f, 2.0f, -1.0f),
			luisa::make_float3(-3.0f, 0.0f, 1.0f),
			luisa::make_float3(3.0f, -2.0f, -1.0f),
			luisa::make_float3(3.0f, 2.0f, -1.0f),
			luisa::make_float3(3.0f, 0.0f, 1.0f)
		};
		mesh_data.num_verts = static_cast<uint>(sim_data.sa_x.size());
		mesh_data.sa_rest_vert_area.assign(sim_data.sa_x.size(), 1.0f);

		const std::array<std::vector<uint>, 2> boundary_verts = {
			std::vector<uint>{ 0u, 1u, 2u },
			std::vector<uint>{ 3u, 4u, 5u }
		};
		const std::vector<luisa::float3> contour_points = {
			luisa::make_float3(0.0f, 0.4f, 0.4f),
			luisa::make_float3(0.0f, 0.6f, 0.4f),
			luisa::make_float3(0.0f, 0.4f, 0.6f),
			luisa::make_float3(0.0f, 0.6f, 0.6f)
		};
		const std::vector<float> contour_weights(contour_points.size(), 1.0f);
		const auto				 directions = lcs::get_raycasting_direction(
			contour_points, contour_weights, boundary_verts, &sim_data, &mesh_data);

		require(directions.size() == 10u,
			"default direction generation must preserve all 10 slots");

		const auto contour_center = lcs::get_weighted_mean(
			contour_points, luisa::make_float3(0.0f), contour_weights);
		std::array<luisa::float3, 2> mesh_centers = {
			luisa::make_float3(0.0f), luisa::make_float3(0.0f)
		};
		luisa::float3x3 scatter = luisa::make_float3x3(0.0f);
		for (uint side = 0u; side < 2u; ++side)
		{
			std::vector<float> weights;
			weights.reserve(boundary_verts[side].size());
			for (const uint vid : boundary_verts[side])
			{
				const float weight = std::max(mesh_data.sa_rest_vert_area[vid], 1e-8f);
				weights.push_back(weight);
				mesh_centers[side] += sim_data.sa_x[vid] * weight;
				const auto centered = sim_data.sa_x[vid] - contour_center;
				scatter = scatter + weight * lcs::outer_product(centered, centered);
			}
			const float sum_weights = std::accumulate(weights.begin(), weights.end(), 0.0f);
			mesh_centers[side] = mesh_centers[side] / sum_weights;
		}

		std::vector<luisa::float3> expected;
		auto					   append_direction = [&](const luisa::float3& direction)
		{
			expected.push_back(luisa::normalize(direction));
		};
		auto append_signed = [&](const luisa::float3& direction)
		{
			append_direction(direction);
			append_direction(-direction);
		};
		for (const auto& eigenvector : lcs::compute_all_eigenvectors(scatter))
			append_signed(eigenvector);
		append_direction(contour_center - mesh_centers[0]);
		append_direction(contour_center - mesh_centers[1]);

		luisa::float3 global_centroid = luisa::make_float3(0.0f);
		float		  total_weight = 0.0f;
		for (uint vid = 0u; vid < mesh_data.num_verts; ++vid)
		{
			const float weight = std::max(mesh_data.sa_rest_vert_area[vid], 1e-8f);
			global_centroid += sim_data.sa_x[vid] * weight;
			total_weight += weight;
		}
		global_centroid /= total_weight;
		append_signed(contour_center - global_centroid);

		for (size_t slot = 0u; slot < expected.size(); ++slot)
		{
			require_direction_near(directions[slot], expected[slot],
				"default direction slot value or ordering changed");
			const float length_squared = luisa::dot(directions[slot], directions[slot]);
			require(std::abs(length_squared - 1.0f) <= 1e-6f,
				"default direction must remain normalized");
		}
	}

	{
		std::vector<PRPMinContourHitInfo> responses;
		responses.push_back(make_response(0u, 10.0f, 1.0f, true));
		responses.push_back(make_response(1u, 0.01f, 1.0f, false));
		const auto selected = lcs::prp_select_response(responses);
		require(selected.has_response() && selected.selected_idx == 0u,
			"an ineligible response must not compete with an eligible response");
	}

	{
		std::vector<PRPMinContourHitInfo> responses;
		responses.push_back(make_response(0u, 4.0f, 1.0f, true));
		responses.push_back(make_response(1u, 1.0f, 0.25f, true));
		const auto selected = lcs::prp_select_response(responses);
		require(selected.has_response() && selected.selected_idx == 0u,
			"selection must compare penetration objective divided by squared coverage");
	}

	{
		lcs::SimulationData<std::vector> sim_data;
		lcs::MeshData<std::vector>		 mesh_data;
		sim_data.sa_x = {
			luisa::make_float3(0.0f, 0.0f, 1.0f),
			luisa::make_float3(0.0f, 0.0f, 0.0f),
			luisa::make_float3(1.0f, 0.0f, 0.0f),
			luisa::make_float3(0.0f, 1.0f, 0.0f)
		};
		mesh_data.sa_vert_mass.assign(sim_data.sa_x.size(), 1.0f);
		const auto initial_direction = luisa::normalize(luisa::make_float3(0.6f, 0.0f, 0.8f));
		const auto hit = lcs::RayCasting::HitInfo::make_vf_hit(
			0u,
			0u,
			0u,
			0u,
			luisa::make_uint3(1u, 2u, 3u),
			luisa::make_float2(0.2f, 0.2f),
			initial_direction,
			1.0f);

		auto make_runtime = [&]()
		{
			std::vector<lcs::PrpStandardContourRuntime> runtimes(1u);
			auto										accepted = make_response(0u, 1.0f, 1.0f, true);
			accepted.hit_list = { hit };
			accepted.separation_direction = initial_direction;
			runtimes[0].contour_idx = 0u;
			runtimes[0].contour_dirs = { initial_direction };
			runtimes[0].active_combos = { 0u };
			runtimes[0].combo_results = { accepted };
			runtimes[0].combo_eval_present = { 1u };
			runtimes[0].combo_objective_pruned = { 0u };
			return runtimes;
		};

		{
			auto runtimes = make_runtime();
			bool evaluator_called = false;
			lcs::prp_optimize_runtime_directions(
				runtimes,
				{ 0u },
				&mesh_data,
				&sim_data,
				0u,
				[&](auto&, const auto&)
				{ evaluator_called = true; });
			require(!evaluator_called,
				"zero direction-optimization iterations must not evaluate a trial");
			require(runtimes[0].combo_results[0].direction_optimization_iteration_count == 0u,
				"zero direction-optimization iterations must preserve a zero attempt count");
		}

		{
			auto runtimes = make_runtime();
			bool evaluator_called = false;
			lcs::prp_optimize_runtime_directions(
				runtimes,
				{ 0u },
				&mesh_data,
				&sim_data,
				1u,
				[&](auto& trial_runtimes, const auto& trial_contours)
				{
					evaluator_called = true;
					require(trial_contours == std::vector<uint>{ 0u },
						"direction optimization must evaluate the requested contour");
					auto trial = make_response(0u, 0.001f, 1.0f, false);
					trial.hit_list = { hit };
					trial.separation_direction = trial_runtimes[0].contour_dirs[0];
					trial_runtimes[0].combo_results[0] = std::move(trial);
				});
			require(evaluator_called,
				"positive direction-optimization iterations must evaluate a valid trial");
			require_direction_near(
				runtimes[0].contour_dirs[0],
				initial_direction,
				"an ineligible trial must not replace an eligible direction");
			require(runtimes[0].combo_results[0].direction_optimization_iteration_count == 1u,
				"a rejected valid trial must increment the attempt count");
			require(runtimes[0].combo_results[0].direction_optimization_accepted_count == 0u,
				"a rejected ineligible trial must not increment the accepted count");
			lcs::SceneParams::PRPDebugInfo debug;
			scene_params->prp_debug = true;
			lcs::export_runtime_combo_debug(runtimes[0], debug);
			scene_params->prp_debug = false;
			require(debug.uint_stats.at("prp.c0.combo0.opt_iter_count") == 1u,
				"debug output must report a rejected optimization attempt");
			require(debug.uint_stats.at("prp.c0.combo0.opt_accepted_count") == 0u,
				"debug output must preserve a zero accepted count");
			require(!debug.bool_stats.at("prp.c0.combo0.opt_eval.present"),
				"a rejected trial must not be exported as an accepted optimized result");
		}

		{
			auto runtimes = make_runtime();
			lcs::prp_optimize_runtime_directions(
				runtimes,
				{ 0u },
				&mesh_data,
				&sim_data,
				1u,
				[&](auto& trial_runtimes, const auto&)
				{
					auto trial = make_response(0u, 0.001f, 1.0f, true);
					trial.hit_list = { hit };
					trial.separation_direction = trial_runtimes[0].contour_dirs[0];
					trial_runtimes[0].combo_results[0] = std::move(trial);
				});
			require(luisa::dot(runtimes[0].contour_dirs[0], initial_direction) < 1.0f - 1e-6f,
				"an improved eligible trial must update the accepted direction");
			require(runtimes[0].combo_results[0].direction_optimization_iteration_count == 1u,
				"an accepted trial must increment the attempt count");
			require(runtimes[0].combo_results[0].direction_optimization_accepted_count == 1u,
				"an improved eligible trial must increment the accepted count");
			lcs::SceneParams::PRPDebugInfo debug;
			scene_params->prp_debug = true;
			lcs::export_runtime_combo_debug(runtimes[0], debug);
			scene_params->prp_debug = false;
			require(debug.bool_stats.at("prp.c0.combo0.opt_eval.present"),
				"an accepted trial must be exported as the optimized result");
			require(debug.uint_stats.at("prp.c0.combo0.opt_eval.opt_accepted_count") == 1u,
				"optimized-result debug output must carry the accepted count");
		}
	}

	{
		std::vector<PRPMinContourHitInfo> responses(2u);
		const auto						  selected = lcs::prp_select_response(responses);
		require(!selected.has_response(), "invalid responses must produce an explicit no-response result");
	}

	{
		const lcs::PRPContourMergeAdjacency directed = { { 1u }, {} };
		const auto							suppressed = lcs::prp_select_pre_finalize_contours_to_suppress(
			directed, { 1.0f, 10.0f }, { 0u, 1u });
		require(suppressed == std::vector<uint>{ 1u },
			"a one-way pre-finalize contour adjacency must keep its owner and suppress the touched contour");
	}

	{
		const lcs::PRPContourMergeAdjacency mutual = { { 1u }, { 0u } };
		const auto							suppressed = lcs::prp_select_pre_finalize_contours_to_suppress(
			mutual, { 1.0f, 2.0f }, { 0u, 1u });
		require(suppressed == std::vector<uint>{ 0u },
			"mutually adjacent contours must keep the larger pre-finalize penetration-area owner");

		const auto tie_suppressed = lcs::prp_select_pre_finalize_contours_to_suppress(
			mutual, { 2.0f, 2.0f }, { 0u, 1u });
		require(tie_suppressed == std::vector<uint>{ 1u },
			"equal-area mutual adjacency must use the smaller contour index as owner");
	}

	{
		auto make_runtime = [](const uint diameter = 8u)
		{
			std::vector<lcs::PrpStandardContourRuntime> runtimes(1u);
			auto&										runtime = runtimes[0];
			runtime.contour_dirs = { luisa::make_float3(0.0f, 1.0f, 0.0f) };
			runtime.active_combos = { 0u };
			runtime.combo_converged = { 0u };
			runtime.combo_objective_pruned = { 0u };
			runtime.combo_results.resize(1u);
			runtime.combo_eval_present = { 0u };
			runtime.init_k_src = runtime.init_k_dst = 2u;
			runtime.mesh_diameter_src = runtime.mesh_diameter_dst = diameter;
			return runtimes;
		};
		{
			auto runtimes = make_runtime();
			uint evaluations = 0u;
			lcs::evaluate_current_runtime_directions_template(runtimes, { 0u }, false, [](const auto&) {}, [&](const auto&, uint, bool)
				{
					++evaluations;
					auto info = make_response(0u, 1.0f, 1.0f, false);
					info.selected_frontier_converged = runtimes[0].curr_k_src > 2u;
					info.max_frontier_hop_src = info.max_frontier_hop_dst = 2u;
					return std::vector<PRPMinContourHitInfo>{ info }; });
			require(evaluations == 2u && runtimes[0].combo_results[0].response_eligible,
				"default expansion is frontier-detachment-only; full coverage alone must not stop a touching support");
			require(runtimes[0].combo_results[0].selected_frontier_converged,
				"a full-coverage response must still finish its component search");
		}
		{
			auto							 runtimes = make_runtime();
			std::vector<std::array<uint, 2>> ranges;
			// A cached inactive direction must not prune a direction-refinement trial.
			runtimes[0].contour_dirs.push_back(luisa::make_float3(1.0f, 0.0f, 0.0f));
			runtimes[0].combo_results.push_back(make_response(1u, 0.001f, 1.0f, true));
			lcs::evaluate_current_runtime_directions_template(runtimes, { 0u }, false, [&](const auto&)
				{ ranges.push_back({ runtimes[0].curr_k_src, runtimes[0].curr_k_dst }); }, [&](const auto&, uint, bool)
				{
					auto info = make_response(0u, 1.0f, 0.5f, false);
					info.boundary_anchor_count = 2u;
					info.max_frontier_hop_src = ranges.size() == 1u ? 2u : 3u;
					info.max_frontier_hop_dst = 1u;
					info.selected_frontier_converged = ranges.size() == 2u;
					return std::vector<PRPMinContourHitInfo>{ info }; });
			require(ranges == std::vector<std::array<uint, 2>>{ { 2u, 2u }, { 4u, 2u } },
				"only the role whose support touches the frontier must expand");
			require(runtimes[0].combo_results[0].response_eligible,
				"detached partial coverage remains eligible for the normalized comparator");
		}
		{
			auto runtimes = make_runtime();
			uint evaluations = 0u;
			lcs::evaluate_current_runtime_directions_template(runtimes, { 0u }, false, [](const auto&) {}, [&](const auto&, uint, bool)
				{
					++evaluations;
					return std::vector<PRPMinContourHitInfo>(1u); });
			require(evaluations == 3u && runtimes[0].no_response_terminal,
				"empty rays must terminate at the finite mesh bound without a response");
			require(!lcs::prp_has_full_contour_coverage(runtimes[0].combo_results[0]),
				"zero anchors must not count as full coverage");
		}
		{
			auto runtimes = make_runtime();
			runtimes[0].mesh_diameter_src = 0u;
			lcs::evaluate_current_runtime_directions_template(runtimes, { 0u }, false, [&](const auto&)
				{ require(runtimes[0].curr_k_src == 2u, "an exhausted role must not shrink while the other expands"); }, [](const auto&, uint, bool)
				{ return std::vector<PRPMinContourHitInfo>(1u); });
			require(runtimes[0].no_response_terminal,
				"an empty response with unequal role diameters must terminate");
		}
		{
			auto runtimes = make_runtime();
			runtimes[0].mesh_diameter_src = 2u;
			uint evaluations = 0u;
			lcs::evaluate_current_runtime_directions_template(runtimes, { 0u }, false, [](const auto&) {}, [&](const auto&, uint, bool)
				{
					++evaluations;
					auto info = make_response(0u, 1.0f, 0.5f, false);
					info.max_frontier_hop_src = 2u;
					info.max_frontier_hop_dst = 1u;
					info.selected_frontier_converged = false;
					return std::vector<PRPMinContourHitInfo>{ info }; });
			require(evaluations == 1u && runtimes[0].combo_results[0].response_eligible,
				"one exhausted role and one detached role must complete without another query");
		}
	}

	{
		require(lcs::Initializer::bending_edge_has_distinct_vertices(luisa::make_uint4(0u, 1u, 2u, 3u)),
			"a four-vertex dihedral must remain eligible");
		require(!lcs::Initializer::bending_edge_has_distinct_vertices(luisa::make_uint4(0u, 1u, 2u, 2u)),
			"a dihedral with a repeated opposite vertex must be rejected");
	}

	{
		auto make_pair = [](const uint pair_idx, const bool loop_vertex)
		{
			lcs::CollisionPair::EfPair pair{};
			pair.set_edge(loop_vertex
					? luisa::make_uint2(pair_idx)
					: luisa::make_uint2(2u * pair_idx, 2u * pair_idx + 1u));
			pair.set_face(luisa::make_uint3(10u + pair_idx, 20u + pair_idx, 30u + pair_idx));
			return pair;
		};
		auto classify = [&](const std::vector<lcs::CollisionPair::EfPair>& pairs,
							const std::vector<std::vector<uint>>&		   adjacency)
		{
			std::vector<uint>		  meshes(pairs.size(), 0u);
			std::vector<luisa::ubyte> boundary_flags(pairs.size(), 0u);
			std::vector<luisa::ubyte> loop_boundary_flags;
			std::vector<uint>		  contour(pairs.size());
			std::iota(contour.begin(), contour.end(), 0u);
			return lcs::classify_contour_type(
				pairs, meshes, adjacency, boundary_flags, loop_boundary_flags, contour, 0u);
		};

		const std::vector<lcs::CollisionPair::EfPair> nonloop_pairs = {
			make_pair(0u, false), make_pair(1u, false), make_pair(2u, false)
		};
		require(classify(nonloop_pairs, { { 1u, 2u }, { 0u, 2u }, { 0u, 1u } }) == lcs::ContourType::Closed,
			"a non-loop contour whose EF graph has no endpoint must classify as Closed");
		require(classify(nonloop_pairs, { { 1u }, { 0u, 2u }, { 1u } }) == lcs::ContourType::Undifined,
			"a non-loop EF graph with degree-one endpoints must not classify as Closed");
		require(classify({ make_pair(0u, true) }, { { 0u } }) == lcs::ContourType::Eight,
			"a loop-vertex storage node must not make an Eight contour open");
		require(classify({ make_pair(0u, true), make_pair(1u, true) }, { { 1u }, { 0u } }) == lcs::ContourType::LL,
			"loop-vertex storage nodes must not make an LL contour open");
	}

	{
		const auto									normal = luisa::make_float3(0.0f, 1.0f, 0.0f);
		luisa::fiber::scheduler						scheduler;
		const std::vector<lcs::RayCasting::HitInfo> hits = {
			lcs::RayCasting::HitInfo::make_vf_hit(0u, 0u, 0u, 0u,
				luisa::make_uint3(1u, 2u, 3u), luisa::make_float2(0.2f, 0.3f), normal, 0.1f)
		};
		lcs::PrpStandardContourRuntime runtime;
		for (uint side = 0u; side < 2u; ++side)
			runtime.vert_to_boundary_hop_dist[side].assign(4u, 0u);
		const auto evaluate = [&](const luisa::float3 target_normal, const bool need_flip)
		{
			return lcs::prp_eval_penetration_v2(hits, { 2.0f }, {},
				std::vector<float>(4u, 3.0f), std::vector<float>(4u, 1.0f),
				{ target_normal }, {}, std::vector<luisa::float3>(4u, normal), need_flip, runtime);
		};
		const auto aligned = evaluate(normal, true);
		const auto opposing = evaluate(-normal, false);
		require(std::abs(aligned.volume - 0.3f) < 1e-6f,
			"aligned normals must not multiply the penetration volume");
		require(std::abs(aligned.objective - 0.03f) < 1e-6f,
			"aligned normals must not multiply the response objective");
		require(std::abs(aligned.displacement_cost - opposing.displacement_cost) < 1e-6f,
			"geometric flip metadata must not penalize the displacement cost");
		require(aligned.invalid_hit_count == 0u && aligned.invalid_area == 0.0f,
			"aligned normals must remain valid response support");
	}

	std::cout << "test_prp_selection passed\n";
	return EXIT_SUCCESS;
}
