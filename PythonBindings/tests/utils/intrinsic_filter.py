"""Shared normalized-coordinate blend options and experiment configuration records."""

import argparse
import math


def unit_interval(value):
    value = float(value)
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise argparse.ArgumentTypeError("expected a finite value in [0, 1]")
    return value


def add_blend_args(parser):
    parser.add_argument("--intrinsic_filter_mode", choices=("and", "blend"), default="and")
    parser.add_argument("--intrinsic_blend_weight", type=unit_interval, default=0.5,
                        help="Rest-coordinate weight; independent of LM regularization")
    parser.add_argument("--rest_geodesic_distance_for_intrinsic_candidates", type=int,
                        choices=(0, 1), default=None)


def apply_blend_args(config, args):
    config.PRP_use_blended_intrinsic_coordinates = args.intrinsic_filter_mode == "blend"
    config.PRP_intrinsic_blend_weight = args.intrinsic_blend_weight
    if args.rest_geodesic_distance_for_intrinsic_candidates is not None:
        config.PRP_use_rest_geodesic_distance_for_intrinsic_candidates = bool(
            args.rest_geodesic_distance_for_intrinsic_candidates)
    if getattr(args, "consistent_solve", None) is not None:
        config.consistent_solve = bool(args.consistent_solve)
        if config.consistent_solve:
            config.use_gpu = False


def configuration_record(config):
    fields = (
        "use_gpu", "use_gpu_untangling", "consistent_solve", "prp_debug",
        "PRP_use_intrinsic_contour_side_candidates", "PRP_intrinsic_tau",
        "PRP_use_blended_intrinsic_coordinates", "PRP_intrinsic_blend_weight",
        "PRP_use_rest_geodesic_distance_for_intrinsic_candidates",
        "PRP_use_deformed_boundary_distance_for_intrinsic_candidates",
        "PRP_cpu_contour_batch_size", "untangling_process_contours_count",
        "untangling_response_depth", "PRP_direction_optimization_iterations",
        "PRP_solid_interior_adjacency", "pcg_iter_count", "nonlinear_iter_count",
        "pcg_lm_adaptive", "pcg_lm_escalation_factor", "use_ccd_linesearch",
        "use_quasi_static_mode", "use_floor",
    )
    return {key: getattr(config, key) for key in fields}
