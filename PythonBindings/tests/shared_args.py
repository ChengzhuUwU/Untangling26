"""Argument groups shared by the retained paper reproduction scripts."""

import argparse
import sys


UNTANGLING_METHODS = ("PRP", "ICM", "GIA")


def create_parser(description="LuisaCompute Python example", **kwargs):
    return argparse.ArgumentParser(description=description, **kwargs)


def add_backend_args(parser, default="auto"):
    if default == "auto":
        default = "metal" if sys.platform == "darwin" else "cuda"
    parser.add_argument("--backend", choices=["cuda", "dx", "vk", "metal"], default=default, help=f"Compute backend to use (default: {default})")


def add_headless_args(parser, advance_frames_default=1):
    parser.add_argument("--headless", action="store_true", help="Run without the interactive viewer")
    parser.add_argument("--advance_frames", type=int, default=advance_frames_default, help=f"Number of frames to simulate in headless mode (default: {advance_frames_default})")


def configure_untangling_method(config, method):
    method = str(method).upper()
    if method not in UNTANGLING_METHODS:
        raise ValueError(f"Unsupported untangling method {method!r}; expected one of {UNTANGLING_METHODS}")

    config.use_untangling = True
    config.use_untangling_PRP = method == "PRP"
    config.use_untangling_ICM = method == "ICM"
    config.use_untangling_GIA = method == "GIA"

    # GPU untangling contains PRP-only kernels. GPU PCG remains independent.
    if method != "PRP":
        config.use_gpu_untangling = False

    # Preserve the baseline variants used by the paper experiments.
    config.accumulate_ICM_correction = True
    config.GIA_use_EF_response = True
    return method
