"""Plot convergence curves for a PRP multi-case sweep.

Four independent figures (one per case: Synthetic, Drop1, Drop3, HairCard),
each showing #self-intersection (EF) pairs vs frame for all five methods
(PRP C0/C2/C3/C4 + baselines ICM/GIA). Reads per-frame debug_info_*.json
directly so partial/crashed arms plot up to their last written frame.

SIGGRAPH-ish aesthetic (per multimodal review): colorblind-safe Okabe-Ito
palette, markers only at events (star=untangle, x=crash) with legend entries,
lines terminate at a crash, thicker lines for print, EF defined in axis label,
informative titles. Vector PDF + 200-dpi PNG.

Usage:
    python plot_convergence_curves.py [SWEEP_ROOT] [OUTPUT_DIR]

    SWEEP_ROOT  directory with <Case>/<Config>/debug_info_*.json + _logs/
                (default: output/Sweep_0729)
    OUTPUT_DIR  where to write figures (default: <SWEEP_ROOT>/figures)
"""
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = "output/Sweep_0729"
CASES = ["Synthetic", "Drop1", "Drop3", "HairCard"]
TITLES = {
	"Synthetic": "Synthetic Fold — cloth untangling convergence",
	"Drop1": "Drop1 (3 cloths + obstacle) — convergence",
	"Drop3": "Drop3 (147k-vertex garment) — convergence",
	"HairCard": "HairCard — convergence",
}
# config -> (display label, color, linestyle). C3 (proposed) gets the accent.
# Okabe-Ito-inspired colorblind-safe palette.
PLOT_ORDER = ["C0_baseline", "C2_dir_opt", "C3_no_ccd_depth005", "C4_all_contours", "ICM", "GIA"]
STYLES = {
	"C0_baseline": ("PRP baseline (C0)", "#999999", "-"),
	"C2_dir_opt": ("PRP + dir-opt (C2)", "#0072B2", (0, (5, 2))),   # blue, dashed
	"C3_no_ccd_depth005": ("PRP + depth-clamp (C3)", "#D55E00", "-"),  # vermilion accent
	"C4_all_contours": ("PRP + all-contours (C4)", "#E69F00", (0, (3, 1, 1, 1))),  # amber, dash-dot
	"ICM": ("ICM", "#009E73", "-."),  # green
	"GIA": ("GIA", "#CC79A7", (0, (1, 2))),  # pink, dotted
}


def load_series(case, cfg, root):
	d = os.path.join(root, case, cfg)
	if not os.path.isdir(d):
		return None, None
	frames, efs = [], []
	for fn in os.listdir(d):
		if not (fn.startswith("debug_info_") and fn.endswith(".json")):
			continue
		try:
			with open(os.path.join(d, fn), encoding="utf-8") as f:
				j = json.load(f)
		except (OSError, json.JSONDecodeError):
			continue
		ef = int(j.get("ef_pair_count", -1))
		if ef < 0:
			continue
		frames.append(int(j.get("frame", 0)))
		efs.append(ef)
	if not frames:
		return None, None
	order = np.argsort(frames)
	return np.array(frames)[order], np.array(efs)[order]


def detect_crash(case, cfg, root):
	log = os.path.join(root, "_logs", f"{case}_{cfg}.log")
	if not os.path.exists(log):
		return None
	try:
		with open(log, encoding="utf-8", errors="replace") as f:
			t = f.read()
	except OSError:
		return None
	for key, label in [
		("BufferBudgetExceeded", "buffer-budget"),
		("CUDA_ERROR_ILLEGAL_ADDRESS", "gpu-illegal-access"),
		("CUDA_ERROR_OUT_OF_MEMORY", "gpu-oom"),
	]:
		if key in t:
			return label
	# CPU-side segfault (C4 process_all_contours on large scenes, exit 127)
	if "CPU PRP contour evaluation: global batched source elements" in t:
		last_eval = t.rfind("CPU PRP contour evaluation: global batched source elements")
		after = t[last_eval:]
		if "Frame " not in after.split("\n", 1)[-1] and "#EF pairs" not in after:
			return "cpu-segfault"
	return None


def plot_case(case, outdir, root):
	fig, ax = plt.subplots(figsize=(6.8, 4.8))
	plotted = False
	for cfg in PLOT_ORDER:
		frames, efs = load_series(case, cfg, root)
		if frames is None:
			continue
		plotted = True
		label, color, ls = STYLES[cfg]
		crash = detect_crash(case, cfg, root)
		# For log-y, floor zero at 0.5 so untangle-to-zero is visible.
		y = np.where(efs == 0, 0.5, efs).astype(float)
		# A crash terminates the line at the last written frame (we don't draw past it).
		ax.plot(frames, y, color=color, ls=ls, lw=2.0, label=label, alpha=0.95, solid_capstyle="round")
		# star at first untangle frame (ef==0), excluding frame 0's trivial clean state
		zero_idx = [i for i, e in enumerate(efs) if e == 0 and frames[i] > 0]
		if zero_idx:
			zf = frames[zero_idx[0]]
			ax.plot([zf], [0.5], marker="*", color="black", ms=13, mfc="white", mec="black", mew=1.4, zorder=6)
		# x at the crash frame
		if crash:
			ax.plot([frames[-1]], [y[-1]], marker="x", color="black", ms=10, mew=2.4, zorder=6)

	# event-marker legend entries (black). Only list the ones that occur in this case.
	handles, labels = ax.get_legend_handles_labels()
	# detect whether any line reached 0 EF (excluding frame 0) or crashed.
	have_star = False
	for cfg in PLOT_ORDER:
		fr, ef = load_series(case, cfg, root)
		if fr is None or ef is None:
			continue
		if np.any((ef == 0) & (fr > 0)):
			have_star = True
	have_x = any(detect_crash(case, cfg, root) for cfg in PLOT_ORDER)
	extra = []
	if have_star:
		extra.append(plt.Line2D([0], [0], marker="*", ls="none", ms=13, mfc="white", mec="black", mew=1.4, label="untangled (0 EF)"))
	if have_x:
		extra.append(plt.Line2D([0], [0], marker="x", ls="none", ms=10, mew=2.4, color="black", label="crashed"))
	ax.legend(handles=handles + extra, labels=labels + [h.get_label() for h in extra],
			  loc="best", frameon=False, fontsize=8, ncol=1)

	ax.set_yscale("log")
	ax.set_xlabel("Frame")
	ax.set_ylabel("Self-intersections (EF pairs, log scale)")
	ax.set_title(TITLES.get(case, case), fontsize=12.5, pad=8)
	ax.grid(True, which="major", ls=":", lw=0.4, alpha=0.35, color="#CCCCCC")
	ax.minorticks_off()
	for s in ("top", "right"):
		ax.spines[s].set_visible(False)
	if not plotted:
		ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes, color="#999999")
	fig.tight_layout()
	os.makedirs(outdir, exist_ok=True)
	fig.savefig(os.path.join(outdir, f"convergence_{case}.pdf"))
	fig.savefig(os.path.join(outdir, f"convergence_{case}.png"), dpi=200)
	plt.close(fig)
	print(f"  {case} -> {os.path.relpath(os.path.join(outdir, f'convergence_{case}.png'))}")


def main():
	root = sys.argv[1] if len(sys.argv) > 1 else ROOT
	outdir = sys.argv[2] if len(sys.argv) > 2 else os.path.join(root, "figures")
	for case in CASES:
		plot_case(case, outdir, root)


if __name__ == "__main__":
	sys.exit(main())