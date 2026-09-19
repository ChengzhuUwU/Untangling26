"""
Mesh deformation utilities for creating self-intersecting test cases.

Provides:
  - BezierCurve3D: composite cubic Bézier curve in 3D
  - curve_deform: Blender-style Curve Modifier deformation
  - laplacian_smooth: uniform Laplacian smoothing
  - generate_self_intersections: random multi-fold self-intersection generator
"""

import numpy as np
from collections import defaultdict


# Bézier curve utilities.

class BezierCurve3D:
    """Composite cubic Bézier curve through control-point segments.

    Global parameter t ∈ [0, 1] spans all segments uniformly.
    """

    def __init__(self, segments):
        """
        Parameters
        ----------
        segments : list of (4, 3) arrays
            Each element is one cubic Bézier segment [P0, P1, P2, P3].
        """
        self.segments = [np.asarray(s, dtype=np.float64) for s in segments]
        self.n_seg = len(self.segments)

    # ── Factory methods ──────────────────────────────────────────

    @classmethod
    def from_keypoints(cls, points, smoothness=0.5):
        """Build a C1-smooth curve interpolating *points* via Catmull-Rom
        tangent estimation.

        Parameters
        ----------
        points : (N, 3) array-like, N >= 2
        smoothness : float
            Handle-length factor (0 = sharp corners, 1 = long handles).
        """
        pts = np.asarray(points, dtype=np.float64)
        n = len(pts)
        assert n >= 2, "Need at least 2 keypoints"
        segments = []
        for i in range(n - 1):
            p0, p3 = pts[i], pts[i + 1]
            # Catmull-Rom tangent
            t0 = (pts[1] - pts[0]) if i == 0 else (pts[i + 1] - pts[i - 1]) * 0.5
            t1 = (pts[-1] - pts[-2]) if i == n - 2 else (pts[i + 2] - pts[i]) * 0.5
            p1 = p0 + smoothness * t0 / 3.0
            p2 = p3 - smoothness * t1 / 3.0
            segments.append(np.stack([p0, p1, p2, p3]))
        return cls(segments)

    @classmethod
    def from_control_points(cls, control_points):
        """Create from 3N+1 raw control points (N cubic segments).

        E.g. 4 points → 1 segment, 7 → 2 segments.
        """
        cp = np.asarray(control_points, dtype=np.float64)
        n = len(cp)
        assert (n - 1) % 3 == 0, f"Need 3N+1 control points, got {n}"
        n_seg = (n - 1) // 3
        return cls([cp[3 * i: 3 * i + 4] for i in range(n_seg)])

    @classmethod
    def make_line(cls, start, end):
        """Degenerate straight-line curve (for use as rest pose)."""
        s = np.asarray(start, np.float64)
        e = np.asarray(end, np.float64)
        d = (e - s) / 3.0
        return cls([np.stack([s, s + d, s + 2 * d, e])])

    # ── Evaluation ───────────────────────────────────────────────

    def evaluate(self, t):
        """Curve position at global parameter *t* ∈ [0, 1]."""
        t = float(np.clip(t, 0.0, 1.0))
        ts = t * self.n_seg
        idx = min(int(ts), self.n_seg - 1)
        u = ts - idx
        c = self.segments[idx]
        return ((1 - u) ** 3 * c[0] + 3 * (1 - u) ** 2 * u * c[1]
                + 3 * (1 - u) * u ** 2 * c[2] + u ** 3 * c[3])

    def evaluate_batch(self, ts):
        """Evaluate at an array of parameter values."""
        return np.array([self.evaluate(t) for t in ts])

    def tangent(self, t, eps=1e-5):
        """Unit tangent at *t*."""
        p0 = self.evaluate(max(0, t - eps))
        p1 = self.evaluate(min(1, t + eps))
        d = p1 - p0
        n = np.linalg.norm(d)
        return d / n if n > 1e-12 else np.array([1.0, 0.0, 0.0])

    def arc_length(self, n_samples=256):
        """Approximate total arc length."""
        ts = np.linspace(0, 1, n_samples)
        pts = self.evaluate_batch(ts)
        return np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1))


# Rotation-minimizing frame utilities (Wang et al. 2008).

def _compute_rmf(curve, ts):
    """Double-reflection RMF along *curve* at parameter samples *ts*.

    Returns
    -------
    positions : (N, 3)
    T : (N, 3)  – unit tangents
    U : (N, 3)  – rotation-minimizing "up" vectors
    """
    n = len(ts)
    positions = curve.evaluate_batch(ts)
    T = np.array([curve.tangent(t) for t in ts])
    U = np.zeros_like(T)

    # Seed initial frame
    ref = np.array([0., 1., 0.]) if abs(T[0, 1]) < 0.9 else np.array([1., 0., 0.])
    U[0] = np.cross(T[0], ref)
    U[0] /= np.linalg.norm(U[0])

    for i in range(n - 1):
        v1 = positions[i + 1] - positions[i]
        c1 = np.dot(v1, v1)
        if c1 < 1e-20:
            U[i + 1] = U[i]
            continue
        rL = U[i] - (2.0 / c1) * np.dot(v1, U[i]) * v1
        tL = T[i] - (2.0 / c1) * np.dot(v1, T[i]) * v1
        v2 = T[i + 1] - tL
        c2 = np.dot(v2, v2)
        U[i + 1] = rL - (2.0 / c2) * np.dot(v2, rL) * v2 if c2 > 1e-20 else rL
        nrm = np.linalg.norm(U[i + 1])
        U[i + 1] = U[i + 1] / nrm if nrm > 1e-12 else U[i]

    return positions, T, U


# Deformation helpers.

def _smoothstep(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def _smoothstep_arr(x):
    x = np.clip(np.asarray(x, dtype=np.float64), 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


# Curve deformation (Blender-style curve modifier).

def curve_deform(vertices, rest_curve, deformed_curve, weights=None):
    """Warp *vertices* from a straight rest-spine onto a deformed Bézier
    curve, analogous to Blender's **Curve Modifier**.

    How it works
    ------------
    1. Project each vertex onto the rest curve's axis → parameter *t*.
    2. Decompose the perpendicular offset in the rest frame (U₀, V₀).
    3. At the same *t* on the deformed curve, reconstruct the position
       using a rotation-minimizing frame (T, U, V).
    4. Blend between original and deformed positions using *weights*.

    Parameters
    ----------
    vertices : (N, 3) array
    rest_curve : BezierCurve3D
        Straight-line (or original) spine.
    deformed_curve : BezierCurve3D
        Target curve — bending / looping creates self-intersection.
    weights : (N,) array or None
        Per-vertex blend weight in [0, 1].  0 = unchanged, 1 = fully
        deformed.  ``None`` treats every vertex as 1.

    Returns
    -------
    (N, 3) ndarray — deformed vertex positions.
    """
    verts = np.array(vertices, dtype=np.float64)
    N = len(verts)
    if weights is None:
        weights = np.ones(N, dtype=np.float64)
    else:
        weights = np.asarray(weights, dtype=np.float64)

    # ── Rest-axis decomposition ──────────────────────────────────
    rs = rest_curve.evaluate(0.0)
    re = rest_curve.evaluate(1.0)
    rd = re - rs
    rlen = np.linalg.norm(rd)
    if rlen < 1e-12:
        return verts
    rhat = rd / rlen

    # Constant rest frame (line → constant T, U, V)
    ref = np.array([0., 1., 0.]) if abs(rhat[1]) < 0.9 else np.array([1., 0., 0.])
    U0 = np.cross(rhat, ref)
    U0 /= np.linalg.norm(U0)
    V0 = np.cross(rhat, U0)

    offsets = verts - rs[None, :]          # (N, 3)
    t_vals = offsets @ rhat / rlen         # (N,)
    local_u = offsets @ U0                 # (N,)
    local_v = offsets @ V0                 # (N,)

    # ── RMF along deformed curve ─────────────────────────────────
    n_samp = 512
    ts = np.linspace(0, 1, n_samp)
    def_pos, def_T, def_U = _compute_rmf(deformed_curve, ts)
    def_V = np.cross(def_T, def_U)        # (n_samp, 3)

    # ── Vectorised deformation ───────────────────────────────────
    t_clipped = np.clip(t_vals, 0.0, 1.0)                      # (N,)
    fi = t_clipped * (n_samp - 1)                               # (N,)
    i0 = np.minimum(fi.astype(int), n_samp - 2)                 # (N,)
    frac = (fi - i0)[:, None]                                   # (N, 1)

    p  = (1 - frac) * def_pos[i0] + frac * def_pos[i0 + 1]     # (N, 3)
    ud = (1 - frac) * def_U[i0]   + frac * def_U[i0 + 1]
    vd = (1 - frac) * def_V[i0]   + frac * def_V[i0 + 1]

    # Re-normalise after interpolation
    ud_n = np.linalg.norm(ud, axis=1, keepdims=True)
    vd_n = np.linalg.norm(vd, axis=1, keepdims=True)
    ud = np.where(ud_n > 1e-12, ud / ud_n, ud)
    vd = np.where(vd_n > 1e-12, vd / vd_n, vd)

    new_pos = p + local_u[:, None] * ud + local_v[:, None] * vd  # (N, 3)

    w = weights[:, None]                                          # (N, 1)
    return (1 - w) * verts + w * new_pos


# Laplacian smoothing.

def laplacian_smooth(vertices, faces, iterations=3, lambda_factor=0.3,
                     pin_boundary=True):
    """Uniform Laplacian smoothing.

    Parameters
    ----------
    vertices, faces : arrays
    iterations : int
    lambda_factor : float  (0 = no change, 1 = fully move to average)
    pin_boundary : bool
        If True, boundary vertices (edges with 1 adjacent face) are fixed.
    """
    N = len(vertices)
    verts = np.array(vertices, dtype=np.float64)
    faces_arr = np.asarray(faces, dtype=np.int32)

    # Build adjacency
    adj = defaultdict(set)
    for f in faces_arr:
        adj[int(f[0])].update([int(f[1]), int(f[2])])
        adj[int(f[1])].update([int(f[0]), int(f[2])])
        adj[int(f[2])].update([int(f[0]), int(f[1])])

    # Boundary detection
    boundary_mask = np.zeros(N, dtype=bool)
    if pin_boundary:
        edge_count = {}
        for f in faces_arr:
            for i in range(3):
                e = (min(int(f[i]), int(f[(i + 1) % 3])),
                     max(int(f[i]), int(f[(i + 1) % 3])))
                edge_count[e] = edge_count.get(e, 0) + 1
        for e, cnt in edge_count.items():
            if cnt == 1:
                boundary_mask[e[0]] = True
                boundary_mask[e[1]] = True

    # Build padded neighbour-index array for vectorised gather
    max_val = max((len(v) for v in adj.values()), default=0)
    nbr_idx = np.full((N, max_val), -1, dtype=np.int32)
    nbr_cnt = np.zeros(N, dtype=np.int32)
    for v in range(N):
        nbrs = sorted(adj.get(v, []))
        nbr_cnt[v] = len(nbrs)
        nbr_idx[v, :len(nbrs)] = nbrs

    valid = nbr_idx >= 0  # (N, max_val)

    for _ in range(iterations):
        safe_idx = np.where(valid, nbr_idx, 0)
        nbr_pos = verts[safe_idx]                               # (N, max_val, 3)
        nbr_pos[~valid] = 0.0
        avg = nbr_pos.sum(axis=1) / np.maximum(nbr_cnt[:, None], 1)  # (N, 3)
        delta = lambda_factor * (avg - verts)
        delta[boundary_mask] = 0.0
        delta[nbr_cnt == 0] = 0.0
        verts = verts + delta

    return verts


# Self-intersection generation.

def generate_self_intersections(vertices, faces,
                                n_deformations=3,
                                severity=0.5,
                                smooth_iterations=2,
                                seed=None):
    """Apply *n_deformations* random Bézier-curve folds to create mesh
    self-intersections.

    Each fold selects a corner or strip region of the mesh, curves it
    out of the mesh plane, and folds it back *through* the rest of the
    mesh to guarantee intersection.

    Parameters
    ----------
    vertices : (V, 3) array
    faces : (F, 3) array
    n_deformations : int
        Number of independent fold passes.
    severity : float
        Controls fold amplitude (0 = gentle, 1 = aggressive).
    smooth_iterations : int
        Laplacian smoothing iterations applied after each fold to
        maintain vertex spacing.  0 = skip smoothing.
    seed : int or None
        For reproducibility.

    Returns
    -------
    deformed_verts : (V, 3) ndarray
    curves_info : list[dict]
        Per-fold metadata (for visualisation / debugging).
    """
    rng = np.random.default_rng(seed)
    verts = np.array(vertices, dtype=np.float64)
    faces_arr = np.asarray(faces, dtype=np.int32)

    # Original geometry analysis
    bbox_min_orig = verts.min(axis=0)
    bbox_max_orig = verts.max(axis=0)
    bbox_size_orig = bbox_max_orig - bbox_min_orig
    diag = np.linalg.norm(bbox_size_orig)

    flat_axis = int(np.argmin(bbox_size_orig))
    span_axes = sorted([i for i in range(3) if i != flat_axis])

    curves_info = []

    for di in range(n_deformations):
        # Current bounding box (updates after each fold)
        bbox_min = verts.min(axis=0)
        bbox_max = verts.max(axis=0)
        bbox_size = bbox_max - bbox_min
        center = (bbox_min + bbox_max) * 0.5

        # ── Choose deformation axis & parameters ─────────────────
        da = span_axes[rng.choice(len(span_axes))]
        other_span = [a for a in span_axes if a != da][0]
        da_min, da_max = float(bbox_min[da]), float(bbox_max[da])
        da_span = max(da_max - da_min, 1e-6)

        side = rng.choice([-1, 1])        # which end to fold
        coverage = rng.uniform(0.2, 0.4)  # fraction of axis to fold

        # ── Influence mask (strip or corner) ─────────────────────
        use_corner = rng.random() < 0.5
        if side > 0:
            threshold = da_max - coverage * da_span
            mask_da = verts[:, da] >= threshold
        else:
            threshold = da_min + coverage * da_span
            mask_da = verts[:, da] <= threshold

        if use_corner:
            os_min, os_max = float(bbox_min[other_span]), float(bbox_max[other_span])
            os_span = max(os_max - os_min, 1e-6)
            os_side = rng.choice([-1, 1])
            os_coverage = rng.uniform(0.3, 0.6)
            if os_side > 0:
                os_threshold = os_max - os_coverage * os_span
                mask_os = verts[:, other_span] >= os_threshold
            else:
                os_threshold = os_min + os_coverage * os_span
                mask_os = verts[:, other_span] <= os_threshold
            mask = mask_da & mask_os
        else:
            mask = mask_da

        if mask.sum() < 5:
            continue

        # ── Per-vertex blend weights (smooth ramp) ───────────────
        if side > 0:
            da_depth = np.clip(
                (verts[:, da] - threshold) / max(da_max - threshold, 1e-6),
                0, 1)
        else:
            da_depth = np.clip(
                (threshold - verts[:, da]) / max(threshold - da_min, 1e-6),
                0, 1)
        weights = _smoothstep_arr(da_depth)

        if use_corner:
            if os_side > 0:
                os_depth = np.clip(
                    (verts[:, other_span] - os_threshold) / max(os_max - os_threshold, 1e-6),
                    0, 1)
            else:
                os_depth = np.clip(
                    (os_threshold - verts[:, other_span]) / max(os_threshold - os_min, 1e-6),
                    0, 1)
            weights *= _smoothstep_arr(os_depth)

        weights[~mask] = 0.0

        # ── Rest curve (straight line along da) ──────────────────
        rest_s = center.copy(); rest_s[da] = da_min
        rest_e = center.copy(); rest_e[da] = da_max
        rest_curve = BezierCurve3D.make_line(rest_s, rest_e)

        # ── Fold geometry ────────────────────────────────────────
        rise_height = severity * diag * rng.uniform(0.25, 0.6)
        fold_sign = rng.choice([-1, 1])

        # Rise direction (primarily along flat axis, slight tilt)
        rise_vec = np.zeros(3)
        rise_vec[flat_axis] = fold_sign
        rise_vec[other_span] += rng.uniform(-0.2, 0.2)
        rise_vec /= np.linalg.norm(rise_vec)

        through_depth = rise_height * rng.uniform(0.2, 0.5)
        # How far the endpoint folds back into the undeformed region
        inward_shift = coverage * da_span * rng.uniform(1.2, 2.0)

        # ── Build deformed-curve keypoints ───────────────────────
        n_keys = 7
        keys = np.zeros((n_keys, 3))
        for k in range(n_keys):
            t = k / (n_keys - 1)
            keys[k] = rest_s + (rest_e - rest_s) * t

        if side > 0:
            # Fold the positive-da end
            keys[4] += rise_vec * rise_height * 0.3     # transition
            keys[5] += rise_vec * rise_height            # peak
            keys[6] -= rise_vec * through_depth          # through mesh
            keys[6][da] -= inward_shift                  # overlap undeformed
        else:
            # Fold the negative-da end
            keys[2] += rise_vec * rise_height * 0.3
            keys[1] += rise_vec * rise_height
            keys[0] -= rise_vec * through_depth
            keys[0][da] += inward_shift

        deformed_curve = BezierCurve3D.from_keypoints(keys, smoothness=0.7)

        # ── Apply deformation ────────────────────────────────────
        verts = curve_deform(verts, rest_curve, deformed_curve,
                             weights=weights)

        # ── Optional inter-fold smoothing ────────────────────────
        if smooth_iterations > 0:
            verts = laplacian_smooth(verts, faces_arr,
                                     iterations=smooth_iterations,
                                     lambda_factor=0.2,
                                     pin_boundary=False)

        curves_info.append({
            'deform_index': di,
            'axis': da,
            'side': int(side),
            'coverage': coverage,
            'use_corner': use_corner,
            'mask_count': int(mask.sum()),
            'rise_height': float(rise_height),
            'through_depth': float(through_depth),
            'inward_shift': float(inward_shift),
            'keypoints': keys.tolist(),
        })

    return verts, curves_info


# Debug and visualization helpers.

def save_deformed_mesh(path, vertices, faces):
    """Save deformed mesh as OBJ."""
    with open(path, 'w') as f:
        for v in vertices:
            f.write(f"v {v[0]:.8f} {v[1]:.8f} {v[2]:.8f}\n")
        for face in faces:
            f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")
    print(f"[mesh_deformation] Saved {path}  "
          f"({len(vertices)} verts, {len(faces)} faces)")


def save_curves_as_obj(path, curves_info, n_samples=64):
    """Export Bézier curve keypoints and sampled curves as OBJ lines
    for visualisation in MeshLab / Blender.
    """
    with open(path, 'w') as f:
        vid = 1
        for ci, info in enumerate(curves_info):
            keys = np.array(info['keypoints'])
            curve = BezierCurve3D.from_keypoints(keys, smoothness=0.7)
            pts = curve.evaluate_batch(np.linspace(0, 1, n_samples))
            for p in pts:
                f.write(f"v {p[0]:.6f} {p[1]:.6f} {p[2]:.6f}\n")
            indices = " ".join(str(vid + j) for j in range(n_samples))
            f.write(f"l {indices}\n")
            vid += n_samples
            # Also write keypoints as separate vertices
            for p in keys:
                f.write(f"v {p[0]:.6f} {p[1]:.6f} {p[2]:.6f}\n")
            vid += len(keys)
    print(f"[mesh_deformation] Saved curves → {path}")
