"""
2D Rope Untangling via RayCasting-based Penetration Response (PRP).

A minimal Python re-implementation of the PRP pipeline used in the C++ solver
(`host_resolve_intersections_PRP` in `Solver/CollisionDetector/intersection_resolver2.cpp`),
specialized to 2D so it is small enough to ship as a SIGGRAPH supplementary example.

Pipeline per simulation step:
    1.  Newton-Raphson iteration on the combined spring + response energy.
    2.  EE intersection detection (O(m*n) brute force, every edge pair).
    3.  For each contour (= one EE pair):
            a. direction r := min-variance axis of the 4 contour points (2D LDA).
                        b. build several 2D combo directions (principal axis, perpendicular,
                             world axes), and for each combo:
                                     - expand k-ring candidates until the valid hit set stabilizes.
                                     - VE raycast: every V along combo direction, keep all
                                         non-incident E hits.
                                     - EV raycast: reverse VE raycast, reinterpret as E->V.
                   - cluster culling via topological adjacency.
                   - keep only clusters reachable to the contour boundary.
                                     - rank by response objective
                                         0.5 * k * sum(area) * (max_depth + d_hat)^2.
                             keep the best converged combo.
    4.  Contour merging: if two contours share any valid hit primitive,
        keep only the contour with the larger penetration objective.
    5.  Build VE response pairs:    p = b1*p1 + b2*p2,
                                    grad weights: V:+1, V1:-b1, V2:-b2.
    6.  Apply attractive response gradient.

Run:
    python Example2D/rope2d_prp.py
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.collections import LineCollection

# ---------------------------------------------------------------------------
# Initial curve (user-supplied control points).
# ---------------------------------------------------------------------------

# You can generate bazier point on the website: https://beziercurve.net/
BEZIER_POINTS = [
    (86, 284),
    (147, 185),
    (194, 284),
    (220, 324),
    (278, 402),
    (299, 285),
    (327, 218),
    (361, 202),
    (389, 230),
    (425, 315),
    (465, 329),
    (516, 304),
    (504, 225),
    (529, 209),
    (567, 184),
    (604, 242),
    (597, 424),
    (653, 295),
    (676, 300),
    (694, 291),
    (688, 245),
    (659, 205),
    (615, 193),
    (476, 76),
    (465, 384),
    (385, 471),
    (322, 320),
    (255, 174),
    (197, 198),
    (151, 301),
    (116, 330),
    (110, 426),
]


def normalize_polyline_to_length(points: np.ndarray, target_length: float = 1.0) -> np.ndarray:
    """Scale the polyline to a 1m total arc length and center it around the origin."""
    if len(points) < 2:
        return points.copy()
    seg_len = np.linalg.norm(np.diff(points, axis=0), axis=1)
    total_length = float(seg_len.sum())
    if total_length < EPS:
        return points.copy()
    scale = target_length / total_length
    normalized = points * scale
    center = 0.5 * (normalized.min(axis=0) + normalized.max(axis=0))
    normalized -= center
    return normalized


def build_polyline(samples_per_segment: int = 6) -> np.ndarray:
    """Catmull-Rom interpolation through the control points (open curve)."""
    P = np.asarray(BEZIER_POINTS, dtype=np.float64)
    # Y-up convention (image coords are Y-down).
    P[:, 1] = -P[:, 1]
    n = len(P)
    out = []
    for i in range(n - 1):
        p0 = P[max(i - 1, 0)]
        p1 = P[i]
        p2 = P[i + 1]
        p3 = P[min(i + 2, n - 1)]
        for k in range(samples_per_segment):
            t = k / samples_per_segment
            t2, t3 = t * t, t * t * t
            pt = 0.5 * (
                (2.0 * p1)
                + (-p0 + p2) * t
                + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * t2
                + (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * t3
            )
            out.append(pt)
    out.append(P[-1])
    polyline = np.asarray(out, dtype=np.float64)
    return normalize_polyline_to_length(polyline, target_length=1.0)


# ---------------------------------------------------------------------------
# Geometry helpers (2D).
# ---------------------------------------------------------------------------

EPS = 1e-9
DEFAULT_SPRING_K = 600.0
DEFAULT_RESPONSE_K = 4000.0
DEFAULT_STEP = 1.0
DEFAULT_NEWTON_ITERS = 1
DEFAULT_NEWTON_REG = 1.0e-6
DEFAULT_LINE_SEARCH_SHRINK = 0.5
DEFAULT_LINE_SEARCH_C = 1.0e-4
DEFAULT_MAX_CORRECTION_DISPLACEMENT = 2.0e-2
DEFAULT_PENALTY_DHAT = 5.0e-3
DEFAULT_PENALTY_VE_BARY_MARGIN = 1.0e-4
DEFAULT_EE_BARY_MARGIN = 1e-5
DEFAULT_EE_VE_DISTANCE_FACTOR = 1e-5

def point_segment_distance(point, a, b) -> Tuple[float, float]:
    """Distance from a point to segment AB, returning (distance, barycentric_u)."""
    ab = b - a
    denom = float(ab @ ab)
    if denom < EPS:
        return float(np.linalg.norm(point - a)), 0.0
    u = float(np.clip(((point - a) @ ab) / denom, 0.0, 1.0))
    proj = a + u * ab
    return float(np.linalg.norm(point - proj)), u


def segments_intersect(a, b, c, d, bary_margin: float = EPS) -> Optional[Tuple[float, float]]:
    """Return (u_ab, u_cd) if segments AB and CD properly cross, else None.
    `u` are barycentric in [0, 1]."""
    r = b - a
    s = d - c
    denom = r[0] * s[1] - r[1] * s[0]
    if abs(denom) < EPS:
        return None
    diff = c - a
    t = (diff[0] * s[1] - diff[1] * s[0]) / denom
    u = (diff[0] * r[1] - diff[1] * r[0]) / denom
    if bary_margin < t < 1 - bary_margin and bary_margin < u < 1 - bary_margin:
        return float(t), float(u)
    return None


def ee_pair_intersection(
    a,
    b,
    c,
    d,
    bary_margin: float,
    ve_distance_threshold: float,
) -> Optional[Tuple[float, float]]:
    hit = segments_intersect(a, b, c, d, bary_margin=bary_margin)
    if hit is None:
        return None
    min_ve_dist = min(
        point_segment_distance(a, c, d)[0],
        point_segment_distance(b, c, d)[0],
        point_segment_distance(c, a, b)[0],
        point_segment_distance(d, a, b)[0],
    )
    if min_ve_dist <= ve_distance_threshold:
        return None
    return hit


def ray_segment_hit(origin, direction, a, b) -> Optional[Tuple[float, float]]:
    """Ray origin + t*direction (t>=0) vs segment AB. Returns (t, u in [0,1])."""
    M00, M01 = direction[0], -(b[0] - a[0])
    M10, M11 = direction[1], -(b[1] - a[1])
    det = M00 * M11 - M01 * M10
    if abs(det) < EPS:
        return None
    rhs0 = a[0] - origin[0]
    rhs1 = a[1] - origin[1]
    t = (rhs0 * M11 - rhs1 * M01) / det
    u = (M00 * rhs1 - M10 * rhs0) / det
    if t <= EPS or u < -EPS or u > 1 + EPS:
        return None
    return float(t), float(np.clip(u, 0.0, 1.0))


# ---------------------------------------------------------------------------
# Hit / cluster data.
# ---------------------------------------------------------------------------

@dataclass
class Hit:
    kind: str           # 'VE' or 'EV'
    v: int              # the vertex involved
    e: int              # the edge involved
    t: float            # ray distance
    bary: float         # u along edge (so p = (1-u)*v1 + u*v2)
    src_side: int = -1  # ordered contour-side label: 0 for e1-side, 1 for e2-side
    dst_side: int = -1


@dataclass
class ContourSolution:
    e1: int
    e2: int
    direction: np.ndarray
    sign: int
    objective: float = math.inf
    support: int = 0
    travel: float = 0.0
    area_sum: float = 0.0
    max_depth: float = 0.0
    combo_name: str = ""
    k_ring: int = 0
    hits: List[Hit] = field(default_factory=list)


@dataclass
class PenaltyContact:
    kind: str
    indices: Tuple[int, ...]
    coeffs: Tuple[float, ...]


# ---------------------------------------------------------------------------
# Core PRP routines.
# ---------------------------------------------------------------------------

class RopePRP:
    def __init__(
        self,
        positions: np.ndarray,
        spring_k: float = DEFAULT_SPRING_K,
        response_k: float = DEFAULT_RESPONSE_K,
        step: float = DEFAULT_STEP,
        pin_endpoints: bool = True,
    ):
        self.x = positions.copy()
        self.n_verts = len(self.x)
        self.edges = np.array(
            [(i, i + 1) for i in range(self.n_verts - 1)], dtype=np.int32
        )
        self.n_edges = len(self.edges)
        self.rest_len = np.linalg.norm(
            self.x[self.edges[:, 0]] - self.x[self.edges[:, 1]], axis=1
        )
        self.vertex_area = np.zeros(self.n_verts, dtype=np.float64)
        np.add.at(self.vertex_area, self.edges[:, 0], 0.5 * self.rest_len)
        np.add.at(self.vertex_area, self.edges[:, 1], 0.5 * self.rest_len)
        # Per-vertex adjacent edges (only 2 each for an open polyline).
        self.vert_adj_edges: List[List[int]] = [[] for _ in range(self.n_verts)]
        for ei, (a, b) in enumerate(self.edges):
            self.vert_adj_edges[a].append(ei)
            self.vert_adj_edges[b].append(ei)

        self.spring_k = spring_k
        self.response_k = response_k
        self.response_dhat = 3.0e-3
        self.step = step
        self.newton_iters = DEFAULT_NEWTON_ITERS
        self.newton_reg = DEFAULT_NEWTON_REG
        self.line_search_shrink = DEFAULT_LINE_SEARCH_SHRINK
        self.line_search_c = DEFAULT_LINE_SEARCH_C
        self.max_correction_displacement = DEFAULT_MAX_CORRECTION_DISPLACEMENT
        self.penalty_k = response_k
        self.penalty_dhat = DEFAULT_PENALTY_DHAT
        self.penalty_ve_bary_margin = DEFAULT_PENALTY_VE_BARY_MARGIN
        self.ee_bary_margin = DEFAULT_EE_BARY_MARGIN
        self.ee_ve_distance_threshold = (
            DEFAULT_EE_VE_DISTANCE_FACTOR * float(np.mean(self.rest_len))
        )
        self.pin_mask = np.ones(self.n_verts, dtype=bool)
        if pin_endpoints:
            self.pin_mask[0] = False
            self.pin_mask[-1] = False

        self.last_ee_pairs: List[Tuple[int, int]] = []
        self.last_responses: List[Tuple[int, int, int, float, float]] = []
        self.last_penalty_contacts: List[PenaltyContact] = []

    # ----- energies / gradients ---------------------------------------------

    def free_dof_indices(self) -> np.ndarray:
        free_vertices = np.flatnonzero(self.pin_mask)
        dofs = np.empty(free_vertices.size * 2, dtype=np.int32)
        dofs[0::2] = 2 * free_vertices
        dofs[1::2] = 2 * free_vertices + 1
        return dofs

    @staticmethod
    def _add_block(H: np.ndarray, i: int, j: int, block: np.ndarray) -> None:
        H[2 * i:2 * i + 2, 2 * j:2 * j + 2] += block

    def spring_energy(self, positions: Optional[np.ndarray] = None) -> float:
        if positions is None:
            positions = self.x
        d = positions[self.edges[:, 0]] - positions[self.edges[:, 1]]
        lengths = np.linalg.norm(d, axis=1)
        return float(0.5 * self.spring_k * np.sum((lengths - self.rest_len) ** 2))

    def spring_energy_grad_hessian(
        self,
        positions: Optional[np.ndarray] = None,
    ) -> Tuple[float, np.ndarray, np.ndarray]:
        if positions is None:
            positions = self.x
        g = np.zeros_like(positions)
        H = np.zeros((2 * self.n_verts, 2 * self.n_verts), dtype=np.float64)
        energy = 0.0
        eye2 = np.eye(2, dtype=np.float64)
        for eid, (a, b) in enumerate(self.edges):
            d = positions[a] - positions[b]
            length = float(np.linalg.norm(d))
            rest = float(self.rest_len[eid])
            if length < EPS:
                continue
            energy += 0.5 * self.spring_k * (length - rest) ** 2
            grad_local = self.spring_k * (length - rest) / length * d
            g[a] += grad_local
            g[b] -= grad_local

            tangent = d / length
            block = self.spring_k * (
                (1.0 - rest / length) * eye2
                + (rest / length) * np.outer(tangent, tangent)
            )
            self._add_block(H, int(a), int(a), block)
            self._add_block(H, int(b), int(b), block)
            self._add_block(H, int(a), int(b), -block)
            self._add_block(H, int(b), int(a), -block)
        return energy, g, H

    def spring_grad(self) -> np.ndarray:
        g = np.zeros_like(self.x)
        d = self.x[self.edges[:, 0]] - self.x[self.edges[:, 1]]
        l = np.linalg.norm(d, axis=1)
        safe = np.maximum(l, EPS)
        f = (self.spring_k * (l - self.rest_len) / safe)[:, None] * d
        np.add.at(g, self.edges[:, 0], f)
        np.add.at(g, self.edges[:, 1], -f)
        return g

    def response_grad(
        self, responses: List[Tuple[int, int, int, float, float]]
    ) -> np.ndarray:
        g = np.zeros_like(self.x)
        for v, a, b, w1, w2 in responses:
            diff = self.x[v] - (w1 * self.x[a] + w2 * self.x[b])
            g[v] += self.response_k * diff
            g[a] -= self.response_k * w1 * diff
            g[b] -= self.response_k * w2 * diff
        return g

    def response_energy(self,
        responses: List[Tuple[int, int, int, float, float]],
        positions: Optional[np.ndarray] = None,
    ) -> float:
        if positions is None:
            positions = self.x
        energy = 0.0
        for v, a, b, w1, w2 in responses:
            diff = positions[v] - (w1 * positions[a] + w2 * positions[b])
            energy += 0.5 * self.response_k * float(diff @ diff)
        return energy

    def response_energy_grad_hessian(
        self,
        responses: List[Tuple[int, int, int, float, float]],
        positions: Optional[np.ndarray] = None,
    ) -> Tuple[float, np.ndarray, np.ndarray]:
        if positions is None:
            positions = self.x
        g = np.zeros_like(positions)
        H = np.zeros((2 * self.n_verts, 2 * self.n_verts), dtype=np.float64)
        energy = 0.0
        eye2 = np.eye(2, dtype=np.float64)
        for v, a, b, w1, w2 in responses:
            diff = positions[v] - (w1 * positions[a] + w2 * positions[b])
            energy += 0.5 * self.response_k * float(diff @ diff)
            g[v] += self.response_k * diff
            g[a] -= self.response_k * w1 * diff
            g[b] -= self.response_k * w2 * diff

            coeffs = ((v, 1.0), (a, -w1), (b, -w2))
            for i, ci in coeffs:
                for j, cj in coeffs:
                    self._add_block(H, int(i), int(j), self.response_k * ci * cj * eye2)
        return energy, g, H

    def collect_penalty_contacts(
        self,
        positions: Optional[np.ndarray] = None,
    ) -> List[PenaltyContact]:
        if positions is None:
            positions = self.x
        contacts: List[PenaltyContact] = []

        i_idx, j_idx = np.triu_indices(self.n_verts, k=1)
        non_adjacent = np.abs(i_idx - j_idx) > 1
        if np.any(non_adjacent):
            vv_i = i_idx[non_adjacent]
            vv_j = j_idx[non_adjacent]
            vv_diff = positions[vv_i] - positions[vv_j]
            vv_dist = np.linalg.norm(vv_diff, axis=1)
            vv_sel = (vv_dist > EPS) & (vv_dist < self.penalty_dhat)
            for i, j in zip(vv_i[vv_sel], vv_j[vv_sel]):
                contacts.append(
                    PenaltyContact(
                        kind='VV',
                        indices=(int(i), int(j)),
                        coeffs=(1.0, -1.0),
                    )
                )

        for v in range(self.n_verts):
            p = positions[v]
            for e, (a, b) in enumerate(self.edges):
                a = int(a)
                b = int(b)
                if abs(v - a) <= 1 or abs(v - b) <= 1:
                    continue
                dist, bary = point_segment_distance(p, positions[a], positions[b])
                if dist >= self.penalty_dhat:
                    continue
                if bary <= self.penalty_ve_bary_margin or bary >= 1.0 - self.penalty_ve_bary_margin:
                    continue
                w2 = float(bary)
                w1 = 1.0 - w2
                contacts.append(
                    PenaltyContact(
                        kind='VE',
                        indices=(int(v), a, b),
                        coeffs=(1.0, -w1, -w2),
                    )
                )
        return contacts

    def penalty_energy(
        self,
        contacts: List[PenaltyContact],
        positions: Optional[np.ndarray] = None,
    ) -> float:
        if positions is None:
            positions = self.x
        energy = 0.0
        for contact in contacts:
            diff = np.zeros(2, dtype=np.float64)
            for idx, coeff in zip(contact.indices, contact.coeffs):
                diff += coeff * positions[idx]
            dist = float(np.linalg.norm(diff))
            if dist >= self.penalty_dhat:
                continue
            energy += 0.5 * self.penalty_k * (dist - self.penalty_dhat) ** 2
        return energy

    def penalty_energy_grad_hessian(
        self,
        contacts: List[PenaltyContact],
        positions: Optional[np.ndarray] = None,
    ) -> Tuple[float, np.ndarray, np.ndarray]:
        if positions is None:
            positions = self.x
        g = np.zeros_like(positions)
        H = np.zeros((2 * self.n_verts, 2 * self.n_verts), dtype=np.float64)
        energy = 0.0
        eye2 = np.eye(2, dtype=np.float64)
        for contact in contacts:
            diff = np.zeros(2, dtype=np.float64)
            for idx, coeff in zip(contact.indices, contact.coeffs):
                diff += coeff * positions[idx]
            dist = float(np.linalg.norm(diff))
            if dist < EPS or dist >= self.penalty_dhat:
                continue

            energy += 0.5 * self.penalty_k * (dist - self.penalty_dhat) ** 2
            scale = self.penalty_k * (dist - self.penalty_dhat) / dist
            grad_local = scale * diff
            tangent = diff / dist
            block = self.penalty_k * (
                (1.0 - self.penalty_dhat / dist) * eye2
                + (self.penalty_dhat / dist) * np.outer(tangent, tangent)
            )

            for idx, coeff in zip(contact.indices, contact.coeffs):
                g[idx] += coeff * grad_local
            for i, ci in zip(contact.indices, contact.coeffs):
                for j, cj in zip(contact.indices, contact.coeffs):
                    self._add_block(H, int(i), int(j), ci * cj * block)
        return energy, g, H

    def fixed_total_energy(
        self,
        responses: List[Tuple[int, int, int, float, float]],
        penalty_contacts: List[PenaltyContact],
        positions: Optional[np.ndarray] = None,
    ) -> float:
        return (
            self.spring_energy(positions)
            + self.response_energy(responses, positions)
            + self.penalty_energy(penalty_contacts, positions)
        )

    def total_energy_grad_hessian(
        self,
        responses: List[Tuple[int, int, int, float, float]],
        penalty_contacts: List[PenaltyContact],
        positions: Optional[np.ndarray] = None,
    ) -> Tuple[float, np.ndarray, np.ndarray]:
        spring_energy, spring_grad, spring_hess = self.spring_energy_grad_hessian(positions)
        response_energy, response_grad, response_hess = self.response_energy_grad_hessian(responses, positions)
        penalty_energy, penalty_grad, penalty_hess = self.penalty_energy_grad_hessian(penalty_contacts, positions)
        return (
            spring_energy + response_energy + penalty_energy,
            spring_grad + response_grad + penalty_grad,
            spring_hess + response_hess + penalty_hess,
        )

    def compute_newton_direction(
        self,
        responses: List[Tuple[int, int, int, float, float]],
        penalty_contacts: List[PenaltyContact],
    ) -> Tuple[Optional[np.ndarray], float, float, float]:
        energy, grad, H = self.total_energy_grad_hessian(responses, penalty_contacts)
        free_dofs = self.free_dof_indices()
        if free_dofs.size == 0:
            return None, energy, 0.0, 0.0
        grad_flat = grad.reshape(-1)
        g_free = grad_flat[free_dofs]
        grad_norm = float(np.linalg.norm(g_free))
        if grad_norm < 1.0e-10:
            return None, energy, 0.0, grad_norm

        H_free = H[np.ix_(free_dofs, free_dofs)]
        reg = self.newton_reg * max(1.0, float(np.max(np.abs(np.diag(H_free)))))
        eye = np.eye(H_free.shape[0], dtype=np.float64)
        delta_free = None
        grad_dot_dir = 0.0
        for _ in range(8):
            try:
                delta_try = np.linalg.solve(H_free + reg * eye, -g_free)
            except np.linalg.LinAlgError:
                reg *= 10.0
                continue
            grad_dot_try = float(g_free @ delta_try)
            if grad_dot_try < 0.0:
                delta_free = delta_try
                grad_dot_dir = grad_dot_try
                break
            reg *= 10.0
        if delta_free is None:
            delta_free = -g_free
            grad_dot_dir = float(g_free @ delta_free)

        delta = np.zeros(2 * self.n_verts, dtype=np.float64)
        delta[free_dofs] = delta_free
        return delta.reshape(self.n_verts, 2), energy, grad_dot_dir, grad_norm

    def backtracking_line_search(
        self,
        responses: List[Tuple[int, int, int, float, float]],
        penalty_contacts: List[PenaltyContact],
        direction: np.ndarray,
        energy0: float,
        grad_dot_dir: float,
    ) -> float:
        alpha = 1.0
        x0 = self.x.copy()
        for _ in range(16):
            candidate = x0 + alpha * direction
            candidate[~self.pin_mask] = x0[~self.pin_mask]
            trial_energy = self.fixed_total_energy(responses, penalty_contacts, candidate)
            if trial_energy <= energy0 + self.line_search_c * alpha * grad_dot_dir:
                return alpha
            alpha *= self.line_search_shrink
        return 0.0

    # ----- EE intersection detection ----------------------------------------

    def detect_ee_pairs(self) -> List[Tuple[int, int]]:
        """Vectorized EE intersection test on all (i<j) edge pairs."""
        E = self.n_edges
        X = self.x
        ea = self.edges[:, 0]
        eb = self.edges[:, 1]
        A = X[ea]
        B = X[eb]
        R = B - A                                 # (E, 2)
        i_idx, j_idx = np.triu_indices(E, k=1)
        # Skip pairs sharing a vertex.
        share = (
            (ea[i_idx] == ea[j_idx]) | (ea[i_idx] == eb[j_idx])
            | (eb[i_idx] == ea[j_idx]) | (eb[i_idx] == eb[j_idx])
        )
        Ri = R[i_idx]
        Rj = R[j_idx]
        Ai = A[i_idx]
        Aj = A[j_idx]
        denom = Ri[:, 0] * Rj[:, 1] - Ri[:, 1] * Rj[:, 0]
        diff = Aj - Ai
        with np.errstate(divide='ignore', invalid='ignore'):
            t = (diff[:, 0] * Rj[:, 1] - diff[:, 1] * Rj[:, 0]) / denom
            u = (diff[:, 0] * Ri[:, 1] - diff[:, 1] * Ri[:, 0]) / denom

        def batched_point_segment_distance(P, A0, B0):
            AB0 = B0 - A0
            denom0 = np.einsum('ij,ij->i', AB0, AB0)
            bary0 = np.zeros(P.shape[0], dtype=np.float64)
            valid0 = denom0 > EPS
            bary0[valid0] = np.einsum('ij,ij->i', P[valid0] - A0[valid0], AB0[valid0]) / denom0[valid0]
            bary0 = np.clip(bary0, 0.0, 1.0)
            proj0 = A0 + bary0[:, None] * AB0
            return np.linalg.norm(P - proj0, axis=1)

        d_ai_to_j = batched_point_segment_distance(Ai, Aj, Aj + Rj)
        d_bi_to_j = batched_point_segment_distance(Ai + Ri, Aj, Aj + Rj)
        d_aj_to_i = batched_point_segment_distance(Aj, Ai, Ai + Ri)
        d_bj_to_i = batched_point_segment_distance(Aj + Rj, Ai, Ai + Ri)
        min_ve_dist = np.minimum.reduce([d_ai_to_j, d_bi_to_j, d_aj_to_i, d_bj_to_i])

        good = (~share) & (np.abs(denom) > EPS) \
            & (t > self.ee_bary_margin) & (t < 1 - self.ee_bary_margin) \
            & (u > self.ee_bary_margin) & (u < 1 - self.ee_bary_margin) \
            & (min_ve_dist > self.ee_ve_distance_threshold)
        sel = np.where(good)[0]
        return [(int(i_idx[k]), int(j_idx[k])) for k in sel]

    # ----- direction (min-variance axis of contour points) ------------------

    def contour_direction(self, e1: int, e2: int) -> np.ndarray:
        idx = [self.edges[e1, 0], self.edges[e1, 1],
               self.edges[e2, 0], self.edges[e2, 1]]
        pts = self.x[idx]
        c = pts - pts.mean(axis=0, keepdims=True)
        cov = c.T @ c
        w, v = np.linalg.eigh(cov)        # ascending
        # Min-variance direction = eigenvector for smallest eigenvalue.
        d = v[:, 0]
        n = np.linalg.norm(d)
        if n < EPS:
            return np.array([1.0, 0.0])
        return d / n

    @staticmethod
    def _normalize_direction(direction: np.ndarray) -> Optional[np.ndarray]:
        n = np.linalg.norm(direction)
        if n < EPS:
            return None
        return np.asarray(direction / n, dtype=np.float64)

    def contour_combo_axes(self, e1: int, e2: int) -> List[Tuple[str, np.ndarray]]:
        """2D combo seeds: principal/perpendicular axes and world axes.

        Directions are deduplicated up to sign.
        """
        dirs: List[Tuple[str, np.ndarray]] = []

        def add_dir(name: str, direction: np.ndarray):
            d = self._normalize_direction(direction)
            if d is None:
                return
            for _, exist in dirs:
                if abs(float(np.dot(d, exist))) > 1.0 - 1.0e-6:
                    return
            dirs.append((name, d))

        principal = self.contour_direction(e1, e2)
        add_dir("principal", principal)
        add_dir("perp", np.array([-principal[1], principal[0]], dtype=np.float64))

        add_dir("world_x", np.array([1.0, 0.0], dtype=np.float64))
        add_dir("world_y", np.array([0.0, 1.0], dtype=np.float64))
        return dirs

    # ----- ray casts --------------------------------------------------------

    def raycast_VE(
        self,
        direction: np.ndarray,
        vert_mask: Optional[np.ndarray] = None,
        edge_mask: Optional[np.ndarray] = None,
        keep_all_hits: bool = True,
    ) -> List[Hit]:
        """Vectorized: every allowed vertex shoots a ray along `direction`.

        In PRP we need all hits to preserve the full topological adjacency graph
        before cluster culling. The old closest-hit truncation made some combos
        look artificially converged and biased the result toward local minima.
        """
        V = self.n_verts
        X = self.x
        ea = self.edges[:, 0]
        eb = self.edges[:, 1]
        A = X[ea]
        B = X[eb]
        AB = B - A
        dx, dy = float(direction[0]), float(direction[1])
        det = (-dx) * AB[:, 1] - (-dy) * AB[:, 0]
        diff = A[None, :, :] - X[:, None, :]
        t_num = diff[..., 0] * (-AB[:, 1])[None, :] - diff[..., 1] * (-AB[:, 0])[None, :]
        u_num = dx * diff[..., 1] - dy * diff[..., 0]
        with np.errstate(divide='ignore', invalid='ignore'):
            t = t_num / det[None, :]
            u = u_num / det[None, :]
        valid = (np.abs(det)[None, :] > EPS) & (t > EPS) \
            & (u > -EPS) & (u < 1 + EPS)
        v_idx = np.arange(V)[:, None]
        incident = (ea[None, :] == v_idx) | (eb[None, :] == v_idx)
        valid &= ~incident
        if edge_mask is not None:
            valid &= edge_mask[None, :]
        out: List[Hit] = []
        for v in range(V):
            if vert_mask is not None and not vert_mask[v]:
                continue
            hit_edges = np.flatnonzero(valid[v])
            if hit_edges.size == 0:
                continue
            order = hit_edges[np.argsort(t[v, hit_edges])]
            if not keep_all_hits:
                order = order[:1]
            for e in order:
                out.append(Hit('VE', v=int(v), e=int(e),
                               t=float(t[v, e]),
                               bary=float(np.clip(u[v, e], 0.0, 1.0))))
        return out

    # ----- cluster culling --------------------------------------------------

    def _hits_share_topology(self, h1: Hit, h2: Hit) -> bool:
        """Adjacency rules requested by user.

        VE-VE : same destination edge.
        EV-EV : same source edge (symmetric to reverse VE casts).
        VE-EV : VE.e is adjacent to EV.v  AND  EV.e is adjacent to VE.v.
        """
        if h1.src_side != h2.src_side or h1.dst_side != h2.dst_side:
            return False
        if h1.kind == 'VE' and h2.kind == 'VE':
            return h1.e == h2.e
        if h1.kind == 'EV' and h2.kind == 'EV':
            return h1.e == h2.e
        # Mixed: choose ve, ev pair.
        ve = h1 if h1.kind == 'VE' else h2
        ev = h2 if h1.kind == 'VE' else h1
        ve_e_endpoints = set(self.edges[ve.e].tolist())
        ev_e_endpoints = set(self.edges[ev.e].tolist())
        return (ev.v in ve_e_endpoints) and (ve.v in ev_e_endpoints)

    def build_clusters(self, hits: List[Hit]) -> List[List[int]]:
        n = len(hits)
        if n == 0:
            return []
        # Build adjacency.
        adj: List[List[int]] = [[] for _ in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                if self._hits_share_topology(hits[i], hits[j]):
                    adj[i].append(j)
                    adj[j].append(i)
        seen = [False] * n
        clusters: List[List[int]] = []
        for s in range(n):
            if seen[s]:
                continue
            stack = [s]
            seen[s] = True
            comp: List[int] = []
            while stack:
                u = stack.pop()
                comp.append(u)
                for w in adj[u]:
                    if not seen[w]:
                        seen[w] = True
                        stack.append(w)
            clusters.append(comp)
        return clusters

    # ----- contour adjacency filter -----------------------------------------

    def _hit_is_contour_adjacent(
        self, h: Hit, e1: int, e2: int
    ) -> bool:
        if h.src_side < 0 or h.dst_side < 0:
            return False
        src_edge = e1 if h.src_side == 0 else e2
        dst_edge = e1 if h.dst_side == 0 else e2
        src_endpoints = (int(self.edges[src_edge, 0]), int(self.edges[src_edge, 1]))
        dst_endpoints = (int(self.edges[dst_edge, 0]), int(self.edges[dst_edge, 1]))
        if h.kind == 'VE':
            return h.v in src_endpoints and h.e == dst_edge
        return h.e == src_edge and h.v in dst_endpoints

    def filter_contour_clusters(
        self, hits: List[Hit], clusters: List[List[int]], e1: int, e2: int
    ) -> List[Hit]:
        best_hits: List[Hit] = []
        best_key: Optional[Tuple[float, float, int, float]] = None
        for comp in clusters:
            comp_hits = [hits[i] for i in comp]
            if not any(self._hit_is_contour_adjacent(hit, e1, e2) for hit in comp_hits):
                continue
            comp_hits = self.compress_hits_by_vertex(comp_hits)
            if not comp_hits:
                continue
            objective, _area_sum, max_depth = self.response_energy_metrics(comp_hits)
            travel = float(sum(hit.t for hit in comp_hits))
            key = (objective, max_depth, len(comp_hits), travel)
            if best_key is None or key < best_key:
                best_hits = comp_hits
                best_key = key
        return best_hits

    @staticmethod
    def compress_hits_by_vertex(hits: List[Hit]) -> List[Hit]:
        """Use all-hits for connectivity, but keep only one representative hit
        per moved vertex when scoring/building responses.

        The representative is the farthest valid hit for that vertex, which keeps
        the large-scale displacement intent of the combo without over-counting the
        same vertex across many destination primitives.
        """
        best_by_vertex: dict[Tuple[int, int, int], Hit] = {}
        for hit in hits:
            key = (hit.src_side, hit.dst_side, hit.v)
            prev = best_by_vertex.get(key)
            if prev is None or hit.t > prev.t:
                best_by_vertex[key] = hit
        return list(best_by_vertex.values())

    # ----- penetration objective --------------------------------------------

    def hit_area(self, hit: Hit) -> float:
        return float(self.vertex_area[hit.v])

    def response_energy_metrics(self, hits: List[Hit]) -> Tuple[float, float, float]:
        """Return (objective, area_sum, max_depth) with the same contour-level
        max-depth semantics as PRP response evaluation."""
        if not hits:
            return math.inf, 0.0, 0.0
        area_sum = float(sum(self.hit_area(hit) for hit in hits))
        max_depth = float(max(hit.t for hit in hits))
        objective = 0.5 * self.response_k * area_sum * (max_depth + self.response_dhat) ** 2
        return objective, area_sum, max_depth

    def penetration_objective(self, hits: List[Hit]) -> float:
        return self.response_energy_metrics(hits)[0]

    def contour_candidate_masks(
        self, e1: int, e2: int, k_ring: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        boundary_verts = {
            int(self.edges[e1, 0]), int(self.edges[e1, 1]),
            int(self.edges[e2, 0]), int(self.edges[e2, 1]),
        }
        vmask = np.zeros(self.n_verts, dtype=bool)
        if k_ring >= self.n_verts:
            vmask[:] = True
        else:
            for bv in boundary_verts:
                lo = max(0, bv - k_ring)
                hi = min(self.n_verts - 1, bv + k_ring)
                vmask[lo:hi + 1] = True
        emask = vmask[self.edges[:, 0]] & vmask[self.edges[:, 1]]
        return vmask, emask

    def contour_vertex_hop_distance(self, vid: int, edge: int) -> int:
        ea, eb = int(self.edges[edge, 0]), int(self.edges[edge, 1])
        return min(abs(vid - ea), abs(vid - eb))

    def contour_vertex_side(self, vid: int, e1: int, e2: int) -> int:
        e1_a, e1_b = int(self.edges[e1, 0]), int(self.edges[e1, 1])
        e2_a, e2_b = int(self.edges[e2, 0]), int(self.edges[e2, 1])
        d1 = min(abs(vid - e1_a), abs(vid - e1_b))
        d2 = min(abs(vid - e2_a), abs(vid - e2_b))
        return 0 if d1 <= d2 else 1

    def contour_edge_side(self, eid: int, e1: int, e2: int) -> int:
        va, vb = int(self.edges[eid, 0]), int(self.edges[eid, 1])
        d1 = min(self.contour_vertex_hop_distance(va, e1), self.contour_vertex_hop_distance(vb, e1))
        d2 = min(self.contour_vertex_hop_distance(va, e2), self.contour_vertex_hop_distance(vb, e2))
        return 0 if d1 <= d2 else 1

    def orient_hit(self, kind: str, hit: Hit, e1: int, e2: int) -> Optional[Hit]:
        vertex_side = self.contour_vertex_side(hit.v, e1, e2)
        edge_side = self.contour_edge_side(hit.e, e1, e2)
        if kind == 'VE':
            src_side, dst_side = vertex_side, edge_side
        else:
            src_side, dst_side = edge_side, vertex_side
        if src_side == dst_side:
            return None
        return Hit(kind=kind, v=hit.v, e=hit.e, t=hit.t, bary=hit.bary,
                   src_side=src_side, dst_side=dst_side)

    @staticmethod
    def hit_signature(hits: List[Hit]) -> Tuple[Tuple[str, int, int, int, int], ...]:
        return tuple(sorted((h.kind, h.v, h.e, h.src_side, h.dst_side) for h in hits))

    @staticmethod
    def is_better_solution(
        candidate: Optional[ContourSolution],
        incumbent: Optional[ContourSolution],
    ) -> bool:
        if candidate is None or not candidate.hits:
            return False
        if incumbent is None or not incumbent.hits:
            return True
        if not math.isclose(candidate.objective, incumbent.objective, rel_tol=1.0e-6, abs_tol=1.0e-6):
            return candidate.objective < incumbent.objective
        if not math.isclose(candidate.max_depth, incumbent.max_depth, rel_tol=1.0e-6, abs_tol=1.0e-6):
            return candidate.max_depth < incumbent.max_depth
        if candidate.support != incumbent.support:
            return candidate.support < incumbent.support
        if not math.isclose(candidate.travel, incumbent.travel, rel_tol=1.0e-6, abs_tol=1.0e-6):
            return candidate.travel < incumbent.travel
        return candidate.k_ring < incumbent.k_ring

    # ----- per-contour PRP evaluation ---------------------------------------

    def solve_contour(self, e1: int, e2: int, initial_k_ring: int = 8) -> ContourSolution:
        """Evaluate several 2D combo directions and expand the candidate ring
        until the retained hit set stabilizes."""
        k_rings = []
        k = max(4, initial_k_ring)
        while k < self.n_verts:
            k_rings.append(k)
            k *= 2
        if not k_rings or k_rings[-1] != self.n_verts:
            k_rings.append(self.n_verts)

        best: Optional[ContourSolution] = None
        for combo_name, base_dir in self.contour_combo_axes(e1, e2):
            for sign in (+1, -1):
                direction = sign * base_dir
                combo_best: Optional[ContourSolution] = None
                prev_signature: Optional[Tuple[Tuple[str, int, int, int, int], ...]] = None
                prev_objective = math.inf

                for k_ring in k_rings:
                    vmask, emask = self.contour_candidate_masks(e1, e2, k_ring)
                    ve_raw = self.raycast_VE(
                        direction,
                        vert_mask=vmask,
                        edge_mask=emask,
                        keep_all_hits=True,
                    )
                    ve_hits = []
                    for hit in ve_raw:
                        oriented = self.orient_hit('VE', hit, e1, e2)
                        if oriented is not None:
                            ve_hits.append(oriented)
                    ev_raw = self.raycast_VE(
                        -direction,
                        vert_mask=vmask,
                        edge_mask=emask,
                        keep_all_hits=True,
                    )
                    ev_hits = []
                    for hit in ev_raw:
                        oriented = self.orient_hit('EV', hit, e1, e2)
                        if oriented is not None:
                            ev_hits.append(oriented)
                    all_hits = ve_hits + ev_hits
                    clusters = self.build_clusters(all_hits)
                    valid_hits = self.filter_contour_clusters(all_hits, clusters, e1, e2)
                    if not valid_hits:
                        continue

                    travel = float(sum(h.t for h in valid_hits))
                    objective, area_sum, max_depth = self.response_energy_metrics(valid_hits)
                    sol = ContourSolution(
                        e1=e1,
                        e2=e2,
                        direction=np.asarray(direction, dtype=np.float64),
                        sign=sign,
                        objective=objective,
                        support=len(valid_hits),
                        travel=travel,
                        area_sum=area_sum,
                        max_depth=max_depth,
                        combo_name=f"{combo_name}{'+' if sign > 0 else '-'}",
                        k_ring=k_ring,
                        hits=valid_hits,
                    )
                    combo_best = sol

                    signature = self.hit_signature(valid_hits)
                    if prev_signature == signature and math.isclose(objective, prev_objective, rel_tol=1.0e-6, abs_tol=1.0e-6):
                        break
                    prev_signature = signature
                    prev_objective = objective

                if self.is_better_solution(combo_best, best):
                    best = combo_best

        if best is None:
            best = ContourSolution(
                e1=e1,
                e2=e2,
                direction=np.array([1.0, 0.0], dtype=np.float64),
                sign=+1,
            )
        return best

    # ----- contour merge ----------------------------------------------------

    @staticmethod
    def merge_contours(
        solutions: List[ContourSolution],
    ) -> List[ContourSolution]:
        """If two contours share any hit primitive (same v, same e),
        keep the globally better solution under the same comparator used for
        combo selection."""
        n = len(solutions)
        if n <= 1:
            return solutions
        # Build (v,e) signature sets per contour.
        sigs: List[set] = []
        for sol in solutions:
            sig = set()
            for h in sol.hits:
                sig.add((h.kind, h.v, h.e))
            sigs.append(sig)
        # Union-find over contours that share any signature.
        parent = list(range(n))
        def find(i):
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i
        def union(i, j):
            ri, rj = find(i), find(j)
            if ri != rj:
                parent[ri] = rj
        for i in range(n):
            for j in range(i + 1, n):
                if sigs[i] & sigs[j]:
                    union(i, j)
        groups: dict = {}
        for i in range(n):
            groups.setdefault(find(i), []).append(i)
        kept: List[ContourSolution] = []
        for members in groups.values():
            best_idx = members[0]
            for idx in members[1:]:
                if RopePRP.is_better_solution(solutions[idx], solutions[best_idx]):
                    best_idx = idx
            kept.append(solutions[best_idx])
        return kept

    # ----- responses from hits ----------------------------------------------

    def build_responses(
        self, solutions: List[ContourSolution]
    ) -> List[Tuple[int, int, int, float, float]]:
        out = []
        for sol in solutions:
            for h in sol.hits:
                a, b = int(self.edges[h.e, 0]), int(self.edges[h.e, 1])
                w2 = float(h.bary)
                w1 = 1.0 - w2
                # p = w1 * a + w2 * b, attractive pull V -> p.
                out.append((int(h.v), a, b, w1, w2))
        return out

    def update_raycast_preview(self) -> None:
        ee_pairs = self.detect_ee_pairs()
        self.last_ee_pairs = ee_pairs
        solutions = [self.solve_contour(e1, e2) for (e1, e2) in ee_pairs]
        solutions = [s for s in solutions if s.hits]
        solutions = self.merge_contours(solutions)
        self.last_responses = self.build_responses(solutions)

    # ----- one simulation step ----------------------------------------------

    def step_once(self):
        for _ in range(self.newton_iters):
            ee_pairs = self.detect_ee_pairs()
            self.last_ee_pairs = ee_pairs

            solutions = [self.solve_contour(e1, e2) for (e1, e2) in ee_pairs]
            solutions = [s for s in solutions if s.hits]
            solutions = self.merge_contours(solutions)
            responses = self.build_responses(solutions)
            self.last_responses = responses
            penalty_contacts = self.collect_penalty_contacts()
            self.last_penalty_contacts = penalty_contacts

            direction, energy0, grad_dot_dir, grad_norm = self.compute_newton_direction(responses, penalty_contacts)
            if direction is None or grad_norm < 1.0e-10:
                break

            alpha = self.backtracking_line_search(responses, penalty_contacts, direction, energy0, grad_dot_dir)
            if alpha <= 0.0:
                break

            step_update = alpha * direction
            if self.max_correction_displacement > 0.0:
                max_update = float(np.max(np.linalg.norm(step_update[self.pin_mask], axis=1)))
                if max_update > self.max_correction_displacement:
                    step_update *= self.max_correction_displacement / max_update
            self.x += step_update

            if float(np.max(np.linalg.norm(step_update[self.pin_mask], axis=1))) < 1.0e-9:
                break


# ---------------------------------------------------------------------------
# Visualization.
# ---------------------------------------------------------------------------

def run_and_animate(n_frames: int = 200, save_path: Optional[str] = None):
    pts = build_polyline(samples_per_segment=4)
    sim = RopePRP(pts, spring_k=DEFAULT_SPRING_K,
                  response_k=DEFAULT_RESPONSE_K,
                  step=DEFAULT_STEP)

    fig, ax = plt.subplots(figsize=(8, 8))
    extent = float(max(np.ptp(pts[:, 0]), np.ptp(pts[:, 1]), 1.0e-3))
    margin = 0.1 * extent
    xmin, xmax = pts[:, 0].min() - margin, pts[:, 0].max() + margin
    ymin, ymax = pts[:, 1].min() - margin, pts[:, 1].max() + margin
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect('equal')
    ax.set_facecolor('#fafafa')

    rope_line, = ax.plot([], [], '-', color='#1f77b4', lw=2.2)
    verts_scat = ax.scatter([], [], s=8, c='#1f77b4')
    cross_scat = ax.scatter([], [], s=70, marker='x', c='red', linewidths=2)
    resp_lines = LineCollection([], colors='orange', linewidths=1.2, alpha=0.8)
    ax.add_collection(resp_lines)
    title = ax.set_title('')

    # Shared frame counter / play state. In headless save mode `playing` is
    # forced True; in interactive mode SPACE toggles, P single-steps, S runs
    # a one-off raycast preview without advancing the simulation.
    state = {'frame': 0, 'playing': save_path is not None, 'step_once': False}

    def redraw():
        rope_line.set_data(sim.x[:, 0], sim.x[:, 1])
        verts_scat.set_offsets(sim.x)

        cross_pts = []
        for (e1, e2) in sim.last_ee_pairs:
            a, b = sim.x[sim.edges[e1, 0]], sim.x[sim.edges[e1, 1]]
            c, d = sim.x[sim.edges[e2, 0]], sim.x[sim.edges[e2, 1]]
            hit = ee_pair_intersection(
                a,
                b,
                c,
                d,
                bary_margin=sim.ee_bary_margin,
                ve_distance_threshold=sim.ee_ve_distance_threshold,
            )
            if hit is not None:
                t, _ = hit
                cross_pts.append(a + t * (b - a))
        cross_scat.set_offsets(np.asarray(cross_pts) if cross_pts
                               else np.empty((0, 2)))

        segs = []
        for v, a, b, w1, w2 in sim.last_responses:
            p = w1 * sim.x[a] + w2 * sim.x[b]
            segs.append([sim.x[v], p])
        resp_lines.set_segments(segs)

        mode = 'PLAY' if state['playing'] else 'PAUSED'
        title.set_text(
            f'[{mode}]  frame {state["frame"]}/{n_frames}   '
            f'EE: {len(sim.last_ee_pairs)}   '
            f'resp: {len(sim.last_responses)}   '
            f'(SPACE=play/pause, P=step, S=raycast, R=reset)'
        )

    def update(_):
        advance = state['playing'] or state['step_once']
        if advance and state['frame'] < n_frames:
            sim.step_once()
            state['frame'] += 1
            state['step_once'] = False
        redraw()
        return rope_line, verts_scat, cross_scat, resp_lines, title

    def on_key(event):
        if event.key == ' ':
            state['playing'] = not state['playing']
        elif event.key in ('p', 'P'):
            state['step_once'] = True
        elif event.key in ('s', 'S'):
            sim.update_raycast_preview()
            redraw()
            fig.canvas.draw_idle()
        elif event.key in ('r', 'R'):
            sim.x = pts.copy()
            sim.last_ee_pairs = []
            sim.last_responses = []
            state['frame'] = 0
            state['playing'] = False
            redraw()
            fig.canvas.draw_idle()

    fig.canvas.mpl_connect('key_press_event', on_key)

    # Initial render so the user sees the starting rope before pressing SPACE.
    redraw()

    if save_path:
        anim = FuncAnimation(fig, update, frames=n_frames,
                             interval=60, blit=False, repeat=False)
        anim.save(save_path, writer='pillow', fps=15)
        print(f'Saved animation to {save_path}')
    else:
        # Interactive: drive frames continuously, but only advance when
        # `state['playing']` or after a P press.
        anim = FuncAnimation(fig, update, frames=10**9,
                             interval=40, blit=False, repeat=False,
                             cache_frame_data=False)
        print('Controls:  SPACE = play/pause   P = step one frame   R = reset')
        plt.show()


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--frames', type=int, default=200)
    p.add_argument('--save', type=str, default=None,
                   help='Optional output path (e.g. rope.gif)')
    args = p.parse_args()
    run_and_animate(n_frames=args.frames, save_path=args.save)
