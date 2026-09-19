# PRP Coverage + Locality Selection — Plan (supersedes old P0/P1/P2/P3 plan)

Status: ACTIVE. Phase A committed. Phase B/C implemented and verified on canonical SMPL_028 gates; Phase D batch regression NOT yet implemented.
Companion doc: `ResearchLog/development.md` (architecture map + lessons; read it too).
Summary of prior fix: `ResearchLog/Constaint/prp_cluster_culling_fix.md`.

---

## 0. CORRECTED problem statement (read first — the old "size rejection" idea was WRONG)

Earlier we "fixed" the whole-mesh explosion by rejecting components that span too
many object vertices (`PRP_reject_whole_mesh_component_fraction = 0.5`) plus a
hit-count cap (5000). The dataset owner confirmed this is the **wrong axis**:

- On a **long contour** (the canonical case = `SMPL_028.obj` contour 1), the
  correct response is a **band of ~1000–3000 hits** running along the contour,
  with **contour coverage ~0.8 (must be > 0.5)**.
- That legitimate band is **fused inside the single large boundary-attached
  component** (5000–13000 hits depending on direction). It is NOT a separate
  neighbor cluster.
- Size rejection throws the whole band away; the fallback then grabs a **1-hit
  splinter** with near-zero coverage. Output like
  `Hit Counts = [2,4,9,7,1,12,2,6,3,1]` is this failure (local optimum =
  low-overlap tiny cluster), NOT success.
- Full coverage (→1.0) would need multiple directions. We DO NOT need that. We
  only need the single **main direction** to reach ~0.8 coverage at 1000–3000 hits.

So the fix is **coverage + locality selection**, not size rejection:
- Keep the **max-coverage** boundary-attached component (the band), do NOT reject
  it for being large.
- **Thin** that component by **target-to-contour hop locality** down to the band
  (~1000–3000), removing the "thickening" that grows away from the contour.
- Pick the **main direction** = the combo whose kept support has highest coverage.

### Canonical reproduction
```powershell
d:/Projects/Untangling25/.venv/Scripts/python.exe d:/Projects/Untangling25/PythonBindings/tests/test_untangling_jang2026.py --headless --advance_frames 1 --PRP_process_single_contour 1 --untangling_process_contour 1 --prp_dump_hits 1
```
- This uses `obj_dir = .../Dataset_distributed/Dataset_distributed/human/SMPL_028.obj`
  (hardcoded at test_untangling_jang2026.py ~L87). num_verts=6890. The contour-1
  intersection is very long.
- WRONG (current): `Contour 1 : Best Combo = 9, Hit count = 1`.
- TARGET: best combo selects ~1000–3000 valid hits, coverage ~0.8.

### Final goal (acceptance)
Pass BOTH:
1. `test_untangling_jang2026.py` (its own hardcoded knot/misc list + the SMPL/bear
   manual obj — run a few `--scene_id` and the default).
2. `test_batch_method_comparison_enhanced.py --dataset_mode jang --methods PRP --begin 1 --end 59 --no_export_per_frame --jang_dataset "D:\Projects\UntanglingProject\UntanglingProject\Demos\Benchmark\Dataset_distributed\Dataset_distributed"`
   (maximize convergence %, compare to AGENTS.md baseline PRP 94.9%).

---

## 1. What is ALREADY DONE (committed + this-session uncommitted)

### Committed (git log: 0e56256, bf91c8a)
- P0-1: single-best-component selection (replaced union-all) — but with the WRONG
  size rejection (to be replaced in Phase B).
- P0-2: contour-crossing adjacency cut re-enabled (`adjacency_crosses_contour_boundary`).
- P0-3: loop-vertex graph barrier in `find_adjacent_hits` (`loop_vertex_flag`).
- `prp_dump_hits` flag + per-hit CSV + `analyze_prp_hits.py`.
- `PRP_reject_whole_mesh_component_fraction` (0.5) + 5000 hit-count cap — **THESE
  ARE THE WRONG AXIS; Phase B disables them in the active path.**

### This session, UNCOMMITTED (Phase A instrumentation — built + verified):
- `intersection_resolver2.h` PRPMinContourHitInfo (~L578): added
  `float contour_coverage`, `uint covered_anchor_count`, `uint boundary_anchor_count`.
- `intersection_resolver2.cpp` assemble_contour_results (after `update_info_from_eval`,
  ~L5719+): compute `contour_coverage` = (distinct boundary anchors touched by kept
  hits via `nearest_boundary_vertex` on BOTH src and dst sides) / (|boundary_verts[0]∪[1]|).
  Stored in `info`.
- Summary stats (`prp_debug`): added `prp.c{N}.coverage`, `.covered_anchor_count`,
  `.boundary_anchor_count`, and per-combo `prp.c{N}.cmb{M}.coverage`,
  `.cmb{M}.valid_hit_count`.
- CSV dump: added columns `src_hop`, `tgt_hop` (min boundary hop distance over
  source / target verts via `runtime.boundary_hop_distance`). "-1" = unreachable.
- `analyze_prp_hits.py`: parses `src_hop`/`tgt_hop`; prints target hop-to-contour
  histogram (band vs leak diagnostic).

NOTE: Phase A coverage uses `runtime.nearest_boundary_vertex(mesh, vid)` and the
dense `boundary_hop_distance`. These come from
`calculate_hop_dist_and_geom_dist_in_mesh(...)` already filled in build_standard_runtime.

---

## 2. Gate GA findings (this session — these refine the plan)

Ran the reproduction with dump, analyzed `prp_hits_f0_c1_cmb*.csv`:

| combo | raw hits | boundary-attached comp 0 | comp0 boundary-adj | leak comp(-1) |
|---|---|---|---|---|
| 0 | 10452 | 9911 | 289 | 537 |
| 1 | 7961 | 5208 | 280 | small |
| 5 | 13668 | 13504 | 553 | 145 |
| 7 | 9346 | 8126 | 271 | 1210 |
| 9 | 12655 | 10710 | 390 | 1941 |

Key facts:
- **q_median ≈ 0.63** → NOT projection-degenerate. Projection-aware graph (old
  Phase P1) is NOT the bottleneck. Do not build it now.
- **Leak is NOT far-field**: combo 9 target hop-to-contour: ≤2=47.9%, ≤4=70.7%,
  ≤8=97.1%, **max=11**. P0-2/P0-3 already stopped whole-mesh spread (raw fell
  62763→12655). The residual issue is a **thick band**, not a mesh-wide leak.
- Current code rejects the 10710-hit band (size cap) → selects a 1-hit splinter.
- Each direction's boundary-attached component is the band+thickening; max-coverage
  selection will correctly prefer it over splinters. Thinning by tgt_hop locality
  gets it to the target 1000–3000.

CONCLUSION: implement coverage selection + hop-locality thinning (Phase B), then
main-direction-by-coverage (Phase C). Skip old P1 (projection graph) and P3 (atlas).

---

## 3. Code anchor points (verified line numbers drift ±; search names)

| Concern | File:approx line | Symbol |
|---|---|---|
| Hot lambda (per-combo eval) | intersection_resolver2.cpp ~4605–5853 | `evaluate_batched_contours_of_single_combo` |
| Flood-fill components | intersection_resolver2.cpp ~5345–5405 | `parallel_cluster_culling`, `candidate_output_hists` |
| **Selection block to REPLACE** | intersection_resolver2.cpp ~5529–5616 | non-winding `else { ... }` with size rejection (`obj_frac_threshold`, `hit_count_cap`, `best_component_idx`) |
| `PRP_not_cull_rays` override | intersection_resolver2.cpp ~5618 | leave as-is |
| Assemble + coverage + CSV | intersection_resolver2.cpp ~5652–5845 | `assemble_contour_results` |
| Coverage compute (Phase A, done) | intersection_resolver2.cpp ~5719+ | after `update_info_from_eval` |
| PRPMinContourHitInfo | intersection_resolver2.h ~543–591 | has `contour_coverage` now |
| Cross-direction comparator | intersection_resolver2.h ~787 | `prp_is_better_response_hit_info` |
| Validity predicate | intersection_resolver2.h ~777 | `prp_has_valid_hit_info` |
| Best-of-combos selection | intersection_resolver2.cpp ~2416 | `find_best_result_in_combo_results`, `select_runtime_best_results` |
| Runtime locality fields | intersection_resolver2.h ~624–716 | `boundary_hop_distance`, `nearest_boundary_vertex`, `vert_to_boundary_hop_dist/geom_dist`, `loop_vertex_flag`, `boundary_verts` |
| Hit visit helpers | intersection_resolver2.cpp ~1103–1141 | `visit_prp_hit_source_vertices`, `visit_prp_hit_target_vertices` |
| Scene params | scene_params.h ~161–167 | add new flags after `PRP_reject_whole_mesh_component_fraction` |
| Python bindings | python_bindings.cpp ~1709 | `def_readwrite` list for SceneParams |
| Response build (depth) | intersection_resolver2.cpp ~3558,3581 | `use_max_depth`, response from `min_hit_info` |

---

## 4. Phase B — Coverage + locality selection (THE CORE FIX)

> Historical design notes below: as of 2026-09-17, `PRP_locality_max_hop`
> is no longer a scene parameter. Both CPU/GPU resolver implementations fix
> `kPrpLocalityMaxHop` to the maximum unsigned integer (unlimited); the Python
> hop sweep described here would now require editing and recompiling C++.

### B0. Scene params (scene_params.h + python_bindings.cpp def_readwrite)
- `bool  PRP_use_coverage_locality_selection = true;`
- `uint  PRP_locality_max_hop = 1;`  // calibrated on SMPL_028: hop=1 balances single-contour target and full-scene convergence
- `float PRP_coverage_target = 0.5f;` // min acceptable coverage (aim 0.8)
- Keep `PRP_reject_whole_mesh_component_fraction` but the new path must NOT use it
  (or set its default so it never triggers when coverage selection is on).

### B1. Replace selection block (intersection_resolver2.cpp ~5529–5616)
When `PRP_use_coverage_locality_selection`:
1. For each boundary-attached component in `candidate_output_hists`, compute its
   **coverage** = distinct boundary anchors touched (reuse the Phase-A coverage
   logic: `nearest_boundary_vertex` on src+dst verts of the component's hits,
   intersect with `boundary_verts[0]∪[1]`). Also record component hit count.
2. **Select the max-coverage component** (tie-break: LARGER hit count → prefer the
   band, not a splinter). DO NOT reject for size/vertex-fraction.
3. **Locality-thin** the selected component: mark a hit valid iff
   `min_tgt_hop(hit) <= PRP_locality_max_hop`. (`min_tgt_hop` = min over target
   verts of `runtime.boundary_hop_distance(dst_mesh_idx, vid)`.) Source side is
   already k-ring-bounded, so target-hop thinning is the lever that removes the
   thickening while keeping the contour-adjacent band.
4. Mark only the thinned band's hits `packed_hit_valid = true`.
Else (flag off): keep current behavior (for A/B comparison).

### B2. Disable wrong-axis rejection in the active path
- In the new path, do NOT apply `obj_frac_threshold` or `hit_count_cap`. Leave the
  old code reachable only when `PRP_use_coverage_locality_selection == false`.

### B3. CALIBRATE `PRP_locality_max_hop` (no recompile — it's a scene param)
- Sweep hop ∈ {3,4,5,6,8} via Python on the reproduction; pick the smallest hop
  that yields **valid hit count ~1000–3000 AND coverage ≥ 0.5 (aim ~0.8)** for the
  selected (max-coverage) combo. Record chosen value as the default.
- Data point from Gate GA (combo 9, ALL hits incl leak): tgt_hop ≤2→6056, ≤4→8950.
  Within the band component only, the count will be lower; measure per-component
  after B1 lands. Expect hop≈3–5 to land in 1000–3000.

### Gate GB (single contour, SMPL_028 contour 1)
- best combo: valid hits ∈ [1000,3000], coverage ≥ 0.5 (target ~0.8).
- adaptive k-ring converges (no 72/69 failure).
- full SMPL_028 scene (advance_frames 30, all contours) still drives EF→0.

---

## 5. Phase C — Main-direction selection by coverage

### C1. Coverage-gated comparator
- `contour_coverage` is already stored in `PRPMinContourHitInfo` (Phase A).
- Wrap/extend `prp_is_better_response_hit_info` (intersection_resolver2.h ~787) so
  ordering is lexicographic:
  1. valid (`prp_has_valid_hit_info`).
  2. coverage tier: `coverage >= PRP_coverage_target` beats below-target.
  3. among same tier: existing order (penetration_objective → depth →
     displacement_cost → hit_count → combo_idx).
- Effect: a high-coverage band (combo with ~0.8) beats a low-coverage splinter even
  if the splinter has lower raw objective. This is what stops the 1-hit pick.
- Keep `use_max_depth = true`; depth is already computed on the finalized (thinned)
  support, so response magnitude follows the band.

### Gate GC
- Selected best combo for SMPL_028 contour 1 is the band (coverage ~0.8), and
  `opt.best_direction` is the LDA/main axis, not a degenerate off-axis splinter.

---

## 6. Phase D — Full regression and harden

### D1. test_untangling_jang2026.py
- Default (SMPL_028) advance_frames 30 → EF should trend to 0; per-contour coverage healthy.
- Spot-check a few `--scene_id` from its hardcoded knot/misc list for regressions.

### D2. batch single (bear / scene 1)
```powershell
... test_batch_method_comparison_enhanced.py --dataset_mode jang --methods PRP --begin 1 --end 1 --no_export_per_frame --jang_dataset "..."
```
- Expect explosion-free, improved EF trend vs prior (was oscillating ~800).
- If still not converging: check coverage per contour (selection) vs leak-not-thinned
  (raise/lower hop) vs genuine solver-rate (not a selection bug). Bear has 11318 verts.

### D3. batch FULL 1–59 (the acceptance run)
```powershell
... test_batch_method_comparison_enhanced.py --dataset_mode jang --methods PRP --begin 1 --end 59 --no_export_per_frame --jang_dataset "D:\Projects\UntanglingProject\UntanglingProject\Demos\Benchmark\Dataset_distributed\Dataset_distributed"
```
- Record convergence %, EF reduction %, time. Compare AGENTS.md baseline (94.9%).
- Report: `output/BatchComparison/jang/batch_summary.md` + per-case
  `output/BatchComparison/jang/PRP/<case>/debug_info_*.json`.

### D4. Hop-threshold generalization (only if D shows density sensitivity)
- Replace fixed `PRP_locality_max_hop` with geodesic-relative
  (`tgt_geom <= factor * contour_geom_extent`) or `hit.dist`-relative band detection,
  so one param generalizes across the 59 meshes of varying density.

### Gate GD (final)
- Maximize converged cases over 59; no whole-mesh explosion in any frame; SMPL_028
  scene still EF→0; bear improved or converged.

---

## 7. Build / run / measure protocol

```powershell
# build (PowerShell: no &&; use ; or if ($?) { })
cmake --build build -j

# single long contour (canonical) + dump
d:/Projects/Untangling25/.venv/Scripts/python.exe d:/Projects/Untangling25/PythonBindings/tests/test_untangling_jang2026.py --headless --advance_frames 1 --PRP_process_single_contour 1 --untangling_process_contour 1 --prp_dump_hits 1

# analyze a specific combo CSV (coverage + tgt_hop band/leak split)
d:/Projects/Untangling25/.venv/Scripts/python.exe d:/Projects/Untangling25/ResearchLog/analyze_prp_hits.py --csv ResearchLog/Tmp/prp_hits_f0_c1_cmb9.csv

# bear single
d:/Projects/Untangling25/.venv/Scripts/python.exe d:/Projects/Untangling25/PythonBindings/tests/test_batch_method_comparison_enhanced.py --dataset_mode jang --methods PRP --begin 1 --end 1 --no_export_per_frame --jang_dataset "D:\Projects\UntanglingProject\UntanglingProject\Demos\Benchmark\Dataset_distributed\Dataset_distributed"

# full 59 (acceptance)
d:/Projects/Untangling25/.venv/Scripts/python.exe d:/Projects/Untangling25/PythonBindings/tests/test_batch_method_comparison_enhanced.py --dataset_mode jang --methods PRP --begin 1 --end 59 --no_export_per_frame --jang_dataset "D:\Projects\UntanglingProject\UntanglingProject\Demos\Benchmark\Dataset_distributed\Dataset_distributed"
```

CSV path: `ResearchLog/Tmp/prp_hits_f{frame}_c{contour}_cmb{combo}.csv`.
Per-hit columns: hit_idx,hit_type,combo_idx,src_mesh_idx,src_id,dst_id,dist,
penetration_depth,q,is_boundary_adj,component_id,valid,src_hop,tgt_hop,direction_{xyz}.

Success criteria:
- per contour: valid hits ∈ [1000,3000] AND coverage ≥ 0.5 (aim 0.8).
- per case: final EF == 0.
- overall: max converged of 59, no explosions, SMPL_028 still EF→0.

---

## 8. Hard-won lessons (do NOT repeat)

1. "Hit count < 4000" is NOT success. A 2-hit cluster passes and is wrong. The real
   criterion is **contour coverage** (≥0.5, aim 0.8). Always measure coverage.
2. Size/vertex-fraction rejection is the WRONG axis — it kills the legitimate band.
   Coverage + target-hop locality is the right axis.
3. The legitimate band is FUSED inside the big boundary-attached component, not a
   separate neighbor. Keep the max-coverage component, then THIN it; don't reject it.
4. The leak here is NOT far-field (max tgt_hop=11). P0-2/P0-3 already stopped
   whole-mesh spread. Residual = thick band → thin by tgt_hop.
5. q_median≈0.63 → not projection-degenerate. Don't build the projection-aware
   graph (old P1) or atlas (old P3) for this; they are not the bottleneck.
6. Single direction need only reach ~0.8 coverage (>0.5), NOT 1.0. Full coverage
   needs multiple directions; we only care about the single main direction.
7. Whole-scene EF→0 can mask a wrong per-contour response (other contours + CCD
   carry it). Verify per-contour coverage, not just scene EF.
8. `untangling_response_depth`: test_untangling_jang2026.py overrides to 5e-3;
   batch `_apply_common_config` uses 1.0; `untangling_config.py` uses 1.0.
9. Scene-id mapping: batch uses `--begin/--end` over alphabetical
   `Dataset_distributed/Dataset_distributed/{animal,human,misc}/*.obj` (1-based;
   1=animal/bear0). It does NOT use `--scene_id`. test_untangling_jang2026.py has
   its own hardcoded knot/misc list + hardcoded `obj_dir` (currently SMPL_028).
10. PowerShell: no `&&`. Use `;` or `if ($?) { }`. Build: `cmake --build build -j`.

---

## 9. Results log

| Date | Phase | contour-1 valid hits | coverage | whole-mesh? | notes |
|---|---|---|---|---|---|
| 2026-06-26 | I baseline | 54591 | n/a | YES | topological flood, q_median 0.632 |
| 2026-06-26 | P0 + size-rejection (WRONG) | 2 | ~0 | NO | splinter; false "success"; dress scene EF→0 only because other contours carried it |
| 2026-06-26 | P0 batch bear (scene 1) | n/a | n/a | NO (capped) | EF oscillates 800-900 in 50 frames |
| 2026-06-27 | A instrumentation + Gate GA | (current code still picks 1) | comp0 high; leak max tgt_hop=11 | NO | confirmed: thick band, not far leak; need coverage+locality (Phase B) |
| 2026-06-27 | B/C coverage+locality | 3094 | 0.964 | NO | hop=1; single-contour gate slightly above hit target but full SMPL_028 reaches EF=0 at frame 21 |
| 2026-06-27 | D slice 1-10 | n/a | n/a | NO | 9/10 converged; bear0 PASS@7, deer3 FAIL ef 340->383 at hop=1; deer3 hop=2 improves to 346 but still fails; hop=3 worse |
| | D full 1-59 | | | | not yet implemented |

---

## 10. Sequencing
1. (DONE) Phase A instrumentation — coverage + tgt_hop columns; Gate GA passed.
2. (DONE) Phase B: scene params + replace selection block (thinned-coverage + tgt_hop thin) +
   disable size rejection in active path + calibrate hop (Gate GB).
3. (DONE) Phase C: coverage-gated comparator (Gate GC).
4. Phase D: jang2026 + batch 1 + batch 1–59 (Gate GD = acceptance).
5. Phase D4 only if density sensitivity appears: geodesic/dist-relative locality.

No-fallback rule: invalid hypotheses are skipped, never silently rerouted.
Commit Phase A instrumentation before starting Phase B so it is not lost on compaction.
