# PRP Pipeline in `host_resolve_intersections_PRP`

This document explains how the current CPU PRP implementation turns discrete EF penetration pairs (`host_collision_data->narrow_phase_list_ef`) into final corrective VF/EE collision templates used by the solver.

The current CPU implementation is no longer organized around the old per-contour helpers named `build_ray_cast_from_contour`, `perform_raycast_from_direction`, or `build_ray_cast_from_direction`. The standard path now builds one `PrpStandardContourRuntime` per active contour, then evaluates contours in batched combo groups through `evaluate_current_runtime_directions_template(...)` and `evaluate_batched_contours_of_single_combo(...)`.

## Scope

Implementation files:
- `Solver/CollisionDetector/intersection_resolver2.cpp`
- `Solver/CollisionDetector/intersection_resolver2.h`
- `Solver/CollisionDetector/intersection_resolver_helper.cpp/.h`

Current focus:
- EF pairs -> contour construction
- runtime construction in `build_standard_runtime`
- signed boundary-normal axis candidate culling
- direction seed generation in `get_raycasting_direction`
- adaptive k-ring candidate expansion
- batched CPU raycast + topology cluster culling
- standard branch and direction-optimization branch
- final hit -> VF/EE response pair conversion

Not current in the CPU standard path:
- `PRP_resolve_src_dst_conflict` is not executed. The scene flag still exists, but the current standard batched path does not call a src/dst conflict-resolution pass.
- The old progressive raycast branch and old per-contour raycast helper names are stale references for this code path.

## 1. Top-level Flow

`host_resolve_intersections_PRP(...)` performs these main stages.

1. Read EF penetration input.
- `num_pairs = host_collision_data->narrow_phase_collision_count[1]`
- EF list: `narrow_phase_list_ef`
- EF indices: `narrow_phase_list_ef_indices`

2. Build intersection contours from EF adjacency.
- Calls `intersection_contour_construction_from_ext_adjacent(host_untangling_data, host_collision_data, false)`.
- Outputs include:
  - `intersection_contours`
  - `ef_pair_contour_index`
  - `ef_pair_mesh_index`

3. Classify each contour type.
- `classify_contour_type(...)` tags contour topology for debug and downstream logic.

4. Precompute geometric normals from current positions.
- `face_normal`: oriented face normals.
- `edge_normal`: adjacent-face weighted average.
- `vert_normal`: adjacent-face weighted average.

5. Build per-vertex contour-boundary flags.
- `vert_in_boundary_flag[vid]` stores labels of form `2 * contour_idx + mesh_idx`.
- Batched cluster culling uses these flags to keep only hit connected components attached to the current contour boundary.

6. Build `PrpStandardContourRuntime` for each active contour.
- Active contours are those not skipped by `SceneParams::should_contour_skip(...)`.
- `build_standard_runtime(...)` fills object ids, full primitive candidates, boundary vertices, optional normal-culling, boundary distances, direction seeds, combo state, and adaptive k-ring initialization.

7. Evaluate runtimes in batched form.
- `run_batched_runtime_evaluation(...)` handles both the standard and direction-optimization branches.
- `evaluate_current_runtime_directions_template(...)` controls adaptive k-ring growth and combo iteration.
- `evaluate_batched_contours_of_single_combo(...)` performs the actual CPU ray dispatch, hit packing, adjacency construction, connected-component culling, anti-flip filtering, and metric evaluation.

8. Select and finalize the best hit set per contour.
- `select_runtime_best_results(...)` uses `find_best_result_in_combo_results(...)`, which uses `prp_is_better_response_hit_info(...)`.
- `prp_finalize_selected_best_hit_infos(...)` finalizes selected hit metadata and debug exports.

9. Convert selected hits to solver templates.
- `make_response_pair_from_raycasting_info(...)` converts selected VF/FV/EE hits into `CollisionPairTemplate` entries.

10. Assemble correction into the solver system.
- `host_apply_untangling_constraint(...)` injects the final templates into `sa_cgB` and `sa_cgA_diag`.

## 2. From EF Pairs to Contours

Input:
- `narrow_phase_list_ef`
- EF adjacency information in `host_untangling_data`

Key operations:

1. Build mesh-flip weights on EF adjacency edges.
- Adjacent EF pairs carry a flip bit describing whether traversal crosses to the opposite mesh side.

2. Flood-fill the EF graph into contours.
- Uses `contour_flood_filling_template(...)` with mesh-flip constraints.

3. Write contour membership back per EF pair.
- `ef_pair_contour_index[pair_idx] = contour_idx`.

Practical meaning: discrete EF penetrations are grouped into contour-level penetration components before raycasting.

## 3. Runtime Construction

For each active contour, `build_standard_runtime(...)` constructs a `PrpStandardContourRuntime`.

### 3.1 Object ids and full primitive candidates

The first EF pair identifies the two object/side ids:
- edge side -> `object_ids[mesh_idx]`
- face side -> `object_ids[mesh_idx ^ 1]`

For each side, full candidate vectors are initialized from the object's primitive ranges:
- `mesh_candidate_verts[mesh_idx]`
- `mesh_candidate_edges[mesh_idx]`
- `mesh_candidate_faces[mesh_idx]`
- plus the corresponding dense range fields.

These vectors form the contour candidate universe. Adaptive k-ring later derives `partial_candidate_*` from this universe.

### 3.2 Contour geometry and boundary vertices

The runtime stores:
- `contour_points_3D` from `ef_pair_pos_3D`
- `contour_points_weight` from `ef_pair_length`
- `boundary_verts[0/1]` from the EF edge side and face side vertices

Boundary vertices are the anchor set for:
- boundary-normal candidate culling
- BFS hop distance
- geodesic distance
- self-collision flip/anti-flip checks
- contour-connected cluster culling

### 3.3 Signed Boundary-Normal Axis Candidate Culling

When `SceneParams::PRP_use_oriented_raycast_culling` is true, `build_standard_runtime(...)` calls:

```cpp
prp_filter_standard_candidates_by_boundary_normals(...)
```

This is a pre-raycast candidate-domain filter. It does not change the objective or response construction; it only removes primitives whose local normal orientation is inconsistent with the contour boundary's dominant normal mode.

#### Axis construction per side

For each mesh side `mesh_idx`, `prp_compute_boundary_normal_axis(...)` uses `runtime.boundary_verts[mesh_idx]` and `vert_normal`:

1. Iterate boundary vertices.
- Skip out-of-range vertices.
- Skip invalid normals.
- Normalize each valid vertex normal `n`.
- Weight it by `sa_rest_vert_area[vid]`, clamped to at least `1e-8`.

2. Accumulate a signed reference and an unsigned normal moment.

```cpp
mean_normal   += w * n;
normal_moment += w * outer_product(n, n);
```

3. Pick the principal normal axis.
- `compute_all_eigenvectors(normal_moment)` returns eigenvectors sorted by ascending eigenvalue.
- The dominant normal axis is eigenvector `[2]`, the largest-eigenvalue direction of the normal moment matrix.
- This is the main normal PCA direction.

4. Assign the sign.
- The PCA moment `outer_product(n, n)` is sign-invariant, so the eigenvector axis is initially unsigned.
- The sign reference is `mean_normal` when it is valid; otherwise it falls back to the first valid boundary normal.
- If `dot(axis, orient_reference) < 0`, the axis is flipped.

5. Fail open on degeneracy.
- If there are no valid boundary normals, zero total weight, or an invalid axis, no normal culling is applied for that side.

So the selected "signed principal normal axis" is:

```text
axis = dominant eigenvector of sum(w * n * n^T), signed to agree with sum(w * n)
```

The stored `mean_alignment` is currently diagnostic-only; it is not used as a threshold.

#### Primitive filtering rule

The threshold is:

```cpp
threshold = clamp(PRP_cull_vertical_hits_cos_threshold, -1, 1)
```

For each side, candidates are filtered independently.

- Vertex candidates use `vert_normal[vid]`.
- Edge candidates use `edge_normal[eid]`; if invalid, they fall back to the sum of endpoint vertex normals.
- Face candidates use `face_normal[fid]`; if invalid, they fall back to the sum of face-vertex normals.

A primitive passes when:

```cpp
dot(normalize(primitive_normal), signed_axis) >= threshold
```

Invalid primitive normals pass through. If filtering a candidate class would keep zero ids, that class is left unchanged. If every id passes, the vector is also left unchanged. This makes the filter fail-open rather than deleting an entire primitive class.

Because the rule uses a signed dot product, not `abs(dot)`, it suppresses opposite-facing surface patches instead of treating the normal axis as purely unoriented.

### 3.4 Boundary distances

After optional normal culling, `calculate_hop_dist_and_geom_dist_in_mesh(...)` computes dense boundary distance fields:
- `vert_to_boundary_hop_dist[side]`
- `vert_to_boundary_geom_dist[side]`
- `vert_nearest_boundary_vert[side]`
- min/max contour-to-opposite-boundary stats

For sparse/adaptive GPU paths there are sparse equivalents, but the current CPU standard path uses these dense vectors.

### 3.5 Direction seeds and combo state

`get_raycasting_direction(...)` fills `runtime.contour_dirs`.

Then the runtime initializes:
- `active_combos`, respecting `should_combo_skip(...)`
- per-combo result buffers
- boundary vertex counts
- total candidate count
- adaptive k-ring initial levels: `init_k_src`, `init_k_dst`, `curr_k_src`, `curr_k_dst`

## 4. Direction Generation

`get_raycasting_direction(...)` selects:
- `InputDirection` when `PRP_use_input_dir` is true.
- otherwise `EmbedMeshDirection`, unless other scene options enable the expanded global/local direction set.

### 4.1 Default `EmbedMeshDirection`

The default direction is a minimum-variance axis of the embedded contour neighbourhood.

1. Choose input vertices.
- If direction optimization is enabled: use candidate vertices.
- Otherwise: use boundary vertices.

2. Compute weighted contour mean from `contour_points_3D` and `contour_points_weight`.

3. Build scatter matrix over both sides.

```cpp
centered = x[vid] - mean;
Sw += weight * outer_product(centered, centered);
```

4. Use the smallest-eigenvalue eigenvector.
- `compute_inner_lda_direction_template(Sw)` returns the minimum-variance direction.

5. Add both signs.
- Standard two-combo mode evaluates `+dir` and `-dir`.

### 4.2 Expanded direction sets

Current code can also produce larger direction sets when scene flags request it:
- fixed 16-direction set around a local frame
- Fisher LDA direction
- mesh-center delta direction
- proxy contour normal direction
- cone perturbations
- low-discrepancy samples
- multi-eigenvector sampling

The runtime stores all produced directions as combo indices and evaluates only non-skipped combos.

## 5. Adaptive K-ring Evaluation

`evaluate_current_runtime_directions_template(...)` is the shared driver for standard evaluation, initial direction-optimization screening, trial evaluations, and final optimization evaluation.

When `PRP_use_adaptive_kring` is true:

1. Reset per-runtime sparse/eval state and set `curr_k_src/dst = init_k_src/dst`.
2. While contours remain unfinished:
- Build `partial_candidate_verts/edges/faces` for each remaining contour via `build_standard_candidates_at_level(...)`.
- Evaluate each active, unconverged combo at the current k level.
- Mark a combo converged when the retained hit set no longer touches the current frontier.
- If early exit is enabled, prune evaluated combos whose displacement cost is already no better than the best converged valid combo.
- Grow `curr_k_src/dst` by `PRP_adaptive_kring_growth` until all active combos converge or the safety loop limit is hit.

When `PRP_use_adaptive_kring` is false:
- Each active combo is evaluated once on the current full candidate universe.

### 5.1 Candidate construction at k

`build_standard_candidates_at_level(runtime, k_src, k_dst, host_mesh_data)` filters the runtime's `mesh_candidate_*` universe by BFS hop distance:
- vertices pass when `dist[vid] <= k`
- edges pass when both endpoints satisfy the hop bound
- faces pass when all three vertices satisfy the hop bound

Because normal culling edits `mesh_candidate_*` before this stage, adaptive k-ring grows inside the normal-filtered candidate universe.

### 5.2 Frontier convergence

After a combo is evaluated, `compute_hit_min_max_frontier_hop(...)` records frontier metrics. The current CPU batched path marks:

```cpp
frontier_converged =
    max_frontier_hop_src < evaluated_kring_src &&
    max_frontier_hop_dst < evaluated_kring_dst;
```

`prp_has_valid_hit_info(...)` requires both:
- non-empty hit list
- `frontier_converged == true`

This validity rule is used by best-result selection and early-exit pruning.

## 6. Batched CPU Raycast and Culling

`evaluate_batched_contours_of_single_combo(...)` evaluates one combo index across a batch of contours.

### 6.1 Request preparation

For every contour slot in the batch, the function chooses either partial adaptive candidates or full mesh candidates.

It emits batched requests grouped by cached BVH:
- VF: source vertices from side 0 -> target faces on side 1.
- EE: source edges from side 0 -> target edges on side 1.
- FV: destination vertices from side 1 -> source faces on side 0, implemented as reverse VF queries against `reverse_face_bvh`.

The request path also computes:
- dense primitive spans for BVH traversal
- exact target candidate id filters
- target-side hop-distance filters for adaptive k-ring
- EE per-request `ray_max_dist`
- EE target projection lower-bound pruning

### 6.2 Raw ray generation

The CPU path calls cached batched ray traversal:
- `ray_cast_from_verts_batched(...)` for VF
- `ray_cast_from_edges_batched(...)` for EE
- `ray_cast_from_verts_batched(...)` for reverse FV

Raw hits are grouped back by local contour slot, then packed into one dense hit array for adjacency/culling.

### 6.3 Hit adjacency graph

For each contour slot, the packed hits populate primitive-to-hit buckets:
- `vert_contains_pairs`
- `edge_contains_pairs`
- `face_contains_pairs`

`find_adjacent_hits(...)` builds `pair_adj_pairs`. The adjacency is type-aware for VF, FV, and EE hits.

The scratch buckets are shared across combo evaluations and cleared by touched primitive rows, avoiding repeated full allocation of `num_primitives` buckets.

### 6.4 Contour-connected cluster culling

The current CPU path keeps all connected hit components that touch the current contour boundary.

1. For each hit, compute whether it is adjacent to the current contour boundary using `is_hit_adj_contour(...)` and `vert_in_boundary_flag`.
2. Flood-fill the hit adjacency graph per contour slot.
3. Keep every connected component that has at least one boundary-adjacent hit.
4. If `PRP_not_cull_rays` is true, keep all hits once any boundary-adjacent component exists.

This cluster culling remains semantic. Hop-distance predicates alone do not replace it because valid components are propagated transitively through the hit adjacency graph.

## 7. Combo Scoring and Best-result Selection

`prp_is_better_response_hit_info(a, b)` is the canonical comparator.

Validity first:
- invalid if `hit_list` is empty
- invalid if `frontier_converged` is false

For two valid results, the ordering is:

1. finite `penetration_objective` preferred
2. smaller `penetration_objective`
3. smaller `penetration_depth`
4. smaller `displacement_cost`
5. smaller `hit_count`
6. smaller `combo_idx`

This replaced older descriptions that treated penetration volume or displacement cost as the sole primary selector.

## 8. Standard Branch (`PRP_use_direction_optimization == false`)

`run_batched_runtime_evaluation(...)` does:

1. Evaluate current runtime directions.
- Calls `evaluate_current_runtime_directions(...)` with stage `Eval`.
- Uses adaptive k-ring if enabled.
- Evaluates one combo index across all active contours in batched form.

2. Select best combo per contour.
- `select_runtime_best_results(...)` picks the best valid combo using the canonical comparator.

3. Finalize selected best hit infos.
- `prp_finalize_selected_best_hit_infos(...)` prepares selected results for response construction and debug export.

There is no per-contour `perform_raycast_from_direction(...)` call in the current standard path.

## 9. Direction-optimization Branch

When `PRP_use_direction_optimization` is true, the current branch is also batched.

1. Initial evaluation.
- `evaluate_current_runtime_directions(...)` evaluates the initial direction combos, optionally with early exit.

2. Combo selection for optimization.
- `prp_select_optimization_combos(...)` marks valid combos for optimization.
- `PRP_optimize_select_combos_count` can keep only the best displacement-cost combos among valid ones.

3. Iterative direction update.
- For each selected combo, the code computes a Newton or IRLS/trust-region direction update from the accepted hit list.
- It temporarily writes the trial direction into `runtime.contour_dirs[combo_idx]`, evaluates that trial through the same batched direction evaluator, then accepts it only if `prp_is_better_response_hit_info(...)` improves the result.
- Trust radius is expanded or shrunk according to the predicted-vs-actual reduction ratio when trust-region mode is enabled.

4. Final evaluation and selection.
- The current directions are reevaluated in stage `Opt`.
- Best results are selected and finalized through the same code used by the standard branch.

## 10. From Best Hits to Solver VF/EE Templates

After a contour has selected `PRPMinContourHitInfo`:

1. `make_response_pair_from_raycasting_info(...)` converts selected hits into `CollisionPairTemplate` records.

Conversion rules:
- VF hit -> VF template.
- FV hit -> VF-layout template with index permutation and normal flip.
- EE hit -> EE template.

2. Template parameters include response depth and stiffness.
- `penetration_depth = min(min_hit_info.penetration_depth, untangling_response_depth)` when max-depth mode is used.
- The response uses the selected hit geometry, not the original EF pair directly.

3. `host_apply_untangling_constraint(...)` injects template contributions.
- Reads indices, weights, normal, `k1`, and `k2`.
- Computes gradient and rank-one Hessian contribution.
- Accumulates into:
  - `sa_cgB`
  - `sa_cgA_diag`

## 11. Compact Current CPU Mapping

1. EF narrow-phase pairs provide discrete penetration samples.
2. EF adjacency flood-fill builds contour components.
3. Each active contour builds a `PrpStandardContourRuntime`.
4. Full primitive candidates are optionally pre-filtered by signed boundary-normal axes.
5. Boundary hop/geodesic fields and direction combos are initialized.
6. Adaptive k-ring builds partial candidates from the normal-filtered universe.
7. Batched CPU raycasting generates VF/EE/FV raw hits per combo.
8. Hit adjacency + contour-connected component culling keeps boundary-attached hit regions.
9. Self-collision flip/anti-flip filtering removes geometrically invalid hits.
10. Penetration metrics and frontier convergence are computed.
11. The canonical comparator selects the best valid combo/hit set.
12. Selected hits become VF/EE response templates.
13. Templates are assembled into the Newton system.

## 12. GPU Note

The GPU PRP path mirrors many of the same runtime concepts and adaptive semantics, but its ray queries and culling are dispatched as GPU kernels. It has its own batching, upload, and optional hardware-RT machinery in `intersection_resolver_gpu.cpp`.

This document is primarily about the current CPU path. CPU/GPU parity work should compare:
- candidate construction
- adaptive k-ring levels
- target filtering
- cluster culling
- self-collision flip/anti-flip semantics
- final best-result comparator
