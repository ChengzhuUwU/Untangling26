# Reproduction Assets

This directory contains only inputs used by the public smoke tests and paper
case reproductions. Simulation output belongs under the repository-level
`output/` directory and is ignored by Git.

## Included inputs

| Case | Files | Purpose |
| --- | --- | --- |
| JSON smoke test | `Scenes/default_scene.json`, `InputMesh/bowl/torus3K.obj`, `InputMesh/square2K.obj` | Small self-contained cloth/torus scene. |
| Canonical units | `InputMesh/SelfIntersectionUnit/unit_*.obj` | Seven Wicke-style contour configurations used by `demo_unit.py`. Subdivision is performed at runtime. |
| Drop3 | `PaperCases/Drop3/drop3_clothes.obj`, `PaperCases/Drop3/state_frame_68.state` | Rest mesh and tangled initial state for the heavy Drop3 paper case. |

Drop3 input integrity:

| File | SHA-256 |
| --- | --- |
| `drop3_clothes.obj` | `AF7CDAA7074044BC4B91687ED2839E17B886C149F5AFC710C9233B460CA1448F` |
| `state_frame_68.state` | `2B5A431F0321B9F5372A502125FB00179C2AF9FB98E21828E1C8CFFFC706C789` |

The state file depends on the exact vertex ordering of `drop3_clothes.obj`;
do not process or reorder that mesh before loading the state.

## External benchmark

The Jang et al. dataset is third-party data and is not redistributed here.
Obtain `Dataset_distributed` from the authors' Instant Mesh Intersection Repair
project, then pass its path to the Jang runner with `--jang_dataset`. The runner
records the SHA-256 of every selected input and an aggregate manifest hash in
its report.
