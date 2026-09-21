<div align="center">

# Type-Free Global Intersection Analysis with Linear Displacement Fields

### **ACM SIGGRAPH Asia 2026**

**[Chengzhu He](https://chengzhuuwu.github.io/)** &nbsp;·&nbsp; **[Xudong Feng](https://rullec.github.io/)** &nbsp;·&nbsp; **[Anjun Chen](https://chen3110.github.io/)** &nbsp;·&nbsp; **[Dan Song](https://zengjianhao.github.io/DanSongGroup/)** &nbsp;·&nbsp; **[Shihui Guo](https://www.humanplus.xyz/)** &nbsp;·&nbsp; **[Kui Wu](https://kuiwuchn.github.io/)**


<p align="center">
  <a href="https://chengzhuuwu.github.io/publication/siggraph-asia-2026-untangling/"><img src="https://img.shields.io/badge/Project-Page-blue?style=flat-square&logo=googlechrome&logoColor=white" alt="Project Page"></a>
  <a href="https://chengzhuuwu.github.io/files/untangling_siggraph_asia_2026.pdf"><img src="https://img.shields.io/badge/Paper-PDF-red?style=flat-square&logo=adobeacrobatreader&logoColor=white" alt="Paper PDF"></a>
  <a href="https://chengzhuuwu.github.io/files/untangling_supplemental.pdf"><img src="https://img.shields.io/badge/Supplemental-Material-orange?style=flat-square" alt="Supplemental Material"></a>
  <a href="https://youtu.be/cs_592tRAZQ"><img src="https://img.shields.io/badge/YouTube-Video-red?style=flat-square&logo=youtube&logoColor=white" alt="Video"></a>
</p>

<p align="center">
  <a href="https://github.com/ChengzhuUwU/Untangling26/actions/workflows/build_cmake.yml"><img src="https://github.com/ChengzhuUwU/Untangling26/actions/workflows/build_cmake.yml/badge.svg" alt="Build Status"></a>
  <a href="https://github.com/ChengzhuUwU/Untangling26/actions/workflows/build_wheels.yml"><img src="https://github.com/ChengzhuUwU/Untangling26/actions/workflows/build_wheels.yml/badge.svg" alt="Wheels"></a>
</p>

<p align="center">
  <img src="Document/Resp.jpg" alt="Type-Free Untangling Teaser" width="100%">
</p>

</div>

---

## Overview

Resolving self-intersections and multi-body penetrations without relying on non-penetrating collision history remains a fundamental challenge in physical simulation. Classical global intersection analysis (**GIA**) relies on explicitly classifying intersection contours into topological taxonomies and constructing 2D separating regions—an approach that easily breaks down on open boundary cuts, self-intersecting figure-eights, or complex multi-sheet nesting. Conversely, local intersection contour minimization (**ICM**) avoids explicit topological classification, but its reliance on local gradient descent along intersection lines frequently leads to severe local-minimum trapping under deep penetrations.

This work introduces a **type-free global intersection analysis** framework. For each contour, given a separation direction $\mathbf{r}\in\mathbb{S}^2$ and its (filtered) hit response support $\mathcal{S}(\mathbf{r})$, we evaluate the separation cost using a mass-weighted $L_2$ displacement metric analogous to the inertial term $\sum_i m_i \|\Delta\mathbf{x}_i\|^2$:

$$
\begin{aligned}
D_{\mathcal{C}}(\mathbf{r})
&=\max_{h\in\mathcal{S}(\mathbf{r})}\delta_h(\mathbf{r})\\
J(\mathbf{r})
&= \sum_{h\in\mathcal{S}(\mathbf{r})} m_h\, D_{\mathcal{C}}(\mathbf{r})^2
 = \rho\, D_{\mathcal{C}}(\mathbf{r})^2 \sum_{h\in\mathcal{S}(\mathbf{r})} A_h
\end{aligned} 
$$

where $m_h = \rho\, A_h$ is the mass associated with hit $h$, computed from the area $A_h$ of its source and destination primitives and the surface density $\rho$, and $D_{\mathcal{C}}(\mathbf{r})$ is the maximum ray travel distance of this contour.
All hits in $\mathcal{S}(\mathbf{r})$ share the same contour depth $D_{\mathcal{C}}$; we treat the area density $\rho$ as a uniform constant and drop it. By minimizing $J(\mathbf{r})$, we select separation paths that reconcile geometric untangling with the physical solver's lowest-energy displacement modes.


Given discrete edge--face (EF) intersection pairs, we treat each intersection contour $\mathcal{C}$ as a topological seed for constructing a separation response. 
For each contour, our pipeline alternates between freezing the physical state $\mathbf{x}$ to find the optimal separation direction $\mathbf{r}^\star = \arg\min J(\mathbf{r})$, 
and freezing $\mathbf{r}^\star$ to advance $\mathbf{x}$ via the global IP solver:

1. **Initialize** directed candidates $\mathbf{r} \in \mathcal{D}_0$ from contour-local geometry;
2. **Cast** to find the raw hit set $\mathcal{H}_{\mathrm{raw}}(\mathbf{r})$ via vertex-face (VF), edge-edge (EE), and face-vertex (FV) intersection tests;
3. **Filter** raw hits via dual-sided product-complex cluster culling, retaining only the physical component $\mathcal{S}(\mathbf{r})$ attached to the detected contour;
4. **Refine** selected candidates using projected Newton or Cauchy steps on $\mathbb{S}^2$, re-evaluating the filtered response objective $J(\mathbf{r})$ at each trial;
5. **Assemble** the final optimized hit set into solver constraints $\{C_k\}$ to advance the physical state $\mathbf{x}$.

<p align="center">
  <img src="Document/pipeline.jpg" alt="Ray-casting and hit-selection pipeline: intersection contour, raw VF/EE/FV hits, two-sided hit components, contour-anchored hits, and the applied response." width="100%">
</p>

The entire pipeline is history-independent, requires no topological case distinctions or manual contour taxonomy, and is implemented on the GPU via [LuisaCompute](https://github.com/LuisaGroup/LuisaCompute) across CUDA, DirectX 12, Vulkan, and Metal backends.

> **注意 / Disclaimer:**
> **本方案并不提供严格的解相交保证：对于自相交而言，在穿透较为复杂的区域，我们的 Cluster Culling 流程可能无法筛选出有修正语义的 RayCasting Hits；对于刚体相交而言，我们的方案可能会卡在物体的镂空区域。如何进一步提升系统的适用性与鲁棒性，是值得深入研究的方向。**  
> *(Note: This framework does not provide a strict untangling guarantee: for self-intersections in complex penetration configurations, the Cluster Culling procedure may fail to isolate RayCasting hits with restorative semantics; for rigid-body intersections, the method may become trapped in hollow or concave regions. Improving the general applicability and robustness of the system is an open and promising research direction.)*


## Quick Start (Pre-built Wheels)

Pre-built Python wheels (built for **Python 3.13**) are available on the [GitHub Releases](https://github.com/ChengzhuUwU/Untangling26/releases/tag/v0.1) page, allowing you to run the simulations and interactive GUI **without needing a C++ compiler or CMake setup**.

We recommend using **[uv](https://docs.astral.sh/uv/)** as the environment manager: it is extremely fast and can **automatically download and isolate Python 3.13** even if you do not have Python 3.13 installed on your machine.

#### 1. Install `uv` (Recommended)

If you do not have `uv` installed, install it following the [official installation guide](https://docs.astral.sh/uv/getting-started/installation/):

- **Windows (PowerShell)**:
  ```powershell
  powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
  ```
- **Linux / macOS**:
  ```bash
  curl -LsSf https://astral.sh/uv/install.sh | sh
  ```
- *Alternative Package Managers*: You can also install `uv` via `pip install uv`, `winget install astral-sh.uv`, or `brew install uv`.

#### 2. Fetch the repository (for demo scripts and mesh assets)

```bash
git clone https://github.com/ChengzhuUwU/Untangling26.git
cd Untangling26
```

#### 3. Setup Python 3.13 environment

Using **`uv`** (automatically downloads and sets up Python 3.13):

```powershell
# Windows PowerShell
uv venv --python 3.13
.venv\Scripts\activate
uv pip install numpy trimesh triangle polyscope
```

```bash
# Linux / macOS
uv venv --python 3.13
source .venv/bin/activate
uv pip install numpy trimesh triangle polyscope
```

<details>
<summary><b>Alternative: Using Conda</b></summary>

```bash
conda create -n untangling python=3.13 -y
conda activate untangling
pip install numpy trimesh triangle polyscope
```
</details>

#### 4. Install the pre-built wheel

Install the wheel matching your platform directly from GitHub Releases into your active Python 3.13 environment:

- **Windows (x86_64)**:
  ```powershell
  uv pip install https://github.com/ChengzhuUwU/Untangling26/releases/download/v0.1/untangling26-0.1.0-cp313-cp313-win_amd64.whl
  ```

- **Linux (x86_64)**:
  ```bash
  uv pip install https://github.com/ChengzhuUwU/Untangling26/releases/download/v0.1/untangling26-0.1.0-cp313-cp313-manylinux_2_28_x86_64.whl
  ```

- **macOS (Apple Silicon arm64)**:
  ```bash
  uv pip install https://github.com/ChengzhuUwU/Untangling26/releases/download/v0.1/untangling26-0.1.0-cp313-cp313-macosx_15_0_arm64.whl
  ```

*(Note: If using Conda, replace `uv pip install` with `pip install`.)*

#### 5. Run an interactive demo

Launch the canonical unit demo with the Polyscope interactive GUI:

```bash
# Interactive Polyscope viewer
python PythonBindings/tests/demo_unit.py --scene_id 0

# Headless mode for automated benchmarks
python PythonBindings/tests/demo_unit.py --scene_id 0 --headless --advance_frames 1
```

---

## Building from Source

If you prefer building the Python module (`lcs_py`) locally from source or wish to develop custom solvers:

#### For Windows

```powershell
# 1. Fetch the repository
git clone https://github.com/ChengzhuUwU/Untangling26.git
cd Untangling26
git submodule update --init --recursive

# 2. Setup Python environment (Python 3.13 recommended)
uv venv --python 3.13
.venv\Scripts\activate
uv pip install numpy trimesh triangle polyscope

# 3. Configure (automatically detects active .venv, uv, or conda environment)
cmake -S . -B build -G Ninja `
  -D CMAKE_BUILD_TYPE=Release `
  -D LCS_BUILD_PYBINDINGS=ON

# 4. Build Python module
cmake --build build --target lcs_py -j 4

# 5. (Optional) Editable install
uv pip install -e . --no-build-isolation -C build-dir=build
```

#### For Linux & MacOS

```bash
# 1. Fetch the repository
git clone https://github.com/ChengzhuUwU/Untangling26.git
cd Untangling26
git submodule update --init --recursive

# 2. Setup Python environment (Python 3.13 recommended)
uv venv --python 3.13
source .venv/bin/activate
uv pip install numpy trimesh triangle polyscope

# 3. Configure (automatically detects active .venv, uv, or conda environment)
cmake -S . -B build -G Ninja \
  -D CMAKE_BUILD_TYPE=Release \
  -D LCS_BUILD_PYBINDINGS=ON

# 4. Build Python module
cmake --build build --target lcs_py -j 4

# 5. (Optional) Editable install
uv pip install -e . --no-build-isolation -C build-dir=build
```

#### Configurations

In Step `# 3. Configure`:

- **Build Tools**:
  - `CMake` (>= 3.22) is recommended: install from the [Release Page](https://github.com/Kitware/CMake/releases).
  - `Ninja` is recommended as the build generator: install from the [Release Page](https://github.com/ninja-build/ninja/releases).
  - `Clang/Clang++` (or MSVC / GCC): specify via `-D CMAKE_C_COMPILER=clang -D CMAKE_CXX_COMPILER=clang++`.
- **Python Environment**: When `-D LCS_BUILD_PYBINDINGS=ON` is enabled, CMake **automatically detects** your active virtual environment (`uv`, `conda`, `venv`, or `.venv` in the repository root) without needing manual paths. You can still explicitly specify an interpreter by passing `-D LCS_PYTHON_EXECUTABLE=/path/to/python`.
- **GPU Computing Backends**: LuisaCompute will automatically configure available backends. To explicitly specify a backend:
  - **Vulkan** (**Cross-platform**, Windows, Linux, and macOS): `-D LUISA_COMPUTE_ENABLE_VULKAN=ON`
  - **CUDA** (NVIDIA GPUs on Windows and Linux): `-D LUISA_COMPUTE_ENABLE_CUDA=ON`
  - **DirectX 12** (Windows): `-D LUISA_COMPUTE_ENABLE_DX=ON`
  - **Metal** (Apple Silicon macOS): `-D LUISA_COMPUTE_ENABLE_METAL=ON`

The compiled module is placed in `build/bin`. Benchmark scripts add this path to `sys.path` automatically.

## Demos and Benchmarks

Run all commands from the repository root after building `lcs_py` (or installing the pre-built wheel).

Ensure your virtual environment is activated:
- **Windows PowerShell**: `.venv\Scripts\activate` (or `conda activate <env>`)
- **Linux / macOS**: `source .venv/bin/activate` (or `conda activate <env>`)

Once activated, you can run all scripts with `python PythonBindings/tests/...`.

On the initial run, LuisaCompute JIT-compiles device kernels (cached under `build/bin/.cache`), so subsequent runs execute significantly faster. Output files are saved to `output/paper_cases/`. Specify `--backend cuda` for NVIDIA GPUs, `--backend metal` on macOS, or `--backend dx` / `--backend vk` for DirectX / Vulkan.

All five demo entry points use **Polyscope GUI by default**. **Remove `--headless` from a demo command to enable the interactive Polyscope GUI shown below.** In the viewer, use **Space / Advance Single Frame** to step and **Run / Start Simulation** to play; pause with **Pause / End Simulation**.

<p align="center">
  <img src="Document/ui1.jpg" alt="Polyscope GUI showing a canonical untangling mesh and interactive simulation controls." width="100%">
  <br><em>Polyscope GUI: inspect meshes and advance the simulation interactively.</em>
</p>

### Canonical Configurations (Wicke et al. 2006)

Evaluates the seven canonical intersecting boundary and closed configurations from Wicke et al. (BBII, BIBI, BLI, Closed, Cross, Eight, and LL). The 2026-09-18 dual-metric Option C validation reached zero reported edge–face (EF) intersections in 2–5 solver iterations on both CPU and GPU contour-evaluation paths. Add `--export_debug` when per-frame OBJ/NPZ files are needed:

<p align="center"><img src="Document/cases_canonical.jpg" alt="The seven canonical contour configurations: inputs on the top row, resolved states on the bottom row." width="95%"></p>

For each case:
```powershell
python PythonBindings/tests/demo_unit.py `
    --backend cuda --headless --advance_frames 100 `
    --use_subdivision --subdiv_levels 3 --scene_id 0
```

Or with batching command:

```powershell
0..6 | ForEach-Object {
  python PythonBindings/tests/demo_unit.py `
    --backend cuda --headless --advance_frames 300 `
    --use_subdivision --subdiv_levels 3 --scene_id $_
  if ($LASTEXITCODE -ne 0) { throw "Canonical unit $_ failed" }
}
```

### Synthetic Fold

A procedurally generated multi-layer folding benchmark exhibiting deep, nested self-intersections:

<p align="center"><img src="Document/cases_synthetic.jpg" alt="Synthetic fold: initial state, initial ray hits, and resolved state." width="85%"></p>

```powershell
python PythonBindings/tests/demo_synthetic.py `
  --backend cuda --headless --advance_frames 400 `
  --start_from_initial 1 --use_ccd_linesearch 0 `
  --untangling_response_depth 0.005 `
  --max_ef_pairs 10000
```

### Jang59 Benchmark

Animated previews of two groups of benchmark cases. Each group contains four animal, four human, and four miscellaneous meshes; red curves mark detected intersections.

<p align="center">
  <img src="Document/jang59_group1.gif" alt="Group 1: untangling four animal, four human, and four miscellaneous meshes." width="100%">
  <br><em></em>
</p>

<p align="center">
  <img src="Document/jang59_group2.gif" alt="Group 2: untangling another four animal, four human, and four miscellaneous meshes." width="100%">
  <br><em></em>
</p>

Evaluates against the [dataset](https://github.com/wonjongg/instant-mesh-intersection-repair) from Instant Self-Intersection Repair (ISIR, Jang et al. 2025) benchmark suite. Sample scenes from the official repository are bundled in the `external/instant-mesh-intersection-repair` submodule:

```powershell
git submodule update --init external/instant-mesh-intersection-repair
```

Validate case ordering and hashes without loading the solver, then run the gate:

```powershell
python PythonBindings/tests/test_batch_method_comparison_enhanced.py `
  --backend cuda --begin 1 --end 59 --dry_run
```

```powershell
python -u PythonBindings/tests/test_batch_method_comparison_enhanced.py `
  --backend cuda --headless --begin 1 --end 59 `
  --advance_frames 100 --timeout 300 --no_export_per_frame
```

The complete 59-case `Dataset_distributed` benchmark is subject to its original distribution terms; request it from the authors (wonjong@postech.ac.kr) and pass `--jang_dataset <directory>` to evaluate the full set. 

To inspect one Jang case, omit `--headless` and select, for example, `--begin 1 --end 1`. For multiple selected cases/methods, close a completed viewer to open the next one; closing before completion cancels the remaining runs and writes the partial report. `--timeout` applies only to headless workers, and `--dry_run` never opens a viewer.

### Static Repair from a Single OBJ

`test_load_from_obj.py` performs static self-intersection repair on any OBJ mesh: it loads the mesh, untangles it in quasi-static mode (no gravity, no floor), and writes the resolved surface as an OBJ, exiting with code 0 only on a clean collision report. The example below repairs the self-intersecting Klein bottle bundled with the ISIR submodule in five iterations (76 initial EF pairs to zero):

```powershell
python PythonBindings/tests/test_load_from_obj.py `
  --backend cuda --headless `
  --input_mesh external/instant-mesh-intersection-repair/data/misc/disc_kleinbottle.obj
```

### Thingi10K Rigid Solids

Tests deep volumetric penetrations between watertight rigid solids (`MaterialType::Rigid`) from the Thingi10K dataset. Enabling `config.PRP_solid_interior_adjacency = True` establishes virtual volumetric chords connecting opposing entry and exit ray hits across the interior volume, avoiding topological disconnection around thick geometry. This benchmark evaluates 8 representative watertight models (300–2,600 faces) across CPU and GPU backends, comparing pure surface adjacency against solid interior adjacency:

```powershell
python PythonBindings/tests/test_thingi10k_rigid_untangling.py `
  --backend cuda --headless --max_frames 35
```


Without `--headless`, Thingi10K previews one of the eight models, selected with `--case_index 1` through `8` (default: `1`). Choose the preview configuration with `--use_gpu 0|1` and `--solid_adj 0|1`; both default to `0`. For example:

```powershell
python PythonBindings/tests/test_thingi10k_rigid_untangling.py `
  --backend cuda --case_index 1 --solid_adj 1 --max_frames 35
```


## Acknowledgments

Special thanks to [Xudong](https://rullec.github.io/) for his guidance and development support from [LuisaCompute](https://github.com/LuisaGroup/LuisaCompute) community.

This project builds upon [LuisaComputeSimulator](https://github.com/ChengzhuUwU/LuisaComputeSimulator) and [LuisaCompute](https://github.com/LuisaGroup/LuisaCompute). 

Reference (on untangling):
- [Unreal Engine](https://github.com/EpicGames/UnrealEngine/blob/release/Engine/Source/Runtime/Experimental/Chaos/Private/Chaos/PBDTriangleMeshCollisions.cpp): Implementation on EF intersection detection, and untangling baseline: ICM and GIA
- [instant-mesh-intersection-repair](https://github.com/wonjongg/instant-mesh-intersection-repair): For mesh-intersection-repair benchmark

## License

The source code is distributed under the [Apache License 2.0](LICENSE).

If you have any questions, feel free to contact with `chengzhuhe@stu.xmu.edu.cn`
