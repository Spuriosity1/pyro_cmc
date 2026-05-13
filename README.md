# classical_MC_v2

Classical Monte Carlo simulated annealing for Heisenberg spin systems on the pyrochlore lattice, using [LatticeLab 2](https://github.com/Spuriosity1/latticelab) (`lattice_lib ≥ 2.2.0`).

Computes the static structure factor (SSF) S(q) via FFT and writes results to HDF5 for downstream analysis.

## Dependencies

| Library | Purpose |
|---------|---------|
| `lattice_lib ≥ 2.2.0` | Supercell geometry, periodic BCs, FFT infrastructure |
| `fftw3` | Underlying FFT engine |
| `hdf5` | Output file format (serial or parallel) |
| `nlohmann_json` | JSON (available for configuration) |
| `meson ≥ 1.5.0` | Build system |
| Python ≥ 3.8 + `numpy`, `h5py`, `matplotlib` | Postprocessing / plotting |

`XoshiroCpp` (Xoshiro128++ PRNG) is bundled as a header in `include/`.

## Build

```bash
meson setup build
meson compile -C build
```

Output: `build/anneal` (installed), `build/spiral_byhand` (test utility, not installed).

## Usage

### `anneal` — main executable

Runs simulated annealing from `T_hot` down to `T_cold` in `n_steps` logarithmically spaced steps, then accumulates SSF and energy statistics at each temperature.

```
anneal -o <output_dir> -s <hex_seed> L <int>
       --J1 <float> [--J2 <float>] [--J3 <float>]
       [--T_hot <float>] --T_cold <float>
       [--T_ref <float>] [--n_steps <int>]
       [--n_sweep <int>] [--n_burn_in <int>] [--n_sample <int>]
       [-B <bx> <by> <bz>] [--save_state]
```

**Arguments:**

| Flag | Default | Description |
|------|---------|-------------|
| `-o / --output_dir` | required | Directory for output files |
| `-s / --seed` | required | 64-bit hex RNG seed (e.g. `0xdeadbeef`) |
| `L` | required | Supercell linear dimension; total spins = 4L³ × 4 |
| `--J1` | required | Nearest-neighbour coupling |
| `--J2` | 0 | Second-neighbour coupling |
| `--J3` | 0 | Third-neighbour coupling |
| `--T_hot` | 10√(J1²+J2²+J3²) | Starting temperature |
| `--T_cold` | required | Final temperature |
| `--T_ref` | 1.0 | Reference temperature for Metropolis proposal width |
| `--n_steps` | 100 | Annealing temperature steps |
| `--n_sweep` | 16 | Metropolis sweeps per temperature |
| `--n_burn_in` | 16 | Burn-in sweeps at `T_hot` |
| `--n_sample` | 64 | Sampling sweeps at each temperature |
| `-B` | (0,0,0) | External magnetic field vector |
| `--save_state` | — | Also write final spin configuration to `.spins.h5` |

**Example:**
```bash
./build/anneal -o results/ -s 0xdeadbeef L 4 \
    --J1 1.0 --J2 0.0 --J3 0.0 \
    --T_cold 0.05 --n_steps 200 --n_sample 128
```

**Output file:** `results/L=4_J1=1.0_J2=0.0_J3=0.0_seed=0xdeadbeef_T_c=0.05.out.h5`

### `spiral_byhand` — validation utility

Constructs a hand-crafted spiral state at wavevector Q and computes its SSF without MC, for comparison against known analytic results.

```
spiral_byhand -o <output_dir> L <int> Q <qx> <qy> <qz>
```

## HDF5 output format

```
<params>.out.h5
├── /geometry/
│   ├── recip_vectors[3, 3]     reciprocal lattice basis (rows: b0, b1, b2)
│   └── index_cell[3, 3]        supercell index matrix (int64)
├── /energy/
│   ├── T_list[n_T]
│   ├── E[n_T]                  sum of energy samples per temperature
│   ├── E2[n_T]                 sum of energy² samples
│   └── n_samples[n_T]
└── /ssf/
    ├── static_corr[n_corr, n_T, n_k, n_sl, n_sl, 2]   last dim: [re, im]
    ├── corr_lookup[n_corr]     e.g. ["xx", "yy", "zz"]
    ├── T_list[n_T]
    ├── n_samples[n_T]
    ├── sl_positions[n_sl, 3]   sublattice coordinates (int64)
    ├── @n_spins                total spin count (attribute)
    └── @k_dims[3]              k-space grid dimensions (attribute)
```

`static_corr` stores raw per-cell DFT correlators **without** sublattice phases; these are applied in postprocessing (see `plot_ssf.py`).

## Postprocessing

### `scripts/plot_ssf.py`

Loads an HDF5 file, applies sublattice phase corrections, and plots 2D k-space slices of S(q).

```bash
python3 scripts/plot_ssf.py <file.h5> [options]
```

| Flag | Default | Description |
|------|---------|-------------|
| `-t / --t-index` | last (coldest) | Temperature index into SSF array |
| `--slice-axis` | 2 | k-axis to fix for the 2D slice (0/1/2) |
| `--slice-idx` | 0 | Index along `--slice-axis`; 0 = Γ plane |
| `--log` | — | Logarithmic colour scale |
| `--cmap` | inferno | Matplotlib colormap |
| `-o / --output` | — | Save to PNG instead of displaying |
| `--energy` | — | Also plot ⟨E⟩ and C_v vs T |

The script applies the sublattice phase correction:
```
w[k, μ, ν] = exp(2πi Σ_j K_j (r_μ − r_ν)_j / k_dims_j)
```
and fftshifts the result so the Γ point appears at the centre of the plot.

## Code structure

```
include/
  MC.hpp                  MC_runner class: spins, bonds, Metropolis update
  ssf_manager.hpp         SSF accumulator: FFTs spin components, writes /ssf
  energy_manager.hpp      Energy accumulator, writes /energy
  abstract_manager.hpp    Base class for temperature-resolved accumulators
  pyrochlore_geometry.hpp Lattice geometry: FCC sites, link directions, neighbours
  format_bits.hpp         Formatting utilities
  XoshiroCpp.hpp          Bundled Xoshiro128++ PRNG (header-only)
src/
  MC.cpp                  MC_runner implementation + HDF5 I/O
  anneal.cpp              Main: CLI parsing, cooling schedule, output
test/
  spiral_byhand.cpp       Reference spiral-state SSF for validation
scripts/
  plot_ssf.py             SSF visualisation
  plot_common.py          Shared HDF5 loading utilities
```

## Physics notes

- **Lattice:** Pyrochlore — 16 sublattices per 8×8×8 cubic conventional cell (4 FCC sites × 4 corner-sharing tetrahedron vertices).
- **Hamiltonian:** H = Σ_{⟨ij⟩_n} J_n S_i · S_j − B · Σ_i S_i (isotropic Heisenberg, up to third neighbours).
- **Metropolis proposal:** Gaussian kick in ℝ³ re-normalised to the unit sphere; width ∝ √(T/T_ref).
- **SSF:** S^{αβ}(q) = (1/N) Σ_{μν} e^{iq·(r_μ−r_ν)} ⟨S^α_μ(q) S^{β*}_ν(q)⟩, with q on a uniform grid set by `k_dims`.
