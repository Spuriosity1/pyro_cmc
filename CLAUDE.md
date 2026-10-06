# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build

```bash
meson setup build          # first time only
meson compile -C build     # rebuild
```

Build type is `release` with `warning_level=3` and `cpp_std=c++2a`. Dependencies: `lattice_lib ≥ 2.2.0`, `fftw3`, `hdf5`, `nlohmann_json`, `eigen3` (for `ltgs` only).

To run the validation test utility:
```bash
./build/test/spiral_byhand -o <out_dir> L <int> Q <qx> <qy> <qz>
```

## Executables

| Binary | Source | Purpose |
|--------|--------|---------|
| `build/anneal` | `src/anneal.cpp` | Simulated annealing + SSF sampling at multiple `--T_sample` temperatures |
| `build/fieldcool` | `src/fieldcool.cpp` | Field-cooling variant (same CLI structure) |
| `build/ltgs` | `src/luttinger_tisza.cpp` | Luttinger-Tisza classical ground-state solver (Eigen, no MC) |
| `build/test/spiral_byhand` | `test/spiral_byhand.cpp` | Constructs a hand-crafted spiral and computes its SSF for analytic comparison |

## Architecture

### Core classes (`include/`, `src/MC.cpp`)

**`CMC::MC_runner`** (`include/MC.hpp`, `src/MC.cpp`) — owns the lattice reference, coupling specs, PRNG, and runs Metropolis sweeps. Couplings are registered via `define_coupling(name, rel_vecs, J_matrix)` then `setup_lattice()` pre-links neighbor pointers into each `HeisenbergSpin::bond_sets`.

**`HeisenbergSpin`** — lattice site type satisfying the `lattice_lib` `GeometricObject` concept. Holds `ipos_t ipos`, `int pyro_sl` (0–3), `vec3<double> S`, and `std::vector<NeighbourSpins> bond_sets` (one entry per coupling shell).

**`abstract_manager`** (`include/abstract_manager.hpp`) — base for temperature-resolved accumulators. Subclasses override `on_new_temp()` to push per-temperature storage. Callers use `new_T(T)` (append a new slot) or `set_T(T)` (find or append), then call `write_group(file_id, path)`.

- **`ssf_manager`** (`include/ssf_manager.hpp`) — FFT-based static structure factor. Accepts 2-char correlator names (`"xx"`, `"yy"`, `"zz"`, `"xy"`, …). Stores raw per-cell DFT correlators *without* sublattice phases; phases are applied in post-processing.
- **`energy_manager`** (`include/energy_manager.hpp`) — accumulates `E` and `E²` per temperature for heat capacity.

### Geometry (`include/pyrochlore_geometry.hpp`)

Integer-coordinate pyrochlore lattice. Conventional cubic cell has side 8 (in integer units). Sublattice layout: 4 FCC sites × 4 tetrahedral vertices = 16 sublattices per cubic cell. `pyro_sl = latlib_sl % 4`.

Bond displacement tables are `static const std::vector<std::vector<ipos_t>>` keyed by pyrochlore sublattice: `pyrochlore::nn1_dist`, `nn2_dist`, `nn3a_dist`, `nn3b_dist`, and per-pair tables `nn1_pair_MN` (used for the XXZ J1 coupling).

`PyroCubicCell()` and `PrimitiveCell()` return `UnitCellSpecifier<HeisenbergSpin>` for use with `lattice_lib`'s `build_supercell`.

### Shared CLI (`include/cli_bits.hpp`)

`declare_LJ123(prog)` — registers `L`, `--J1/J2/J3`, `--Qz`, `--Jzz`, `-B` on an `argparse::ArgumentParser`.

`build_pyro_lat(prog)` / `build_J1J2J3_h(prog, lat, seed)` — construct the supercell and a fully wired-up `MC_runner`.

**`--Qz` and `--J3` are mutually exclusive.** Specifying `--Qz` computes a J3 that minimises the spiral energy at that wavevector (in units of 2π/a\_cubic).

`--Jzz` controls local-frame XXZ anisotropy (J₁ coupling only); `Jzz=0` is isotropic Heisenberg. The six nn1 sublattice-pair couplings are registered separately.

`generate_T_profile(T_hot, T_cold, T_sample, n_steps)` returns a log-spaced temperature schedule that guarantees the `--T_sample` values appear exactly as sampling stops.

### Output format (HDF5)

Files written to `<output_dir>/<prefix>_L=…_J1=…_seed=…_Tc=….out.h5`:
- `/geometry/` — `recip_vectors[3,3]`, `index_cell[3,3]`
- `/energy/` — `T_list`, `E`, `E2`, `n_samples`
- `/ssf/` — `static_corr[n_corr, n_T, n_k, n_sl, n_sl, 2]`, `corr_lookup`, `T_list`, `n_samples`, `sl_positions`, attributes `n_spins` and `k_dims[3]`

## Postprocessing

Python scripts require `numpy`, `h5py`, `matplotlib`.

```bash
python3 scripts/plot_ssf.py <file.h5> [--t-index N] [--slice-axis 0|1|2] [--log] [-o out.png]
python3 scripts/plot_q_dep.py ...
python3 scripts/plot_QaB.py ...
python3 scripts/acc_runs.py ...   # accumulate multiple runs
python3 scripts/ltgs_drive.py ... # drive the ltgs binary
```

`plot_ssf.py` applies the sublattice phase correction `w[k,μ,ν] = exp(2πi Σⱼ Kⱼ(rμ−rν)ⱼ/k_dims_j)` and fftshifts so Γ appears at centre. Shared HDF5 utilities live in `scripts/plot_common.py`.
