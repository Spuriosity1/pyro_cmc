#!/usr/bin/env python3
"""Plot SSF intensity at (0,0,Qz) and symmetry-equivalent q-points vs a scan parameter."""

from plot_ssf import load_file, apply_phases, split_fixed_varying, compute_phase_factors
import argparse
import numpy as np
import h5py
import matplotlib.pyplot as plt
import os
import sys


def qz_to_idx(qz_frac, k_dim):
    return int(round(qz_frac * k_dim)) % k_dim


def load_corr2(path):
    """Load static_corr_2 and n_seeds from a merged HDF5 file.

    Returns (corr_2, n_seeds) or (None, None) if the datasets are absent.
    corr_2 shape: [n_corr, n_T, n_k, n_sl, n_sl] complex.
    """
    with h5py.File(path, "r") as f:
        ssf = f["/ssf"]
        if "static_corr_2" not in ssf:
            return None, None
        raw = ssf["static_corr_2"][:]
        corr_2 = raw[..., 0] + 1j * raw[..., 1]
        energy = f["/energy"]
        n_seeds = int(energy["n_seeds"][()]) if "n_seeds" in energy else None
    return corr_2, n_seeds


def contract_corr2(corr_2, corr_lookup, sl_positions, k_dims):
    """Phase-contract corr_2 without normalising by n_ssf.

    Returns dict label -> array [n_T, k0, k1, k2], real-valued.
    The raw contracted value is Σ_{mu,nu} Re(w * corr_2), which can be
    divided by (n_ssf^2 * n_spins^2) in post-processing to get <S^2> per seed.
    """
    w = compute_phase_factors(k_dims, sl_positions)
    contracted = np.real(np.einsum("kmn,ctkmn->ctk", w, corr_2))
    k0, k1, k2 = k_dims
    contracted = contracted.reshape(len(corr_lookup), -1, k0, k1, k2)
    return {label: contracted[i] for i, label in enumerate(corr_lookup)}


def cross_seed_se(W_q, intensity, n_seeds, n_ssf_t, n_spins):
    """Approximate standard error of the mean SSF from cross-seed second moment.

    W_q      : Re(Σ_{mu,nu} w * corr_2)[k_q], i.e. unscaled contracted second moment
    intensity : already-computed per-spin SSF mean = Re(Σ w * corr) / (n_ssf * n_spins)
    n_seeds  : number of seeds K
    n_ssf_t  : total sample count for this temperature (= K * n_per_seed)
    n_spins  : total spin count

    Approximation: cross-terms between different (mu,nu) pairs are dropped when
    computing Var[Σ_mu,nu w C_{mu,nu}] from the element-wise second moment.
    """
    if n_seeds < 2:
        return np.nan
    # <s^2>_approx per seed = W_q * K / (n_ssf_t^2 * n_spins^2)
    s2_approx = W_q * n_seeds / (n_ssf_t**2 * n_spins**2)
    var_between = (n_seeds / (n_seeds - 1)) * (s2_approx - intensity**2)
    # SE of the mean = sqrt(var_between / K)
    return np.sqrt(max(float(var_between), 0.0) / n_seeds)


def main():
    p = argparse.ArgumentParser(
        description="Plot SSF intensity at Bragg/symmetry-equivalent q-points vs a scan parameter."
    )
    p.add_argument("file", help="Path(s) to HDF5 file", nargs='+')
    p.add_argument("-x", "--x-axis", required=True, help="Tag to plot along the x-axis")
    p.add_argument("-s", "--series-axis", help="Tag to use as a series label")
    p.add_argument("-t", "--t-index", type=int, default=None,
                   help="Temperature index into the SSF array (default: last = coldest)")
    p.add_argument("--vmin", type=float, default=None)
    p.add_argument("--vmax", type=float, default=None)
    p.add_argument("-o", "--output", default=None,
                   help="Save figure to file instead of displaying")
    args = p.parse_args()

    files = args.file
    fixed, _, all_params = split_fixed_varying(files)

    if args.series_axis:
        series_vals = sorted(set(pm.get(args.series_axis, '?') for pm in all_params))
    else:
        series_vals = [None]

    def get_series(params):
        return params.get(args.series_axis, '?') if args.series_axis else None

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    ax_flat = [axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]]

    panel_labels = [
        r"$(0,0,Q_z)$  [Bragg]",
        r"$(Q_z,0,0)$",
        r"$(0,Q_z,0)$",
        r"$\Gamma = (0,0,0)$",
    ]
    for ax, title in zip(ax_flat, panel_labels):
        ax.set_title(title)
        ax.set_xlabel(args.x_axis)
        ax.set_ylabel(r"$S(\mathbf{q})$ / spin")

    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    series_color = {v: colors[i % len(colors)] for i, v in enumerate(series_vals)}

    series_data = {v: {'x': [], 'I': [[], [], [], []], 'SE': [[], [], [], []]}
                   for v in series_vals}

    for fpath, params in zip(files, all_params):
        x_str = params.get(args.x_axis)
        if x_str is None:
            print(f"Warning: '{args.x_axis}' not found in {os.path.basename(fpath)}, skipping",
                  file=sys.stderr)
            continue
        try:
            x_val = float(x_str)
        except ValueError:
            x_val = x_str

        qz_str = params.get('Qz')
        if qz_str is None:
            print(f"Warning: 'Qz' not found in {os.path.basename(fpath)}, skipping",
                  file=sys.stderr)
            continue
        qz = float(qz_str)

        (_, _, _, _, _, _,
         corr, corr_lookup, sl_positions, k_dims, n_spins, _, n_ssf) = load_file(fpath)

        n_T = corr.shape[1]
        t_idx = args.t_index if args.t_index is not None else n_T - 1
        if not (0 <= t_idx < n_T):
            sys.exit(f"--t-index {t_idx} out of range [0, {n_T - 1}]")

        S = apply_phases(corr, corr_lookup, sl_positions, k_dims, n_ssf)
        diag = [c for c in ("xx", "yy", "zz") if c in S]
        if not diag:
            print(f"Warning: no diagonal correlators in {os.path.basename(fpath)}, skipping",
                  file=sys.stderr)
            continue

        corr_2, n_seeds = load_corr2(fpath)
        if corr_2 is not None:
            S2 = contract_corr2(corr_2, corr_lookup, sl_positions, k_dims)
        else:
            S2 = None

        # Qz is in units of 2π/a_cubic; k_dims[i] = L for cubic supercell
        qi = qz_to_idx(qz, k_dims[0])

        q_indices = [
            (0,  0,  qi),   # (0, 0, Qz)
            (qi, 0,  0),    # (Qz, 0, 0)
            (0,  qi, 0),    # (0, Qz, 0)
            (0,  0,  0),    # Gamma
        ]

        ser = get_series(params)
        series_data[ser]['x'].append(x_val)
        for panel, (i0, i1, i2) in enumerate(q_indices):
            intensity = sum(S[c][t_idx, i0, i1, i2] for c in diag) / n_spins
            series_data[ser]['I'][panel].append(intensity)

            if S2 is not None and n_seeds is not None:
                W_q = sum(S2[c][t_idx, i0, i1, i2] for c in diag)
                se = cross_seed_se(W_q, intensity, n_seeds, n_ssf[t_idx], n_spins)
            else:
                se = np.nan
            series_data[ser]['SE'][panel].append(se)

    plotted_any = False
    for ser in series_vals:
        d = series_data[ser]
        if not d['x']:
            continue
        order = np.argsort(d['x'])
        x_sorted = np.array(d['x'])[order]
        label = str(ser) if ser is not None else None
        color = series_color[ser]
        for panel, ax in enumerate(ax_flat):
            I_sorted = np.array(d['I'][panel])[order]
            SE_sorted = np.array(d['SE'][panel])[order]
            if np.any(np.isfinite(SE_sorted)):
                ax.errorbar(x_sorted, I_sorted, yerr=SE_sorted,
                            fmt='o-', label=label, color=color, ms=4,
                            capsize=3, elinewidth=1)
            else:
                ax.plot(x_sorted, I_sorted, 'o-', label=label, color=color, ms=4)
        plotted_any = True

    if not plotted_any:
        sys.exit("No data to plot — check that --x-axis and Qz tags exist in the filenames.")

    if args.series_axis:
        ax_flat[0].legend(title=args.series_axis, fontsize=8)

    if args.vmin is not None or args.vmax is not None:
        for ax in ax_flat:
            ax.set_ylim(args.vmin, args.vmax)

    fixed_str = "  ".join(f"{k}={v}" for k, v in fixed.items())
    suptitle = r"$S(\mathbf{q})$ at symmetry-equivalent wavevectors"
    if fixed_str:
        suptitle += f"\n{fixed_str}"
    fig.suptitle(suptitle)

    plt.tight_layout()

    if args.output:
        fig.savefig(args.output, dpi=150, bbox_inches="tight")
        print(f"Saved to {args.output}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
