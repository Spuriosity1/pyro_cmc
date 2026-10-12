#!/usr/bin/env python3
"""Plot static structure factor from anneal HDF5 output."""

import argparse
import os
import re
import sys
import numpy as np
import h5py

from ssf_backend import load_file, normalize_ssf, parse_params, split_fixed_varying
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib as mpl 
mpl.rcParams["savefig.directory"] = os.getcwd()



def label_axes(ax, slice_axis=2):
    axes_l = ['yz', 'xz', 'xy']
    a1, a2 = tuple(axes_l[slice_axis])

    ax.set_xlabel(rf"$k_{a1}$")
    ax.set_ylabel(rf"$k_{a2}$")

def kgrid_xy(k_dims, recip, slice_axis=2):
    """Cartesian x,y coordinates for a 2D slice of the k-grid.

    slice_axis: which of 0,1,2 to fix (default 2 → hk0 plane).
    Returns x,y arrays of shape [na0, na1].
    """
    axes = [a for a in range(3) if a != slice_axis]
    a0, a1 = axes
    na0, na1 = k_dims[a0], k_dims[a1]

    h = np.arange(na0) - na0 // 2
    k = np.arange(na1) - na1 // 2
    hh, kk = np.meshgrid(h, k, indexing="ij")

    q = (hh[..., None] / na0 * recip[a0]
       + kk[..., None] / na1 * recip[a1])  # [na0, na1, 3]
    return q[..., a0], q[..., a1]


def plot_ssf(ax, file, args, title=""):
    (recip, index_cell,
     T_list, E, E2, n_E,
     corr, corr_lookup, sl_positions,
     k_dims, n_spins, ssf_T, n_ssf) = load_file(file)

    n_T = corr.shape[1]
    t_idx = args.t_index if args.t_index is not None else n_T - 1
    if not (0 <= t_idx < n_T):
        sys.exit(f"--t-index must be in [0, {n_T - 1}]")
    t_val = ssf_T[t_idx]

    S = normalize_ssf(corr, corr_lookup, k_dims, n_ssf)

    # Build derived observables from whatever components are present
    diag = [c for c in ("xx", "yy", "zz") if c in S]
    data_3d = sum(S[c] for c in diag)

    # sa = args.slice_axis
    sl = args.slice_idx


    ax[0].set_title(title, fontsize=6)

    for sa in (0,1,2):
        x, y = kgrid_xy(k_dims, recip, slice_axis=sa)

        # Build index tuple for the 2D slice
        idx = [slice(None), slice(None), slice(None)]
        idx[sa] = sl
        idx = tuple(idx)  # applied as data_3d[t_idx][idx]


        data = np.fft.fftshift(data_3d[t_idx][idx])
        data /= n_spins

        vmax = args.vmax
        if args.vmin:
            vmin = args.vmin
        else:
            vmin = max(data.min(), 1e-6 * vmax) if args.log else 0
        norm = (mcolors.LogNorm(vmin=vmin, vmax=vmax) if args.log
                else mcolors.Normalize(vmin=0, vmax=vmax))
        a0, a1 = [a for a in range(3) if a != sa]
        # For FCC: tile with the 4 nearest in-plane BCC shifts (±1,±1) so the
        # full FCC reciprocal lattice is visible beyond one conventional cubic BZ.
        shifts = ((0,0), (1,1), (1,-1), (-1,1), (-1,-1)) if args.fcc else ((0,0),)
        for p, q in shifts:
            xs = x + p * recip[a0][a0] + q * recip[a1][a0]
            ys = y + p * recip[a0][a1] + q * recip[a1][a1]
            c = ax[sa].pcolormesh(xs, ys, data, cmap=args.cmap, norm=norm, shading="auto")

        label_axes(ax[sa], slice_axis=sa)
        ax[sa].set_aspect("equal")
    return c



def plot_energy(args):
    _, _, T_list, E, E2, n_E, *_ = load_file(args.file)

    if not np.all(np.isfinite(n_E)):
        sys.exit("No /energy group in this file (minimise output?); "
                 "nothing to plot for energy/heat-capacity.")

    E_mean  = E / n_E
    E2_mean = E2 / n_E
    C = (E2_mean - E_mean**2) / T_list**2

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 6), sharex=True)
    ax1.plot(T_list, E_mean, "o-", ms=3)
    ax1.set_ylabel("⟨E⟩ / site")
    ax1.invert_xaxis()

    ax2.plot(T_list, C, "o-", ms=3, color="tab:orange")
    ax2.set_ylabel("C / site")
    ax2.set_xlabel("T")

    fig.suptitle(args.file)
    plt.tight_layout()

    if args.output:
        out = (args.output.replace(".png", "_energy.png")
               if args.output.endswith(".png") else args.output + "_energy.png")
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print(f"Saved to {out}")
    else:
        plt.show()


def main():
    p = argparse.ArgumentParser(
        description="Plot SSF (and optionally energy) from anneal HDF5 output.")
    p.add_argument("file", help="Path(s) to HDF5 file", nargs='+')
    p.add_argument("-t", "--t-index", type=int, default=None,
                   help="Temperature index into the SSF array (default: last = coldest)")
    # p.add_argument("--slice-axis", type=int, default=2, choices=[0, 1, 2],
    #                help="Which k-axis to fix for the 2D slice (default: 2 → hk0 plane)")
    p.add_argument("--slice-idx", type=int, default=0,
                   help="Index along --slice-axis to plot (default: 0)")
    p.add_argument("--log", action="store_true", help="Use logarithmic colour scale")
    p.add_argument("--cmap", default="inferno", help="Matplotlib colormap (default: inferno)")

    p.add_argument("--fcc", action="store_true",
                   help="Tile with BCC-type shifts to show full FCC reciprocal lattice")
    p.add_argument("--vmin", type=float)
    p.add_argument("--vmax", default=1, type=float)

    p.add_argument("-o", "--output", default=None,
                   help="Save figure to file instead of displaying")
    args = p.parse_args()
    
    files = args.file

    # Arrange panels in ascending numeric K order. Files without a numeric K
    # token keep a stable order after those that have one.
    def k_sort_key(f):
        v = parse_params(f).get("K")
        try:
            return (0, float(v))
        except (TypeError, ValueError):
            return (1, 0.0)
    files = sorted(files, key=k_sort_key)

    n_panels = len(files)

    fixed, varying_keys, all_params = split_fixed_varying(files)

    print(varying_keys)
    panel_titles = [
        "  ".join(f"{k}={p.get(k, '?')}" for k in varying_keys) or os.path.basename(f)
        for f, p in zip(files, all_params)
    ]
    fixed_str = "  ".join(f"{k}={v}" for k, v in fixed.items())
    suptitle = r"$S\cdot S$ (xx+yy+zz)"
    if fixed_str:
        suptitle += f"\n{fixed_str}"

    fig, axes = plt.subplots(3, n_panels, figsize=(2 * n_panels, 2*3), sharex=True, sharey=True, squeeze=False)

    fig.suptitle(suptitle)

    c = [plot_ssf(ax, f, args, title=t) for ax, f, t in zip(axes.T, files, panel_titles)]
    cax = fig.add_axes([0.05,0.9,0.3,0.02])
    plt.colorbar(c[0], cax=cax, orientation='horizontal')
        
    plt.tight_layout()

    if args.output:
        fig.savefig(args.output, dpi=150, bbox_inches="tight")
        print(f"Saved to {args.output}")
    else:
        plt.show()

if __name__ == "__main__":
    main()
