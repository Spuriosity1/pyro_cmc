#!/usr/bin/env python3
"""Plot static structure factor from anneal HDF5 output."""

import argparse
import sys
import numpy as np
import h5py
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


def load_file(path):
    with h5py.File(path, "r") as f:
        recip = f["/geometry/recip_vectors"][:]   # (3, 3), rows are a*, b*, c*
        index_cell = f["/geometry/index_cell"][:]  # (3, 3)

        T_list = f["/energy/T_list"][:]
        E      = f["/energy/E"][:]
        E2     = f["/energy/E2"][:]
        n_E    = f["/energy/n_samples"][:]

        SdotS    = f["/ssf/SdotS"][:]     # (n_T, Nk0, Nk1)
        SzSz     = f["/ssf/SzSz"][:]
        n_ssf    = f["/ssf/n_samples"][:]
    return recip, index_cell, T_list, E, E2, n_E, SdotS, SzSz, n_ssf


def kgrid_coords(N0, N1, recip, index_cell):
    """
    Return (x, y) arrays of k-point positions in Cartesian reciprocal space.

    The BZ grid is assumed to span the first two rows of recip_vectors,
    with points at h/N0 * b0 + k/N1 * b1 for h in [0,N0), k in [0,N1).
    index_cell encodes the supercell, so reciprocal grid spacing is
    b_i / index_cell[i,i] if diagonal, generalised below.
    """
    # Supercell reciprocal vectors: the SSF BZ is folded by index_cell
    # grid vectors = recip @ inv(index_cell^T) -- but we just need 2D plot
    # so use the first two primitive recip vectors scaled to the grid size
    b0 = recip[0]  # first recip vector
    b1 = recip[1]  # second recip vector

    h = np.arange(N0)
    k = np.arange(N1)
    hh, kk = np.meshgrid(h, k, indexing="ij")

    pts = (hh[..., None] / N0) * b0 + (kk[..., None] / N1) * b1
    x = pts[..., 0]
    y = pts[..., 1]
    return x, y


def plot_ssf(args):
    recip, index_cell, T_list, E, E2, n_E, SdotS, SzSz, n_ssf = load_file(args.file)

    n_T, N0, N1 = SdotS.shape

    # Normalise by number of samples
    SdotS = SdotS / n_ssf
    SzSz  = SzSz  / n_ssf

    # Select temperature index
    t_idx = args.t_index if args.t_index is not None else n_T - 1
    if not (0 <= t_idx < n_T):
        sys.exit(f"--t-index must be in [0, {n_T - 1}]")

    # Find corresponding temperature from T_list (SSF saved at n_T checkpoints)
    # We assume checkpoints are evenly spaced through the 100-step anneal
    checkpoint_stride = len(T_list) // n_T
    t_val = T_list[t_idx * checkpoint_stride]

    x, y = kgrid_coords(N0, N1, recip, index_cell)

    norm = mcolors.LogNorm if args.log else mcolors.Normalize
    cmap = args.cmap

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    fig.suptitle(f"{args.file}  —  T = {t_val:.4g}  (index {t_idx})")

    for ax, data, label in zip(axes, [SdotS[t_idx], SzSz[t_idx]], [r"$S\cdot S$", r"$S^z S^z$"]):
        vmin, vmax = data.min(), data.max()
        if args.log:
            vmin = max(vmin, 1e-6 * vmax)
            c = ax.pcolormesh(x, y, data, cmap=cmap, norm=mcolors.LogNorm(vmin=vmin, vmax=vmax), shading="auto")
        else:
            c = ax.pcolormesh(x, y, data, cmap=cmap, norm=mcolors.Normalize(vmin=0, vmax=vmax), shading="auto")
        plt.colorbar(c, ax=ax, label=label)
        ax.set_title(label)
        ax.set_xlabel(r"$k_x$")
        ax.set_ylabel(r"$k_y$")
        ax.set_aspect("equal")

    plt.tight_layout()

    if args.output:
        fig.savefig(args.output, dpi=150, bbox_inches="tight")
        print(f"Saved to {args.output}")
    else:
        plt.show()


def plot_energy(args):
    _, _, T_list, E, E2, n_E, *_ = load_file(args.file)

    # Specific heat: C = (⟨E²⟩ - ⟨E⟩²) / T²
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
        out = args.output.replace(".png", "_energy.png") if args.output.endswith(".png") else args.output + "_energy.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print(f"Saved to {out}")
    else:
        plt.show()


def main():
    p = argparse.ArgumentParser(description="Plot SSF (and optionally energy) from anneal HDF5 output.")
    p.add_argument("file", help="Path to HDF5 file")
    p.add_argument("-t", "--t-index", type=int, default=None,
                   help="Temperature index into the SSF array (default: last = coldest)")
    p.add_argument("--log", action="store_true", help="Use logarithmic colour scale")
    p.add_argument("--cmap", default="inferno", help="Matplotlib colormap (default: inferno)")
    p.add_argument("-o", "--output", default=None, help="Save figure to file instead of displaying")
    p.add_argument("--energy", action="store_true", help="Also plot E and specific heat vs T")
    args = p.parse_args()

    plot_ssf(args)
    if args.energy:
        plot_energy(args)


if __name__ == "__main__":
    main()
