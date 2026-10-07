#!/usr/bin/env python3
"""Plot the ratio S(Q2)/S(Q1) of the two strongest spiral star-point Bragg peaks
vs a scan parameter (-x), with series labelling (-s), à la plot_QaB.py.

The spiral wavevector Q has a three-arm star {(0,0,Q), (Q,0,0), (0,Q,0)}. For
each file the three peaks are ranked by intensity at a reference temperature
into Q1 >= Q2 >= Q3 (regardless of the order they appear in the file), and the
ratio S(Q2)/S(Q1) in [0, 1] is plotted (Q1 is the dominant peak, Q2 the
subdominant one that may be small). A perfectly single-Q spiral drives the ratio
toward 0; a symmetric multi-Q / paramagnetic state drives it toward 1.
"""

from plot_ssf import load_file, normalize_ssf, split_fixed_varying
from plot_QaB import (qz_to_idx, load_ssf_variance, se_from_inter,
                      se_from_intra, filter_corrupted)
import argparse
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import os
import sys


def ratio_se(A, B, seA, seB):
    """Standard error of R = A/B via linear error propagation (A, B assumed
    independent). Returns NaN unless both SEs are finite and B != 0."""
    if B == 0 or not (np.isfinite(seA) and np.isfinite(seB)):
        return np.nan
    R = A / B
    return abs(R) * np.sqrt((seA / A) ** 2 + (seB / B) ** 2) if A != 0 else \
        abs(seA / B)


def main():
    p = argparse.ArgumentParser(
        description="Plot the ratio S(Q2)/S(Q1) of the two strongest spiral "
                    "star-point Bragg peaks vs a scan parameter."
    )
    p.add_argument("file", help="Path(s) to HDF5 file", nargs='+')
    p.add_argument("-x", "--x-axis", default="T",
                   help="Parameter to plot along the x-axis (default: 'T' = "
                        "temperature). Any other value is a filename tag.")
    p.add_argument("-s", "--series-axis",
                   help="Parameter to use as a series label; 'T' selects "
                        "temperature, any other value is a filename tag.")
    p.add_argument("-t", "--t-index", type=int, default=None,
                   help="Temperature index into the SSF array (default: last = coldest); "
                        "used only when neither -x nor -s is 'T'.")
    p.add_argument("--err-source", choices=["inter", "intra", "total"],
                   default="inter",
                   help="Error bar source for the propagated ratio SE "
                        "(default: inter)")
    p.add_argument("--vmin", type=float, default=None)
    p.add_argument("--vmax", type=float, default=None)
    p.add_argument("-o", "--output", default=None,
                   help="Save figure to file instead of displaying")
    args = p.parse_args()

    files = filter_corrupted(args.file)

    fixed, _, all_params = split_fixed_varying(files)

    x_label = args.x_axis
    # T is a genuine parameter: expand each file over all its sampled
    # temperatures exactly when -x or -s asks for it; otherwise collapse to a
    # single temperature (coldest, or --t-index).
    uses_T = 'T' in (args.x_axis, args.series_axis)

    def fmt_val(v):
        return f"{v:g}" if isinstance(v, float) else str(v)

    # ---- load + expand each file into per-(file, temperature) records ----
    records = []
    title_T = None
    for fpath, file_params in zip(files, all_params):
        qz_str = file_params.get('Q') or file_params.get('Qz')
        if qz_str is None:
            print(f"Warning: 'Q' or 'Qz' not found in {os.path.basename(fpath)}, skipping",
                  file=sys.stderr)
            continue
        qz = float(qz_str)

        try:
            (_, _, _, _, _, _,
             corr, corr_lookup, sl_positions, k_dims, n_spins, ssf_T, n_ssf) = load_file(fpath)
        except Exception:
            print(f"{fpath}")
            continue

        n_T = corr.shape[1]
        if uses_T:
            t_indices = range(n_T)
        else:
            t_idx0 = args.t_index if args.t_index is not None else n_T - 1
            if not (0 <= t_idx0 < n_T):
                sys.exit(f"--t-index {t_idx0} out of range [0, {n_T - 1}]")
            t_indices = [t_idx0]
            title_T = float(ssf_T[t_idx0])

        S = normalize_ssf(corr, corr_lookup, k_dims, n_ssf)
        diag = [c for c in ("xx", "yy", "zz") if c in S]
        if not diag:
            print(f"Warning: no diagonal correlators in {os.path.basename(fpath)}, skipping",
                  file=sys.stderr)
            continue

        var_inter, var_intra, n_seeds = load_ssf_variance(fpath)

        # Qz is in units of 2π/a_cubic; k_dims[i] = L for cubic supercell.
        qi = qz_to_idx(qz, k_dims[0])
        star = [(0, 0, qi), (qi, 0, 0), (0, qi, 0)]  # the three spiral arms

        # Rank the three star points by intensity once, at the coldest
        # temperature, so Q1/Q2/Q3 stay the same physical q-point across the
        # whole x-axis for this file.
        ref_t = n_T - 1
        ref_I = [sum(S[c][ref_t, i0, i1, i2] for c in diag)
                 for (i0, i1, i2) in star]
        perm = list(np.argsort(ref_I)[::-1])  # perm[rank] -> star index; 0=Q1

        q1 = star[perm[0]]
        q2 = star[perm[1]]

        for t_idx in t_indices:
            pm = dict(file_params)
            pm['T'] = float(ssf_T[t_idx])

            x_raw = pm.get(args.x_axis)
            if x_raw is None:
                print(f"Warning: '{args.x_axis}' not found in {os.path.basename(fpath)}, skipping",
                      file=sys.stderr)
                continue
            try:
                x_val = float(x_raw)
            except (TypeError, ValueError):
                x_val = x_raw

            ser = pm.get(args.series_axis) if args.series_axis else None
            n_per_seed = n_ssf[t_idx] / n_seeds if n_seeds else np.nan

            def intensity(q):
                return sum(S[c][t_idx, q[0], q[1], q[2]] for c in diag)

            def se(q):
                if var_inter is None or n_seeds is None:
                    return np.nan
                # norm cancels in the ratio, so SE is computed on the raw
                # (per-spin, /n_spins) intensity consistently with intensity().
                si = sj = np.nan
                if args.err_source in ("inter", "total"):
                    W = sum(var_inter[c][t_idx, q[0], q[1], q[2]] for c in diag)
                    si = se_from_inter(W, n_seeds, n_spins)
                if args.err_source in ("intra", "total") and var_intra is not None:
                    W = sum(var_intra[c][t_idx, q[0], q[1], q[2]] for c in diag)
                    sj = se_from_intra(W, n_seeds, n_per_seed, n_spins)
                parts = [v for v in (si, sj) if np.isfinite(v)]
                return np.sqrt(sum(v ** 2 for v in parts)) if parts else np.nan

            # Q2/Q1: Q1 is the dominant (large) peak, Q2 the subdominant one
            # that may be small, so this ratio stays bounded in [0, 1].
            num, den = intensity(q2), intensity(q1)
            R = num / den if den != 0 else np.nan
            # intensity() is summed over samples then /n_spins in normalize_ssf;
            # se() returns the per-spin SE, matching the num/den scaling.
            SE = ratio_se(num / n_spins, den / n_spins, se(q2), se(q1))

            records.append({'pm': pm, 'x': x_val, 'ser': ser, 'R': R, 'SE': SE})

    if not records:
        sys.exit("No data to plot — check the requested -x/-s parameters and that "
                 "the Qz tag exists in the filenames.")

    # ---- accumulate runs that differ only in 'seed' ----
    # Group by every parameter except the seed (unless seed is itself an axis),
    # then plot the mean of the per-seed ratios with their sample stdev as the
    # error bar. A lone-seed group keeps its propagated within-file SE instead.
    drop = {'seed'} - {args.x_axis, args.series_axis}
    groups = {}
    for r in records:
        key = tuple(sorted((k, v) for k, v in r['pm'].items() if k not in drop))
        groups.setdefault(key, []).append(r)

    points = []
    for grp in groups.values():
        Rs = np.array([g['R'] for g in grp], dtype=float)
        n = int(np.count_nonzero(np.isfinite(Rs)))
        R_mean = np.nanmean(Rs) if n else np.nan
        R_err = np.nanstd(Rs, ddof=1) if n > 1 else grp[0]['SE']
        points.append({'x': grp[0]['x'], 'ser': grp[0]['ser'],
                       'R': R_mean, 'SE': R_err})

    # ---- derive series values + colours from the accumulated points ----
    series_vals = sorted(set(p['ser'] for p in points),
                         key=lambda v: (v is None, v))

    try:
        vals_np = [float(v) for v in series_vals]
        cmap = plt.colormaps['viridis']
        # T spans a log-spaced schedule, so colour it on a log scale.
        if args.series_axis == 'T' and min(vals_np) > 0:
            cnorm = mpl.colors.LogNorm(min(vals_np), max(vals_np))
        else:
            cnorm = mpl.colors.Normalize(min(vals_np), max(vals_np))
        series_color = {v: cmap(cnorm(float(v))) for v in series_vals}
    except (TypeError, ValueError):
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        series_color = {v: colors[i % len(colors)] for i, v in enumerate(series_vals)}

    series_data = {v: {'x': [], 'R': [], 'SE': []} for v in series_vals}
    for p in points:
        d = series_data[p['ser']]
        d['x'].append(p['x'])
        d['R'].append(p['R'])
        d['SE'].append(p['SE'])

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.set_xlabel(x_label)
    ax.set_ylabel(r"$S(\mathbf{Q}_2)\,/\,S(\mathbf{Q}_1)$")
    ax.axhline(1.0, color='k', lw=0.8, ls='--', alpha=0.5)

    plotted_any = False
    for ser in series_vals:
        d = series_data[ser]
        if not d['x']:
            continue
        order = np.argsort(d['x'])
        x_sorted = np.array(d['x'])[order]
        R_sorted = np.array(d['R'])[order]
        SE_sorted = np.array(d['SE'])[order]
        label = fmt_val(ser) if ser is not None else None
        color = series_color[ser]

        if np.any(np.isfinite(SE_sorted)):
            ax.errorbar(x_sorted, R_sorted, yerr=SE_sorted,
                        fmt='o-', label=label, color=color, ms=4,
                        capsize=3, elinewidth=1)
        else:
            ax.plot(x_sorted, R_sorted, 'o-', label=label, color=color, ms=4)
        plotted_any = True

    if not plotted_any:
        sys.exit("No data to plot — check the requested -x/-s parameters and that "
                 "the Qz tag exists in the filenames.")

    if args.series_axis:
        ax.legend(title=args.series_axis, fontsize=8)

    if args.vmin is not None or args.vmax is not None:
        ax.set_ylim(args.vmin, args.vmax)

    fixed_str = "  ".join(f"{k}={v}" for k, v in fixed.items())
    base = r"Spiral peak-ratio $S(\mathbf{Q}_2)/S(\mathbf{Q}_1)$"
    if args.x_axis == 'T':
        suptitle = base + r" vs $T$"
    elif args.series_axis == 'T':
        suptitle = base + r" (series: $T$)"
    else:
        suptitle = base + (f", T={title_T:g}" if title_T is not None else "")
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
