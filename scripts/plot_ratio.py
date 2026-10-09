#!/usr/bin/env python3
"""Plot the ratio S(Q2)/S(Q1) of the two strongest spiral star-point Bragg peaks
vs a scan parameter (-x), with series labelling (-s), à la plot_QaB.py.

The spiral wavevector Q has a three-arm star {(0,0,Q), (Q,0,0), (0,Q,0)}. For
each file the three peaks are ranked by intensity independently at every
temperature into Q1 >= Q2 >= Q3 (regardless of the order they appear in the
file), and the ratio S(Q2)/S(Q1) in [0, 1] is plotted (Q1 is the dominant peak,
Q2 the
subdominant one that may be small). A perfectly single-Q spiral drives the ratio
toward 0; a symmetric multi-Q / paramagnetic state drives it toward 1.
"""

from bragg import load_records, se_from_inter, se_from_intra
import argparse
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import sys


def ratio_se(A, B, seA, seB):
    """Standard error of R = A/B via linear error propagation (A, B assumed
    independent). Returns NaN unless both SEs are finite and B != 0."""
    if B == 0 or not (np.isfinite(seA) and np.isfinite(seB)):
        return np.nan
    R = A / B
    return abs(R) * np.sqrt((seA / A) ** 2 + (seB / B) ** 2) if A != 0 else \
        abs(seA / B)


def _peak_se(r, idx, err_source):
    """Per-spin SE of one Bragg intensity (index into the record's sorted
    arms) from the carried variance arrays; NaN when they are absent (e.g. a
    .bragg.csv input, which drops them)."""
    if r['var_inter'] is None or r['n_seeds'] is None:
        return np.nan
    si = sj = np.nan
    if err_source in ("inter", "total"):
        si = se_from_inter(r['var_inter'][idx], r['n_seeds'], r['n_spins'])
    if err_source in ("intra", "total") and r['var_intra'] is not None:
        sj = se_from_intra(r['var_intra'][idx], r['n_seeds'],
                           r['n_per_seed'], r['n_spins'])
    parts = [v for v in (si, sj) if np.isfinite(v)]
    return np.sqrt(sum(v ** 2 for v in parts)) if parts else np.nan


def load_and_expand(args):
    # T is a genuine parameter: expand each file over all its sampled
    # temperatures exactly when -x or -s asks for it; otherwise collapse to a
    # single temperature (coldest, or --t-index). load_records reads either raw
    # *.out.h5 files or a pre-extracted *.bragg.csv summary.
    uses_T = 'T' in (args.x_axis, args.series_axis)
    records_b, title_T, fixed = load_records(args.file, uses_T, args.t_index)

    records = []
    for r in records_b:
        pm = r['pm']
        x_raw = pm.get(args.x_axis)
        if x_raw is None:
            print(f"Warning: '{args.x_axis}' not found in a record, skipping",
                  file=sys.stderr)
            continue
        try:
            x_val = float(x_raw)
        except (TypeError, ValueError):
            x_val = x_raw

        ser = pm.get(args.series_axis) if args.series_axis else None

        # The arms are already ranked per T into [Q1, Q2, Q3, Γ]: Q1 is the
        # dominant peak, Q2 the subdominant one that may be small, so the ratio
        # Q2/Q1 stays bounded in [0, 1]. (Re-ranking per T rather than once at a
        # reference T keeps Q2 on the true subdominant peak — a fixed ranking
        # picks the arm that dies for ~half the seeds, inflating the variance.)
        den, num = r['I'][0], r['I'][1]
        R = num / den if den != 0 else np.nan

        # norm cancels in the ratio, so the SE is propagated on the per-spin
        # intensities. Absent variance (a .bragg.csv input) leaves SE = NaN and
        # the caller falls back to seed-to-seed spread.
        if r['var_inter'] is not None and r['n_spins']:
            SE = ratio_se(num / r['n_spins'], den / r['n_spins'],
                          _peak_se(r, 1, args.err_source),
                          _peak_se(r, 0, args.err_source))
        else:
            SE = np.nan

        records.append({'pm': pm, 'x': x_val, 'ser': ser, 'R': R, 'SE': SE})

    return records, title_T, fixed

def provide_selector_args(p : argparse.ArgumentParser):
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

def main():
    p = argparse.ArgumentParser(
        description="Plot the ratio S(Q2)/S(Q1) of the two strongest spiral "
                    "star-point Bragg peaks vs a scan parameter."
    )
    provide_selector_args(p)
    p.add_argument("--vmin", type=float, default=None)
    p.add_argument("--vmax", type=float, default=None)
    p.add_argument("-o", "--output", default=None,
                   help="Save figure to file instead of displaying")
    args = p.parse_args()


    x_label = args.x_axis

    def fmt_val(v):
        return f"{v:g}" if isinstance(v, float) else str(v)

    # ---- load + expand each file into per-(file, temperature) records ----
    records, title_T, fixed = load_and_expand(args)

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
