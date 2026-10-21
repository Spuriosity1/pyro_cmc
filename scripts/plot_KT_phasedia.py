#!/usr/bin/env python3
"""Extract the ratio S(Q2)/S(Q1) of the two strongest spiral star-point Bragg peaks as a function of the scan parameter -x. When the ratio is greater than eps (i.e. the peak is discernible), we fit a linear extrapolation curve to extract the K point where the peak vanishes. We render this in 2D, vs. an extra parameter -y.

The spiral wavevector Q has a three-arm star {(0,0,Q), (Q,0,0), (0,Q,0)}. For
each file the three peaks are ranked by intensity independently at every
temperature into Q1 >= Q2 >= Q3 (regardless of the order they appear in the
file), and the ratio S(Q2)/S(Q1) in [0, 1] is plotted (Q1 is the dominant peak,
Q2 the
subdominant one that may be small). A perfectly single-Q spiral drives the ratio
toward 0; a symmetric multi-Q / paramagnetic state drives it toward 1.
"""

import argparse

from plot_ratio import (provide_selector_args, load_and_expand,
                        use_tex_serif, phase_boundary, draw_phase_boundary)
import sys
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt


def fmt_val(v):
    return f"{v:g}" if isinstance(v, float) else str(v)

def main():
    p = argparse.ArgumentParser(
        description="Plot the ratio S(Q2)/S(Q1) of the two strongest spiral "
                    "star-point Bragg peaks vs a scan parameter."
    )
    provide_selector_args(p)

    p.add_argument("--eps", help="Cutoff ratio  S2/S2 to be treated as zero (defaults to 1e-4)",type=float, default=1e-4)
    p.add_argument("-o", "--output", default=None,
                   help="Save figure to file instead of displaying")
    args = p.parse_args()

    use_tex_serif()

    # ---- load + expand each file into per-(file, temperature) records ----
    records, title_T, fixed = load_and_expand(args)

    if not records:
        sys.exit("No data to plot — check the requested -x/-s parameters and that "
                 "the Qz tag exists in the filenames.")

    # ---- accumulate runs that differ only in the binned parameters ----
    # Group by every parameter except the binned ones (unless a binned
    # parameter is itself an axis), then plot the mean of the accumulated ratios
    # with their sample stdev as the error bar. A lone-member group keeps its
    # propagated within-file SE instead. 'seed' is binned by default; --bin adds
    # others (e.g. a J3 that merely tracks the scanned K).
    drop = set(args.bin) - {args.x_axis, args.series_axis}
    groups = {}
    for r in records:
        key = tuple(sorted((k, v) for k, v in r['pm'].items() if k not in drop))
        groups.setdefault(key, []).append(r)

    # Each ax2 trace is one "trace group": the unique combination of parameters
    # that are neither the x-axis, the series-axis, nor binned (already averaged
    # over) — e.g. a distinct L. Fixed parameters are excluded so the trace
    # labels stay concise.
    trace_drop = {args.x_axis, args.series_axis} | set(args.bin) | set(fixed.keys())

    points = []
    for grp in groups.values():
        Rs = np.array([g['R'] for g in grp], dtype=float)
        n = int(np.count_nonzero(np.isfinite(Rs)))
        R_mean = np.nanmean(Rs) if n else np.nan
        R_err = np.nanstd(Rs, ddof=1) if n > 1 else grp[0]['SE']
        pm0 = grp[0]['pm']
        trace = tuple(sorted((k, v) for k, v in pm0.items()
                             if k not in trace_drop))
        points.append({'x': grp[0]['x'], 'ser': grp[0]['ser'],
                       'R': R_mean, 'SE': R_err, 'trace': trace})

    def fmt_trace(trace):
        return ", ".join(f"{k}={fmt_val(v)}" for k, v in trace)

    trace_vals = sorted(set(p['trace'] for p in points))
    multi_trace = len(trace_vals) > 1

    def model_(x, m, x0, m2):
        return  (x - x0)*(m2*x+m) + args.eps

    fig, ax = plt.subplots()
    fig2, ax2 = plt.subplots()

    for trace in trace_vals:
        tpoints = [p for p in points if p['trace'] == trace]
        trace_label = fmt_trace(trace)

        series_vals = sorted(set(p['ser'] for p in tpoints),
                             key=lambda v: (v is None, v))
        series_data = {v: {'x': [], 'R': [], 'SE': []} for v in series_vals}
        for p in tpoints:
            d = series_data[p['ser']]
            d['x'].append(p['x'])
            d['R'].append(p['R'])
            d['SE'].append(p['SE'])

        for ser in series_vals:
            d = series_data[ser]
            if not d['x']:
                continue
            order = np.argsort(d['x'])
            x_sorted = np.array(d['x'])[order]
            R_sorted = np.array(d['R'])[order]
            SE_sorted = np.array(d['SE'])[order]

            parts = []
            if ser is not None:
                parts.append(fmt_val(ser))
            if multi_trace and trace_label:
                parts.append(trace_label)
            label = " / ".join(parts) if parts else None

            ax.errorbar(x_sorted, R_sorted, SE_sorted, label=label, lw=0, marker='s')

            # Draw the vanishing-peak extrapolation on the diagnostic axes; the
            # (K, T) boundary itself is assembled once below via phase_boundary.
            R_finite = R_sorted > args.eps
            try:
                popt, _ = curve_fit(model_, x_sorted[R_finite],
                                    R_sorted[R_finite], [1, -0.1, 0],
                                    sigma=SE_sorted[R_finite])
                ax.plot(x_sorted, model_(x_sorted, *popt),
                        color=ax.lines[-1].get_color())
            except Exception:
                print(f"Fit failed for series {ser}"
                      + (f" ({trace_label})" if trace_label else ""))

    ax.axhline(args.eps, ls=':', color='k')
    ax.set_xlabel(r"$S(S+1)\kappa/|J_1|$")
    ax.set_ylabel(r"$S(\bm{q}_{\rm PG1})\,/\,S(\bm{q}_{\rm B})$")
    ax.set_ylim(0, None)
    fig.legend(fontsize=7)

    # ---- fig2: polished (K, T) phase boundary where the PG1 peak vanishes ----
    boundary = phase_boundary(args, args.eps)
    draw_phase_boundary(ax2, boundary,
                        labeler=(fmt_trace if multi_trace else None))
    ax2.set_xlabel(r"$S(S+1)\kappa/|J_1|$")
    ax2.set_ylabel(r"$T/|J_1|$")
    ax2.grid(True, which='both', lw=0.3, alpha=0.4)
    if multi_trace:
        ax2.legend(title=r"$L$", fontsize=8)

    fig2.tight_layout()
    if args.output:
        fig2.savefig(args.output, dpi=150, bbox_inches="tight")
        print(f"Saved to {args.output}")
    plt.show()


main()
