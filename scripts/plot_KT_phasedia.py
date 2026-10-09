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

from plot_ratio import provide_selector_args, load_and_expand
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

    # Each ax2 trace is one "trace group": the unique combination of parameters
    # that are neither the x-axis, the series-axis, nor the seed (already
    # averaged over) — e.g. a distinct L. Fixed parameters are excluded so the
    # trace labels stay concise.
    trace_drop = {args.x_axis, args.series_axis, 'seed'} | set(fixed.keys())

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

    def model_(x, m, x0):
        return m * (x - x0)

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

        fit_xint = []
        fit_yval = []
        sigma_fit_xint = []

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

            R_finite = R_sorted > args.eps
            xx = x_sorted[R_finite]
            yy = R_sorted[R_finite]
            sigma_yy = SE_sorted[R_finite]

            try:
                popt, pcov = curve_fit(model_, xx, yy, [1, -0.1], sigma=sigma_yy)
                ax.plot(x_sorted, model_(x_sorted, *popt), color=ax.lines[-1].get_color())
                fit_xint.append(popt[1])
                fit_yval.append(float(ser))
                sigma_fit_xint.append(np.sqrt(pcov[1, 1]))
            except Exception:
                print(f"Fit failed for series {ser}"
                      + (f" ({trace_label})" if trace_label else ""))

        if not fit_xint:
            continue
        order2 = np.argsort(fit_yval)
        ax2.errorbar(np.array(fit_xint)[order2], np.array(fit_yval)[order2],
                     xerr=np.array(sigma_fit_xint)[order2], marker='s',
                     label=trace_label if trace_label else None)

    ax.axhline(args.eps, ls=':', color='k')
    ax.set_xlabel(args.x_axis)
    ax.set_ylabel("ratio $S(Q_1)/S(Q_2)$")

    ax2.set_xlabel(args.x_axis)
    ax2.set_ylabel(args.series_axis)
    if multi_trace:
        ax2.legend()

    ax.set_ylim(0, None)
    fig.legend()
    plt.show()


main()
