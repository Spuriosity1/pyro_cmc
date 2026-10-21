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
import shutil
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import sys


def use_tex_serif():
    """Switch matplotlib to a LaTeX serif style for publication figures.

    Uses a real LaTeX install when one is on PATH; otherwise falls back to
    matplotlib's built-in Computer-Modern mathtext so the serif look survives
    without an external latex binary."""
    rc = {
        'font.family': 'serif',
        'font.serif': ['Computer Modern Roman', 'cmr10', 'DejaVu Serif'],
        'font.size': 8,
        'mathtext.fontset': 'cm',
        'axes.formatter.use_mathtext': True,
    }
    if shutil.which('latex'):
        rc['text.usetex'] = True
        rc['text.latex.preamble'] = r'\usepackage{bm}\usepackage{sansmath}'
    mpl.rcParams.update(rc)


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


def load_and_expand(args, num_idx=1):
    # num_idx selects which ranked arm is the ratio numerator: 1 => the
    # subdominant peak Q2 (the default S(Q2)/S(Q1)), 2 => the third-ranked peak
    # Q3 (for S(Q3)/S(Q1)). The denominator is always the dominant peak Q1.
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
        den, num = r['I'][0], r['I'][num_idx]
        R = num / den if den != 0 else np.nan

        # norm cancels in the ratio, so the SE is propagated on the per-spin
        # intensities. Absent variance (a .bragg.csv input) leaves SE = NaN and
        # the caller falls back to seed-to-seed spread.
        if r['var_inter'] is not None and r['n_spins']:
            SE = ratio_se(num / r['n_spins'], den / r['n_spins'],
                          _peak_se(r, num_idx, args.err_source),
                          _peak_se(r, 0, args.err_source))
        else:
            SE = np.nan

        records.append({'pm': pm, 'x': x_val, 'ser': ser, 'R': R, 'SE': SE})

    return records, title_T, fixed

def accumulate_points(args, num_idx=1):
    """Load every file, expand over T, then accumulate runs that differ only in
    the binned parameters into one point apiece. Returns (points, title_T,
    fixed) where each point is a dict with x, ser, L, R (mean ratio) and SE.

    num_idx forwards to load_and_expand: 1 for the default S(Q2)/S(Q1), 2 for
    S(Q3)/S(Q1).

    Grouping key is every filename parameter except the binned ones (minus the
    x/series axes, which must keep splitting). The mean of the per-seed ratios
    is reported with their sample stdev as the error bar; a lone-seed group
    keeps its propagated within-file SE instead.
    """
    records, title_T, fixed = load_and_expand(args, num_idx)

    drop = set(args.bin) - {args.x_axis, args.series_axis}
    groups = {}
    for r in records:
        key = tuple(sorted((k, v) for k, v in r['pm'].items() if k not in drop))
        groups.setdefault(key, []).append(r)

    points = []
    for grp in groups.values():
        Rs = np.array([g['R'] for g in grp], dtype=float)
        Ls = np.array([g['pm']['L'] for g in grp], dtype=int)
        n = int(np.count_nonzero(np.isfinite(Rs)))
        R_mean = np.nanmean(Rs) if n else np.nan
        R_err = np.nanstd(Rs, ddof=1) if n > 1 else grp[0]['SE']
        points.append({'x': grp[0]['x'], 'ser': grp[0]['ser'],
                       'L': int(np.round(np.mean(Ls))),
                       'R': R_mean, 'SE': R_err})
    return points, title_T, fixed


def series_colors(series_vals, series_axis,cmap = plt.colormaps['viridis']):
    """Map each series value to a colour. Numeric series get a viridis ramp
    (log-scaled for T, which spans a log-spaced schedule); non-numeric series
    fall back to the property cycle. Returns (series_color, cnorm, cmap) with
    cnorm/cmap None in the non-numeric case."""
    try:
        vals_np = [float(v) for v in series_vals]
        if series_axis == 'T' and min(vals_np) > 0:
            cnorm = mpl.colors.LogNorm(min(vals_np), max(vals_np))
        else:
            cnorm = mpl.colors.Normalize(min(vals_np), max(vals_np))
        series_color = {v: cmap(cnorm(float(v))) for v in series_vals}
    except (TypeError, ValueError):
        cmap = cnorm = None
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        series_color = {v: colors[i % len(colors)]
                        for i, v in enumerate(series_vals)}
    return series_color, cnorm, cmap


def phase_boundary(args, eps):
    """Fit, per trace and temperature, the scan value where the subdominant
    Bragg peak vanishes. The ratio S(Q2)/S(Q1) is fit above `eps` with the
    model (x - x0)(m2 x + m) + eps, and x0 — the extrapolated x where the peak
    disappears — is kept with its fit error.

    Returns {trace: {'T', 'x0', 'x0_err'}} with arrays sorted by the (numeric)
    series value. `trace` is a tuple of (key, value) pairs for the parameters
    that are neither the x-axis, series axis, binned, nor fixed (e.g. a distinct
    L). Shared by plot_KT_phasedia's fig2 and the plot_final_ratio inset so both
    read exactly the same boundary."""
    from scipy.optimize import curve_fit

    records, _, fixed = load_and_expand(args)
    drop = set(args.bin) - {args.x_axis, args.series_axis}
    groups = {}
    for r in records:
        key = tuple(sorted((k, v) for k, v in r['pm'].items() if k not in drop))
        groups.setdefault(key, []).append(r)

    trace_drop = ({args.x_axis, args.series_axis}
                  | set(args.bin) | set(fixed.keys()))
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

    def model_(x, m, x0, m2):
        return (x - x0) * (m2 * x + m) 

    def _tracekey(trace):
        # Sort traces by value numerically where possible, so e.g. an L-string
        # trace orders 6, 8, 10, 20 rather than lexicographically '10','20','6'.
        key = []
        for _, v in trace:
            try:
                key.append(float(v))
            except (TypeError, ValueError):
                key.append(v)
        return tuple(key)

    boundary = {}
    for trace in sorted(set(p['trace'] for p in points), key=_tracekey):
        tpoints = [p for p in points if p['trace'] == trace]
        T_out, x0_out, x0_err = [], [], []
        for ser in sorted(set(p['ser'] for p in tpoints),
                          key=lambda v: (v is None, v)):
            if ser is None:
                continue
            d = [p for p in tpoints if p['ser'] == ser]
            order = np.argsort([p['x'] for p in d])
            xs = np.array([p['x'] for p in d])[order]
            Rs = np.array([p['R'] for p in d])[order]
            SEs = np.array([p['SE'] for p in d])[order]
            mask = Rs > eps
            if mask.sum() < 3:
                continue
            try:
                popt, pcov = curve_fit(model_, xs[mask], Rs[mask],
                                       [1, -0.1, 0], sigma=SEs[mask])
            except Exception:
                continue
            T_out.append(float(ser))
            x0_out.append(float(popt[1]))
            x0_err.append(float(np.sqrt(pcov[1, 1])))
        if T_out:
            o = np.argsort(T_out)
            boundary[trace] = {'T': np.array(T_out)[o],
                               'x0': np.array(x0_out)[o],
                               'x0_err': np.array(x0_err)[o]}
    return boundary


def draw_phase_boundary(ax, boundary, colors=None, labeler=None,
                        marker='s', ms=3, lw=1, **kw):
    """Render phase_boundary() output as the vanishing-peak locus x0 (x-axis)
    vs the series value T (y-axis), one error-bar trace apiece. `colors` maps a
    trace to a colour; `labeler` maps a trace to a legend label."""
    for i, (trace, b) in enumerate(boundary.items()):
        color = (colors.get(trace) if isinstance(colors, dict)
                 else colors[i] if colors else None)
        label = labeler(trace) if labeler else None
        
        ax.errorbar(b['x0'], b['T'], xerr=b['x0_err'], marker=marker, ms=ms,
                    lw=lw, capsize=1.5, color=color, label=label, **kw)


def parse_final_args(ddir="/home/als/Projects/016_spiral_cmc/out_db"):
    """Build the fixed argument set shared by the 'final' plots: the run13
    L-scan files, K on the x-axis, T as the series, binning J3 (slaved to K)
    and seed."""
    p = argparse.ArgumentParser()
    provide_selector_args(p)
    files = [f'{ddir}/run13_L={L}.csv' for L in (6, 8, 10, 12, 14, 16, 18, 20)]
    # files = [f'{ddir}/run13_L={L:02d}.csv' for L in (8, 12, 16, 20)]
    return p.parse_args([*files, "-x", "K", "-s", "T", "--bin", "J3", "seed"])


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
    p.add_argument("--bin", nargs='+', default=['seed'],
                   help="Filename parameters to bin over, i.e. ignore when "
                        "separating distinct parameter sets into series/traces "
                        "(their records are accumulated together, as seeds are). "
                        "Defaults to ['seed']; pass e.g. '--bin seed J3' to fold "
                        "a parameter such as J3 that is slaved to the scanned K "
                        "and so must not split the series.")
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

    # ---- load, expand over T, and accumulate runs over the binned params ----
    points, title_T, fixed = accumulate_points(args)

    if not points:
        sys.exit("No data to plot — check the requested -x/-s parameters and that "
                 "the Qz tag exists in the filenames.")

    # ---- derive series values + colours from the accumulated points ----
    series_vals = sorted(set(p['ser'] for p in points),
                         key=lambda v: (v is None, v))
    series_color, cnorm, cmap = series_colors(series_vals, args.series_axis)

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
                        fmt='x', label=label, color=color, ms=3,
                        capsize=2, elinewidth=1)
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
