#!/usr/bin/env python3
"""Plot SSF intensity at (0,0,Qz) and symmetry-equivalent q-points vs a scan parameter."""

from bragg import load_records, se_from_inter, se_from_intra
import argparse
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import sys


def main():
    p = argparse.ArgumentParser(
        description="Plot SSF intensity at Bragg/symmetry-equivalent q-points vs a scan parameter."
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
    p.add_argument("--err-source", choices=["inter", "intra", "total", "both"],
                   default="inter",
                   help="Error bar source (default: inter). 'inter' = seed-to-seed "
                        "SE of the mean; 'intra' = within-run sampling-noise SE "
                        "only; 'total' = both combined in quadrature; 'both' = draw "
                        "intra as a faint wide band behind the solid inter-seed bars")
    p.add_argument("--vmin", type=float, default=None)
    p.add_argument("--vmax", type=float, default=None)
    p.add_argument("--per-site", action="store_true",
                   help="Normalise by N^2 instead of N, i.e. plot the intensive "
                        "order parameter m^2 = S(Q)/N. A Bragg peak scales as "
                        "S(Q) ~ N*m^2, so this makes the ordered-peak curves for "
                        "different L overlap (the default /N leaves it extensive). "
                        "Note: diffuse (~N) points then scale as 1/N.")
    p.add_argument("-o", "--output", default=None,
                   help="Save figure to file instead of displaying")
    args = p.parse_args()

    x_label = args.x_axis
    # T is a genuine parameter: it is "in play" (so we expand each file over all
    # its sampled temperatures) exactly when -x or -s asks for it. Otherwise we
    # collapse to a single temperature (coldest, or --t-index).
    uses_T = 'T' in (args.x_axis, args.series_axis)

    # Bragg records come from either raw *.out.h5 files or a pre-extracted
    # *.bragg.csv summary; load_records dispatches on the filename. Each record
    # carries the merged parameter dict (filename tags + 'T'), the four
    # intensities [Q1, Q2, Q3, Γ] (arms already sorted desc), n_spins, and the
    # per-seed variance arrays (None when they came from a lossy .bragg.csv).
    records_b, title_T, fixed = load_records(args.file, uses_T, args.t_index)

    def fmt_val(v):
        return f"{v:g}" if isinstance(v, float) else str(v)

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    ax_flat = [axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]]

    # First three panels hold the Bragg peaks ranked (per file) by intensity,
    # not fixed directions; Gamma is pinned to the last panel.
    panel_labels = [
        r"highest $S(\mathbf{q})$",
        r"2nd highest",
        r"3rd highest",
        r"$\Gamma = (0,0,0)$",
    ]
    y_label = (r"$m^2 = S(\mathbf{q})/N$" if args.per_site
               else r"$S(\mathbf{q})$ / spin")
    for ax, title in zip(ax_flat, panel_labels):
        ax.set_title(title)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)

    # ---- derive x / series / error bars from each Bragg record ----
    # The four intensities are already [Q1, Q2, Q3, Γ] (arms ranked per T),
    # matching the four panel labels above.
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

        # Default /N gives the standard structure factor S(q); --per-site divides
        # by N again to give the intensive order parameter m^2 = S(Q)/N, so the
        # ordered Bragg peak overlaps across system sizes L.
        norm = r['n_spins']**2 if args.per_site else r['n_spins']

        I = [r['I'][panel] / norm for panel in range(4)]
        SE_inter, SE_intra = [], []
        for panel in range(4):
            if r['var_inter'] is not None:
                SE_inter.append(se_from_inter(r['var_inter'][panel], r['n_seeds'], norm))
            else:
                SE_inter.append(np.nan)
            if r['var_intra'] is not None:
                SE_intra.append(se_from_intra(r['var_intra'][panel], r['n_seeds'],
                                              r['n_per_seed'], norm))
            else:
                SE_intra.append(np.nan)

        records.append({'x': x_val, 'ser': ser,
                        'I': I, 'SE_inter': SE_inter, 'SE_intra': SE_intra})

    if not records:
        sys.exit("No data to plot — check the requested -x/-s parameters and that "
                 "the Qz tag exists in the filenames.")

    # ---- derive series values + colours from the collected records ----
    series_vals = sorted(set(r['ser'] for r in records),
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

    series_data = {
        v: {'x': [], 'I': [[], [], [], []],
            'SE_inter': [[], [], [], []], 'SE_intra': [[], [], [], []]}
        for v in series_vals
    }
    for r in records:
        d = series_data[r['ser']]
        d['x'].append(r['x'])
        for panel in range(4):
            d['I'][panel].append(r['I'][panel])
            d['SE_inter'][panel].append(r['SE_inter'][panel])
            d['SE_intra'][panel].append(r['SE_intra'][panel])

    plotted_any = False
    for ser in series_vals:
        d = series_data[ser]
        if not d['x']:
            continue
        order = np.argsort(d['x'])
        x_sorted = np.array(d['x'])[order]
        label = fmt_val(ser) if ser is not None else None
        color = series_color[ser]
        for panel, ax in enumerate(ax_flat):
            I_sorted = np.array(d['I'][panel])[order]
            SEi_sorted = np.array(d['SE_inter'][panel])[order]
            SEa_sorted = np.array(d['SE_intra'][panel])[order]

            if args.err_source == "inter":
                se_main = SEi_sorted
            elif args.err_source == "intra":
                se_main = SEa_sorted
            else:  # total or both -> main bar is the combined/inter estimate
                have_either = np.isfinite(SEi_sorted) | np.isfinite(SEa_sorted)
                se_main = np.sqrt(np.nan_to_num(SEi_sorted)**2
                                   + np.nan_to_num(SEa_sorted)**2)
                se_main = np.where(have_either, se_main, np.nan)

            if args.err_source == "both" and np.any(np.isfinite(SEa_sorted)):
                # faint wide band behind the main bars shows intra-seed spread
                ax.errorbar(x_sorted, I_sorted, yerr=np.nan_to_num(SEa_sorted),
                            lw=0,
                            fmt='o', ecolor=color, alpha=0.3,
                            
                            elinewidth=5, capsize=0, zorder=1)
                se_main = SEi_sorted

            if np.any(np.isfinite(se_main)):
                ax.errorbar(x_sorted, I_sorted, yerr=se_main,
                            fmt='o', label=label, color=color, ms=4,
                            lw=0,
                            capsize=3, elinewidth=1, zorder=2)
            else:
                ax.plot(x_sorted, I_sorted, 'o', alpha=0.2, label=label, color=color, ms=4, zorder=2)
        plotted_any = True

    if not plotted_any:
        sys.exit("No data to plot — check the requested -x/-s parameters and that "
                 "the Qz tag exists in the filenames.")

    if args.series_axis:
        ax_flat[0].legend(title=args.series_axis, fontsize=8)

    if args.vmin is not None or args.vmax is not None:
        for ax in ax_flat:
            ax.set_ylim(args.vmin, args.vmax)

    fixed_str = "  ".join(f"{k}={v}" for k, v in fixed.items())

    base = r"$S(\mathbf{q})$ at symmetry-equivalent wavevectors"
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
