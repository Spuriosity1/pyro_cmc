"""Joy plot (ridgeline) of the spiral peak-ratio S(Q2)/S(Q1) vs the scan
parameter K, with one stacked ridge per system size L (the z-axis).

The spiral wavevector Q has a three-arm star {(0,0,Q), (Q,0,0), (0,Q,0)}. For
each file the three peaks are ranked by intensity independently at every
temperature into Q1 >= Q2 >= Q3, and the ratio S(Q2)/S(Q1) in [0, 1] is the
height of each ridge. A perfectly single-Q spiral drives the ratio toward 0; a
symmetric multi-Q / paramagnetic state drives it toward 1.

Each L is drawn as its own baseline-shifted ridge, stacked bottom (small L) to
top (large L), so the finite-size evolution of R(K) reads off as depth. Within
a ridge the temperature series are coloured lines sharing the ridge baseline.

This shares its data pipeline with plot_final_ratio.py (which flattens the same
points onto a single axes with per-L offsets); the only difference is the
rendering.
"""


from plot_ratio import *


use_tex_serif()

args = parse_final_args()

# ---- load, expand over T, accumulate over the binned params ----
points, title_T, fixed = accumulate_points(args)

# ---- derive series (T) values + colours from the accumulated points ----
series_vals = sorted(set(p['ser'] for p in points),
                     key=lambda v: (v is None, v))
series_color, cnorm, cmap = series_colors(series_vals, args.series_axis)

# ---- bin the points by (L, T) so each ridge (L) holds its T curves ----
L_vals = sorted(set(p['L'] for p in points))
ridge = {L: {v: {'x': [], 'R': [], 'SE': []} for v in series_vals}
         for L in L_vals}
for p in points:
    d = ridge[p['L']][p['ser']]
    d['x'].append(p['x'])
    d['R'].append(p['R'])
    d['SE'].append(p['SE'])

# Vertical spacing between ridge baselines, set from the tallest ridge so
# neighbouring ridges overlap by ~40% — the classic joy-plot look.
all_R = np.array([p['R'] for p in points], dtype=float)
R_max = np.nanmax(all_R) if np.any(np.isfinite(all_R)) else 1.0
spacing = 0.2 * R_max


def fmt_val(v):
    return f"{v:g}" if isinstance(v, float) else str(v)


fig, ax = plt.subplots(figsize=(7, 8))

plotted_any = False
# Draw back-to-front: the highest-L ridge sits furthest back; lower L in front
# so its white fill occludes the ridges behind it.
for i, L in list(enumerate(L_vals))[::-1]:
    base = i * spacing
    z = (len(L_vals) - i) * 10  # front ridges (small L) get larger zorder

    for ser in series_vals:
        d = ridge[L][ser]
        if not d['x']:
            continue
        order = np.argsort(d['x'])
        x_sorted = np.array(d['x'])[order]
        R_sorted = np.array(d['R'])[order]
        SE_sorted = np.array(d['SE'])[order]
        color = series_color[ser]

        # White fill under each curve occludes the ridges drawn behind this one.
        ax.fill_between(x_sorted, base, base + R_sorted,
                        color='white', zorder=z)
        ax.errorbar(x_sorted, base + R_sorted, yerr=SE_sorted,
                    fmt='none', ecolor=color, capsize=1.5,
                    elinewidth=0.8, zorder=z + 1, alpha=0.6)
        ax.plot(x_sorted, base + R_sorted, color=color, lw=1.2, zorder=z + 2)
        plotted_any = True

    # Ridge baseline.
    ax.axhline(base, color='0.6', lw=0.6, zorder=z - 1)

if not plotted_any:
    sys.exit("No data to plot — check the requested -x/-s parameters and that "
             "the Qz tag exists in the filenames.")

# ---- z-axis: label each baseline with its L ----
ax.set_yticks([i * spacing for i in range(len(L_vals))])
ax.set_yticklabels([str(L) for L in L_vals])
ax.set_ylabel(r"$L$")
ax.set_xlabel(f"${args.x_axis}$")
ax.set_ylim(-0.1 * spacing, (len(L_vals) - 1) * spacing + 1.2 * R_max)

xpad = 0.01
ax.set_xlim(-xpad, 0.065)

# A colourbar carries the T series, which the stacked ridges leave no room for
# in a legend.
if cnorm is not None and args.series_axis:
    sm = mpl.cm.ScalarMappable(norm=cnorm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, pad=0.02, fraction=0.05)
    cbar.set_label(f"${args.series_axis}$")
elif args.series_axis:
    ax.legend([fmt_val(v) for v in series_vals], title=args.series_axis,
              fontsize=8)

plt.tight_layout()

# if args.output:
#     fig.savefig(args.output, dpi=150, bbox_inches="tight")
#     print(f"Saved to {args.output}")
# else:
plt.show()
