"""Maintext ratio figure: the L=20 view of the paramagnetic Bragg-peak ratios.

Two panels stacked on a shared K axis — the dominant S(q_PG1)/S(q_B) on top and
the weaker S(q_PG2)/S(q_B) in a shorter panel below — coloured by temperature.
The multi-L supplementary version (fig1) lives in plot_final_ratio.py; both
share the data pipeline in plot_ratio.py.
"""


from plot_ratio import *
from plot_helpers import label_subplot, minor_ticks, mark_series_on_cbar


use_tex_serif()


p = argparse.ArgumentParser()
provide_selector_args(p)
ddir="/home/als/Projects/016_spiral_cmc/out_db"
files = [f'{ddir}/run16_L{L:d}.csv' for L in (9,)]
#files = [f'{ddir}/run14_L={L:d}.csv' for L in (9, 18)]
#files = [f'{ddir}/run13_L={L:d}.csv' for L in (18,)]

target_L=9

args= p.parse_args([*files, "-x", "T", "-s", "K", "--bin", "J3", "seed"])


output = "/home/als/Projects/016_spiral_cmc/fig/200_plot_ratio_main.pdf"

# ---- load, expand over T, accumulate over the binned params ----
points, _, _ = accumulate_points(args)
# S(Q3)/S(Q1) over the same runs: num_idx=2 picks the third-ranked arm q_PG2.
points_pg2, _, _ = accumulate_points(args, num_idx=2)

all_L = sorted(set(p['L'] for p in points))


# ---- series (T) values + colours ----
# series_vals = sorted(set(p['ser'] for p in points),
#                      key=lambda v: (v is None, v))
# Match the CSV's K strings exactly: the files store K=0 as "0", not "0.0", so
# str() on the numpy floats drops the K=0 trace. '%g' reproduces "0", "0.02", …
series_vals = [f"{p:g}" for p in np.arange(0,0.2+0.001,0.02)]
print(series_vals)
series_color, cnorm, cmap = series_colors(series_vals, args.series_axis)

pt = 0.0138888889
xpad = 0.01


def series_data_at_L(pts, L):
    sd = {v: {'x': [], 'R': [], 'SE': []} for v in series_vals}
    for p in pts:
        if p['L'] != L or p['ser'] not in sd:
            continue
        d = sd[p['ser']]
        d['x'].append(p['x'])
        d['R'].append(p['R'])
        d['SE'].append(p['SE'])
    return sd


def draw_clean(target_ax, sd, ms=3, lw=0.6, scale=1):
    n = len(series_vals)
    for i, ser in enumerate(series_vals):
        d = sd[ser]
        if not d['x']:
            continue
        order = np.argsort(d['x'])
        xs = np.array(d['x'])[order]
        Rs = np.array(d['R'])[order]
        SEs = np.array(d['SE'])[order]
        # series_vals is ascending in K, so the smallest K (K=0) gets the
        # largest zorder and is drawn in the foreground.
        z = 3 + (n - i)
        eb = target_ax.errorbar(xs, scale*Rs, yerr=scale*SEs, fmt='+-', ms=2,
                                lw=lw, color=series_color[ser], capsize=1.5,
                                elinewidth=0.8, zorder=z)
        # Fade just the error bars (bar lines + caps), leaving the trace opaque.
        for artist in (*eb[1], *eb[2]):
            artist.set_alpha(0.2)


# Two stacked panels sharing the K axis: the dominant ratio S(q_PG1)/S(q_B) up
# top, and the weaker S(q_PG2)/S(q_B) in a much shorter panel below.
fig = plt.figure(figsize=(246*pt, 200*pt), layout='constrained')
gs = fig.add_gridspec(2, 1, height_ratios=[2, 1], hspace=0.04)
ax_pg1 = fig.add_subplot(gs[0])
ax_pg2 = fig.add_subplot(gs[1], sharex=ax_pg1)

# Top panel.
draw_clean(ax_pg1, series_data_at_L(points, target_L))
ax_pg1.set_ylabel(r"$\mathcal{S}(\bm{q}_{\rm PG1})\,/\,\mathcal{S}(\bm{q}_{\rm B})$")
ax_pg1.axhline(1.0, color='k', lw=0.8, ls='--', alpha=0.5)
ax_pg1.set_ylim(-0.001, 0.08)
ax_pg1.tick_params(labelbottom=False)   # x labels live on the bottom panel only

# Label the experimental ratio
ax_pg1.axhline(8e-3,color='r', ls='--',lw=0.5)
ax_pg1.axvline(0.0120,color='r', ls='--',lw=0.5)

minor_ticks(ax_pg1, 0.01, 'x')
minor_ticks(ax_pg1, 0.001, 'y')

# Bottom panel: the weaker ratio, in a shorter panel.
draw_clean(ax_pg2, series_data_at_L(points_pg2, target_L), ms=2, lw=0.8)
ax_pg2.set_ylabel(r"$\mathcal{S}(\bm{q}_{\rm PG2})\,/\,\mathcal{S}(\bm{q}_{\rm B})$")
ax_pg2.set_xlabel(r"$T/S(S+1)|J_1|$")
#ax_pg2.set_xlim(-xpad, 0.01+xpad)
ax_pg2.set_ylim(-0.00001, 0.0058)
ax_pg2.set_xscale('log')
#ax_pg2.set_yscale('log')
minor_ticks(ax_pg2, 1e-4, 'y')

for a, l in zip([ax_pg1, ax_pg2], 'ab'):
    label_subplot(fig, a, r"{\sffamily (%s)}" % l, "top right", fontsize=10)

# One colourbar spanning both panels.
sm = mpl.cm.ScalarMappable(norm=cnorm, cmap=cmap)
cbar = fig.colorbar(sm, ax=[ax_pg1, ax_pg2], pad=0.01, fraction=0.05)
cbar.set_label(f"${args.series_axis if args.series_axis!='K' else r'S(S+1)\kappa'}/|J_1|$")
mark_series_on_cbar(cbar, series_vals)



fig.savefig(output, dpi=150, bbox_inches="tight")
print(f"Saved to {output}")
