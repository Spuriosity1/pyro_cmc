#!/bin/env python3

"""Supplementary ratio figure (fig1): S(q_PG1)/S(q_B) vs K for every system size
L at once, markers sized by L and coloured by temperature, with a (K, T)
phase-boundary inset. The maintext L=20 companion lives in
plot_final_ratio_L20.py; both share the data pipeline in plot_ratio.py.

The spiral wavevector Q has a three-arm star {(0,0,Q), (Q,0,0), (0,Q,0)}. For
each file the three peaks are ranked by intensity independently at every
temperature into Q1 >= Q2 >= Q3 (regardless of the order they appear in the
file), and the ratio S(Q2)/S(Q1) in [0, 1] is plotted (Q1 is the dominant peak,
Q2 the subdominant one that may be small). A perfectly single-Q spiral drives
the ratio toward 0; a symmetric multi-Q / paramagnetic state drives it toward 1.
"""

import glob

from plot_ratio import *
from plot_helpers import minor_ticks, mark_series_on_cbar


use_tex_serif()

args = parse_final_args()

output="/home/als/Projects/016_spiral_cmc/fig/200_plot_ratio.pdf"

# ---- load, expand over T, accumulate over the binned params ----
points, title_T, fixed = accumulate_points(args)

# View window: trim the data to the plotted kappa range here so that no markers,
# arrows, or their labels are rendered outside the axes (savefig's tight bbox
# would otherwise expand to include anything drawn past set_xlim).
XMIN, XMAX = -0.008, 0.105
points = [p for p in points if XMIN <= p['x'] <= XMAX]

# Marker area encodes the system size L: ms grows steeply from the smallest L
# present so the finite-size trend reads off the point sizes.
all_L = sorted(set(p['L'] for p in points))
L_min = min(all_L) if all_L else 0


def ms_of(L):
    return 0.2*(-4 + 1*L)
    # return 1 + 0.05 * (np.asarray(L) - L_min) ** 3


L_hi = max(all_L) if all_L else 0


def offset_of(L):
    # Horizontal fan offset per L: the largest L sits at the true kappa, smaller
    # L shifted left, so points that overlap at each kappa spread out by size.
    return 0.0003 * (np.asarray(L) - L_hi)


# ---- derive series values + colours from the accumulated points ----
cmap=plt.colormaps['coolwarm']
series_vals = sorted(set(p['ser'] for p in points),
                     key=lambda v: (v is None, v))
series_color, cnorm, cmap = series_colors(series_vals, args.series_axis, cmap)

series_data = {v: {'x': [], 'L': [], 'R': [], 'SE': []} for v in series_vals}
for p in points:
    d = series_data[p['ser']]
    d['x'].append(p['x'])
    d['L'].append(p['L'])
    d['R'].append(p['R'])
    d['SE'].append(p['SE'])

pt=0.0138888889

fig, ax = plt.subplots(figsize=(500*pt, 220*pt))
ax.set_xlabel(r"$S(S+1)\kappa/|J_1|$")
ax.set_ylabel(r"$\mathcal{S}(\bm{q}_{\rm PG1})\,/\,\mathcal{S}(\bm{q}_{\rm B})$")
ax.axhline(1.0, color='k', lw=0.8, ls='--', alpha=0.5)


def fmt_val(v):
    return f"{v:g}" if isinstance(v, float) else str(v)

plotted_any = False
for ser in series_vals:
    d = series_data[ser]
    if not d['x']:
        continue
    order = np.argsort(d['x'])
    x_sorted = np.array(d['x'])[order]
    R_sorted = np.array(d['R'])[order]
    L_sorted = np.array(d['L'])[order]
    SE_sorted = np.array(d['SE'])[order]


    label = fmt_val(ser) if ser is not None else None
    color = series_color[ser]

    offset = offset_of(L_sorted)

    ms_sorted = ms_of(L_sorted)

    ax.errorbar(x_sorted+offset, R_sorted, yerr=SE_sorted,
                fmt=',', color=color, ms=0,
                capsize=0, elinewidth=1, zorder=-10)
    ax.scatter(x_sorted+offset, R_sorted,
                marker='s', label=label, color=color, s=ms_sorted)
    

    mticks = set()
    for x, o, L in zip(x_sorted, offset, L_sorted):
        c=1- L/20
        mticks.add(x+o)
    
    ax.set_xticks(list(mticks),minor=True)
        # ax.axvline(x+o,lw=0.2,ls='-', color=(c,c,c), zorder=-22)

    if not plotted_any:
        plotted_any = True

if not plotted_any:
    sys.exit("No data to plot — check the requested -x/-s parameters and that "
             "the Q tag exists in the filenames.")

# series_leg = ax.legend( fontsize=6,
#                        loc='upper left')
# ax.add_artist(series_leg)


sm = mpl.cm.ScalarMappable(norm=cnorm, cmap=cmap)
cbar = fig.colorbar(sm, ax=ax, pad=0.02, fraction=0.05)

cbar.set_label(r"$T/S(S+1)|J_1|$")
mark_series_on_cbar(cbar, series_vals, edgecolors='k')
minor_ticks(ax,0.01, 'x')
minor_ticks(ax,0.001, 'y')

# Fine-line L scale: under every kappa column, a faint vertical line per L at
# the same horizontal offset as the fanned points, terminating just above the
# tallest (highest-S, low-T) point in that column so the lines stay clear of the
# inset. The L values are labelled once, on the column nearest kappa=0.06.
y_bot = -0.01
label_kappa = 0.06

col_L = {}     # kappa -> set of L present
col_top = {}   # kappa -> max R (finite) over that column
for p in points:
    col_L.setdefault(p['x'], set()).add(p['L'])
    if np.isfinite(p['R']):
        col_top[p['x']] = max(col_top.get(p['x'], p['R']), p['R'])

cols = sorted(col_top)
label_col = min(cols, key=lambda c: abs(c - label_kappa)) if cols else None
for c in cols:
    for L in sorted(col_L[c]):
        top = col_top[c]
        il = all_L.index(L)
        xl = c + float(offset_of(L))
        lw, ltop = (0.7, top+0.0015) if il % 3 == 1 else (0.5, top+0.0010)
        ax.plot([xl, xl], [y_bot, ltop], lw=lw, color='0.55', zorder=-20)
        if c == label_col and il % 3 == 1:
            ax.text(xl, top + 0.002, str(L), ha='center', va='bottom',
                    rotation=90, fontsize=5, color='0.3')

# A single 'L' header centred over the scale's numbering, above the rotated L
# values of the labelled column.
if label_col is not None:
    x_hdr = label_col + 0.5 * float(offset_of(min(all_L)))
    ax.text(x_hdr, col_top[label_col] + 0.007, r"$L$",
            ha='center', va='bottom', fontsize=7, color='0.3')

    y_top = col_top[label_col]+0.006
    dx=0.003

    x_lo, x_hi = x_hdr -dx, x_hdr + dx
    ax.annotate('', xy=(x_hi, y_top), xytext=(x_lo, y_top),
                    arrowprops=dict(arrowstyle='->', color='k', lw=0.6))

# # Over the coldest (T=1e-5) points at each kappa, float a single black arrow
# # spanning the full L range (smallest -> largest) to show the direction the
# # offset fans the points, its tail/head labelled with those L.
# if series_vals and all_L:
#     cold = min(series_vals, key=lambda v: abs(float(v) - 1e-5))
#     cd = series_data[cold]
#     L_lo_g, L_hi_g = min(all_L), max(all_L)
#     dx_lo, dx_hi = float(offset_of(L_lo_g)), float(offset_of(L_hi_g))
#     by_x = {}
#     for x, L, R in zip(cd['x'], cd['L'], cd['R']):
#         by_x.setdefault(x, []).append(R)
#     for x, Rs in by_x.items():
#         ytop = np.nanmax(Rs) + 0.0025
#         x_lo, x_hi = x + dx_lo, x + dx_hi
#         ax.annotate('', xy=(x_hi, ytop), xytext=(x_lo, ytop),
#                     arrowprops=dict(arrowstyle='->', color='k', lw=0.6))
#         ax.text(x_lo, ytop, str(L_lo_g), ha='right', va='center',
#                 fontsize=6, color='k')
#         ax.text(x_hi, ytop, str(L_hi_g), ha='left', va='center',
#                 fontsize=6, color='k')
#
# Inset (top left): the (K, T) boundary where the subdominant PG1 peak vanishes,
# extrapolated per T — the polished plot_KT_phasedia fig2, dropped in here.
axins = ax.inset_axes([0.1, 0.57, 0.35, 0.38])

# The inset uses *all* available L (run13_L=*.csv), not just the subset drawn on
# the main axes, and shades each by L: L=20 black, smallest L pale grey.
_ins_parser = argparse.ArgumentParser()
provide_selector_args(_ins_parser)
_ins_files = sorted(glob.glob(
    '/home/als/Projects/016_spiral_cmc/out_db/run13_L=*.csv'))
args_all = _ins_parser.parse_args([*_ins_files, '-x', 'K', '-s', 'T',
                                   '--bin', 'J3', 'seed'])
boundary = phase_boundary(args_all, 0.003)

ins_L = {int(dict(tr)['L']): tr for tr in boundary}
if ins_L:
    L_lo, L_hi = min(ins_L), max(ins_L)

    def _gray(L):
        frac = (L - L_lo) / (L_hi - L_lo) if L_hi > L_lo else 1.0
        g = 0.8 * (1 - frac)          # L_lo -> pale grey, L_hi -> black
        return (g, g, g)
    ins_colors = {tr: _gray(L) for L, tr in ins_L.items()}
else:
    ins_colors = None

draw_phase_boundary(axins, boundary, colors=ins_colors, marker='+', ms=2, lw=0.8,
                    labeler=lambda tr: r"$Q=1/2$ $L$=%s" % fmt_val(dict(tr)['L']))

# Overlay the run14, L=18 boundary in red for comparison with the run13 ramp.
_p14 = argparse.ArgumentParser()
provide_selector_args(_p14)
args14 = _p14.parse_args(
    ['/home/als/Projects/016_spiral_cmc/out_db/run14_L=18.csv',
     '-x', 'K', '-s', 'T', '--bin', 'J3', 'seed'])
boundary14 = phase_boundary(args14, 0.003)
draw_phase_boundary(axins, boundary14,
                    colors={tr: 'red' for tr in boundary14}, marker='o', ms=1, lw=0.8,
                    labeler=lambda tr: r"$Q=1/9$ $L$=18")

# Shade the double-Q region below the L=20 boundary (K > x0 — the subdominant
# peak is present), leaving the single-Q region above it white. Lock the limits
# first so the fill, extended flat to the axes edges, doesn't rescale them.
tr20 = ins_L.get(20) if ins_L else None
if tr20 is not None:
    _xlim, _ylim = axins.get_xlim(), axins.get_ylim()
    b20 = boundary[tr20]
    xs = np.concatenate(([_xlim[0]], b20['x0'], [_xlim[1]]))
    ys = np.concatenate(([b20['T'][0]], b20['T'], [b20['T'][-1]]))
    axins.fill_between(xs, _ylim[0], ys, color='1', hatchcolor='0', zorder=-5, hatch='......')
    axins.set_xlim(_xlim)
    axins.set_ylim(0, 0.015)
    tbox = dict(boxstyle='round,pad=0.15', facecolor='1', edgecolor='none')
    axins.text(0.72, 0.22, r"double-$q$", transform=axins.transAxes,
               ha='center', va='center', fontsize=6, bbox=tbox, zorder=6)
    axins.text(0.55, 0.78, r"single-$q$", transform=axins.transAxes,
               ha='center', va='center', fontsize=6, bbox=tbox, zorder=6)

# Dashed linear-fit extrapolation off the hot end of every trace, continued up
# to the top of the T axis (fit from each trace's last few points).
target_T = axins.get_ylim()[1]


def _extrapolate(bnd, color_of, n=3):
    for tr, b in bnd.items():
        T, x0 = np.asarray(b['T']), np.asarray(b['x0'])
        if len(T) < 2 or target_T <= T[-1]:
            continue
        k = min(n, len(T))
        slope, intc = np.polyfit(T[-k:-1], x0[-k:-1], 1)
        Tseg = np.array([T[-2], target_T])
        axins.plot(slope * Tseg + intc, Tseg, ls='--', lw=0.8,
                   color=color_of(tr), zorder=3)


_extrapolate(boundary, lambda tr: ins_colors[tr] if ins_colors else 'k')
_extrapolate(boundary14, lambda tr: 'red', 4)

#axins.set_yscale('log')
axins.set_xlabel(r"$S(S+1)\kappa/|J_1|$", fontsize=6, labelpad=1)
axins.set_ylabel(r"$T/S(S+1)|J_1|$", fontsize=6, labelpad=1)
axins.tick_params(labelsize=5)
if boundary:
    axins.legend(fontsize=4, labelspacing=0.2, handletextpad=0.3,
                 borderpad=0.2, framealpha=0.9)

ax.set_ylim(-0.001, 0.05)
ax.set_xlim(XMIN, XMAX)


plt.tight_layout()


fig.savefig(output, dpi=150, bbox_inches="tight")
print(f"Saved to {output}")

