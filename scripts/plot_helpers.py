import matplotlib as mpl

def label_subplot(fig, ax, label, where='top left', hpad=0.05, vpad=0.05,
                  **kwargs):
    """Place a panel label (e.g. '(a)') in a corner of `ax`.

    The anchor is an axes corner — (0,1) top-left, (1,1) top-right, (0,0)
    bottom-left, (1,0) bottom-right — and the inward (hpad, vpad) offset is in
    *inches*, so the gap from the corner is a fixed physical distance,
    independent of the axes' or figure's size. Extra kwargs pass to ax.text."""
    parts = where.replace('-', ' ').split()
    horiz = 'right' if 'right' in parts else 'left'    # -> ha
    vert = 'bottom' if 'bottom' in parts else 'top'     # -> va

    x = 0.0 if horiz == 'left' else 1.0
    y = 1.0 if vert == 'top' else 0.0
    dx = hpad if horiz == 'left' else -hpad             # push inward
    dy = -vpad if vert == 'top' else vpad

    # dpi_scale_trans turns (dx, dy) inches into a display-space translation;
    # composed after transAxes it offsets the corner anchor by a fixed physical
    # distance regardless of this axes' size.
    offset = mpl.transforms.ScaledTranslation(dx, dy, fig.dpi_scale_trans)
    return ax.text(x, y, label, transform=ax.transAxes + offset,
                   ha=horiz, va=vert, **kwargs)


def minor_ticks(target_ax, step=0.001, axis='both'):
    """Put minor ticks every `step` on the x axis, y axis, or both."""
    if axis in ('both', 'x'):
        target_ax.xaxis.set_minor_locator(mpl.ticker.MultipleLocator(step))
    if axis in ('both', 'y'):
        target_ax.yaxis.set_minor_locator(mpl.ticker.MultipleLocator(step))


def mark_series_on_cbar(cbar, series_vals, mew=0.7, s=6, edgecolors='red'):
    """Put a small diamond on the (vertical) colourbar at every series value
    present, filled with the colourbar colour at that value. x is axes-fraction
    (centred), y is the data value, so the diamonds track the colourbar's
    (possibly log) norm. `mew` sets the diamond edge width (scatter's
    `linewidths`); `s` its area."""
    tvals = [float(v) for v in series_vals]
    trans = mpl.transforms.blended_transform_factory(
        cbar.ax.transAxes, cbar.ax.transData)
    cbar.ax.scatter([0.5] * len(tvals), tvals, transform=trans,
                    marker='s', s=s, facecolors=cbar.mappable.to_rgba(tvals),
                    edgecolors=edgecolors, linewidths=mew, zorder=5, clip_on=False)
