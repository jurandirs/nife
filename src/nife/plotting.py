"""The look of NIFE's plots, shared by every kind of explanation.

Colors have fixed roles, so that a reader learns them once:

- ``instance`` (blue) — the instance being explained, what *is*;
- ``counterfactual`` (orange) — the alternatives, what *would be*;
- ``neutral`` (gray) — context: other points, unselected marks;

plus their light ``*_wash`` versions for areas, the ink of the text and the gray of
grids and tracks. The blue and the orange are the first two hues of a palette
validated for color-vision deficiencies; text never takes the color of the marks.

Every plot of NIFE takes an optional matplotlib ``ax`` and returns the Axes it drew
on, so plots can be combined in a figure of your own and saved with
``ax.figure.savefig(...)``. They draw inside `style`, without changing your
matplotlib settings. To change a color everywhere, change `COLORS` (for example,
``nife.plotting.COLORS["counterfactual"] = "#d62728"``).

New plots use the helpers of this module: `styled` (decorator), `new_axes`,
`set_titles`, `clean_axes` and `legend_below`.
"""

from __future__ import annotations

import functools
from contextlib import contextmanager
from typing import Iterator, Optional, Sequence, Tuple

#: The colors of NIFE's plots, by role.
COLORS = {
    "instance": "#2a78d6",
    "instance_wash": "#dbe8fa",
    "counterfactual": "#eb6834",
    "counterfactual_wash": "#fbdccd",
    "neutral": "#a8a69f",
    "neutral_wash": "#efeee9",
    "ink": "#0b0b0b",
    "ink_secondary": "#52514e",
    "ink_muted": "#898781",
    "grid": "#e1e0d9",
    "axis": "#c3c2b7",
    "surface": "#ffffff",
}

_FONTS = ["Inter", "Helvetica Neue", "Helvetica", "Arial", "Liberation Sans", "DejaVu Sans"]


def _rc() -> dict:
    return {
        "font.family": "sans-serif",
        "font.sans-serif": _FONTS,
        "font.size": 9.5,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.titlecolor": COLORS["ink"],
        "axes.titlelocation": "left",
        "axes.labelsize": 9.5,
        "axes.labelcolor": COLORS["ink_secondary"],
        "axes.edgecolor": COLORS["axis"],
        "axes.linewidth": 0.8,
        "axes.facecolor": COLORS["surface"],
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": False,
        "grid.color": COLORS["grid"],
        "grid.linewidth": 0.8,
        "xtick.color": COLORS["ink_muted"],
        "ytick.color": COLORS["ink_muted"],
        "xtick.labelcolor": COLORS["ink_secondary"],
        "ytick.labelcolor": COLORS["ink_secondary"],
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "xtick.major.size": 0,
        "ytick.major.size": 0,
        "legend.frameon": False,
        "legend.fontsize": 9,
        "legend.labelcolor": COLORS["ink_secondary"],
        "text.color": COLORS["ink"],
    }


@contextmanager
def style() -> Iterator[None]:
    """Draw with NIFE's look inside a ``with`` block, without changing matplotlib's settings.

    Only the settings of NIFE's look are restored at the end, unlike
    `matplotlib.rc_context`: what else changes inside the block stays, such as the
    interactive mode that Jupyter turns on when the first figure is drawn.
    """
    import matplotlib as mpl

    settings = _rc()
    saved = {key: mpl.rcParams[key] for key in settings}
    mpl.rcParams.update(settings)
    try:
        yield
    finally:
        mpl.rcParams.update(saved)


def styled(plot):
    """Decorator: run a plotting function inside `style`."""

    @functools.wraps(plot)
    def wrapper(*args, **kwargs):
        with style():
            return plot(*args, **kwargs)

    return wrapper


def new_axes(ax=None, figsize: Tuple[float, float] = (7.0, 4.0)):
    """`ax`, or the Axes of a new figure of size `figsize` (inches).

    New figures use matplotlib's constrained layout, so that the titles, labels and
    legends drawn around the Axes fit in the figure, also when it is saved.
    """
    if ax is not None:
        return ax
    import matplotlib.pyplot as plt

    figure, ax = plt.subplots(figsize=figsize, layout="constrained")
    figure.set_facecolor(COLORS["surface"])
    return ax


def _chars_per_line(ax, fraction: float = 0.85, size: float = 9.0) -> int:
    """About how many characters of text of `size` fit in `fraction` of the width of the Axes."""
    width = ax.figure.get_figwidth() * ax.get_position().width
    return max(30, int(width * fraction * 72 / (size * 0.55)))


def wrap_parts(text: str, width: int, separator: str = " · ") -> str:
    """`text` in lines of at most `width` characters, breaking at `separator` (then at spaces)."""
    import textwrap

    lines, line = [], ""
    for part in text.split(separator):
        candidate = f"{line}{separator}{part}" if line else part
        if len(candidate) <= width or not line:
            line = candidate
        else:
            lines.append(line)
            line = part
    lines.append(line)
    return "\n".join(textwrap.fill(line, width) for line in lines)


def set_titles(ax, title: str, subtitle: Optional[str] = None, pad: float = 0) -> None:
    """A left-aligned title and, below it, a subtitle in the secondary ink, wrapped to the figure's width.

    `pad` adds space (points) between the subtitle and the Axes, for what is drawn above them.
    """
    if subtitle:
        subtitle = wrap_parts(subtitle, _chars_per_line(ax))
    lines = subtitle.count("\n") + 1 if subtitle else 0
    ax.set_title(title, loc="left", pad=10 + pad + (13 * lines + 4 if subtitle else 0), fontsize=12,
                 fontweight="bold", color=COLORS["ink"])
    if subtitle:
        ax.annotate(
            subtitle,
            xy=(0, 1),
            xycoords="axes fraction",
            xytext=(0, 6 + pad),
            textcoords="offset points",
            va="bottom",
            ha="left",
            fontsize=9,
            color=COLORS["ink_secondary"],
        )


def clean_axes(ax, grid: Optional[str] = None, left: bool = True, bottom: bool = True) -> None:
    """Hairline axes: no top or right spine, optional hairline grid on ``"x"``, ``"y"`` or ``"both"``.

    It styles the Axes itself, so that Axes created outside `style` (by the user, for
    a figure of their own) look like the others.
    """
    ax.set_facecolor(COLORS["surface"])
    for side, visible in (("top", False), ("right", False), ("left", left), ("bottom", bottom)):
        ax.spines[side].set_visible(visible)
        ax.spines[side].set_color(COLORS["axis"])
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(length=0, colors=COLORS["ink_muted"], labelcolor=COLORS["ink_secondary"], labelsize=9)
    ax.xaxis.label.set_color(COLORS["ink_secondary"])
    ax.yaxis.label.set_color(COLORS["ink_secondary"])
    if grid:
        ax.grid(True, axis=grid, color=COLORS["grid"], linewidth=0.8)
        ax.set_axisbelow(True)


def legend_below(ax, handles: Sequence, labels: Sequence[str], below: Optional[float] = None):
    """A horizontal legend under the Axes, in as many columns as fit in the figure's width.

    `below` is its distance (points) from the bottom of the Axes; by default it clears
    the tick labels and the label of the x axis.
    """
    from matplotlib.transforms import ScaledTranslation

    room = _chars_per_line(ax, fraction=1.1)
    ncol = len(labels)
    while ncol > 1 and sum(len(label) + 6 for label in labels[:ncol]) > room:
        ncol -= 1
    if below is None:
        ticks = any(label.get_text() for label in ax.get_xticklabels()) and ax.xaxis.get_visible()
        below = 8 + (16 if ticks else 0) + (14 if ax.get_xlabel() else 0)
    anchor = ax.transAxes + ScaledTranslation(0, -below / 72, ax.figure.dpi_scale_trans)
    return ax.legend(
        handles,
        labels,
        loc="upper left",
        bbox_to_anchor=(0, 0),
        bbox_transform=anchor,
        ncol=ncol,
        frameon=False,
        handletextpad=0.4,
        columnspacing=1.6,
    )


def dot(color: str, size: float = 8, marker: str = "o"):
    """A legend handle: a dot of `color` with a ring of the surface."""
    from matplotlib.lines import Line2D

    return Line2D(
        [], [], linestyle="none", marker=marker, markersize=size, markerfacecolor=color,
        markeredgecolor=COLORS["surface"], markeredgewidth=1.5,
    )


def swatch(color: str):
    """A legend handle: a square of `color`."""
    from matplotlib.patches import Patch

    return Patch(facecolor=color, edgecolor="none")


def format_value(value) -> str:
    """A value for a label: integers as they are, other numbers with 4 significant digits."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    return f"{int(number)}" if number.is_integer() and abs(number) < 1e6 else f"{number:.4g}"
