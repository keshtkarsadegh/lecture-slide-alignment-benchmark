"""The one heatmap drawing of the project: methods as rows, category columns in groups.

Q4's heatmap (scripts/analysis_q4.py, draw_heatmap) and the paper's Q2 and Q4 heatmaps
(scripts/paper_assets.py, draw_paper_heatmap) are all drawn here, so the approved layout
(column groups with a gap between them, a bold header over each group, the cell value
printed in the cell, the column counts with the headers) is defined in one module and
the figures cannot drift apart. The paper's version adds a fixed geometry in inches,
the Q1 rank and kind of every method, and the colour bar under the cells. matplotlib is imported inside the function, not at the top, so that
the modules importing this one (and their tests) do not need it until they draw.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional, Sequence


def column_positions(groups: Sequence[tuple[str, int, int]], gap: float) -> list[float]:
    """The x position of every column: one unit per column, `gap` between groups."""
    xpos: list[float] = []
    x = 0.0
    for _, a, b in groups:
        for _ in range(a, b + 1):
            xpos.append(x)
            x += 1
        x += gap
    return xpos


def save_options(path: Path, png_dpi: int) -> dict:
    """savefig's options: a PNG at `png_dpi`; a PDF without its creation date, so the
    same results give the same bytes and a rerun does not change a file in git."""
    if Path(path).suffix == ".pdf":
        return {"metadata": {"CreationDate": None}}
    return {"dpi": png_dpi}


def draw_heatmap(values: Sequence[Sequence[Optional[float]]], row_labels: Sequence[str],
                 col_labels: Sequence[str], groups: Sequence[tuple[str, int, int]],
                 out_paths: Sequence[Path], *, cmap: str, vmin: float, vmax: float,
                 fmt: Callable[[float], str], white_above: float, cbar_label: str,
                 title: str = "", figsize: tuple[float, float] = (8.2, 11),
                 cell_fontsize: float = 8.5, row_fontsize: float = 9,
                 col_fontsize: float = 8.5, group_fontsize: float = 10,
                 title_fontsize: float = 11, group_y: float = -3.3, gap: float = 0.35,
                 cbar_extend: str = "neither", cbar_below: bool = False,
                 png_dpi: int = 150) -> None:
    """Draw `values` (rows x columns; None = no number, a grey cell with "-") and
    save it to every path of `out_paths` (the suffix picks the format).

    `groups`: (header, first column, last column) in column order, "" for a group
    without a header. A value beyond `vmax` gets the darkest colour and its true
    number in the cell (with `cbar_extend="max"` the colour bar says so), so a
    single large cell does not wash out the scale of all the others. `group_y` is
    the header's height in row units above the first row (negative = above).
    `cbar_below` puts the colour bar under the cells instead of at their right, which
    leaves the cells more width on a narrow page.
    """
    import matplotlib
    matplotlib.use("Agg")
    matplotlib.rcParams["pdf.fonttype"] = 42
    matplotlib.rcParams["ps.fonttype"] = 42
    import matplotlib.pyplot as plt

    xpos = column_positions(groups, gap)
    fig, ax = plt.subplots(figsize=figsize)
    colours = plt.get_cmap(cmap)
    norm = plt.Normalize(vmin, vmax, clip=True)
    for i, row in enumerate(values):
        for j, v in enumerate(row):
            if v is None:
                ax.add_patch(plt.Rectangle((xpos[j] - 0.5, i - 0.5), 1, 1, color="0.9",
                                           ec="white", lw=0.8))
                ax.text(xpos[j], i, "-", ha="center", va="center", fontsize=cell_fontsize)
                continue
            ax.add_patch(plt.Rectangle((xpos[j] - 0.5, i - 0.5), 1, 1,
                                       color=colours(norm(v)), ec="white", lw=0.8))
            ax.text(xpos[j], i, fmt(v), ha="center", va="center", fontsize=cell_fontsize,
                    color="white" if v > white_above else "black")
    ax.set_xlim(-0.6, xpos[-1] + 0.6)
    ax.set_ylim(len(row_labels) - 0.5, -0.5)
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=row_fontsize)
    ax.set_xticks(xpos)
    ax.set_xticklabels(col_labels, fontsize=col_fontsize)
    ax.xaxis.tick_top()
    for g, a, b in groups:
        if g:
            ax.text((xpos[a] + xpos[b]) / 2, group_y, g, ha="center", va="bottom",
                    fontsize=group_fontsize, fontweight="bold")
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)
    sm = plt.cm.ScalarMappable(cmap=colours, norm=plt.Normalize(vmin, vmax))
    if cbar_below:
        cb = fig.colorbar(sm, ax=ax, orientation="horizontal", fraction=0.02, pad=0.015,
                          aspect=45, shrink=0.7, extend=cbar_extend)
    else:
        cb = fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.02, extend=cbar_extend)
    cb.set_label(cbar_label, fontsize=col_fontsize)
    cb.ax.tick_params(labelsize=col_fontsize)
    if title:
        fig.suptitle(title, fontsize=title_fontsize, y=0.995)
    fig.tight_layout()
    for p in out_paths:
        fig.savefig(p, **save_options(p, png_dpi))
    plt.close(fig)


# --------------------------------------------------------------------------- #
# The paper's heatmaps (scripts/paper_assets.py): one fixed geometry
# --------------------------------------------------------------------------- #

#: The paper heatmaps' geometry, in inches from the canvas's bottom-left corner. Fixed
#: (not tight_layout), so that the three paper heatmaps are identical in geometry and
#: are printed 1:1 at the journal's text width (4.69 in). The widths follow from the
#: widest texts at 7.5 pt, the smallest size the paper prints: the longest method name
#: sets the label column (the names under the top band may reach into the band's
#: bracket, which spans only the band's rows), and the cell numbers set the cell width.
#: A level name ("medium") or a unit count ("(18,315)") is wider than a cell, so the
#: level and the count over each column are set upright, side by side (head_h is the
#: room for the longest of them); the colour bar sits under the cells because at their
#: right it would take the width the cells need. The rows are as low as the 7.5 pt cell
#: numbers allow with a clear margin, so that a heatmap and its caption leave room for
#: text under them on the page (the class's \topfraction and \textfraction).
PAPER_GEOMETRY = {
    "size": (4.69, 5.30),
    "label_right": 1.52,      # row labels right-aligned here
    "rank_x": 1.62,           # Q1 rank column (centre)
    "kind_x": 1.75,           # kind glyph column (centre)
    "cells_x": 1.82,          # first cell's left edge
    "cells_right": 4.67,      # last cell's right edge
    "group_gap": 0.05,        # gap between column groups
    "cells_y": 0.55,          # bottom of the last row
    "row_h": 0.130,
    "head_pad": 0.04,         # from the top row to the foot of the upright headers
    "head_h": 0.47,           # the upright headers' room (the longest is about 0.45 in)
    "cbar": (1.95, 0.36, 4.50, 0.44),  # x0, y0, x1, y1 (room for the end labels)
    "legend_y": 0.40,
}
#: The paper heatmaps' text sizes (points): nothing under 7.5, so that every text
#: prints at 7.5 pt or more at the text width.
PAPER_PT = 7.5
PAPER_PT_GROUP = 8.0


def luminance(rgba: Sequence[float]) -> float:
    """Rec. 709 luma of an sRGB colour (0 black .. 1 white): the cell text turns white
    below 0.5, where black text would lose contrast."""
    r, g, b = rgba[:3]
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def paper_columns(groups: Sequence[tuple[str, int, int]], g: dict) -> tuple[list[float], float]:
    """The left edge of every cell column and the cell width: the columns of each group
    side by side, `group_gap` between groups, filling cells_x .. cells_right."""
    n_cols = groups[-1][2] + 1
    width = (g["cells_right"] - g["cells_x"] - g["group_gap"] * (len(groups) - 1)) / n_cols
    xs, x = [], g["cells_x"]
    for _, a, b in groups:
        for _ in range(a, b + 1):
            xs.append(x)
            x += width
        x += g["group_gap"]
    return xs, width


def draw_paper_heatmap(values: Sequence[Sequence[float]], row_labels: Sequence[str],
                       ranks: Sequence[int], kinds: Sequence[str],
                       groups: Sequence[tuple[str, int, int]], levels: Sequence[str],
                       counts: Sequence[str], *, count_label: str, cmap: str, vmin: float,
                       vmax: float, fmt: Callable[[float], str], cbar_label: str,
                       kind_style: dict, extend: str = "neither",
                       band_rows: Optional[int] = None, geometry: Optional[dict] = None):
    """The paper's heatmap: rows = methods (in the given order), columns in groups, the
    value printed in each cell; at the left the method name, its Q1 rank (grey) and a
    glyph for its kind; above the cells the group names, a rule under each group, the
    level names and the count per column (upright: they are wider than a cell);
    under the cells the kind legend and a
    horizontal colour bar. No title: the caption carries it.

    `counts` are the column counts as plain numbers ("18,315"); they are printed in
    brackets, in grey, under the grey `count_label` ("lectures", "units"), the same in
    every heatmap.

    `band_rows`: draw a rule under that many rows (the top band of Q1) with a bracket
    and "top band" at the left; None for a figure whose rows are not in Q1 order.
    Returns the figure, unsaved, so the caller can check its layout before saving."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker
    from matplotlib.patches import Rectangle

    g = dict(PAPER_GEOMETRY, **(geometry or {}))
    W, H = g["size"]
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.axis("off")
    xs, cw = paper_columns(groups, g)
    rh = g["row_h"]
    top = g["cells_y"] + rh * len(row_labels)
    colours = plt.get_cmap(cmap)
    norm = plt.Normalize(vmin, vmax, clip=True)
    grey, dark = "#777777", "#222222"
    pt = PAPER_PT
    for i, (row, label, rank, kind) in enumerate(zip(values, row_labels, ranks, kinds)):
        yc = top - (i + 0.5) * rh
        ax.text(g["label_right"], yc, label, ha="right", va="center", fontsize=pt, color=dark)
        ax.text(g["rank_x"], yc, str(rank), ha="center", va="center", fontsize=pt, color=grey)
        c, mk = kind_style[kind]
        ax.plot([g["kind_x"]], [yc], marker=mk, ms=4.5, color=c, ls="none", mew=0)
        for x, v in zip(xs, row):
            fc = colours(norm(v))
            ax.add_patch(Rectangle((x, yc - rh / 2), cw, rh, facecolor=fc, edgecolor="white",
                                   lw=0.8))
            ax.text(x + cw / 2, yc, fmt(v), ha="center", va="center", fontsize=pt,
                    color="white" if luminance(fc) < 0.5 else "black")
    # headers, from the cells up: level and count upright side by side (the level
    # first, as the first line of a two-line upright label), then a rule under each
    # group (trimmed like \cmidrule(lr)) and the group name
    foot = top + g["head_pad"]
    rule_y = foot + g["head_h"] + 0.03
    for x, lv, n in zip(xs, levels, counts):
        xc = x + cw / 2
        if lv:
            ax.text(xc - 0.058, foot, lv, ha="center", va="bottom", fontsize=pt, color=dark,
                    rotation=90)
        ax.text(xc + 0.058 if lv else xc, foot, f"({n})", ha="center", va="bottom",
                fontsize=pt, color=grey, rotation=90)
    for name, a, b in groups:
        x0, x1 = xs[a], xs[b] + cw
        ax.text((x0 + x1) / 2, rule_y + 0.035, name, ha="center", va="bottom",
                fontsize=PAPER_PT_GROUP, fontweight="bold", color=dark)
        ax.plot([x0 + 0.02, x1 - 0.02], [rule_y] * 2, color="#444444", lw=0.6,
                solid_capstyle="butt")
    ax.text(g["label_right"], foot, count_label, ha="right", va="bottom", fontsize=pt,
            color=grey)
    ax.text(g["rank_x"], foot, "Q1 rank", ha="center", va="bottom", fontsize=pt, color=grey,
            rotation=90)
    ax.text(g["kind_x"], foot, "Kind", ha="center", va="bottom", fontsize=pt, color=grey,
            rotation=90)
    if band_rows:
        yb = top - band_rows * rh
        ax.plot([0.03, xs[-1] + cw], [yb, yb], color="#444444", lw=0.7, solid_capstyle="butt")
        ax.plot([0.20, 0.17, 0.17, 0.20], [top - 0.03, top - 0.03, yb + 0.03, yb + 0.03],
                color=grey, lw=0.6)
        ax.text(0.09, (top + yb) / 2, "top band", ha="center", va="center", fontsize=pt,
                color=grey, rotation=90)
    # the kind legend, under the labels
    x = 0.05
    for kind, (c, mk) in kind_style.items():
        ax.plot([x + 0.04], [g["legend_y"]], marker=mk, ms=4.5, color=c, ls="none", mew=0)
        t = ax.text(x + 0.10, g["legend_y"], kind, ha="left", va="center", fontsize=pt,
                    color=dark)
        fig.canvas.draw_idle()
        x += 0.10 + t.get_window_extent(fig.canvas.get_renderer()).width / fig.dpi + 0.12
    cx0, cy0, cx1, cy1 = g["cbar"]
    cax = fig.add_axes((cx0 / W, cy0 / H, (cx1 - cx0) / W, (cy1 - cy0) / H))
    sm = plt.cm.ScalarMappable(cmap=colours, norm=plt.Normalize(vmin, vmax))
    cb = fig.colorbar(sm, cax=cax, orientation="horizontal", extend=extend,
                      extendfrac=0.03)
    cb.locator = matplotlib.ticker.MaxNLocator(nbins=7, steps=[1, 2, 5, 10])
    cb.update_ticks()
    cb.outline.set_linewidth(0.6)
    cb.ax.tick_params(labelsize=pt, length=2.5, width=0.6, pad=1.5)
    cb.set_label(cbar_label, fontsize=pt, labelpad=2)
    return fig
