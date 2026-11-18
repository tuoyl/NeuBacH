"""
plotstyle.py -- one consistent, colour-blind-checked look for all figures.

Palette: fixed categorical order (blue, orange, aqua, yellow, magenta, green,
violet, red), single-hue blue sequential ramp, blue<->red diverging ramp with a
neutral grey midpoint.  Thin 2-px lines, solid hairline grid, no dual axes.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED = CAT

SEQ_BLUE = LinearSegmentedColormap.from_list(
    "seq_blue", ["#eef4fc", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"])
DIV_BLUE_RED = LinearSegmentedColormap.from_list(
    "div_blue_red", ["#0d366b", "#256abf", "#6da7ec", "#cde2fb", "#f0efec",
                     "#f6cfc9", "#ee8f89", "#e34948", "#8f1f1f"])


def apply():
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "font.family": "DejaVu Sans", "font.size": 9.5,
        "axes.edgecolor": AXIS, "axes.linewidth": 0.8,
        "axes.labelcolor": INK2, "axes.titlecolor": INK, "axes.titlesize": 10.5,
        "axes.titleweight": "bold", "axes.titlelocation": "left",
        "xtick.color": MUTED, "ytick.color": MUTED,
        "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "grid.linestyle": "-",
        "axes.axisbelow": True, "axes.spines.top": False, "axes.spines.right": False,
        "lines.linewidth": 2.0, "lines.solid_capstyle": "round", "lines.solid_joinstyle": "round",
        "legend.frameon": False, "legend.fontsize": 8.5, "legend.labelcolor": INK2,
        "savefig.dpi": 200, "figure.dpi": 100,
        "axes.prop_cycle": matplotlib.cycler(color=CAT),
    })


def note(fig, text, y=0.005):
    """small grey footnote under a figure (assumptions / forecast disclaimer)."""
    fig.text(0.01, y, text, fontsize=7.2, color=MUTED, ha="left", va="bottom")
