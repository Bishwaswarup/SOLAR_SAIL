"""
paperstyle.py — plain journal-figure style for matplotlib.

White background, black marks distinguished by dash pattern rather than colour,
serif text, full axes box with inward ticks, no decorative elements.  Import and
call `use()` before creating a figure.

    from src.paperstyle import use, panel_label
    use()
"""

from __future__ import annotations

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# Dash patterns, in the order a figure should consume them.  Distinguishing
# series by dash rather than colour keeps the figure legible in greyscale print.
DASHES = [
    (None, None),        # solid
    (5, 2),              # dashed
    (1, 1.6),            # dotted
    (6, 2, 1, 2),        # dash-dot
    (3, 1.4, 1, 1.4),    # short dash-dot
    (8, 2, 1, 2, 1, 2),  # dash-dot-dot
]

MARKERS = ['o', 's', '^', 'D', 'v', 'x']


def use(font_size: float = 9.0, serif: bool = True) -> None:
    """Apply the paper style globally."""
    plt.rcParams.update({
        'figure.facecolor':   'white',
        'figure.dpi':          150,
        'savefig.facecolor':  'white',
        'savefig.dpi':         300,
        'savefig.bbox':       'tight',

        'font.family':        'serif' if serif else 'sans-serif',
        'font.serif':         ['DejaVu Serif', 'Times New Roman', 'Times'],
        'font.size':           font_size,
        'mathtext.fontset':   'dejavuserif' if serif else 'dejavusans',

        'axes.facecolor':     'white',
        'axes.edgecolor':     'black',
        'axes.linewidth':      0.8,
        'axes.labelsize':      font_size,
        'axes.titlesize':      font_size,
        'axes.titleweight':   'normal',
        'axes.grid':           False,
        'axes.spines.top':     True,
        'axes.spines.right':   True,
        'axes.prop_cycle':    plt.cycler(color=['black']),

        'lines.linewidth':     1.0,
        'lines.markersize':    3.5,
        'lines.color':        'black',

        'xtick.direction':    'in',
        'ytick.direction':    'in',
        'xtick.top':           True,
        'ytick.right':         True,
        'xtick.major.size':    3.5,
        'ytick.major.size':    3.5,
        'xtick.minor.size':    2.0,
        'ytick.minor.size':    2.0,
        'xtick.major.width':   0.8,
        'ytick.major.width':   0.8,
        'xtick.labelsize':     font_size - 0.5,
        'ytick.labelsize':     font_size - 0.5,

        'legend.frameon':      False,
        'legend.fontsize':     font_size - 0.5,
        'legend.handlelength': 2.6,
        'legend.borderpad':    0.2,
        'legend.labelspacing': 0.3,
    })


def dashed(ax, x, y, i: int = 0, lw: float = 1.0, **kw):
    """Plot black with the i-th dash pattern."""
    d = DASHES[i % len(DASHES)]
    line, = ax.plot(x, y, color=kw.pop('color', 'black'), lw=lw, **kw)
    if d[0] is not None:
        line.set_dashes(list(d))
    return line


def panel_label(ax, text: str, dx: float = -0.15, dy: float = 1.04,
                size: float = 9.5):
    """Put a plain '(a)' style label at the top-left, outside the axes."""
    ax.text(dx, dy, text, transform=ax.transAxes, ha='left', va='bottom',
            fontsize=size)


def thin_guide(ax, *, y=None, x=None, label: str = None,
               label_x: float = 0.02, label_dy: float = 1.03,
               size: float = 7.5):
    """A light reference line (threshold, asymptote) with an optional label."""
    if y is not None:
        ax.axhline(y, color='0.45', lw=0.6, dashes=[2, 2], zorder=0)
        if label:
            ax.text(label_x, y * label_dy, label, transform=ax.get_yaxis_transform(),
                    fontsize=size, color='0.3', va='bottom')
    if x is not None:
        ax.axvline(x, color='0.45', lw=0.6, dashes=[2, 2], zorder=0)
        if label:
            ax.text(x, label_dy, label, transform=ax.get_xaxis_transform(),
                    fontsize=size, color='0.3', ha='center', va='bottom')
