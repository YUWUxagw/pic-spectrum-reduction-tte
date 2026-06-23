"""Figure 3: Publication-quality Forest Plot"""

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# ==========================================================
# DATA
# ==========================================================

results = [
    ("Primary: IPCW-weighted Cox", 0.392, 0.322, 0.477, '#2C3E50', 'primary'),

    ("S1: Exclude CoNS outcomes", 0.377, 0.300, 0.474, '#2980B9', 'sens'),
    ("S2: Allowable gap = 12 h", 0.392, 0.322, 0.477, '#2980B9', 'sens'),
    ("S3: Exclude unrecognized baseline resistance", 0.115, 0.072, 0.184, '#2980B9', 'sens'),
    ("S4: First ICU episode only", 0.381, 0.312, 0.466, '#2980B9', 'sens'),
    ("S5: Exclude ICD-only infection context", 0.391, 0.322, 0.475, '#2980B9', 'sens'),
    ("S6: Exclude single-drug stop", 0.262, 0.196, 0.352, '#2980B9', 'sens'),
    ("S7: Follow-up from end of grace period", 0.090, 0.053, 0.151, '#2980B9', 'sens'),

    ("Multivariable-adjusted Cox (24 covariates)", 0.380, 0.311, 0.465, '#1E8449', 'adj'),
    ("PS-IPTW (propensity score)", 0.377, 0.310, 0.459, '#1E8449', 'adj'),
    ("IPCW × IPTW (doubly weighted)", 0.383, 0.314, 0.467, '#1E8449', 'adj'),
    ("Fine-Gray subdistribution HR", 0.396, 0.294, 0.535, '#1E8449', 'adj'),
]

n = len(results)

# ==========================================================
# FIGURE LAYOUT
# ==========================================================

fig = plt.figure(
    figsize=(12, 7),
    facecolor="white"
)

gs = fig.add_gridspec(
    1,
    3,
    width_ratios=[3.0, 5.0, 1.5],
    wspace=0.02
)

ax_label = fig.add_subplot(gs[0, 0])
ax_forest = fig.add_subplot(gs[0, 1])
ax_num = fig.add_subplot(gs[0, 2])

# ==========================================================
# LABEL COLUMN
# ==========================================================

ax_label.set_xlim(0, 1)
ax_label.set_ylim(-0.8, n + 0.8)
ax_label.axis("off")

ax_label.text(
    0,
    n + 0.3,
    "Analysis",
    fontsize=10,
    fontweight="bold"
)

# ==========================================================
# NUMBER COLUMN
# ==========================================================

ax_num.set_xlim(0, 1)
ax_num.set_ylim(-0.8, n + 0.8)
ax_num.axis("off")

ax_num.text(
    0,
    n + 0.3,
    "HR (95% CI)",
    fontsize=10,
    fontweight="bold"
)

# ==========================================================
# FOREST AXIS
# ==========================================================

ax_forest.set_ylim(-0.8, n + 0.8)
ax_forest.set_xscale("log")

ax_forest.set_xlim(0.06, 0.65)

ax_forest.set_xticks(
    [0.08, 0.10, 0.15, 0.20, 0.30, 0.40, 0.60]
)

ax_forest.set_xticklabels(
    ["0.08", "0.10", "0.15", "0.20", "0.30", "0.40", "0.60"]
)

ax_forest.set_yticks([])

ax_forest.set_xlabel(
    "Hazard Ratio",
    fontsize=11
)

# ==========================================================
# REFERENCE LINE
# ==========================================================

ax_forest.axvline(
    1.0,
    color="#E74C3C",
    lw=1.0,
    ls="--",
    alpha=0.5
)

# ==========================================================
# PLOT ROWS
# ==========================================================

for i, (label, hr, lo, hi, color, kind) in enumerate(results):

    y = n - i - 1

    if kind == "primary":
        marker = "D"
        ms = 9
        lw = 2.8
        fw = "bold"
    else:
        marker = "s"
        ms = 6.5
        lw = 1.8
        fw = "normal"

    # --------------------------
    # Left labels
    # --------------------------

    ax_label.text(
        0.00,
        y,
        label,
        fontsize=9,
        va="center",
        fontweight=fw
    )

    # --------------------------
    # Forest plot
    # --------------------------

    ax_forest.plot(
        [lo, hi],
        [y, y],
        color=color,
        lw=lw,
        alpha=0.9,
        solid_capstyle="round"
    )

    ax_forest.plot(
        hr,
        y,
        marker=marker,
        color=color,
        markersize=ms,
        markeredgecolor="white",
        markeredgewidth=0.8
    )

    # --------------------------
    # Right numbers
    # --------------------------

    ax_num.text(
        0.0,
        y,
        f"{hr:.3f} ({lo:.3f}–{hi:.3f})",
        fontsize=8.5,
        family="monospace",
        va="center"
    )

# ==========================================================
# SECTION HEADERS
# ==========================================================

sens_y = n - 1.6

ax_label.text(
    0,
    sens_y,
    "Sensitivity analyses",
    fontsize=9,
    fontweight="bold",
    color="gray",
    style="italic"
)

adj_y = 2.4

ax_label.text(
    0,
    adj_y,
    "Adjusted and propensity-score analyses",
    fontsize=9,
    fontweight="bold",
    color="gray",
    style="italic"
)

# ==========================================================
# DIVIDER
# ==========================================================

for ax in [ax_label, ax_forest, ax_num]:

    ax.axhline(
        y=3.5,
        color="#E5E7E9",
        lw=1
    )

# ==========================================================
# FAVOR DIRECTION
# ==========================================================

ax_forest.text(
    0.07,
    n + 1.0,
    "← Favors Spectrum Reduction",
    fontsize=9,
    fontweight="bold",
    color="#1E8449"
)

ax_forest.text(
    0.33,
    n + 1.0,
    "Favors Continue Broad →",
    fontsize=9,
    fontweight="bold",
    color="#C0392B"
)

# ==========================================================
# STYLE
# ==========================================================

for sp in ["top", "right", "left"]:
    ax_forest.spines[sp].set_visible(False)

ax_forest.spines["bottom"].set_color("#CCCCCC")

ax_forest.grid(
    axis="x",
    alpha=0.25,
    linewidth=0.4
)

ax_forest.set_axisbelow(True)

# ==========================================================
# LEGEND
# ==========================================================

legend_elements = [

    Line2D(
        [0],[0],
        marker='D',
        color='w',
        markerfacecolor='#2C3E50',
        markersize=8,
        label='Primary analysis'
    ),

    Line2D(
        [0],[0],
        marker='s',
        color='w',
        markerfacecolor='#2980B9',
        markersize=6,
        label='Sensitivity analyses'
    ),

    Line2D(
        [0],[0],
        marker='s',
        color='w',
        markerfacecolor='#1E8449',
        markersize=6,
        label='Adjusted analyses'
    )
]

ax_forest.legend(
    handles=legend_elements,
    loc="lower left",
    fontsize=8,
    frameon=False
)

# ==========================================================
# TITLE
# ==========================================================

fig.suptitle(
    "Figure 2. Forest Plot of Hazard Ratios for Resistant-Organism Detection",
    fontsize=13,
    fontweight="bold",
    y=0.98
)

# ==========================================================
# SAVE
# ==========================================================

fig.savefig(
    r"F:\test\Figure2_Publication_Forest.tiff",
    dpi=600,
    bbox_inches="tight",
    facecolor="white"
)


plt.close()

print("Figure 2 saved")