"""Figure 1: STROBE Flow Diagram — publication-ready layout"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

# ---------- palette ----------
NAVY      = '#1B4F72'
LT_BLUE   = '#EBF5FB'
LT_GREEN  = '#EAFAF1'
LT_RED    = '#FDEDEC'
LT_GRAY   = '#F4F6F6'
E_BLUE    = '#5499C7'
E_GRN     = '#52BE80'
E_RED     = '#E6736B'
E_GRAY    = '#AEB6BF'
TEXT      = '#1C2833'
RED       = '#C0392B'
ARR       = '#5499C7'
BG        = '#FFFFFF'

FONT = 'DejaVu Sans'

fig, ax = plt.subplots(figsize=(11, 12.7), facecolor=BG)
ax.set_facecolor(BG)
ax.set_xlim(0, 11)
ax.set_ylim(1.55, 14.25)
ax.axis('off')

CX = 4.7          # main-flow column centre
SX = 9.0          # exclusion-box column centre

# ---------- low-level helpers ----------

def box(x, y, w, h, txt, color=LT_BLUE, edge=E_BLUE, fs=10, fc=TEXT,
        weight='bold', lw=1.2, z=2):
    r = FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                        boxstyle='round,pad=0.045,rounding_size=0.09',
                        facecolor=color, edgecolor=edge, linewidth=lw,
                        zorder=z)
    ax.add_patch(r)
    ax.text(x, y, txt, ha='center', va='center', fontsize=fs,
            fontweight=weight, color=fc, fontfamily=FONT,
            zorder=z + 1, linespacing=1.45)
    return dict(x=x, y=y, w=w, h=h)


def vline(b_top, b_bottom, color=ARR, lw=1.6):
    """Arrow strictly between the bottom edge of b_top and top edge of b_bottom."""
    y1 = b_top['y'] - b_top['h'] / 2
    y2 = b_bottom['y'] + b_bottom['h'] / 2
    ax.annotate('', xy=(CX, y2), xytext=(CX, y1),
                arrowprops=dict(arrowstyle='-|>', color=color, lw=lw,
                                 mutation_scale=16, shrinkA=0, shrinkB=0))


def branch(y, side_box, color=E_RED, lw=1.3):
    """Horizontal branch from a point on the main vertical line to the
    left edge of a side (exclusion) box. Anchored exactly at the box edge."""
    x_end = side_box['x'] - side_box['w'] / 2
    ax.annotate('', xy=(x_end, y), xytext=(CX, y),
                arrowprops=dict(arrowstyle='-|>', color=color, lw=lw,
                                 mutation_scale=13, shrinkA=0, shrinkB=0))
    # small tick marking the branch point on the main line
    ax.plot(CX, y, marker='o', markersize=2.6, color=color, zorder=4)


def exbox(y, w, h, txt, x=SX, fs=8.3):
    r = FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                        boxstyle='round,pad=0.045,rounding_size=0.08',
                        facecolor=LT_RED, edgecolor=E_RED, linewidth=1.0,
                        zorder=2)
    ax.add_patch(r)
    ax.text(x, y, txt, ha='center', va='center', fontsize=fs,
            color=RED, fontfamily=FONT, zorder=3, linespacing=1.5)
    return dict(x=x, y=y, w=w, h=h)


# ================= TITLE =================
ax.text(CX, 13.85, 'Figure 1. Cohort construction and trial emulation (STROBE flow diagram)',
        ha='center', fontsize=12.3, fontweight='bold', color=NAVY, fontfamily=FONT)

# ================= ROW 1 =================
r1 = box(CX, 13.15, 4.8, 0.62,
         'All ICU admissions assessed for eligibility\n(n = 13,729 stays)',
         color=LT_GRAY, edge=E_GRAY, fs=10)

# ================= ROW 2 =================
r2 = box(CX, 11.85, 4.8, 0.9,
         'Eligible episodes\n(n = 7,483 episodes; 7,119 patients)\nCandidate person-time trials\n(n = 22,449)',
         fs=10)

ex1 = exbox(y=(r1['y'] - r1['h']/2 + r2['y'] + r2['h']/2)/2, w=3.0, h=0.6,
            txt='Excluded\nICU length of stay < 48 h\n(n = 6,246)')
vline(r1, r2)
branch(ex1['y'], ex1)

# ================= ROW 3 =================
r3 = box(CX, 10.15, 4.8, 0.62,
         'Eligible trials: n = 3,771  (16.8%)',
         fs=10)

ex2 = exbox(y=(r2['y'] - r2['h']/2 + r3['y'] + r3['h']/2)/2, w=3.3, h=1.65,
            txt=('Excluded trials (n = 18,678)\n'
                 'No broad-spectrum antibiotics:  9,357\n'
                 'Not in ICU at trial start:  4,466\n'
                 'No infection context:  3,323\n'
                 'Death before landmark:  1,010\n'
                 'Baseline antimicrobial resistance:  522'),
            fs=8.0)
vline(r2, r3)
branch(ex2['y'], ex2)

# ================= ROW 4 =================
r4 = box(CX, 8.65, 4.8, 0.62,
         'Grace-period adherence assessed\nAssignable trials: n = 3,691',
         color=LT_GRAY, edge=E_GRAY, fs=10)

ex3 = exbox(y=(r3['y'] - r3['h']/2 + r4['y'] + r4['h']/2)/2, w=3.0, h=0.55,
            txt='Excluded\nDied during grace period\n(n = 80)')
vline(r3, r4)
branch(ex3['y'], ex3)

# ================= ROW 5: design step (highlighted) =================
r5 = box(CX, 7.35, 4.8, 0.66,
         'Clone–censor–weight design\n3,691 trials × 2 strategies = 7,382 clones',
         color=NAVY, edge=NAVY, fc='white', fs=10.3)
vline(r4, r5)

# ================= SPLIT INTO TWO ARMS =================
y_split  = r5['y'] - r5['h']/2 - 0.30
LX, RX   = 2.55, 6.85
arm_y    = 5.95

ax.annotate('', xy=(CX, y_split), xytext=(CX, r5['y'] - r5['h']/2),
            arrowprops=dict(arrowstyle='-', color=ARR, lw=1.6))
ax.plot([LX, RX], [y_split, y_split], color=ARR, lw=1.6, solid_capstyle='round')
for bx in (LX, RX):
    ax.annotate('', xy=(bx, arm_y + 0.46), xytext=(bx, y_split),
                arrowprops=dict(arrowstyle='-|>', color=ARR, lw=1.6,
                                 mutation_scale=16, shrinkA=0, shrinkB=0))

armA = box(LX, arm_y, 3.6, 0.92,
           'Strategy A — Spectrum reduction\nAdherent: 591  (16%)\nCensored: 3,100  (84%)',
           color=LT_GREEN, edge=E_GRN, fs=9.3)
armB = box(RX, arm_y, 3.6, 0.92,
           'Strategy B — Continue broad-spectrum\nAdherent: 2,802  (76%)\nCensored: 889  (24%)',
           color=LT_RED, edge=E_RED, fs=9.3)

# ================= MERGE =================
y_merge = arm_y - armA['h']/2 - 0.30
ax.annotate('', xy=(LX, y_merge), xytext=(LX, arm_y - armA['h']/2),
            arrowprops=dict(arrowstyle='-', color=ARR, lw=1.6))
ax.annotate('', xy=(RX, y_merge), xytext=(RX, arm_y - armB['h']/2),
            arrowprops=dict(arrowstyle='-', color=ARR, lw=1.6))
ax.plot([LX, RX], [y_merge, y_merge], color=ARR, lw=1.6, solid_capstyle='round')

r7y = 3.85
ax.annotate('', xy=(CX, r7y + 0.43), xytext=(CX, y_merge),
            arrowprops=dict(arrowstyle='-|>', color=ARR, lw=1.6,
                             mutation_scale=16, shrinkA=0, shrinkB=0))

r7 = box(CX, r7y, 5.6, 0.86,
         '28-day incident resistant-organism detections: 505 events\n(145 in spectrum-reduction arm vs. 360 in continue-broad arm)',
         fs=10)

# ================= FOOTER: sensitivity landmarks =================
ax.plot([2.0, 10.65], [2.95, 2.95], color=E_GRAY, lw=0.8)
ax.text(CX, 2.55,
        'Sensitivity analyses — alternative grace-period landmarks',
        ha='center', fontsize=9, fontweight='bold', color='#566573', fontfamily=FONT)
ax.text(CX, 2.10,
        '48 h: n = 1,390 (38%)        72 h: n = 1,229 (33%)        96 h: n = 1,072 (29%)',
        ha='center', fontsize=9.3, color='#7F8C8D', fontfamily=FONT)

plt.tight_layout(pad=0.3)
plt.savefig(r'F:\test\Figure1_STROBE.tiff', dpi=300, bbox_inches='tight', facecolor=BG, pil_kwargs={'compression': 'tiff_lzw'})
plt.close()
print('Done')
