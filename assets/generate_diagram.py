"""
Run once to regenerate assets/architecture_evolution.png
    python assets/generate_diagram.py
Requires: matplotlib
"""

import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from matplotlib.gridspec import GridSpec

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'architecture_evolution.png')

# ── Palette ────────────────────────────────────────────────────────────────────
BLUE   = '#1565C0'
GREEN  = '#2E7D32'
RED    = '#B71C1C'
ORANGE = '#BF360C'
PURPLE = '#4527A0'
GRAY   = '#455A64'
TEXT   = '#1A1A1A'
SUB    = '#546E7A'
BORDER = '#B0BEC5'
WHITE  = '#FFFFFF'

# ── Figure: 28 × 20 in @ 130 dpi  →  3640 × 2600 px ──────────────────────────
FW, FH, DPI = 28, 20, 130
fig = plt.figure(figsize=(FW, FH), facecolor=WHITE, dpi=DPI)
gs  = GridSpec(2, 1, figure=fig,
               height_ratios=[1.05, 1.45],
               hspace=0.04,
               left=0.01, right=0.99, top=0.97, bottom=0.01)
ax1 = fig.add_subplot(gs[0])   # Section 1 — cards
ax2 = fig.add_subplot(gs[1])   # Section 2 — flow

for ax in (ax1, ax2):
    ax.set_facecolor(WHITE)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis('off')


# ── Helpers ───────────────────────────────────────────────────────────────────

def rbox(ax, x0, y0, w, h, ec, fc=WHITE, lw=2.5, z=2):
    ax.add_patch(FancyBboxPatch(
        (x0, y0), w, h,
        boxstyle='round,pad=0.35',
        facecolor=fc, edgecolor=ec, linewidth=lw, zorder=z))

def fbox(ax, x0, y0, w, h, fc, z=3):
    ax.add_patch(FancyBboxPatch(
        (x0, y0), w, h,
        boxstyle='round,pad=0.3',
        facecolor=fc, edgecolor='none', zorder=z))

def shadow(ax, x0, y0, w, h):
    fbox(ax, x0+0.28, y0-0.32, w, h, '#DDDDDD', z=1)

def harrow(ax, x1, y, x2, color=GRAY, lw=2.4, label='', dy=2.4):
    ax.annotate('', xy=(x2, y), xytext=(x1, y),
                arrowprops=dict(arrowstyle='->', color=color,
                                lw=lw, mutation_scale=24), zorder=9)
    if label:
        ax.text((x1+x2)/2, y+dy, label,
                ha='center', va='bottom', fontsize=9.5, color=SUB,
                bbox=dict(boxstyle='round,pad=0.25', fc=WHITE, ec='none', alpha=0.92),
                zorder=10)


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 1 — Architecture Evolution Cards
# ══════════════════════════════════════════════════════════════════════════════

ax1.text(50, 99.2, 'Architecture Evolution',
         ha='center', va='top', fontsize=22, fontweight='bold', color=TEXT, zorder=6)

CARDS = [
    dict(title='v0',      sub='Colab Baseline',
         color=RED,
         enc='InceptionV3',
         dec='LSTM (no attention)',
         dat='Flickr30k',
         bleu='—',
         note='Training\ncollapsed'),

    dict(title='v1',      sub='LSTM + Attention',
         color=BLUE,
         enc='InceptionV3 (frozen)',
         dec='LSTM + Bahdanau',
         dat='Flickr30k 30k',
         bleu='0.0674'),

    dict(title='v2',      sub='Multi-head Attention',
         color=BLUE,
         enc='InceptionV3 (frozen)',
         dec='Multi-head Self-Attn',
         dat='Instagram ~21k',
         bleu='—',
         note='Instagram eval\nno Flickr BLEU'),

    dict(title='v3a',     sub='Custom Transformer',
         color=BLUE,
         enc='ConvNeXt-Tiny\nmulti-scale',
         dec='Custom Transformer',
         dat='Flickr30k (LMDB)',
         bleu='Phase 1 baseline'),

    dict(title='v3b',     sub='Perceiver + GPT-2',
         color=BLUE,
         enc='ConvNeXt-Tiny\nmulti-scale',
         dec='Perceiver + GPT-2\ncross-attn inject',
         dat='Flickr30k',
         bleu='0.0656'),

    dict(title='v3c  ✓',  sub='COCO Fine-tuned  ·  FINAL',
         color=GREEN,
         enc='ConvNeXt-Tiny\nmulti-scale',
         dec='Perceiver + GPT-2\ncross-attn inject',
         dat='COCO 118k',
         bleu='0.1341'),

    dict(title='v3d',     sub='Instagram Fine-tune',
         color=RED,
         enc='ConvNeXt-Tiny\nmulti-scale',
         dec='Perceiver + GPT-2\ncross-attn inject',
         dat='Instagram CLIP-filtered\n4,022 samples',
         bleu='0.0711\n(recovery)',
         note='Abandoned —\ndomain gap'),
]

N   = len(CARDS)
CW  = 12.9                             # card width
GAP = (100 - N * CW) / (N + 1)        # ≈ 1.27
CH  = 87
CT  = 95

for i, c in enumerate(CARDS):
    x0 = GAP + i * (CW + GAP)
    y0 = CT - CH
    col = c['color']

    shadow(ax1, x0, y0, CW, CH)
    rbox(ax1, x0, y0, CW, CH, col, lw=3)

    # ── header strip (color fill + version title)
    SH = 11
    fbox(ax1, x0, y0+CH-SH, CW, SH, col)
    ax1.text(x0+CW/2, y0+CH-SH/2, c['title'],
             ha='center', va='center', fontsize=14, fontweight='bold',
             color=WHITE, zorder=5)

    # ── subtitle below strip
    ax1.text(x0+CW/2, y0+CH-SH-1.5, c['sub'],
             ha='center', va='top', fontsize=7.5,
             color=col, fontweight='bold', zorder=5)

    # ── horizontal rule
    ry = y0 + CH - SH - 11
    ax1.plot([x0+0.7, x0+CW-0.7], [ry, ry], color=BORDER, lw=1.0, zorder=5)

    # ── 4 fields
    note_h = 8 if 'note' in c else 0
    avail  = ry - y0 - note_h - 0.5
    row_h  = avail / 4

    for j, (lbl, val) in enumerate([
            ('Encoder', c['enc']),
            ('Decoder', c['dec']),
            ('Dataset', c['dat']),
            ('BLEU-4',  c['bleu']),
    ]):
        yy      = ry - 1.5 - j * row_h
        is_bleu = (lbl == 'BLEU-4')
        ax1.text(x0+0.8, yy, lbl,
                 ha='left', va='top', fontsize=6.8, color=SUB, zorder=5)
        ax1.text(x0+0.8, yy-3.2, val,
                 ha='left', va='top', fontsize=8.5 if is_bleu else 8,
                 color=col if is_bleu else TEXT,
                 fontweight='bold' if is_bleu else 'normal', zorder=5)

    # ── note badge (failed / abandoned)
    if 'note' in c:
        ax1.text(x0+CW/2, y0+0.8, c['note'],
                 ha='center', va='bottom', fontsize=7,
                 color=RED, style='italic', zorder=5)

    # ── arrow to next card
    if i < N - 1:
        ax1.annotate('',
                     xy=(x0+CW+GAP, y0+CH/2),
                     xytext=(x0+CW, y0+CH/2),
                     arrowprops=dict(arrowstyle='->', color=GRAY, lw=2),
                     zorder=6)


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 2 — Final Architecture Flow (v3c)
# ══════════════════════════════════════════════════════════════════════════════

ax2.text(50, 99.5,
         'Final Architecture  —  v3c  |  Perceiver + GPT-2 with Cross-Attention Injection',
         ha='center', va='top', fontsize=17, fontweight='bold', color=TEXT, zorder=6)

CY  = 46
BH  = 76
SH  = 9

def flow_box(ax, cx, w, title, color, lines, cy=CY, bh=BH, sh=SH):
    x0, y0 = cx - w/2, cy - bh/2
    shadow(ax, x0, y0, w, bh)
    rbox(ax, x0, y0, w, bh, color, lw=3)
    fbox(ax, x0, y0+bh-sh, w, sh, color)
    ax.text(cx, y0+bh-sh/2, title,
            ha='center', va='center', fontsize=11,
            fontweight='bold', color=WHITE, zorder=5)

    ry = y0 + bh - sh - 0.5
    ax.plot([x0+0.5, x0+w-0.5], [ry, ry], color=BORDER, lw=0.9, zorder=5)

    avail = ry - y0 - 0.5
    rh    = avail / max(len(lines), 1)

    for k, ln in enumerate(lines):
        yy       = ry - 0.8 - k * rh
        is_dim   = ln.strip().startswith('(B,')
        is_sep   = ln.strip() and set(ln.strip()) <= {'-', '─', ' '}
        is_head  = ln.strip().endswith(':') and not ln.startswith(' ')
        ax.text(cx, yy, ln,
                ha='center', va='top', zorder=5,
                fontsize=7 if is_sep else 9,
                color=SUB if (is_dim or is_sep) else (col if is_head else TEXT),
                fontweight='bold' if is_head else 'normal',
                family='monospace' if ('(B,' in ln or '→' in ln) else 'sans-serif')

# Component positions   cx     w
# Image                  4.5    8
# ConvNextMultiScale    21.5   21
# Perceiver             44     20
# GPT2xAttn             71.5   28
# Caption               92.5   12

# ── 1 — Input Image ──────────────────────────────────────────────────────────
col = GRAY
flow_box(ax2, cx=4.5, w=8, title='Input', color=col,
         lines=['(B, 3,',
                '224, 224)',
                '',
                'JPEG',
                'PNG'])

harrow(ax2, 8.5, CY, 11.0, label='(B, 3, 224, 224)')

# ── 2 — ConvNextMultiScale ────────────────────────────────────────────────────
col = ORANGE
flow_box(ax2, cx=21.5, w=21, title='ConvNextMultiScale', color=col,
         lines=[
             'timm convnext_tiny  (FROZEN)',
             'out_indices = (1, 2, 3)',
             '─'*28,
             'Stage 1  →  192 ch  28×28',
             'pool(8,8)  →  (B, 64, 192)',
             'Linear  192 → 256',
             '─'*28,
             'Stage 2  →  384 ch  14×14',
             'pool  +  Linear  384 → 256',
             '─'*28,
             'Stage 3  →  768 ch  7×7',
             'pool  +  Linear  768 → 256',
             '─'*28,
             'cat(dim=2)  →  (B, 64, 768)',
             'LayerNorm(768)',
         ])

harrow(ax2, 32.0, CY, 34.0, label='(B, 64, 768)')

# ── 3 — Perceiver ─────────────────────────────────────────────────────────────
col = BLUE
flow_box(ax2, cx=44, w=20, title='Perceiver', color=col,
         lines=[
             '16 learnable latents',
             '(B, 16, 768)  randn init',
             '─'*26,
             'Cross-Attention  (8 heads):',
             '  Q  :  latents  (B, 16, 768)',
             '  K,V:  image    (B, 64, 768)',
             '─'*26,
             'LayerNorm  (post-attention)',
             '─'*26,
             'FFN:  768 → 3072 → 768',
             'activation: GELU',
             '─'*26,
             'LayerNorm  (post-FFN)',
             '─'*26,
             'Output:  (B, 16, 768)',
         ])

harrow(ax2, 54.0, CY, 57.5, label='(B, 16, 768)')

# ── 4 — GPT2withCrossAttention ────────────────────────────────────────────────
col = PURPLE
flow_box(ax2, cx=71.5, w=27, title='GPT2withCrossAttention  (gpt2-small)', color=col,
         lines=[
             'wte  (50257 → 768)  +  wpe  (1024 → 768)',
             '─'*36,
             '12 ×',
             '    GPT-2 Block  (causal self-attn)',
             '    12 heads  ·  768 hidden dim',
             '    ─'*18,
             '    CrossAttentionBlock:',
             '    Q  :  token      (B, T, 768)',
             '    K,V:  Perceiver  (B, 16, 768)',
             '    MultiheadAttn  (12 heads)',
             '    LayerNorm',
             '─'*36,
             'ln_f  +  lm_head  →  (B, T, 50257)',
             'Loss:  CrossEntropyLoss(ignore_index=50256)',
         ])

harrow(ax2, 85.0, CY, 88.5, label='(B, T, 50257)')

# ── 5 — Caption ───────────────────────────────────────────────────────────────
col = GREEN
flow_box(ax2, cx=94.5, w=11, title='Caption', color=col,
         lines=[
             'MAX_LEN = 64',
             '',
             'Top-k sampling',
             '  k = 50',
             '  temp = 1.0',
             '',
             'Beam search',
             '  width = 5',
             '',
             '"A dog runs',
             ' through a field"',
         ])

# ── Bottom note ───────────────────────────────────────────────────────────────
ax2.text(50, 1.5,
         'Training: Kaggle 2×T4 GPU  ·  MAX_LEN = 64  ·  BATCH_SIZE = 32  ·  '
         'LR: Perceiver 1e-5 / GPT-2 1e-6  ·  '
         'Early stopping patience = 7  ·  COCO 118k BLEU-4: 0.1341',
         ha='center', va='bottom', fontsize=10, color=SUB, style='italic', zorder=6)

plt.savefig(OUT, dpi=DPI, bbox_inches='tight', facecolor=WHITE)
print(f'Saved: {OUT}  ({int(FW*DPI)} × {int(FH*DPI)} px before bbox_inches)')
