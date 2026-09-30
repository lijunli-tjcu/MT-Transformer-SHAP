# =============================================================================
# generate_fig1_framework.py (修复版)
# Fig. 1: Overall framework with grouped layers
# 输出：fig1.pdf + fig1.png
# =============================================================================
import os
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

matplotlib.rcParams.update({
    'font.family': 'Arial',
    'font.size': 8,
    'axes.titlesize': 0,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
})

FIG_DIR = 'd:/yuceshuju2026/figures'
os.makedirs(FIG_DIR, exist_ok=True)

fig, ax = plt.subplots(figsize=(7.0, 2.0))
ax.set_xlim(0, 10)
ax.set_ylim(0, 2.6)
ax.axis('off')

# ============ 三个分组底色 ============
# 输入组
group1 = FancyBboxPatch((0.1, 0.3), 3.0, 1.9,
                        boxstyle='round,pad=0.05,rounding_size=0.1',
                        linewidth=0, facecolor='#f5f5f5', zorder=0)
ax.add_patch(group1)
ax.text(1.6, 2.35, 'Input', ha='center', va='center',
        fontsize=7, fontstyle='italic', color='gray', zorder=1)

# 模型组
group2 = FancyBboxPatch((3.4, 0.3), 2.6, 1.9,
                        boxstyle='round,pad=0.05,rounding_size=0.1',
                        linewidth=0, facecolor='#e8f0f8', zorder=0)
ax.add_patch(group2)
ax.text(4.7, 2.35, 'Model', ha='center', va='center',
        fontsize=7, fontstyle='italic', color='gray', zorder=1)

# 输出组
group3 = FancyBboxPatch((6.3, 0.3), 3.6, 1.9,
                        boxstyle='round,pad=0.05,rounding_size=0.1',
                        linewidth=0, facecolor='#eef5ee', zorder=0)
ax.add_patch(group3)
ax.text(8.1, 2.35, 'Output', ha='center', va='center',
        fontsize=7, fontstyle='italic', color='gray', zorder=1)

# ============ 各阶段框 ============
def add_box(x, y, w, h, title, body, zorder=2):
    box = FancyBboxPatch((x, y), w, h,
                         boxstyle='round,pad=0.03,rounding_size=0.08',
                         linewidth=0.8, edgecolor='black',
                         facecolor='white', zorder=zorder)
    ax.add_patch(box)
    ax.text(x + w/2, y + h - 0.18, title,
            ha='center', va='top', fontsize=7.5, fontweight='bold', zorder=zorder)
    ax.text(x + w/2, y + h - 0.55, body,
            ha='center', va='top', fontsize=6.5, zorder=zorder)

# Data Collection
add_box(0.2, 0.6, 1.3, 1.4, 'Data\nCollection',
        '343 SET\nresponses\n9 input features')

# Feature Encoding
add_box(1.7, 0.6, 1.3, 1.4, 'Feature\nEncoding',
        'Scalar\nprojection\nTransformer\nencoder')

# Multi-Task Prediction
add_box(3.5, 0.6, 2.4, 1.4, 'Multi-Task\nPrediction',
        'Shared encoder\n+ 13 task-specific\nheads\n(1 overall + 12 sub)')

# SHAP Analysis
add_box(6.4, 0.6, 1.5, 1.4, 'SHAP\nAnalysis',
        'Global\nimportance\n+ Local\nexplanations')

# Outputs
add_box(8.1, 0.6, 1.7, 1.4, 'Outputs',
        '1 overall\nscore\n12 sub-\nindicator\ndiagnostics')

# ============ 箭头 ============
def add_arrow(x1, y1, x2, y2):
    arrow = FancyArrowPatch((x1, y1), (x2, y2),
                            arrowstyle='->', mutation_scale=10,
                            linewidth=0.8, color='black', zorder=3)
    ax.add_patch(arrow)

add_arrow(1.5, 1.3, 1.7, 1.3)
add_arrow(3.0, 1.3, 3.5, 1.3)
add_arrow(5.9, 1.3, 6.4, 1.3)
add_arrow(7.9, 1.3, 8.1, 1.3)

plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'fig1.pdf'))
plt.savefig(os.path.join(FIG_DIR, 'fig1.png'), dpi=300)
plt.close()
print('[OK] fig1.pdf + fig1.png generated (fixed version)')