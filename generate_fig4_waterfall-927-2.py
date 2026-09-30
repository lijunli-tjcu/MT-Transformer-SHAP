# =============================================================================
# generate_fig4_waterfall.py (自然语言标签版)
# Fig. 4: SHAP Waterfall Plots
# 输出：fig4a.pdf + fig4a.png + fig4b.pdf + fig4b.png
#
# 说明：
#   - 术语统一：所有特征标签改为自然语言（去下划线）
#   - 与 Fig. 3、正文一致
# =============================================================================
import os
import numpy as np
import shap
import matplotlib
import matplotlib.pyplot as plt

matplotlib.rcParams.update({
    'font.family': 'Arial',
    'font.size': 9,
    'axes.labelsize': 9,
    'axes.titlesize': 0,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'axes.linewidth': 0.8,
    'lines.linewidth': 0.8,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
})

FIG_DIR = 'd:/yuceshuju2026/figures'
os.makedirs(FIG_DIR, exist_ok=True)

# ---------- 加载 SHAP 值 ----------
shap_mean = np.load('d:/yuceshuju2026/shap/shap_mean.npy')
X_explain = np.load('d:/yuceshuju2026/shap/X_explain.npy')

# ---------- 自然语言特征标签（与 Fig. 3、正文一致） ----------
feature_names = ['gender', 'year of study', 'prior GPA', 'course difficulty',
                 'class size', 'teacher experience', 'instructor academic rank',
                 'student interest', 'teaching style']

base_value = float(np.mean(shap_mean))

# ---------- 自动选择 below/above 样本 ----------
shap_sum = shap_mean.sum(axis=1)
idx_below = int(np.argmin(shap_sum))
idx_above = int(np.argmax(shap_sum))

print(f'Below-average sample index: {idx_below}, SHAP sum = {shap_sum[idx_below]:.4f}')
print(f'Above-average sample index: {idx_above}, SHAP sum = {shap_sum[idx_above]:.4f}')

# ---------- Fig 4a: below-average ----------
explanation_below = shap.Explanation(
    values=shap_mean[idx_below],
    base_values=base_value,
    data=X_explain[idx_below],
    feature_names=feature_names
)
plt.figure(figsize=(3.5, 3.5))   # 略加宽，给更长的标签留空间
shap.plots.waterfall(explanation_below, show=False)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'fig4a.pdf'))
plt.savefig(os.path.join(FIG_DIR, 'fig4a.png'), dpi=300)
plt.close()
print('[OK] fig4a.pdf + fig4a.png generated')

# ---------- Fig 4b: above-average ----------
explanation_above = shap.Explanation(
    values=shap_mean[idx_above],
    base_values=base_value,
    data=X_explain[idx_above],
    feature_names=feature_names
)
plt.figure(figsize=(3.5, 3.5))
shap.plots.waterfall(explanation_above, show=False)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'fig4b.pdf'))
plt.savefig(os.path.join(FIG_DIR, 'fig4b.png'), dpi=300)
plt.close()
print('[OK] fig4b.pdf + fig4b.png generated')