# =============================================================================
# generate_fig3_shap_bar.py
# Fig. 3: SHAP Feature Importance Bar Plot
#
# 说明：
#   - 术语统一：instructor academic rank
#   - 方式 A：mean 和 std 都用 per-run 数据（尺度一致）
#   - 输出：fig3.pdf + fig3.png
# =============================================================================
import os
import numpy as np
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

# ---------- 加载 25 次运行的 SHAP 值 ----------
shap_all = np.array([
    np.load(f'd:/yuceshuju2026/shap/shap_run{i}.npy')
    for i in range(25)
])  # (25, 50, 9)

# ---------- 方式 A：mean 和 std 都用 per-run 数据 ----------
per_run_means = np.mean(np.abs(shap_all), axis=1)  # (25, 9)
mean_abs = np.mean(per_run_means, axis=0)           # (9,)
std_abs = np.std(per_run_means, axis=0)             # (9,)

# ---------- 术语统一后的特征名 ----------
feature_names = ['gender', 'year', 'gpa', 'course_difficulty', 'class_size',
                 'teacher_experience', 'instructor_academic_rank',
                 'student_interest', 'teaching_style']

# ---------- 用于显示的标签（图里显示的文本） ----------
feature_labels = ['gender', 'year', 'gpa', 'course difficulty', 'class size',
                  'teacher experience', 'instructor academic rank',
                  'student interest', 'teaching style']

order = np.argsort(-mean_abs)
sorted_features = [feature_labels[i] for i in order]
sorted_means = mean_abs[order]
sorted_stds = std_abs[order]

# ---------- Bar plot ----------
fig, ax = plt.subplots(figsize=(3.8, 3.0))
y_pos = np.arange(len(sorted_features))

ax.barh(y_pos, sorted_means, xerr=sorted_stds,
        color='steelblue', edgecolor='black', linewidth=0.5,
        error_kw={'elinewidth': 0.8, 'capsize': 2, 'capthick': 0.8})

ax.set_yticks(y_pos)
ax.set_yticklabels(sorted_features, fontsize=8)
ax.set_xlabel('mean |SHAP|', fontsize=9)
ax.invert_yaxis()

# ---------- 数值标注在条形右端内侧（白色） ----------
for i, (mean, std) in enumerate(zip(sorted_means, sorted_stds)):
    ax.text(mean - 0.02, i, f'{mean:.2f}',
            va='center', ha='right', fontsize=7,
            color='white', fontweight='bold')

# ---------- x 轴范围扩展 ----------
ax.set_xlim(0, max(sorted_means + sorted_stds) * 1.15)

plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'fig3.pdf'))
plt.savefig(os.path.join(FIG_DIR, 'fig3.png'), dpi=300)
plt.close()

print('[OK] fig3.pdf + fig3.png generated')
print('\nFeature ranking (mean ± std, Method A):')
for i, idx in enumerate(order):
    print(f'  {i+1}. {feature_labels[idx]}: {mean_abs[idx]:.4f} ± {std_abs[idx]:.4f}')
print(f'\nSum of mean|SHAP| = {mean_abs.sum():.4f}')