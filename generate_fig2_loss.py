# =============================================================================
# generate_fig2_loss.py
# Fig. 2: Training/Validation Loss Curve
# =============================================================================

import os
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

matplotlib.rcParams.update({
    'font.family': 'Arial',
    'font.size': 10,
    'axes.labelsize': 10,
    'axes.titlesize': 0,
    'legend.fontsize': 8,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'axes.linewidth': 0.8,
    'lines.linewidth': 1.0,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
})

FIG_DIR = 'd:/yuceshuju2026/figures'
os.makedirs(FIG_DIR, exist_ok=True)

log = pd.read_csv('d:/yuceshuju2026/logs/mt_transformer_loss.csv')

fig, ax = plt.subplots(figsize=(3.3, 2.5))

ax.plot(log['epoch'], log['train_loss'], 'b-', linewidth=1.0,
        label='Training Loss')
ax.plot(log['epoch'], log['val_loss'], 'r--', linewidth=1.0,
        label='Validation Loss')

best_epoch = log['val_loss'].idxmin()
best_loss = log['val_loss'][best_epoch]
ax.scatter(best_epoch, best_loss, color='red', edgecolor='black',
           s=40, zorder=5)

# ---------- 修正：Best model 标注移到左侧 ----------
ax.annotate(f'Best: epoch {best_epoch}, loss={best_loss:.3f}',
            xy=(best_epoch, best_loss),
            xytext=(best_epoch - 25, best_loss + 0.07),
            arrowprops=dict(arrowstyle='->', color='gray', linewidth=0.7),
            fontsize=6.5, color='black',
            ha='left', va='bottom')

# Early stop 文字
last_epoch = log['epoch'].iloc[-1]
ax.axvline(last_epoch, color='gray', linestyle=':', linewidth=0.8)
ax.text(last_epoch - 1, 0.65, f'Early stop\n(epoch {last_epoch})',
        fontsize=6.5, color='gray', ha='right', va='center')

ax.set_xlabel('Epoch')
ax.set_ylabel('Loss')
ax.legend(frameon=False, loc='upper right')

plt.tight_layout()

pdf_path = os.path.join(FIG_DIR, 'fig2.pdf')
png_path = os.path.join(FIG_DIR, 'fig2.png')
plt.savefig(pdf_path)
plt.savefig(png_path, dpi=300)
plt.close()

print('[OK] fig2.pdf generated')
print(f'     path = {pdf_path}')
print('[OK] fig2.png generated')
print(f'     path = {png_path}')
print(f'     Best epoch: {best_epoch}, loss = {best_loss:.4f}')
print(f'     Last epoch: {last_epoch}')