import matplotlib.pyplot as plt
import numpy as np
import re

# 更新后适配你的日志格式
log_file = 'total=15000_epoch=100_batch=6_with_MoE=True(CBAM_K=8)_With_HiLo=True/total=15000_epochs=100_batch=6_with_MoE=True(CBAM_K=8)_with_HiLo=True.log'
patches_per_epoch = 2500

global_patches = []
seg_f1_scores = []
seg_losses = []
cls_accuracies = []
cls_losses = []

# 正则模式
epoch_batch_pattern = re.compile(r'\[Epoch\s+(\d+), Batch\s+(\d+)\]')
f1_pattern = re.compile(r'F1 Score:\s*([\d\.]+)')
seg_loss_pattern = re.compile(r'seg_loss:\s*([\d\.]+)')
cls_acc_pattern = re.compile(r'Classification Accuracy:.*\]\s*([\d\.]+)%')
cls_loss_pattern = re.compile(r'cls_loss:\s*([\d\.]+)')

with open(log_file, 'r') as f:
    for line in f:
        if 'F1 Score:' in line:
            # 提取 epoch 和 batch
            m = epoch_batch_pattern.search(line)
            if not m:
                continue
            epoch = int(m.group(1))
            batch = int(m.group(2))
            global_patch = (epoch - 1) * patches_per_epoch + batch
            # 只取每10个batch记录一次
            if global_patch % 10 != 0:
                continue

            # 提取指标
            f1_match = f1_pattern.search(line)
            seg_loss_match = seg_loss_pattern.search(line)
            cls_acc_match = cls_acc_pattern.search(line)
            cls_loss_match = cls_loss_pattern.search(line)

            if f1_match and seg_loss_match and cls_acc_match and cls_loss_match:
                f1 = float(f1_match.group(1))
                seg_loss = float(seg_loss_match.group(1))
                cls_acc = float(cls_acc_match.group(1))
                cls_loss = float(cls_loss_match.group(1))
                global_patches.append(global_patch)
                seg_f1_scores.append(f1)
                seg_losses.append(seg_loss)
                cls_accuracies.append(cls_acc)
                cls_losses.append(cls_loss)

def plot_and_save(x, y, title, ylabel, filename, window=20, x_range=None, y_range=None):
    plt.figure(figsize=(10, 6))
    x_arr = np.array(x)
    y_arr = np.array(y)

    # 过滤x范围
    if x_range:
        mask = (x_arr >= x_range[0]) & (x_arr <= x_range[1])
        x_arr = x_arr[mask]
        y_arr = y_arr[mask]

    # 原始曲线
    plt.plot(x_arr, y_arr, color='black', alpha=0.1, label='Raw')

    # 平滑曲线
    if len(y_arr) >= window:
        y_smooth = np.convolve(y_arr, np.ones(window)/window, mode='valid')
        x_smooth = x_arr[len(x_arr) - len(y_smooth):]
        plt.plot(x_smooth, y_smooth, label=f'{window}-step MA', linewidth=2)

    plt.title(title)
    plt.xlabel('Global Patch Number')
    plt.ylabel(ylabel)
    if x_range:
        plt.xlim(*x_range)
    if y_range:
        plt.ylim(*y_range)
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()
    print(f"Saved plot: {filename}")

# 绘制并保存
plot_and_save(global_patches, seg_f1_scores,
              'Segmentation F1 over Training', 'F1 Score', 'seg_f1.png',
              window=20, x_range=(0, 175000), y_range=(0, 1.0))

plot_and_save(global_patches, seg_losses,
              'Segmentation Loss over Training', 'Loss', 'seg_loss.png',
              window=20, x_range=(0, 175000), y_range=(0, max(seg_losses) * 1.1))

plot_and_save(global_patches, cls_accuracies,
              'Classification Accuracy over Training', 'Accuracy (%)', 'cls_accuracy.png',
              window=20, x_range=(0, 175000), y_range=(0, 100))

plot_and_save(global_patches, cls_losses,
              'Classification Loss over Training', 'Loss', 'cls_loss.png',
              window=20, x_range=(0, 175000), y_range=(0, max(cls_losses) * 1.1))
