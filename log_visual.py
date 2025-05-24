import matplotlib.pyplot as plt
import numpy as np

log_file = './log/Biformer+HiLo.log'
patch_per_batch = 100
patches_per_epoch = 25000

global_patches = []
loc_accuracies = []
seg_losses = []
cls_accuracies = []
cls_losses = []

with open(log_file, 'r') as f:
	for line in f:
		if '[Epoch' in line and 'Localization Accuracy:' in line:
			try:
				# 提取 Epoch 和 Batch
				epoch = int(line.split('[Epoch ')[1].split(',')[0])
				batch = int(line.split('Batch ')[1].split(']')[0])
				global_patch = (epoch - 1) * patches_per_epoch + batch

				if global_patch % 10 != 0:
					continue

				# 提取 Localization Accuracy 百分比
				loc_acc_str = line.split('Localization Accuracy:')[1].split('%')[0].strip().split()[-1]
				loc_acc = float(loc_acc_str)

				# 提取 seg_loss
				seg_loss_str = line.split('seg_loss:')[1].split(';')[0].strip()
				seg_loss = float(seg_loss_str)

				# 提取 Classification Accuracy 百分比
				cls_acc_str = line.split('Classification Accuracy:')[1].split('%')[0].strip().split()[-1]
				cls_acc = float(cls_acc_str)

				# 提取 cls_loss
				cls_loss_str = line.split('cls_loss:')[1].strip()
				cls_loss = float(cls_loss_str)

				# 存储
				global_patches.append(global_patch)
				loc_accuracies.append(loc_acc)
				seg_losses.append(seg_loss)
				cls_accuracies.append(cls_acc)
				cls_losses.append(cls_loss)

			except Exception as e:
				print(f"解析失败: {line.strip()}")
				print(f"错误原因: {e}")


def plot_and_save(x, y, title, ylabel, filename, window=20, x_range=None, y_range=None):
	plt.figure(figsize=(14, 7))

	# 过滤指定x区间的数据
	if x_range is not None:
		x = np.array(x)
		y = np.array(y)
		mask = (x >= x_range[0]) & (x <= x_range[1])
		x = x[mask]
		y = y[mask]

	# 画原始曲线
	plt.plot(x, y, alpha=0.25, label='Raw', color='cornflowerblue')

	# 画平滑曲线
	if len(y) >= window:
		y_smooth = np.convolve(y, np.ones(window)/window, mode='valid')
		x_smooth = x[len(x) - len(y_smooth):]
		plt.plot(x_smooth, y_smooth, label=f'Moving Avg (window={window})', 
						 color='orangered', linewidth=3)

	# 设置标题和标签
	plt.title(title, fontsize=18, weight='bold', color='#333333')
	plt.xlabel('Global Patch Number', fontsize=15, weight='medium')
	plt.ylabel(ylabel, fontsize=15, weight='medium')

	# 设置坐标轴范围（如果有指定）
	if x_range is not None:
		plt.xlim(x_range)
	if y_range is not None:
		plt.ylim(y_range)

	# 美化网格和图例
	plt.grid(True, linestyle='--', alpha=0.4, linewidth=1)
	plt.legend(fontsize=13, frameon=True, shadow=True)
	plt.xticks(fontsize=12)
	plt.yticks(fontsize=12)
	plt.tight_layout()

	# 保存并显示
	plt.savefig(filename, dpi=300)
	print(f"图像已保存为 {filename}")

# 示例调用
plot_and_save(global_patches, loc_accuracies, 
			'Localization Accuracy Over Training', 'Accuracy (%)', 'loc_accuracy.png',
			window=20, x_range=(0, 25000), y_range=(0, 100))

plot_and_save(global_patches, seg_losses, 
			'Segmentation Loss Over Training', 'Loss', 'seg_loss.png',
			window=20, x_range=(0, 25000), y_range=(0, 3.5))

plot_and_save(global_patches, cls_accuracies, 
		'Classification Accuracy Over Training', 'Accuracy (%)', 'cls_accuracy.png',
		window=20, x_range=(0, 25000), y_range=(0, 100))

plot_and_save(global_patches, cls_losses, 
		'Classification Loss Over Training', 'Loss', 'cls_loss.png',
		window=20, x_range=(0, 25000), y_range=(0, 3.5))
