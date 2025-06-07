import os
import random

# 数据集路径
dataset_path = "/media/yst/Elements SE/datasets/PSCC/Training Dataset/"
dataset_names = ['copymove', 'splice']
folder = "fake"
sample_num = 100

# 支持的图片后缀
img_exts = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')

# 存储采样结果
sampled_files = []
for dataset_name in dataset_names:
    folder_path = os.path.join(dataset_path, dataset_name, folder)
    # 列出该文件夹下所有图片文件
    all_files = [f for f in os.listdir(folder_path) if f.lower().endswith(img_exts)]
    
    # 随机采样，若不足sample_num则采全部
    sampled = random.sample(all_files, min(sample_num, len(all_files)))
    
    # 记录带相对路径的文件名，比如 fake/xxx.jpg
    sampled_files.extend([os.path.join(dataset_path, dataset_name, folder, f) for f in sampled])

# 保存文件名到文本文件
save_path = "sampled_files.txt"
with open(save_path, 'w') as f:
    for filename in sampled_files:
        f.write(filename + '\n')

print(f"共采样了{len(sampled_files)}张图片，文件名已保存到 {save_path}")
