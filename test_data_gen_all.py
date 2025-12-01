import os

# 数据集路径
dataset_path = "/mnt/e/datasets/PSCC/Training Dataset/"
dataset_names = ['copymove', 'splice', 'splice_randmask', 'removal']
folder = "fake"

# 支持的图片后缀
img_exts = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')

# 存储结果
sampled_files = []
for dataset_name in dataset_names:
    folder_path = os.path.join(dataset_path, dataset_name, folder)
    # 列出该文件夹下所有图片文件
    all_files = [f for f in os.listdir(folder_path) if f.lower().endswith(img_exts)]

    # 不再随机采样，直接全量加入
    sampled_files.extend(
        [os.path.join(dataset_path, dataset_name, folder, f) for f in all_files]
    )

# 保存文件名到文本文件
save_path = "sampled_files_all.txt"
with open(save_path, 'w') as f:
    for filename in sampled_files:
        f.write(filename + '\n')

print(f"共收集了{len(sampled_files)}张图片，文件名已保存到 {save_path}")
