import os

import torch
import torchvision.utils as tv_utils


def adjust_learning_rate(optimizer, epoch, lr_strategy, lr_decay_step):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    current_learning_rate = lr_strategy[epoch // lr_decay_step]
    for param_group in optimizer.param_groups:
        param_group['lr'] = current_learning_rate
        print('Learning rate sets to {}.'.format(param_group['lr']))

def save_image(image, image_name, category):
    """
    image: Tensor, shape (B, C, H, W) or (1, C, H, W)
    image_name: str 或 list/tuple[str]，可以带子目录，如 "epoch_001/img1.png"
    category: 比如 "mask"，最终根目录会是 "./mask_results/"
    """
    # 按 batch 维拆开
    images = torch.split(image, 1, dim=0)  # list of [1, C, H, W]
    batch_num = len(images)

    # 统一把 image_name 变成列表形式，方便按 idx 对应
    if isinstance(image_name, (list, tuple)):
        names = list(image_name)
    else:
        # 单个名字，重复使用 batch_num 次
        names = [image_name] * batch_num

    base_dir = f'./{category}_results'

    for ind in range(batch_num):
        raw_name = names[ind]

        # 兼容 bytes
        if isinstance(raw_name, bytes):
            raw_name = raw_name.decode('utf-8')

        # 统一用正斜杠分隔
        raw_name = raw_name.replace('\\', '/')
        parts = raw_name.split('/')  # 例如 ["epoch_001", "img1.png"] 或 ["img1.png"]

        filename = parts[-1]         # "img1.png"
        subdirs = parts[:-1]         # ["epoch_001"] 或 []

        # 去掉原始扩展名，只保留 stem
        stem, _ = os.path.splitext(filename)

        # save_dir: ./mask_results/epoch_001  或 ./mask_results
        if subdirs:
            save_dir = os.path.join(base_dir, *subdirs)
        else:
            save_dir = base_dir

        os.makedirs(save_dir, exist_ok=True)

        save_path = os.path.join(save_dir, stem + '.png')
        tv_utils.save_image(images[ind], save_path)

def findLastCheckpoint(save_dir):
    if os.path.exists(save_dir):
        file_list = os.listdir(save_dir)
        result = 0
        for file in file_list:
            try:
                num = int(file.split('.')[0].split('_')[-1])
                result = max(result, num)
            except:
                continue
        return result
    else:
        os.mkdir(save_dir)
        return 0