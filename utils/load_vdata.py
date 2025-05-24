import os
import torch
import numpy as np
import imageio.v2 as imageio
from torch.utils import data

class TestData(data.Dataset):
    def __init__(self, args):
        super(TestData, self).__init__()

        txt_file = "./sampled_files.txt"

        with open(txt_file, 'r') as f:
            lines = f.readlines()

        # 去除换行符，并加上根目录（如果有）
        self.image_names = [os.path.join(line.strip()) for line in lines]

        # 根据文件名自动判断类别（0表示authentic，1表示fake）
        self.image_class = [0 if 'authentic' in name.lower() else 1 for name in self.image_names]

    def rgba2rgb(self, rgba, background=(255, 255, 255)):
        row, col, ch = rgba.shape
        rgb = np.zeros((row, col, 3), dtype='float32')
        r, g, b, a = rgba[:, :, 0], rgba[:, :, 1], rgba[:, :, 2], rgba[:, :, 3]
        a = np.asarray(a, dtype='float32') / 255.0
        R, G, B = background
        rgb[:, :, 0] = r * a + (1.0 - a) * R
        rgb[:, :, 1] = g * a + (1.0 - a) * G
        rgb[:, :, 2] = b * a + (1.0 - a) * B
        return np.asarray(rgb, dtype='uint8')

    def __getitem__(self, index):
        image_name = self.image_names[index]
        cls = self.image_class[index]

        image = imageio.imread(image_name)

        # 若是 RGBA，则转为 RGB
        if image.ndim == 3 and image.shape[-1] == 4:
            image = self.rgba2rgb(image)

        # 标准化为 torch tensor
        image = torch.from_numpy(image.astype(np.float32) / 255).permute(2, 0, 1)

        return image, cls, image_name

    def __len__(self):
        return len(self.image_names)
