import os
from os.path import join
import random
from random import randrange
from PIL import Image, ImageFile
import numpy as np
import imageio.v2 as imageio
import torch
import torch.utils.data as data
from pathlib import Path


IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}

def _list_images_recursive(root: str):
    out = []
    for p in Path(root).rglob("*"):
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            out.append(str(p))
    out.sort()
    return out


def _is_img(p: str) -> bool:
    return os.path.splitext(p)[1].lower() in IMG_EXTS


def _list_images(root: str):
    out = []
    for dp, dn, fn in os.walk(root):
        for f in fn:
            p = os.path.join(dp, f)
            if _is_img(p):
                out.append(p)
    out.sort()
    return out


def _index_masks_by_stem(mask_root: str):
    """
    建立 stem -> mask_path 索引
    （stem 不含后缀；大小写不敏感）
    """
    idx = {}
    masks = _list_images(mask_root)
    for p in masks:
        stem = os.path.splitext(os.path.basename(p))[0].lower()
        idx.setdefault(stem, []).append(p)
    return idx


def _pair_hard(fake_root: str, mask_root: str):
    """
    hard 数据配对：
      - 优先同名同后缀
      - 其次同 stem（后缀不同也行）
      - mask 若是 RGB，后面读的时候会取单通道
    返回：[(fake_path, mask_path), ...]
    """
    fake_paths = _list_images(fake_root)
    mask_idx = _index_masks_by_stem(mask_root)

    pairs = []
    missing = []

    for fp in fake_paths:
        base = os.path.basename(fp)
        stem = os.path.splitext(base)[0].lower()

        # 1) 先尝试同名文件（包括后缀）
        mp1 = os.path.join(mask_root, base)
        if os.path.exists(mp1) and _is_img(mp1):
            pairs.append((fp, mp1))
            continue

        # 2) 再尝试同 stem 的任意 mask
        cands = mask_idx.get(stem, [])
        if len(cands) > 0:
            # 多候选时优先 png
            cands = sorted(cands, key=lambda x: (0 if x.lower().endswith(".png") else 1, x))
            pairs.append((fp, cands[0]))
        else:
            missing.append(fp)

    if len(missing) > 0:
        print(f"[HARD][WARN] missing masks for {len(missing)} fakes (show 10):")
        for p in missing[:10]:
            print("  ", p)

    print(f"[HARD] paired={len(pairs)}, missing={len(missing)}")
    return pairs


def generate_4masks(mask):
    # mask: np array
    mask_pil = Image.fromarray(mask)

    (width2, height2) = (mask_pil.width // 2, mask_pil.height // 2)
    (width3, height3) = (mask_pil.width // 4, mask_pil.height // 4)
    (width4, height4) = (mask_pil.width // 8, mask_pil.height // 8)

    # 对 mask 用 NEAREST 更合理（避免插值污染标签）
    mask2 = mask_pil.resize((width2, height2), resample=Image.NEAREST)
    mask3 = mask_pil.resize((width3, height3), resample=Image.NEAREST)
    mask4 = mask_pil.resize((width4, height4), resample=Image.NEAREST)

    def _bin01(x):
        x = x.astype(np.float32) / 255.0
        x = (x > 0.5).astype(np.float32)
        return x

    mask = _bin01(mask)
    mask2 = _bin01(np.asarray(mask2))
    mask3 = _bin01(np.asarray(mask3))
    mask4 = _bin01(np.asarray(mask4))

    # 重要：copy/clone，避免 DataLoader 多进程时 “storage not resizable”
    mask = torch.from_numpy(mask.copy()).contiguous().clone()
    mask2 = torch.from_numpy(mask2.copy()).contiguous().clone()
    mask3 = torch.from_numpy(mask3.copy()).contiguous().clone()
    mask4 = torch.from_numpy(mask4.copy()).contiguous().clone()

    return mask, mask2, mask3, mask4


def data_aug(img, data_aug_ind):
    img = Image.fromarray(img)
    if data_aug_ind == 0:
        return np.asarray(img)
    elif data_aug_ind == 1:
        return np.asarray(img.rotate(90, expand=True))
    elif data_aug_ind == 2:
        return np.asarray(img.rotate(180, expand=True))
    elif data_aug_ind == 3:
        return np.asarray(img.rotate(270, expand=True))
    elif data_aug_ind == 4:
        return np.asarray(img.transpose(Image.FLIP_TOP_BOTTOM))
    elif data_aug_ind == 5:
        return np.asarray(img.rotate(90, expand=True).transpose(Image.FLIP_TOP_BOTTOM))
    elif data_aug_ind == 6:
        return np.asarray(img.rotate(180, expand=True).transpose(Image.FLIP_TOP_BOTTOM))
    elif data_aug_ind == 7:
        return np.asarray(img.rotate(270, expand=True).transpose(Image.FLIP_TOP_BOTTOM))
    else:
        raise Exception('Data augmentation index is not applicable.')


class TrainData(data.Dataset):
    def __init__(self, args):
        super(TrainData, self).__init__()
        path, crop_size, train_num, train_ratio, val_num = args['path'], \
            args['crop_size'], args['train_num'], \
            args['train_ratio'], args['val_num']

        # ======================= 原有数据 =======================
        # authentic
        authentic_names = []
        authentic_path = join(path, 'authentic')
        with open(join(authentic_path, 'authentic.txt')) as f:
            contents = f.readlines()
            for content in contents[val_num:]:
                authentic_names.append(join(authentic_path, content.strip()))

        # splice + splice_randmask
        splice_names = []
        splice_path = join(path, 'splice')
        with open(join(splice_path, 'fake.txt')) as f:
            contents = f.readlines()
            for content in contents[val_num:]:
                splice_names.append(join(splice_path, content.strip()))

        splice_randmask = []
        splice_randmask_path = join(path, 'splice_randmask')
        with open(join(splice_randmask_path, 'fake.txt')) as f:
            contents = f.readlines()
            for content in contents:
                splice_randmask.append(join(splice_randmask_path, content.strip()))
        splice_names = splice_names + splice_randmask

        # copymove
        copymove_names = []
        copymove_path = join(path, 'copymove')
        with open(join(copymove_path, 'fake.txt')) as f:
            contents = f.readlines()
            for content in contents[val_num:]:
                copymove_names.append(join(copymove_path, content.strip()))

        # removal
        removal_names = []
        removal_path = join(path, 'removal')
        with open(join(removal_path, 'fake.txt')) as f:
            contents = f.readlines()
            for content in contents[val_num:]:
                removal_names.append(join(removal_path, content.strip()))

        self.image_names = [authentic_names, splice_names, copymove_names, removal_names]
        self.train_num = train_num
        self.train_ratio = train_ratio
        self.crop_size = crop_size

        # ======================= 新增 hard 数据 =======================
        # WSL 下建议用 /mnt/e/...（你给的 E:\... 在 WSL 里是 /mnt/e/...）
        # hard_root = args.get("hard_path", "/mnt/c/datasets/datasets_hard")
        hard_root = args.get("hard_path", "/mnt/c/datasets/splice_columbia/")
        # hard_root = args.get('hard_path', '/mnt/c/datasets/datasets_hard_hard')
        self.hard_train_pick = int(args.get('hard_train_pick', 80))

        self.hard_fake_root = join(hard_root, 'fake_256')
        self.hard_mask_root = join(hard_root, 'mask_256')

        self.hard_pairs_all = _pair_hard(self.hard_fake_root, self.hard_mask_root)  # [(fake, mask),...]
        self.hard_train_pairs = []  # 每个 epoch 重抽

        # hard 的 cls 怎么给：默认固定为 1（当作 splice）
        self.hard_cls_fixed = int(args.get('hard_cls_fixed', 1))

        # ======================= hard authentic =======================
        self.hard_auth_roots = [
            # join(hard_root, "C1.0_Au_256"),
            # join(hard_root, "C2.0_Au_256"),
            join(hard_root, "4cam_auth_256"),
            # join(hard_root, "Cover_Au_256")
        ]

        self.hard_auth_all = []
        for r in self.hard_auth_roots:
            if os.path.exists(r):
                self.hard_auth_all += _list_images_recursive(r)

        print(f"[HARD][AUTH] total={len(self.hard_auth_all)}")

        self.hard_auth_train_pick = int(args.get("hard_auth_train_pick", 100))
        self.hard_auth_train = []

        # 初始化抽一次（epoch=0）
        self.set_epoch(0)

        self.authentic_names = authentic_names
        self.splice_names = splice_names
        self.copymove_names = copymove_names
        self.removal_names = removal_names

    def dump_epoch0_lists(self, out_dir: str, epoch: int = 0, print_n: int = 30):
        """
        导出：
        1) hard epoch 抽样的 5000 对 fake/mask
        2) train 四类候选池（用于确认数据源）
        """
        import os
        os.makedirs(out_dir, exist_ok=True)

        # 1) hard 抽样对
        hard_pair_path = os.path.join(out_dir, f"hard_train_pairs_epoch{epoch}.txt")
        hard_fake_path = os.path.join(out_dir, f"hard_train_fake_epoch{epoch}.txt")

        with open(hard_pair_path, "w", encoding="utf-8") as fpair, \
            open(hard_fake_path, "w", encoding="utf-8") as ffake:
            for fp, mp in self.hard_train_pairs:
                fpair.write(fp + "\n")
                fpair.write(mp + "\n")
                ffake.write(fp + "\n")

        print(f"[DEBUG] saved hard train list: {hard_fake_path}")
        print(f"[DEBUG] saved hard train pairs: {hard_pair_path}")

        # 打印前 N 条
        print(f"[DEBUG] hard train sample (first {print_n}):")
        for i, (fp, mp) in enumerate(self.hard_train_pairs[:print_n]):
            print(f"  [{i:04d}] fake={fp}")
            print(f"         mask={mp}")

        # 2) train 四类候选池（base 部分是随机抽样，无法列出“将被抽到的具体序列”，但池子可核对）
        def dump_pool(name, lst):
            p = os.path.join(out_dir, f"base_train_pool_{name}.txt")
            with open(p, "w", encoding="utf-8") as f:
                for x in lst:
                    f.write(x + "\n")
            print(f"[DEBUG] saved base train pool {name}: {p} (n={len(lst)})")

        dump_pool("authentic", self.authentic_names)
        dump_pool("splice", self.splice_names)
        dump_pool("copymove", self.copymove_names)
        dump_pool("removal", self.removal_names)

    def set_epoch(self, epoch: int):
        """
        每个 epoch 开始调用一次：
          train: 从 hard_pairs_all 无放回抽 hard_train_pick 个
        """
        # -------- hard fake --------
        n = len(self.hard_pairs_all)
        k = self.hard_train_pick
        rng = random.Random(12345 + epoch)
        if n >= k:
            self.hard_train_pairs = rng.sample(self.hard_pairs_all, k)
        else:
            # 不够就允许重复抽
            self.hard_train_pairs = [rng.choice(self.hard_pairs_all) for _ in range(k)]

        # -------- hard authentic --------
        n2 = len(self.hard_auth_all)
        k2 = self.hard_auth_train_pick
        if n2 > 0:
            if n2 >= k2:
                self.hard_auth_train = rng.sample(self.hard_auth_all, k2)
            else:
                self.hard_auth_train = [rng.choice(self.hard_auth_all) for _ in range(k2)]
        else:
            self.hard_auth_train = []

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

    def _read_mask_any(self, mask_path: str):
        m = imageio.imread(mask_path)
        if m.ndim == 3:
            m = m[:, :, 0]
        return m

    def get_item(self, index):
        sample_name = None
        crop_width, crop_height = self.crop_size
        train_num = self.train_num
        train_ratio = self.train_ratio

        # ============ 1) 原有数据（按你的 4 类采样） ============
        if index < train_num:
            # get 4 class
            if index < train_num * train_ratio[0]:
                cls = 0
            elif train_num * train_ratio[0] <= index < train_num * (train_ratio[0] + train_ratio[1]):
                cls = 1
            elif train_num * (train_ratio[0] + train_ratio[1]) <= index < train_num * (
                    train_ratio[0] + train_ratio[1] + train_ratio[2]):
                cls = 2
            else:
                cls = 3

            one_cls_names = self.image_names[cls]
            index2 = randrange(0, len(one_cls_names))
            image_name = one_cls_names[index2]
            sample_name = f"base|{image_name}"
            image = imageio.imread(image_name)

            # 处理 RGBA
            if image.ndim == 3 and image.shape[-1] == 4:
                image = self.rgba2rgb(image)

            if image.ndim != 3 or image.shape[2] != 3:
                raise Exception(f'Image channel is not 3: {image_name}')

            im_height, im_width, _ = image.shape

            # authentic
            if cls == 0:
                if im_height != crop_height or im_width != crop_width:
                    image = Image.fromarray(image.astype(np.uint8))
                    image = image.resize((crop_height, crop_width), resample=Image.BICUBIC)
                    image = np.asarray(image)
                mask = np.zeros((crop_height, crop_width)).astype(np.uint8)

            # splice/copy-move/removal
            else:
                if cls == 1:
                    if '.jpg' in image_name:
                        mask_name = image_name.replace('fake', 'mask').replace('.jpg', '.png')
                    else:
                        mask_name = image_name.replace('fake', 'mask').replace('.tif', '.png')
                elif cls == 2:
                    mask_name = image_name.replace('fake', 'mask')
                else:  # cls == 3
                    if '.tif' in image_name:
                        mask_name = image_name.replace('fake', 'mask').replace('.tif', '.png')
                    else:
                        mask_name = image_name.replace('fake', 'mask').replace('.jpg', '.png')

                mask = self._read_mask_any(mask_name)

                if im_height != crop_height or im_width != crop_width:
                    image = Image.fromarray(image)
                    image = image.resize((crop_height, crop_width), resample=Image.BICUBIC)
                    image = np.asarray(image)

                    mask = Image.fromarray(mask)
                    # mask 用 NEAREST
                    mask = mask.resize((crop_height, crop_width), resample=Image.NEAREST)
                    mask = np.asarray(mask)

        # ============ 2) hard 数据（每个 epoch 抽取的 5000 条） ============
        else:
            idx = index - train_num

            # ---------- hard fake ----------
            if idx < self.hard_train_pick:
                cls = self.hard_cls_fixed  # 1
                fp, mp = self.hard_train_pairs[idx]
                sample_name = f"hard_fake|{fp}"

                image = imageio.imread(fp)
                if image.ndim == 3 and image.shape[-1] == 4:
                    image = self.rgba2rgb(image)
                if image.ndim != 3:
                    image = np.stack([image] * 3, axis=-1)
                if image.shape[2] != 3:
                    image = image[:, :, :3]

                mask = self._read_mask_any(mp)

            # ---------- hard authentic ----------
            else:
                cls = 0
                aidx = idx - self.hard_train_pick
                ap = self.hard_auth_train[aidx]
                sample_name = f"hard_auth|{ap}"

                image = imageio.imread(ap)
                if image.ndim == 3 and image.shape[-1] == 4:
                    image = self.rgba2rgb(image)
                if image.ndim != 3:
                    image = np.stack([image] * 3, axis=-1)
                if image.shape[2] != 3:
                    image = image[:, :, :3]

                mask = np.zeros(image.shape[:2], dtype=np.uint8)

        # ============ augmentation + to tensor ============
        aug_index = randrange(0, 8)

        image = data_aug(image, aug_index)
        image = torch.from_numpy(np.ascontiguousarray(image).astype(np.float32) / 255.0).permute(2, 0, 1)
        image = image.contiguous().clone()

        mask = data_aug(mask, aug_index)
        mask, mask2, mask3, mask4 = generate_4masks(mask)

        if sample_name is not None:
            return image, [mask, mask2, mask3, mask4], cls, sample_name
        else:
            return image, [mask, mask2, mask3, mask4], cls

    def __getitem__(self, index):
        return self.get_item(index)

    def __len__(self):
        return (
            self.train_num
            + self.hard_train_pick          # hard fake
            + self.hard_auth_train_pick     # hard authentic
        )

ImageFile.LOAD_TRUNCATED_IMAGES = True

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def _is_img(p: Path) -> bool:
    return p.is_file() and p.suffix.lower() in IMG_EXTS


def _read_image_rgb(path: str) -> np.ndarray:
    """稳健读 RGB，保证输出 HxWx3 uint8"""
    im = Image.open(path)
    if im.mode != "RGB":
        im = im.convert("RGB")
    return np.asarray(im, dtype=np.uint8)


def _read_mask_gray(path: str) -> np.ndarray:
    """读 mask 为单通道 HxW uint8"""
    im = Image.open(path).convert("L")
    return np.asarray(im, dtype=np.uint8)


def _resize_mask_nearest(mask_u8: np.ndarray, size_hw: tuple[int, int]) -> np.ndarray:
    """把 mask 用最近邻 resize 到 (H,W)"""
    H, W = size_hw
    im = Image.fromarray(mask_u8, mode="L")
    im = im.resize((W, H), resample=Image.NEAREST)
    return np.asarray(im, dtype=np.uint8)


def _pair_hard(fake_root: str, mask_root: str):
    """
    在 hard 数据集里配对 fake/mask。
    规则：
      - 以 fake 文件为准遍历
      - 优先匹配：mask_root 下与 fake 相同的相对路径（仅替换根目录）
      - 其次匹配：mask_root 任意位置同 stem（建立 stem->path 索引）
    返回：[(fake_path, mask_path), ...] 都是 str
    """
    fake_root_p = Path(fake_root)
    mask_root_p = Path(mask_root)

    if (not fake_root_p.exists()) or (not mask_root_p.exists()):
        print(f"[HARD] fake_root or mask_root not exists: {fake_root} | {mask_root}")
        return []

    # 建 mask stem 索引（用于兜底）
    mask_index = {}
    for mp in mask_root_p.rglob("*"):
        if not _is_img(mp):
            continue
        stem = mp.stem
        if stem not in mask_index:
            mask_index[stem] = str(mp)

    pairs = []
    missing = 0

    fake_files = [fp for fp in fake_root_p.rglob("*") if _is_img(fp)]
    fake_files.sort()

    for fp in fake_files:
        rel = fp.relative_to(fake_root_p)

        # 1) 直接相对路径对应（保留文件名）
        cand = mask_root_p / rel
        if cand.exists() and _is_img(cand):
            pairs.append((str(fp), str(cand)))
            continue

        # 2) 相对路径对应但后缀不同：用 stem 找同目录
        cand_dir = (mask_root_p / rel.parent)
        found = None
        if cand_dir.exists():
            for ext in IMG_EXTS:
                cand2 = cand_dir / (fp.stem + ext)
                if cand2.exists() and _is_img(cand2):
                    found = str(cand2)
                    break
        if found is not None:
            pairs.append((str(fp), found))
            continue

        # 3) 全局 stem 索引兜底
        if fp.stem in mask_index:
            pairs.append((str(fp), mask_index[fp.stem]))
        else:
            missing += 1

    print(f"[HARD] paired={len(pairs)}, missing={missing}")
    return pairs


class ValData(data.Dataset):
    def __init__(self, args):
        super(ValData, self).__init__()

        path, val_num = args["path"], args["val_num"]

        # ======================= 原有 val 数据 =======================
        authentic_names = []
        authentic_path = join(path, "authentic")
        with open(join(authentic_path, "authentic.txt")) as f:
            contents = f.readlines()
            for content in contents[:val_num]:
                authentic_names.append(join(authentic_path, content.strip()))
        authentic_cls = [0] * len(authentic_names)

        splice_names = []
        splice_path = join(path, "splice")
        with open(join(splice_path, "fake.txt")) as f:
            contents = f.readlines()
            for content in contents[:val_num]:
                splice_names.append(join(splice_path, content.strip()))
        splice_cls = [1] * len(splice_names)

        copymove_names = []
        copymove_path = join(path, "copymove")
        with open(join(copymove_path, "fake.txt")) as f:
            contents = f.readlines()
            for content in contents[:val_num]:
                copymove_names.append(join(copymove_path, content.strip()))
        copymove_cls = [2] * len(copymove_names)

        removal_names = []
        removal_path = join(path, "removal")
        with open(join(removal_path, "fake.txt")) as f:
            contents = f.readlines()
            for content in contents[:val_num]:
                removal_names.append(join(removal_path, content.strip()))
        removal_cls = [3] * len(removal_names)

        self.image_names = authentic_names + splice_names + copymove_names + removal_names
        self.image_class = authentic_cls + splice_cls + copymove_cls + removal_cls

        # ======================= 新增 hard val（固定） =======================
        # hard_root = args.get("hard_path", "/mnt/c/datasets/datasets_hard")
        hard_root = args.get("hard_path", "/mnt/c/datasets/splice_columbia/")
        # hard_root = args.get("hard_path", "/mnt/c/datasets/datasets_hard_hard")
        self.hard_val_pick_req = int(args.get("hard_val_pick", 100))
        self.hard_cls_fixed = int(args.get("hard_cls_fixed", 1))

        # self.hard_fake_root = join(hard_root, "fake")
        # self.hard_mask_root = join(hard_root, "mask")
        self.hard_fake_root = join(hard_root, "fake_256")
        self.hard_mask_root = join(hard_root, "mask_256")

        self.hard_pairs_all = _pair_hard(self.hard_fake_root, self.hard_mask_root)

        # 固定 val：优先读 pairs 文件（两行一对 fake/mask）
        self.hard_val_pairs_file = args.get("hard_val_pairs_file", "")
        self.hard_val_seed = int(args.get("hard_val_seed", 54321))
        self.hard_auth_val_seed = int(args.get("hard_auth_val_seed", 12345))

        self.hard_val_pairs = self._build_fixed_hard_val_pairs()

        # ✅ hard 实际数量（避免 hard 空但 __len__ 还 +500）
        self.hard_val_pick = len(self.hard_val_pairs)

        # ======================= hard authentic val =======================
        self.hard_auth_roots = [
            # join(hard_root, "C1.0_Au_256"),
            # join(hard_root, "C2.0_Au_256"),
            join(hard_root, "4cam_auth_256"),
            # join(hard_root, "Cover_Au_256")
        ]

        self.hard_auth_all = []
        for r in self.hard_auth_roots:
            if os.path.exists(r):
                self.hard_auth_all += _list_images_recursive(r)

        self.hard_auth_val_pick_req = int(args.get("hard_auth_val_pick", 100))

        rng = random.Random(self.hard_auth_val_seed)
        if len(self.hard_auth_all) >= self.hard_auth_val_pick_req:
            self.hard_auth_val = rng.sample(self.hard_auth_all, self.hard_auth_val_pick_req)
        else:
            self.hard_auth_val = self.hard_auth_all[:]

    def _build_fixed_hard_val_pairs(self):
        # 1) 从文件读取（最稳）
        p = self.hard_val_pairs_file
        if p and os.path.exists(p):
            pairs = []
            with open(p, "r", encoding="utf-8") as f:
                lines = [x.strip() for x in f.readlines() if x.strip()]
            if len(lines) % 2 != 0:
                raise ValueError(f"hard_val_pairs_file 行数必须为偶数(两行一对): {p}")
            for i in range(0, len(lines), 2):
                pairs.append((lines[i], lines[i + 1]))

            if len(pairs) == 0:
                return []

            # 截断/补齐到 hard_val_pick_req
            if len(pairs) >= self.hard_val_pick_req:
                return pairs[: self.hard_val_pick_req]
            out = pairs[:]
            while len(out) < self.hard_val_pick_req:
                out.append(pairs[len(out) % len(pairs)])
            return out

        # 2) 没文件就固定 seed 抽样一次
        n = len(self.hard_pairs_all)
        if n == 0:
            return []

        rng = random.Random(self.hard_val_seed)
        if n >= self.hard_val_pick_req:
            return rng.sample(self.hard_pairs_all, self.hard_val_pick_req)
        else:
            return [rng.choice(self.hard_pairs_all) for _ in range(self.hard_val_pick_req)]

    def dump_epoch0_lists(self, out_dir: str, epoch: int = 0, print_n: int = 30):
        """
        把当前 ValData 实际会用到的样本列表（含 hard fake / hard authentic）写到文件，方便 double check。
        输出：
        val_base_epoch{epoch}.txt        # 原有 val（auth/splice/copymove/removal）
        val_hard_fake_epoch{epoch}.txt  # hard fake（fake/mask 成对）
        val_hard_auth_epoch{epoch}.txt  # hard authentic（仅原图）
        """
        import os
        os.makedirs(out_dir, exist_ok=True)

        base_path = os.path.join(out_dir, f"val_base_epoch{epoch}.txt")
        hard_fake_path = os.path.join(out_dir, f"val_hard_fake_epoch{epoch}.txt")
        hard_auth_path = os.path.join(out_dir, f"val_hard_auth_epoch{epoch}.txt")

        # ---------- base val ----------
        with open(base_path, "w", encoding="utf-8") as f:
            for p, c in zip(self.image_names, self.image_class):
                f.write(f"{int(c)}\t{p}\n")

        # ---------- hard fake val ----------
        with open(hard_fake_path, "w", encoding="utf-8") as f:
            for fake_p, mask_p in getattr(self, "hard_val_pairs", []):
                f.write(f"{int(self.hard_cls_fixed)}\t{fake_p}\t{mask_p}\n")

        # ---------- hard authentic val ----------
        with open(hard_auth_path, "w", encoding="utf-8") as f:
            for p in getattr(self, "hard_auth_val", []):
                f.write(f"0\t{p}\n")  # cls=0 表示 authentic

        # ---------- 打印前 print_n 条 ----------
        def _print_head(title: str, fp: str):
            print(f"\n[DEBUG] {title}: {fp}")
            try:
                with open(fp, "r", encoding="utf-8") as rf:
                    for i, line in enumerate(rf):
                        if i >= print_n:
                            break
                        print(line.rstrip("\n"))
            except Exception as e:
                print(f"[DEBUG] cannot read {fp}: {e}")

        _print_head("VAL_BASE_HEAD", base_path)
        _print_head("VAL_HARD_FAKE_HEAD", hard_fake_path)
        _print_head("VAL_HARD_AUTH_HEAD", hard_auth_path)

        print(f"\n[DEBUG] dump_epoch0_lists done -> {out_dir}")
        print(
            f"[DEBUG] base_val={len(self.image_names)} "
            f"hard_fake_val={len(getattr(self, 'hard_val_pairs', []))} "
            f"hard_auth_val={len(getattr(self, 'hard_auth_val', []))}"
        )


    def set_epoch(self, epoch: int):
        # ✅ val 固定：什么都不做（保留接口）
        return

    def get_item(self, index):
        # ---------------- 原 val ----------------
        if index < len(self.image_names):
            image_name = self.image_names[index]
            cls = self.image_class[index]

            image = _read_image_rgb(image_name)  # HxWx3
            H, W = image.shape[:2]

            if cls == 0:
                mask = np.zeros((H, W), dtype=np.uint8)
            elif cls == 1:
                # splice：fake.jpg -> mask.png (你原来的逻辑)
                # 兼容 jpg/tif
                if image_name.lower().endswith(".jpg") or image_name.lower().endswith(".jpeg"):
                    mask_name = image_name.replace("fake", "mask").rsplit(".", 1)[0] + ".png"
                else:
                    # 例如 tif
                    mask_name = image_name.replace("fake", "mask").rsplit(".", 1)[0] + ".png"
                mask = _read_mask_gray(mask_name)
            elif cls == 2:
                # copymove：路径里 fake -> mask（你原来的逻辑）
                mask_name = image_name.replace("fake", "mask")
                mask = _read_mask_gray(mask_name)
            elif cls == 3:
                # removal：fake.xxx -> mask.png
                mask_name = image_name.replace("fake", "mask").rsplit(".", 1)[0] + ".png"
                mask = _read_mask_gray(mask_name)
            else:
                raise Exception("class is not defined!")

            # 对齐尺寸（防止个别数据不一致）
            if mask.shape[0] != H or mask.shape[1] != W:
                mask = _resize_mask_nearest(mask, (H, W))

            # tensor
            image_t = torch.from_numpy(image.astype(np.float32) / 255.0).permute(2, 0, 1)  # [3,H,W]
            mask_t = torch.from_numpy(mask.astype(np.float32) / 255.0)                     # [H,W]
            return image_t, mask_t, cls, image_name

        # hard fake val
        hard_idx = index - len(self.image_names)
        if 0 <= hard_idx < self.hard_val_pick:
            fake_path, mask_path = self.hard_val_pairs[hard_idx]
            image = _read_image_rgb(fake_path)
            H, W = image.shape[:2]
            mask = _read_mask_gray(mask_path)

            if mask.shape[:2] != (H, W):
                mask = _resize_mask_nearest(mask, (H, W))

            image_t = torch.from_numpy(image.astype(np.float32) / 255.0).permute(2, 0, 1)
            mask_t  = torch.from_numpy(mask.astype(np.float32) / 255.0)

            cls = 1  # fake
            return image_t, mask_t, cls, fake_path

        # hard authentic val
        hard_auth_idx = index - len(self.image_names) - self.hard_val_pick
        if 0 <= hard_auth_idx < len(self.hard_auth_val):
            image_name = self.hard_auth_val[hard_auth_idx]
            image = _read_image_rgb(image_name)
            H, W = image.shape[:2]
            mask = np.zeros((H, W), dtype=np.uint8)

            image_t = torch.from_numpy(image.astype(np.float32) / 255.0).permute(2, 0, 1)
            mask_t  = torch.from_numpy(mask.astype(np.float32))  # 全 0 没问题

            cls = 0
            return image_t, mask_t, cls, image_name


        raise IndexError(f"index out of hard val range: {index}")

    def __getitem__(self, index):
        return self.get_item(index)

    def __len__(self):
        return (
            len(self.image_names)
            + self.hard_val_pick
            + len(self.hard_auth_val)
        )
