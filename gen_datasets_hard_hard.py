#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import re
import argparse
import shutil
from pathlib import Path
from typing import List, Set, Dict

IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp")

# 匹配一行里出现的“文件名或路径”，取第一个图像后缀的token
# 例如：hard_fake_xxx.png cls=1 f1=0.1407
PAT = re.compile(r'([^\s]+?\.(?:png|jpg|jpeg|bmp|tif|tiff|webp))', re.IGNORECASE)


def parse_filenames(txt_path: Path) -> List[str]:
    names = []
    for line in txt_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line:
            continue
        m = PAT.search(line)
        if not m:
            continue
        token = m.group(1).strip()
        # 你说“解析文件名”，这里取 basename
        names.append(Path(token).name)
    return names


def derive_mask_name(fake_name: str) -> str:
    """
    经验规则：hard_fake_xxx 与 hard_mask_xxx 对应
    你示例里还包含 datasets_hard_fake / _fake_ 字样，所以也一起替换成 mask
    """
    mask = fake_name

    # 只替换第一个前缀 hard_fake_
    if mask.startswith("hard_fake_"):
        mask = mask.replace("hard_fake_", "hard_mask_", 1)

    # 替换路径编码里的 fake -> mask（更像你数据里存的命名方式）
    mask = mask.replace("datasets_hard_fake", "datasets_hard_mask")
    mask = mask.replace("_fake_", "_mask_")

    return mask


def build_hit_map(src_root: Path, want_names: Set[str]) -> Dict[str, List[Path]]:
    """
    只遍历一次 src_root，把命中的文件路径收集起来（避免为每个 name rglob）
    返回：name -> [fullpath, ...]
    """
    hits: Dict[str, List[Path]] = {n: [] for n in want_names}

    for p in src_root.rglob("*"):
        if not p.is_file():
            continue
        if p.suffix.lower() not in IMG_EXTS:
            continue
        name = p.name
        if name in hits:
            hits[name].append(p)

    return hits


def copy_preserve_rel(src_root: Path, dst_root: Path, src_file: Path):
    rel = src_file.relative_to(src_root)
    dst_file = dst_root / rel
    dst_file.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src_file, dst_file)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--txt", default="./hard_train_mining/meta.txt", help="包含 hard 样本记录的 txt（每行含 xxx.png cls=... f1=...）")
    ap.add_argument("--src", default="/mnt/c/datasets/datasets_hard", help="源数据根目录（只读）")
    ap.add_argument("--dst", default="/mnt/c/datasets/datasets_hard_hard", help="目标目录（写入复制文件）")
    ap.add_argument("--copy_pair", action="store_true", help="同时复制对应 hard_mask（从 hard_fake 推导）")
    ap.add_argument("--missing_log", default="missing_files.txt", help="缺失文件记录输出")
    args = ap.parse_args()

    txt_path = Path(args.txt)
    src_root = Path(args.src)
    dst_root = Path(args.dst)

    assert txt_path.exists(), f"txt 不存在: {txt_path}"
    assert src_root.exists(), f"src 不存在: {src_root}"
    dst_root.mkdir(parents=True, exist_ok=True)

    fake_names = parse_filenames(txt_path)
    print("fake_names:", fake_names[:5])
    if not fake_names:
        raise RuntimeError("未从 txt 解析到任何图片文件名（检查格式/后缀）")

    want: Set[str] = set(fake_names)

    pair_map = {}
    if args.copy_pair:
        for fn in fake_names:
            mn = derive_mask_name(fn)
            pair_map[fn] = mn
            want.add(mn)

    print(f"[INFO] parsed fake names: {len(fake_names)}")
    print(f"[INFO] total wanted files (with pair={args.copy_pair}): {len(want)}")

    hits = build_hit_map(src_root, want)

    copied = 0
    missing = []

    # 按 txt 顺序复制 fake（以及可选 mask）
    for fn in fake_names:
        paths = hits.get(fn, [])
        if not paths:
            missing.append(fn)
        else:
            for p in paths:
                copy_preserve_rel(src_root, dst_root, p)
                copied += 1

        if args.copy_pair:
            mn = pair_map[fn]
            mpaths = hits.get(mn, [])
            if not mpaths:
                missing.append(mn)
            else:
                for mp in mpaths:
                    copy_preserve_rel(src_root, dst_root, mp)
                    copied += 1

    # 去重缺失列表
    missing_uniq = sorted(set(missing))
    if missing_uniq:
        Path(args.missing_log).write_text("\n".join(missing_uniq) + "\n", encoding="utf-8")
        print(f"[WARN] missing files: {len(missing_uniq)} -> {args.missing_log}")

    print(f"[DONE] copied files: {copied}")
    print(f"[DONE] dst: {dst_root}")
    print("[NOTE] 源目录未做任何修改（仅 copy2 复制）。")


if __name__ == "__main__":
    main()
