#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def iter_images(in_dir: Path):
    for p in in_dir.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            yield p


def _save_image_fast(im: Image.Image, dst: Path, is_mask: bool, png_compress_level: int, jpg_quality: int):
    """
    更快保存：
    - PNG: compress_level 低一点更快（文件更大一些）
    - JPG: quality 控制
    """
    suf = dst.suffix.lower()
    if suf == ".png":
        # compress_level: 0最快(最大)，9最慢(最小)
        im.save(dst, format="PNG", compress_level=png_compress_level)
    elif suf in (".jpg", ".jpeg"):
        im.save(dst, format="JPEG", quality=jpg_quality, optimize=False)  # optimize=True 很耗时
    else:
        # 其他格式直接保存
        im.save(dst)


def _process_one(src: Path, dst: Path, size: int, is_mask: bool,
                 png_compress_level: int, jpg_quality: int, skip_exist: bool):
    if skip_exist and dst.exists():
        return "skip"

    try:
        with Image.open(src) as im:
            if is_mask:
                # mask：离散标签，用 NEAREST
                if im.mode != "L":
                    im = im.convert("L")
                im = im.resize((size, size), resample=Image.NEAREST)
            else:
                # fake：一般 RGB + BILINEAR
                if im.mode == "RGBA":
                    im = im.convert("RGB")
                elif im.mode != "RGB":
                    im = im.convert("RGB")
                im = im.resize((size, size), resample=Image.BILINEAR)

            ensure_dir(dst.parent)
            _save_image_fast(im, dst, is_mask=is_mask,
                             png_compress_level=png_compress_level,
                             jpg_quality=jpg_quality)
        return "ok"
    except Exception as e:
        return f"err: {e}"


def resize_folder_fast(
    in_dir: Path,
    out_dir: Path,
    size: int = 256,
    is_mask: bool = False,
    workers: int | None = None,
    png_compress_level: int = 1,
    jpg_quality: int = 95,
    skip_exist: bool = True,
):
    in_dir = in_dir.resolve()
    out_dir = out_dir.resolve()
    ensure_dir(out_dir)

    # 先收集任务（扫描一次）
    tasks = []
    for src in iter_images(in_dir):
        rel = src.relative_to(in_dir)
        dst = out_dir / rel
        tasks.append((src, dst))

    total = len(tasks)
    print(f"[SCAN] {in_dir} -> {total} files")

    if workers is None:
        # I/O + 解码混合，线程数可以略高一点
        workers = min(32, (os.cpu_count() or 8) * 2)

    ok = 0
    skip = 0
    err = 0

    # 并行处理
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [
            ex.submit(
                _process_one, src, dst, size, is_mask,
                png_compress_level, jpg_quality, skip_exist
            )
            for src, dst in tasks
        ]

        done_n = 0
        for fut in as_completed(futs):
            done_n += 1
            r = fut.result()
            if r == "ok":
                ok += 1
            elif r == "skip":
                skip += 1
            else:
                err += 1
                # 打印少量错误即可，避免刷屏
                if err <= 20:
                    print(f"[ERR] {r}")

            # 进度提示（每 500 个打印一次）
            if done_n % 500 == 0 or done_n == total:
                print(f"[PROG] {done_n}/{total} | ok={ok} skip={skip} err={err}")

    print(f"[DONE] {in_dir} -> {out_dir} | ok={ok}, skip={skip}, err={err}, workers={workers}")


def main():
    # datasets\COVERAGE
    base = Path(r"/mnt/c/datasets/COVERAGE/")
    fake_in = base / "image"
    mask_in = base / "mask"
    fake_out = base / "image_256"
    mask_out = base / "mask_256"

    # 你可以按机器调 workers，比如 16/32
    resize_folder_fast(
        fake_in, fake_out,
        size=256, is_mask=False,
        workers=None,              # None 自动
        png_compress_level=1,      # 更快
        jpg_quality=95,
        skip_exist=True,
    )
    resize_folder_fast(
        mask_in, mask_out,
        size=256, is_mask=True,
        workers=None,
        png_compress_level=1,
        jpg_quality=95,
        skip_exist=True,
    )


if __name__ == "__main__":
    main()
