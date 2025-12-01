# filter_copy_c1_with_mask.py
import argparse
import os
import shutil
from pathlib import Path

IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp")

def norm_path(p: str) -> Path:
    p = p.strip().replace("\\", "/")
    p = os.path.normpath(p)
    return Path(p)

def safe_copy(src: Path, dst: Path, try_hardlink: bool = False):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return "skip_exists"
    try:
        if try_hardlink:
            os.link(src, dst)
            return "hardlink"
        shutil.copy2(src, dst)
        return "copy"
    except Exception:
        if try_hardlink:
            shutil.copy2(src, dst)
            return "copy"
        raise

def find_corresponding_mask(src_root: Path, rel_under_fake: Path) -> Path | None:
    """
    rel_under_fake: fake/ 下面的相对路径（不包含 fake 前缀），例如:
      Sp_D_CND_A_pla0005_pla0023_0281.png
      subdir/xxx.jpg
    在 src_root/mask 或 src_root/masks 下找对应 mask：
      1) 完全同路径同文件名
      2) 同路径同 stem，优先 .png，其它扩展名兜底
      3) stem + '_mask' 的常见命名
    """
    mask_dirs = [src_root / "mask", src_root / "masks"]

    stem = rel_under_fake.stem
    suffix = rel_under_fake.suffix.lower()

    # 1) 同路径同名
    for md in mask_dirs:
        cand = md / rel_under_fake
        if cand.exists() and cand.is_file():
            return cand

    # 2) 同路径同 stem，不同扩展名（优先 png）
    exts_try = (".png",) + tuple(e for e in IMG_EXTS if e != ".png")
    for md in mask_dirs:
        parent = (md / rel_under_fake).parent
        for ext in exts_try:
            cand = parent / (stem + ext)
            if cand.exists() and cand.is_file():
                return cand

    # 3) 常见：xxx_mask.png / xxx-mask.png
    for md in mask_dirs:
        parent = (md / rel_under_fake).parent
        for pat in [f"{stem}_mask", f"{stem}-mask", f"{stem}.mask"]:
            for ext in exts_try:
                cand = parent / (pat + ext)
                if cand.exists() and cand.is_file():
                    return cand

    return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--txt", default="./datasets_CASIA1.0.txt", help="你的txt文件路径（每行一个样本路径）")
    ap.add_argument("--src_root", default="/mnt/c/datasets/datasets_hard", help="源数据根目录")
    ap.add_argument("--dst_root", default="/mnt/c/datasets/datasets_hard_C1", help="输出根目录")
    ap.add_argument("--hardlink", action="store_true", help="优先用硬链接（失败自动copy）")
    args = ap.parse_args()

    src_root = Path(os.path.normpath(args.src_root))
    dst_root = Path(os.path.normpath(args.dst_root))

    dst_fake = dst_root / "fake"
    dst_au = dst_root / "C1.0_Au_256"
    dst_mask = dst_root / "mask"

    txt_path = Path(args.txt)
    assert txt_path.exists(), f"txt 不存在: {txt_path}"

    total = 0
    kept = 0
    ignored = 0

    missing_imgs = []
    missing_masks = []

    copied_imgs = 0
    copied_masks = 0
    hardlinked_imgs = 0
    hardlinked_masks = 0
    skipped_imgs = 0
    skipped_masks = 0

    with txt_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            total += 1

            src = norm_path(line)

            # 必须在 src_root 下
            try:
                rel = src.relative_to(src_root)
            except Exception:
                ignored += 1
                continue

            if len(rel.parts) == 0:
                ignored += 1
                continue

            top = rel.parts[0]

            # 情况A：fake 图像（要拷贝图像 + 对应mask）
            if top == "fake":
                kept += 1
                rel_under_fake = Path(*rel.parts[1:])  # fake/ 后面的部分

                # 1) 拷贝 fake 图像
                dst_img = dst_fake / rel_under_fake
                if not src.exists() or not src.is_file():
                    missing_imgs.append(str(src))
                else:
                    r = safe_copy(src, dst_img, try_hardlink=args.hardlink)
                    if r == "copy": copied_imgs += 1
                    elif r == "hardlink": hardlinked_imgs += 1
                    elif r == "skip_exists": skipped_imgs += 1

                # 2) 找并拷贝对应 mask
                msrc = find_corresponding_mask(src_root, rel_under_fake)
                if msrc is None:
                    # 记录：给出“期望的相对路径”方便你排查
                    missing_masks.append(f"(from fake) {src} -> (mask not found under {src_root}/mask or {src_root}/masks)")
                else:
                    # mask 输出路径：dst_root/mask/ 下保持 rel_under_fake 的目录结构
                    # 文件名用实际找到的 mask 文件名（可能扩展名不同）
                    dst_m = (dst_mask / rel_under_fake).with_suffix(msrc.suffix)
                    r = safe_copy(msrc, dst_m, try_hardlink=args.hardlink)
                    if r == "copy": copied_masks += 1
                    elif r == "hardlink": hardlinked_masks += 1
                    elif r == "skip_exists": skipped_masks += 1

            # 情况B：C1.0_Au_256（只拷贝图像）
            elif top == "C1.0_Au_256":
                kept += 1
                rel_under = Path(*rel.parts[1:])
                dst_img = dst_au / rel_under
                if not src.exists() or not src.is_file():
                    missing_imgs.append(str(src))
                else:
                    r = safe_copy(src, dst_img, try_hardlink=args.hardlink)
                    if r == "copy": copied_imgs += 1
                    elif r == "hardlink": hardlinked_imgs += 1
                    elif r == "skip_exists": skipped_imgs += 1

            # 情况C：如果 txt 里本来就有 mask/masks 路径，也顺带拷贝到 dst_root/mask
            elif top in ("mask", "masks"):
                kept += 1
                rel_under = Path(*rel.parts[1:])
                dst_m = dst_mask / rel_under
                if not src.exists() or not src.is_file():
                    missing_masks.append(str(src))
                else:
                    r = safe_copy(src, dst_m, try_hardlink=args.hardlink)
                    if r == "copy": copied_masks += 1
                    elif r == "hardlink": hardlinked_masks += 1
                    elif r == "skip_exists": skipped_masks += 1

            else:
                ignored += 1
                continue

    dst_root.mkdir(parents=True, exist_ok=True)

    (dst_root / "missing_imgs.txt").write_text("\n".join(missing_imgs) + ("\n" if missing_imgs else ""), encoding="utf-8")
    (dst_root / "missing_masks.txt").write_text("\n".join(missing_masks) + ("\n" if missing_masks else ""), encoding="utf-8")

    print("====== DONE ======")
    print(f"txt lines total           : {total}")
    print(f"kept (fake/C1.0_Au_256/mask) : {kept}")
    print(f"ignored                   : {ignored}")
    print("")
    print(f"[Images] copied={copied_imgs} hardlinked={hardlinked_imgs} skipped={skipped_imgs} missing={len(missing_imgs)} -> {dst_root/'missing_imgs.txt'}")
    print(f"[Masks ] copied={copied_masks} hardlinked={hardlinked_masks} skipped={skipped_masks} missing={len(missing_masks)} -> {dst_root/'missing_masks.txt'}")
    print("")
    print(f"output root: {dst_root}")
    print(f"  - {dst_fake}")
    print(f"  - {dst_au}")
    print(f"  - {dst_mask}")

if __name__ == "__main__":
    main()
