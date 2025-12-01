# recompress_to_jpeg.py
import os
import argparse
from PIL import Image

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}

def read_list(txt_path: str):
    paths = []
    with open(txt_path, "r", encoding="utf-8") as f:
        for line in f:
            p = line.strip().strip('"').strip("'")
            if not p or p.startswith("#"):
                continue
            paths.append(p)
    return paths

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def to_jpg_path(out_root: str, rel_path: str):
    rel_no_ext = os.path.splitext(rel_path)[0]
    return os.path.join(out_root, rel_no_ext + ".jpg")

def main():
    quality = 95
    q = quality
    if not (1 <= q <= 100):
        raise ValueError("quality 必须在 1~100")

    list = "sampled_files.txt"
    in_paths = read_list(list)
    if not in_paths:
        raise RuntimeError("txt 里没有读到任何路径")

    # 输出目录：Robost_Jpeg_<质量因子>
    out_base = "./"
    out_root = os.path.join(out_base, f"Robost_Jpeg_{q}")
    ensure_dir(out_root)

    # 用于保留目录结构
    base_root = os.path.commonpath([os.path.abspath(p) for p in in_paths])

    ok, fail = 0, 0
    for i, p in enumerate(in_paths, 1):
        try:
            p_abs = os.path.abspath(p)
            ext = os.path.splitext(p_abs)[1].lower()
            if ext and ext not in IMG_EXTS:
                # 不是常见图像扩展名也允许尝试打开，但这里先跳过更安全
                pass

            rel = os.path.relpath(p_abs, base_root)

            out_path = to_jpg_path(out_root, rel)
            ensure_dir(os.path.dirname(out_path))

            with Image.open(p_abs) as im:
                # JPEG 需要 RGB / L 等模式，这里统一转 RGB（含 RGBA 会丢 alpha）
                if im.mode != "RGB":
                    im = im.convert("RGB")
                im.save(
                    out_path,
                    format="JPEG",
                    quality=q,
                    optimize=True,
                    subsampling=0,     # 4:4:4，尽量少引入额外损失
                    progressive=False
                )

            ok += 1
            if i % 200 == 0:
                print(f"[{i}/{len(in_paths)}] done... (ok={ok}, fail={fail})")

        except Exception as e:
            fail += 1
            print(f"[FAIL] {p} -> {e}")

    print(f"\nFinished. out_dir = {out_root}")
    print(f"ok={ok}, fail={fail}")

if __name__ == "__main__":
    main()
