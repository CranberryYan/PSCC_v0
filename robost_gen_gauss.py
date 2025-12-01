# blur_gauss.py
import os
from PIL import Image, ImageFilter

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

def to_out_path(out_root: str, rel_path: str):
    # 保留原始扩展名（png 还是 png，jpg 还是 jpg）
    return os.path.join(out_root, rel_path)

def main():
    # ===== 你手动控制的参数 =====
    sigma = 3   # 高斯模糊标准差（float，>0）
    list_path = "sampled_files.txt"
    out_base = "./"
    # ==========================

    if sigma <= 0:
        raise ValueError("sigma 必须 > 0")

    in_paths = read_list(list_path)
    if not in_paths:
        raise RuntimeError("txt 里没有读到任何路径")

    # 输出目录：Robost_Gauss_<sigma>
    # 目录名里避免小数点太长，做个格式化
    sigma_tag = f"{sigma:.3f}".rstrip("0").rstrip(".")
    out_root = os.path.join(out_base, f"Robost_Gauss_{sigma_tag}")
    ensure_dir(out_root)

    # 用于保留目录结构
    base_root = os.path.commonpath([os.path.abspath(p) for p in in_paths])

    ok, fail = 0, 0
    for i, p in enumerate(in_paths, 1):
        try:
            p_abs = os.path.abspath(p)
            ext = os.path.splitext(p_abs)[1].lower()
            if ext and ext not in IMG_EXTS:
                # 不认识的后缀也可以尝试打开；这里不强制跳过
                pass

            rel = os.path.relpath(p_abs, base_root)
            out_path = to_out_path(out_root, rel)
            ensure_dir(os.path.dirname(out_path))

            with Image.open(p_abs) as im:
                # 保持原模式（比如 PNG 的 RGBA），避免丢 alpha
                blurred = im.filter(ImageFilter.GaussianBlur(radius=sigma))
                # 按原格式保存：由扩展名决定
                blurred.save(out_path)

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
