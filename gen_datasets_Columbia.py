#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
from pathlib import Path
from datetime import datetime

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}

def norm_line(s: str) -> str:
    s = s.strip()
    if not s:
        return ""
    # 修复类似: "hard_auth|/mnt/c/xxx/xxx.jpg.png"
    if "|" in s:
        s = s.split("|")[-1].strip()

    # 去掉可能的引号
    if (s.startswith('"') and s.endswith('"')) or (s.startswith("'") and s.endswith("'")):
        s = s[1:-1].strip()

    return s

def fix_double_ext(p: str) -> list[str]:
    """
    尝试把 xxx.jpg.png / xxx.jpeg.png / xxx.png.jpg 之类修回去
    返回候选路径列表（按优先级）
    """
    cands = [p]
    lower = p.lower()
    for a, b in [(".jpg.png", ".jpg"), (".jpeg.png", ".jpeg"), (".png.jpg", ".jpg"), (".png.jpeg", ".jpeg")]:
        if lower.endswith(a):
            cands.append(p[: -len(a)] + b)
    return list(dict.fromkeys(cands))

def scan_files(scan_root: Path) -> tuple[set[str], dict[str, list[str]]]:
    all_files: set[str] = set()
    by_name: dict[str, list[str]] = {}
    for fp in scan_root.rglob("*"):
        if not fp.is_file():
            continue
        if fp.suffix.lower() not in IMG_EXTS:
            continue
        f = str(fp)
        all_files.add(f)
        bn = fp.name
        by_name.setdefault(bn, []).append(f)
    return all_files, by_name

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--txt", default="./datasets_Columbia.txt", help="要清理的 txt 路径")
    ap.add_argument("--scan_root", default="/mnt/c/datasets/datasets_hard/4cam_auth_256/", help="要扫描的目录(递归)")
    ap.add_argument("--mode", choices=["filter", "rebuild"], default="rebuild",
                    help="filter: 按原txt过滤修复；rebuild: 直接用扫描结果重建txt")
    args = ap.parse_args()

    txt_path = Path(args.txt)
    scan_root = Path(args.scan_root)

    if not scan_root.exists():
        raise FileNotFoundError(f"scan_root 不存在: {scan_root}")
    if not txt_path.exists():
        raise FileNotFoundError(f"txt 不存在: {txt_path}")

    all_files, by_name = scan_files(scan_root)
    print(f"[SCAN] root={scan_root}")
    print(f"[SCAN] found {len(all_files)} image files")

    # 备份
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    bak = txt_path.with_suffix(txt_path.suffix + f".bak_{ts}")
    bak.write_text(txt_path.read_text(encoding="utf-8", errors="ignore"), encoding="utf-8")
    print(f"[BACKUP] {bak}")

    kept: list[str] = []
    fixed = 0
    dropped = 0
    dup = 0

    if args.mode == "rebuild":
        kept = sorted(all_files)
        txt_path.write_text("\n".join(kept) + ("\n" if kept else ""), encoding="utf-8")
        print(f"[WRITE] rebuild -> {txt_path} ({len(kept)} lines)")
        return

    # mode == filter：尽量按原 txt 修复/保留
    seen = set()
    lines = txt_path.read_text(encoding="utf-8", errors="ignore").splitlines()

    for raw in lines:
        s = norm_line(raw)
        if not s:
            continue

        candidates = fix_double_ext(s)

        chosen = None
        # 1) 直接存在
        for c in candidates:
            if os.path.exists(c):
                chosen = c
                if c != s:
                    fixed += 1
                break

        # 2) 不存在但 basename 在 scan_root 里能匹配到
        if chosen is None:
            bn = Path(candidates[-1]).name  # 用“最后一个修复候选”的basename更接近真实名
            if bn in by_name and len(by_name[bn]) >= 1:
                # 同名多路径时，默认取第一个；你也可以改成按更短路径/更近目录挑选
                chosen = by_name[bn][0]
                fixed += 1

        if chosen is None:
            dropped += 1
            continue

        if chosen in seen:
            dup += 1
            continue
        seen.add(chosen)
        kept.append(chosen)

    txt_path.write_text("\n".join(kept) + ("\n" if kept else ""), encoding="utf-8")
    print(f"[WRITE] filter -> {txt_path}")
    print(f"[STAT] kept={len(kept)} fixed={fixed} dropped={dropped} dup_removed={dup}")

if __name__ == "__main__":
    main()
