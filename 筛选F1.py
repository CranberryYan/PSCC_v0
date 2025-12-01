#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import ast
import re

# ================= 文件路径 =================
# base = Path("./total=15000_epoch=100_batch=6_with_MoE=True(MoE_attn=CBAM_K=8)_With_HiLo=True/")
# K1_path = base / "K1/f1_results_K1.txt"
# K2_path = base / "K2/f1_results_K2.txt"
# K3_path = base / "K3/f1_results_K3.txt"
# K4_path = base / "K4/f1_results_K4.txt"
# K5_path = base / "K5/f1_results_K5.txt"
# K6_path = base / "K6/f1_results_K6.txt"
# K7_path = base / "K7/f1_results_K7.txt"
# K8_path = base / "K8/f1_results_K8.txt"
K1_path = "./待筛选F1/f1_results_CNN+注意力.txt"
K2_path = "./待筛选F1/f1_results_MoE+普通卷积.txt"
K3_path = "./待筛选F1/f1_results_本研究.txt"
K4_path = "./待筛选F1/f1_results_判别器+普通卷积.txt"


def extract_metric(items, key: str):
    """
    items: list[str] 或者可迭代字符串token
    key: 'F1' / 'IoU' / 'P' / 'R'
    返回 float 或 None
    支持: 'F1:0.8238'、'F1=0.8238'、'F1 0.8238'
    """
    # 1) 先找类似 'F1:0.8238'
    for s in items:
        if not isinstance(s, str):
            continue
        if s.startswith(key + ":"):
            try:
                return float(s.split(":", 1)[1])
            except:
                pass
        if s.startswith(key + "="):
            try:
                return float(s.split("=", 1)[1])
            except:
                pass

    # 2) 再用正则兜底（整行拼起来找）
    text = " ".join([str(x) for x in items])
    m = re.search(rf"{re.escape(key)}\s*[:=]\s*([0-9]*\.?[0-9]+)", text)
    if m:
        return float(m.group(1))

    # 3) 支持 'F1 0.8238' 这种
    for i, s in enumerate(items):
        if s == key and i + 1 < len(items):
            try:
                return float(items[i + 1])
            except:
                pass
    return None


def parse_f1_file(path: Path):
    """
    返回字典：sample_id -> (tag, f1)
    sample_id 建议用相对路径(更不容易重名)，例如 'Dataset/.../93956.png'
    """
    d = {}
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            # === 解析一行：优先按 Python list 字符串解析 ===
            items = None
            if line.startswith("[") and line.endswith("]"):
                try:
                    items = ast.literal_eval(line)
                except Exception:
                    items = None

            if items is None:
                # 兜底：普通文本就 split
                items = line.replace(",", " ").split()

            # === 你的格式里：items[0] 是 root，items[1] 才是图片路径 ===
            if isinstance(items, list) and len(items) >= 2:
                sample_path = str(items[1])
            else:
                sample_path = str(items[0])

            # 用“相对路径字符串”做 key，避免 basename 重名
            sample_id = sample_path.replace("\\", "/")

            f1_value = extract_metric(items, "F1")
            if f1_value is None:
                print(f"[WARN] 第{line_no}行未找到F1，跳过：{line[:120]}")
                continue

            # === 分档 ===
            if f1_value < 0.55:
                tag = "bad"
            elif f1_value >= 0.85:
                tag = "good"
            else:
                tag = "normal"

            d[sample_id] = (tag, f1_value)

    return d


# ================= 读取八个文件 =================
f1_dict = parse_f1_file(K1_path)
f2_dict = parse_f1_file(K2_path)
f3_dict = parse_f1_file(K3_path)
f4_dict = parse_f1_file(K4_path)
# f5_dict = parse_f1_file(K5_path)
# f6_dict = parse_f1_file(K6_path)
# f7_dict = parse_f1_file(K7_path)
# f8_dict = parse_f1_file(K8_path)

# ================= 筛选条件 =================
selected_samples = []

# keys = set(f1_dict) & set(f2_dict) & set(f3_dict) & set(f4_dict) & set(f5_dict) & set(f6_dict) & set(f7_dict) & set(f8_dict)
keys = set(f1_dict) & set(f2_dict) & set(f3_dict) & set(f4_dict)

for sample_id in keys:
    # cond1 = (f1_dict[sample_id][0] in ("normal", "good"))
    # cond2 = (f2_dict[sample_id][0] in ("normal", "good"))
    # cond3 = (f3_dict[sample_id][0] in ("normal", "bad"))
    # cond4 = (f3_dict[sample_id][0] in ("normal", "bad"))
    # cond5 = (f3_dict[sample_id][0] in ("normal", "bad"))
    # cond6 = (f3_dict[sample_id][0] in ("bad", "bad"))
    # cond7 = (f3_dict[sample_id][0] in ("bad", "bad"))
    # cond8 = (f3_dict[sample_id][0] in ("bad", "bad"))

    # if cond1 and cond2 and cond3 and cond4 and cond5 and cond6 and cond7 and cond8:
    #     selected_samples.append(sample_id)

    cond1 = (f1_dict[sample_id][0] in ("bad"))
    cond2 = (f2_dict[sample_id][0] in ("bad"))
    cond3 = (f3_dict[sample_id][0] in ("good", "normal"))
    cond4 = (f3_dict[sample_id][0] in ("normal", "bad"))

    if cond1 and cond2 and cond3 and cond4:
        selected_samples.append(sample_id)

# ================= 输出结果 =================
output_path = "selected_samples.txt"
with open(output_path, "w", encoding="utf-8") as f:
    for s in sorted(selected_samples):
        f.write(s + "\n")

print(f"筛选完成，符合条件的样本数: {len(selected_samples)}")
print(f"结果已保存到 {output_path}")
