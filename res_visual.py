import os
import shutil
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import imageio.v2 as imageio


# ------------------------- 低开销工具函数 -------------------------
_THRESH = 255*0.90  # int(255*0.90 + 0.5)，与 (mask > 255*0.90) 对 uint8 等价

def load_binary_mask(path: str) -> np.ndarray:
    """读取并二值化 mask, 返回 uint8 的 2D (0/1)"""
    mask = imageio.imread(path)
    if mask.ndim == 3:
        mask = mask[:, :, 0]
    # 保持和原逻辑一致：> 229.5 等价于 >= 230（对 uint8）
    return (mask >= _THRESH).astype(np.uint8)


def _to_three_channel_uint8(img: np.ndarray) -> np.ndarray:
    """灰度/RGBA -> HxWx3 uint8，用于拼接可视化"""
    if img.ndim == 2:
        # broadcast_to 是 view（不拷贝），后面 concatenate 会拷贝成连续内存
        img = np.broadcast_to(img[:, :, None], (img.shape[0], img.shape[1], 3))
    elif img.ndim == 3 and img.shape[2] == 4:
        img = img[:, :, :3]

    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    return img


def _align_min_hw(a2d: np.ndarray, b2d: np.ndarray):
    """裁剪到共同最小尺寸（不 flatten）"""
    h = min(a2d.shape[0], b2d.shape[0])
    w = min(a2d.shape[1], b2d.shape[1])
    return a2d[:h, :w], b2d[:h, :w]


def _prf1_from_tp_fp_fn(tp: int, fp: int, fn: int):
    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0
    return p, r, f1


# ------------------------- 主流程 -------------------------
def compute_f1_scores(
    ref_txt,
    pred_root,
    save_path='f1_results.txt',
    case_root='cases',
    bad_threshold=0.55,
    good_threshold=0.85,
    original_root=None,
    max_workers=None,
):
    """
    - bad:    F1 < bad_threshold
    - normal: bad_threshold <= F1 < good_threshold
    - good:   F1 >= good_threshold

    所有样本都会保存拼接图: [ 原图 | GT | Pred ]
    同时输出 per-image 平均 P/R/F1 + micro(all-pixel) P/R/F1
    """

    if good_threshold <= bad_threshold:
        raise ValueError("good_threshold 必须大于 bad_threshold")

    pred_root = Path(pred_root)
    case_root = Path(case_root)

    # 清空 case_root
    if case_root.exists():
        shutil.rmtree(case_root)
    (case_root / 'good_cases').mkdir(parents=True, exist_ok=True)
    (case_root / 'normal_cases').mkdir(parents=True, exist_ok=True)
    (case_root / 'bad_cases').mkdir(parents=True, exist_ok=True)

    good_dir = case_root / 'good_cases'
    normal_dir = case_root / 'normal_cases'
    bad_dir = case_root / 'bad_cases'

    good_list_path = good_dir / 'good_cases_list.txt'
    normal_list_path = normal_dir / 'normal_cases_list.txt'
    bad_list_path = bad_dir / 'bad_cases_list.txt'

    with open(ref_txt, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f if line.strip()]

    # 线程数：I/O 为主，默认给个温和的并发
    if max_workers is None:
        cpu = os.cpu_count() or 8
        max_workers = min(16, cpu * 2)

    def _process_one(idx_line):
        """单样本处理：读mask->算指标->存可视化/坏例"""
        idx, line = idx_line
        img_path = Path(line)
        base_name = img_path.name
        stem = img_path.stem

        # GT mask: fake -> mask, 后缀换 .png
        ref_path = Path(str(img_path).replace('fake', 'mask'))
        ref_path = ref_path.with_suffix('.png')

        # Pred mask: pred_root/stem.png
        pred_path = pred_root / f"{stem}.png"

        if not pred_path.exists():
            return (idx, None, f"预测文件不存在: {pred_path}")
        if not ref_path.exists():
            return (idx, None, f"GT mask 不存在: {ref_path}")
        if not img_path.exists():
            return (idx, None, f"原图不存在: {img_path}")

        try:
            y_true_img = load_binary_mask(str(ref_path))
            y_pred_img = load_binary_mask(str(pred_path))
            y_true_img, y_pred_img = _align_min_hw(y_true_img, y_pred_img)

            # TP/FP/FN（numpy 统计比 sklearn 快很多）
            t = y_true_img.astype(bool, copy=False)
            p = y_pred_img.astype(bool, copy=False)
            tp = int(np.count_nonzero(t & p))
            fp = int(np.count_nonzero((~t) & p))
            fn = int(np.count_nonzero(t & (~p)))

            prec, rec, f1 = _prf1_from_tp_fp_fn(tp, fp, fn)

            # 分组
            if f1 < bad_threshold:
                target_dir = bad_dir
                case_tag = 'bad'
            elif f1 >= good_threshold:
                target_dir = good_dir
                case_tag = 'good'
            else:
                target_dir = normal_dir
                case_tag = 'normal'

            # 拼接图（所有 case 都做）
            orig_img = imageio.imread(str(img_path))
            orig_vis = _to_three_channel_uint8(orig_img)

            ref_vis = _to_three_channel_uint8((y_true_img * 255).astype(np.uint8, copy=False))
            pred_vis = _to_three_channel_uint8((y_pred_img * 255).astype(np.uint8, copy=False))

            h = min(orig_vis.shape[0], ref_vis.shape[0], pred_vis.shape[0])
            w = min(orig_vis.shape[1], ref_vis.shape[1], pred_vis.shape[1])
            orig_vis = orig_vis[:h, :w]
            ref_vis = ref_vis[:h, :w]
            pred_vis = pred_vis[:h, :w]

            concat_img = np.concatenate([orig_vis, ref_vis, pred_vis], axis=1)
            imageio.imwrite(str(target_dir / f"concat_{stem}.png"), concat_img)

            # bad case 额外保存
            if case_tag == 'bad':
                shutil.copy(str(ref_path), str(target_dir / f"ref_{ref_path.name}"))
                shutil.copy(str(pred_path), str(target_dir / f"pred_{pred_path.name}"))
                imageio.imwrite(str(target_dir / f"ref_bin_{ref_path.name}"), (y_true_img * 255).astype(np.uint8))
                imageio.imwrite(str(target_dir / f"pred_bin_{pred_path.name}"), (y_pred_img * 255).astype(np.uint8))
                shutil.copy(str(img_path), str(target_dir / f"img_{base_name}"))

            return (idx, (line, prec, rec, f1, tp, fp, fn, case_tag), None)

        except Exception as e:
            return (idx, None, f"处理失败: {line} 错误: {e}")

    # 并行处理
    results = []
    errors = []
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(_process_one, (i, line)) for i, line in enumerate(lines)]
        for fu in as_completed(futures):
            idx, payload, err = fu.result()
            if err is not None:
                errors.append((idx, err))
            elif payload is not None:
                results.append((idx, payload))

    # 按原顺序写结果
    results.sort(key=lambda x: x[0])

    good_lines, normal_lines, bad_lines = [], [], []
    out_lines = []

    p_scores, r_scores, f1_scores = [], [], []
    micro_tp = micro_fp = micro_fn = 0
    good_cnt = normal_cnt = bad_cnt = 0

    for _, (line, prec, rec, f1, tp, fp, fn, tag) in results:
        p_scores.append(prec)
        r_scores.append(rec)
        f1_scores.append(f1)
        micro_tp += tp
        micro_fp += fp
        micro_fn += fn

        out_lines.append(f"{line}\tP: {prec:.4f}\tR: {rec:.4f}\tF1: {f1:.4f}\n")

        one = f"{line}\tP: {prec:.4f}\tR: {rec:.4f}\tF1: {f1:.4f}\n"
        if tag == 'bad':
            bad_lines.append(one); bad_cnt += 1
        elif tag == 'good':
            good_lines.append(one); good_cnt += 1
        else:
            normal_lines.append(one); normal_cnt += 1

    # 写 txt（一次性写更快）
    with open(save_path, 'w', encoding='utf-8') as out_file:
        out_file.writelines(out_lines)

        if f1_scores:
            avg_p = float(np.mean(p_scores))
            avg_r = float(np.mean(r_scores))
            avg_f1 = float(np.mean(f1_scores))

            micro_p = micro_tp / (micro_tp + micro_fp) if (micro_tp + micro_fp) > 0 else 0.0
            micro_r = micro_tp / (micro_tp + micro_fn) if (micro_tp + micro_fn) > 0 else 0.0
            micro_f1 = (2 * micro_tp) / (2 * micro_tp + micro_fp + micro_fn) if (2 * micro_tp + micro_fp + micro_fn) > 0 else 0.0

            out_file.write("\n")
            out_file.write(f"Average (per-image)  P: {avg_p:.4f}\tR: {avg_r:.4f}\tF1: {avg_f1:.4f}\n")
            out_file.write(f"Micro   (all-pixel) P: {micro_p:.4f}\tR: {micro_r:.4f}\tF1: {micro_f1:.4f}\n")
            out_file.write(f"Good: {good_cnt}, Normal: {normal_cnt}, Bad: {bad_cnt}\n")
        else:
            out_file.write("\n未成功计算任何指标。\n")

        # 可选：把错误也写进去，方便排查
        if errors:
            errors.sort(key=lambda x: x[0])
            out_file.write("\n[WARN] Skipped / Errors:\n")
            for _, msg in errors:
                out_file.write(msg + "\n")

    # 分组列表
    good_list_path.write_text(''.join(good_lines), encoding='utf-8')
    normal_list_path.write_text(''.join(normal_lines), encoding='utf-8')
    bad_list_path.write_text(''.join(bad_lines), encoding='utf-8')

    # 控制台输出
    if f1_scores:
        print(f"平均(逐图) P/R/F1: {avg_p:.4f} / {avg_r:.4f} / {avg_f1:.4f}")
        print(f"Micro(全像素) P/R/F1: {micro_p:.4f} / {micro_r:.4f} / {micro_f1:.4f}")
        print(f"Good: {good_cnt}, Normal: {normal_cnt}, Bad: {bad_cnt}")
        if errors:
            print(f"[WARN] 跳过/错误样本数: {len(errors)}")
    else:
        print("未成功计算任何指标。")


if __name__ == "__main__":
    compute_f1_scores(
        ref_txt="./sampled_files_allTemp.txt",
        pred_root='mask_results/result/',
        save_path='f1_results.txt',
        case_root='cases_all',
        bad_threshold=0.55,
        good_threshold=0.85,
        original_root=None,
        max_workers=128,  # 需要更快可手动设 8/16/32 试试（看磁盘IO）
    )
