import os
import shutil
import numpy as np
import imageio.v2 as imageio
from sklearn.metrics import f1_score, precision_score, recall_score


# 读取并二值化mask, 返回二维二值图
def load_binary_mask(path):
    mask = imageio.imread(path)
    if mask.ndim == 3:
        mask = mask[:, :, 0]  # 若是三通道, 取第一个通道
    binary_mask = (mask > 255 * 0.95).astype(np.uint8)
    return binary_mask


def _to_three_channel_uint8(img):
    """把灰度 / RGBA 统一成 HxWx3 的 uint8, 方便拼接显示"""
    if img.ndim == 2:  # HxW
        img = np.stack([img] * 3, axis=-1)
    elif img.ndim == 3 and img.shape[2] == 4:  # HxWx4 -> 去掉 Alpha
        img = img[:, :, :3]
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    return img


def _align_and_flatten(a2d, b2d):
    """把两张 2D mask 裁剪到共同最小尺寸，并展平"""
    h_min = min(a2d.shape[0], b2d.shape[0])
    w_min = min(a2d.shape[1], b2d.shape[1])
    a2d = a2d[:h_min, :w_min]
    b2d = b2d[:h_min, :w_min]
    return a2d, b2d, a2d.reshape(-1), b2d.reshape(-1)


def compute_f1_scores(ref_txt,
                      pred_root,
                      save_path='f1_results.txt',
                      case_root='cases',
                      bad_threshold=0.55,
                      good_threshold=0.85,
                      original_root=None):
    """
    根据 F1 分数把所有样本分成三类:
      - bad case:   F1 < bad_threshold
      - normal case: bad_threshold <= F1 < good_threshold
      - good case:  F1 >= good_threshold

    并且无论 good / normal / bad, 都保存一张:
        [ 原图 | GT mask | Pred mask ]
    的拼接图, 方便论文展示。

    同时计算并输出 Precision / Recall / F1。
    """

    if good_threshold <= bad_threshold:
        raise ValueError("good_threshold 必须大于 bad_threshold")

    # 先清空 case_root
    if os.path.exists(case_root):
        shutil.rmtree(case_root)
    os.makedirs(case_root, exist_ok=True)

    # 三个子目录
    good_dir = os.path.join(case_root, 'good_cases')
    normal_dir = os.path.join(case_root, 'normal_cases')
    bad_dir = os.path.join(case_root, 'bad_cases')

    os.makedirs(good_dir, exist_ok=True)
    os.makedirs(normal_dir, exist_ok=True)
    os.makedirs(bad_dir, exist_ok=True)

    # 三个列表文件
    good_list_path = os.path.join(good_dir, 'good_cases_list.txt')
    normal_list_path = os.path.join(normal_dir, 'normal_cases_list.txt')
    bad_list_path = os.path.join(bad_dir, 'bad_cases_list.txt')

    with open(ref_txt, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]

    f1_scores = []
    p_scores = []
    r_scores = []

    good_cnt = normal_cnt = bad_cnt = 0

    # micro 汇总（全像素 TP/FP/FN）
    micro_tp = 0
    micro_fp = 0
    micro_fn = 0

    with open(save_path, 'w') as out_file, \
         open(good_list_path, 'w') as good_file, \
         open(normal_list_path, 'w') as normal_file, \
         open(bad_list_path, 'w') as bad_file:

        for line in lines:
            base_name = os.path.basename(line)     # xxx.tif
            stem, _ = os.path.splitext(base_name)  # xxx

            # -------- GT mask 路径: fake -> mask, 后缀统一换成 .png --------
            ref_path = line.replace('fake', 'mask')
            ref_path = os.path.splitext(ref_path)[0] + '.png'

            # -------- 预测 mask 路径: 在 pred_root/stem.png --------
            pred_path = os.path.join(pred_root, stem + '.png')

            if not os.path.exists(pred_path):
                print(f"预测文件不存在: {pred_path}")
                continue
            if not os.path.exists(ref_path):
                print(f"GT mask 不存在: {ref_path}")
                continue

            img_path = line
            if not os.path.exists(img_path):
                print(f"原图不存在: {img_path}")
                continue

            try:
                # 读取 GT / Pred mask（用于 P/R/F1）
                y_true_img = load_binary_mask(ref_path)
                y_pred_img = load_binary_mask(pred_path)

                # 尺寸对齐（很关键：避免 mask 尺寸不一致导致 metric 报错）
                y_true_img, y_pred_img, y_true, y_pred = _align_and_flatten(y_true_img, y_pred_img)

                # 三个指标（zero_division=0：全0时不报错，返回0）
                p = precision_score(y_true, y_pred, zero_division=0)
                r = recall_score(y_true, y_pred, zero_division=0)
                f1 = f1_score(y_true, y_pred, zero_division=0)

                p_scores.append(p)
                r_scores.append(r)
                f1_scores.append(f1)

                # micro 累积（全像素汇总）
                micro_tp += int(np.sum((y_true == 1) & (y_pred == 1)))
                micro_fp += int(np.sum((y_true == 0) & (y_pred == 1)))
                micro_fn += int(np.sum((y_true == 1) & (y_pred == 0)))

                # 写总结果
                out_file.write(f"{line}\tP: {p:.4f}\tR: {r:.4f}\tF1: {f1:.4f}\n")

                # -------- 按 F1 分组 --------
                if f1 < bad_threshold:
                    target_dir = bad_dir
                    target_file = bad_file
                    bad_cnt += 1
                    case_tag = 'bad'
                elif f1 >= good_threshold:
                    target_dir = good_dir
                    target_file = good_file
                    good_cnt += 1
                    case_tag = 'good'
                else:
                    target_dir = normal_dir
                    target_file = normal_file
                    normal_cnt += 1
                    case_tag = 'normal'

                # 写入对应分组列表
                target_file.write(f"{line}\tP: {p:.4f}\tR: {r:.4f}\tF1: {f1:.4f}\n")

                # ================== 拼接图生成（所有 case 都做） ==================
                orig_img = imageio.imread(img_path)
                orig_vis = _to_three_channel_uint8(orig_img)

                # GT / Pred 可视化：二值 * 255 -> 3 通道
                ref_vis = _to_three_channel_uint8(y_true_img * 255)
                pred_vis = _to_three_channel_uint8(y_pred_img * 255)

                # 与原图对齐（展示用）
                h_min = min(orig_vis.shape[0], ref_vis.shape[0], pred_vis.shape[0])
                w_min = min(orig_vis.shape[1], ref_vis.shape[1], pred_vis.shape[1])
                orig_vis = orig_vis[:h_min, :w_min]
                ref_vis = ref_vis[:h_min, :w_min]
                pred_vis = pred_vis[:h_min, :w_min]

                concat_img = np.concatenate([orig_vis, ref_vis, pred_vis], axis=1)
                concat_name = os.path.join(target_dir, f"concat_{stem}.png")
                imageio.imwrite(concat_name, concat_img)

                # ================== bad case 额外保存 ==================
                if case_tag == 'bad':
                    shutil.copy(ref_path, os.path.join(target_dir, f"ref_{os.path.basename(ref_path)}"))
                    shutil.copy(pred_path, os.path.join(target_dir, f"pred_{os.path.basename(pred_path)}"))

                    imageio.imwrite(
                        os.path.join(target_dir, f"ref_bin_{os.path.basename(ref_path)}"),
                        y_true_img * 255
                    )
                    imageio.imwrite(
                        os.path.join(target_dir, f"pred_bin_{os.path.basename(pred_path)}"),
                        y_pred_img * 255
                    )

                    shutil.copy(img_path, os.path.join(target_dir, f"img_{os.path.basename(img_path)}"))

            except Exception as e:
                print(f"处理失败: {line} 错误: {e}")

        if f1_scores:
            avg_p = float(np.mean(p_scores))
            avg_r = float(np.mean(r_scores))
            avg_f1 = float(np.mean(f1_scores))

            # micro（全像素汇总）
            micro_p = micro_tp / (micro_tp + micro_fp) if (micro_tp + micro_fp) > 0 else 0.0
            micro_r = micro_tp / (micro_tp + micro_fn) if (micro_tp + micro_fn) > 0 else 0.0
            micro_f1 = (2 * micro_tp) / (2 * micro_tp + micro_fp + micro_fn) if (2 * micro_tp + micro_fp + micro_fn) > 0 else 0.0

            out_file.write("\n")
            out_file.write(f"Average (per-image)  P: {avg_p:.4f}\tR: {avg_r:.4f}\tF1: {avg_f1:.4f}\n")
            out_file.write(f"Micro   (all-pixel) P: {micro_p:.4f}\tR: {micro_r:.4f}\tF1: {micro_f1:.4f}\n")
            out_file.write(f"Good: {good_cnt}, Normal: {normal_cnt}, Bad: {bad_cnt}\n")

            print(f"平均(逐图) P/R/F1: {avg_p:.4f} / {avg_r:.4f} / {avg_f1:.4f}")
            print(f"Micro(全像素) P/R/F1: {micro_p:.4f} / {micro_r:.4f} / {micro_f1:.4f}")
            print(f"Good: {good_cnt}, Normal: {normal_cnt}, Bad: {bad_cnt}")
        else:
            print("未成功计算任何指标。")


if __name__ == "__main__":
    compute_f1_scores(
        ref_txt='sampled_files.txt',
        pred_root='total=15000_epochs=100_batch=4_with_MoE=True(MoE_attn=CBAM_K=32)_with_HiLo=True/mask_results_软路由/result/',
        save_path='f1_results.txt',
        case_root='total=15000_epochs=100_batch=4_with_MoE=True(MoE_attn=CBAM_K=32)_with_HiLo=True/cases_all_软路由/',
        bad_threshold=0.55,
        good_threshold=0.85,
        original_root=None
    )
