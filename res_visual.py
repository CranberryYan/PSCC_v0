import os
import shutil
import numpy as np
import imageio.v2 as imageio
from sklearn.metrics import f1_score


# 读取并二值化mask, 同时返回二维图像和展平后的数组
def load_binary_mask(path):
    mask = imageio.imread(path)
    if mask.ndim == 3:
        mask = mask[:, :, 0]  # 若是三通道, 取第一个通道
    binary_mask = (mask > 240).astype(np.uint8)
    return binary_mask, binary_mask.flatten()


def _to_three_channel_uint8(img):
    """把灰度 / RGBA 统一成 H×W×3 的 uint8，方便拼接显示"""
    if img.ndim == 2:  # H×W
        img = np.stack([img] * 3, axis=-1)
    elif img.ndim == 3 and img.shape[2] == 4:  # H×W×4 -> 去掉 Alpha
        img = img[:, :, :3]
    # 若 dtype 不是 uint8，简单裁剪后转成 0~255
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255)
        img = img.astype(np.uint8)
    return img


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

    并且无论 good / normal / bad，都保存一张:
        [ 原图 | GT mask | Pred mask ]
    的拼接图，方便论文展示。
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
    good_cnt = normal_cnt = bad_cnt = 0

    with open(save_path, 'w') as out_file, \
         open(good_list_path, 'w') as good_file, \
         open(normal_list_path, 'w') as normal_file, \
         open(bad_list_path, 'w') as bad_file:

        for line in lines:
            # line 示例: /mnt/e/datasets/PSCC/Training Dataset/copymove/fake/xxx.tif
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

            # 原图路径：这里直接用 line（你之前 sampled_files 写的是完整路径）
            img_path = line
            if not os.path.exists(img_path):
                print(f"原图不存在: {img_path}")
                continue

            try:
                # 读取 GT / Pred mask（用于 F1）
                y_true_img, y_true = load_binary_mask(ref_path)
                y_pred_img, y_pred = load_binary_mask(pred_path)

                score = f1_score(y_true, y_pred)
                f1_scores.append(score)

                # 写总结果
                out_file.write(f"{line}\tF1: {score:.4f}\n")

                # -------- 按 F1 分组 --------
                if score < bad_threshold:
                    target_dir = bad_dir
                    target_file = bad_file
                    bad_cnt += 1
                    case_tag = 'bad'
                elif score >= good_threshold:
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
                target_file.write(f"{line}\tF1: {score:.4f}\n")

                # ================== 拼接图生成（所有 case 都做） ==================
                # 原图
                orig_img = imageio.imread(img_path)
                orig_vis = _to_three_channel_uint8(orig_img)

                # GT / Pred 可视化：二值 * 255 -> 3 通道
                ref_vis = _to_three_channel_uint8(y_true_img * 255)
                pred_vis = _to_three_channel_uint8(y_pred_img * 255)

                # 尺寸不一致时简单对齐一下高度/宽度（如果你的数据一定匹配，可以省略这块）
                h_min = min(orig_vis.shape[0], ref_vis.shape[0], pred_vis.shape[0])
                w_min = min(orig_vis.shape[1], ref_vis.shape[1], pred_vis.shape[1])
                orig_vis = orig_vis[:h_min, :w_min]
                ref_vis = ref_vis[:h_min, :w_min]
                pred_vis = pred_vis[:h_min, :w_min]

                concat_img = np.concatenate([orig_vis, ref_vis, pred_vis], axis=1)
                concat_name = os.path.join(target_dir, f"concat_{stem}.png")
                imageio.imwrite(concat_name, concat_img)

                # ================== 仍然对 bad case 做更详细的保存 ==================
                if case_tag == 'bad':
                    # 保存 GT mask & pred mask 原图
                    shutil.copy(
                        ref_path,
                        os.path.join(target_dir,
                                     f"ref_{os.path.basename(ref_path)}")
                    )
                    shutil.copy(
                        pred_path,
                        os.path.join(target_dir,
                                     f"pred_{os.path.basename(pred_path)}")
                    )

                    # 保存二值图(255 显示)
                    imageio.imwrite(
                        os.path.join(target_dir,
                                     f"ref_bin_{os.path.basename(ref_path)}"),
                        y_true_img * 255
                    )
                    imageio.imwrite(
                        os.path.join(target_dir,
                                     f"pred_bin_{os.path.basename(pred_path)}"),
                        y_pred_img * 255
                    )

                    # 保存原始图像
                    shutil.copy(
                        img_path,
                        os.path.join(target_dir,
                                     f"img_{os.path.basename(img_path)}")
                    )

            except Exception as e:
                print(f"处理失败: {line} 错误: {e}")

        if f1_scores:
            avg_score = sum(f1_scores) / len(f1_scores)
            out_file.write(f"\nAverage F1 Score: {avg_score:.4f}\n")
            out_file.write(f"Good: {good_cnt}, Normal: {normal_cnt}, Bad: {bad_cnt}\n")
            print(f"平均 F1 分数: {avg_score:.4f}")
            print(f"Good: {good_cnt}, Normal: {normal_cnt}, Bad: {bad_cnt}")
        else:
            print("未成功计算任何 F1 分数。")

if __name__ == "__main__":
    compute_f1_scores(
        ref_txt='sampled_files.txt',
        pred_root='mask_results/result/',
        save_path='f1_results.txt',
        case_root='cases_all',          # 根目录，下面会有 good/normal/bad 三个子目录
        bad_threshold=0.55,             # <0.55 认为是 bad
        good_threshold=0.85,            # ≥0.85 认为是 good
        original_root=None              # 或者填你的 fake 根目录；目前代码直接用 line 原路径拷贝
    )