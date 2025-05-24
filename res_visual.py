import os
import shutil
import numpy as np
import imageio.v2 as imageio
from sklearn.metrics import f1_score

def load_binary_mask(path):
  mask = imageio.imread(path)
  if mask.ndim == 3:
      mask = mask[:, :, 0] # 若是三通道, 取第一个通道
  mask = (mask > 127).astype(np.uint8) # 二值化: 0/255 → 0/1
  return mask.flatten() # 展平用于 f1_score

def compute_f1_scores(ref_txt, pred_root,
                      save_path='f1_results.txt',
                      bad_case_dir='bad_cases',
                      f1_threshold=0.5,
                      original_root=None):
  os.makedirs(bad_case_dir, exist_ok=True)
  bad_list_path = os.path.join(bad_case_dir, 'bad_cases_list.txt')

  with open(ref_txt, 'r') as f:
    lines = [line.strip() for line in f if line.strip()]

  f1_scores = []
  with open(save_path, 'w') as out_file, open(bad_list_path, 'w') as bad_file:
    for line in lines:
      ref_path = line.replace('fake', 'mask').replace('jpg', 'png')
      pred_path = os.path.join(
        pred_root, os.path.basename(line)).replace('jpg', 'png')

      if not os.path.exists(pred_path):
        print(f"预测文件不存在：{pred_path}")
        continue

      try:
        y_true = load_binary_mask(ref_path)
        y_pred = load_binary_mask(pred_path)

        score = f1_score(y_true, y_pred)
        f1_scores.append(score)

        out_file.write(f"{line}\tF1: {score:.4f}\n")

        # 判断是否为bad case
        if score < f1_threshold:
          bad_file.write(f"{line}\tF1: {score:.4f}\n")

          shutil.copy(ref_path,
                      os.path.join(
                        bad_case_dir, f"ref_{os.path.basename(ref_path)}"))
          shutil.copy(pred_path,
                      os.path.join(
                        bad_case_dir, f"pred_{os.path.basename(pred_path)}"))

          if original_root is not None:
            image_path = os.path.join(original_root, line)
            image_path = image_path.replace('jpg', 'png')
            if os.path.exists(image_path):
              shutil.copy(image_path,
                          os.path.join(bad_case_dir,
                                       f"img_{os.path.basename(image_path)}"))

      except Exception as e:
        print(f"处理失败: {line} 错误: {e}")

    if f1_scores:
      avg_score = sum(f1_scores) / len(f1_scores)
      out_file.write(f"\nAverage F1 Score: {avg_score:.4f}\n")
      print(f"平均 F1 分数: {avg_score:.4f}")
    else:
      print("未成功计算任何 F1 分数。")

compute_f1_scores(
  ref_txt='sampled_files.txt',
  pred_root='mask_results/',
  save_path='f1_results.txt'
)

