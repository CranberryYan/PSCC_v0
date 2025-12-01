from yacs.config import CfgNode as CN

pscc_args = CN()
# pscc_args.path = '../dataset/'
pscc_args.path = "/mnt/e/datasets/PSCC/Training Dataset/"
pscc_args.num_epochs = 40
pscc_args.lr_strategy = [2e-4, 1e-4, 5e-5, 2.5e-5, 1.25e-5,
                         1e-5, 1e-5, 1e-5, 1e-5, 1e-5]
pscc_args.learning_rate = pscc_args.lr_strategy[0]
pscc_args.learning_step = 10

pscc_args.lr_decay_step = pscc_args.num_epochs // pscc_args.learning_step

pscc_args.crop_size = [256, 256]
pscc_args.val_num = 200

pscc_args.save_tag = False

pscc_args.train_bs = 6
pscc_args.val_bs = 1
pscc_args.train_num = 15000
# authentic, splice, copymove, removal
pscc_args.train_ratio = [0.20, 0.20, 0.45, 0.15]

def get_pscc_args():
  return pscc_args.clone()
