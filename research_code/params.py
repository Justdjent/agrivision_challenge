import argparse
import distutils.util

parser = argparse.ArgumentParser()
arg = parser.add_argument
arg('--gpu', default="0")
arg('--clr')
arg('--seed', type=int, default=80)
arg('--test_size_float', type=float, default=0.1)
arg('--epochs', type=int, default=30)

# reshape is first
arg('--reshape_height', type=int, default=512)
arg('--reshape_width', type=int, default=512)

# crop is last
arg('--crop_height', type=int, default=256)
arg('--crop_width', type=int, default=256)
arg('--use_crop', type=distutils.util.strtobool, default='true')

arg('--learning_rate', type=float, default=0.001)
arg('--batch_size', type=int, default=1)

#TODO Merge train and val
arg('--train_dir', default='/mnt/storage/cal/data/agrivision/train')
arg('--val_dir', default='/mnt/storage/cal/data/agrivision/val')
arg('--test_dir', default='/mnt/storage/cal/data/agrivision/test')

arg('--dataset_df', default='/mnt/storage/cal/data/agrivision/train_val_initial_2020-03-25.csv')
arg('--test_df', default='/mnt/storage/cal/data/agrivision/test.csv')

arg('--exclude_bad_labels_df')

arg('--experiments_dir', default='/mnt/storage/cal/artifacts/agrivision')
arg('--class_names', nargs='+', default=['cloud_shadow', 'double_plant', 'planter_skip', 'standing_water', 'waterway', 'weed_cluster'])
arg('--models_dir', default='models')
arg('--log_dir', default='logs')
arg('--weights')
arg('--freeze_till_layer', default='input_1')
arg('--show_summary', type=bool, default=False)
arg('--network', default='instance_unet')

arg('--net_alias', default='')
arg('--loss_function', default='')

arg('--pred_sample_csv', default='input/sample_submission.csv')
arg('--edges', action='store_true')

arg('--threshold', type=float, default=0.5)

arg('--exp_name')
arg('--coord_conv', type=bool, default=False)
arg('--use_aug', action='store_true')
arg('--activation', default='sigmoid', choices=['sigmoid', 'softmax'])
arg('--add_classification_head', action='store_true')
arg('--cls_head_loss_weight', '-cw', default=0.2)

arg('--channels', nargs='+', default=['r', 'g', 'b', 'nir', 'ndvi', 'ndwi', 'l'])

args = parser.parse_args()
