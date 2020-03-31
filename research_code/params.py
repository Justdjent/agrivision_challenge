import argparse
import distutils.util

parser = argparse.ArgumentParser()
arg = parser.add_argument
arg('--gpu', default="0")
arg('--fold', type=int, default=None)
arg('--n_folds', type=int, default=5)
arg('--folds_source')
arg('--clr')
arg('--seed', type=int, default=80)
arg('--test_size_float', type=float, default=0.1)
arg('--epochs', type=int, default=30)
arg('--img_height', type=int, default=512)
arg('--img_width', type=int, default=512)
arg('--out_height', type=int, default=512)
arg('--out_width', type=int, default=512)
arg('--input_width', type=int, default=256)
arg('--input_height', type=int, default=256)
arg('--use_crop', type=distutils.util.strtobool, default='true')
arg('--learning_rate', type=float, default=0.001)

arg('--batch_size', type=int, default=1)
arg('--auto_dataset_dir', default='/mnt/storage_4tb/ymi/datasets/Agriculture-Vision/train_val')
arg('--manual_dataset_dir', default='/mnt/storage_4tb/ymi/datasets/Agriculture-Vision/train_val')
arg('--test_data_dir', default='/mnt/storage_4tb/ymi/datasets/Agriculture-Vision/train_val')
arg('--test_mask_dir', default='/mnt/storage_4tb/ymi/datasets/Agriculture-Vision/train_val')

arg('--class_names', type=str, nargs='+', default=['cloud_shadow', 'double_plant', 'planter_skip', 'standing_water', 'waterway', 'weed_cluster'])
arg('--models_dir', default='models')
arg('--weights')
arg('--loss_function', default='crossentropy_time')
arg('--freeze_till_layer', default='input_1')
arg('--show_summary', type=bool, default=False)
arg('--network', default='instance_unet')
arg('--net_alias', default='')
arg('--preprocessing_function', default='tf')
arg('--mask_suffix', default='.png')
arg('--train_df', default='/mnt/storage_4tb/ymi/datasets/Agriculture-Vision/train_simple.csv') # /mnt/storage/ymi/geo_data/train_only_man_agr_7116.df
arg('--val_df', default='/mnt/storage_4tb/ymi/datasets/Agriculture-Vision/val_simple.csv')
arg('--test_df', default='/mnt/storage_4tb/ymi/datasets/Agriculture-Vision/val_simple.csv') # '/mnt/storage/ymi/geo_data/test_nrg_3580.df'
arg('--inp_list', default='input_list_pegasus2_5E') # latest df train_man_nrg_cleaned_6116
arg('--r_type', default='rgb', choices=['nrg', 'rgg', 'rgb'])

arg('--pred_mask_dir')
arg('--pred_tta', action='store_true')
arg('--pred_batch_size', default=1)
arg('--pred_threads', type=int, default=1)
arg('--submissions_dir', default='submissions')
arg('--pred_sample_csv', default='input/sample_submission.csv')
arg('--predict_on_val', type=bool, default=False)
arg('--stacked_channels', type=int, default=0)
arg('--stacked_channels_dir', default="blue")
arg('--edges', action='store_true')
# Dir names
arg('--train_data_dir_name', default='train')
arg('--val_data_dir_name', default='val')
arg('--train_mask_dir_name', default='train_masks')
arg('--val_mask_dir_name', default='val_masks')

arg('--threshold', type=float, default=0.5)

arg('--dirs_to_ensemble', nargs='+')
arg('--ensembling_strategy', default='average')
arg('--folds_dir')
arg('--ensembling_dir')
arg('--ensembling_cpu_threads', type=int, default=6)
arg('--output_csv')
arg('--exp_name')
arg('--coord_conv', type=bool, default=False)

arg('--use_aug', action='store_true')
args = parser.parse_args()