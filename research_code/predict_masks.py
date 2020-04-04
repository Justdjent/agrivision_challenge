import os
import cv2
import warnings

import numpy as np
import pandas as pd
import tensorflow as tf

from tqdm import tqdm
from research_code.params import args
from research_code.models import make_model
from keras.applications import imagenet_utils
from tensorflow.image import flip_left_right


def setup_env():
    tqdm.monitor_interval = 0
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
                logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)


def do_tta(x, tta_type):
    if tta_type == 'hflip':
        return flip_left_right(x, 2)
    else:
        return x


def undo_tta(pred, tta_type):
    if tta_type == 'hflip':
        return flip_left_right(pred, 2)
    else:
        return pred


def get_new_shape(img_shape, max_shape=224):
    if img_shape[0] > img_shape[1]:
        new_shape = (max_shape, int(img_shape[1] * max_shape / img_shape[0]))
    else:
        new_shape = (int(img_shape[0] * max_shape / img_shape[1]), max_shape)
    return new_shape


def predict(experiments_dir, class_names, weights_path, test_df_path, test_data_dir, stacked_channels, network):
    output_dir = os.path.join(experiments_dir, "predictions")
    os.makedirs(output_dir, exist_ok=True)
    # if args.stacked_channels != 0:
    warnings.showwarning("Currently there is only rgb image being read", UserWarning, 'predict_masks.py', 57)
    model = make_model((None, None, args.stacked_channels + 3),
                    network=args.network,
                    channels=len(args.class_names),
                    activation=args.activation)
    model.load_weights(weights_path)
    test_df = pd.read_csv(test_df_path)
    test_df = test_df[test_df['ds_part'] == 'val']
    nbr_test_samples = len(test_df)
    for idx, row in tqdm(test_df.iterrows(), total=nbr_test_samples):
        img_path = os.path.join(test_data_dir, 'images', "rgb", row['name'])
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        x = np.expand_dims(img, axis=0)
        x = imagenet_utils.preprocess_input(x, 'channels_last', mode='tf')
        preds = model.predict(x)
        #print(preds.shape)
        for num in range(len(class_names)):
            bin_mask = (preds[0, :, :, num] * 255).astype(np.uint8)
            cur_class = class_names[num]
            filename = row['name']
            save_folder_masks = os.path.join(output_dir, cur_class)
            os.makedirs(save_folder_masks, exist_ok=True)
            save_path_masks = os.path.join(save_folder_masks, filename)
            cv2.imwrite(save_path_masks, bin_mask)


if __name__ == '__main__':
    setup_env()
    
    predict(experiments_dir=args.experiments_dir,
            class_names=args.class_names,
            weights_path=args.weights,
            test_df_path=args.dataset_df,
            test_data_dir=args.val_dir,
            stacked_channels=args.stacked_channels,
            network=args.network)
