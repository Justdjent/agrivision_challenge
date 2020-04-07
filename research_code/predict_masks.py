import os
import cv2
import warnings

import numpy as np
import pandas as pd
import tensorflow as tf

from tqdm import tqdm
from research_code.data_generator import read_channels
from research_code.params import args
from research_code.models import make_model
from keras.applications import imagenet_utils
from tensorflow.image import flip_left_right
from research_code.utils import calculate_ndvi


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


def predict(experiment_dir, class_names, weights_path, test_df_path, test_data_dir, input_channels, network):
    output_dir = os.path.join(experiment_dir, "predictions")
    os.makedirs(output_dir, exist_ok=True)
    model = make_model((None, None, len(input_channels)),
                       network=network,
                       channels=len(args.class_names),
                       activation=args.activation)
    model.load_weights(weights_path)
    test_df = pd.read_csv(test_df_path)
    test_df = test_df[test_df['ds_part'] == 'val']
    nbr_test_samples = len(test_df)
    for idx, row in tqdm(test_df.iterrows(), total=nbr_test_samples):
        x = read_channels(input_channels, row["name"], test_data_dir)
        x = imagenet_utils.preprocess_input(x, 'channels_last', mode='tf')
        preds = model.predict(x)
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

    predict(experiment_dir=args.experiments_dir,
            class_names=args.class_names,
            weights_path=args.weights,
            test_df_path=args.dataset_df,
            test_data_dir=args.val_dir,
            stacked_channels=args.stacked_channels,
            network=args.network)
