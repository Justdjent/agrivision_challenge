import os
import cv2

import numpy as np
import pandas as pd
import tensorflow as tf

from tqdm import tqdm
from research_code.params import args
from research_code.models import make_model
from research_code.random_transform_mask import pad_size
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


def predict(output_dir, class_names, weights_path, test_df_path, test_data_dir, stacked_channels):
    os.makedirs(output_dir, exist_ok=True)
    model = make_model((None, None, 3 + stacked_channels))
    model.load_weights(weights_path)
    test_df = pd.read_csv(test_df_path)
    class_name = class_names[0]
    for ds_part in ['train', 'val']:
        input_df = test_df[(test_df['ds_part'] == ds_part) & (test_df[class_name] != 0)]
        nbr_test_samples = len(input_df)
        for idx, row in tqdm(input_df.iterrows(), total=nbr_test_samples):
            img_path = os.path.join(test_data_dir, 'images', "rgb", row['name'])
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img, pads = pad_size(img)
            x = np.expand_dims(img, axis=0)
            x = imagenet_utils.preprocess_input(x, 'channels_last', mode='tf')
            preds = model.predict(x)

            for num, pred in enumerate(preds):
                bin_mask = (pred * 255).astype(np.uint8)
                filename = row['name']
                save_folder_masks = os.path.join(output_dir, class_name)
                os.makedirs(save_folder_masks, exist_ok=True)
                save_path_masks = os.path.join(save_folder_masks, filename)
                cv2.imwrite(save_path_masks, bin_mask)


if __name__ == '__main__':
    setup_env()
    predict(output_dir=args.pred_mask_dir,
            class_names=args.class_names,
            weights_path=args.weights,
            test_df_path=args.test_df,
            test_data_dir=args.test_data_dir,
            stacked_channels=args.stacked_channels)
