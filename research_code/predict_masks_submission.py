import os
import shutil

import numpy as np
from keras.applications import imagenet_utils
import pandas as pd
import cv2
import tensorflow as tf
from research_code.models import make_model
from research_code.params import args
from research_code.evaluate import iou
from tqdm import tqdm
tqdm.monitor_interval = 0
from collections import OrderedDict

# from research_code.random_transform_mask import pad_size, unpad
#from research_code.utils import calculate_ndvi
from research_code.data_generator import read_channels
from research_code.data_generator import HEAD_CHANNELS

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
# prediction_dir = args.pred_mask_dir
SUBMISSION_NUMS = {'cloud_shadow': 1,
                   'double_plant': 2,
                   'planter_skip': 3,
                   'standing_water': 4,
                   'waterway': 5,
                   'weed_cluster': 6}

def do_tta(x, tta_type):
    if tta_type == 'hflip':
        # batch, img_col = 2
        return tf.image.flip_axis(x, 2)
    else:
        return x

def undo_tta(pred, tta_type):
    if tta_type == 'hflip':
        # batch, img_col = 2
        return tf.image.flip_axis(pred, 2)
    else:
        return pred


def get_new_shape(img_shape, max_shape=224):
    if img_shape[0] > img_shape[1]:
        new_shape = (max_shape, int(img_shape[1] * max_shape / img_shape[0]))
    else:
        new_shape = (int(img_shape[0] * max_shape / img_shape[1]), max_shape)
    return new_shape


def generate_submission(thresh, weights_path):
    output_dir = os.path.join(args.experiments_dir, args.exp_name,  "submission")
    output_filename = output_dir + ".zip"
    class_names = args.class_names
    os.makedirs(output_dir, exist_ok=True)
    model = make_model((None, None, len(args.channels)),
                       network=args.network,
                       channels=len(args.class_names),
                       activation=args.activation,
                       add_classification_head=args.add_classification_head,
                       classes=args.class_names)

    model.load_weights(weights_path)
    batch_size = 1
    test_images_list = os.listdir(os.path.join(args.test_dir, "images", "rgb"))
    nbr_test_samples = len(test_images_list)

    pbar = tqdm()
    for idx in range(nbr_test_samples):
        img_name = test_images_list[idx]
        x = read_channels(args.channels, img_name, args.test_dir)
        x = imagenet_utils.preprocess_input(x, 'channels_last', mode='tf')
        x = np.expand_dims(x, axis=0)
        preds = model.predict(x)
        prediction_dict = {name: pred for name, pred in zip(model.output_names, preds)}

        mixed_prediction = np.zeros((x.shape[1], x.shape[2]))
        for head, preds in prediction_dict.items():
            class_names = HEAD_CHANNELS.get(head)
            if not isinstance(class_names, list):
                # print(head)
                continue
            for ind in range(len(class_names)):
                bin_mask = (preds[0, :, :, ind] * 255).astype(np.uint8)
                bin_mask = bin_mask > (thresh * 255)
                mixed_prediction[bin_mask] = SUBMISSION_NUMS.get(class_names[ind])

        save_path_masks = os.path.join(output_dir, img_name.replace(".jpg", ".png"))
        cv2.imwrite(save_path_masks, mixed_prediction)
        pbar.update(1)
    
    shutil.make_archive(output_filename, 'zip', output_dir)

if __name__ == '__main__':
    generate_submission(args.threshold, args.weights)