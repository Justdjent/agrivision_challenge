import os

import numpy as np
from keras.applications import imagenet_utils
import pandas as pd
import cv2
from research_code.models import make_model
from research_code.params import args
from research_code.evaluate import iou
from tqdm import tqdm
tqdm.monitor_interval = 0
from collections import OrderedDict

from research_code.random_transform_mask import pad_size, unpad

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
prediction_dir = args.pred_mask_dir


def do_tta(x, tta_type):
    if tta_type == 'hflip':
        # batch, img_col = 2
        return flip_axis(x, 2)
    else:
        return x


def undo_tta(pred, tta_type):
    if tta_type == 'hflip':
        # batch, img_col = 2
        return flip_axis(pred, 2)
    else:
        return pred


def get_new_shape(img_shape, max_shape=224):
    if img_shape[0] > img_shape[1]:
        new_shape = (max_shape, int(img_shape[1] * max_shape / img_shape[0]))
    else:
        new_shape = (int(img_shape[0] * max_shape / img_shape[1]), max_shape)
    return new_shape


def predict(thresh=0.5):
    output_dir = args.pred_mask_dir
    class_names = args.class_names
    os.makedirs(output_dir, exist_ok=True)
    model = make_model((None, None, 3 + args.stacked_channels))
    model.load_weights(args.weights)
    batch_size = 1
    test_images_list = os.listdir(os.path.join(args.test_data_dir, "rgb"))
    nbr_test_samples = len(test_images_list)

    pbar = tqdm()
    class_ious = {cls: [] for cls in class_names}
    for i in tqdm(range(int(nbr_test_samples / batch_size)), total=nbr_test_samples):
        img_sizes = []

        for j in range(batch_size):
            if i * batch_size + j < nbr_test_samples:
                class_labels = {}
                img_name = test_images_list[i * batch_size + j]
                img_path = os.path.join(args.test_data_dir, "rgb", img_name)
                img = cv2.imread(img_path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img, pads = pad_size(img)
        x = np.expand_dims(img, axis=0)
        x = imagenet_utils.preprocess_input(x,  'channels_last', mode='tf')
        batch_x = x
        preds = model.predict_on_batch(batch_x)
        mixed_prediction = np.zeros((img.shape[0], img.shape[1]))
        for num, pred in enumerate(preds):
            # print(pred.max())
            bin_mask = (pred[0, :, :, 0] * 255).astype(np.uint8)
            # print(bin_mask.max())
            bin_mask = bin_mask > (thresh * 255)
            mixed_prediction[bin_mask] = num + 1
            # print(np.unique(mixed_prediction))
        os.makedirs(output_dir, exist_ok=True)
        save_path_masks = os.path.join(output_dir, img_name.replace(".jpg", ".png"))
        cv2.imwrite(save_path_masks, mixed_prediction)
        pbar.update(1)

if __name__ == '__main__':
    predict(args.threshold)
