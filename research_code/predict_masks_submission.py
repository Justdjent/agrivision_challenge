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
from research_code.utils import calculate_ndvi
from research_code.predict_masks import run_tta, run_model
from research_code.data_generator import read_channels
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
# prediction_dir = args.pred_mask_dir


def generate_submission(thresh, weights_path):
    output_dir = os.path.join(args.experiments_dir, args.exp_name,  "submission")
    output_filename = output_dir + ".zip"
    class_names = args.class_names
    os.makedirs(output_dir, exist_ok=True)
    # model = make_model((None, None, len(args.channels)),
    #                   network=args.network,
    #                   channels=len(args.class_names),
    #                   activation=args.activation,
    #                   add_classification_head=args.add_classification_head,
    #                   classes=args.class_names)
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
    class_ious = {cls: [] for cls in class_names}
    for i in tqdm(range(int(nbr_test_samples / batch_size)), total=nbr_test_samples):
        img_sizes = []

        for j in range(batch_size):
            if i * batch_size + j < nbr_test_samples:
                class_labels = {}
                img_name = test_images_list[i * batch_size + j]
                x = read_channels(args.channels, img_name, args.test_dir)
        # batch_x = x

        if args.tta:
            preds = run_tta(x, model, args.add_classification_head)
        else:
            preds = run_model(model, x, args.add_classification_head)

        for num in range(len(class_names)):
            bin_mask = (preds[:, :, num] * 255).astype(np.uint8)
            cur_class = class_names[num]
            filename = img_name
            save_folder_masks = os.path.join(output_dir, cur_class)
            os.makedirs(save_folder_masks, exist_ok=True)
            save_path_masks = os.path.join(save_folder_masks, filename)
            cv2.imwrite(save_path_masks, bin_mask)
        pbar.update(1)

    shutil.make_archive(output_filename, 'zip', output_dir)

if __name__ == '__main__':
    generate_submission(args.threshold, args.weights)
