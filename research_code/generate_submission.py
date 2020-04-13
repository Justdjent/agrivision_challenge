import os
from collections import OrderedDict
import shutil

import cv2
import numpy as np
import pandas as pd
from keras.applications import imagenet_utils
from tqdm import tqdm
tqdm.monitor_interval = 0

from research_code.data_generator import read_channels
from research_code.evaluate import iou
from research_code.models import make_model
from research_code.params import args
# from research_code.random_transform_mask import pad_size, unpad
from research_code.utils import calculate_ndvi



os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
# prediction_dir = args.pred_mask_dir​


def generate_submission(weights_path, thresh=0.5):
    output_dir = os.path.join(args.experiments_dir, args.exp_name,  "submission")
    class_names = args.class_names
    os.makedirs(output_dir, exist_ok=True)
    model = make_model((None, None, len(args.channels)),
                       network=args.network,
                       channels=len(args.class_names),
                       activation=args.activation,
                       add_classification_head=args.add_classification_head,
                       classes=args.class_names)

    model.load_weights(weights_path)
    test_images_list = os.listdir(os.path.join(args.test_dir, "images", "rgb"))
    nbr_test_samples = len(test_images_list)

    pbar = tqdm()
    for i in tqdm(range(int(nbr_test_samples)), total=nbr_test_samples):
        img_name = test_images_list[i]
        img = read_channels(args.channels, img_name, args.test_dir)

        x = np.expand_dims(img, axis=0)
        x = imagenet_utils.preprocess_input(x, 'channels_last', mode='tf')
        # batch_x = x
        if args.add_classification_head:
            preds, _ = model.predict(x)
        else:
            preds = model.predict(x)

        mixed_prediction = np.zeros((img.shape[0], img.shape[1]))
        for num in range(len(class_names)):
            bin_mask = (preds[0, :, :, num] * 255).astype(np.uint8)
            bin_mask = bin_mask > (thresh * 255)
            mixed_prediction[bin_mask] = num + 1

        save_path_masks = os.path.join(output_dir, img_name.replace(".jpg", ".png"))
        cv2.imwrite(save_path_masks, mixed_prediction)
        pbar.update(1)

        shutil.make_archive(os.path.join(args.experiments_dir, args.exp_name, "submission"), "zip", os.path.join(args.experiments_dir, args.exp_name, 'submission'))
if __name__ == '__main__':
    generate_submission(args.weights, args.threshold)
