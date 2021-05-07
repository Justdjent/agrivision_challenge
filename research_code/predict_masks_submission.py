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
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
# prediction_dir = args.pred_mask_dir

CLASSES_DICT = {'double_plant': 1,
                'drydown': 2,
                'endrow': 3,
                'nutrient_deficiency': 4,
                'planter_skip': 5,
                'water': 6,
                'waterway': 7,
                'weed_cluster': 8}


def generate_submission(thresh, weights_path):
    output_dir = os.path.join(args.experiments_dir, args.exp_name,  "submission")
    output_filename = output_dir + ".zip"
    class_names = args.class_names
    os.makedirs(output_dir, exist_ok=True)
    print(len(args.channels))
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
    for i in tqdm(range(int(nbr_test_samples / batch_size)), total=nbr_test_samples):
        img_sizes = []

        for j in range(batch_size):
            if i * batch_size + j < nbr_test_samples:
                class_labels = {}
                img_name = test_images_list[i * batch_size + j]
                img_path = os.path.join(args.test_dir, "images", "rgb", img_name)
                nir_img_path = os.path.join(args.test_dir, 'images', "nir", img_name)
                nir_img = cv2.imread(nir_img_path, cv2.IMREAD_GRAYSCALE)
                img = cv2.imread(img_path)
                red = img[:, :, -1]
                ndvi = calculate_ndvi(red=red, nir_img=nir_img)
                nir_img = np.expand_dims(nir_img, axis=-1)
                ndvi = np.expand_dims(ndvi, axis=-1)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = np.concatenate([img, nir_img, ndvi], axis=-1)
                not_valid_mask = read_masks_borders(img_name, args.test_dir)

                # img = nir_img
                # img, pads = pad_size(img)
                x = img
        # batch_x = x

        if args.tta:
            preds = run_tta(x, model, args.add_classification_head)
        else:
            preds = run_model(model, x, args.add_classification_head)

        # preds = model.predict_on_batch(batch_x)
        mixed_prediction = np.zeros((img.shape[0], img.shape[1]))
        #for num, pred in enumerate(preds):
        for num, cur_cl_name in enumerate(class_names):
            if cur_cl_name == 'storm_damage':
                continue
            # print(pred.max())
            bin_mask = (preds[:, :, num] * 255).astype(np.uint8)
            # print(bin_mask.max())
            bin_mask = bin_mask > (thresh * 255)
            mixed_prediction[bin_mask] = CLASSES_DICT.get(cur_cl_name)
            # print(np.unique(mixed_prediction))
        # os.makedirs(output_dir, exist_ok=True)
        mixed_prediction[not_valid_mask] = 0
        save_path_masks = os.path.join(output_dir, img_name.replace(".jpg", ".png"))
        cv2.imwrite(save_path_masks, mixed_prediction)
        pbar.update(1)

    shutil.make_archive(output_filename, 'zip', output_dir)


def read_masks_borders(name, img_dir):
    border_path = os.path.join(
        img_dir, "boundaries", name.replace(".jpg", ".png")
    )

    border_img = cv2.imread(border_path, cv2.IMREAD_GRAYSCALE)
    try:
        border_img = border_img > 0
    except Exception as error:
        print(border_path)
    border_img = np.invert(border_img)
    return border_img

if __name__ == '__main__':
    generate_submission(args.threshold, args.weights)
