import os

import pandas as pd
import numpy as np
from tqdm import tqdm
import cv2
from collections import OrderedDict

from research_code.params import args

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu


def dice_coef(y_true, y_pred, smooth=1.0):
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = np.sum(y_true_f * y_pred_f)
    return (2. * intersection) / (np.sum(y_true_f) + np.sum(y_pred_f) + smooth)


def iou(y_true, y_pred, smooth=0.00001):
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = np.sum(y_true_f * y_pred_f)
    return (2. * intersection) / (np.sum(y_true_f) + np.sum(y_pred_f) + smooth)


def calculate_metrics(base_mask, transformed_mask, eta=0.0000001):
    """
    Calculates IoU/Jaccard and F1/Dice metric from masked rasters
    :param base_mask: base image thresholded mask
    :param transformed_mask: transformed image thresholded mask
    :return: IoU/Jaccard and F1/Dice metric
    """
    base_mask = base_mask > 0
    transformed_mask = transformed_mask > 0
    inter = np.sum(base_mask * transformed_mask)
    union = np.sum(base_mask) + np.sum(transformed_mask)
    iou = inter / (union - inter + eta)
    fn = np.sum(base_mask) - inter
    fp = np.sum(transformed_mask) - inter
    dice = 2 * inter / (2 * inter + fp + fn + eta)
    return iou, dice


def evaluate(masks_dir, results_dir, tfr_df_name):
    """
    Creates dataframe and tfrecords file for results visualization
    :param masks_dir: Ground truth masks dir
    :param results_dir: Predicted masks dir
    :param tfr_df_name: Name of output DataFrame and TFRecords, Will be saved to results_dir/<name>
    :return:
    """
    test_df = pd.read_csv(args.test_df)
    test_df = test_df[test_df['ds_part'] == 'val']
    thresh = args.threshold
    ious = {cls: [] for cls in args.class_names}
    pbar = tqdm()
    for num, row in test_df.iterrows():
        mean_iou = []
        mixed_prediction = np.zeros((512, 512))
        for num, cls in enumerate(args.class_names):
            result_path = os.path.join(results_dir, "labels", cls, row['name'])
            result = cv2.imread(result_path, cv2.IMREAD_GRAYSCALE)
            mask_path = os.path.join(masks_dir, "labels", cls, row['name'].replace(".jpg", ".png"))
            boundary_path = os.path.join(masks_dir, "boundaries", row['name'].replace(".jpg", ".png"))
            boundary = cv2.imread(boundary_path, cv2.IMREAD_GRAYSCALE)
            result[boundary != 255] = 0
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            # result = result > thresh * 255
            # mixed_prediction[bin_mask] = num + 1
            mask = mask > thresh * 255
            cur_iou = iou(mask, result)
            test_df.loc[num, "{}_iou".format(cls)] = cur_iou
            mean_iou.append(cur_iou)
            ious[cls].append(cur_iou)
        test_df.loc[num, "mean_iou"] = np.mean(mean_iou)
        cur_iou_state = []
        for cls, _ious in ious.items():
            cur_iou_state.append((cls, np.mean(_ious)))
            # print("{}: {}".format(cls, np.)mean(ious)))
        iou_dict = OrderedDict(cur_iou_state)
        pbar.set_postfix(iou_dict)
        pbar.update(1)
    test_df.to_csv(os.path.join(os.path.dirname(results_dir), tfr_df_name), index=False)
    
if __name__ == '__main__':
    prediction_dir = args.pred_mask_dir
    mask_dir = args.test_mask_dir
    output_csv_name = args.output_csv
    evaluate(mask_dir,
             prediction_dir,
             output_csv_name)
