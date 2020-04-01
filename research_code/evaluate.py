import os
import cv2
import json

import numpy as np
import pandas as pd
import seaborn as sn
import tensorflow as tf
import matplotlib.pyplot as plt

from tqdm import tqdm
from research_code.params import args

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu


def dice_coef(y_true, y_pred, smooth=1.0):
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = np.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (np.sum(y_true_f) + np.sum(y_pred_f) + smooth)


def iou(y_true, y_pred, smooth=1.0):
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = np.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (np.sum(y_true_f) + np.sum(y_pred_f) + smooth)


def m_iou(confusion_matrix: np.ndarray, class_names):
    class_ious = {}
    overall_iou = 0
    for cls, cls_name in enumerate(class_names):
        intersection = confusion_matrix[cls, cls]
        prediction = confusion_matrix[cls, :].sum()
        target = confusion_matrix[:, cls].sum()
        union = prediction + target - intersection
        if union == 0:
            class_ious[cls_name] = np.NAN
        else:
            class_iou = round(intersection / union, 3)
            class_ious[cls_name] = class_iou
            overall_iou += class_iou
    return round(overall_iou / len(class_names), 3), class_ious


def compute_confusion_matrix(predictions: np.ndarray, ground_truths: np.ndarray, num_classes):
    # pick prediction with the highest score and convert it into 3d binary array
    highest_score_prediction = predictions.argmax(axis=-1)
    predictions_mask = tf.one_hot(highest_score_prediction, num_classes, dtype=np.uint8).numpy()
    ground_truths = ground_truths.astype(np.uint8)
    true_positive = (predictions_mask * ground_truths).sum(axis=2)
    highest_score_prediction = predictions.argmax(axis=-1)
    incorrect_predictions = true_positive == 0
    correct_predictions = true_positive == 1
    transposed_conf_matrix = np.zeros((num_classes, num_classes), dtype=np.int32)
    confusion_matrix_tp = ground_truths[correct_predictions].sum(axis=0)
    for cls in range(num_classes):
        transposed_conf_matrix[cls, :] += ground_truths[incorrect_predictions & (highest_score_prediction == cls)].sum(
            axis=0, dtype=np.int32)
    confusion_matrix = transposed_conf_matrix.T
    np.fill_diagonal(confusion_matrix, confusion_matrix_tp)
    assert confusion_matrix.sum() == ground_truths.sum()
    return confusion_matrix.astype(np.uint64)


def calculate_metrics(ground_truth, prediction, threshold, eta=0.0000001):
    """
    Calculates IoU/Jaccard and F1/Dice metric from masked rasters
    :param ground_truth: ground_truth thresholded mask
    :param prediction: transformed image thresholded mask
    :return: IoU/Jaccard and F1/Dice metric
    """
    prediction = prediction > threshold
    inter = np.sum(ground_truth * prediction)
    union = np.sum(ground_truth) + np.sum(prediction)
    iou = inter / (union - inter + eta)
    fn = np.sum(ground_truth) - inter
    fp = np.sum(prediction) - inter
    dice = 2 * inter / (2 * inter + fp + fn + eta)
    return iou, dice


def plot_confusion_matrix(confusion_matrix, class_names, x_title, pred_dir, output_filename):
    confusion_matrix = confusion_matrix.astype('float') / confusion_matrix.sum(axis=1)[:, np.newaxis]
    df_cm = pd.DataFrame(confusion_matrix, index=class_names,
                         columns=class_names)
    plt.figure(figsize=(15, 15))
    heat_map = sn.heatmap(df_cm, annot=True)
    heat_map.set_ylabel("Ground truth")
    heat_map.set_xlabel(x_title)
    plt.show()
    heat_map.figure.savefig(os.path.join(pred_dir, f"{output_filename}_conf_matrix.png"))


def evaluate(test_dir, prediction_dir, output_csv, test_df_path, threshold, class_names):
    """
    Creates dataframe and tfrecords file for results visualization
    :param test_dir: Directory with images, boundaries, masks and ground truth of the test
    :param prediction_dir: Predicted masks dir
    :param output_csv: Name for output_csv
    :param test_df_path: Path to dataframe with data about test
    :param threshold: Threshold for predictions
    :param class_names: Array of class names
    :return:
    """
    test_df = pd.read_csv(test_df_path)
    output_df = pd.DataFrame(columns=['name', 'iou', 'dice', 'ds_part', 'class_name'])
    class_iou = {}
    class_dice = {}
    for ds_part in ['train', 'val']:
        input_dir = os.path.join(test_dir, ds_part)
        input_df = test_df[test_df['ds_part'] == ds_part]
        for class_name in class_names:
            class_df = input_df[input_df[class_name] != 0]
            class_iou[class_name] = 0
            class_dice[class_name] = 0
            for idx, row in tqdm(class_df.iterrows(), total=len(class_df)):
                filename = row['name']
                boundary_path = os.path.join(input_dir, "boundaries", filename.replace('.jpg', '.png'))
                boundary = cv2.imread(boundary_path, cv2.IMREAD_GRAYSCALE).astype(bool)
                mask_path = os.path.join(input_dir, "masks", filename.replace('.jpg', '.png'))
                if os.path.exists(mask_path):
                    invalid_pixels_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE).astype(bool)
                else:
                    invalid_pixels_mask = np.ones(boundary.shape)
                valid_pixels_mask = np.logical_and(boundary, invalid_pixels_mask).astype(np.uint8)
                ground_truth_path = os.path.join(input_dir, "labels", class_name, filename.replace('.jpg', '.png'))
                prediction_path = os.path.join(prediction_dir, class_name, filename)
                ground_truth = cv2.imread(ground_truth_path, cv2.IMREAD_GRAYSCALE)
                prediction = cv2.imread(prediction_path, cv2.IMREAD_GRAYSCALE)
                ground_truth = (ground_truth / 255) * valid_pixels_mask
                prediction = (prediction / 255) * valid_pixels_mask
                img_iou, img_dice = calculate_metrics(ground_truth, prediction, threshold)
                class_iou[class_name] += img_iou
                class_dice[class_name] += img_dice
                output_df = output_df.append({
                    "name": filename,
                    "iou": img_iou,
                    "dice": img_dice,
                    "ds_part": ds_part,
                    "class_name": class_name
                }, ignore_index=True)
            class_iou[class_name] /= len(class_df)
            class_dice[class_name] /= len(class_df)
    for class_name in class_names:
        train_df = output_df[(output_df["class_name"] == class_name) & (output_df["ds_part"] == 'train')]
        val_df = output_df[(output_df["class_name"] == class_name) & (output_df["ds_part"] == 'val')]
        print(f"{class_name}: Train IoU={round(train_df['iou'].mean(), 3)}, "
              f"Train Dice={round(train_df['dice'].mean(), 3)}, "
              f"Val IoU={round(val_df['iou'].mean(), 3)}, "
              f"Val Dice={round(val_df['dice'].mean(), 3)}")
    output_df.to_csv(os.path.join(prediction_dir, "train_val_evaluation.csv"), index=None)


if __name__ == '__main__':
    evaluate(test_dir=args.test_data_dir,
             prediction_dir=args.pred_mask_dir,
             output_csv=args.output_csv,
             test_df_path=args.test_df,
             threshold=args.threshold,
             class_names=args.class_names)
