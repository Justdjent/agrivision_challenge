import os
import cv2

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


def plot_confusion_matrix(confusion_matrix, class_names, x_title, pred_dir):
    confusion_matrix = confusion_matrix.astype('float') / confusion_matrix.sum(axis=1)[:, np.newaxis]
    df_cm = pd.DataFrame(confusion_matrix, index=class_names,
                         columns=class_names)
    plt.figure(figsize=(15, 15))
    heat_map = sn.heatmap(df_cm, annot=True)
    heat_map.set_ylabel("Ground truth")
    heat_map.set_xlabel(x_title)
    plt.show()
    heat_map.figure.savefig(os.path.join(pred_dir, "conf_matrix.png"))


def evaluate(test_dir, prediction_dir, output_csv, test_df_path, threshold, class_names):
    """
    Creates dataframe and tfrecords file for results visualization
    :param test_dir: Directory with images, boundaries, masks and ground truth of the test
    :param prediction_dir: Predicted masks dir
    :param output_csv: Name for output_csv
    :param test_df_path: Path to dataframe with data about test
    :param threshold: Threshold for predictions
    :param class_names: Array of class names
    :param background_is_class: Name of activation function - currently supported 'sigmoid' and 'softmax'
    :return:
    """
    test_df = pd.read_csv(test_df_path)
    test_df = test_df[test_df['ds_part'] == 'val']
    if 'background' in class_names:
        df = pd.DataFrame(columns=class_names + ['name'])
        num_classes = len(class_names)
    else:
        df = pd.DataFrame(columns=class_names + ['background', 'name'])
        num_classes = len(class_names) + 1
    confusion_matrix = np.zeros((num_classes, num_classes), dtype=np.uint64)
    for idx, row in tqdm(test_df.iterrows(), total=len(test_df)):
        filename = row['name']
        boundary_path = os.path.join(test_dir, "boundaries", filename.replace('.jpg', '.png'))
        boundary = cv2.imread(boundary_path, cv2.IMREAD_GRAYSCALE).astype(bool)
        mask_path = os.path.join(test_dir, "masks", filename.replace('.jpg', '.png'))
        if os.path.exists(mask_path):
            invalid_pixels_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE).astype(bool)
        else:
            invalid_pixels_mask = np.ones(boundary.shape)
        valid_pixels_mask = np.logical_and(boundary, invalid_pixels_mask).astype(np.uint8)
        # since there is no background class at the moment it should be calculated based on other classes
        if 'background' not in class_names:
            background_prediction = np.zeros(boundary.shape)
            background_ground_truth = np.zeros(boundary.shape)
        ground_truths = []
        predictions = []
        for class_idx, class_name in enumerate(class_names):
            ground_truth_path = os.path.join(test_dir, "labels", class_name, filename.replace('.jpg', '.png'))
            prediction_path = os.path.join(prediction_dir, class_name, filename)
            ground_truth = cv2.imread(ground_truth_path, cv2.IMREAD_GRAYSCALE)
            prediction = cv2.imread(prediction_path, cv2.IMREAD_GRAYSCALE)
            ground_truth = (ground_truth / 255)
            prediction = (prediction / 255)
            if 'background' not in class_names:
                prediction[prediction < threshold] = 0
                background_prediction = np.logical_or(background_prediction, prediction)
                background_ground_truth = np.logical_or(background_ground_truth, ground_truth)
            ground_truths.append(ground_truth)
            predictions.append(prediction)
        if 'background' not in class_names:
            background_ground_truth = np.logical_not(background_ground_truth)
            background_prediction = np.logical_not(background_prediction)
            ground_truths.append(background_ground_truth)
            predictions.append(background_prediction)
        predictions = np.moveaxis(np.array(predictions) * valid_pixels_mask, 0, -1)
        ground_truths = np.moveaxis(np.array(ground_truths) * valid_pixels_mask, 0, -1)
        img_confusion_matrix = compute_confusion_matrix(predictions, ground_truths, num_classes)
        confusion_matrix += img_confusion_matrix
        if 'background' not in class_names:
            _, img_ious = m_iou(img_confusion_matrix, class_names + ['background'])
        else:
            _, img_ious = m_iou(img_confusion_matrix, class_names)
        img_ious["name"] = filename
        df = df.append(img_ious, ignore_index=True)
    if 'background' not in class_names:
        class_names.append('background')
    mean_iou, class_ious = m_iou(confusion_matrix, class_names)
    x_title = f"Mean IoU - {mean_iou}\n{class_ious}"
    print(x_title)
    plot_confusion_matrix(confusion_matrix, class_names, x_title, prediction_dir)
    df.to_csv(os.path.join(prediction_dir, output_csv), index=False)


if __name__ == '__main__':
    evaluate(test_dir=args.test_data_dir,
             prediction_dir=args.pred_mask_dir,
             output_csv=args.output_csv,
             test_df_path=args.test_df,
             threshold=args.threshold,
             class_names=args.class_names)
