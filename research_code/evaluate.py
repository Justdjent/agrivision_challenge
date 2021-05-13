import os
import cv2
import json
import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sn
import tensorflow as tf
from concurrent.futures import ProcessPoolExecutor

from tqdm import tqdm
from typing import List
from research_code.params import args


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


def evaluate(test_dir: str, experiment_dir: str, test_df_path: str, threshold: float,
             class_names: List[str]):
    """
    Creates dataframe and tfrecords file for results visualization
    :param test_dir: Directory with images, boundaries, masks and ground truth of the test
    :param experiment_dir: Predicted masks dir
    :param test_df_path: Path to dataframe with data about test
    :param threshold: Threshold for predictions
    :param class_names: Array of class names
    :return:
    """
    prediction_dir = experiment_dir #os.path.join(experiment_dir, "predictions")
    test_df = pd.read_csv(test_df_path)
    test_df = test_df[test_df['ds_part'] == 'val']
    class_names = class_names + ['background']
    df = pd.DataFrame(columns=class_names + ['name'])
    num_classes = len(class_names)
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

        background_prediction = np.zeros(boundary.shape)
        ground_truths = []
        predictions = []
        for class_idx, class_name in enumerate(class_names):
            ground_truth_path = os.path.join(test_dir, "labels", class_name, filename.replace('.jpg', '.png'))
            ground_truth = cv2.imread(ground_truth_path, cv2.IMREAD_GRAYSCALE)
            try:
                ground_truth = (ground_truth  > 0)
            except:
                print(ground_truth_path)
                raise
            if class_name == 'background':
                prediction = np.logical_not(background_prediction)
            else:
                prediction_path = os.path.join(prediction_dir, class_name, filename)
                # print(prediction_path)
                prediction = cv2.imread(prediction_path, cv2.IMREAD_GRAYSCALE)
                prediction = (prediction / 255)
                prediction[prediction < threshold] = 0
                background_prediction = np.logical_or(background_prediction, prediction)
            ground_truths.append(ground_truth)
            predictions.append(prediction)

        ground_truths = np.moveaxis(np.array(ground_truths) * valid_pixels_mask, 0, -1)
        predictions = np.moveaxis(np.array(predictions) * valid_pixels_mask, 0, -1)

        img_confusion_matrix = compute_confusion_matrix(predictions, ground_truths, num_classes)
        confusion_matrix += img_confusion_matrix
        _, img_ious = m_iou(img_confusion_matrix, class_names)
        img_ious["name"] = filename
        df = df.append(img_ious, ignore_index=True)

    mean_iou, class_ious = m_iou(confusion_matrix, class_names)
    x_title = f"Mean IoU - {mean_iou}\n{class_ious}"
    class_ious["mean_iou"] = mean_iou
    print(x_title)
    output_filename = os.path.basename(experiment_dir)
    output_csv = output_filename + ".csv"
    plot_confusion_matrix(confusion_matrix, class_names, x_title, prediction_dir, output_filename)
    with open(os.path.join(prediction_dir, f"{output_filename}_mean_ious.json"), 'w') as f:
        json.dump(class_ious, f)
    df.to_csv(os.path.join(prediction_dir, output_csv), index=False)


class Evaluator:
    def __init__(self, test_dir: str, experiment_dir: str, test_df_path: str, threshold: float,
             class_names: List[str]):
        self.test_dir = test_dir
        self.experiment_dir = experiment_dir
        self.threshold = threshold
        self.thresholds = {'planter_skip': 0.1,
                           'cloud_shadow': 0.4,
                           'double_plant': 0.4,
                           'standing_water': 0.4,
                           'waterway': 0.7,
                           'weed_cluster': 0.4}
        self.test_df_path = test_df_path
        self.class_names = class_names
        self.class_names = self.class_names + ['background']
        self.prediction_dir = os.path.join(experiment_dir, "predictions")

    def process_data(self, data, num_cores=4, parallel=True):
        """
        process data using one ore multiple threads with given processing and
        post-processing
        :param data:
        :param processing_func:
        :param postprocessing_func:
        :return:
            results: processing results
        """
        if parallel:
            # data = data[:20]
            data_batches = np.array_split(data, num_cores)
            df_merged = pd.DataFrame()
            confusion_matrix_full = None

            with ProcessPoolExecutor(num_cores) as executor:
                results = executor.map(self.evaluate_df, data_batches)
            for confusion_matrix, df in results:
                if not isinstance(confusion_matrix_full, np.ndarray):
                    confusion_matrix_full = confusion_matrix.copy()
                else:
                    confusion_matrix_full += confusion_matrix
                df_merged = df_merged.append(df)
        else:
            confusion_matrix_full, df_merged = self.evaluate_df(data)

        return confusion_matrix_full, df_merged

    def evaluate_df(self, test_df):
        df = pd.DataFrame(columns=self.class_names + ['name'])
        num_classes = len(self.class_names)
        confusion_matrix = np.zeros((num_classes, num_classes), dtype=np.uint64)
        for idx, row in test_df.iterrows():
            filename = row['name']
            boundary_path = os.path.join(self.test_dir, row['ds_part'], "boundaries", filename.replace('.jpg', '.png'))
            boundary = cv2.imread(boundary_path, cv2.IMREAD_GRAYSCALE).astype(bool)
            mask_path = os.path.join(self.test_dir, row['ds_part'], "masks", filename.replace('.jpg', '.png'))
            if os.path.exists(mask_path):
                invalid_pixels_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE).astype(bool)
            else:
                invalid_pixels_mask = np.ones(boundary.shape)
            valid_pixels_mask = np.logical_and(boundary, invalid_pixels_mask).astype(np.uint8)

            background_prediction = np.zeros(boundary.shape)
            ground_truths = []
            predictions = []
            for class_idx, class_name in enumerate(self.class_names):
                ground_truth_path = os.path.join(self.test_dir, row['ds_part'], "labels", class_name, filename.replace('.jpg', '.png'))
                ground_truth = cv2.imread(ground_truth_path, cv2.IMREAD_GRAYSCALE)
                ground_truth[ground_truth < 127] = 0
                ground_truth[ground_truth >= 127] = 255
                try:
                    # ground_truth = (ground_truth  < self.threshold * 255)
                    ground_truth = ground_truth > 127
                except:
                    print(ground_truth_path)
                if class_name == 'background':
                    prediction = np.logical_not(background_prediction)
                else:
                    prediction_path = os.path.join(self.prediction_dir, class_name, filename)
                    prediction = cv2.imread(prediction_path, cv2.IMREAD_GRAYSCALE)
                    prediction = (prediction / 255)
                    prediction[prediction < self.threshold] = 0
                    background_prediction = np.logical_or(background_prediction, prediction)
                ground_truths.append(ground_truth)
                predictions.append(prediction)

            ground_truths = np.moveaxis(np.array(ground_truths) * valid_pixels_mask, 0, -1)
            predictions = np.moveaxis(np.array(predictions) * valid_pixels_mask, 0, -1)

            img_confusion_matrix = compute_confusion_matrix(predictions, ground_truths, num_classes)
            _, img_ious = m_iou(img_confusion_matrix, self.class_names)
            img_ious["name"] = filename
            df = df.append(img_ious, ignore_index=True)
            confusion_matrix += img_confusion_matrix
        return confusion_matrix, df

    def evaluate_multiprocess(self):
        """
        Creates dataframe and tfrecords file for results visualization
        :param test_dir: Directory with images, boundaries, masks and ground truth of the test
        :param experiment_dir: Predicted masks dir
        :param test_df_path: Path to dataframe with data about test
        :param threshold: Threshold for predictions
        :param class_names: Array of class names
        :return:
        """
        test_df = pd.read_csv(self.test_df_path)
        test_df = test_df[test_df['ds_part'] == 'val']
        confusion_matrix, df = self.process_data(test_df, num_cores=4, parallel=True)

        mean_iou, class_ious = m_iou(confusion_matrix, self.class_names)
        x_title = f"Mean IoU - {mean_iou}\n{class_ious}"
        class_ious["mean_iou"] = mean_iou
        print(x_title)
        output_filename = os.path.basename(self.experiment_dir)
        output_csv = output_filename + ".csv"
        plot_confusion_matrix(confusion_matrix, self.class_names, x_title, self.prediction_dir, output_filename)
        with open(os.path.join(self.prediction_dir, f"{output_filename}_mean_ious.json"), 'w') as f:
            json.dump(class_ious, f)
        df.to_csv(os.path.join(self.prediction_dir, output_csv), index=False)
        return x_title

    def get_best_threshold(self):
        threshlds = np.linspace(0.1, 0.9, 9)
        mean_ios = []
        for thresh in threshlds:
            self.threshold = thresh
            thresh_mean_iou = self.evaluate_multiprocess()
            mean_ios.append(thresh_mean_iou)
            print("thresh: {}, mean_iou: ".format(thresh) + thresh_mean_iou)

        for thresh, mean_iou in zip(threshlds, mean_ios):
            print("thresh: {}, mean_iou: ".format(thresh) + mean_iou)


if __name__ == '__main__':
    evaluator = Evaluator(test_dir=args.val_dir,
             experiment_dir=args.experiments_dir,
             test_df_path=args.dataset_df,
             threshold=args.threshold,
             class_names=args.class_names)
    evaluator.evaluate_multiprocess()
    # evaluator.get_best_threshold()
    # evaluate(test_dir=args.val_dir,
    #          experiment_dir=args.experiments_dir,
    #          test_df_path=args.dataset_df,
    #          threshold=args.threshold,
    #         class_names=args.class_names)
