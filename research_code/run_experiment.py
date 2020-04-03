import os
import cv2

import numpy as np
import pandas as pd

from tqdm import tqdm
from typing import List
from research_code.params import args
from research_code.train import train
from research_code.predict_masks import predict
from research_code.evaluate import evaluate


def find_best_model(model_dir):
    min_loss = 10e+5
    best_model_name = None
    for file in os.listdir(model_dir):
        if file.endswith("h5"):
            loss_value = float('.'.join(file.split("-")[-1].split(".")[:-1]))
            if loss_value < min_loss:
                min_loss = loss_value
                best_model_name = file
    return best_model_name


def generate_ndvi():
    pass


def generate_ndwi():
    pass


def precompute_background_class(test_dir: str, test_df: pd.DataFrame, class_names: List[str]):
    background_class_path = os.path.join(test_dir, "labels", "background")
    if os.path.exists(background_class_path):
        print("Background class has been precomputed. Skipping background class computation")
        return
    else:
        print("Precomputing background class ground truth")
        os.makedirs(background_class_path)

    for idx, row in tqdm(test_df.iterrows(), total=len(test_df)):
        filename = row['name']
        background_ground_truth = None
        for class_idx, class_name in enumerate(class_names):
            ground_truth_path = os.path.join(test_dir, "labels", class_name, filename.replace('.jpg', '.png'))
            ground_truth = cv2.imread(ground_truth_path, cv2.IMREAD_GRAYSCALE)
            ground_truth = (ground_truth / 255)
            if background_ground_truth is None:
                background_ground_truth = np.zeros(ground_truth.shape)
            background_ground_truth = np.logical_or(background_ground_truth, ground_truth)
        background_ground_truth = np.logical_not(background_ground_truth).astype(np.uint8) * 255
        cv2.imwrite(os.path.join(background_class_path, f"{filename.replace('jpg', 'png')}"), background_ground_truth)
    print(f"Background class was saved into {background_class_path}")


def run_experiment():
    dataset_df = pd.read_csv(args.dataset_df)
    classes = args.class_names
    if 'background' in classes:
        classes.remove('background')
    precompute_background_class(args.train_dir, dataset_df[dataset_df["ds_part"] == 'train'], classes)
    precompute_background_class(args.val_dir, dataset_df[dataset_df["ds_part"] == 'val'], classes)

    experiment_dir, model_dir, experiment_name = train()
    prediction_dir = os.path.join(experiment_dir, "predictions")
    best_model_name = find_best_model(model_dir)
    weights_path = os.path.join(model_dir, best_model_name)

    test_df_path = args.dataset_df
    test_data_dir = args.val_dir
    print(f"Starting prediction process. Using {best_model_name} for prediction")
    predict(output_dir=prediction_dir,
            class_names=args.class_names,
            weights_path=weights_path,
            test_df_path=test_df_path,
            test_data_dir=test_data_dir,
            stacked_channels=args.stacked_channels,
            network=args.network)
    print(f"Starting evaluation process of results in {prediction_dir}")
    evaluate(test_dir=test_data_dir,
             prediction_dir=prediction_dir,
             output_csv=f"{experiment_name}_img_ious.csv",
             test_df_path=test_df_path,
             threshold=args.threshold,
             class_names=args.class_names)


if __name__ == "__main__":
    run_experiment()
