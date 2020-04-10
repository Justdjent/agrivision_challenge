import os
import cv2

import numpy as np
import pandas as pd

from tqdm import tqdm
from typing import List
from research_code.params import args
from research_code.train import train
from research_code.predict_masks import predict, predict_multihead
from research_code.evaluate import evaluate
from research_code.predict_masks_submission import generate_submission
from research_code.utils import calculate_ndvi, calculate_ndwi, calculate_lightness


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


def generate_ndvi(img_dir: str):
    ndvi_dir_path = os.path.join(img_dir, "images", "ndvi")
    rgb_dir = os.path.join(img_dir, "images", "rgb")
    rgb_images = os.listdir(rgb_dir)
    if os.path.exists(ndvi_dir_path):
        ndvi_images = os.listdir(ndvi_dir_path)
        if len(ndvi_images) == len(rgb_images):
            print("NDVI has been precomputed. Skipping NDVI computation")
            return
        else:
            rgb_images = set(rgb_images) - set(ndvi_images)
    print("Precomputing NDVI channel")
    os.makedirs(ndvi_dir_path, exist_ok=True)
    for filename in tqdm(rgb_images, total=len(rgb_images)):
        if not filename.endswith(".jpg"):
            continue
        rgb_path = os.path.join(img_dir, "images", "rgb", filename)
        nir_path = os.path.join(img_dir, "images", "nir", filename)
        ndvi_path = os.path.join(ndvi_dir_path, filename)
        rgb_img = cv2.cvtColor(cv2.imread(rgb_path), cv2.COLOR_BGR2RGB)
        nir_img = cv2.imread(nir_path, cv2.IMREAD_GRAYSCALE)
        ndvi = calculate_ndvi(rgb_img[:, :, 0], nir_img)
        cv2.imwrite(ndvi_path, ndvi)


def generate_ndwi(img_dir: str):
    ndwi_dir_path = os.path.join(img_dir, "images", "ndwi")
    rgb_dir = os.path.join(img_dir, "images", "rgb")
    rgb_images = os.listdir(rgb_dir)
    if os.path.exists(ndwi_dir_path):
        ndwi_images = os.listdir(ndwi_dir_path)
        if len(ndwi_images) == len(rgb_images):
            print("NDVI has been precomputed. Skipping NDVI computation")
            return
        else:
            rgb_images = set(rgb_images) - set(ndwi_images)
    print("Precomputing NDWI channel")
    os.makedirs(ndwi_dir_path, exist_ok=True)
    for filename in tqdm(rgb_images, total=len(rgb_images)):
        if not filename.endswith(".jpg"):
            continue
        rgb_path = os.path.join(img_dir, "images", "rgb", filename)
        nir_path = os.path.join(img_dir, "images", "nir", filename)
        ndwi_path = os.path.join(ndwi_dir_path, filename)
        rgb_img = cv2.cvtColor(cv2.imread(rgb_path), cv2.COLOR_BGR2RGB)
        nir_img = cv2.imread(nir_path, cv2.IMREAD_GRAYSCALE)
        ndwi = calculate_ndwi(rgb_img[:, :, 1], nir_img)
        cv2.imwrite(ndwi_path, ndwi)


def generate_lightness(img_dir: str):
    lightness_dir_path = os.path.join(img_dir, "images", "l")
    rgb_dir = os.path.join(img_dir, "images", "rgb")
    rgb_images = os.listdir(rgb_dir)
    if os.path.exists(lightness_dir_path):
        lightness_images = os.listdir(lightness_dir_path)
        if len(lightness_images) == len(rgb_images):
            print("NDVI has been precomputed. Skipping NDVI computation")
            return
        else:
            rgb_images = set(rgb_images) - set(lightness_images)
    print("Precomputing lightness channel")
    os.makedirs(lightness_dir_path, exist_ok=True)
    for filename in tqdm(rgb_images, total=len(rgb_images)):
        if not filename.endswith(".jpg"):
            continue
        rgb_path = os.path.join(img_dir, "images", "rgb", filename)
        lightness_path = os.path.join(lightness_dir_path, filename)
        rgb_img = cv2.cvtColor(cv2.imread(rgb_path), cv2.COLOR_BGR2RGB)
        ndwi = calculate_lightness(rgb_img)
        cv2.imwrite(lightness_path, ndwi)


def precompute_background_class(test_dir: str, class_names: List[str]):
    background_class_path = os.path.join(test_dir, "labels", "background")
    rgb_dir = os.path.join(test_dir, "images", "rgb")
    rgb_images = os.listdir(rgb_dir)
    if os.path.exists(background_class_path):
        background_images = os.listdir(background_class_path)
        background_images = [bg_img.replace(".png", ".jpg") for bg_img in background_images]
        if len(background_images) == len(rgb_images):
            print("background has been precomputed. Skipping background computation")
            return
        else:
            rgb_images = set(rgb_images) - set(background_images)
    print("Precomputing background class ground truth")
    os.makedirs(background_class_path, exist_ok=True)

    for filename in tqdm(rgb_images, total=len(rgb_images)):
        if not filename.endswith(".jpg"):
            continue
        background_ground_truth = None
        for class_idx, class_name in enumerate(class_names):
            ground_truth_path = os.path.join(test_dir, "labels", class_name, filename.replace('.jpg', '.png'))
            ground_truth = cv2.imread(ground_truth_path, cv2.IMREAD_GRAYSCALE)
            if not isinstance(ground_truth, np.ndarray):
                print(ground_truth_path)
            ground_truth = (ground_truth / 255)
            if background_ground_truth is None:
                background_ground_truth = np.zeros(ground_truth.shape)
            background_ground_truth = np.logical_or(background_ground_truth, ground_truth)
        background_ground_truth = np.logical_not(background_ground_truth).astype(np.uint8) * 255
        cv2.imwrite(os.path.join(background_class_path, f"{filename.replace('jpg', 'png')}"), background_ground_truth)
    print(f"Background class was saved into {background_class_path}")


def run_experiment():
    dataset_df = pd.read_csv(args.dataset_df)
    classes = list(args.class_names)
    if 'background' in classes:
        classes.remove('background')
    for num, ds_dir in enumerate([args.train_dir, args.val_dir, args.test_dir]):

        generate_lightness(ds_dir)
        generate_ndvi(ds_dir)
        generate_ndwi(ds_dir)
        if num != 2:
            precompute_background_class(ds_dir, classes)

    experiment_dir, model_dir, experiment_name = train()
    prediction_dir = os.path.join(experiment_dir, "predictions")
    best_model_name = find_best_model(model_dir)
    weights_path = os.path.join(model_dir, best_model_name)

    test_df_path = args.dataset_df
    test_data_dir = args.val_dir
    print(f"Starting prediction process. Using {best_model_name} for prediction")
    pred_function = predict

    if args.multihead:
        pred_function = predict_multihead
    pred_function(experiment_dir=experiment_dir,
            class_names=args.class_names,
            weights_path=weights_path,
            test_df_path=test_df_path,
            test_data_dir=test_data_dir,
            input_channels=args.channels,
            network=args.network,
            add_classification_head=args.add_classification_head)
    print(f"Starting evaluation process of results in {prediction_dir}")
    evaluate(test_dir=test_data_dir,
             experiment_dir=experiment_dir,
             test_df_path=test_df_path,
             threshold=args.threshold,
             class_names=args.class_names)

    print("Generating submission")
    generate_submission(thresh=args.threshold, weights_path=weights_path)


if __name__ == "__main__":
    run_experiment()
