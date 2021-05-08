import os
import cv2
import glob

import numpy as np
import pandas as pd
from pandarallel import pandarallel
pandarallel.initialize(verbose=2)
from tqdm import tqdm
from typing import List
from research_code.params import args
from research_code.train import train
from research_code.predict_masks import predict
from research_code.evaluate import evaluate, Evaluator
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


def generate_ndvi(img_dir: str, input_df: pd.DataFrame, ds_part):
    ndvi_dir_path = os.path.join(img_dir, ds_part, "images", "ndvi")
    if os.path.exists(ndvi_dir_path):
        if len(os.listdir(ndvi_dir_path)) == len(input_df):
            print("NDVI has been precomputed. Skipping NDVI computation")
            return
    else:
        print("Precomputing NDVI channel")
        os.makedirs(ndvi_dir_path)
    for idx, row in tqdm(input_df.iterrows(), total=len(input_df)):
        filename = row["name"]
        rgb_path = os.path.join(img_dir, ds_part, "images", "rgb", filename)
        nir_path = os.path.join(img_dir, ds_part, "images", "nir", filename)
        ndvi_path = os.path.join(ndvi_dir_path, filename)
        if os.path.exists(ndvi_path):
            continue
        rgb_img = cv2.cvtColor(cv2.imread(rgb_path), cv2.COLOR_BGR2RGB)
        nir_img = cv2.imread(nir_path, cv2.IMREAD_GRAYSCALE)
        ndvi = calculate_ndvi(rgb_img[:, :, 0], nir_img)
        cv2.imwrite(ndvi_path, ndvi)

def generate_ndvi_parralel(img_dir: str, input_df: pd.DataFrame, ds_part):
    ndvi_dir_path = os.path.join(img_dir, ds_part, "images", "ndvi")
    if os.path.exists(ndvi_dir_path):
        if len(os.listdir(ndvi_dir_path)) == len(input_df):
            print("NDVI has been precomputed. Skipping NDVI computation")
            return
    else:
        print("Precomputing NDVI channel")
        os.makedirs(ndvi_dir_path)
    input_df.parallel_apply(calc_ndvi, ds_part=ds_part, axis=1)

def calc_ndvi(row, ds_part):
    filename = row["name"]
    rgb_path = os.path.join(img_dir, ds_part, "images", "rgb", filename)
    nir_path = os.path.join(img_dir, ds_part, "images", "nir", filename)
    ndvi_path = os.path.join(ndvi_dir_path, filename)
    if os.path.exists(ndvi_path):
        return
    rgb_img = cv2.cvtColor(cv2.imread(rgb_path), cv2.COLOR_BGR2RGB)
    nir_img = cv2.imread(nir_path, cv2.IMREAD_GRAYSCALE)
    ndvi = calculate_ndvi(rgb_img[:, :, 0], nir_img)
    cv2.imwrite(ndvi_path, ndvi)


def generate_ndwi(img_dir: str, input_df: pd.DataFrame, ds_part):
    ndwi_dir_path = os.path.join(img_dir, ds_part, "images", "ndwi")
    if os.path.exists(ndwi_dir_path):
        if len(os.listdir(ndwi_dir_path)) == len(input_df):
            print("NDWI has been precomputed. Skipping NDWI computation")
            return
    else:
        print("Precomputing NDWI channel")
        os.makedirs(ndwi_dir_path)
    for idx, row in tqdm(input_df.iterrows(), total=len(input_df)):
        filename = row["name"]
        rgb_path = os.path.join(img_dir, ds_part, "images", "rgb", filename)
        nir_path = os.path.join(img_dir, ds_part, "images", "nir", filename)
        ndwi_path = os.path.join(ndwi_dir_path, filename)
        if os.path.exists(ndwi_path):
            continue
        rgb_img = cv2.cvtColor(cv2.imread(rgb_path), cv2.COLOR_BGR2RGB)
        nir_img = cv2.imread(nir_path, cv2.IMREAD_GRAYSCALE)
        ndwi = calculate_ndwi(rgb_img[:, :, 1], nir_img)
        cv2.imwrite(ndwi_path, ndwi)


def generate_lightness(img_dir: str, input_df: pd.DataFrame, ds_part):
    lightness_dir_path = os.path.join(img_dir, ds_part, "images", "l")
    if os.path.exists(lightness_dir_path):
        if len(os.listdir(lightness_dir_path)) == len(input_df):
            print("Lightness has been precomputed. Skipping lightness computation")
            return
    else:
        print("Precomputing lightness channel")
        os.makedirs(lightness_dir_path)
    for idx, row in tqdm(input_df.iterrows(), total=len(input_df)):
        filename = row["name"]
        rgb_path = os.path.join(img_dir, ds_part, "images", "rgb", filename)
        if not os.path.exists(rgb_path):
            raise("{} not exist".format(rgb_path))
        lightness_path = os.path.join(lightness_dir_path, filename)
        if os.path.exists(lightness_path):
            continue
        rgb_img = cv2.cvtColor(cv2.imread(rgb_path), cv2.COLOR_BGR2RGB)
        ndwi = calculate_lightness(rgb_img)
        cv2.imwrite(lightness_path, ndwi)


def precompute_background_class(test_dir: str, test_df: pd.DataFrame, class_names: List[str], ds_part):
    background_class_path = os.path.join(test_dir, ds_part, "labels", "background")
    if os.path.exists(background_class_path):
        if len(os.listdir(background_class_path)) == len(test_df):
            print("Background class has been precomputed. Skipping background class computation")
            return
    else:
        print("Precomputing background class ground truth")
        os.makedirs(background_class_path)
    print(test_dir)
    for idx, row in tqdm(test_df.iterrows(), total=len(test_df)):
        filename = row['name']
        background_ground_truth = None
        bg_save_path = os.path.join(background_class_path, f"{filename.replace('jpg', 'png')}")
        if os.path.exists(bg_save_path):
            continue
        for class_idx, class_name in enumerate(class_names):
            ground_truth_path = os.path.join(test_dir, ds_part, "labels", class_name, filename.replace('.jpg', '.png'))
            if not os.path.exists(ground_truth_path):
                print(ground_truth_path)
                raise
            ground_truth = cv2.imread(ground_truth_path, cv2.IMREAD_GRAYSCALE)
            ground_truth = (ground_truth > 127).astype(np.float32)
            if background_ground_truth is None:
                background_ground_truth = np.zeros(ground_truth.shape)
            background_ground_truth = np.logical_or(background_ground_truth, ground_truth)
        background_ground_truth = np.logical_not(background_ground_truth).astype(np.uint8) * 255
        cv2.imwrite(bg_save_path, background_ground_truth)
    print(f"Background class was saved into {background_class_path}")


def precompute_background_class_parralel(test_dir: str, test_df: pd.DataFrame, class_names: List[str], ds_part):
    background_class_path = os.path.join(test_dir, ds_part, "labels", "background")
    if os.path.exists(background_class_path):
        if len(os.listdir(background_class_path)) == len(test_df):
            print("Background class has been precomputed. Skipping background class computation")
            return
    else:
        print("Precomputing background class ground truth")
        os.makedirs(background_class_path)
    test_df.parallel_apply(calc_background, background_class_path=background_class_path, class_names=class_names, test_dir=test_dir, axis=1)


def calc_background(row, background_class_path, class_names, test_dir):
    filename = row['name']
    background_ground_truth = None
    bg_save_path = os.path.join(background_class_path, f"{filename.replace('jpg', 'png')}")
    if os.path.exists(bg_save_path):
        return
    for class_idx, class_name in enumerate(class_names):
        ground_truth_path = os.path.join(test_dir, row['ds_part'], "labels", class_name, filename.replace('.jpg', '.png'))
        if not os.path.exists(ground_truth_path):
            print(ground_truth_path)
            raise
        ground_truth = cv2.imread(ground_truth_path, cv2.IMREAD_GRAYSCALE)
        ground_truth = (ground_truth > 127).astype(np.float32)
        if background_ground_truth is None:
            background_ground_truth = np.zeros(ground_truth.shape)
        background_ground_truth = np.logical_or(background_ground_truth, ground_truth)
    background_ground_truth = np.logical_not(background_ground_truth).astype(np.uint8) * 255
    cv2.imwrite(bg_save_path, background_ground_truth)


def run_experiment():
    dataset_df = pd.read_csv(args.dataset_df)
    print(dataset_df.columns)
    classes = list(args.class_names)
    print(len(dataset_df))
    if 'background' in classes:
        classes.remove('background')
    for ds_part, ds_dir in zip(['train', 'val'], [args.train_dir, args.val_dir]):
        df = dataset_df[dataset_df['ds_part'] == ds_part]
        precompute_background_class_parralel(ds_dir, df, classes, ds_part)
        # generate_lightness(ds_dir, df, ds_part)
        generate_ndvi_parralel(ds_dir, df, ds_part)
        # generate_ndwi(ds_dir, df, ds_part)

    iterations_number = 6
    prev_mask_dir = None
    iteration_dataframe = None
    weights_path = None
    for iteration in range(iterations_number):
        experiment_dir, model_dir, experiment_name = train(iteration, masks_dir=prev_mask_dir, iter_df=iteration_dataframe, weights_path=weights_path)
        prediction_dir = os.path.join(experiment_dir, "predictions")
        best_model_name = find_best_model(model_dir)
        weights_path = os.path.join(model_dir, best_model_name)

        test_df_path = args.dataset_df
        test_data_dir = args.val_dir
        print(f"Starting prediction process. Using {best_model_name} for prediction")
        predict(experiment_dir=experiment_dir,
                class_names=args.class_names,
                weights_path=weights_path,
                test_df_path=test_df_path,
                test_data_dir=test_data_dir,
                input_channels=args.channels,
                network=args.network,
                add_classification_head=args.add_classification_head)
        print(f"Starting evaluation process of results in {prediction_dir}")
        #TODO only first
        confidence_df = calculate_confidence_for_preds(prediction_dir, args.class_names[0])
        new_iteration_training = confidence_df[confidence_df['conf'] > 0.9]
        new_iteration_training.reset_index(inplace=True, drop=True)
        print("{} images after iteration {}".format(len(new_iteration_training), iteration))
        prev_mask_dir = prediction_dir
        iteration_dataframe = new_iteration_training
        new_iteration_training.to_csv(args.dataset_df.replace(".csv", "_{}_{}".format(args.class_names[0], iteration)))


def get_confidence(row):
    img = cv2.imread(row['path'], cv2.IMREAD_GRAYSCALE)
    gray = ((0.3 * 255 < img) & (img < 0.7 * 255)).sum()
    good = (img > 0.5 * 255).sum()
    confidence = 1 - (gray + 0.0001) / (good + 0.0001)
    return confidence


def calculate_confidence_for_preds(predictions_path, cls):
    path_list = glob.glob(os.path.join(predictions_path, cls, "*"))
    output_df = pd.DataFrame(columns=['path', 'name', 'conf'])
    output_df['path'] = path_list
    output_df['name'] = output_df['path'].str.split("/").str[-1]
    output_df['conf'] = output_df.parallel_apply(get_confidence, axis=1)
    return output_df


if __name__ == "__main__":
    run_experiment()
