import os

import pandas as pd
import numpy as np
from tqdm import tqdm
import cv2
from collections import OrderedDict
import rasterio
from rasterio.plot import reshape_as_raster
from research_code.params import args
from rasterio import Affine
from rasterio.windows import Window
from tqdm import tqdm
from contextlib import ExitStack


def generate_mosaics(dataframe_path, save_folder, dataset_path):
    full_df = pd.read_csv(dataframe_path)
    full_df['id'] = full_df['name'].str.split(".", n=1, expand=True)[0]
    full_df['coords'] = full_df['id'].str.split("_", n=1, expand=True)[1]
    full_df[['x_start', 'y_start', 'x_end', 'y_end']] = full_df['coords'].str.split("-", expand=True)
    full_df[['x_start', 'y_start', 'x_end', 'y_end']] = full_df[['x_start', 'y_start', 'x_end', 'y_end']].astype(int)
    field_list = list(full_df['field'].unique())

    rasters_to_save = args.class_names
    rasters_to_save.extend(['nir', 'rgb', 'boundary'])
    for field in tqdm(field_list):
        field_folder = os.path.join(save_folder, field)
        os.makedirs(field_folder, exist_ok=True)
        field_df = full_df[full_df['field'] == field]
        height = field_df['y_end'].max()
        width = field_df['x_end'].max()
        rgb_meta = {"nodata": 0,
                'crs': 'epsg:32611',
                'compress': 'JPEG',
                "height": height,
                "width": width,
                "dtype": 'uint8',
                "count": 3,
                #166021.44 0.00
                "transform": Affine(0.1, 0, 166021.44, 0, -0.1, 0),
                'driver': 'GTiff',
                'tiled': True,
                'block_size': 256
                }
        meta = rgb_meta.copy()
        meta.update({"count": 1})

        with ExitStack() as stack:
            save_contexts = []
            for raster in rasters_to_save:
                if raster == 'rgb':
                    cur_meta = rgb_meta
                else:
                    cur_meta = meta
                filename = os.path.join(field_folder, raster + ".tif")
                save_contexts.append(stack.enter_context(rasterio.open(filename, "w", **cur_meta)))

        # with rasterio.open(filename, "w", **meta) as dst:
            for num, row in field_df.iterrows():
                name = row['name']
                x_off = row['x_start']
                y_off = row['y_start']

                for cls, raster in zip(rasters_to_save, save_contexts):

                    if cls in ['rgb', 'nir']:
                        name = name.replace(".png", ".jpg")
                        cur_img = read_image(cls, dataset_path, name, 'images')
                    elif cls == 'boundary':
                        name = name.replace(".jpg", ".png")
                        cur_img = read_image(cls, dataset_path, name, 'boundaries')
                    else:
                        name = name.replace(".jpg", ".png")
                        cur_img = read_image(cls, dataset_path, name, 'labels')
                    try:
                        raster.write(cur_img, window=Window(int(x_off), int(y_off), cur_img.shape[1], cur_img.shape[2]))
                    except:
                        print('hi')


def read_image(cls, dataset_path, name, folder):
    if cls == 'boundary':
        img_path = os.path.join(dataset_path, folder, name)
    else:
        img_path = os.path.join(dataset_path, folder, cls, name)

    if cls == 'rgb':
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = reshape_as_raster(img)
    else:
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img = np.expand_dims(img, axis=0)
    return img


if __name__ == '__main__':
    dataframe_path = args.full_df_path
    save_raster_folder = args.raster_folder
    dataset_path = args.manual_dataset_dir
    generate_mosaics(dataframe_path, save_raster_folder, dataset_path)