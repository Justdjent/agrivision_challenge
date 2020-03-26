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


def generate_mosaics(dataframe_path, save_folder, dataset_path):
    src = rasterio.open("/home/user/projects/geo/alignment-dev/1-167/non_aligned/project_transparent_mosaic_red.tif")

    full_df = pd.read_csv(dataframe_path)
    full_df['id'] = full_df['name'].str.split(".", n=1, expand=True)[0]
    full_df['coords'] = full_df['id'].str.split("_", n=1, expand=True)[1]
    full_df[['x_start', 'y_start', 'x_end', 'y_end']] = full_df['coords'].str.split("-", expand=True)
    full_df[['x_start', 'y_start', 'x_end', 'y_end']] = full_df[['x_start', 'y_start', 'x_end', 'y_end']].astype(int)
    field_list = list(full_df['field'].unique())
    for field in tqdm(field_list):
        field_df = full_df[full_df['field'] == field]
        height = field_df['y_end'].max()
        width = field_df['x_end'].max()
        meta = {"nodata": 0,
                'crs': 'epsg:32611',
                'compress': 'JPEG',
                "height": height,
                "width": width,
                "dtype": 'uint8',
                "count": 3,
                #166021.44 0.00
                "transform": Affine(1, 0, 166021.44, 0, 1, 0),
                'driver': 'GTiff'
                }
        print(height, width)
        filename = os.path.join(save_folder, field + ".tif")
        with rasterio.open(filename, "w", **meta) as dst:
            for num, row in field_df.iterrows():
                name = row['name']
                x_off = row['x_start']
                y_off = row['y_start']
                img_path = os.path.join(dataset_path, "images", "rgb", name)
                img = cv2.imread(img_path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = reshape_as_raster(img)
                try:
                    dst.write(img, window=Window(int(x_off), int(y_off), img.shape[1], img.shape[2]))
                except:
                    print('hi')

if __name__ == '__main__':
    dataframe_path = args.full_df_path
    save_raster_folder = args.raster_folder
    dataset_path = args.manual_dataset_dir
    generate_mosaics(dataframe_path, save_raster_folder, dataset_path)