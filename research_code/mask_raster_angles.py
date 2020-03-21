import sys
from tqdm import tqdm
tqdm.monitor_interval = 0
import shutil


import json
from osgeo import ogr
import rasterio.mask
from keras.applications import imagenet_utils
from shapely.geometry import Polygon
import logging
import os
import cv2
# from tensorflow.keras.preprocessing.image import flip_axis
from rasterio.plot import reshape_as_raster, reshape_as_image
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
import tensorflow.keras.backend as K

from row_detection_research_code.models import make_model

from row_detection_research_code.params import args
from row_detection_research_code.random_transform_mask import pad_size, unpad, get_bigger_window

logging.basicConfig(stream=sys.stdout, format='%(asctime)s %(message)s', datefmt=' %I:%M:%S ', level="INFO")

import rasterio
import numpy as np
from row_detection_research_code.utils import read_json, create_raster_list_for_root

def flip_axis(x, axis):
  x = np.asarray(x).swapaxes(axis, 0)
  x = x[::-1, ...]
  x = x.swapaxes(0, axis)
  return x


def do_tta(x, tta_type):
    if tta_type == 'hflip':
        # batch, img_col = 2
        return flip_axis(x, 2)
    elif tta_type == 'vflip':
        # batch, img_col = 2
        return flip_axis(x, 1)
    else:
        return x


def undo_tta(pred, tta_type):
    if tta_type == 'hflip':
        # batch, img_col = 2
        return flip_axis(pred, 2)
    elif tta_type == 'vflip':
        # batch, img_col = 2
        return flip_axis(pred, 1)
    else:
        return pred

class Reader:
    def __init__(self, raster_list: dict):
        self.raster_array = None
        self.meta = None
        self.raster_list = raster_list

    def load_stack(self):
        self.raster_array = {}
        self.meta = {}
        for r_type, path in self.raster_list.items():
            with rasterio.open(path, 'r') as src:
                self.raster_array[r_type] = src.read()
                self.meta[r_type] = src.meta

    def save_raster(self, raster_type, save_path=None):
        if not save_path:
            save_path = self.raster_list[raster_type]
        with rasterio.open(save_path, 'w', **self.meta[raster_type]) as dst:
            for i in range(1, self.meta[raster_type]['count'] + 1):
                src_array = self.raster_array[raster_type][i - 1]
                dst.write(src_array, i)

    @staticmethod
    def min_max(X, X_min=None, X_max=None):
        if X_min:
            X_std = (X - X_min) / (X_max - X_min)
            X_scaled = X_std * (1 - 0) + 0
        else:
            X_std = (X - X.min()) / (X.max() - X.min())
            X_scaled = X_std * (1 - 0) + 0
        return X_scaled


class SegmentatorNN:
    def __init__(self, inp_list, thresh, r_type):
        # self.raster_path = raster_path
        # self.raster_type = raster_type
        self.r_type = r_type
        if isinstance(inp_list, dict):
            inp = inp_list
        else:
            inp = read_json(inp_list)
        self.thresh = thresh
        self.reader = Reader(inp)

    def mask_tiles(self, save_mask=True, window_size=60):

        model = make_model((2*args.input_width, 2*args.input_height, 3))
        model.load_weights(args.weights)
        max_values = [1, 1, 1]
        min_values = [0, 0, 0]
        with rasterio.open(self.reader.raster_list[self.r_type]['path'], 'r') as dataset:
            meta = dataset.meta
            if args.network in ['angle_net', 'dist_net', 'dist_net_unet', 'dist_net_resnet']:
                num_channels = 2
            else:
                num_channels = 2
            raster_array = np.zeros((num_channels, dataset.meta['height'], dataset.meta['width']), np.uint8)
            xs = dataset.bounds.left
            window_size_meters = window_size
            window_size_pixels = window_size / (dataset.res[0])
            cnt = 0
            pbar = tqdm()
            while xs < dataset.bounds.right:
                ys = dataset.bounds.bottom
                while ys < dataset.bounds.top:
                    row, col = dataset.index(xs, ys)
                    pbar.set_postfix(Row='{}'.format(row), Col='{}'.format(col))
                    step_row = row - int(window_size_pixels)
                    step_col = col + int(window_size_pixels)
                    new_window, pads = get_bigger_window(step_row, row,
                                                         step_col, col,
                                                         window_size_pixels,
                                                         dataset.height, dataset.width)
                    # res = dataset.read(window=((max(0, step_row), row),
                    #                            (col, step_col)))
                    res = dataset.read(window=((max(0, new_window[0]), new_window[1]),
                                               (max(0, new_window[2]), new_window[3])))

                    rect = [[max(0, step_row), row], [col, step_col]]
                    # if res.max() > 0:
                    #     print('hi')
                    if res.dtype == 'float32':
                        if res.max() > 1 or res.max() < 0.02:
                            res[res < 0] = 0
                            if 'rgb' not in self.reader.raster_list.keys() and 'rgbn' not in self.reader.raster_list.keys():
                                res = self.min_max(res, min=min_values, max=max_values)
                        if 'rgb' not in self.reader.raster_list.keys() and 'rgbn' not in self.reader.raster_list.keys():
                            if res.max() < 2:
                                res = self.process_float(res)
                        res = res.astype(np.uint8)
                    # img_size = tuple([res.shape[2], res.shape[1]])
                    cv_res = reshape_as_image(res)[:, :, :3]
                    if not all(pad_size == 0 for pad_size in pads):
                        cv_res = cv2.copyMakeBorder(cv_res, pads[0], pads[1], pads[2], pads[3], cv2.BORDER_REFLECT_101)
                    img_size = tuple([cv_res.shape[1], cv_res.shape[0]])
                    try:
                        cv_res = cv2.resize(cv_res, (2*args.input_width, 2*args.input_width), cv2.INTER_NEAREST)
                    except:
                        pbar.update(1)

                        cnt += 1
                        ys = ys + 0.5 * window_size_meters
                        continue
                    # cv_res = cv2.resize(cv_res, (448, 448))

                    cv_res = np.expand_dims(cv_res, axis=0)
                    x = imagenet_utils.preprocess_input(cv_res, mode=args.preprocessing_function)
                    if args.pred_tta:
                        x_tta = do_tta(x, "hflip")
                        batch_x = np.concatenate((x, x_tta), axis=0)
                    else:
                        batch_x = x

                    preds = model.predict_on_batch(batch_x)

                    if args.network in ['angle_net', 'dist_net', 'dist_net_unet', 'dist_net_resnet']:
                        #@ preds_prepared = np.argmax(preds[0], axis=-1).astype(np.uint8)
                        preds_prepared = np.stack([np.argmax(preds[0][0], axis=-1).astype(np.uint8),
                                                   preds[1][0, :, :, 0] * 255], axis=-1)
                    else:
                        preds_prepared = np.stack([preds[0][0, :, :, 0] * 255,
                                                   preds[1][0, :, :, 0] * 255], axis=-1)
                    if args.pred_tta:
                        preds_tta = undo_tta(preds[1:2], "hflip")
                        pred = (preds[:1] + preds_tta) / 2
                    else:
                        pred = preds

                    preds_prepared = cv2.resize(preds_prepared, img_size)
                    # if args.network in ['angle_net', 'dist_net', 'dist_net_unet', 'dist_net_resnet']:
                    #     #preds_prepared = np.rollaxis(preds_prepared, 2, 0).astype(np.uint8)
                    #     # pred = unpad(pred[0], pads)
                    #     stack_arr = np.stack([preds_prepared, raster_array[0, rect[0][0]:rect[0][1], rect[1][0]:rect[1][1]]], axis=-1)
                    #     stack_arr = np.amax(stack_arr, axis=-1)
                    # else:
                    # if not all(pad_size == 0 for pad_size in pads):
                        #preds_prepared = unpad(preds_prepared, pads)
                    preds_prepared = preds_prepared[max(pads[0], int(window_size_pixels/2)):preds_prepared.shape[0] - max(int(window_size_pixels/2), pads[1]),
                                                    max(pads[2], int(window_size_pixels/2)):preds_prepared.shape[1] - max(pads[3], int(window_size_pixels/2))]
                    preds_prepared = np.rollaxis(preds_prepared, 2, 0).astype(np.uint8)

                    # pred = unpad(pred[0], pads)
                    try:
                        stack_arr = np.stack([preds_prepared, raster_array[:, rect[0][0]:rect[0][1], rect[1][0]:rect[1][1]]], axis=-1)
                        stack_arr = np.amax(stack_arr, axis=-1)
                    except:
                        print('hi')
                    #stack_arr = np.stack([pred, raster_array[:, rect[0][0]:rect[0][1], rect[1][0]:rect[1][1]]])

                    raster_array[:, rect[0][0]:rect[0][1], rect[1][0]:rect[1][1]] = stack_arr
                    # raster_array[rect[0][0]:rect[0][1], rect[1][0]:rect[1][1]] = np.mean(stack_arr, axis=2)

                    pbar.update(1)

                    cnt += 1
                    ys = ys + 0.5 * window_size_meters

                xs = xs + 0.5 * window_size_meters

        # Save raster
        bin_meta = meta.copy()
        bin_meta.update({"count": num_channels,
                         "dtype": "uint8",
                         "nodata": 0})

        bin_raster_path = self.reader.raster_list[self.r_type]['path'].replace(self.r_type, '{}_bin'.format(self.r_type))
        raster_name = bin_raster_path.split("/")[-1]
        raster_dir = os.path.dirname(bin_raster_path)
        save_raster_dir = os.path.join(raster_dir, "dist_net_resnet_1570_256_tuned_lines")
        os.makedirs(save_raster_dir, exist_ok=True)
        save_raster_path = os.path.join(save_raster_dir, raster_name)
        save_single_raster(raster_array, bin_meta, save_raster_path)

        del model
        K.clear_session()
        return raster_array


    def polygonize(self, contours, meta, transform=True):
        polygons = []
        for i in tqdm(range(len(contours))):
            c = contours[i]
            n_s = (c.shape[0], c.shape[2])
            if n_s[0] > 2:
                if transform:
                    polys = [tuple(i) * meta['transform'] for i in c.reshape(n_s)]
                else:
                    polys = [tuple(i) for i in c.reshape(n_s)]
                polygons.append(Polygon(polys))
        return polygons

    @staticmethod
    def process_float(array):
        array = array.copy()
        array[array < 0] = 0
        array_ = np.uint8(array * 255)
        return array_

    @staticmethod
    def min_max(X, min, max):
        X_scaled = np.zeros(X.shape)
        for i in range(X.shape[0]):
            X_std = (X[i] - min[i]) / (max[i] - min[i])
            X_scaled[i] = X_std * (1 - 0) + 0

        return X_scaled


def save_single_raster(raster_array, meta, save_path):
    with rasterio.open(save_path, 'w', **meta) as dst:
        for i in range(1, meta['count'] + 1):
            src_array = raster_array[i - 1]
            dst.write(src_array, i)


def save_polys_as_shp(polys, name):
    # Now convert it to a shapefile with OGR
    driver = ogr.GetDriverByName('Esri Shapefile')
    ds = driver.CreateDataSource(name)
    layer = ds.CreateLayer('', None, ogr.wkbPolygon)
    # Add one attribute
    layer.CreateField(ogr.FieldDefn('id', ogr.OFTInteger))
    defn = layer.GetLayerDefn()

    # If there are multiple geometries, put the "for" loop here
    for i in range(len(polys)):
        # Create a new feature (attribute and geometry)
        feat = ogr.Feature(defn)
        feat.SetField('id', i)

        # Make a geometry, from Shapely object
        geom = ogr.CreateGeometryFromWkb(polys[i].wkb)
        feat.SetGeometry(geom)

        layer.CreateFeature(feat)
        # feat = geom = None  # destroy these

    # Save and close everything
    # ds = layer = feat = geom = None





def raster_to_img(raster, raster_list, r_type):
    if raster.shape[0] > 1:
        cv_res = np.zeros((raster.shape[1], raster.shape[2], raster.shape[0]))
        if 'rgb' in raster_list.keys() and np.mean(cv_res) > 90:
            cv_res[:, :, 0] = raster[0] / 5.87
            cv_res[:, :, 1] = raster[1] / 5.95
            cv_res[:, :, 2] = raster[2] / 5.95
        elif 'rgbn' in raster_list.keys() and r_type == 'rgg':
            cv_res[:, :, 0] = raster[0] / 5.76
            cv_res[:, :, 1] = raster[1] / 5.08
            cv_res[:, :, 2] = raster[2] / 5.08
        elif 'rgbn' in raster_list.keys() and r_type == 'nrg':
            cv_res[:, :, 0] = raster[0] / 2.27
            cv_res[:, :, 1] = raster[1] / 5.76
            cv_res[:, :, 2] = raster[2] / 5.08
        else:
            cv_res[:, :, 0] = raster[0]
            cv_res[:, :, 1] = raster[1]
            cv_res[:, :, 2] = raster[2]
    else:
        cv_res = raster[0]
    return cv_res





def run():
    if os.path.isdir(args.inp_list):
        d = create_raster_list_for_root(args.inp_list)
    else:

        d = read_json(args.inp_list)
    thresh = args.threshold
    r_type = args.r_type
    for grove, inp in d.items():
        print(grove)
        segmentator = SegmentatorNN(inp, thresh, r_type)
        #try:
        segmentator.mask_tiles()
        # except Exception as e:
        #     print(e)



if __name__ == '__main__':
    # predict()
    run()