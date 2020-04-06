import threading
import rasterio
import cv2
import numpy as np
import geopandas as gpd
from tqdm import tqdm
import os
import math
from rasterio.windows import Window
import rasterio.mask
from shapely.ops import split as shapely_split, cascaded_union
from shapely.geometry import Polygon, Point, MultiPoint, LineString, mapping
import json
import shapely
import argparse
# from random_transform_mask import tiles_with_overlap_shape
from rasterio.plot import reshape_as_raster, reshape_as_image
from rasterio import features
import shutil

POLYGON_WIDTH = 20


def calculate_ndvi(red, nir_img):
    diff = nir_img.astype(np.float32)- red.astype(np.float32)
    nir_red = nir_img.astype(np.float32) + red.astype(np.float32)
    ndvi_img = ((1 + diff/nir_red) * 127).astype(np.uint8)
    return ndvi_img

    
def mask_raster_with_polygon(dataset, save_path, shapes, tile_size=500, all_touched=False, invert=False,
                             crop=False, pad=False):
    shape_mask, transform, window = rasterio.mask.raster_geometry_mask(dataset,
                                                                       shapes,
                                                                       all_touched=all_touched,
                                                                       invert=invert,
                                                                       crop=crop,
                                                                       pad=pad)
    masked_meta = dataset.meta.copy()
    masked_meta.update({"driver": "GTiff",
                        "height": window.height,
                        "width": window.width,
                        "transform": transform})
    # Cut window on tiles
    dst = rasterio.open(save_path, "w+", **masked_meta)

    for i in tqdm(range(masked_meta.get('count')), total=masked_meta.get('count')):
        row_s = window.row_off
        col_s = window.col_off
        col_smax = window.col_off + window.width
        col_d = 0
        col_dmax = window.width
        while row_s < window.row_off + window.height:
            row_smax = min(row_s + tile_size, window.row_off + window.height)
            res = dataset.read(window=((row_s, row_smax), (col_s, col_smax)), indexes=i + 1)

            row_d = row_s - window.row_off
            row_dmax = row_smax - window.row_off
            if crop:
                nan_mask = shape_mask[row_d:row_dmax, col_d:col_dmax]
                no_data_value = masked_meta.get('nodata')
                if not no_data_value:
                    if masked_meta.get('dtype') == 'uint8':
                        no_data_value = 0

                    else:
                        no_data_value = -10000
                    masked_meta.update({"nodata": no_data_value})
                res[nan_mask] = no_data_value
            w = Window.from_slices(slice(row_d, row_dmax), slice(col_d, col_dmax))
            dst.write(res, window=w, indexes=i + 1)
            row_s = row_smax
            del res
    dst.close()


def dist(a, b):
    """
    Distance between two points
    """
    return np.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


def freeze_model(model, freeze_before_layer):
    if freeze_before_layer == "ALL":
        for l in model.layers:
            l.trainable = False
    else:
        freeze_before_layer_index = -1
        for i, l in enumerate(model.layers):
            if l.name == freeze_before_layer:
                freeze_before_layer_index = i
        for l in model.layers[:freeze_before_layer_index]:
            l.trainable = False


def save_gpd_df(save_path, geom_df, meta=None):
    if meta:
        geom_df.crs = meta.get('crs')
    if os.path.exists(save_path):
        os.remove(save_path)
    geom_df.to_file(save_path, driver='GeoJSON', encoding='utf-8')


class ThreadsafeIter(object):
    def __init__(self, it):
        self.lock = threading.Lock()
        self.it = it.__iter__()

    def __iter__(self): return self

    def __next__(self):
        with self.lock:
            return next(self.it)


def split_lines(multiline):
    # get vertices as shapely points
    vertices_list = [Point(pt) for pt in list(zip(*multiline.coords.xy))]

    # get lines fron polygon
    lines = shapely_split(multiline, MultiPoint(vertices_list))
    return lines


def read_json(path):
    #logging.info(path)
    with open(path, 'r') as json_data:
        d = json.load(json_data)
    return d

def get_azimuth_polyline(line):
    radian = math.atan((line.coords[1][0] - line.coords[0][0])/(line.coords[1][1] - line.coords[0][1]))
    degrees = np.rad2deg(radian)
    return degrees


def create_raster_list_for_root(folder):
    d = {}
    groves = os.listdir(folder)
    for grove in groves:
        grove_folder = os.path.join(folder, grove)
        if not os.path.isdir(grove_folder):
            continue
        print(grove)
        d[grove] = create_raster_list(grove_folder)
    return d


def create_raster_list(grove_folder):
    rasters_to_search = ['rgb', 'nir', 'red', 'green', 'blue']
    raster_list = {}
    for file in os.listdir(grove_folder):
        if not file.endswith(".tif"):
            continue
        for channel in rasters_to_search:
            if file.lower().startswith(channel) or file.lower().endswith(channel + ".tif"):
                path = os.path.join(grove_folder, file)
                raster_list[channel] = {'path': path}
                continue
    return raster_list

def get_perpendicular(point, angle, polygon_width=POLYGON_WIDTH):
    """
    point = np.array
    """
    # line_coords_np = np.array(line.coords)
    # print(np.linalg.norm(line_coords_np[0] - line_coords_np[1]))
    beta = - angle
    half_line = polygon_width / 2
    x_coord = math.cos(np.deg2rad(beta)) * half_line
    y_coord = math.sin(np.deg2rad(beta)) * half_line
    new_line = np.empty((2, 2))
    new_line[0][0] = point[0] + x_coord
    new_line[0][1] = point[1] + y_coord
    new_line[1][0] = point[0] - x_coord
    new_line[1][1] = point[1] - y_coord
    #     new_start = [point[0] + x_coord, point[1] - y_coord]
    #     new_end = [point[0] - x_coord, point[1] + y_coord]
    shp_line = LineString(new_line)
    return shp_line, new_line


def get_perpendiculars(lines, angles):
    perpendiculars = []
    polygons = []
    for line, angle in zip(lines, angles):
        point_arr = []
        # only for second point without last
        for i in range(1, 2):
            inp = line.coords[i]
            line_shp, line_arr = get_perpendicular(inp, angle)
            perpendiculars.append(line_shp)
            point_arr.append(line_arr)
        #polygon_shp = Polygon((point_arr[0][0], point_arr[0][1], point_arr[1][1], point_arr[1][0]))
        #polygons.append(polygon_shp)
    return perpendiculars[1:-1]


def get_perpendiculars_each_n_meters(lines, angles, step=1):
    perpendiculars = []
    for line, angle in zip(lines, angles):
        point_arr = []
        # only for second point without last
        pts = np.arange(0, line.length, step)
        for pt in pts:
        #for i in range(1, 2):
            inp = line.interpolate(pt, normalized=False)
            inp = inp.coords[0]
            line_shp, line_arr = get_perpendicular(inp, angle, polygon_width=3)
            perpendiculars.append(line_shp)
            point_arr.append(line_arr)
        #polygon_shp = Polygon((point_arr[0][0], point_arr[0][1], point_arr[1][1], point_arr[1][0]))
        #polygons.append(polygon_shp)
    return perpendiculars


def polygons_to_image_view(window_cut_polygons, dataset, rect=None, downscale=None):
    if rect:
        col_offset = rect[0]
        row_offset = rect[1]
    img_polyg_array = []
    to_img_mat = ~dataset.meta['transform']
    if not downscale:
        downscale = 1
    for item in window_cut_polygons.iterrows():
        if item[1]['geometry'].type != 'Polygon':
            # polyg_spati = np.array(cascaded_union(item[1]['geometry']).exterior.coords)
            continue
        else:
            polyg_spati = np.array(item[1]['geometry'].exterior.coords)
        polyg_img = [np.array(tuple(pt) * to_img_mat) / downscale for pt in polyg_spati]
        if rect:
            polyg_img = np.subtract(polyg_img, (col_offset, row_offset))
        polyg_img = Polygon(polyg_img)
        # polyg_img = shapely_scale(polyg_img, 1/downscale, 1/downscale, origin=(0, 0, 0))
        # polyg_img = polyg_img.convex_hull
        img_polyg_array.append(polyg_img)

    return img_polyg_array


def points_to_image_view(points, rect, dataset):
    col_offset = rect[0]
    row_offset = rect[1]
    img_polyg_array = []
    to_img_mat = ~dataset.meta['transform']
    for item in points.iterrows():
        polyg_spati = np.array(item[1]['geometry'].coords)
        polyg_img = [tuple(pt) * to_img_mat for pt in polyg_spati]
        polyg_img = np.subtract(polyg_img, (col_offset, row_offset))
        polyg_img = LineString(polyg_img)
        # polyg_img = polyg_img.convex_hull
        img_polyg_array.append(polyg_img)

    return img_polyg_array


def get_polygons_in_window(dataframe, window):
    filtered = dataframe.geometry.apply(lambda p: window.intersects(p))
    filtered = gpd.GeoDataFrame(filtered)
    filtered = filtered.loc[(filtered['geometry'] == True)]
    merged_DB = dataframe.join(filtered.drop(columns=['geometry']), how="inner")
    return merged_DB


def lines_to_image_view(points, dataset, downscale=None, rect=None):
    if rect:
        col_offset = rect[0]
        row_offset = rect[1]
    img_polyg_array = []

    to_img_mat = ~dataset.meta['transform']
    if not downscale:
        downscale = 1
    for item in points.iterrows():
        polyg_spati = np.array(item[1]['geometry'].coords)
        polyg_img = [np.array(tuple(pt) * to_img_mat) / downscale for pt in polyg_spati]
        if rect:
            polyg_img = np.subtract(polyg_img, (col_offset, row_offset))
        polyg_img = LineString(polyg_img)
        # polyg_img = polyg_img.convex_hull
        img_polyg_array.append(polyg_img)

    return img_polyg_array