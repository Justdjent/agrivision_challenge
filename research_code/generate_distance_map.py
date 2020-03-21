import os
import cv2
import geopandas as gpd
import numpy as np
from rasterio import features
from tqdm import tqdm
from shapely.ops import cascaded_union
from itertools import combinations
from shapely.affinity import translate
from shapely.geometry import Polygon, LineString, Point, MultiPoint
from shapely.ops import nearest_points
import shutil
import argparse
# from utils import polygons_to_image_view, get_polygons_in_window, lines_to_image_view
from tqdm import tqdm
import rasterio
from rasterio import Affine
import matplotlib.pyplot as plt
from row_detection_research_code.utils import save_gpd_df, dist, polygons_to_image_view, get_polygons_in_window, lines_to_image_view

def img_to_proba(lines, polys, shape, threshold=30):
    """
    Calculates a "probability" of a polygon center

    :param line: list of polygon centers
    :param polys: list of polygons
    :param shape: image shape
    :return: array with tree center probabilities
    """
    dist_mask = np.expand_dims(np.zeros(shape), axis=2)
    for polygon, line in tqdm(zip(polys, lines)):
        if len(polygon.bounds) == 0:
            continue
        minx, miny, maxx, maxy = [int(x) for x in polygon.bounds]
        # make sure we are in bounds
        minx = np.clip(minx, 0, shape[1])
        maxx = np.clip(maxx, 0, shape[1])
        miny = np.clip(miny, 0, shape[0])
        maxy = np.clip(maxy, 0, shape[0])

        mask = features.rasterize([polygon], shape).astype("float32")
        window = mask[miny : maxy + 1, minx : maxx + 1].copy()
        window_binary = window.copy()

        tree = np.nonzero(window)

        for y, x in zip(tree[0], tree[1]):
            pt_shp = Point([x + minx, y + miny])
            #nearest_point = line.interpolate(line.project(pt_shp))
            nearest_pt = nearest_points(pt_shp, line)[1]
            length = pt_shp.distance(nearest_pt)
            if length > threshold:
                length = threshold
            window[y, x] = length
            #window[x, y] = dist([x + minx, y + miny], nearest_point.coords[0])
        try:
            window = window / window.max()
        except Exception as e:
            print(e)
            break
        if not np.isfinite(dist_mask).all():
            break

        window = np.ones_like(window) - window
        mask[miny : maxy + 1, minx : maxx + 1] = window * window_binary
        dist_mask = np.concatenate([dist_mask, np.expand_dims(mask, axis=2)], axis=2)

        dist_mask = np.max(dist_mask, axis=2, keepdims=True)
    return dist_mask


def img_to_distance(lines, polys, shape, threshold=30):
    """
    Calculates a "probability" of a polygon center

    :param line: list of polygon centers
    :param polys: list of polygons
    :param shape: image shape
    :return: array with tree center probabilities
    """
    dist_mask = np.expand_dims(np.zeros(shape), axis=2)
    for polygon, line in tqdm(zip(polys, lines)):
        if len(polygon.bounds) == 0:
            continue
        minx, miny, maxx, maxy = [int(x) for x in polygon.bounds]
        # make sure we are in bounds
        minx = np.clip(minx, 0, shape[1])
        maxx = np.clip(maxx, 0, shape[1])
        miny = np.clip(miny, 0, shape[0])
        maxy = np.clip(maxy, 0, shape[0])

        mask = features.rasterize([polygon], shape).astype("float32")
        window = mask[miny : maxy + 1, minx : maxx + 1].copy()
        window_binary = window.copy()

        tree = np.nonzero(window)

        for y, x in zip(tree[0], tree[1]):
            pt_shp = Point([x + minx, y + miny])
            #nearest_point = line.interpolate(line.project(pt_shp))
            nearest_pt = nearest_points(pt_shp, line)[1]
            length = pt_shp.distance(nearest_pt)
            if length > threshold:
                length = threshold
            window[y, x] = 3 * length
            #window[x, y] = dist([x + minx, y + miny], nearest_point.coords[0])
        try:
            window = 90 - window
        except Exception as e:
            print(e)
            break
        if not np.isfinite(dist_mask).all():
            break

        #window = np.ones_like(window) - window
        mask[miny : maxy + 1, minx : maxx + 1] = window * window_binary
        dist_mask = np.concatenate([dist_mask, np.expand_dims(mask, axis=2)], axis=2)

        dist_mask = np.max(dist_mask, axis=2, keepdims=True)
    return dist_mask


def run_dist_map_generation(raster_path, row_poly_path, row_path, save_path, downscale, distance=False):
    lines_geo = gpd.read_file(row_path)
    lines_poly_geo = gpd.read_file(row_poly_path)


    dataset = rasterio.open(raster_path, "r")
    polys_pixels = polygons_to_image_view(lines_poly_geo, dataset, rect=None, downscale=downscale)
    lines_pixels = lines_to_image_view(lines_geo, dataset, rect=None, downscale=downscale)
    polys_pixels = gpd.GeoDataFrame(geometry=polys_pixels)
    lines_pixels = gpd.GeoDataFrame(geometry=lines_pixels)
    polys = []
    lines = []
    for index, polyg in tqdm(polys_pixels.iterrows()):
        lines_poly = get_polygons_in_window(lines_pixels, polyg['geometry'])
        if len(lines_poly) > 1:
            print("2 lines detected in one polygon")
            for num, line in lines_poly.iterrows():
                lines.append(line['geometry'])
                polys.append(polyg['geometry'])
        else:
            lines.append(lines_poly['geometry'].values[0])
            polys.append(polyg['geometry'])

    down_shape = (np.array(dataset.shape) / downscale).astype(np.int64)
    #thresh =
    if distance:
        dist_mask = img_to_distance(lines, polys, (down_shape[0], down_shape[1]), threshold=30)
        dist_mask = dist_mask.astype(np.uint8)
    else:
        dist_mask = img_to_proba(lines, polys, (down_shape[0], down_shape[1]), threshold=30)
        dist_mask = (dist_mask * 255).astype(np.uint8)
    dist_mask = cv2.resize(dist_mask[:, :, 0], (dataset.shape[1], dataset.shape[0]))
    save_dir = os.path.dirname(save_path)
    os.makedirs(save_dir, exist_ok=True)
    save_meta = dataset.meta.copy()
    save_meta.update({"count": 1,
                      "dtype": "uint8",
                      "nodata": 0})
    with rasterio.open(save_path, 'w', **save_meta) as dst:
        dst.write(dist_mask, 1)

    print('done')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='preprocess raster')

    parser.add_argument('--raster_path',
                        type=str,
                        required=True,
                        help="""path to binary mask""")
    parser.add_argument('--row_poly_path',
                        type=str,
                        required=True,
                        help="""path to binary mask""")
    parser.add_argument('--save_path',
                        type=str,
                        required=True,
                        help="""path to save polygons with angles""")
    parser.add_argument('--row_path',
                        type=str,
                        required=True,
                        help="""path to rows""")
    parser.add_argument('--downscale',
                        type=int,
                        required=False,
                        default=10,
                        help="""path to rows""")
    parser.add_argument('--window_size_meters',
                        default=60,
                        type=int,
                        help="""window size meters""")


    args = parser.parse_args()
    run_dist_map_generation(args.raster_path, args.row_poly_path, args.row_path, args.save_path, args.downscale)