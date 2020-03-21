import rasterio
import cv2
import numpy as np
import geopandas as gpd
from tqdm import tqdm
import os
import math

from shapely.ops import split as shapely_split
from shapely.geometry import Polygon, Point, MultiPoint, LineString
import shapely
import argparse
from utils import save_gpd_df, get_perpendiculars, get_azimuth_polyline, split_lines


POLYGONIZATION_THRESHOLD = 0.99


def polygonize(contours, meta, transform=True):
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





def get_azimuth_360(line):
    '''azimuth between 2 shapely points (interval 0 - 360)'''
    angle = np.arctan2(line.coords[1][0] - line.coords[0][0], line.coords[1][1] - line.coords[0][1])
    return np.degrees(angle)if angle>0 else np.degrees(angle) + 360


def split_polygon(polygon_to_split, list_of_lines):
    list_to_polygonize = list_of_lines.copy()
    list_to_polygonize.append(polygon_to_split.boundary) # collection of individual linestrings for splitting in a list and add the polygon lines to it.
    merged_lines = shapely.ops.linemerge(list_to_polygonize)
    border_lines = shapely.ops.unary_union(merged_lines)
    decomposition = shapely.ops.polygonize(border_lines)
    return list(decomposition)


def get_polygons(raster_mask_path, threshold=POLYGONIZATION_THRESHOLD):
    with rasterio.open(raster_mask_path, "r") as src:
        raster_array = src.read(1)
        meta = src.profile
        if meta.get('dtype') == 'uint8':
            threshold = 255 * threshold
        raster_array = (raster_array > threshold).astype(np.uint8)
        im2, contours, hierarchy = cv2.findContours(raster_array.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        polygons = polygonize(contours, meta, transform=True)
        polygon_df = gpd.GeoDataFrame(geometry=polygons)
        polygon_df.crs = meta.get('crs')
    return polygon_df, meta


def assign_angles(splitted_df, lines, angles):
    cntr = 0
    for num, row in splitted_df.iterrows():
        for line, angle in zip(lines, angles):
            pt_1 = line.interpolate(0.1, normalized=True)
            pt_2 = line.interpolate(0.9, normalized=True)
            new_line = LineString([pt_1, pt_2])
            if new_line.intersects(row['geometry']):
                splitted_df.loc[num, "angle"] = angle + 90
                cntr += 1
    return splitted_df


def run_data_preparation(raster_mask_path, rows_path, save_path):

    # get tree polygons
    polygon_df, meta = get_polygons(raster_mask_path)

    # read lines
    row_df = gpd.read_file(rows_path)
    complete_df = None
    # iterate and generate ground truth
    for num, row in tqdm(row_df.iterrows(), total=len(row_df)):
        cur_line = row['geometry']
        lines = split_lines(cur_line)
        angles = [get_azimuth_polyline(line) for line in lines]
        #angles_to_sign = [get_azimuth_360(line) for line in lines]
        perpendiculars = get_perpendiculars(lines, angles)

        inter_polys = polygon_df[polygon_df.intersects(cur_line)]
        splitted_list = []

        for poly in inter_polys['geometry'].values:
            splitted_polygons = split_polygon(poly, perpendiculars)
            #print(len(splitted_polygons))
            splitted_list.extend(splitted_polygons)

        splitted_df = gpd.GeoDataFrame(geometry=splitted_list)
        splitted_df = assign_angles(splitted_df, lines, angles)
        if not isinstance(complete_df, gpd.GeoDataFrame):
            complete_df = splitted_df.copy()
        else:
            complete_df = complete_df.append(splitted_df)
    save_gpd_df(save_path, complete_df, meta)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='preprocess raster')

    parser.add_argument('--raster_mask_path',
                        default=1,
                        type=str,
                        required=True,
                        help="""path to binary mask""")
    parser.add_argument('--rows_path',
                        default=1,
                        type=str,
                        required=True,
                        help="""path to rows geojson""")
    parser.add_argument('--save_path',
                        default=1,
                        type=str,
                        required=True,
                        help="""path to save polygons with angles""")

    args = parser.parse_args()
    run_data_preparation(args.raster_mask_path, args.rows_path, args.save_path)