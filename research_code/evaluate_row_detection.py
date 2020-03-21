import geopandas as gpd
import cv2
import rasterio
import argparse
import numpy as np
from tqdm import tqdm
from shapely.geometry import Polygon
import seetree.preprocessPointData as preprocessPointData

from row_detection_research_code.utils import (get_polygons_in_window,
                                               save_gpd_df,
                                               get_perpendiculars_each_n_meters,
                                               split_lines,
                                               get_azimuth_polyline)


def run_metric_each_n_meters(rows_path, predicted_path, save_path, coverage_percentage=0.85):

    predicted_df = gpd.read_file(predicted_path)
    predicted_df = convert_crs(predicted_df)
    gt_rows = gpd.read_file(rows_path)
    gt_rows = convert_crs(gt_rows)

    # iterate and generate ground truth
    pbar = tqdm(total=len(gt_rows))

    for num, row in gt_rows.iterrows():
        line_score = []
        cur_line = row['geometry']
        lines = split_lines(cur_line)
        try:
            angles = [get_azimuth_polyline(line) for line in lines]
        except:
            print("angle can't be calculated")
            continue
        # angles_to_sign = [get_azimuth_360(line) for line in lines]
        perpendiculars = get_perpendiculars_each_n_meters(lines, angles, step=1)
        row_poly = row['geometry'].buffer(2)
        window_cut_rows = get_polygons_in_window(predicted_df, row_poly)
        for perpendicular in perpendiculars:
            inter_lines = predicted_df[predicted_df.intersects(perpendicular)]
            error_m = -1
            if len(inter_lines) > 0:
                for inter_line_num, inter_line in inter_lines.iterrows():
                    predicted_point = inter_line['geometry'].intersection(perpendicular)
                    gt_point = cur_line.intersection(perpendicular)
                    error_m = predicted_point.distance(gt_point)
                line_score.append(error_m)
            else:
                line_score.append(error_m)
        line_score = np.array(line_score)
        coverage = np.sum(line_score != -1) / len(perpendiculars)
        mean_error = np.mean(line_score[line_score != -1])
        num_lines = len(window_cut_rows)
        if coverage > coverage_percentage:
            detected = 1
        else:
            detected = 0
        #inter_polys = predicted_df[predicted_df.intersects(cur_line)]
        gt_rows.loc[num, 'coverage'] = coverage
        gt_rows.loc[num, 'mean_error_meters'] = mean_error
        gt_rows.loc[num, 'num_lines'] = num_lines
        gt_rows.loc[num, 'detected'] = detected
        percentage_detected = gt_rows['detected'].sum() / num
        mean_error = gt_rows['mean_error_meters'].mean()
        mean_coverage = gt_rows['coverage'].mean()
        pbar.set_postfix(percentage_detected="%.4f" % percentage_detected,
                         mean_error="%.4f" % mean_error,
                         mean_coverage="%.4f" % mean_coverage,
                         cur_coverage="%.4f" % coverage,
                         mean_lines="%.4f" % gt_rows['num_lines'].mean())
        pbar.update(1)

    print("Percentage detected: {}".format(percentage_detected))
    print("Average number lines per line detected: {}".format(gt_rows['num_lines'].mean()))
    print("Detected/All: {}/{}".format(gt_rows['detected'].sum(), len(gt_rows)))
    save_gpd_df(save_path, gt_rows)


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


def convert_crs(row_df, def_crs='epsg:4326'):
    if row_df.crs.get('init') == def_crs:
        row_df = preprocessPointData.df_wgs_to_utm(row_df)
    return row_df


def run_evaluation(gt_row_path, detected_lines_path, save_path, misalignment_percentage=0.15):

    gt_rows = gpd.read_file(gt_row_path)
    gt_rows = convert_crs(gt_rows)
    grove_rows = gpd.read_file(detected_lines_path)
    grove_rows = convert_crs(grove_rows)

    for num, row in tqdm(gt_rows.iterrows()):
        row_poly = row['geometry'].buffer(2)
        window_cut_rows = get_polygons_in_window(grove_rows, row_poly)
        num_lines = len(window_cut_rows)
        gt_length = row['geometry'].length
        detected_length = 0
        for num_line, line in window_cut_rows.iterrows():
            detected_length += line['geometry'].length
        line_length_ratio = (gt_length - detected_length) / gt_length
        #print(line_length_ratio)
        if abs(line_length_ratio) < misalignment_percentage:
            detected = 1
        else:
            detected = 0
        gt_rows.loc[num, 'gt_length'] = gt_length
        gt_rows.loc[num, 'detected_length'] = detected_length
        gt_rows.loc[num, 'line_length_ratio'] = line_length_ratio
        gt_rows.loc[num, 'num_lines'] = num_lines
        gt_rows.loc[num, 'detected'] = detected
    percentage_detected = gt_rows['detected'].sum() / len(gt_rows)
    print("Percentage detected: {}".format(percentage_detected))
    print("Average number liens per line detected: {}".format(gt_rows['num_lines'].mean()))
    print("Detected/All: {}/{}".format(gt_rows['detected'].sum(), len(gt_rows)))
    save_gpd_df(save_path, gt_rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='preprocess raster')

    parser.add_argument('--gt_row_path',
                        type=str,
                        required=True,
                        help="""path to edge detection output""")
    parser.add_argument('--detected_lines_path',
                        type=str,
                        required=True,
                        help="""downscale factor""")
    parser.add_argument('--save_path',
                        type=str,
                        required=True,
                        help="""path to save geojson""")

    args = parser.parse_args()
    # run_evaluation(args.gt_row_path, args.detected_lines_path, args.save_path)

    run_metric_each_n_meters(args.gt_row_path, args.detected_lines_path, args.save_path)