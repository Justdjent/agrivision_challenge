import os
import numpy as np
import matplotlib.pyplot as plt
import rasterio
from skimage.morphology import skeletonize
from skimage import data
import sknw
import cv2
from scipy import ndimage as ndi
from skimage.morphology import watershed
from tqdm import tqdm
from scipy.spatial.distance import cdist
from shapely.geometry import LineString
import argparse
import geopandas as gpd

from row_detection_service.utils import save_gpd_df, read_json


def sort_points_recur(graph_ps_x):
    input_shape = graph_ps_x.shape
    graph_index = 0
    sorted_graph_pts = [graph_ps_x[graph_index]]

    for i in range(len(graph_ps_x)):
        # point =
        graph_ps_x = np.delete(graph_ps_x, graph_index, axis=0)
        dists = cdist(np.expand_dims(sorted_graph_pts[i], axis=0), graph_ps_x)
        # print(dists.shape)
        if len(dists[0]) > 0:
            graph_index = np.argmin(dists)
            graph_min_val = np.min(dists)
            sorted_graph_pts.append(graph_ps_x[graph_index])
    sorted_graph_pts = np.array(sorted_graph_pts)
    # print(sorted_graph_pts.shape, input_shape)
    assert input_shape == sorted_graph_pts.shape
    return sorted_graph_pts


def my_watershed(energy, markers, mask):
    """
    Watershed wrapper.

    :param energy: Watershed "energy" mask
    :param markers: "kernels" for the watershed algorithm
    :param mask: Binary mask separating a single class
    :return: Image with every instance labeled with a separate number.
    """
    markers = ndi.label(markers, output=np.uint32)[0]
    labels = watershed(255-energy, markers, watershed_line=True, mask=mask)
    return labels


def make_lines(labels):
    graphs = []
    for num, label in tqdm(enumerate(range(1, len(np.unique(labels)) + 1)), total=len(np.unique(labels))):
        label_mask = labels == label
        ske = skeletonize(label_mask).astype(np.uint16)
        graph = sknw.build_sknw(ske)
        graphs.append(graph)

    #         if num == 100:
    #             break
    return graphs


def graph_to_shp_lines(graph_lines, meta, each=50, downscale=0.2):
    lines = []
    for graph in tqdm(graph_lines, len(lines)):

        node, nodes = graph.node, graph.nodes()
        graph_ps = []
        for (s, e) in graph.edges():
            ps = []
            weight = graph[s][e]['weight']
            if weight < 10:
                continue
            ps.append(nodes[s]['o'])
            pts = graph[s][e]['pts']
            ps.extend(pts)
            ps.append(nodes[e]['o'])
            ps = np.array(ps)
            # print(len(ps))
            if len(ps) > each:
                ps = ps[::each]
            graph_ps.extend(ps)
        graph_ps = np.array(graph_ps)
        if len(graph_ps) < 2:
            continue

        # sort points in one line
        y_axis_var = np.var(graph_ps[:, 1])
        x_axis_var = np.var(graph_ps[:, 0])
        if y_axis_var > x_axis_var:
            ind = np.lexsort((graph_ps[:, 0], graph_ps[:, 1]))
        else:
            ind = np.lexsort((graph_ps[:, 1], graph_ps[:, 0]))
        all_coords = graph_ps[ind]

        # sort points recursively
        graph_ps = sort_points_recur(all_coords)

        # transform points to world coordinates
        graph_ps = graph_ps * 1 / downscale
        ps_yx = graph_ps.copy()
        ps_yx[:, 1] = graph_ps[:, 0]
        ps_yx[:, 0] = graph_ps[:, 1]
        converted = [meta['transform'] * pt for pt in ps_yx]

        line = LineString(converted)
        lines.append(line)
    return lines


def postprocess_single_channel_pred(raster_path, downscale, save_labels=True):
    connection_meters = 1
    with rasterio.open(raster_path, "r") as src:
        image = src.read(1)
        if src.meta['dtype'] != 'uint8':
            image = (image * 255).astype(np.uint8)
        #image = cv2.resize(image, None, fx=downscale, fy=downscale)
        meta = src.meta
        kernel_size = int(connection_meters / meta['transform'][0])
        if (kernel_size % 2) == 0:
            kernel_size += 1
        kernel_size = (kernel_size, kernel_size)
        kernel = np.ones(kernel_size)
        blur = cv2.GaussianBlur(image, kernel_size, 0)
        lines = cv2.dilate(blur.astype(np.uint8), kernel_size)
        lines = cv2.morphologyEx(lines, cv2.MORPH_CLOSE, kernel)
        tree_line_mask = cv2.resize(lines, None, fx=downscale, fy=downscale, interpolation=cv2.INTER_NEAREST)

    mask = tree_line_mask > 50
    markers = tree_line_mask > 50
    energy = tree_line_mask.copy()
    energy[energy < 50] = 0
    labels = my_watershed(energy, markers, mask)
    if save_labels:
        save_meta = meta.copy()
        save_meta.update({'dtype': 'uint16',
                          'count': '1'})
        save_label_path = raster_path.replace(".tif", "_labels.tif")
        with rasterio.open(save_label_path, "w", **save_meta) as dst:
            dst.write(labels.astype(np.uint16), indexes=1)
    return labels, meta


def prepare_lines_masks(image, downscale=0.2, kernel_size=(13, 13)):
    lines = (image[1] > 150) | (image[0] > 150)
    kernel = np.ones(kernel_size)
    blur = cv2.GaussianBlur(lines.astype(np.uint8),kernel_size,0)
    lines = cv2.dilate(blur.astype(np.uint8), kernel_size)
    lines = cv2.morphologyEx(lines, cv2.MORPH_CLOSE, kernel)
    tree_line_mask = cv2.resize(lines, None, fx=downscale, fy=downscale, interpolation=cv2.INTER_NEAREST)
    return tree_line_mask


def postprocess_double_channel_pred(raster_path, downscale, connection_meters=1, save_labels=True):
    with rasterio.open(raster_path, "r") as src:
        image = src.read()
        #
        meta = src.meta
        # get size of connection
        kernel_size = int(connection_meters / meta['transform'][0])
        if (kernel_size % 2) == 0:
            kernel_size += 1

        lines = prepare_lines_masks(image, downscale=downscale, kernel_size=(kernel_size, kernel_size))

    lines = (lines * 255).astype(np.uint8)
    mask = lines > 220
    markers = lines > 220
    energy = lines.copy()
    # energy[energy < 150] = 0
    labels = my_watershed(energy, markers, mask)
    if save_labels:
        save_meta = meta.copy()
        save_meta.update({'dtype': 'uint16',
                          'count': '1'})
        save_label_path = raster_path.replace(".tif", "_labels.tif")
        with rasterio.open(save_label_path, "w", **save_meta) as dst:
            dst.write(labels.astype(np.uint16), indexes=1)
    return labels, meta


def run_lines_generation(raster_path, save_path, downscale=0.2, network_channels=1):
    if network_channels == 1:
        labels, meta = postprocess_single_channel_pred(raster_path, downscale)
    elif network_channels == 2:
        labels, meta = postprocess_double_channel_pred(raster_path, downscale)
    else:
        print('stop')
        return None


    # creating lines
    graph_lines = make_lines(labels)

    # transforming lines to shapely objects
    lines = graph_to_shp_lines(graph_lines, meta, each=5, downscale=downscale)
    line_gdf = gpd.GeoDataFrame(geometry=lines)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    save_gpd_df(save_path, line_gdf, meta=meta)


def run_generation_from_local(args):
    d = read_json(args.inp_list)
    for grove, inp in d.items():
        print(grove)
        raster_list = read_json(inp)
        predicted_folder = 'dist_net_resnet_1349_256_aug'
        # segmentator = SegmentatorNN(inp, thresh, r_type)
        # path = raster_list.get('line_gap_1269', None)
        red_path = raster_list.get('rgb', None)
        path = os.path.join(os.path.dirname(red_path), predicted_folder, 'project_transparent_mosaic_rgb_bin.tif')
        if not os.path.exists(path):
            raster_name = os.path.basename(red_path)
            raster_name = raster_name.split("_")[-1]
            path = os.path.join(os.path.dirname(red_path), predicted_folder, 'rgb_bin_{}'.format(raster_name))
        if not path:
            print("There is no segmentation for {}".format(grove))
            break
        save_path = path.replace(".tif", "_{}.geojson".format(args.postfix))
        run_lines_generation(path, save_path, args.downscale, args.network_channels)


def run_generation_from_jobs(args):
    groves = os.listdir(args.raster_folder)
    for grove in groves:
        grove_folder = os.path.join(args.raster_folder, grove)
        if not os.path.isdir(grove_folder):
            continue
        print(grove)
        path = os.path.join(grove_folder, 'distance_mask.tif')
        if not os.path.exists(path):
            print("There is no segmentation for {}".format(grove))
            continue
        if not path:
            print("There is no segmentation for {}".format(grove))
            break
        save_path = path.replace(".tif", "_{}.geojson".format(args.postfix))
        run_lines_generation(path, save_path, args.downscale, args.network_channels)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='preprocess raster')

    parser.add_argument('--raster_folder',
                        type=str,
                        required=False,
                        help="""path to edge detection output""")
    parser.add_argument('--inp_list',
                        type=str,
                        required=False,
                        default="/mnt/storage_4tb/ymi/geo_data/angle_net_data/rio_grande/inp_list_rio",
                        help="""path to edge detection output""")
    parser.add_argument('--downscale',
                        type=float,
                        required=False,
                        default=0.2,
                        help="""downscale factor""")
    parser.add_argument('--network_channels',
                        type=int,
                        required=False,
                        default=1,
                        help="""downscale factor""")
    parser.add_argument('--save_path',
                        type=str,
                        required=False,
                        help="""path to save geojson""")
    parser.add_argument('--postfix',
                        type=str,
                        required=False,
                        help="""path to save geojson""")

    args = parser.parse_args()
    if args.raster_folder:
        print('running generation from job result')
        run_generation_from_jobs(args)
    else:
        run_generation_from_local(args)