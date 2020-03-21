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
import shapely
import argparse
# from random_transform_mask import tiles_with_overlap_shape
from rasterio.plot import reshape_as_raster, reshape_as_image
from rasterio import features
import shutil
from row_detection_research_code.utils import polygons_to_image_view, points_to_image_view, mask_raster_with_polygon
POLYGONIZATION_THRESHOLD = 0.99
IMAGES_FOLDER = "images"
EDGES_FOLDER = "edges"
BIN_MASK_FOLDER = "masks"

def tiles_with_overlap_shape(w_img, h_img, window_size, overlap):
    sp = []
    pnt_step = int(window_size * overlap)
    step_h = h_img // pnt_step
    step_w = w_img // pnt_step
    pointerh_min = 0
    for h in range(step_h + 1):
        if h != 0:
            pointerh_min += pnt_step
        pointerh = min(pointerh_min, h_img)
        pointerh_max = min(pointerh_min + window_size, h_img)
        pointerw_min = 0
        if pointerh == pointerh_max:
            continue
        for w in range(step_w + 1):
            if w != 0:
                pointerw_min += pnt_step
            pointerw = min(pointerw_min, w_img)
            pointerw_max = min(pointerw_min + window_size, w_img)
            if pointerw == pointerw_max:
                continue
            sp.append([pointerw, pointerw_max, pointerh, pointerh_max])
    return sp


def window_to_geo(bottom, left, width, height):
    ret = Polygon([(bottom, left), (bottom, left + width), (bottom + height, left + width), (bottom + height, left)])
    return ret


def get_polygons_in_window(dataframe, window):
    filter = dataframe.geometry.apply(lambda p: window.intersects(p))
    filter = gpd.GeoDataFrame(filter)
    filter = filter.loc[(filter['geometry'] == True)]
    merged_DB = dataframe.join(filter.drop(columns=['geometry']), how="inner")
    return merged_DB


def generate_mask(size, points):
    mask = np.zeros(size)
    for num, kp in enumerate(points):
        try:
            gen_mask = makeGaussian(size, fwhm=20, center=kp)
            mask = np.dstack([mask, gen_mask])
            mask = np.amax(mask, axis=2)
        except:
            mask = np.zeros(size)
    return mask


def makeGaussian(size, fwhm = 3, center=None):
    """ Make a square gaussian kernel.

    size is the length of a side of the square
    fwhm is full-width-half-maximum, which
    can be thought of as an effective radius.
    """

    x = np.arange(0, size[0], 1, float)
    y = np.arange(0, size[1], 1, float)
    y = y[:,np.newaxis]

    if center is None:
        x0 = y0 = size // 2
    else:
        x0 = center[0]
        y0 = center[1]

    return np.exp(-4*np.log(2) * ((x-x0)**2 + (y-y0)**2) / fwhm**2)


def run_mask_generaion(raster_path, polygons_path, window_size_meters, save_path, grove_bound=None):
    images_path = os.path.join(save_path, IMAGES_FOLDER)
    angles_path = os.path.join(save_path, EDGES_FOLDER)
    masks_path = os.path.join(save_path, BIN_MASK_FOLDER)
    os.makedirs(images_path, exist_ok=True)
    os.makedirs(angles_path, exist_ok=True)
    os.makedirs(masks_path, exist_ok=True)
    grove_polygons = gpd.read_file(polygons_path)
    if grove_bound:
        gdf_grove = gpd.read_file(grove_bound)
        united_bound = cascaded_union(gdf_grove['geometry'].tolist())
        masked_path = raster_path.replace(".tif", "_masked.tif")
        if not os.path.exists(masked_path):
            with rasterio.open(raster_path, "r") as dataset:
                mask_raster_with_polygon(dataset, masked_path, [mapping(united_bound)], tile_size=5120, crop=True, pad=False)
        raster_path = masked_path
    with rasterio.open(raster_path, "r") as dataset:
        xres = dataset.meta.get('transform')[0]
        window_size_pixels = np.ceil(window_size_meters / xres)
        window_config = tiles_with_overlap_shape(dataset.meta['height'], dataset.meta['width'],
                                                 window_size_pixels, 1)
        window_config = np.array(window_config)
        poly_list = []
        windows_coordinates = [pt for pt in window_config]
        progress_bar = tqdm()
        for num, window in enumerate(windows_coordinates):
            progress_bar.set_postfix(Row='{}'.format(int(window[0])), Col='{}'.format(int(window[2])))
            img_window = ((window[0], window[1]), (window[2], window[3]))
            res = dataset.read(window=img_window)

            ys, xs = (window[2], window[1]) * dataset.meta['transform']
            polygon_window = window_to_geo(ys, xs, res.shape[1] * xres, res.shape[2] * xres)
            poly_list.append(polygon_window)
            window_cut_polygons = get_polygons_in_window(grove_polygons, polygon_window)
            if len(window_cut_polygons.index) > 0:
                # logger.info("Original polygon is = %s", len(window_cut_polygons))
                # logger.info("Current rect = %s", rect)

                segments = polygons_to_image_view(window_cut_polygons, (window[2], window[0]), dataset)
                mask = np.zeros((res.shape[1], res.shape[2]), dtype=np.uint8)

                for segment, angle in zip(segments, window_cut_polygons['angle'].values.astype(np.uint8)):
                    seg_mask = features.rasterize([segment], out_shape=(res.shape[1], res.shape[2]))
                    seg_mask = (seg_mask * angle).astype(np.uint8)
                    stack_mask = np.dstack([mask, seg_mask])
                    mask = np.amax(stack_mask, axis=2)
                # mask = features.rasterize(segments, out_shape=(res.shape[1], res.shape[2]))
                # mask = (mask * 255).astype(np.uint8)
                img = reshape_as_image(res)
                img_name = "{}_{}_{}_{}_{}.png".format(num, int(window[0]), int(window[1]), int(window[2]), int(window[3]))
                # save mask
                save_name = os.path.join(masks_path, img_name)
                result_mask = cv2.imwrite(save_name, mask)

                # save img
                save_name = os.path.join(images_path, img_name)
                result = cv2.imwrite(save_name, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
                if not result or not result_mask:
                    raise Exception("save error")
            progress_bar.update(1)


def run_dist_mask_preparation(raster_path, dist_mask_path, window_size_meters, save_path, grove_bound=None):
    images_path = os.path.join(save_path, IMAGES_FOLDER)
    edges_path = os.path.join(save_path, EDGES_FOLDER)
    masks_path = os.path.join(save_path, BIN_MASK_FOLDER)
    if os.path.exists(images_path):
        shutil.rmtree(images_path)
    if os.path.exists(edges_path):
        shutil.rmtree(edges_path)
    if os.path.exists(masks_path):
        shutil.rmtree(masks_path)
    os.makedirs(images_path, exist_ok=True)
    os.makedirs(edges_path, exist_ok=True)
    os.makedirs(masks_path, exist_ok=True)
    if grove_bound:
        gdf_grove = gpd.read_file(grove_bound)
        united_bound = cascaded_union(gdf_grove['geometry'].tolist())
        masked_path = raster_path.replace(".tif", "_masked.tif")
        if not os.path.exists(masked_path):
            with rasterio.open(raster_path, "r") as dataset:
                mask_raster_with_polygon(dataset, masked_path, [mapping(united_bound)], tile_size=5120, crop=True, pad=False)
        raster_path = masked_path
    with rasterio.open(raster_path, "r") as dataset, rasterio.open(dist_mask_path, "r") as dist_dataset:
        xres = dataset.meta.get('transform')[0]
        window_size_pixels = np.ceil(window_size_meters / xres)
        window_config = tiles_with_overlap_shape(dataset.meta['height'], dataset.meta['width'],
                                                 window_size_pixels, 1)
        window_config = np.array(window_config)
        poly_list = []
        windows_coordinates = [pt for pt in window_config]
        progress_bar = tqdm()
        for num, window in enumerate(windows_coordinates):
            progress_bar.set_postfix(Row='{}'.format(int(window[0])), Col='{}'.format(int(window[2])))
            img_window = ((window[0], window[1]), (window[2], window[3]))
            res = dataset.read(window=img_window)
            res_dist = dist_dataset.read(window=img_window)
            if res.shape[1] != res.shape[2]:
                continue
            ys, xs = (window[2], window[1]) * dataset.meta['transform']
            polygon_window = window_to_geo(ys, xs, res.shape[1] * xres, res.shape[2] * xres)
            poly_list.append(polygon_window)
            # mask = features.rasterize(segments, out_shape=(res.shape[1], res.shape[2]))
            # mask = (mask * 255).astype(np.uint8)
            img = reshape_as_image(res)
            img_dist = reshape_as_image(res_dist)
            img_name = "{}_{}_{}_{}_{}.png".format(num, int(window[0]), int(window[1]), int(window[2]), int(window[3]))
            # save mask
            # save_name = os.path.join(masks_path, img_name)
            # result_mask = cv2.imwrite(save_name, mask)

            # save edge
            save_name = os.path.join(edges_path, img_name)
            result_edge = cv2.imwrite(save_name, img_dist)

            # save img
            save_name = os.path.join(images_path, img_name)
            result = cv2.imwrite(save_name, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            if not result or not result_edge:
                raise Exception("save error")
            progress_bar.update(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='preprocess raster')

    parser.add_argument('--raster_path',
                        type=str,
                        required=True,
                        help="""path to binary mask""")
    parser.add_argument('--dist_mask_path',
                        type=str,
                        required=True,
                        help="""path to binary mask""")
    parser.add_argument('--window_size_meters',
                        default=60,
                        type=int,
                        help="""window size meters""")
    parser.add_argument('--save_path',
                        type=str,
                        required=True,
                        help="""path to save polygons with angles""")
    parser.add_argument('--grove_bound_path',
                        type=str,
                        required=False,
                        help="""path to polygons""")


    args = parser.parse_args()
    # run_mask_generaion(args.raster_path, args.polygons_path, args.window_size_meters, args.save_path, args.grove_bound_path)
    run_dist_mask_preparation(args.raster_path, args.dist_mask_path, args.window_size_meters, args.save_path, args.grove_bound_path)