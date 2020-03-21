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
from utils import get_polygons_in_window


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





def dist(a, b):
    """
    Distance between two points
    """
    return np.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


def img_to_proba(centers, polys, shape):
    """
    Calculates a "probability" of a polygon center

    :param centers: list of polygon centers
    :param polys: list of polygons
    :param shape: image shape
    :return: array with tree center probabilities
    """
    dist_mask = np.expand_dims(np.zeros(shape), axis=2)
    for polygon, center in zip(polys, centers):
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
        offset_center = [center[1] - miny, center[0] - minx]

        tree = np.nonzero(window)
        for x, y in zip(tree[0], tree[1]):
            window[x, y] = dist([x, y], offset_center)
        try:
            window = window / window.max()
        except:
            continue
        if not np.isfinite(dist_mask).all():
            continue

        window = np.ones_like(window) - window
        mask[miny : maxy + 1, minx : maxx + 1] = window * window_binary
        dist_mask = np.concatenate([dist_mask, np.expand_dims(mask, axis=2)], axis=2)

        dist_mask = np.max(dist_mask, axis=2, keepdims=True)
    return dist_mask


def polygons_to_image_view(window_cut_polygons, rect, dataset):
    col_offset = rect[0]
    row_offset = rect[1]
    img_polyg_array = []
    to_img_mat = ~dataset.meta['transform']
    for item in window_cut_polygons.iterrows():
        if item[1]['geometry'].type != 'Polygon':
            # polyg_spati = np.array(cascaded_union(item[1]['geometry']).exterior.coords)
            continue
        else:
            polyg_spati = np.array(item[1]['geometry'].exterior.coords)
        polyg_img = [tuple(pt) * to_img_mat for pt in polyg_spati]
        polyg_img = np.subtract(polyg_img, (col_offset, row_offset))
        polyg_img = Polygon(polyg_img)
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


def run_mask_row_generation(raster_path, polygons_path, window_size_meters, save_path, row_path, grove_bound=None):
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
    grove_polygons = gpd.read_file(polygons_path)
    grove_rows = gpd.read_file(row_path)
    if grove_bound:
        gdf_grove = gpd.read_file(grove_bound)
        united_bound = cascaded_union(gdf_grove['geometry'].tolist())
        masked_path = raster_path.replace(".tif", ".masked")
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
            window_cut_rows = get_polygons_in_window(grove_rows, polygon_window)
            if len(window_cut_polygons.index) > 0:
                # logger.info("Original polygon is = %s", len(window_cut_polygons))
                # logger.info("Current rect = %s", rect)

                segments = polygons_to_image_view(window_cut_polygons, (window[2], window[0]), dataset)
                row_segments = points_to_image_view(window_cut_rows, (window[2], window[0]), dataset)
                img_to_proba(row_segments, segments, (res.shape[1], res.shape[2]))
                mask = np.zeros((res.shape[1], res.shape[2]), dtype=np.uint8)
                if res.shape[1] != res.shape[2]:
                    continue
                try:
                    mid_line = features.rasterize(row_segments, out_shape=(res.shape[1], res.shape[2]))
                    kernel = np.ones((60, 60))
                    dilated_mid_line = cv2.dilate(mid_line, kernel)
                    blurred = dilated_mid_line * 255
                    #blurred = cv2.GaussianBlur(dilated_mid_line * 255, (29, 29), 0)
                except:
                    continue
                # points_to_make_gaussian = np.stack(np.where(mid_line > 0)).T
                # gaus_mask = generate_mask((res.shape[1], res.shape[2]), points_to_make_gaussian)
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

                # save edge
                save_name = os.path.join(edges_path, img_name)
                result_edge = cv2.imwrite(save_name, blurred)

                # save img
                save_name = os.path.join(images_path, img_name)
                result = cv2.imwrite(save_name, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
                if not result or (not result_mask) or not result_edge:
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
                        required=False,
                        help="""path to distance mask""")
    parser.add_argument('--window_size_meters',
                        default=60,
                        type=int,
                        help="""window size meters""")
    parser.add_argument('--polygons_path',
                        type=str,
                        required=True,
                        help="""window size meters""")
    parser.add_argument('--save_path',
                        type=str,
                        required=True,
                        help="""path to save polygons with angles""")
    parser.add_argument('--grove_bound_path',
                        type=str,
                        required=False,
                        help="""path to polygons""")
    parser.add_argument('--row_path',
                        type=str,
                        required=False,
                        help="""path to rows""")

    args = parser.parse_args()
    # run_mask_generaion(args.raster_path, args.polygons_path, args.window_size_meters, args.save_path, args.grove_bound_path)
    run_mask_row_generation(args.raster_path, args.polygons_path, args.window_size_meters, args.save_path, args.row_path,
                       args.grove_bound_path)