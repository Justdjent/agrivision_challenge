import cv2

import os
import numpy as np
from scipy import ndimage as ndi
import geopandas as gpd
from skimage.morphology import watershed
import rasterio
from shapely.geometry import Polygon, mapping
from rasterio import features
from shapely.affinity import scale as shapely_scale
from tqdm import tqdm
from shapely.ops import cascaded_union
import argparse
from row_detection_research_code.utils import lines_to_image_view, mask_raster_with_polygon, save_gpd_df


def get_line_mask(rows_img, dataset, downscale=None):
    mid_line = features.rasterize(rows_img, out_shape=(int(dataset.height / downscale), int(dataset.width / downscale)))
    kernel = np.ones((10, 10))
    dilated_mid_line = cv2.dilate(mid_line, kernel)
    return dilated_mid_line


def get_watershed(image):
    mask_x = np.ones(image.shape) * 255
    # Now we want to separate the two objects in image
    # Generate the markers as local maxima of the distance to the background
    distance = ndi.distance_transform_edt(image)
    markers = ndi.label(image)[0]
    labels = watershed(-distance, markers, mask=mask_x)
    return labels


def watershed_to_polygons(watershed_result, dataset, downscale=None):
    polys_shapely = []
    for point in tqdm(np.unique(watershed_result)):
        tmp = watershed_result.copy()
        tmp[np.where(watershed_result != point)] = 0
        tmp = (tmp > 0).astype("uint8")
        _, contour, _ = cv2.findContours(tmp, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # contours.append(contour)
        c_points = [pt for pt in contour[0].squeeze()]
        try:
            polygon = shapely_scale(Polygon(c_points), downscale, downscale, origin=(0, 0, 0))
            polyg_geo = [np.array(tuple(pt) * dataset.meta['transform']) for pt in polygon.exterior.coords]
            polyg_geo = Polygon(polyg_geo)
            polys_shapely.append(polyg_geo)
        except:
            raise
    return polys_shapely


def run_polygons_preparation(raster_path, row_path, save_path, downscale, grove_bound=None):
    lines_geo = gpd.read_file(row_path)

    if grove_bound:
        gdf_grove = gpd.read_file(grove_bound)
        united_bound = cascaded_union(gdf_grove['geometry'].tolist())
        masked_path = raster_path.replace(".tif", "_masked.tif")
        if not os.path.exists(masked_path):
            with rasterio.open(raster_path, "r") as dataset:
                mask_raster_with_polygon(dataset, masked_path, [mapping(united_bound)], tile_size=5120, crop=True,
                                         pad=False)
        raster_path = masked_path
    dataset = rasterio.open(raster_path, "r")
    lines_img_coords = lines_to_image_view(lines_geo, dataset, downscale=downscale)
    line_mask = get_line_mask(lines_img_coords, dataset, downscale=downscale)
    watersheded_lines = get_watershed(line_mask)
    polygons = watershed_to_polygons(watersheded_lines, dataset, downscale=downscale)
    polygon_df = gpd.GeoDataFrame(geometry=polygons)
    # filtering polygons
    # for index, polyg in polygon_df.iterrows():
    #     lines_poly = get_polygons_in_window(lines_geo, polyg['geometry'])
    #     break
    save_gpd_df(save_path, polygon_df, dataset.meta)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='preprocess raster')

    parser.add_argument('--raster_path',
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
                        default=5,
                        help="""path to rows""")
    parser.add_argument('--grove_bound_path',
                        type=str,
                        required=False,
                        help="""path to polygons""")
    args = parser.parse_args()
    # run_mask_generaion(args.raster_path, args.polygons_path, args.window_size_meters, args.save_path, args.grove_bound_path)
    run_polygons_preparation(args.raster_path, args.row_path, args.save_path, args.downscale, args.grove_bound_path)