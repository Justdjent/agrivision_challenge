from row_detection_research_code.generate_distance_map import run_dist_map_generation
from row_detection_research_code.prepare_masks_dists import run_dist_mask_preparation
from row_detection_research_code.prepare_line_polys import run_polygons_preparation
from row_detection_research_code.utils import create_raster_list
from row_detection_service.reader import Reader
import argparse
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='preprocess raster')

    parser.add_argument('--raster_path',
                        type=str,
                        required=True,
                        help="""path to raster""")
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
                        default=None,
                        help="""path to polygons""")
    parser.add_argument('--row_path',
                        type=str,
                        required=True,
                        help="""path to rows""")
    parser.add_argument('--downscale',
                        type=int,
                        required=False,
                        default=5,
                        help="""path to rows""")

    r_type = 'rgb'
    args = parser.parse_args()
    # run_mask_generaion(args.raster_path, args.polygons_path, args.window_size_meters, args.save_path, args.grove_bound_path)
    poly_save_dir = os.path.dirname(args.row_path)
    if os.path.isdir(args.raster_path):
        raster_list = create_raster_list(args.raster_path)
        print(raster_list)
        reader = Reader(raster_list)
        if r_type not in raster_list.keys():
            reader.create_raster(r_type, True)
        raster_save_dir = args.raster_path
        input_raster_path = reader.raster_list[r_type]['path']
    else:
        raster_save_dir = os.path.dirname(args.raster_path)
        input_raster_path = args.raster_path
    polygons_path = os.path.join(poly_save_dir, "row_poly.geojson")
    # dist_map_path = os.path.join(raster_save_dir, "dist_mat.tif")
    # masked_raster_path = os.path.join(raster_save_dir, "rgb_masked.tif")
    #polygons_path = args.row_path.replace(".geojson", "_poly.geojson")
    dist_map_path = input_raster_path.replace(".tif", "_dist_mat_true_dist.tif")
    masked_raster_path = input_raster_path.replace(".tif", "_masked.tif")
    if not os.path.exists(polygons_path):
        run_polygons_preparation(input_raster_path, args.row_path, polygons_path, args.downscale, grove_bound=args.grove_bound_path)
    run_dist_map_generation(masked_raster_path, polygons_path, args.row_path, dist_map_path, args.downscale)
    #raster_path, row_poly_path, row_path, save_path, downscale
    run_dist_mask_preparation(masked_raster_path, dist_map_path, args.window_size_meters, args.save_path, grove_bound=args.grove_bound_path)