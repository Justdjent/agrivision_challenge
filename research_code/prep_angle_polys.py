import os
import argparse


from prepare_masks_angles import run_mask_row_generaion
from prepare_angle import run_data_preparation
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='preprocess raster')

    parser.add_argument('--raster_path',
                        type=str,
                        required=True,
                        help="""path to binary mask""")
    parser.add_argument('--bin_raster_path',
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
    parser.add_argument('--row_path',
                        type=str,
                        required=False,
                        help="""path to rows""")

    args = parser.parse_args()
    # prepare angle polys
    poly_save_dir = os.path.dirname(args.row_path)
    poly_save_path = os.path.join(poly_save_dir, "polygons_angle_180.geojson")
    if not os.path.exists(poly_save_path):
        run_data_preparation(args.bin_raster_path, args.row_path, poly_save_path)
    # run_mask_generaion(args.raster_path, args.polygons_path, args.window_size_meters, args.save_path, args.grove_bound_path)
    run_mask_row_generaion(args.raster_path,
                           poly_save_path,
                           args.window_size_meters,
                           args.save_path,
                           args.row_path,
                           args.grove_bound_path)
