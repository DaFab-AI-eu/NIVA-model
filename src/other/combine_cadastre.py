import argparse
import os
from glob import glob
import geopandas as gpd
import pandas as pd


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--input_folder", type=str, required=True)
    parser.add_argument("-o", "--output", type=str, required=True)
    parser.add_argument("--min_field_num_tile", type=int, required=False, default=200)

    args = parser.parse_args()
    input_folder = args.input_folder
    output_path = args.output
    min_field_num_tile = args.min_field_num_tile

    file_paths = glob(os.path.join(input_folder, "*.gpkg"))
    final_list = []
    for file_path in file_paths:
        df = gpd.read_file(file_path)
        if len(df) >= min_field_num_tile:
            final_list.append(df)
        else:
            print(
                f"Number of cadastre crop fields for the tile_grid = {len(df)} < {min_field_num_tile}"
                f"\n. not enough Ground Truth data for the region {file_path}. ")

    final_df = pd.concat(final_list)
    final_df.reset_index(drop=True, inplace=True)
    final_df.to_file(output_path)
    print(f"Final records number {len(final_df)}")