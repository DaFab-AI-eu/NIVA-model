import argparse
import os
import geopandas as gpd


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", type=str, required=True)
    parser.add_argument("-o", "--output", type=str, required=True)

    args = parser.parse_args()
    input_path = args.input
    output_path = args.output

    df = gpd.read_file(input_path)
    df.to_file(output_path)
    print(f"Final crs {df.crs}")