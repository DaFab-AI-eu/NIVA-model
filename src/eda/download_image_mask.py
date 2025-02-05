import argparse
import os
import asyncio
import pandas as pd
import sys

# Add the src directory to the path
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(src_path)


from training.download_data import download_images

RETRY_LIMIT = 5

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--country", type=str, required=True,
                        default="NL")
    parser.add_argument("-s", "--sentinel_dir", type=str, required=True,
                        default="ai4boundaries_dataset")
    parser.add_argument("-i", "--path_file", type=str, required=True,
                        default="NL.csv",
                        help="Input file containing images/masks urls to download")
    args = parser.parse_args()

    country = args.country
    SENTINEL2_DIR = args.sentinel_dir

    folder_save = os.path.join(SENTINEL2_DIR, country)
    path_file = f"{SENTINEL2_DIR}/stats/{country}_sm_num_fields.csv"

    fold_data = pd.read_csv(path_file)

    os.makedirs(folder_save, exist_ok=True)
    os.makedirs(os.path.join(folder_save, "masks"), exist_ok=True)
    os.makedirs(os.path.join(folder_save, "images"), exist_ok=True)

    asyncio.run(download_images(fold_data, folder_save))