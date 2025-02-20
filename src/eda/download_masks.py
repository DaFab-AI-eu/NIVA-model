import argparse
import os
import asyncio
import aiohttp
import pandas as pd
import sys

# Add the src directory to the path
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(src_path)

# Import logger
from niva_utils.logger import get_logger  # noqa: E402
LOGGER = get_logger(__name__)

# Import config
from niva_utils.config_loader import load_config  # noqa: E402
CONFIG = load_config()

from training.download_data import help_func
from training.download_data import download_images

RETRY_LIMIT = 5
async def download_mask(fold_data, folder_save):

    async with aiohttp.ClientSession() as session:
        num_files = len(fold_data)
        # Download masks
        # sentinel2_images_file_url
        LOGGER.info("Downloading sentinel2 masks...")
        fail_fns_mask = await help_func(
            session, fold_data["sentinel2_masks_file_url"], path=os.path.join(
                folder_save, "masks")
        )
        download_count_masks = num_files - len(fail_fns_mask)
        LOGGER.info(f"Downloaded {download_count_masks} files to masks folder\n"
                    f"failed files: {fail_fns_mask}")
        for i in range(RETRY_LIMIT):
            if len(fail_fns_mask) == 0:
                break
            LOGGER.info(f"Retrying failed downloads, attempt {i + 1}")
            fold_data_retry = fold_data[fold_data["sentinel2_masks_file_url"].isin(
                fail_fns_mask)]
            fail_fns_mask = await help_func(
                session,
                fold_data_retry,
                path=os.path.join(folder_save, "masks")
            )

def main():
    # input params
    split_filepath = "ai4boundaries_ftp_urls_sentinel2_split.csv"

    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--country", type=str, required=True,
                        default="NL")
    parser.add_argument("-s", "--sentinel_dir", type=str, required=False,
                        default="ai4boundaries_dataset")
    parser.add_argument("-i", "--split_filepath", type=str, required=False,
                        help="Path to ai4boundaries_ftp_urls_sentinel2_split.csv",
                        default=None)
    parser.add_argument("-f", "--flag_masks_only", action='store_true', default=False)
    args = parser.parse_args()

    country = args.country
    SENTINEL2_DIR = args.sentinel_dir
    split_filepath = os.path.join(SENTINEL2_DIR, split_filepath) if not args.split_filepath else args.split_filepath
    flag_masks_only = args.flag_masks_only

    data = pd.read_csv(split_filepath)
    if country in ['NL', 'AT', 'SE', 'LU', 'ES', 'SI', 'FR']:
        SENTINEL2_DIR = os.path.join(SENTINEL2_DIR, country)
        data = data[data["Country"] == country]
    os.makedirs(os.path.join(SENTINEL2_DIR, "masks"), exist_ok=True)
    if not flag_masks_only:
        os.makedirs(os.path.join(SENTINEL2_DIR, "images"), exist_ok=True)
        asyncio.run(download_images(data, SENTINEL2_DIR))
    else:
        asyncio.run(download_mask(data, SENTINEL2_DIR))


if __name__ == "__main__":
    main()
