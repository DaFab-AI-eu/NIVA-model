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
    split_filepath = "ai4boundaries_ftp_urls_sentinel2_split.csv"
    SENTINEL2_DIR = "ai4boundaries_dataset"

    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--country", type=str, required=True,
                        default="NL")
    args = parser.parse_args()

    country = args.country

    SENTINEL2_DIR = os.path.join(SENTINEL2_DIR, country)

    data = pd.read_csv(split_filepath)
    fold_data = data[data["Country"] == country]
    os.makedirs(os.path.join(SENTINEL2_DIR, "masks"), exist_ok=True)
    asyncio.run(download_mask(fold_data, SENTINEL2_DIR))


if __name__ == "__main__":
    main()
