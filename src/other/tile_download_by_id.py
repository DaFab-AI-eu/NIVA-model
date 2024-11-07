import argparse
import os
import sys
import logging
from time import time
import geopandas as gpd
from pystac_client import Client
import odc.ui
import xarray as xr
from dask.diagnostics import ProgressBar
from odc.stac import stac_load
from tqdm import tqdm

# Add the src directory to the path
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(src_path)

# Import logger
from niva_utils.logger import get_logger  # noqa: E402
LOGGER = get_logger(__name__)


api_url = "https://earth-search.aws.element84.com/v1"
collection = "sentinel-2-l2a"


def get_search_json(ids, url_l=api_url,
                    collections=[collection]):
    catalog = Client.open(url_l)
    query = catalog.search(
        collections=collections, limit=100,
        ids=ids,  # search by ids !!!!!!!
    )
    items = list(query.items())
    LOGGER.info(f"Found: {len(items):d} datasets")
    # Convert STAC items into a GeoJSON FeatureCollection
    stac_json = query.item_collection_as_dict()
    return stac_json, items


def get_gdf(stac_json):
    gdf = gpd.GeoDataFrame.from_features(stac_json, "epsg:4326")

    # Compute granule id from components
    gdf["granule"] = (
            gdf["mgrs:utm_zone"].apply(lambda x: f"{x:02d}")
            + gdf["mgrs:latitude_band"]
            + gdf["mgrs:grid_square"]
    )
    return gdf


def load_item(item, output_folder, use_dask=True, chunk_size=2048):
    bands = ["blue", "green", "red", "nir", "scl"]  # ["B02", "B03", "B04", "B08", "SCL"]
    # downloading the tile specified bands
    if use_dask:
        ds: xr.Dataset = odc.stac.load(items=[item],
                                       bands=bands,
                                       chunks={'x': chunk_size,
                                               'y': chunk_size},
                                       resolution=10)
        with ProgressBar():
            ds.load()
    else:
        ds: xr.Dataset = odc.stac.load(items=[item],
                                       bands=bands,
                                       resolution=10,
                                       progress=tqdm.tqdm)

    ds = ds.rename_vars({"blue": "B2", "green": "B3", "red": "B4", "nir": "B8"})
    # saving the tile, takes some time
    ds.to_netcdf(os.path.join(output_folder, f"{item.id}.nc"))
    return ds


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--tile_id", type=str, required=True)
    parser.add_argument("-r", "--project_root", type=str, required=True)
    parser.add_argument("--chunk_size", type=int, required=False, default=2048)

    args = parser.parse_args()
    TILE_ID = [args.tile_id]
    TILE_FOLDER = args.project_root
    XARRAY_CHUNK_SIZE = args.chunk_size

    durations = []
    LOGGER.info(f"Searching STAC endpoint {api_url} for items ids {TILE_ID}")
    start_time = time()
    stac_json, items = get_search_json(TILE_ID)
    end_time = time()
    durations.append(('search_stac_endpoint', end_time - start_time))

    gdf = get_gdf(stac_json)
    os.makedirs(TILE_FOLDER, exist_ok=True)
    file_path = os.path.join(TILE_FOLDER, f"stac_s_{'_'.join(TILE_ID)}.gpkg")
    gdf.to_file(file_path)
    LOGGER.info(f"Saved search results to {file_path}")

    columns = ['datetime', 'granule', 'eo:cloud_cover', 's2:vegetation_percentage',
               's2:nodata_pixel_percentage']  # 's2:product_uri'
    LOGGER.info(f"Simple information for items:\n{gdf[columns]}")

    start_time = time()
    for ind, item in tqdm(enumerate(items), desc="Process of tiles download"):
        output_folder = os.path.join(TILE_FOLDER, item.id, "tile")
        os.makedirs(output_folder, exist_ok=True)

        gdf.iloc[ind: ind + 1].to_file(os.path.join(output_folder, "metadata.gpkg"))

        load_item(item, output_folder, chunk_size=XARRAY_CHUNK_SIZE)
    end_time = time()
    durations.append(('downloading', end_time - start_time))

    LOGGER.info("\n--- Overview ---")
    total_time = 0
    for step, duration in durations:
        LOGGER.info(f"Total time for {step}: {duration:.2f} seconds")
        total_time += duration
    logging.info(f"Total time for all steps: {total_time:.2f} seconds")


if __name__ == "__main__":
    main()
