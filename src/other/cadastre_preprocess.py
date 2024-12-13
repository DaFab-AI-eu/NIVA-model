import argparse
import os
import sys
import geopandas as gpd
import numpy as np

# Add the src directory to the path
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(src_path)

# Import logger
from niva_utils.logger import get_logger  # noqa: E402
LOGGER = get_logger(__name__)


# The ground truth data for France from https://geoservices.ign.fr/rpg

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--cadastre_path", type=str, required=True)
    parser.add_argument("-o", "--cadastre_tile_path", type=str, required=True)
    parser.add_argument("--tile_meta", type=str, required=True)
    parser.add_argument("--sim_tolerance", type=float, required=False,
                        default=5)

    args = parser.parse_args()
    sim_tolerance = args.sim_tolerance
    cadastre_path = args.cadastre_path
    cadastre_tile_path = args.cadastre_tile_path
    tile_meta_path = args.tile_meta

    tile_meta = gpd.read_file(tile_meta_path, columns=["geometry"])
    # read only the data for the given tile bounds
    if cadastre_path.endswith(".parquet"):
        # for fiboa data source
        # year filtering?
        data_gt = gpd.read_parquet(cadastre_path, columns=["geometry"])
        # determination_datetime filtering
        LOGGER.info(f"Whole file len {len(data_gt)}")
        data_gt = data_gt.clip(tile_meta.to_crs(data_gt.crs))
        LOGGER.info(f"File len after clipping to tile_grid {len(data_gt)}")
    else:
        # for France cadastre
        data_gt = gpd.read_file(cadastre_path, bbox=tile_meta.geometry, columns=["geometry"])  # [386588 rows x 7 columns]

    """Simple data analysis below"""
    # Columns for France : [ID_PARCEL, SURF_PARC, CODE_CULTU, CODE_GROUP, CULTURE_D1, CULTURE_D2, geometry]
    LOGGER.info(f"file {cadastre_path} len {len(data_gt)} and columns {data_gt.columns}")
    LOGGER.info(f"file {cadastre_path} crs {data_gt.crs}")
    LOGGER.info(f"invalid geometries in dataset = {data_gt[~data_gt.geometry.is_valid]}")
    data_gt = data_gt[data_gt.geometry.is_valid]
    # no MultiPolygons
    multi_pol = data_gt[data_gt.geometry.type != 'Polygon']
    if len(multi_pol):
        org_len = len(data_gt)
        data_gt = data_gt.explode(ignore_index=True)
        LOGGER.info(f"Other geometries except Polygon in dataset = {multi_pol.geometry.type.unique()}. "
                    f"Length before explode {org_len} and after {len(data_gt)}")

    coor_counts = data_gt.geometry.count_coordinates()
    rings_count = data_gt.geometry.count_interior_rings()
    LOGGER.info(f"Interior ring count min = {np.min(rings_count)} max = {np.max(rings_count)}")  # 0, 35
    LOGGER.info(f"Number of geometries with interior ring = {len(data_gt[rings_count != 0])}")  # 6015
    LOGGER.info(f"Number of coordinates for geometries min = {np.min(coor_counts)}, "
                f"max = {np.max(coor_counts)}, mean = {np.mean(coor_counts)}")  # 4 836 23.5

    data_gt_sim = data_gt.geometry.simplify(tolerance=sim_tolerance,
                                            preserve_topology=True)
    coor_counts = data_gt_sim.geometry.count_coordinates()
    LOGGER.info(f"Number of coordinates for geometries after simplification min = {np.min(coor_counts)}, "
                f"max = {np.max(coor_counts)}, mean = {np.mean(coor_counts)}")  # 4 352 9.4
    LOGGER.info(f"Average area after simplification = {np.mean(data_gt_sim.geometry.area)}, "
               f"Average area before simplification = {np.mean(data_gt.geometry.area)}")

    data_gt_sim.to_file(cadastre_tile_path)


if __name__ == "__main__":
    main()
