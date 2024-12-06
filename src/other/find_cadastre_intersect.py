import argparse
import json

import geopandas as gpd

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--tile_meta", type=str, required=True)
    parser.add_argument("-c", "--cadastre_meta", type=str, required=True)
    parser.add_argument("-o", "--output", type=str, required=True)

    args = parser.parse_args()
    path_to_borders = args.cadastre_meta
    path_tile_meta = args.tile_meta
    output_path = args.output

    def get_intersection_cadatre_urls(path_to_borders, path_tile_meta):
        # finds intersection of tile grid and France regions with cadastre data available
        data_borders = gpd.read_file(path_to_borders)
        tile_meta = gpd.read_file(path_tile_meta)
        # for France the cadastre data is valid for previous year
        # https://geoservices.ign.fr/rpg#telechargementrpg2023
        data_borders = data_borders[data_borders.determination_year == tile_meta.datetime[0].year]
        # TODO add filtering of regions with low area of intersection
        data_borders = data_borders[data_borders.intersects(tile_meta.geometry.iloc[0], align=False)]
        data_borders = data_borders[["region", "url", "source_path"]]
        return data_borders

    data_borders = get_intersection_cadatre_urls(path_to_borders, path_tile_meta)
    print(f"{data_borders}")
    data_borders.to_json(output_path, orient='records')