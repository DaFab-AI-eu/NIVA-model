import json
import os

import numpy as np
import pandas as pd
from pyproj import Transformer
from skimage.measure import regionprops_table

import rioxarray as rxr
from tqdm import tqdm


def get_patch_stats(label_ds):
    # get position metadata
    meta = {}
    bbox = label_ds.rio.bounds()
    x = (bbox[1] + bbox[3]) / 2
    y = (bbox[0] + bbox[2]) / 2
    transformer = Transformer.from_crs("EPSG:3035", "EPSG:4326")
    meta["lat"], meta["lon"] = transformer.transform(x, y)
    # calculation of overall tile stats
    field_enum_np = np.array(label_ds[-1]).astype(np.int16)
    extent_mask_np = np.array(label_ds[0])
    meta["n_fields"] = len(np.unique(field_enum_np)) - 1  # -10000 for background
    meta["prc_fields"] = np.round(100 * extent_mask_np.sum() / extent_mask_np.size, 1)

    # calculate field obj stats
    # https://scikit-image.org/docs/stable/api/skimage.measure.html#skimage.measure.regionprops
    props = regionprops_table(
        field_enum_np,
        properties=(
            "label",
            "area",
            "perimeter",
            "solidity",  # Ratio of pixels in the region to pixels of the convex hull image. area/area_convex
            "axis_major_length",
            # The length of the major axis of the ellipse that has the same normalized second central moments as the region.
            "axis_minor_length",
            # The length of the minor axis of the ellipse that has the same normalized second central moments as the region.
            "eccentricity",  # The value is in the interval [0, 1). When it is 0, the ellipse becomes a circle.
            "area_filled",
            "extent",
            # Ratio of pixels in the region to pixels in the total bounding box. Computed as area / (rows * cols). area/area_bbox
            "orientation",
        # Angle between the 0th axis (rows) and the major axis of the ellipse that has the same second moments as the region, ranging from -pi/2 to pi/2 counter-clockwise.
        ),
    )
    props = pd.DataFrame(props)
    props["area/area_filled"] = props["area"] / props["area_filled"]
    props["axis_minor_length/axis_major_length"] = props["axis_minor_length"] / props["axis_major_length"]
    props.drop(columns=["area_filled", "label", "axis_minor_length", "axis_major_length"], inplace=True)

    # general stats
    for col in ["min", "mean", "max"]:
        meta.update(props.apply(col, axis=0).rename(lambda x: f"{x}_{col}").to_dict())
    return meta


def main():
    # input params
    SENTINEL2_DIR = "ai4boundaries_dataset"
    country = "NL"

    final_path = os.path.join(SENTINEL2_DIR, "stats")
    SENTINEL2_DIR = os.path.join(SENTINEL2_DIR, country)

    folder_masks = os.path.join(SENTINEL2_DIR, "masks")
    folder_stats = os.path.join(SENTINEL2_DIR, "stats")
    os.makedirs(folder_stats, exist_ok=True)
    os.makedirs(final_path, exist_ok=True)

    masks_path = [f.path for f in os.scandir(
        folder_masks) if f.is_file() and f.name.endswith('.tif')]

    for mask_path in tqdm(masks_path):
        label_ds = rxr.open_rasterio(mask_path)
        meta_stats = get_patch_stats(label_ds)
        file_id = os.path.split(mask_path)[-1][:-20]
        with open(os.path.join(folder_stats, f'{file_id}.json'), 'w') as file:
            json.dump(meta_stats, file)

    stats_path = [f.path for f in os.scandir(
        folder_stats) if f.is_file() and f.name.endswith('.json')]
    data_all = []
    for stat_path in stats_path:
        with open(stat_path, 'r') as file:
            json_object = json.load(file)
        json_object["file_id"] = os.path.splitext(os.path.split(stat_path)[-1])[0]
        data_all.append(json_object)
    data = pd.DataFrame(data_all)

    data.to_csv(os.path.join(final_path, f"{country}.csv"), index=False)


if __name__ == "__main__":
    main()
