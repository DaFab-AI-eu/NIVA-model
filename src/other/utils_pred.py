import sys
import os
from typing import Any, TypeAlias, cast
from numpy.typing import NDArray
import numpy as np
import onnxruntime as ort  # pyright: ignore[reportMissingTypeStubs]
import xarray as xr
from tqdm import tqdm
from matplotlib.colors import ListedColormap
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib import pyplot as plt

import rioxarray as rxr
import xarray as xr

# Add the src directory to the path
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(src_path)

from eda.utils_plot import show_single_ts
from accuracy_computation_sim import comp_scores

FloatArray: TypeAlias = NDArray[np.floating[Any]]

BANDS_ORDER = ("blue", "green", "red", "nir")

def normalize_meanstd(data):
    # code for normalization follows parameters here
    # https://github.com/zhu-xlab/SSL4EO-S12/blob/main/src/download_data/convert_rgb.py
    S2A_MEAN = {
        "blue": 889.6,
        "green": 1151.7,
        "red": 1307.6,
        "nir": 2538.9,
    }
    S2A_STD = {
        "blue": 1159.1,
        "green": 1188.1,
        "red": 1375.2,
        "nir": 1476.4,
    }

    mean = np.array([S2A_MEAN[band] for band in BANDS_ORDER], dtype=np.float64)
    std = np.array([S2A_STD[band] for band in BANDS_ORDER], dtype=np.float64)

    # follows normalization presented https://github.com/zhu-xlab/SSL4EO-S12/tree/main
    features = data.astype(np.float64)

    std_2 = 2.0 * std
    min_value = mean - std_2
    max_value = mean + std_2

    return np.clip(
        (features - min_value) / (max_value - min_value), a_min=0.0, a_max=1.0
    )

def predict(
    model: ort.InferenceSession, input: FloatArray
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """
    Simple method to call model.net.predict with normalization, but typed.
    """
    return cast(
        tuple[FloatArray, FloatArray, FloatArray],
        model.run(None, {"features": normalize_meanstd(input).astype(np.float32)}),
    )

def visualize_predictions(metrics_df, path_to_pdf, representative_imgs_cat, SENTINEL2_DIR,
                          path2predicted, category = "normal", date_idx = list(range(0, 6)), viz_factor=3.5):
    with PdfPages(path_to_pdf) as pdf:
        for file_id in tqdm(np.array(representative_imgs_cat[category])):
            # predicted
            p_extent, p_boundary, _ = get_predicted(path2predicted, file_id)
            combined, combined_s = get_combined(p_extent, p_boundary, p_limit=0.5)
            # ground truth
            extent, _, _, enum = get_mask(SENTINEL2_DIR, file_id)
            images, time_data = get_image(abs_path=SENTINEL2_DIR, file_id=file_id)
            images = np.clip(viz_factor * images / 10000, 0, 1)
            fig, axs = plt.subplots(figsize=(15, 20), ncols=6, nrows=len(date_idx), sharey=True)

            for idx in date_idx:
                sub_metrics = metrics_df[(metrics_df.file_id == file_id) & (metrics_df.date == time_data[idx])]['IoU_pixel_combined']
                sub_metrics = sub_metrics.values[0]
                show_single_ts(axis=axs[idx][0], msk=images[idx], ts=time_data[idx], feature_name=f"{file_id} RGB\nIoU={sub_metrics}",
                               alpha=1, grid=False)
                # enums
                num_fields = max(0, enum.max()) + 1
                colors = np.random.rand(num_fields, 3)
                colors[0] = np.array([0., 0., 0.]) # background color / without fields
                cmap = ListedColormap(colors, name='enum_cmap', N=num_fields)
                show_single_ts(axis=axs[idx][1], msk=enum, ts=None, feature_name=f"{file_id} field instances\n"
                                                                                 f"#fields={num_fields-1}\n"
                                                                                 f"%fields={int(np.round(100 * extent.sum( ) /np.prod(extent.shape)))}",
                               vmin=0, vmax=num_fields, cmap=cmap)

                # predicted extent
                show_single_ts(axis=axs[idx][2], msk=p_extent[idx], ts=None, feature_name=f"{file_id} field extent\n(predicted)",
                               alpha=1, grid=False)
                # predicted extent combined with boundary
                show_single_ts(axis=axs[idx][4], msk=combined[idx], ts=None, feature_name=f"{file_id} field combined\n(predicted)",
                               alpha=1, grid=False)
                # boundary
                show_single_ts(axis=axs[idx][3], msk=p_boundary[idx], ts=None, feature_name=f"{file_id} field boundary\n(predicted)",
                               alpha=1, grid=False)
                # combined binary (0, 1)
                show_single_ts(axis=axs[idx][5], msk=combined_s[idx], ts=None, feature_name=f"{file_id} field combined\n(predicted)",
                               alpha=1, grid=False)

            plt.tight_layout()
            pdf.savefig(fig) # problem. Doesn't save overlay with boundary at all!!!
            plt.show()


def smooth(array, disk_size: int = 2):
    # taken from https://github.com/sentinel-hub/field-delineation/blob/main/fd/post_processing.py
    """Blur input array using a disk element of a given disk size"""
    import skimage.util
    from skimage.filters import rank
    from skimage.morphology import disk  # pyright: ignore[reportUnknownVariableType]

    assert array.ndim == 2

    smoothed = rank.mean(  # pyright: ignore[reportUnknownMemberType]
            skimage.util.img_as_ubyte(array),  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]
            footprint=disk(disk_size).astype(np.float32),  # pyright: ignore[reportUnknownMemberType, reportAttributeAccessIssue]
    ).astype(np.float32)

    smoothed = (smoothed - np.min(smoothed)) / (np.max(smoothed) - np.min(smoothed))
    assert np.sum(~np.isfinite(smoothed)) == 0
    return smoothed

def get_combined(p_extent, p_boundary, p_limit=0.5):
    combined = np.clip(1 + p_extent - p_boundary, 0, 2)
    combined = (combined - np.min(combined)) / (np.max(combined) - np.min(combined))
    combined = np.stack([smooth(combined_i, disk_size = 1) for combined_i in combined])
    combined_s = np.where(combined >= p_limit, 1, 0)
    return combined, combined_s

def get_mask(abs_path, file_id):
    mask_path = os.path.join(abs_path, "masks", f"{file_id}_S2label_10m_256.tif")
    label_ds = rxr.open_rasterio(mask_path)
    extent, boundary, distance, enum = np.array(label_ds[:4])
    enum = enum.astype(int)
    enum[enum < 0] = 1 # background is -1000
    enum -= 1 # enumeration starts from 2 from fields
    extent, boundary = extent.astype(np.uint8), boundary.astype(np.uint8)  # need for correct visualization cmap
    return extent, boundary, distance, enum

def get_image(abs_path, file_id):
    image_path = os.path.join(abs_path, "images", f"{file_id}_S2_10m_256.nc")
    image_ds = xr.open_dataset(image_path)
    images = np.stack([image_ds["B4"], image_ds["B3"], image_ds["B2"]], axis=-1)
    time_data = np.array(image_ds.variables['time'][:], dtype='datetime64[M]')
    return images, time_data

def get_predicted(abs_path, file_id):
    predicted = np.load(os.path.join(abs_path, f"{file_id}.npy"))
    # "extent", "boundary", "distance" - (6, 256, 256, 2)
    extent, boundary, distance = predicted[..., 1]
    return extent, boundary, distance

def get_upack_patch(file_id, abs_path):
    image_path = os.path.join(abs_path, "images", f"{file_id}_S2_10m_256.nc")
    image_ds = xr.open_dataset(image_path)
    images = np.stack([image_ds["B2"], image_ds["B3"], image_ds["B4"], image_ds["B8"]], axis=-1)
    return images