import os

from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
from matplotlib import pyplot as plt
import rioxarray as rxr
import xarray as xr
from matplotlib.colors import ListedColormap


def get_upack_patch(file_id,
                    abs_path):
    mask_path = os.path.join(abs_path, "masks", f"{file_id}_S2label_10m_256.tif")
    image_path = os.path.join(abs_path, "images", f"{file_id}_S2_10m_256.nc")
    label_ds = rxr.open_rasterio(mask_path)
    image_ds = xr.open_dataset(image_path)
    images = np.stack([image_ds["B4"], image_ds["B3"], image_ds["B2"]], axis=-1)
    time_data = np.array(image_ds.variables['time'][:], dtype='datetime64[D]')
    extent, boundary, distance, enum = np.array(label_ds[:4])
    enum = enum.astype(int)
    enum[enum < 0] = 1 # background is -1000
    enum -= 1 # enumeration starts from 2 from fields
    return images, extent, boundary, distance, enum, time_data

def show_single_ts(axis, msk, ts=None, feature_name="extent", vmin=0, vmax=1, alpha=1., grid=False, cmap='viridis'):
    axis.imshow(msk, vmin=vmin, vmax=vmax, alpha=alpha,  cmap=cmap)
    if grid:
        axis.grid()
    title = f'{feature_name} {ts}' if ts else f'{feature_name}'
    axis.set_title(title)
    axis.set_axis_off()


def plot_patch(file_id, images, extent, boundary, distance, enum, time_data):
    # image shape 256*256 pixels
    factor = 3.5 / 10000
    fig, ax = plt.subplots(ncols=len(images) + 1, nrows=3, figsize=(24, 10))
    alpha = 1.
    ind_row = 0
    show_single_ts(axis=ax[ind_row][0], msk=distance, ts=None, feature_name=f"distance {file_id}",
                   vmin=0, vmax=1, alpha=1., grid=False, cmap='gray')
    for ind in range(len(images)):
        mask = np.clip(images[ind] * factor, 0, 1)
        show_single_ts(axis=ax[ind_row][ind + 1], msk=mask, ts=time_data[ind], feature_name=f"{file_id}",
                       vmin=0, vmax=1, alpha=alpha, grid=False)

    alpha = 0.8
    ind_row = 1

    num_fields = enum.max() + 1
    colors = np.random.rand(num_fields, 3)
    colors[0] = np.array([0., 0., 0.]) # background color / without fields
    cmap = ListedColormap(colors, name='enum_cmap', N=num_fields)
    show_single_ts(axis=ax[ind_row][0], msk=enum, ts=None, feature_name=f"extent/enum {file_id}", vmin=0, vmax=num_fields, alpha=1., grid=False,
                   cmap=cmap)

    for ind in range(len(images)):
        mask = np.clip(images[ind] * factor, 0, 1)
        show_single_ts(axis=ax[ind_row][ind + 1], msk=mask, ts=time_data[ind], feature_name=f"{file_id}",
                       vmin=0, vmax=1, alpha=alpha, grid=False)
        ##
        show_single_ts(axis=ax[ind_row][ind + 1], msk=extent, ts=None,
                       feature_name=f"extent {file_id} {time_data[ind]}",
                       vmin=0, vmax=1, alpha=1 - alpha, grid=False, cmap='gray')

    alpha = 0.8
    ind_row = 2
    show_single_ts(axis=ax[ind_row][0], msk=boundary, ts=None, feature_name=f"boundary {file_id}",
                   vmin=0, vmax=1, alpha=1, grid=False, cmap='gray')
    for ind in range(len(images)):
        mask = np.clip(images[ind] * factor, 0, 1)
        show_single_ts(axis=ax[ind_row][ind + 1], msk=mask, ts=time_data[ind], feature_name=f"{file_id}",
                       vmin=0, vmax=1, alpha=alpha, grid=False)
        ##
        show_single_ts(axis=ax[ind_row][ind + 1], msk=boundary, ts=None,
                       feature_name=f"boundary {file_id} {time_data[ind]}",
                       vmin=0, vmax=1, alpha=1 - alpha, grid=False, cmap='gray')
    return fig


def save2pdf(path_to_pdf, abs_path, file_ids):
    # visualize patches
    with PdfPages(path_to_pdf) as pdf:
        for file_id in enumerate(file_ids):
            images, extent, boundary, distance, enum, time_data = get_upack_patch(file_id,
                    abs_path)
            fig = plot_patch(file_id, images, extent, boundary, distance, enum, time_data)
            plt.show()
            pdf.savefig(fig)
            plt.close()