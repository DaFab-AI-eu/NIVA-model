import os
import random

from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
from matplotlib import pyplot as plt
import rioxarray as rxr
import xarray as xr
from matplotlib.colors import ListedColormap


def get_upack_patch(file_id, abs_path, num_img=6):
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
    extent, boundary = extent.astype(np.uint8), boundary.astype(np.uint8)  # need for correct visualization cmap
    if num_img == 1:
        ids = [len(image_ds) // 2]
    elif num_img == 2:
        ids = [0, len(image_ds) - 1]
    elif num_img == 3:
        ids = [0, len(image_ds) // 2, len(image_ds) - 1]
    elif num_img == 6: # visualize all images
        ids = list(range(num_img))
    else:
        raise ValueError(f"Unsupported value {num_img} provided!")
    images, time_data = images[ids], time_data[ids]
    return images, extent, boundary, distance, enum, time_data

def show_single_ts(axis, msk, ts=None, feature_name="extent", vmin=0, vmax=1, alpha=1., grid=False, cmap=None):
    axis.imshow(msk, vmin=vmin, vmax=vmax, alpha=alpha,  cmap=cmap)
    if grid:
        axis.grid()
    title = f'{feature_name} {ts}' if ts else f'{feature_name}'
    axis.set_title(title)
    axis.set_axis_off()


def plot_patch(file_id, images, extent, boundary, distance, enum, time_data):
    # image shape 256*256 pixels
    factor = 3.5 / 10000
    fig, ax = plt.subplots(ncols=len(images) + 1, nrows=2,
                           figsize=(24 * (len(images)+1)/7., 8))
    ind_row = 0
    # show_single_ts(axis=ax[ind_row][0], msk=distance, ts=None, feature_name=f"distance {file_id}",
    #                vmin=0, vmax=1, alpha=1., grid=False, cmap='gray')
    for ind in range(len(images)):
        mask = np.clip(images[ind] * factor, 0, 1)
        show_single_ts(axis=ax[ind_row][ind + 1], msk=mask, ts=time_data[ind], feature_name=f"{file_id}",
                      alpha=1, grid=False)

    alpha = 0.2
    ind_row = 0

    num_fields = max(0, enum.max()) + 1
    colors = np.random.rand(num_fields, 3)
    colors[0] = np.array([0., 0., 0.]) # background color / without fields
    cmap = ListedColormap(colors, name='enum_cmap', N=num_fields)
    show_single_ts(axis=ax[ind_row][0], msk=enum, ts=None, feature_name=f"extent/enum {file_id}\n"
                                                                        f"#fields={num_fields-1}\n"
                                                                        f"%fields={int(np.round(100 * extent.sum()/np.prod(extent.shape)))}",
                   vmin=0, vmax=num_fields, cmap=cmap)

    # for ind in range(len(images)):
    #     mask = np.clip(images[ind] * factor, 0, 1)
    #     show_single_ts(axis=ax[ind_row][ind + 1], msk=mask, ts=time_data[ind], feature_name=f"{file_id}",
    #                   alpha=1)
    #     ##
    #     show_single_ts(axis=ax[ind_row][ind + 1], msk=extent, ts=None,
    #                    feature_name=f"extent {file_id} {time_data[ind]}",
    #                    alpha=alpha, cmap='Reds')

    alpha = 0.2
    ind_row = 1
    show_single_ts(axis=ax[ind_row][0], msk=boundary, ts=None, feature_name=f"boundary {file_id}",
                   vmin=0, vmax=1, alpha=1, grid=False, cmap='gray')
    for ind in range(len(images)):
        mask = np.clip(images[ind] * factor, 0, 1)
        show_single_ts(axis=ax[ind_row][ind + 1], msk=mask, ts=time_data[ind], feature_name=f"{file_id}",
                       alpha=1, grid=False)
        ##
        show_single_ts(axis=ax[ind_row][ind + 1], msk=boundary, ts=None,
                       feature_name=f"boundary {file_id} {time_data[ind]}",
                       alpha=alpha, cmap='Reds')
    return fig


def save2pdf(path_to_pdf, abs_path, file_ids, num_img=6):
    # visualize patches
    with PdfPages(path_to_pdf) as pdf:
        for file_id in file_ids:
            images, extent, boundary, distance, enum, time_data = get_upack_patch(file_id,
                    abs_path, num_img=num_img)
            fig = plot_patch(file_id, images, extent, boundary, distance, enum, time_data)
            plt.tight_layout()
            pdf.savefig(fig) # problem. Doesn't save overlay with boundary at all!!!
            plt.show()
            # plt.close()

def visualize_col_data(data, SENTINEL2_DIR, country='NL', col='sm_num_fields', num_img=2,
                       num_plots=5):
    file_ids = data[data[col]].reset_index(drop=True)
    file_ids = file_ids.file_id.tolist()
    file_ids = random.sample(file_ids, k=num_plots)

    abs_path = os.path.join(SENTINEL2_DIR, country)
    path_to_pdf = os.path.join(SENTINEL2_DIR, f"{col}_{country}_v1.pdf")

    save2pdf(path_to_pdf, abs_path, file_ids, num_img=num_img)