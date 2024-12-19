import numpy as np
from matplotlib import pyplot as plt


def show_single_ts(axis, msk, ts=None, feature_name="extent", vmin=0, vmax=1, alpha=1., grid=False, cmap='viridis'):
    axis.imshow(msk, vmin=vmin, vmax=vmax, alpha=alpha,  cmap=cmap)
    if grid:
        axis.grid()
    title = f'{feature_name} {ts}' if ts else f'{feature_name}'
    axis.set_title(title)


def plot_patch(file_id, images, extent, boundary, distance, time_data):
    # TODO visualize enum, distance
    factor = 3.5 / 10000
    fig, ax = plt.subplots(ncols=len(images) + 1, nrows=3, figsize=(24, 15))
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
    show_single_ts(axis=ax[ind_row][0], msk=extent, ts=None, feature_name=f"extent {file_id}",
                   vmin=0, vmax=1, alpha=1., grid=False, cmap='gray')
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