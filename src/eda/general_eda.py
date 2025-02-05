import argparse
import os

import numpy as np
from matplotlib import pyplot as plt
import pandas as pd


def general_stats(data, sm_prc_fields=10, sm_num_fields=50):
    n_len = len(data) / 100
    cols = ['n_fields', 'prc_fields',
            'sm_area_cnt', 'bg_area_cnt',
            'sm_solidity_cnt', 'bg_solidity_cnt',
            'sm_eccentricity_cnt',
            'area_min', 'area_mean', 'area_max',
            'solidity_min', 'solidity_mean', 'solidity_max',
            'eccentricity_min', 'eccentricity_mean', 'eccentricity_max',
            'extent_min', 'extent_mean', 'extent_max']
    print(f"General information: \n {data[cols].describe()}")
    # histograms
    fig, ax = plt.subplots(len(cols), figsize=(16, 28))
    plt.title("Histograms")
    for ind in range(len(cols)):
        data.plot(column=cols[ind], kind="hist", bins=10, ax=ax[ind],
                  grid=True)
        # data.plot(column=cols[ind], kind="kde", ax=ax[ind],
        #                    grid=True)
    plt.show()

    print(f"Total number of fields {np.round(data['n_fields'].sum() / (10 ** 6), 2)} M")
    print(f"Total area of fields in ha {np.round(data['prc_fields'].sum() * (256. / 10. / 1000.)**2, 2)} M") # in millions
    data["no_field"] = data['n_fields'] == 0
    print(f"Number of patches without fields present {len(data[data['no_field']])} "
          f"({np.round(len(data[data['no_field']])/n_len, 2)}%)")
    data["sm_prc_fields"] = (~data["no_field"]) & (data["prc_fields"] < sm_prc_fields)
    print \
        (f"Number of patches with small percentage  (< {sm_prc_fields}) "
         f"of fields present {len(data[data['sm_prc_fields']])} "
         f"({np.round(len(data[data['sm_prc_fields']])/n_len, 2)}%)")
    data["sm_num_fields"] = (~data["no_field"]) & (data["n_fields"] < sm_num_fields)
    print \
        (f"Number of patches with small number (< {sm_num_fields}) of fields present "
         f"{len(data[data['sm_num_fields']])} ({np.round(len(data[data['sm_num_fields']])/n_len, 2)}%)")
    print \
        (f"Number of patches with small number or small percentage of fields present combined "
         f"{len(data[data['sm_num_fields'] | data['sm_prc_fields']])} "
         f"({np.round(len(data[data['sm_num_fields'] | data['sm_prc_fields']])/n_len, 2)}%)")
    print \
        (f"Number of patches with small number & small percentage of fields present combined "
         f"{len(data[data['sm_num_fields'] & data['sm_prc_fields']])} "
         f"({np.round(len(data[data['sm_num_fields'] & data['sm_prc_fields']]) / n_len, 2)}%)")

    cols = ['n_fields', 'prc_fields',
            'sm_area_cnt', 'bg_area_cnt',
            'sm_solidity_cnt', 'bg_solidity_cnt',
            'sm_eccentricity_cnt']
    for col in cols:
        # log transformation for screwed distributions
        mean, std = data[col].mean(), data[col].std()
        # quantile -> % 25 percent of values
        # ls_val, gt_val =  data[col].quantile([0.25, 0.75])
        if col != 'prc_fields':
            ls_val, gt_val = np.round(mean - 2 * std, 2), np.round(mean + 2 * std, 2)
        else:
            ls_val, gt_val = 25., 75.
        fun_gt = lambda gt_val, col: data[col] > gt_val
        fun_ls = lambda ls_val, col: data[col] < ls_val
        data[f'{col}_gt'] = fun_gt(gt_val, col)
        data[f'{col}_ls'] = fun_ls(ls_val, col)
        print(f"Number of patches with {col} (>= {gt_val}) of fields present "
              f"{len(data[data[f'{col}_gt']])} ({np.round(len(data[data[f'{col}_gt']])/n_len, 2)}%)")
        if ls_val > 0:
            print(f"Number of patches with {col} (< {ls_val}) of fields present "
                  f"{len(data[data[f'{col}_ls']])} ({np.round(len(data[data[f'{col}_ls']])/n_len, 2)}%)")
    return data

def general_eda_main(SENTINEL2_DIR, country,
                     sm_prc_fields=10, sm_num_fields=40):
    path_file = os.path.join(SENTINEL2_DIR, "stats", f"{country}.csv")
    path_out = os.path.join(SENTINEL2_DIR, "stats", f"{country}_stats.csv")
    data = pd.read_csv(path_file)
    data = general_stats(data, sm_prc_fields=sm_prc_fields, sm_num_fields=sm_num_fields)
    data.to_csv(path_out, index=False)
    return data

def main():
    # input params
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--country", type=str, required=True,
                        default="NL")
    parser.add_argument("-s", "--sentinel_dir", type=str, required=True,
                        default="ai4boundaries_dataset")
    args = parser.parse_args()

    country = args.country
    SENTINEL2_DIR = args.sentinel_dir

    general_eda_main(SENTINEL2_DIR, country)


if __name__ == "__main__":
    main()
