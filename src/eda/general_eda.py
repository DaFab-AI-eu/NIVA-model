import os
from matplotlib import pyplot as plt
import pandas as pd


def general_stats(data, sm_prc_fields=10, sm_num_fields=50):
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

    data["no_field"] = data['n_fields'] == 0
    print(f"Number of patches without fields present {len(data[data['no_field']])}")
    data["sm_prc_fields"] = (~data["no_field"]) & (data["prc_fields"] < sm_prc_fields)
    print \
        (f"Number of patches with small percentage  (< {sm_prc_fields}) of fields present {len(data[data['sm_prc_fields']])}")
    data["sm_num_fields"] = (~data["no_field"]) & (data["n_fields"] < sm_num_fields)
    print \
        (f"Number of patches with small number (< {sm_num_fields}) of fields present {len(data[data['sm_num_fields']])}")
    print \
        (f"Number of patches with small number + small percentage of fields present combined {len(data[data['sm_num_fields'] | data['sm_prc_fields']])}")

    cols = ['n_fields', 'prc_fields',
            'sm_area_cnt', 'bg_area_cnt',
            'sm_solidity_cnt', 'bg_solidity_cnt',
            'sm_eccentricity_cnt']
    for col in cols:
        # log transformation
        mean, std = data[col].mean(), data[col].std()
        ls_val, gt_val =  data[col].quantile([0.25, 0.75]) # mean - 2 * std, mean + 2 * std
        fun_gt = lambda gt_val, col: data[col] > gt_val
        fun_ls = lambda ls_val, col: data[col] < ls_val
        data[f'{col}_gt'] = fun_gt(gt_val, col)
        data[f'{col}_ls'] = fun_ls(ls_val, col)
        print(f"Number of patches with {col} (>= {gt_val}) of fields present {len(data[data[f'{col}_gt']])}")
        print(f"Number of patches with {col} (< {ls_val}) of fields present {len(data[data[f'{col}_ls']])}")
    return data

def main():
    SENTINEL2_DIR = "C:\\Users\\alena.bakhoryna\\Documents\\code\\ai4boundaries_dataset"
    country = "NL"
    path_file = os.path.join(SENTINEL2_DIR, "stats", f"{country}.csv")
    path_out = os.path.join(SENTINEL2_DIR, "stats", f"{country}_stats.csv")
    data = pd.read_csv(path_file)
    data = general_stats(data)
    data.to_csv(path_out)


if __name__ == "__main__":
    main()
