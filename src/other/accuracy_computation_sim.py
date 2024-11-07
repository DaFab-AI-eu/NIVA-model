import argparse
import glob
import os
import sys
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from tqdm import tqdm
from sklearn.metrics import matthews_corrcoef, cohen_kappa_score, confusion_matrix, accuracy_score, \
    ConfusionMatrixDisplay

import fs.move  # required by eopatch.save
from eolearn.core import FeatureType, EOPatch, OverwritePermission


from utils_plot import draw_true_color, draw_bbox, draw_vector_timeless, draw_mask
from transform_vector2mask import main_rastorize_vector

# Add the src directory to the path
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(src_path)

# Import logger
from niva_utils.logger import get_logger  # noqa: E402
LOGGER = get_logger(__name__)


VISUALIZE = False

def display_cm(cm):
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['no field', 'field'])
    # 0 - no field, 1 - field
    disp.plot(cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()


def compute_iou_from_cm(cm):
    # compute mean iou
    # iou = true_positives / (true_positives + false_positives + false_negatives)
    intersection = np.diag(cm)
    ground_truth_set = cm.sum(axis=1)
    predicted_set = cm.sum(axis=0)
    union = ground_truth_set + predicted_set - intersection
    IoU = intersection / union.astype(np.float32)
    return np.nanmean(IoU)


def compute_iou(y_true, y_pred, cm_display=True):
    y_pred = y_pred.flatten()
    y_true = y_true.flatten()
    LOGGER.info(f"y_true fields ={np.count_nonzero(y_true) / len(y_true)}, "
                f"y_pred fields ={np.count_nonzero(y_pred) / len(y_pred)}")
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    if cm_display:
        display_cm(cm / len(y_true))
    return cm, compute_iou_from_cm(cm)


def comp_scores(mask_gt, mask_pred):
    mask_gt = mask_gt.flatten()
    mask_pred = mask_pred.flatten()
    score_names = ["accuracy_score", "matthews_corrcoef", "cohen_kappa_score"]
    score_vals = []
    score_vals.append(accuracy_score(mask_gt, mask_pred))
    score_vals.append(matthews_corrcoef(mask_gt, mask_pred))
    score_vals.append(cohen_kappa_score(mask_gt, mask_pred))
    cm, iou = compute_iou(y_pred=mask_pred, y_true=mask_gt, cm_display=VISUALIZE)
    score_names.append("iou")
    score_vals.append(iou)

    cm_normalized = cm / cm.sum()
    score_name_ = ["TN (no field t, no field p)", "FP (no field t, field p)",
                   "FN (field t, no field p)", "TP (field t, field p)"]
    score_names.extend(score_name_)
    score_vals.extend(cm_normalized.flatten().tolist())

    score_names.append("cm")
    score_vals.append(cm)
    return score_names, score_vals


def get_masks_pred_gt(eopatch_folder, cadastre_tile_path,
    pred_file_path):
    for feature_name, vector_file_path in zip(["CADASTRE", "PREDICTED"],
                                              [cadastre_tile_path,
                                               pred_file_path]):
        rasterise_gsaa_config = {
            "vector_file_path": vector_file_path,
            "eopatches_folder": eopatch_folder,
            "feature_name": feature_name,
            "vector_feature": ["vector_timeless", feature_name],
            "extent_feature": ["mask_timeless", f"EXTENT_{feature_name}"],
            "boundary_feature": ["mask_timeless", f"BOUNDARY_{feature_name}"],
            # "distance_feature": ["data_timeless", "DISTANCE"],
        }
        main_rastorize_vector(rasterise_gsaa_config)


def display_metrics(eopatch_folder):
    eopatch = EOPatch.load(eopatch_folder)
    mask_gt = eopatch.mask_timeless['EXTENT_CADASTRE'].squeeze()
    mask_pred = eopatch.mask_timeless['EXTENT_PREDICTED'].squeeze()
    scores_names, score_vals = comp_scores(mask_pred=mask_pred, mask_gt=mask_gt)
    LOGGER.info(f"Metrics for patch {eopatch_folder} \n {dict(zip(scores_names, score_vals))}")
    return scores_names, score_vals


def visualize_eopatch(eop):
    fig, axis = plt.subplots(figsize=(15, 10), ncols=1, sharey=True)
    eop.vector_timeless['CADASTRE'].plot(ax=axis, color='green', alpha=0.5)
    eop.vector_timeless['PREDICTED'].plot(ax=axis, color='red', alpha=0.5)
    plt.show()

    for mask_name in ["EXTENT", "BOUNDARY"]:
        time_idx = 0  # only one tile stemp
        fig, ax = plt.subplots(ncols=4, figsize=(15, 20))
        draw_true_color(ax[0], eop, time_idx=time_idx, factor=3.5 / 10000, feature_name='BANDS', bands=(2, 1, 0),
                        grid=False)
        draw_bbox(ax[0], eop)
        draw_vector_timeless(ax[0], eop, vector_name='CADASTRE', alpha=.3)

        draw_true_color(ax[1], eop, time_idx=time_idx, factor=3.5 / 10000, feature_name='BANDS', bands=(2, 1, 0),
                        grid=False)
        draw_bbox(ax[1], eop)
        draw_mask(ax[1], eop, time_idx=None, feature_name=f'{mask_name}_CADASTRE', alpha=.3)

        draw_true_color(ax[2], eop, time_idx=time_idx, factor=3.5 / 10000, feature_name='BANDS', bands=(2, 1, 0),
                        grid=False)
        draw_bbox(ax[2], eop)
        draw_mask(ax[2], eop, time_idx=None, feature_name=f'{mask_name}_PREDICTED', alpha=.3)

        draw_true_color(ax[3], eop, time_idx=time_idx, factor=3.5 / 10000, feature_name='BANDS', bands=(2, 1, 0),
                        grid=False)
        ax[3].grid()
        plt.show()

    for mask_name in ["EXTENT", "BOUNDARY"]:
        time_idx = 0  # only one tile stemp
        fig, ax = plt.subplots(ncols=4, figsize=(15, 20))
        draw_true_color(ax[0], eop, time_idx=time_idx, factor=3.5 / 10000, feature_name='BANDS', bands=(2, 1, 0),
                        grid=False)
        draw_bbox(ax[0], eop)
        draw_vector_timeless(ax[0], eop, vector_name='PREDICTED', alpha=.3)

        draw_true_color(ax[1], eop, time_idx=time_idx, factor=3.5 / 10000, feature_name='BANDS', bands=(2, 1, 0),
                        grid=False)
        draw_bbox(ax[1], eop)
        draw_mask(ax[1], eop, time_idx=None, feature_name=f'{mask_name}_PREDICTED', alpha=.3)

        draw_true_color(ax[2], eop, time_idx=time_idx, factor=3.5 / 10000, feature_name='BANDS', bands=(2, 1, 0),
                        grid=False)
        draw_bbox(ax[2], eop)
        draw_mask(ax[2], eop, time_idx=None, feature_name=f'{mask_name}_PREDICTED', alpha=.3,
                  data_timeless=True)  # !!!!!!!!! visualize predicted masks before vectorization step
        draw_true_color(ax[3], eop, time_idx=time_idx, factor=3.5 / 10000, feature_name='BANDS', bands=(2, 1, 0),
                        grid=False)
        ax[3].grid()
        plt.show()



def main():
    # read GT files for region, and read Tile predicted agr
    # convert back to pixels and compare by grid (partial loading) geoJson?
    # https://cadastre.data.gouv.fr/data/etalab-cadastre/2024-07-01/geojson/departements/18/
    # contains in parcels buildings too (needs to filter non crop parcels)
    # Luxembourg 2024 (bad weather during March/April > 50% cloud cover)

    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--cadastre_tile_path", type=str, required=True)
    parser.add_argument("-p", "--pred_file_path", type=str, required=True)
    parser.add_argument("-e", "--eopatches_folder", type=str, required=True)
    parser.add_argument("-o", "--metrics_path", type=str, required=True)

    args = parser.parse_args()

    cadastre_tile_path = args.cadastre_tile_path
    pred_file_path = args.pred_file_path
    metrics_path = args.metrics_path

    eopatches_path = [f.path for f in os.scandir(
        args.eopatches_folder) if f.is_dir() and f.name.startswith('eopatch')]

    score_vals_list = []
    cms_list = []
    for eopatch_folder in tqdm(eopatches_path):
        # creates gt and predicted masks from vectors (MOST important method)
        get_masks_pred_gt(eopatch_folder, cadastre_tile_path, pred_file_path)

        if VISUALIZE:
            eopatch = EOPatch.load(eopatch_folder)
            visualize_eopatch(eopatch)

        scores_names, score_vals = display_metrics(eopatch_folder)
        score_vals, cm = score_vals[:-1], score_vals[-1]
        score_vals_list.append(score_vals)
        cms_list.append(cm)

    cm_all = np.stack(cms_list).sum(axis=0)
    iou_all = compute_iou_from_cm(cm_all)
    cm_all = cm_all.astype(float)
    cm_all /= cm_all.sum().astype(cm_all.dtype)
    display_cm(cm_all)

    # create metrics file
    score_vals_list.append(np.array(score_vals_list).mean(axis=0).tolist())
    metrics_final = pd.DataFrame(columns=["eopatch_folder", "iou_all"] + scores_names[:-1])
    metrics_final["eopatch_folder"] = eopatches_path + ["mean"]
    metrics_final[scores_names[:-1]] = score_vals_list
    metrics_final.loc[metrics_final["eopatch_folder"] == "mean", scores_names[-5:-1]] = cm_all.flatten().tolist()
    metrics_final.loc[metrics_final["eopatch_folder"] == "mean", "iou_all"] = iou_all
    metrics_final.to_csv(metrics_path)

    LOGGER.info(f"confusion matrics for the all eopatches combined {cm_all}")
    LOGGER.info(f"Mean Metrics \n {metrics_final.iloc[-1]}")


if __name__ == "__main__":
    main()
