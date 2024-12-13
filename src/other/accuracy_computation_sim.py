import argparse
import os
import sys
import pandas as pd
import geopandas as gpd
import shapely
from matplotlib import pyplot as plt
import numpy as np
from tqdm import tqdm
from sklearn.metrics import matthews_corrcoef, cohen_kappa_score, confusion_matrix, accuracy_score, \
    ConfusionMatrixDisplay

import fs.move  # required by eopatch.save
from eolearn.core import EOPatch


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
    # TODO when there is no field pred and gt IoU should be 1
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
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    if cm_display:
        display_cm(cm / len(y_true))
    return cm, compute_iou_from_cm(cm)


def comp_scores(mask_gt, mask_pred):
    mask_gt = mask_gt.flatten()
    mask_pred = mask_pred.flatten()
    metrics_dict = {}
    func_perc = lambda arr: arr[arr == 1].sum() / len(arr)
    metrics_dict["CADASTRE_fields_prc"] = func_perc(mask_gt)
    metrics_dict["PREDICTED_fields_prc"] = func_perc(mask_pred)
    metrics_dict["accuracy_score"] = accuracy_score(mask_gt, mask_pred)
    # TODO when there is no field pred and gt matthews_corrcoef should be 1
    metrics_dict["matthews_corrcoef"] = matthews_corrcoef(mask_gt, mask_pred)
    # TODO when there is no field pred and gt cohen_kappa_score should be 1
    metrics_dict["cohen_kappa_score"] = cohen_kappa_score(mask_gt, mask_pred)
    cm, iou = compute_iou(y_pred=mask_pred, y_true=mask_gt, cm_display=VISUALIZE)
    metrics_dict["iou"] = iou
    cm_normalized = cm / cm.sum()
    score_names = ["TN (no field t, no field p)", "FP (no field t, field p)",
                   "FN (field t, no field p)", "TP (field t, field p)"]
    metrics_dict.update(dict(zip(score_names, cm_normalized.flatten().tolist())))
    return metrics_dict


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


def vector_analyses(vector, sub_name='CADASTRE'):
    areas = vector.area # in square meters
    metrics_dict = {
        f"{sub_name}_num_pol": len(vector),
        f"{sub_name}_max_areas/m^2": areas.max(),
        f"{sub_name}_min_areas/m^2": areas.min(),
        f"{sub_name}_avg_areas/m^2": areas.mean(),
    }
    return metrics_dict

def get_object_level_metrics(y_true_shapes, y_pred_shapes, iou_threshold=0.5):
    # according to https://github.com/fieldsoftheworld/ftw-baselines/blob/main/src/ftw/metrics.py#L5
    """Get object level metrics for a single mask / prediction pair.

        Args:
            y_pred_shapes (geopandas): Predicted vectors.
            y_true_shapes (geopandas): Truth vectors.
            iou_threshold (float, optional): IoU threshold for matching predictions to ground truths. Defaults to 0.5.

        Returns
            tuple (int, int, int): Number of true positives, false positives, and false negatives.
        """
    if iou_threshold < 0.5:
        raise ValueError(
            "iou_threshold must be greater than 0.5")  # If we go lower than 0.5 then it is possible for a single prediction to match with multiple ground truths and we have to do de-duplication

    tps = 0
    fns = 0
    tp_is = set()  # keep track of which of the true shapes are true positives
    tp_js = set()  # keep track of which of the predicted shapes are true positives
    fn_is = set()  # keep track of which of the true shapes are false negatives
    matched_js = set()
    # takes long to compute
    for i, y_true_shape in enumerate(y_true_shapes.geometry):
        matching_j = None
        if not shapely.is_valid(y_true_shape):
            continue

        y_intersect = y_pred_shapes[y_true_shape.intersects(y_pred_shapes.geometry)] # shape 0, 1, 2, 3
        for j, y_pred_shape in y_intersect.geometry.items():
            intersection = y_true_shape.intersection(y_pred_shape)
            union = y_true_shape.union(y_pred_shape)
            iou = intersection.area / union.area
            if iou > iou_threshold:
                matching_j = j
                matched_js.add(j)
                tp_js.add(j)
                break

        if matching_j is not None:
            tp_is.add(i)
            tps += 1
        else:
            fn_is.add(i)
            fns += 1
    fps = len(y_pred_shapes) - len(matched_js)
    fp_js = set(range(len(y_pred_shapes))) - matched_js  # compute which of the predicted shapes are false positives
    # Create masks/vectors of the true positives, false positives, and false negatives
    masks = {
    "tp_true": y_true_shapes.iloc[list(tp_is)],
    "tp_pred": y_pred_shapes.iloc[list(tp_js)],
    "fp_pred": y_pred_shapes.iloc[list(fp_js)],
    "fn_true": y_true_shapes.iloc[list(fn_is)],
    }
    return tps, fps, fns, masks


def display_metrics(eopatch_folder, obj_metrics=False):
    eopatch = EOPatch.load(eopatch_folder)
    metrics_dict = {}
    for sub_name in ['CADASTRE', 'PREDICTED']:
        metrics_dict.update(vector_analyses(eopatch.vector_timeless[sub_name],
                                            sub_name=sub_name))

    if obj_metrics:
        tps, fps, fns, masks = get_object_level_metrics(eopatch.vector_timeless['CADASTRE'],
                                                 eopatch.vector_timeless['PREDICTED'])
        metrics_dict.update({
            "tps_obj": tps,
            "fps_obj": fps,
            "fns_obj": fns,
        })
        # visualization of object vectors for tp, fn, fp, tn
        # visualize_eopatch_obj(eopatch, masks)

    mask_gt = eopatch.mask_timeless['EXTENT_CADASTRE'].squeeze()
    mask_pred = eopatch.mask_timeless['EXTENT_PREDICTED'].squeeze()
    metrics_dict.update(comp_scores(mask_pred=mask_pred, mask_gt=mask_gt))
    return metrics_dict

def get_obj_metrics(metrics_df, flag_data):
    all_tps, all_fps, all_fns = metrics_df[~flag_data][["tps_obj", "fps_obj", "fns_obj"]].sum(axis=0)
    # TODO values close to zero -> eps
    if all_tps + all_fps > 0:
        object_precision = all_tps / (all_tps + all_fps)
    else:
        object_precision = float('nan')

    if all_tps + all_fns > 0:
        object_recall = all_tps / (all_tps + all_fns)
    else:
        object_recall = float('nan')
    metrics_df["object_recall"] = object_recall
    metrics_df["object_precision"] = object_precision
    return metrics_df


def visualize_eopatch_obj(eop, masks):
    fig, axis = plt.subplots(figsize=(15, 10), ncols=5, sharey=True)
    eop.vector_timeless['CADASTRE'].plot(ax=axis[0], color='green', alpha=0.5)
    eop.vector_timeless['PREDICTED'].plot(ax=axis[0], color='red', alpha=0.5)

    for ind, (key, mask) in enumerate(masks.items()):
        mask.plot(ax=axis[ind+1], color='green', alpha=0.5)
        axis[ind+1].grid()
        axis[ind+1].set_title(f"{key}")
    plt.show()


def visualize_eopatch(eop):
    fig, axis = plt.subplots(figsize=(15, 10), ncols=1, sharey=True)
    eop.vector_timeless['CADASTRE'].plot(ax=axis, color='green', alpha=0.5)
    eop.vector_timeless['PREDICTED'].plot(ax=axis, color='red', alpha=0.5)
    plt.show()

    flag_bands = 'BANDS' in eop.data
    # holes in polygons are preserved
    for mask_name in ["EXTENT", "BOUNDARY"]:
        time_idx = 0  # only one tile stamp
        fig, ax = plt.subplots(ncols=4, figsize=(15, 20))
        if flag_bands:
            draw_true_color(ax[0], eop, time_idx=time_idx, factor=3.5 / 10000, feature_name='BANDS', bands=(2, 1, 0),
                            grid=False)
        draw_bbox(ax[0], eop)
        draw_vector_timeless(ax[0], eop, vector_name='CADASTRE', alpha=.3)
        if flag_bands:
            draw_true_color(ax[1], eop, time_idx=time_idx, factor=3.5 / 10000, feature_name='BANDS', bands=(2, 1, 0),
                            grid=False)
        draw_bbox(ax[1], eop)
        draw_mask(ax[1], eop, time_idx=None, feature_name=f'{mask_name}_CADASTRE', alpha=.3)
        if flag_bands:
            draw_true_color(ax[2], eop, time_idx=time_idx, factor=3.5 / 10000, feature_name='BANDS', bands=(2, 1, 0),
                            grid=False)
        draw_bbox(ax[2], eop)
        draw_mask(ax[2], eop, time_idx=None, feature_name=f'{mask_name}_PREDICTED', alpha=.3)
        if flag_bands:
            draw_true_color(ax[3], eop, time_idx=time_idx, factor=3.5 / 10000, feature_name='BANDS', bands=(2, 1, 0),
                            grid=False)
        ax[3].grid()
        plt.show()

    # for mask_name in ["EXTENT", "BOUNDARY"]:
    #     time_idx = 0  # only one tile stemp
    #     fig, ax = plt.subplots(ncols=4, figsize=(15, 20))
    #     draw_true_color(ax[0], eop, time_idx=time_idx, factor=3.5 / 10000, feature_name='BANDS', bands=(2, 1, 0),
    #                     grid=False)
    #     draw_bbox(ax[0], eop)
    #     draw_vector_timeless(ax[0], eop, vector_name='PREDICTED', alpha=.3)
    #
    #     draw_true_color(ax[1], eop, time_idx=time_idx, factor=3.5 / 10000, feature_name='BANDS', bands=(2, 1, 0),
    #                     grid=False)
    #     draw_bbox(ax[1], eop)
    #     draw_mask(ax[1], eop, time_idx=None, feature_name=f'{mask_name}_PREDICTED', alpha=.3)
    #
    #     draw_true_color(ax[2], eop, time_idx=time_idx, factor=3.5 / 10000, feature_name='BANDS', bands=(2, 1, 0),
    #                     grid=False)
    #     draw_bbox(ax[2], eop)
    #     draw_mask(ax[2], eop, time_idx=None, feature_name=f'{mask_name}_PREDICTED', alpha=.3,
    #               data_timeless=True)  # !!!!!!!!! visualize predicted masks before vectorization step
    #     draw_true_color(ax[3], eop, time_idx=time_idx, factor=3.5 / 10000, feature_name='BANDS', bands=(2, 1, 0),
    #                     grid=False)
    #     ax[3].grid()
    #     plt.show()



def main():
    # read GT files for region, and read Tile predicted agr
    # convert back to pixels and compare by grid (partial loading)
    # https://cadastre.data.gouv.fr/data/etalab-cadastre/2024-07-01/geojson/departements/18/
    # contains in parcels buildings too (needs to filter non crop parcels)
    # Luxembourg 2024 (bad weather during March/April > 50% cloud cover)

    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--cadastre_tile_path", type=str, required=True)
    parser.add_argument("-p", "--pred_file_path", type=str, required=True)
    parser.add_argument("-e", "--eopatches_folder", type=str, required=True)
    parser.add_argument("-o", "--metrics_path", type=str, required=True)
    parser.add_argument("--field_num", type=int, required=False, default=10)
    parser.add_argument("--min_field_num_tile", type=int, required=False, default=200)
    parser.add_argument("--obj_metrics", type=bool, required=False, default=True)

    args = parser.parse_args()

    cadastre_tile_path = args.cadastre_tile_path
    pred_file_path = args.pred_file_path
    metrics_path = args.metrics_path
    field_num = args.field_num
    obj_metrics = args.obj_metrics
    # number of fields in the patch for cadastre data to compute accuracy

    cadastre_data = gpd.read_file(cadastre_tile_path)
    if len(cadastre_data) < args.min_field_num_tile:
        LOGGER.info(f"Number of cadastre crop fields for the tile_grid = {len(cadastre_data)} < {args.min_field_num_tile}"
                    f"\n. Terminate the metrics computation with the reason not enough Ground Truth data. ")
        return

    eopatches_path = [f.path for f in os.scandir(
        args.eopatches_folder) if f.is_dir() and f.name.startswith('eopatch')]

    score_list = []
    for eopatch_folder in tqdm(eopatches_path):
        # creates gt and predicted masks from vectors (MOST important method)
        get_masks_pred_gt(eopatch_folder, cadastre_tile_path, pred_file_path)

        if VISUALIZE:  # for all visualizations eopatch image (rgb) bands are needed
            eopatch = EOPatch.load(eopatch_folder)
            visualize_eopatch(eopatch)

        metrics_dict = display_metrics(eopatch_folder, obj_metrics=obj_metrics)
        score_list.append(metrics_dict)

    metrics_df = pd.DataFrame(score_list)

    col_mean = ['CADASTRE_num_pol', 'CADASTRE_max_areas/m^2',
                'CADASTRE_min_areas/m^2', 'CADASTRE_avg_areas/m^2',
                'PREDICTED_num_pol', 'PREDICTED_max_areas/m^2',
                'PREDICTED_min_areas/m^2', 'PREDICTED_avg_areas/m^2',
                'CADASTRE_fields_prc', 'PREDICTED_fields_prc', 'accuracy_score',
                'matthews_corrcoef', 'cohen_kappa_score', 'iou',
                'TN (no field t, no field p)', 'FP (no field t, field p)',
                'FN (field t, no field p)', 'TP (field t, field p)',
                ]

    # in case cadastre data is not available for the whole tile
    flag_data = ((metrics_df['CADASTRE_num_pol'] < field_num) &
                 (metrics_df['PREDICTED_num_pol'] >= field_num))
    LOGGER.info(f"EOpatches number with predicted fields "
                f"but not available cadastre data for comparison "
                f"\n {len(metrics_df[flag_data])} from {len(metrics_df)}")
    # compute mean metrics only for the patches with available cadastre data
    mean = metrics_df[~flag_data][col_mean].mean(axis=0)
    metrics_df["no_CADASTRE_data"] = flag_data
    metrics_df = pd.concat([metrics_df, pd.DataFrame([mean])], axis=0)

    metrics_df["eopatch_folder"] = eopatches_path + ["mean"]
    metrics_df["count_no_CADASTRE_data"] = metrics_df["no_CADASTRE_data"].sum()

    if obj_metrics:
        metrics_df = get_obj_metrics(metrics_df, flag_data)
    # rounding
    col_r_int = ['CADASTRE_num_pol', 'CADASTRE_max_areas/m^2',
                'CADASTRE_min_areas/m^2', 'CADASTRE_avg_areas/m^2',
                'PREDICTED_num_pol', 'PREDICTED_max_areas/m^2',
                'PREDICTED_min_areas/m^2', 'PREDICTED_avg_areas/m^2']
    col_r_float = ['CADASTRE_fields_prc', 'PREDICTED_fields_prc', 'accuracy_score',
                'matthews_corrcoef', 'cohen_kappa_score', 'iou',
                'TN (no field t, no field p)', 'FP (no field t, field p)',
                'FN (field t, no field p)', 'TP (field t, field p)',]
    if obj_metrics:
        col_r_float.extend(['object_recall', 'object_precision'])

    metrics_df[col_r_int] = metrics_df[col_r_int].apply(lambda x: np.round(x))
    metrics_df[col_r_float] = metrics_df[col_r_float].apply(lambda x: np.round(x, 2))
    # create metrics file
    metrics_df.to_csv(metrics_path, index=False)
    LOGGER.info(f"Mean Metrics \n {metrics_df.iloc[-1]}")


if __name__ == "__main__":
    main()
