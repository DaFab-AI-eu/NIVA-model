import argparse
import json
import os
import sys
import pandas as pd
import geopandas as gpd
import shapely
from matplotlib import pyplot as plt
import numpy as np
from tqdm import tqdm
from sklearn.metrics import matthews_corrcoef, cohen_kappa_score, confusion_matrix, accuracy_score


import fs.move  # required by eopatch.save
from eolearn.core import EOPatch


from utils_plot import visualize_eopatch, display_cm, visualize_eopatch_obj
from transform_vector2mask import get_masks_pred_gt

# Add the src directory to the path
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(src_path)

# Import logger
from niva_utils.logger import get_logger  # noqa: E402
LOGGER = get_logger(__name__)


VISUALIZE = False



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
    metrics_dict["Accuracy_pixel"] = accuracy_score(mask_gt, mask_pred)
    # TODO when there is no field pred and gt matthews_corrcoef should be 1
    metrics_dict["MCC_pixel"] = matthews_corrcoef(mask_gt, mask_pred)
    # TODO when there is no field pred and gt cohen_kappa_score should be 1
    metrics_dict["Kappa_pixel"] = cohen_kappa_score(mask_gt, mask_pred)
    cm, iou = compute_iou(y_pred=mask_pred, y_true=mask_gt, cm_display=VISUALIZE)
    metrics_dict["IoU_pixel"] = iou
    cm_normalized = cm / cm.sum()
    score_names = ["TN (no field t, no field p)", "FP (no field t, field p)",
                   "FN (field t, no field p)", "TP (field t, field p)"]
    metrics_dict.update(dict(zip(score_names, cm_normalized.flatten().tolist())))
    return metrics_dict


def vector_analyses(vector, sub_name='CADASTRE'):
    areas = vector.area # in square meters
    metrics_dict = {
        f"{sub_name}_num_pol": len(vector),
        f"{sub_name}_max_areas/m^2": areas.max(),
        f"{sub_name}_min_areas/m^2": areas.min(),
        f"{sub_name}_avg_areas/m^2": areas.mean(),
    }
    return metrics_dict

def over_under_segmentation_metrics(gt_polygon, pred_polygons):
    """
    Compute over-segmentation (0 is perfect score, 1 high degree of over-segm)
    and under-segmentation following the logic
    First mentioning of metrics
    https://www.researchgate.net/publication/224602240_A_Novel_Protocol_for_Accuracy_Assessment_in_Classification_of_Very_High_Resolution_Images
    ResUnet-a (https://arxiv.org/pdf/1910.12023)
    niva code (https://github.com/sentinel-hub/eo-flow/blob/70a426fe3ab07d8ec096e06af4db7e445af1e740/eoflow/models/metrics.py#L169)

        Args:
            gt_polygon (shapely): one reference geometry polygon
            pred_polygons (geopandas): several predicted polygons that intersect with reference one
        Returns
            (over_segmentation, under_segmentation)
    """
    intersections = gt_polygon.intersection(pred_polygons)
    pred_polygon = pred_polygons.iloc[intersections.area.argmax()]
    max_intersection = intersections.area.max()
    under_segm = 1. - max_intersection / pred_polygon.geometry.area # small minus values happen
    over_segm = 1. - max_intersection / gt_polygon.area # small minus values happen

    return np.clip(over_segm, 0, 1), np.clip(under_segm, 0, 1)



def total_general_metrics(cadastre_tile_path, pred_file_path, tile_meta_path):
    metrics_dict = {}
    total_tile_area = (10980 * 10) ** 2 # in meters not considering missing values

    general_meta_col = ['eo:cloud_cover',
                        's2:nodata_pixel_percentage',
                        's2:vegetation_percentage',
                        's2:not_vegetated_percentage',
                        's2:water_percentage',
                        's2:high_proba_clouds_percentage',
                        's2:medium_proba_clouds_percentage',
                        's2:thin_cirrus_percentage',
                        's2:cloud_shadow_percentage',]

    with open(tile_meta_path, "r") as fp:
        boundaries = gpd.GeoDataFrame.from_features([json.load(fp)], crs="epsg:4326")
    epsg = boundaries["proj:epsg"][0]
    boundaries = boundaries.to_crs(epsg=epsg)
    total_tile_area_cur = boundaries.area.sum()

    LOGGER.info(f"Data {tile_meta_path} crs {boundaries.crs} {boundaries.crs.axis_info[0].unit_name}")

    # area histogram
    fig, ax = plt.subplots(ncols=2, figsize=(15, 20))
    tile_id = os.path.splitext(os.path.split(tile_meta_path)[1])[0]

    for ind, (feature_name, vector_file_path) in enumerate(zip(["CADASTRE", "PREDICTED"],
                                              [cadastre_tile_path,
                                               pred_file_path])):
        data = gpd.read_file(vector_file_path)
        data = data.to_crs(epsg=epsg)
        LOGGER.info(f"Data {vector_file_path} crs {data.crs} {data.crs.axis_info[0].unit_name}")
        metrics_dict.update(vector_analyses(data, sub_name=f'Total_{feature_name}'))
        metrics_dict[f'Total_{feature_name}_area'] = data.area.sum()
        metrics_dict[f'Total_{feature_name}_prc'] = 100 * metrics_dict[f'Total_{feature_name}_area'] / total_tile_area
        metrics_dict[f'Total_{feature_name}_prc_cur'] = 100 * metrics_dict[f'Total_{feature_name}_area'] / total_tile_area_cur

        data["area_ha"] = data.area / 10**4
    #     data[["area_ha"]].plot(kind='kde', title =f"{tile_id} {feature_name} area/ha "
    #                                               f"{int(data['area_ha'].min())}"
    #                                               f"{int(data['area_ha'].mean())}"
    #                                               f" {int(data['area_ha'].max())}", ax=ax[ind], legend=True, fontsize=12)
    # plt.show()

    # add general metadata of the tile
    metrics_dict.update(boundaries[general_meta_col].iloc[0].to_dict())
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

    over_segm, under_segm, num_cor = 0, 0, 0

    # takes long to compute
    for i, y_true_shape in enumerate(y_true_shapes.geometry):
        matching_j = None
        if not shapely.is_valid(y_true_shape):
            continue

        y_intersect = y_pred_shapes[y_true_shape.intersects(y_pred_shapes.geometry)] # shape 0, 1, 2, 3

        if len(y_intersect) >= 1:
            # compute over(under)-segmentation
            over_segm_cur, under_segm_cur = over_under_segmentation_metrics(gt_polygon=y_true_shape,
                                                                            pred_polygons=y_pred_shapes)
            over_segm += over_segm_cur
            under_segm += under_segm_cur
            num_cor += 1

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

    if num_cor: # in case no gt / reference data
        over_segm /= num_cor
        under_segm /= num_cor
    else:
        over_segm, under_segm = np.nan, np.nan

    masks = {
    "tp_true": y_true_shapes.iloc[list(tp_is)],
    "tp_pred": y_pred_shapes.iloc[list(tp_js)],
    "fp_pred": y_pred_shapes.iloc[list(fp_js)],
    "fn_true": y_true_shapes.iloc[list(fn_is)],
    }
    metrics_dict = {
        "tps_obj": tps,
        "fps_obj": fps,
        "fns_obj": fns,
        "over_segm": over_segm,
        "under_segm": under_segm,
    }
    return metrics_dict, masks


def display_metrics(eopatch_folder, obj_metrics=False):
    eopatch = EOPatch.load(eopatch_folder)
    metrics_dict = {}
    for sub_name in ['CADASTRE', 'PREDICTED']:
        metrics_dict.update(vector_analyses(eopatch.vector_timeless[sub_name],
                                            sub_name=sub_name))

    if obj_metrics:
        metrics_dict_c, masks = get_object_level_metrics(eopatch.vector_timeless['CADASTRE'],
                                                 eopatch.vector_timeless['PREDICTED'])
        metrics_dict.update(metrics_dict_c)
        # visualization of object vectors for tp, fn, fp, tn
        # visualize_eopatch_obj(eopatch, masks)

    mask_gt = eopatch.mask_timeless['EXTENT_CADASTRE'].squeeze()
    mask_pred = eopatch.mask_timeless['EXTENT_PREDICTED'].squeeze()
    metrics_dict.update(comp_scores(mask_pred=mask_pred, mask_gt=mask_gt))
    return metrics_dict

def get_obj_metrics(metrics_df, flag_data):
    (all_tps, all_fps, all_fns) = metrics_df[~flag_data][["tps_obj", "fps_obj", "fns_obj",
                                                      ]].sum(axis=0)

    if all_tps + all_fps > 0:
        object_precision = all_tps / (all_tps + all_fps)
    else:
        object_precision = float('nan')

    if all_tps + all_fns > 0:
        object_recall = all_tps / (all_tps + all_fns)
    else:
        object_recall = float('nan')
    metrics_df["Recall_object"] = object_recall
    metrics_df["Precision_object"] = object_precision

    (metrics_df["Oversegmentation"],
     metrics_df["Undersegmentation"]) = metrics_df[~flag_data][["over_segm", "under_segm"]].nanmean(axis=0)
    return metrics_df


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
    parser.add_argument("-m", "--tile_meta_path", type=str, required=True)
    parser.add_argument("--field_num", type=int, required=False, default=10)
    parser.add_argument("--min_field_num_tile", type=int, required=False, default=200)
    parser.add_argument("--obj_metrics", type=bool, required=False, default=True)

    args = parser.parse_args()

    cadastre_tile_path = args.cadastre_tile_path
    pred_file_path = args.pred_file_path
    metrics_path = args.metrics_path
    field_num = args.field_num
    obj_metrics = args.obj_metrics
    tile_meta_path = args.tile_meta_path
    # number of fields in the patch for cadastre data to compute accuracy

    cadastre_data = gpd.read_file(cadastre_tile_path)
    if len(cadastre_data) < args.min_field_num_tile:
        LOGGER.info(f"Number of cadastre crop fields for the tile_grid = {len(cadastre_data)} < {args.min_field_num_tile}"
                    f"\n. Terminate the metrics computation with the reason not enough Ground Truth data. ")
        return

    eopatches_path = [f.path for f in os.scandir(
        args.eopatches_folder) if f.is_dir() and f.name.startswith('eopatch')]

    total_metrics_dict = total_general_metrics(cadastre_tile_path, pred_file_path, tile_meta_path)
    LOGGER.info(total_metrics_dict) # TODO resolve for area visualization (log transform)

    score_list = []
    for eopatch_folder in tqdm(eopatches_path):
        # creates gt and predicted masks from vectors (MOST important method)
        # get_masks_pred_gt(eopatch_folder, cadastre_tile_path, pred_file_path)
        VISUALIZE = False
        if VISUALIZE:  # for all visualizations eopatch image (rgb) bands are needed
            eopatch = EOPatch.load(eopatch_folder)
            visualize_eopatch(eopatch, eopatch_folder=eopatch_folder)

        metrics_dict = display_metrics(eopatch_folder, obj_metrics=obj_metrics)
        score_list.append(metrics_dict)
        LOGGER.info(metrics_dict)


    metrics_df = pd.DataFrame(score_list)

    col_mean = ['CADASTRE_num_pol', 'CADASTRE_max_areas/m^2',
                'CADASTRE_min_areas/m^2', 'CADASTRE_avg_areas/m^2',
                'PREDICTED_num_pol', 'PREDICTED_max_areas/m^2',
                'PREDICTED_min_areas/m^2', 'PREDICTED_avg_areas/m^2',
                'CADASTRE_fields_prc', 'PREDICTED_fields_prc', 'Accuracy_pixel',
                'MCC_pixel', 'Kappa_pixel', 'IoU_pixel',
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

    # additional metrics
    eps = 1e-6
    metrics_df["Precision_pixel"] = metrics_df['TP (field t, field p)'] / (
                metrics_df['TP (field t, field p)'] + metrics_df['FP (no field t, field p)'] + eps)
    metrics_df["Recall_pixel"] = metrics_df['TP (field t, field p)'] / (
                metrics_df['TP (field t, field p)'] + metrics_df['FN (field t, no field p)'] + eps)
    metrics_df["F1_pixel"] = ((2 * metrics_df["Precision_pixel"] * metrics_df["Recall_pixel"]) /
                              (metrics_df["Precision_pixel"] + metrics_df["Recall_pixel"] + eps))

    # general metrics
    for key, val in total_metrics_dict.items():
        metrics_df[key] = val

    if obj_metrics:
        metrics_df = get_obj_metrics(metrics_df, flag_data)
    # rounding
    col_r_int = ['CADASTRE_num_pol', 'CADASTRE_max_areas/m^2',
                'CADASTRE_min_areas/m^2', 'CADASTRE_avg_areas/m^2',
                'PREDICTED_num_pol', 'PREDICTED_max_areas/m^2',
                'PREDICTED_min_areas/m^2', 'PREDICTED_avg_areas/m^2',

                 'Total_CADASTRE_num_pol', 'Total_CADASTRE_max_areas/m^2',
                 'Total_CADASTRE_min_areas/m^2',
                 'Total_CADASTRE_avg_areas/m^2', 'Total_CADASTRE_prc_cur',
                 'Total_PREDICTED_num_pol',
                 'Total_PREDICTED_max_areas/m^2', 'Total_PREDICTED_min_areas/m^2',
                 'Total_PREDICTED_avg_areas/m^2', 'Total_PREDICTED_area',]

    col_r_float = ['CADASTRE_fields_prc', 'PREDICTED_fields_prc', 'Accuracy_pixel',
                    'MCC_pixel', 'Kappa_pixel', 'IoU_pixel',
                    'TN (no field t, no field p)', 'FP (no field t, field p)',
                    'FN (field t, no field p)', 'TP (field t, field p)',
                    'Precision_pixel', 'Recall_pixel', 'F1_pixel',

                    'Total_CADASTRE_area', 'Total_CADASTRE_prc',
                    'Total_PREDICTED_prc', 'Total_PREDICTED_prc_cur',
                    'eo:cloud_cover', 's2:nodata_pixel_percentage', 's2:vegetation_percentage',
                    's2:not_vegetated_percentage', 's2:water_percentage',
                    's2:high_proba_clouds_percentage',
                    's2:medium_proba_clouds_percentage', 's2:thin_cirrus_percentage',
                    's2:cloud_shadow_percentage']


    if obj_metrics:
        col_r_float.extend(['Recall_object', 'Precision_object',
                            'Oversegmentation', 'Undersegmentation'])

    metrics_df[col_r_int] = metrics_df[col_r_int].apply(lambda x: np.round(x))
    metrics_df[col_r_float] = metrics_df[col_r_float].apply(lambda x: np.round(x, 2))
    # create metrics file
    metrics_df.to_csv(metrics_path, index=False)
    LOGGER.info(f"Mean Metrics \n {metrics_df.iloc[-1]}")


if __name__ == "__main__":
    main()
