import os
import geopandas as gpd
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import random
import shapely as shp

import folium
#import folium.plugins
from branca.element import Figure
from IPython.display import display


def convert_bounds(bbox, invert_y=False):
    """
    Helper method for changing bounding box representation to leaflet notation

    ``(lon1, lat1, lon2, lat2) -> ((lat1, lon1), (lat2, lon2))``
    """
    x1, y1, x2, y2 = bbox
    if invert_y:
        y1, y2 = y2, y1
    return ((y1, x1), (y2, x2))


def display_tile(gdf, bounds=None, id_name='Name', tooltip=['Name'], name_g="Tiles",
                 width="400px", height="500px"):
    fig = Figure(width=width, height=height)
    map1 = folium.Map()
    fig.add_child(map1)

    gdf.explore(
        id_name,
        categorical=True,
        tooltip=tooltip,
        popup=True,
        style_kwds=dict(fillOpacity=0.1, width=2),
        name=name_g,
        m=map1,
    )
    if bounds is not None:  # lon, lat
        map1.fit_bounds(bounds=convert_bounds(gdf.unary_union.bounds))
    display(fig)


def preprocess_polygons(data, limit_area=1000):
    # fill in holes, filter small parcels (im meters), crs must be in meters!!!
    data["geometry"] = data.apply(lambda row: shp.geometry.Polygon(row.geometry.exterior), axis=1)
    data = data[data.area > limit_area]
    return data

def match_predicted_validation_fields(fields_val, fields):
    # spatial join - match delineated and validation fields
    # fill in holes, filter too small parcels
    # fields_val, fields = (preprocess_polygons(gt_tile, limit_area=gt_area_ls),
    #                       preprocess_polygons(pred_tile, limit_area=pred_area_ls))

    fields_intersect = fields_val.sjoin(fields, how="left")[["geometry", "index_right"]]
    fields_intersect = fields_intersect.rename(columns={"geometry": "geom_fields_val", "index_right": "idx_fields"})
    fields_intersect = pd.merge(fields_intersect, fields, left_on="idx_fields", right_index=True).reset_index()
    fields_intersect = fields_intersect.rename(columns={"geometry": "geom_fields", "index": "idx_fields_val"})
    fields_intersect = fields_intersect[["idx_fields_val", "geom_fields_val", "idx_fields", "geom_fields"]]

    fields_intersect["fields_val_area"] = fields_intersect.geom_fields_val.area
    fields_intersect["fields_area"] = fields_intersect.geom_fields.area
    return fields_intersect


def categorize_area(area_ha, list_areas=[
    0, 0.1, 0.5, 1.5, 2.5, 5, 12, 25, 50, 100, 1000
]):
    # compute number of parcels less than 0.5ha, 0.5-1, 1-2, 2-5, 5-10, 10-50, 50-100, 100-1000
    if area_ha >= list_areas[-1]:
        return f"{list_areas[-1]}-"
    low_ind, high_ind = 0, len(list_areas) - 1
    while low_ind + 1 < high_ind:
        middle_ind = (low_ind + high_ind) // 2
        if area_ha >= list_areas[middle_ind]:
            low_ind = middle_ind
        else:
            high_ind = middle_ind
    # LOGGER.info(f"------------------- {low_ind} {high_ind} {area_ha}")
    return f"{list_areas[low_ind]}-{list_areas[high_ind]}"

def visualize_missing_stats(fields_val, fields_intersect_ids,
                            tile_id, list_areas, areas_cat, title='validation(cadastre)',
                            plot_field=True,
                            path2save="folder"):
    dict_metrics = {}
    idx_unique, counts = np.unique(fields_intersect_ids, return_counts=True)
    dict_metrics[f"{title}_idx_unique_matched"] = len(idx_unique)
    dict_metrics[f"{title}_max_counts_matched"] = np.max(counts)
    dict_metrics[f"{title}_mean_counts_matched"] = np.mean(counts)

    if len(idx_unique) < len(fields_val):  # area stats
        missing_fields = len(fields_val) - len(idx_unique)
        dict_metrics[f"{title}_num_missing_fields"] = missing_fields
        dict_metrics[f"{title}_missing_fields%"] = int(100*missing_fields/len(fields_val))
        print(f"Warning: For {missing_fields}({int(100*missing_fields/len(fields_val))}%)"
              f" {title} field(s) no overlapping segmented fields were found.")
        missed_fields_val = fields_val[~fields_val.index.isin(idx_unique)]
        missed_fields_val["area"] = np.round(missed_fields_val.geometry.area / 10 ** 4, 2)
        missed_fields_val["area_cat"] = missed_fields_val["area"].apply(lambda row: categorize_area(row,
                                                                                                    list_areas=list_areas))
        fig, ax = plt.subplots(ncols=1, figsize=(8, 6))
        tile_img = (f"{tile_id} Missing Number of polygons belonging\n"
                    f" to the area (in ha) category")
        missed_fields_val["area_cat"].value_counts().plot(kind='barh',
                                                          title=tile_img,
                                                          ax=ax,
                                                          legend=True, fontsize=12,
                                                          alpha=0.5,
                                                          y=areas_cat, color='green',
                                                          label=title)
        plt.savefig(os.path.join(path2save, f"{tile_id}_missing_{title}_area.png"))
        plt.show()

        if plot_field:
            missed_fields_val = missed_fields_val.reset_index()
            display_tile(gdf=missed_fields_val, bounds=None, id_name='index', tooltip=['area'],
                         name_g=f"Visualization of {missing_fields} {title} field(s) with no overlapping segmented fields")
    return dict_metrics

def metrics_computation(fields_intersect, col_max='ov_combined'):
    fields_intersect["geom_fields_val_intersection"] = fields_intersect["geom_fields_val"].intersection(
        fields_intersect["geom_fields"])
    fields_intersect["fields_val_intersection_area"] = fields_intersect["geom_fields_val_intersection"].area
    fields_intersect["geom_fields_val_union"] = fields_intersect["geom_fields_val"].union(
        fields_intersect["geom_fields"])
    fields_intersect["fields_val_union_area"] = fields_intersect["geom_fields_val_union"].area

    # calculate areal overlap measures to find best matching segemented fields
    # determine overlaps relative to validation fields & segmented fields
    # additionally intersection over union (aka Jaccard-Index)
    fields_intersect["ov_val"] = np.round(
        fields_intersect["fields_val_intersection_area"] / fields_intersect["fields_val_area"], 2)
    fields_intersect["ov_seg"] = np.round(
        fields_intersect["fields_val_intersection_area"] / fields_intersect["fields_area"], 2)
    fields_intersect["ov_combined"] = (fields_intersect["ov_seg"] + fields_intersect["ov_val"]) / 2
    fields_intersect["ov_IoU"] = np.round(
        fields_intersect["fields_val_intersection_area"] / fields_intersect["fields_val_union_area"], 2)

    # minus values if pred_area >>> gt_area
    fields_intersect['HDS-0004'] = np.round(1. - np.clip((fields_intersect["fields_val_union_area"] -
                                                          fields_intersect["fields_val_intersection_area"]) /
                                                         fields_intersect["fields_val_area"], 0, 1),
                                            2)


    fields_intersect_m = get_one2one_match(fields_intersect, col_max)

    return fields_intersect, fields_intersect_m


def get_one2one_match(fields_intersect, col_max):
    # filter delineated fields with maximum overlap for each validation field
    idx_max_overlap = fields_intersect.groupby(['idx_fields_val'])[col_max].idxmax()
    # 'ov_combined' not 'fields_val_intersection_area' as in oversegm/undersegm rate calculation rules?
    fields_intersect_m = fields_intersect.iloc[idx_max_overlap].reset_index(drop=True)
    return fields_intersect_m

def points_along_boundary(geom, distance_delta=5):
    # distance_delta=5 meters distance between sample points along the boundary
    distances = np.arange(0, geom.length, distance_delta)
    points = [geom.exterior.interpolate(distance) for distance in distances]
    multipoint = shp.unary_union(points) # ops.
    return multipoint


def mean_absolute_edge_error(source_geom, replica_geom):
    dists = []
    for ref_point in source_geom.geoms:
        near_points = shp.ops.nearest_points(ref_point, replica_geom)
        dists.append(near_points[0].distance(near_points[1]))
    return np.mean(np.array(dists))


def mean_absolute_edge_error_(source_geom, replica_geom):
    dists = [ref_point.distance(replica_geom) for ref_point in source_geom.geoms]
    return np.mean(np.array(dists))

def get_edge_metrics(fields_intersect):
    # calculate edge-based metrics for all fields
    fields_intersect["geom_fields_val_points"] = (fields_intersect["geom_fields_val"].
                                                  apply(lambda row: points_along_boundary(row)))
    # fields_intersect["geom_fields_points"] = fields_intersect["geom_fields"].apply(lambda row: points_along_boundary(row))

    fields_intersect["mae_val"] = (fields_intersect.
                                   apply(lambda row: mean_absolute_edge_error_(row["geom_fields_val_points"],
                                                                                               row["geom_fields"]),
                             axis=1))
    # The Hausdorff distance is the maximum distance between any point on the
    # first set and its nearest point on the second set, and vice-versa
    fields_intersect["hausdorff_dist"] = fields_intersect.geom_fields.hausdorff_distance(
        fields_intersect.geom_fields_val, densify=0.25)
    # frechet_distance
    fields_intersect["frechet_dist"] = fields_intersect.geom_fields.frechet_distance(
        fields_intersect.geom_fields_val, densify=0.25)
    return fields_intersect


def get_area_category(list_areas = [
        0, 0.1, 0.5, 1.5, 2.5, 5, 12, 25, 50, 100, 1000
    ]):
    areas_cat = [(f"{list_areas[low_ind]}-"
                  f"{list_areas[low_ind + 1] if low_ind + 1 < len(list_areas) else ''}") for low_ind in
                 range(len(list_areas))]
    return areas_cat


def compute_all_global_object_metrics(gt_tile, pred_tile, tile_id, folder_save, gt_area_ls=1000, pred_area_ls=900):
    dict_metrics = {"ALL_num_cadastre_pol": len(gt_tile), "ALL_num_predicted_pol": len(pred_tile)}
    # fill in holes, filter too small parcels
    # buffering 10 predicted fields ???
    fields_val, fields = (preprocess_polygons(gt_tile, limit_area=gt_area_ls),
                          preprocess_polygons(pred_tile, limit_area=pred_area_ls))
    # biggest/smallest hole ?
    dict_metrics.update({
        "ALL_num_cadastre_pol_post": len(fields_val), "ALL_num_predicted_pol_post": len(fields)
    })

    fields_intersect = match_predicted_validation_fields(fields_val, fields)

    list_areas = [
        0, 0.5, 1.5, 2.5, 5, 12, 25, 50, 100, 1000
    ]
    areas_cat = get_area_category(list_areas)
    metrics_curr = visualize_missing_stats(fields_val, fields_intersect_ids=fields_intersect["idx_fields_val"],
                            title='validation(cadastre)',
                            tile_id=tile_id, list_areas=list_areas, areas_cat=areas_cat, plot_field=False,
                                           path2save=folder_save)
    dict_metrics.update(metrics_curr)
    metrics_curr = visualize_missing_stats(fields, fields_intersect_ids=fields_intersect["idx_fields"],
                            title='predicted',
                            tile_id=tile_id, list_areas=list_areas, areas_cat=areas_cat, plot_field=False,
                                           path2save=folder_save)
    dict_metrics.update(metrics_curr)

    fields_intersect, fields_intersect_m = metrics_computation(fields_intersect, col_max='ov_combined')
    # over(under)segmentation rate, obj > 0.5 precision, recall global; missed fields; no gt fields; filtered fields
    # segmentation number of matching fields
    # "idx_fields_val", "geom_fields_val", "idx_fields", "geom_fields"
    fields_intersect[["idx_fields_val", "idx_fields"]].to_csv(os.path.join(folder_save, f"fields_intersect_{tile_id}.csv"))

    """under_segm = 1. - max_intersection / pred_polygon.geometry.area # small minus values happen
    over_segm = 1. - max_intersection / gt_polygon.area # small minus values happen
    fields_intersect["ov_val"] = np.round(
        fields_intersect["fields_val_intersection_area"] / fields_intersect["fields_val_area"], 2)
    fields_intersect["ov_seg"] = np.round(
        fields_intersect["fields_val_intersection_area"] / fields_intersect["fields_area"], 2)
    """
    # ov_val - over_segm, ov_seg - under_segm
    fields_intersect_m_ = get_one2one_match(fields_intersect, col_max="fields_val_intersection_area")
    fields_intersect_m_["over_segm_global"] = 1. - np.clip(fields_intersect_m_["ov_val"], 0, 1)
    fields_intersect_m_["under_segm_global"] = 1. - np.clip(fields_intersect_m_["ov_seg"], 0, 1)
    dict_metrics.update(fields_intersect_m_[["over_segm_global",
                                             "under_segm_global"]].mean().round(2).to_dict())
    # object recall, precision
    dict_metrics["TP_obj_IoU>0.5"] = len(fields_intersect_m[fields_intersect_m['ov_IoU'] >= 0.5])
    dict_metrics["precision_obj_IoU>0.5"] = dict_metrics["TP_obj_IoU>0.5"] / len(fields)
    dict_metrics["recall_obj_IoU>0.5"] = dict_metrics["TP_obj_IoU>0.5"] / len(fields_val)

    fields_intersect_hds = fields_intersect[
        fields_intersect['HDS-0004'] >= 0.9]  # crop yield forecast requirements Use Case
    dict_metrics["num_HDS-0004>0.9"] = len(fields_intersect_hds)
    dict_metrics["num_HDS-0004>0.8"] = len(fields_intersect[fields_intersect['HDS-0004'] >= 0.8])
    dict_metrics["HDS-0004>0.9%_from_matched"] = int(100 * len(fields_intersect_hds) /
                                                     fields_intersect['idx_fields_val'].nunique())
    print(f"Number of fields {len(fields_intersect_hds)}"
          f" ({int(100 * len(fields_intersect_hds) / fields_intersect['idx_fields_val'].nunique())}% of all matching val/pred pairs) that follow the rule 'HDS-0004'")
    print(f"Validation area stats that follow the rule 'HDS-0004'\n {fields_intersect_hds.fields_val_area.describe()}")
    fields_intersect['HDS-0004'].hist(bins=10)
    plt.savefig(os.path.join(folder_save, f"{tile_id}_HDS-0004_bins.png"))
    plt.show()

    # edge only if fields_intersect['HDS-0004'] >= 0.9, no sense otherwise
    fields_intersect_hds = get_edge_metrics(fields_intersect_hds)

    # summarise stats for all fields
    dict_metrics.update(fields_intersect_m[['ov_val', 'ov_seg', 'ov_combined',
                                           'ov_IoU', 'HDS-0004']].mean(numeric_only=True).round(2).to_dict())
    dict_metrics.update(fields_intersect_hds[["mae_val",
                                              "frechet_dist",
                                              "hausdorff_dist"]].mean(numeric_only=True).round(2).to_dict())
    print(f"areal stats")
    print(
        fields_intersect_m[['ov_val', 'ov_seg', 'ov_combined', 'ov_IoU', 'HDS-0004']].mean(numeric_only=True).round(2))

    print(f"\nedge-based stats")
    print(
        fields_intersect_hds[["mae_val", "frechet_dist", "hausdorff_dist"]].mean(numeric_only=True).round(2))
    fields_intersect_m['ov_IoU'].hist(bins=10)
    plt.savefig(os.path.join(folder_save, f"{tile_id}_ov_IoU_bins.png"))
    plt.show()

    return dict_metrics


