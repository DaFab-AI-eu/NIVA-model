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


def display_tile(gdf, bounds=None, id_name='Name', tooltip=['Name'], name_g="Tiles"):
    fig = Figure(width="400px", height="500px")
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


def match_predicted_validation_fields(gt_tile, pred_tile, gt_area_ls=1000,
                                      pred_area_ls=900):
    # spatial join - match delineated and validation fields
    # fill in holes
    gt_tile["geometry"] = gt_tile.apply(lambda row: shp.geometry.Polygon(row.geometry.exterior), axis=1)
    pred_tile["geometry"] = pred_tile.apply(lambda row: shp.geometry.Polygon(row.geometry.exterior), axis=1)
    # filter too small parcels
    fields_val, fields = gt_tile[gt_tile.area > gt_area_ls], pred_tile[pred_tile.area > pred_area_ls]

    fields_intersect = fields_val.sjoin(fields, how="left")[["geometry", "index_right"]]
    fields_intersect = fields_intersect.rename(columns={"geometry": "geom_fields_val", "index_right": "idx_fields"})
    fields_intersect = pd.merge(fields_intersect, fields, left_on="idx_fields", right_index=True).reset_index()
    fields_intersect = fields_intersect.rename(columns={"geometry": "geom_fields", "index": "idx_fields_val"})
    fields_intersect = fields_intersect[["idx_fields_val", "geom_fields_val", "idx_fields", "geom_fields"]]

    fields_intersect["fields_val_area"] = fields_intersect.geom_fields_val.area
    fields_intersect["fields_area"] = fields_intersect.geom_fields.area
    return fields_intersect, fields_val, fields


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
                            tile_id, list_areas, areas_cat, title='validation/cadastre',
                            plot_field=True):
    idx_unique = fields_intersect_ids.unique()
    if len(idx_unique) < len(fields_val):  # area stats
        missing_fields = len(fields_val) - len(idx_unique)
        print(f"Warning: For {missing_fields}({int(100*missing_fields/len(fields_val))}%)"
              f" {title} field(s) no overlapping segmented fields were found.")
        missed_fields_val = fields_val[~fields_val.index.isin(idx_unique)]
        missed_fields_val["area"] = np.round(missed_fields_val.geometry.area / 10 ** 4, 2)
        missed_fields_val["area_cat"] = missed_fields_val["area"].apply(lambda row: categorize_area(row,
                                                                                                    list_areas=list_areas))
        fig, ax = plt.subplots(ncols=1, figsize=(8, 6))
        missed_fields_val["area_cat"].value_counts().plot(kind='barh',
                                                          title=f"{tile_id} Missing Number of polygons belonging\n"
                                                                f" to the area (in ha) category",
                                                          ax=ax,
                                                          legend=True, fontsize=12,
                                                          alpha=0.5,
                                                          y=areas_cat, color='green',
                                                          label=title)
        plt.show()

        if plot_field:
            missed_fields_val = missed_fields_val.reset_index()
            display_tile(gdf=missed_fields_val, bounds=None, id_name='index', tooltip=['area'],
                         name_g=f"Visualization of {missing_fields} {title} field(s) with no overlapping segmented fields")


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

    # filter delineated fields with maximum overlap for each validation field
    idx_max_overlap = fields_intersect.groupby(['idx_fields_val'])[col_max].idxmax()
    # 'ov_combined' not 'fields_val_intersection_area' as in oversegm/undersegm rate calculation rules?
    fields_intersect_m = fields_intersect.iloc[idx_max_overlap].reset_index(drop=True)

    return fields_intersect, fields_intersect_m


def points_along_boundary(geom, dist=5):
    distance_delta = dist
    distances = np.arange(0, geom.length, distance_delta)
    points = [geom.exterior.interpolate(distance) for distance in distances]
    multipoint = shp.ops.unary_union(points)
    return multipoint


def mean_absolute_edge_error(source_geom, replica_geom):
    dists = []
    for ref_point in source_geom.geoms:
        near_points = shp.ops.nearest_points(ref_point, replica_geom)
        dists.append(near_points[0].distance(near_points[1]))
    return np.mean(np.array(dists))


def get_edge_metrics(fields_intersect):
    # initialise empty lists
    hausdorff_dists = []
    mae_val_dists = []
    mae_seg_dists = []

    # calculate edge-based metrics for all fields
    for index, row in fields_intersect.iterrows(): # TODO redo into apply func
        field_val_geom = shp.geometry.shape(row["geom_fields_val"])
        field_geom = shp.geometry.shape(row["geom_fields"])
        mae_val_dist = mean_absolute_edge_error(
            points_along_boundary(field_val_geom),
            points_along_boundary(field_geom)
        )
        mae_seg_dist = mean_absolute_edge_error(
            points_along_boundary(field_geom),
            points_along_boundary(field_val_geom)
        )
        mae_val_dists.append(mae_val_dist)
        mae_seg_dists.append(mae_seg_dist)
        # The Hausdorff distance is the maximum distance between any point on the
        # first set and its nearest point on the second set, and vice-versa
        hausdorff_dists.append(field_geom.hausdorff_distance(field_val_geom))

    fields_intersect["mae_val"] = mae_val_dists
    fields_intersect["mae_seg"] = mae_seg_dists
    fields_intersect["mae_combined"] = (fields_intersect["mae_val"] + fields_intersect["mae_seg"]) / 2
    fields_intersect["hausdorff_dist"] = hausdorff_dists
    return fields_intersect


def get_area_category(list_areas = [
        0, 0.1, 0.5, 1.5, 2.5, 5, 12, 25, 50, 100, 1000
    ]):
    areas_cat = [(f"{list_areas[low_ind]}-"
                  f"{list_areas[low_ind + 1] if low_ind + 1 < len(list_areas) else ''}") for low_ind in
                 range(len(list_areas))]
    return areas_cat