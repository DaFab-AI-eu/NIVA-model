#
# Copyright (c) Sinergise, 2019 -- 2021.
#
# This file belongs to subproject "field-delineation" of project NIVA (www.niva4cap.eu).
# All rights reserved.
#
# This source code is licensed under the MIT license found in the LICENSE
# file in the root directory of this source tree.


# based on https://github.com/sentinel-hub/field-delineation/blob/main/fd/vectorisation.py

import copy
import os
import sys
import time
from distutils.dir_util import copy_tree
from functools import partial
from glob import glob
from typing import Tuple, Union

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from lxml import etree
from shapely.geometry import Polygon
from shapely.ops import unary_union
from tqdm.auto import tqdm as tqdm

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from typing import Callable, List, Any, Iterable, Optional

# Add the src directory to the path
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(src_path)

# Import logger
from niva_utils.logger import get_logger  # noqa: E402
LOGGER = get_logger(__name__)

# Load configuration
from niva_utils.config_loader import load_config  # noqa: E402
CONFIG = load_config()

# Constants
VECTORIZE_CONFIG = CONFIG['vectorize_config']


def multiprocess(process_fun: Callable, arguments: Iterable[Any],
                 total: Optional[int] = None, max_workers: int = 4) -> List[Any]:
    """
    Executes multiprocessing with tqdm.
    Parameters
    ----------
    process_fun: A function that processes a single item.
    arguments: Arguments with which te function is called.
    total: Number of iterations to run (for cases of iterators)
    max_workers: Max workers for the process pool executor.

    Returns A list of results.
    -------


    """
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        results = list(tqdm(executor.map(process_fun, arguments), total=total))
    return results


@dataclass
class VectorisationConfig:
    gpkg_path: str  # added new
    shape: Tuple[int, int]
    buffer: Tuple[int, int]
    weights_file: str
    vrt_dir: str
    predictions_dir: str
    contours_dir: str
    max_workers: int = 4
    chunk_size: int = 500
    chunk_overlap: int = 10
    threshold: float = 0.6
    cleanup: bool = True
    skip_existing: bool = True
    rows_merging: bool = True


def average_function(no_data: Union[int, float] = 0, round_output: bool = False) -> str:
    """ A Python function that will be added to VRT and used to calculate weighted average over overlaps

    :param no_data: no data pixel value (default = 0)
    :param round_output: flag to round the output (to 0 decimals). Useful when the final result will be in Int.
    :return: Function (as a string)
    """
    rounding = 'out = np.round(out, 0)' if round_output else ''
    return f"""
import numpy as np

def average(in_ar, out_ar, xoff, yoff, xsize, ysize, raster_xsize, raster_ysize, buf_radius, gt, **kwargs):
    p, w = np.split(np.array(in_ar), 2, axis=0)  # shape of in_ar (2*num_splits, 384/116, 500)
    n_overlaps = np.sum(p!={no_data}, axis=0)
    w_sum = np.sum(w, axis=0, dtype=np.float32) 
    p_sum = np.sum(p, axis=0, dtype=np.float32) 
    weighted = np.sum(p*w, axis=0, dtype=np.float32)
    out = np.where((n_overlaps>1) & (w_sum>0) , weighted/w_sum, p_sum/n_overlaps)
    {rounding}
    out_ar[:] = out

"""


def p_simplify(r, tolerance=2.5):
    """ Helper function to parallelise simplification of geometries """
    return r.geometry.simplify(tolerance)


def p_union(r):
    """ Helper function to parallelise union of geometries """
    return r.l_geom.union(r.r_geom)


def get_weights(shape: Tuple[int, int], buffer: Tuple[int, int], low: float = 0, high: float = 1) -> np.ndarray:
    """ Create weights array

    Function to create a numpy array of dimension, that outputs a linear gradient from low to high from the edges
    to the 2*buffer, and 1 elsewhere.
    """
    weight = np.ones(shape)
    weight[..., :2 * buffer[0]] = np.tile(np.linspace(low, high, 2 * buffer[0]), shape[0]).reshape(
        (shape[0], 2 * buffer[0]))
    weight[..., -2 * buffer[0]:] = np.tile(np.linspace(high, low, 2 * buffer[0]), shape[0]).reshape(
        (shape[0], 2 * buffer[0]))
    weight[:2 * buffer[1], ...] = weight[:2 * buffer[1], ...] * np.repeat(np.linspace(low, high, shape[1]),
                                                                          2 * buffer[1]).reshape(
        (2 * buffer[1], shape[1]))
    weight[-2 * buffer[1]:, ...] = weight[-2 * buffer[1]:, ...] * np.repeat(np.linspace(high, low, 2 * buffer[1]),
                                                                            shape[1]).reshape(
        (2 * buffer[1], shape[1]))
    return weight.astype(np.float32)


def write_vrt(files: List[str], weights_file: str, out_vrt: str, function: Optional[str] = None):
    """ Write virtual raster

    Function that will first build a temp.vrt for the input files, and then modify it for purposes of spatial merging
    of overlaps using the provided function
    """

    if not function:
        function = average_function()

    # build a vrt from list of input files
    # https://gdal.org/en/latest/programs/gdalbuildvrt.html
    # gdal_str = f'gdalbuildvrt temp.vrt -b 1 {" ".join(files)}'
    # overcoming the problem of command line length in windows (limit 8,191) -> save files to one file
    folder, _ = os.path.split(out_vrt)
    files_list_path = os.path.join(folder, "input_file_list.txt")
    with open(files_list_path, "w") as file:
        file.write("\n".join(files))

    gdal_str = f'gdalbuildvrt temp.vrt -b 1 -input_file_list {files_list_path}'
    os.system(gdal_str)

    # fix the vrt
    root = etree.parse('temp.vrt').getroot()
    vrtrasterband = root.find('VRTRasterBand')
    rasterbandchildren = list(vrtrasterband)
    root.remove(vrtrasterband)

    dict_attr = {'dataType': 'Float32', 'band': '1', 'subClass': 'VRTDerivedRasterBand'}
    raster_band_tag = etree.SubElement(root, 'VRTRasterBand', dict_attr)

    # Add childern tags to derivedRasterBand tag
    pix_func_tag = etree.SubElement(raster_band_tag, 'PixelFunctionType')
    pix_func_tag.text = 'average'

    pix_func_tag2 = etree.SubElement(raster_band_tag, 'PixelFunctionLanguage')
    pix_func_tag2.text = 'Python'

    pix_func_code = etree.SubElement(raster_band_tag, 'PixelFunctionCode')
    pix_func_code.text = etree.CDATA(function)

    new_sources = []
    for child in rasterbandchildren:
        if child.tag == 'NoDataValue':
            pass
        else:
            raster_band_tag.append(child)
        if child.tag == 'ComplexSource':
            new_source = copy.deepcopy(child)
            new_source.find('SourceFilename').text = weights_file
            new_source.find('SourceProperties').attrib['DataType'] = 'Float32'
            for nodata in new_source.xpath('//NODATA'):
                nodata.getparent().remove(nodata)
            new_sources.append(new_source)

    for new_source in new_sources:
        raster_band_tag.append(new_source)

    os.remove('temp.vrt')

    with open(out_vrt, 'w') as out:
        out.writelines(etree.tounicode(root, pretty_print=True))


def run_contour(col: int, row: int, size: int, vrt_file: str, threshold: float = 0.6,
                contours_dir: str = '.', cleanup: bool = True, skip_existing: bool = True) -> Tuple[str, bool, str]:
    """ Will create a (small) tiff file over a srcwin (row, col, size, size) and run gdal_contour on it. """

    file = f'merged_{row}_{col}_{size}_{size}'
    if skip_existing and os.path.exists(file):
        return file, True, 'Loaded existing file ...'
    try:
        # https://gdal.org/en/latest/programs/gdal_translate.html
        gdal_str = f'gdal_translate --config GDAL_VRT_ENABLE_PYTHON YES -srcwin {col} {row} {size} {size} {vrt_file} {file}.tiff'
        os.system(gdal_str)
        # https://gdal.org/en/latest/programs/gdal_contour.html
        gdal_str = f'gdal_contour -of gpkg {file}.tiff {contours_dir}/{file}.gpkg -i {threshold} -amin amin -amax amax -p'
        os.system(gdal_str)
        if cleanup:
            os.remove(f'{file}.tiff')
        return f'{contours_dir}/{file}.gpkg', True, None
    except Exception as exc:
        return f'{contours_dir}/{file}.gpkg', False, exc


def runner(arg: List):
    """Function that wraps run_contour to be used with sg_utils.postprocessing"""
    return run_contour(*arg)


def unpack_contours(df_filename: str, threshold: float = 0.6) -> gpd.GeoDataFrame:
    """ Convert multipolygon contour row above given threshold into multiple Polygon rows. """
    df = gpd.read_file(df_filename)
    if len(df) <= 2:
        if len(df[df.amax > threshold]):
            # 'ID', 'amin', 'amax', 'geometry', len == 2
            return gpd.GeoDataFrame(geometry=[geom for geom in df[df.amax > threshold].iloc[0].geometry.geoms],
                                    crs=df.crs)
            # added .geoms for shapely version == 2.0
        else:
            return gpd.GeoDataFrame(geometry=[], crs=df.crs)
    raise ValueError(
        f"gdal_contour dataframe {df_filename} has {len(df)} contours, "
        f"but should have maximal 2 entries (one below and/or one above threshold)!")


def split_intersecting(df: gpd.GeoDataFrame, overlap: Polygon) -> Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """ Find entries that overlap with a given polygon """
    index = df.sindex
    possible_matches_index = list(index.intersection(overlap.bounds))
    possible_matches = df.iloc[possible_matches_index]
    precise_matches = possible_matches.intersects(overlap).index

    if len(precise_matches):
        return df[~df.index.isin(precise_matches)].copy(), df[df.index.isin(precise_matches)].copy()

    return df, gpd.GeoDataFrame(geometry=[], crs=df.crs)


def merge_intersecting(df1: gpd.GeoDataFrame, df2: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """ Merge two dataframes of geometries into one """
    multi = unary_union(list(df1.geometry) + list(df2.geometry))
    if multi.is_empty:
        return gpd.GeoDataFrame(geometry=[], crs=df1.crs)
    if multi.geom_type == 'Polygon':
        return gpd.GeoDataFrame(geometry=[multi], crs=df1.crs)
    return gpd.GeoDataFrame(geometry=[g for g in multi.geoms], crs=df1.crs)


def concat_consecutive(merged: gpd.GeoDataFrame, previous: gpd.GeoDataFrame, current: gpd.GeoDataFrame,
                       current_offset: Tuple[int, int], overlap_size: Tuple[int, int] = (10, 500),
                       direction: Tuple[int, int] = (490, 0), transform=None) -> Tuple[gpd.GeoDataFrame,
gpd.GeoDataFrame]:
    list_dfs = []
    if merged is not None:
        list_dfs = [merged]

    if not (len(previous) or len(current)):
        if merged is not None:
            return merged, gpd.GeoDataFrame(geometry=[], crs=merged.crs)
        else:
            return merged, gpd.GeoDataFrame(geometry=[])

    x, y = current_offset
    a, b = overlap_size
    overlap_poly = Polygon.from_bounds(*(transform * (x, y)), *(transform * (x + a, y + b)))

    if len(previous) == 0:
        return merged, current

    if len(current) == 0:
        merged = gpd.GeoDataFrame(pd.concat([merged, previous]), crs=previous.crs)
        return merged, gpd.GeoDataFrame(geometry=[], crs=merged.crs)

    previous_non, previous_int = split_intersecting(previous, overlap_poly)
    current_non, current_int = split_intersecting(current, overlap_poly)
    intersecting = merge_intersecting(previous_int, current_int)
    if len(intersecting):
        # check if intersecting "touches" the "right edge", if so, add it to current_non
        x = x + direction[0]
        y = y + direction[1]
        overlap_poly_end = Polygon.from_bounds(*(transform * (x, y)), *(transform * (x + a, y + b)))
        intersecting_ok, intersecting_next = split_intersecting(intersecting, overlap_poly_end)
        merged = gpd.GeoDataFrame(pd.concat(list_dfs + [previous_non, intersecting_ok]), crs=previous.crs)
        intersecting_next = gpd.GeoDataFrame(pd.concat([intersecting_next, current_non]), crs=previous.crs)
        return merged, intersecting_next

    return gpd.GeoDataFrame(pd.concat(list_dfs + [previous_non]), crs=previous.crs), current_non


def _process_row(row: int, vrt_file: str, vrt_dim: Tuple, contours_dir: str = '.', size: int = 500, buff: int = 10,
                 threshold: float = 0.6, cleanup: bool = True, transform=None, skip_existing: bool = True) \
        -> Tuple[str, bool, str]:
    merged_file = f'{contours_dir}/merged_row_{row}.gpkg'
    if skip_existing and os.path.exists(merged_file):
        return merged_file, True, 'Loaded existing file ...'
    try:
        col = 0
        merged = None
        prev_name, finished, exc = run_contour(col, row, size, vrt_file, threshold, contours_dir, cleanup,
                                               skip_existing)
        if not finished:
            return merged_file, finished, exc
        prev = unpack_contours(prev_name, threshold=threshold)

        if cleanup:
            os.remove(prev_name)
        while col <= (vrt_dim[0] - size):
            col = col + size - buff
            offset = col, row
            cur_name, finished, exc = run_contour(col, row, size, vrt_file, threshold, contours_dir, cleanup,
                                                  skip_existing)
            if not finished:
                return merged_file, finished, exc
            cur = unpack_contours(cur_name, threshold=threshold)
            merged, prev = concat_consecutive(merged, prev, cur, offset, (buff, size), (size - buff, 0), transform)
            if cleanup:
                os.remove(cur_name)
        merged = gpd.GeoDataFrame(pd.concat([merged, prev]), crs=prev.crs)

        merged.to_file(merged_file, driver='GPKG', index=False)  # upgrade geopandas-1.0.1, to make version compatible
        return merged_file, True, None
    except Exception as exc:
        return merged_file, False, exc


def merge_rows(rows: List[str], vrt_file: str, size: int = 500, buffer: int = 10) -> gpd.GeoDataFrame:
    with rasterio.open(vrt_file) as src:
        meta = src.meta
        vrt_dim = meta['width'], meta['height']
        transform = meta['transform']

    merged = None
    prev_name = rows[0]
    prev = gpd.read_file(prev_name)
    for ridx, cur_name in tqdm(enumerate(rows[1:], start=1), total=len(rows) - 1):
        cur = gpd.read_file(cur_name)
        merged, prev = concat_consecutive(merged, prev, cur, (0, ridx * (size - buffer)), (vrt_dim[0], buffer),
                                          (0, size - buffer), transform)
    merged = gpd.GeoDataFrame(pd.concat([merged, prev]), crs=prev.crs)

    return merged


def process_rows(vrt_file: str, contours_dir: str = '.', size: int = 500, buffer: int = 10,
                 threshold: float = 0.6, cleanup: bool = True, skip_existing: bool = True,
                 max_workers: int = 4) -> Tuple[str, bool, str]:
    with rasterio.open(vrt_file) as src:
        meta = src.meta
        vrt_dim = meta['width'], meta['height']
        transform = meta['transform']

    partial_process_row = partial(_process_row, vrt_file=vrt_file, vrt_dim=vrt_dim, contours_dir=contours_dir,
                                  size=size, buff=buffer, threshold=threshold, cleanup=cleanup,
                                  skip_existing=skip_existing,
                                  transform=transform)

    rows = list(range(0, vrt_dim[1], size - buffer))

    return multiprocess(partial_process_row, rows, max_workers=max_workers)


def merging_rows(row_dict: dict, skip_existing: bool = True) -> str:
    """ merge row files into a single file per utm """
    start = time.time()
    # merged_contours_file = f'{row_dict["contours_dir"]}/merged_{row_dict["time_interval"]}_{row_dict["utm"]}.gpkg'
    merged_contours_file = row_dict['gpkg_path']
    if skip_existing and os.path.exists(merged_contours_file):
        return merged_contours_file

    # add logger
    LOGGER.info(f'Merging rows {row_dict["rows"]}')
    merged = merge_rows(rows=row_dict['rows'], vrt_file=row_dict['vrt_file'],
                        size=row_dict['chunk_size'], buffer=row_dict['chunk_overlap'])
    merged.to_file(merged_contours_file, driver='GPKG')

    LOGGER.info(f'Merging rows and writing results for {row_dict["time_interval"]}/{row_dict["utm"]} done'
                f' in {(time.time() - start) / 60} min!\n\n')
    return merged_contours_file


def run_vectorisation(config: VectorisationConfig) -> List[str]:
    """ Run vectorisation process on entire AOI for the given time intervals """
    LOGGER.info(f'Move files to new folders')

    utm_dir = f'{config.predictions_dir}'
    os.makedirs(utm_dir, exist_ok=True)

    LOGGER.info(f'Create weights file {config.weights_file}')
    with rasterio.open(config.weights_file, 'w', driver='gTIFF', width=config.shape[0], height=config.shape[1], count=1,
                       dtype=np.float32) as dst:
        dst.write_band(1, get_weights(config.shape, config.buffer))

    rows = []
    # TODO delete time_interval and utm as we work with only one tile -> one utm, time_interval
    time_interval = 0
    utm = "32631"
    start = time.time()
    LOGGER.info(f'Running contours for {time_interval}/{utm}!')

    contours_dir = f'{config.contours_dir}/'
    LOGGER.info(f'Create contour folder {contours_dir}')
    os.makedirs(contours_dir, exist_ok=True)

    predictions_dir = f'{config.predictions_dir}/'
    tifs = glob(f'{predictions_dir}*.tiff')
    output_vrt = f'{config.vrt_dir}/vrt_{time_interval}_{utm}.vrt'
    write_vrt(tifs, config.weights_file, output_vrt)

    results = process_rows(output_vrt, contours_dir,
                           max_workers=config.max_workers,
                           size=config.chunk_size,
                           buffer=config.chunk_overlap,
                           threshold=config.threshold,
                           cleanup=config.cleanup,
                           skip_existing=config.skip_existing)

    failed = [(file, excp) for file, finished, excp in results if not finished]
    if len(failed):
        LOGGER.warning('Some rows failed:')
        LOGGER.warning('\n'.join([f'{file}: {excp}' for file, excp in failed]))
        # raise Exception(f'{len(failed)} rows failed! ')
        LOGGER.warning(f'{len(failed)} rows failed! ')

    rows.append({'time_interval': time_interval,
                 'utm': utm,
                 'vrt_file': output_vrt,
                 'rows': [file for file, finished, _ in results if finished],
                 'chunk_size': config.chunk_size,
                 'chunk_overlap': config.chunk_overlap,
                 'contours_dir': config.contours_dir,
                 'gpkg_path': config.gpkg_path,
                 })

    LOGGER.info(
        f'Row contours processing for {time_interval}/{utm} done in {(time.time() - start) / 60} min!\n\n')

    # TODO don't need multiprocessing as only one row in rows
    list_of_merged_files = multiprocess(merging_rows, rows, max_workers=config.max_workers)

    return list_of_merged_files


def main_vectorisation(GPKG_FILE_PATH, PROJECT_DATA_ROOT, PREDICTIONS_DIR, CONTOURS_DIR):
    """ Main function to perform vectorisation on the predictions """

    # Complete the VECTORIZE_CONFIG dictionary with necessary paths
    VECTORIZE_CONFIG["weights_file"] = os.path.join(PROJECT_DATA_ROOT, "weights.tiff")
    VECTORIZE_CONFIG["vrt_dir"] = PROJECT_DATA_ROOT
    VECTORIZE_CONFIG["predictions_dir"] = PREDICTIONS_DIR
    VECTORIZE_CONFIG["contours_dir"] = CONTOURS_DIR
    VECTORIZE_CONFIG["gpkg_path"] = GPKG_FILE_PATH

    # Create VectorisationConfig object
    vector_config = VectorisationConfig(
        gpkg_path=VECTORIZE_CONFIG["gpkg_path"],
        shape=tuple(VECTORIZE_CONFIG['shape']),
        buffer=tuple(VECTORIZE_CONFIG['buffer']),
        weights_file=VECTORIZE_CONFIG['weights_file'],
        vrt_dir=VECTORIZE_CONFIG['vrt_dir'],
        predictions_dir=VECTORIZE_CONFIG['predictions_dir'],
        contours_dir=VECTORIZE_CONFIG['contours_dir'],
        max_workers=VECTORIZE_CONFIG['max_workers'],
        chunk_size=VECTORIZE_CONFIG['chunk_size'],
        chunk_overlap=VECTORIZE_CONFIG['chunk_overlap'],
        threshold=VECTORIZE_CONFIG['threshold'],
        cleanup=VECTORIZE_CONFIG['cleanup'],
        skip_existing=VECTORIZE_CONFIG['skip_existing'],
        rows_merging=VECTORIZE_CONFIG['rows_merging']
    )
    results = run_vectorisation(config=vector_config)
    LOGGER.info(f'Vectorisation completed for {len(results)} files')


if __name__ == "__main__":
    # Constants
    PROJECT_DATA_ROOT = CONFIG['niva_project_data_root_inf']
    TILE_ID = CONFIG['TILE_ID']
    # Inferred constants
    PROJECT_DATA_ROOT = os.path.join(PROJECT_DATA_ROOT, TILE_ID)
    # the folders that will be created during the pipeline run
    EOPATCHES_FOLDER = os.path.join(PROJECT_DATA_ROOT, "eopatches")
    PREDICTIONS_DIR = os.path.join(PROJECT_DATA_ROOT, "predictions")
    CONTOURS_DIR = os.path.join(PROJECT_DATA_ROOT, "contours")
    GPKG_FILE_PATH = os.path.join(CONTOURS_DIR, f"{TILE_ID}.gpkg")
    main_vectorisation(GPKG_FILE_PATH, PROJECT_DATA_ROOT, PREDICTIONS_DIR, CONTOURS_DIR)
