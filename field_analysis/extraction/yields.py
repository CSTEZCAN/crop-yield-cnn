import os
import time
import traceback

import geopandas
import matplotlib.pyplot as plt
import numpy as np
from osgeo import gdal, ogr

from ..settings import data as data_settings

from .coverage import calculate_shape_coverage

gdal.UseExceptions()


def drop_null_points(test=False):
    """
    Remove any points containing no useful data for the attribute `Mass_Yield` from every exported yield shapefile. This function is performed effectively inplace, as it operates on the files directly.

    Args:

        test: Optional. Whether the function call is for testing or not. Default is False.
    """
    print("Processing")
    start = time.time()

    for shape in [f for f in os.listdir(data_settings.YIELD_SHAPES_DIR) if '.shp' in f]:

        start_shape = time.time()

        shape_path = os.path.join(data_settings.YIELD_SHAPES_DIR, shape)
        gdf = geopandas.read_file(shape_path)
        gdf.dropna(axis=0, subset=['Mass_Yield'], inplace=True)
        gdf.to_file(shape_path)

        delta_shape = time.time() - start_shape
        print("\t{} - Done in {:.0f} m {:.0f} s".format(
            shape, delta_shape // 60, delta_shape % 60))

        if test:
            break

    delta = time.time() - start
    print("Processing done in {:.0f} m {:.0f} s".format(
        delta // 60, delta % 60))


def rasterize_shapefiles(algorithm, short_side, test=False):
    """
    Rasterize yield datasets with inbound algorithm. The generated image preserves the dimensions of the corresponding Block shape.

    The list of applicable algorithms can be found in http://www.gdal.org/gdal_grid.html.

    Args:

        algorithm: The algorithm with which to interpolate the data points. 
        short_side: The length of the shorter side of the generated raster.
        test: Optional. Whether the function call is for testing or not. Default is False.
    """
    print("Processing")
    start = time.time()
    shape_paths = [os.path.join(data_settings.YIELD_SHAPES_DIR, f)
                   for f in os.listdir(data_settings.YIELD_SHAPES_DIR) if '.shp' in f]

    for i, shape_path in enumerate(shape_paths):

        start_grid = time.time()

        target_path = os.path.join(
            data_settings.YIELD_RASTERS_DIR, "grid_{:03d}.tif".format(i))
        print("\tFrom\t{}\n\tTo\t{}".format(shape_path, target_path))

        try:

            shape = ogr.Open(shape_path)
            layer_name = shape.GetLayer(0).GetName()
            x_min, x_max, y_min, y_max = shape.GetLayer(0).GetExtent()

        finally:

            shape = None

        x_to_y = (x_max - x_min) / (y_max - y_min)
        x_ratio = max(1, x_to_y)
        y_ratio = min(1, x_to_y)

        try:

            raster = gdal.Grid(
                destName=target_path,
                srcDS=shape_path,
                format='GTiff',
                noData=0.0,
                width=short_side * x_ratio,
                height=short_side / y_ratio,
                outputType=gdal.GDT_Float32,
                outputSRS='EPSG:4326',
                outputBounds=[x_min, y_max, x_max, y_min],
                zfield='Mass_Yield',
                layers=[layer_name],
                algorithm=algorithm)

            delta_grid = time.time() - start_grid
            print("\t\tDone in {:.0f} m {:.0f} s".format(
                delta_grid // 60, delta_grid % 60))

            arr = raster.ReadAsArray()

            plt.rcParams['figure.figsize'] = 2, 2
            plt.imshow(arr, cmap='gray', vmin=1500, vmax=15000)
            plt.axis('off')
            plt.show()

        finally:

            raster = None

            if test:
                break

    delta = time.time() - start
    print("Processing done in {:.0f} m {:.0f} s".format(
        delta // 60, delta % 60))


def reproject(source_srs, target_srs, resolution, test=False):
    """
    Reproject images from coordinate system to another with desired resolution.

    Args:

        source_srs: Source coordinate system.
        target_srs: Target coordinate system.
        resolution: Resolution in coordinate units per pixel to which the reprojected raster is resampled to.
        test: Optional. Whether the function call is for testing or not. Default is False.
    """
    print("Processing")
    start = time.time()

    grid_rasters = [f for f in os.listdir(data_settings.YIELD_RASTERS_DIR)
                    if 'grid' in f and 'aux.xml' not in f]

    for i, grid_raster in enumerate(grid_rasters):

        start_warp = time.time()

        target_path = os.path.join(data_settings.YIELD_RASTERS_DIR,
                                   "warp_{:03d}.tif".format(i))
        source_path = os.path.join(
            data_settings.YIELD_RASTERS_DIR, grid_raster)

        print("\tFrom\t{}\n\tTo\t{}".format(source_path, target_path))

        try:

            raster = gdal.Warp(
                target_path, source_path,
                srcSRS=source_srs,
                dstSRS=target_srs,
                xRes=resolution,
                yRes=resolution,
                srcNodata=0,
                dstNodata=0,
                multithread=True)

            delta_warp = time.time() - start_warp
            print("\tDone in {:.0f} m {:.0f} s".format(
                delta_warp // 60, delta_warp % 60))

            arr = raster.ReadAsArray()

            plt.rcParams['figure.figsize'] = 2, 2
            plt.imshow(arr, cmap='gray', vmin=1500, vmax=15000)
            plt.axis('off')
            plt.show()

        finally:

            raster = None

            if test:
                break

    delta = time.time() - start
    print("Processing done in {:.0f} m {:.0f} s".format(
        delta // 60, delta % 60))


def shape_to_polygon(block_id):
    """
    Create a copy of a Block's shape geometry as Polygon.

    Args:

        block_id: The ID of the Block to generate the Polygon for.

    Returns:

        A Block's shape-corresponding Polygon.
    """
    try:

        shapes = ogr.Open(data_settings.BLOCK_SHAPEFILE_PATH)
        shape = shapes.ExecuteSQL("SELECT * FROM {} WHERE LOHKO = '{}'".format(
            shapes.GetLayer().GetName(), block_id))
        feature = shape.GetNextFeature()
        geometry = feature.GetGeometryRef()

        return ogr.CreateGeometryFromWkt(geometry.ExportToWkt())

    finally:

        shapes = None


def raster_to_polygon(raster_path):
    """
    Create a boundary polygon for a raster.

    Args:

        raster_path: A path to the raster's file.

    Returns:

        A Polygon of the raster's boundaries.
    """
    try:

        raster = gdal.Open(raster_path)
        x_min, x_step, _, y_max, _, y_step = raster.GetGeoTransform()
        x_max = x_min+x_step*raster.RasterXSize
        y_min = y_max+y_step*raster.RasterYSize

        raster_bounds = ((x_min, y_max),
                         (x_max, y_max),
                         (x_max, y_min),
                         (x_min, y_min))

        ring = ogr.Geometry(ogr.wkbLinearRing)

        for x, y in raster_bounds:
            ring.AddPoint(x, y)

        raster_poly = ogr.Geometry(ogr.wkbPolygon)
        raster_poly.AddGeometry(ring)
        raster_poly.CloseRings()

        return raster_poly

    finally:

        raster = None


def calculate_shape_intersection_ratio(shape_poly, raster_poly):
    """
    Calculate the raster's and shape's intersection and return the area ratios between intersection and shape.

    Args:

        shape_poly: A Polygon derived from a Shapefile.
        raster_poly: A Polygon derived from a raster's boundaries.

    Returns:

        Ratio of overlap between intersection and the Shapefile's polygon.
    """
    intersection = shape_poly.Intersection(raster_poly)

    return intersection.Area() / shape_poly.Area() * 100


def extract_block_shapes(test=False):
    """
    Extract only the Block-wise boundaries from rasterized images by comparing the raster's and block-corresponding shape's intersection and then comparing the intersection's area to the shape's area. If the ratio is almost 1, conclude that a match has been foung between the raster and a distinct Block ID. The pairs of reprojected yield rasters and cut-to-shape rasters are also visualized to allow for manual validation.

    Args:

        test: Optional. Whether the function call is for testing or not. Default is False.
    """
    start = time.time()

    raster_files = [f for f in os.listdir(data_settings.YIELD_RASTERS_DIR)
                    if 'warp' in f and 'aux.xml' not in f]

    for block_id in data_settings.BLOCK_IDS:

        start_block = time.time()
        print("Processing", block_id, end=" - ")
        match_found = False

        for raster_file in raster_files:

            raster_path = os.path.join(
                data_settings.YIELD_RASTERS_DIR, raster_file)
            target_dir = [d for d in data_settings.BLOCK_DATA_DIRS
                          if block_id in d][0]
            target_path = os.path.join(target_dir,
                                       "{}_yield.tif".format(block_id))

            shape_poly = shape_to_polygon(block_id)
            raster_poly = raster_to_polygon(raster_path)
            coverage = calculate_shape_intersection_ratio(
                shape_poly, raster_poly)

            if coverage < 90:

                continue

            print("Found a match with {:.1f}% overlap:".format(coverage))
            match_found = True

            try:

                raster = gdal.Warp(
                    target_path, raster_path,
                    cutlineDSName=data_settings.BLOCK_SHAPEFILE_PATH,
                    cropToCutline=True,
                    cutlineWhere="LOHKO = '{}'".format(
                        block_id),
                    srcNodata=np.nan,
                    dstNodata=np.nan,
                    multithread=True)
                arr = raster.ReadAsArray()

            finally:

                raster = None

            try:

                source_raster = gdal.Open(raster_path)
                source_arr = source_raster.ReadAsArray()

            finally:

                source_raster = None

            delta_block = time.time() - start_block
            print("\tDone in {:.0f} m {:.0f} s".format(
                delta_block // 60, delta_block % 60))

            plt.rcParams['figure.figsize'] = 4, 2
            plt.subplot(121)
            plt.imshow(source_arr)
            plt.axis('off')
            plt.title("source")
            plt.subplot(122)
            plt.imshow(arr)
            plt.axis('off')
            plt.title("target")
            plt.tight_layout()
            plt.show()

            break

        if not match_found:

            print("No sufficiently overlapping match found!")

            try:

                source_raster = gdal.Open(raster_path)
                source_arr = source_raster.ReadAsArray()

            finally:

                source_raster = None

            plt.rcParams['figure.figsize'] = 4, 2
            plt.subplot(121)
            plt.imshow(source_arr)
            plt.axis('off')
            plt.title("source")
            plt.tight_layout()
            plt.show()

        if test:

            if match_found:

                return target_path

            break

    delta = time.time() - start
    print("Processing done in {:.0f} m {:.0f} s".format(
        delta // 60, delta % 60))
