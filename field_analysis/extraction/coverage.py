import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from osgeo import gdal, ogr, osr

from ..settings import data as data_settings


def calculate_shape_coverage(shape_path, raster_path, block_id):
    """
    Calculates the data coverage of a raster within a shape's boundary with true data coverage.

    Args:

        shape_path: The path to the raster-related shapefile.
        raster_path: The path to the source raster.
        block_id: The ID of the field Block in question.

    Returns:

        A value depicting the raster's data coverage ratio for a Block.
    """

    data_coverage = 0.0

    try:

        shape = ogr.Open(shape_path)
        raster = gdal.Open(raster_path)

        shape_name = shape.GetLayer(0).GetName()
        shape_layer = shape.ExecuteSQL("SELECT * FROM {} WHERE LOHKO = '{}'".format(
            shape_name, block_id))
        feature = shape_layer.GetNextFeature()
        geometry = feature.GetGeometryRef()
        ring = geometry.GetGeometryRef(0)

        points_x, points_y = [], []

        for p in range(ring.GetPointCount()):

            lon, lat, z = ring.GetPoint(p)
            points_x.append(lon)
            points_y.append(lat)

        xmin, xmax = min(points_x), max(points_x)
        ymin, ymax = min(points_y), max(points_y)

        gt = raster.GetGeoTransform()
        px_width, px_height = gt[1], gt[5]

        xcount = round((xmax - xmin) / px_width)
        ycount = round((ymax - ymin) / abs(px_height))

        mask = gdal.GetDriverByName('MEM').Create(
            '', xcount, ycount, 1, gdal.GDT_Byte)
        mask.SetGeoTransform(gt)

        raster_srs = osr.SpatialReference()
        raster_srs.ImportFromWkt(raster.GetProjectionRef())
        mask.SetProjection(raster_srs.ExportToWkt())
        gdal.RasterizeLayer(mask, [1], shape_layer, burn_values=[
                            1], options=["GDAL_PAM_ENABLED=NO"])

        raster_data = raster.ReadAsArray()
        mask_data = mask.ReadAsArray()

        raster_data = np.nan_to_num(raster_data)

        mask_data_count = mask_data[mask_data > 0].size
        raster_data_count = sum(raster_data[mask_data > 0] > 0)
        data_coverage = raster_data_count / mask_data_count * 100

    except Exception as ex:

        raise Exception(
            "Error[layer={}, raster={}]: {}".format(shape_layer.GetName(), raster.GetDescription(), ex))

    finally:

        mask = None
        raster = None
        shape = None

    return data_coverage


def calculate_overall_coverage():
    """
    Calculate the data coverage for each existing date entry block wise. If the dataset in question is a drone dataset, the data is automatically fully covering for that certain date. Otherwise the data is Sentinel-based and employs the possibility of cloud coverage. In that case the data coverage, which essentially is data-to-None ratio, is calculated.

    Returns:

        A list of Block and date wise raster coverage ratios.
    """

    coverage_data = []

    print("Calculating overall data coverage")

    for data_dir in data_settings.BLOCK_DATA_DIRS:

        filenames = os.listdir(data_dir)
        block_id = os.path.split(data_dir)[-1]

        for filename in filenames:

            if 'aux' in filename or 'yield' in filename:

                continue

            file_path = os.path.join(data_dir, filename)
            filename_split = filename.split("_")
            date = filename_split[0]

            if 'sentinel_ndvi' not in filename:

                coverage_data.append(
                    [block_id, pd.to_datetime(date), 100])

                continue

            data_coverage = calculate_shape_coverage(
                shape_path=data_settings.BLOCK_SHAPEFILE_PATH,
                raster_path=file_path,
                block_id=block_id)

            coverage_data.append(
                [block_id, pd.to_datetime(date), data_coverage])

            print(".", end="")

    print()
    return coverage_data


def draw_heatmap():
    """
    Visualize the data coverage date wise using a heatmap.
    """
    plt.rcParams['figure.figsize'] = 16, 10

    coverage_data = calculate_overall_coverage()

    print("Drawing the heatmap")

    dates_data_df = pd.DataFrame(coverage_data, columns=[
        'block_id', 'date', 'data_coverage'])

    block_stats = dates_data_df.groupby('block_id').agg([np.size, np.mean])
    dates_linear = pd.date_range(
        dates_data_df.date.min(), dates_data_df.date.max())
    product = [dates_linear.date, pd.unique(
        dates_data_df.block_id).tolist()]

    multi_index = pd.MultiIndex.from_product(
        product, names=['date', 'block_id'])
    dates_linear_df = pd.DataFrame(
        index=multi_index, columns=['data_coverage'])

    for i, row in dates_data_df.iterrows():

        block_id, date, data_coverage = row.values

        if pd.isnull(dates_linear_df.loc[date].loc[block_id, 'data_coverage']):

            dates_linear_df.loc[date].loc[
                block_id, 'data_coverage'] = data_coverage

        else:

            existing_value = dates_linear_df.loc[date].loc[
                block_id, 'data_coverage']
            dates_linear_df.loc[date].loc[block_id, 'data_coverage'] = max(existing_value,
                                                                           data_coverage)

    dates_linear_df.reset_index(inplace=True)
    dates_pivot = dates_linear_df.pivot_table(
        index='block_id', columns='date', values='data_coverage', aggfunc=np.max)

    x_labels = []

    for i, date in enumerate(dates_linear):

        if i % 2 == 0:

            x_labels.append("{:02d}/{:02d}".format(date.day, date.month))

        else:

            x_labels.append("")

    y_labels = []

    for block_id, row in block_stats.iterrows():

        n = row['data_coverage']['size']
        m = row['data_coverage']['mean']
        y_labels.append("{} (n={:.0f}, m={:.2f})".format(block_id, n, m))

    title = "Data coverage of datasets from {} to {} where n: number of images, m: the mean data coverage".format(
        dates_linear[0].date(), dates_linear[-1].date())

    sns.heatmap(dates_pivot, linewidths=0.01, cmap='Spectral',
                xticklabels=x_labels, yticklabels=y_labels).set_title(title)
    plt.show()
