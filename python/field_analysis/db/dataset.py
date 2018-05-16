import datetime as dt
import os
import pickle
import traceback
from time import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from osgeo import gdal, ogr

from . import models
from ..settings import data as data_settings
from ..settings import model as model_settings
from .models import Area, DataPeriod, Target, db

gdal.UseExceptions()


class DatasetBuilder:
    """
    Build the dataset according to multiple corresponding settings found in package `field_analysis.settings`.
    """

    def __init__(self,  initialize=True, period_length_days=7, area_size_meters=10, area_size_px=32):
        """
        Args:

            initialize: A Boolean for whether to format and overwrite an already existing database file. Defaults to True.
            period_length_days: The number of days to use as the dataset time period. Defaults to 7.
            area_size_meters: The size of the extracted Areas in meters, affects Shapefile-wise extraction. Defaults to 10.
            area_size_px: The size of the extracted Areas in pixels, affects imaga-wise array resizing. Defaults to 32.
        """

        self.period_length_days = period_length_days
        self.area_size_meters = area_size_meters
        self.area_size_px = area_size_px

        self.soil_data = pd.DataFrame()
        self.weather_data = pd.DataFrame()
        self.block_areas = {}

        self.db_path = os.path.join(
            model_settings.DATABASES_DIR,
            "field_analysis_{}m_{}px.db".format(area_size_meters,
                                                area_size_px)
        )

        if initialize and os.path.isfile(self.db_path):
            os.remove(self.db_path)

        models.initialize_db(self.db_path)

    def get_soil_data(self):
        """
        Fetch soil data from the corresponding CSV. The retrieved weather dataset is persisted to the `DatasetBuilder` instance.

        Returns:

            The retrieved soil data as a Pandas DataFrame.
        """
        print("Reading Soil data..", end=" ")

        self.soil_data = pd.read_csv(data_settings.SOIL_PATH,
                                     index_col='block_id',
                                     sep=';',
                                     decimal=",")

        print("OK")

        return self.soil_data

    def get_weather_data(self):
        """
        Fetch weather data from the corresponding CSV and resample the time period to `self.period_length_days`. The retrieved weather dataset is persisted to the `DatasetBuilder` instance.

        Returns:

            The retrieved weather data as a Pandas DataFrame.
        """
        print("Reading Weather data..", end=" ")

        self.weather_data = pd.read_csv(data_settings.WEATHER_PATH,
                                        index_col='Date',
                                        parse_dates=True)

        resample_period = "{}D".format(self.period_length_days)
        self.weather_data = (self.weather_data.resample(resample_period, level=0)
                             .agg([np.mean, np.median, np.sum]))

        print("OK")

        return self.weather_data

    def extract_area_grids(self, show_image=True, test=False):
        """
        Divide the blocks to sliding window grids of Areas with side-wise dimensions corresponding to `self.area_size_meters`. The sliding window uses `settings.SENTINEL_UNIT_LENGTH` as the step size. Only areas with non-null data are considered valid.

        Args:

            show_image: A Boolean for whether to show the accessed Block's shape as an image or not.
            test: Optional. Whether the function call is for testing or not. Default is False.

        Returns:

            The computed coordinates as dictionary of Block-wise lists of Area coordinates.
        """
        print("Extracting Area grids..")
        plt.rcParams['figure.figsize'] = (1, 1)

        shapes = ogr.Open(data_settings.BLOCK_SHAPEFILE_PATH)
        shapes_name = shapes.GetLayer().GetName()

        areas_count = 0

        for block_id in data_settings.BLOCK_IDS:

            print(block_id, end=" ")

            shape = shapes.ExecuteSQL(
                "SELECT * FROM {} WHERE LOHKO = '{}'".format(shapes_name, block_id))
            feature = shape.GetNextFeature()
            geometry = feature.GetGeometryRef()

            x_min, x_max, y_min, y_max = feature.geometry().GetEnvelope()
            x_min, x_max = round(x_min), round(x_max)
            y_min, y_max = round(y_min), round(y_max)

            print("centroid={}".format(
                (round((x_min + x_max) / 2), round((y_min + y_max) / 2))))

            x_delta, y_delta = x_max - x_min, y_max - y_min
            print("\tx_delta={:.1f}\ty_delta={:.1f}".format(x_delta, y_delta))

            # Calculate number sliding window top corners.
            area_cols = int(round((x_delta - self.area_size_meters +
                                   data_settings.SENTINEL_UNIT_LENGTH) / data_settings.SENTINEL_UNIT_LENGTH))
            area_rows = int(round((y_delta - self.area_size_meters +
                                   data_settings.SENTINEL_UNIT_LENGTH) / data_settings.SENTINEL_UNIT_LENGTH))
            n_areas = area_cols * area_rows
            print("\tarea_cols={}\tarea_rows={}".format(area_cols, area_rows))

            # Initialize first area's envelope
            top_corner_x = x_min
            bot_corner_x = x_min + self.area_size_meters
            top_corner_y = y_max
            bot_corner_y = y_max - self.area_size_meters

            coord_grid = []
            block_area = []
            block_areas_count = 0

            for _ in range(area_rows):

                coord_grid_row = []
                block_area_row = []

                for _ in range(area_cols):

                    # Calculate area's center point
                    area_centroid = ogr.Geometry(ogr.wkbPoint)
                    area_centroid.AssignSpatialReference(shape.GetSpatialRef())
                    mean_x = (bot_corner_x + top_corner_x) / 2
                    mean_y = (top_corner_y + bot_corner_y) / 2
                    area_centroid.SetPoint(0, mean_x, mean_y)

                    # Does the centroid fall within the shape?
                    if area_centroid.Within(geometry):

                        block_area_row.append(1)
                        coord_grid_row.append(
                            [(top_corner_x, top_corner_y),
                             (bot_corner_x, bot_corner_y)]
                        )
                        block_areas_count += 1

                    else:

                        block_area_row.append(0)

                    # Move x pointers to next column
                    top_corner_x = top_corner_x + data_settings.SENTINEL_UNIT_LENGTH
                    bot_corner_x = bot_corner_x + data_settings.SENTINEL_UNIT_LENGTH

                coord_grid.append(coord_grid_row)
                block_area.append(block_area_row)

                # Move y pointers to next row
                top_corner_y = top_corner_y - data_settings.SENTINEL_UNIT_LENGTH
                bot_corner_y = bot_corner_y - data_settings.SENTINEL_UNIT_LENGTH

                # Reset x pointers to first column
                top_corner_x = x_min
                bot_corner_x = x_min + self.area_size_meters

            self.block_areas[block_id] = coord_grid

            print("\ttotal_areas={}\tvalid_areas={}".format(
                n_areas, block_areas_count))

            if show_image:

                arr = np.array(block_area)
                plt.imshow(arr)
                plt.xticks([])
                plt.yticks([])
                plt.show()

            areas_count += block_areas_count

            if test:
                break

        print("Total number of areas:", areas_count)
        shapes = None

        return self.block_areas

    def get_files_in_date_range(self, date_range_start, block_file_list):
        """
        Retrieve list of Block-wise files in date range.

        Args:

            date_range_start: The starting date for time period computation.
            block_file_list: A list of file paths.

        Returns:

            A list of files within the computed date range.
        """

        date_range = [date_range_start + dt.timedelta(days=x)
                      for x in range(self.period_length_days)]

        files_in_range = []

        for date in date_range:

            date_file = "{:04d}{:02d}{:02d}".format(date.year,
                                                    date.month,
                                                    date.day)
            [files_in_range.append(f) for f in block_file_list
             if date_file in f and 'aux.xml' not in f]

        return files_in_range

    def get_block_statistics(self, block_dir, file_list):
        """
        Extract mean and median values for block wise images in the `file_list`. Rasters with no data (i.e. full cloud coverage) are skipped.

         Args:

            block_dir: The path to the directory to which the generated yield image is persisted.
            file_list: A list of files to process.

        Returns:

            A dictionary of datatype-wise lists of statistics.
        """
        block_stats = {
            'sentinel_ndvi': [None, None],
            'drone_ndvi': [None, None],
            'drone_rgb': [None, None],
            'yield': [None, None],
        }

        for file in file_list:

            file_path = os.path.join(block_dir, file)

            try:

                block_raster = gdal.Open(file_path)
                arr = block_raster.ReadAsArray().astype('float')

            except Exception as ex:

                traceback.print_exc()
                print("\tError: {}, date={}".format(ex, file_path))
                raise ex

            finally:

                block_raster = None

            arr[arr == 0] = np.nan

            if np.isnan(arr).all():

                continue

            block_raster_mean = np.nanmean(arr)
            block_raster_median = np.nanmedian(arr)

            if 'sentinel_ndvi' in file:

                block_stats['sentinel_ndvi'] = [block_raster_mean,
                                                block_raster_median]

            if 'drone_ndvi' in file:

                block_stats['drone_ndvi'] = [block_raster_mean,
                                             block_raster_median]

            if 'drone_rgb' in file:

                block_stats['drone_rgb'] = [block_raster_mean,
                                            block_raster_median]

            if 'yield' in file:

                block_stats['yield'] = [block_raster_mean,
                                        block_raster_median]

        return block_stats

    def get_area_rasters(self, block_dir, file_list, projection_window):
        """
        Extract area rasters from block wise image datasets within a certain date range. The recommendation is that there should exists only a single item for each datatype for the desired date range, i.e. one datatype per week. If there are multiple items per daterange and datatype, the latest one is used.

        The images are saved as pickled Numpy arrays. Empty Area rasters are skipped.

        For RGB rasters only the first three bands (red, green, blue) are persisted. The rasters are initially in the shape of (bands,x,y), i.e. 4x32x32 corresponging to RGBA channels (A is for alpha). The imaging libraries are able to show the images when the band information is present pixel wise, wherefore the arrays are reshaped to (x,y,bands). Because we want only RGB, the number of selected bands will be a constant of 3.

        Args:

            block_dir: The path to the directory to which the generated yield image is persisted.
            file_list: A list of files to process.
            projection_window: A tuple of coordinate points for the projection area, effectively in the order (corner1_x, corner1_y, corner2_x, corner2_y).

        Returns:

            A dictionary of rasters processed.
        """

        error_message = "Generated Area has incorrect dimensions"
        area_rasters = {
            'sentinel_ndvi': None,
            'drone_ndvi': None,
            'drone_rgb': None,
            'yield': None,
        }

        for file in file_list:

            file_path = os.path.join(block_dir, file)

            try:

                area_raster = gdal.Translate(destName=data_settings.TEMP_PATH,
                                             srcDS=file_path,
                                             projWin=projection_window,
                                             creationOptions=['NUM_THREADS=ALL_CPUS'])
                arr = area_raster.ReadAsArray()

                if len(arr.shape) >= 3:

                    arr = arr[:3].transpose(1, 2, 0)

                if np.isnan(arr).all():

                    continue

                if 'sentinel_ndvi' in file:

                    area_rasters['sentinel_ndvi'] = pickle.dumps(arr)

                arr = arr[:self.area_size_px, :self.area_size_px]

                if 'drone_ndvi' in file:

                    area_rasters['drone_ndvi'] = pickle.dumps(arr)

                if 'yield' in file:

                    area_rasters['yield'] = pickle.dumps(arr)

                if 'drone_rgb' in file:

                    area_rasters['drone_rgb'] = pickle.dumps(arr)

            except Exception as ex:

                traceback.print_exc()
                print("\t{}:, file_path={}, projection_window={}, shape={}".format(
                    ex, file_path, projection_window, arr.shape))
                raise ex

            finally:

                area_raster = None

        return area_rasters

    def generate_areas(self, block_id_only=None):
        """
        Generate Block-wise Areas to the database. The generation process happens inplace, meaning the function alters the database file directly.
        """
        start = time()
        area_count_total = 0

        with db:

            for i, block_id in enumerate(data_settings.BLOCK_IDS):

                if block_id_only is not None and not block_id == block_id_only:
                    continue

                start_block = time()
                print(block_id)
                area_count = 0

                try:

                    for block_row in self.block_areas[block_id]:

                        for block_area in block_row:

                            top_corner, bot_corner = block_area

                            Area.get_or_create(
                                block_id=int(block_id),
                                top_left_x=top_corner[0],
                                top_left_y=top_corner[1],
                                bot_right_x=bot_corner[0],
                                bot_right_y=bot_corner[1]
                            )

                            area_count += 1

                except (AttributeError, KeyError) as ex:

                    traceback.print_exc()
                    print("\tError: {}, block_id={}. Did you remember to run extract_area_grids()? ".format(
                        ex, block_id))
                    raise ex

                area_count_total += area_count

                delta_block = time() - start_block
                print("\tProcessed {} Areas in {:.0f} m {:.1f} s".format(
                    area_count, delta_block // 60, delta_block % 60))

            delta = time() - start
            print("Processed {} Areas in {:.0f} m {:.1f} s".format(
                area_count_total, delta // 60, delta % 60))

    @db.connection_context()
    def generate_dataperiods(self):
        """
        Generate Area-wise DataPeriods to the database. The generation process happens inplace, meaning the function alters the database file directly.
        """
        start = time()

        dataperiod_count_total = 0

        for i, block_id in enumerate(data_settings.BLOCK_IDS):

            start_block = time()

            block_id = int(block_id)
            block_dir = data_settings.BLOCK_DATA_DIRS[i]
            block_dir_files = os.listdir(block_dir)

            print(block_id)

            dataperiod_count = 0

            try:

                block_soil = self.soil_data.loc[block_id]
                block_soil_exists = True

            except (AttributeError, KeyError):

                block_soil_exists = False

            print("\tSoil:", block_soil_exists)

            for db_area in list(Area.select().where(Area.block_id == block_id)):

                projection_window = [
                    db_area.top_left_x, db_area.top_left_y,
                    db_area.bot_right_x, db_area.bot_right_y
                ]

                for date in self.weather_data.index:

                    file_list = self.get_files_in_date_range(
                        date, block_dir_files
                    )

                    block_stats = self.get_block_statistics(
                        block_dir, file_list
                    )

                    block_weather = self.weather_data.loc[date]

                    db_dataperiod, _ = DataPeriod.get_or_create(
                        date=date.to_pydatetime(), area=db_area
                    )

                    area_rasters = self.get_area_rasters(
                        block_dir, file_list, projection_window
                    )

                    db_dataperiod.area_sentinel = area_rasters['sentinel_ndvi']
                    db_dataperiod.block_sentinel_mean = block_stats['sentinel_ndvi'][0]
                    db_dataperiod.block_sentinel_median = block_stats['sentinel_ndvi'][1]

                    db_dataperiod.area_drone_ndvi = area_rasters['drone_ndvi']
                    db_dataperiod.block_drone_ndvi_mean = block_stats['drone_ndvi'][0]
                    db_dataperiod.block_drone_ndvi_median = block_stats['drone_ndvi'][1]

                    db_dataperiod.area_drone_rgb = area_rasters['drone_rgb']
                    db_dataperiod.block_drone_rgb_mean = block_stats['drone_rgb'][0]
                    db_dataperiod.block_drone_rgb_median = block_stats['drone_rgb'][1]

                    if block_soil_exists:

                        db_dataperiod.soil_type_cat = block_soil['soil_type_cat']
                        db_dataperiod.soil_earthiness_cat = block_soil['soil_earthiness_cat']
                        db_dataperiod.soil_conductivity = block_soil['soil_conductivity']
                        db_dataperiod.soil_acidity = block_soil['soil_acidity']
                        db_dataperiod.soil_calcium = block_soil['soil_calcium']
                        db_dataperiod.soil_phosphorus = block_soil['soil_phosphorus']
                        db_dataperiod.soil_potassium = block_soil['soil_potassium']
                        db_dataperiod.soil_magnesium = block_soil['soil_magnesium']
                        db_dataperiod.soil_sulphur = block_soil['soil_sulphur']
                        db_dataperiod.soil_copper = block_soil['soil_copper']
                        db_dataperiod.soil_manganese = block_soil['soil_manganese']
                        db_dataperiod.soil_zinc = block_soil['soil_zinc']
                        db_dataperiod.soil_cec = block_soil['soil_cec']
                        db_dataperiod.soil_cec_ca = block_soil['soil_cec_ca']
                        db_dataperiod.soil_cec_k = block_soil['soil_cec_k']
                        db_dataperiod.soil_cec_mg = block_soil['soil_cec_mg']
                        db_dataperiod.soil_cec_na = block_soil['soil_cec_na']

                    db_dataperiod.weather_air_temperature_mean = block_weather[
                        'Air temperature']['mean']
                    db_dataperiod.weather_cloud_amount_mean = block_weather[
                        'Cloud amount']['mean']
                    db_dataperiod.weather_dewpoint_temperature_mean = block_weather[
                        'Dew-point temperature']['mean']
                    db_dataperiod.weather_gust_speed_mean = block_weather['Gust speed']['mean']
                    db_dataperiod.weather_horizontal_visibility_mean = block_weather[
                        'Horizontal visibility']['mean']
                    db_dataperiod.weather_precipitation_amount_sum = block_weather[
                        'Precipitation amount']['sum']
                    db_dataperiod.weather_precipitation_intensity_mean = block_weather[
                        'Precipitation intensity']['mean']
                    db_dataperiod.weather_pressure_mean = block_weather['Pressure (msl)'][
                        'mean']
                    db_dataperiod.weather_relative_humidity_mean = block_weather[
                        'Relative humidity']['mean']
                    db_dataperiod.weather_wind_direction_median = block_weather[
                        'Wind direction']['median']
                    db_dataperiod.weather_wind_speed_mean = block_weather['Wind speed']['mean']

                    db_dataperiod.save()
                    dataperiod_count += 1

            dataperiod_count_total += dataperiod_count

            delta_block = time() - start_block
            print("\tProcessed {} DataPeriods in {:.0f} m {:.1f} s".format(
                dataperiod_count, delta_block // 60, delta_block % 60))

        delta = time() - start
        print("Processed {} DataPeriods in {:.0f} m {:.1f} s".format(
            dataperiod_count_total, delta // 60, delta % 60))

    @db.connection_context()
    def generate_targets(self):
        """
        Generate Area-wise Targets to the database. The generation process happens inplace, meaning the function alters the database file directly.
        """
        start = time()

        target_count_total = 0

        for i, block_id in enumerate(data_settings.BLOCK_IDS):

            start_block = time()

            block_id = int(block_id)
            block_dir = data_settings.BLOCK_DATA_DIRS[i]
            block_dir_files = os.listdir(block_dir)
            yield_file = [f for f in block_dir_files if 'yield'
                          in f and 'aux.xml' not in f]

            print(block_id)

            if not yield_file:

                print("\tNo yield data")
                continue

            target_count = 0

            block_stats = self.get_block_statistics(block_dir, yield_file)

            for db_area in list(Area.select().where(Area.block_id == block_id)):

                projection_window = [db_area.top_left_x, db_area.top_left_y,
                                     db_area.bot_right_x, db_area.bot_right_y]

                area_rasters = self.get_area_rasters(
                    block_dir, yield_file, projection_window
                )

                Target.get_or_create(
                    area=db_area,
                    area_yield=area_rasters['yield'],
                    block_yield_mean=block_stats['yield'][0],
                    block_yield_median=block_stats['yield'][1]
                )

                target_count += 1

            target_count_total += target_count

            delta_block = time() - start_block
            print("\tProcessed {} Targets in {:.0f} m {:.1f} s".format(
                target_count, delta_block // 60, delta_block % 60))

        delta = time() - start
        print("Processed {} Targets in {:.0f} m {:.1f} s".format(
            target_count_total, delta // 60, delta % 60))

    def build_dataset(self):
        """
        A wrapper function for running the required functions in order to create a complete dataset.
        """

        self.get_soil_data()
        self.get_weather_data()
        self.extract_area_grids()
        self.generate_areas()
        self.generate_dataperiods()
        self.generate_targets()
