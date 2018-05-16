import os
import time

import matplotlib.pyplot as plt
import numpy as np
from osgeo import gdal

from ..settings import data as data_settings

gdal.UseExceptions()


class BlockExtractor:
    """
    A base class for extracting resampled and rescaled Block-wise datasets from distinct source images.
    """

    def __init__(self, target_resolution, source_range, dest_range, source_nodata, source_paths, block_id=None):
        """
        Args:

            target_resolution: The resolution in coordinate units per pixels to which the extracted images are to resampled.
            source_range: The absolute data range of source data.
            dest_range: The absolute data range of output data.
            source_nodata: The `NoData` value of the source data.
            source_paths: A list of source file paths.
            block_id: Optional. Perform operations only for a single Block if provided.
        """

        self.target_resolution = target_resolution
        self.source_range = source_range
        self.dest_range = dest_range
        self.source_nodata = source_nodata
        self.source_paths = source_paths

        if block_id is not None:

            print("Extracting Block only for {}".format(block_id))
            self.source_paths = [p for p in self.source_paths
                                 if str(block_id) in p]

        self.path_pairs = []

    def build_path_pairs(self):
        """
        Build source-target path pairs to be fed to gdal processing functions. The file containing folder has to either match the target block ID or granule name for paths to be paired. Processes items inplace.
        """

        if self.source_paths is None:

            raise ValueError("self.source_paths uninitialized!")

        for source_path in self.source_paths:

            for block_data_dir in data_settings.BLOCK_DATA_DIRS:

                block_id = os.path.split(block_data_dir)[-1]

                source_data_dir, filename = os.path.split(source_path)
                containing_dir = os.path.split(source_data_dir)[-1]

                if not containing_dir in [block_id, data_settings.GRANULE]:

                    continue

                block_data_path = os.path.join(block_data_dir, filename)
                self.path_pairs.append((source_path, block_data_path))

    def extract_block_projection(self, source_path):
        """
        Extract a raster's projection found in file path `source_path`. The `gdal.Open(source_path).GetProjection()` return something in the lines of the following:

            'PROJCS["ETRS89 / TM35FIN(E,N)",GEOGCS["ETRS89",DATUM["European_Terrestrial_Reference_System_1989",SPHEROID["GRS 1980",6378137,298.257222101,AUTHORITY["EPSG","7019"]],TOWGS84[0,0,0,0,0,0,0],AUTHORITY["EPSG","6258"]],PRIMEM["Greenwich",0,AUTHORITY["EPSG","8901"]],UNIT["degree",0.0174532925199433,AUTHORITY["EPSG","9122"]],AUTHORITY["EPSG","4258"]],PROJECTION["Transverse_Mercator"],PARAMETER["latitude_of_origin",0],PARAMETER["central_meridian",27],PARAMETER["scale_factor",0.9996],PARAMETER["false_easting",500000],PARAMETER["false_northing",0],UNIT["metre",1,AUTHORITY["EPSG","9001"]],AXIS["Easting",EAST],AXIS["Northing",NORTH],AUTHORITY["EPSG","3067"]]'

        The last AUTHORITY corresponds to the EPSG projection of the raster and that is what will be extracted.

        Args:

            source_path: A file path to the source raster.

        Returns:

            The raster's projection in format 'AUTHORITY:NUMBER', i.e. 'EPSG:3067'.
        """
        raster = gdal.Open(source_path)
        projection = (raster
                      .GetProjection()[-16:]
                      .replace("[", "")
                      .replace("]", "")
                      .replace("\"", "")
                      .split(","))
        return "{}:{}".format(projection[0], projection[1])

    def extract_field_blocks(self):
        """
        Extract block field datasets from source data files for distinct block IDs. The extraction is two-phase and happens effectively inplace, as the function processes files directly. 

        In the first phase, the source image is transformed with `gdal.Warp`. It is resampled to wished resolution and cut to shape using Block-corresponding Shape. The generated dataset is persisted to `settings.TEMP_PATH` regardless.

        If first phase executes without problems, the processed dataset is scaled to wished data range. If succesful, the image is then persisted to the corresponding target path.

        The paths should follow the convention of ../[DATASET]/[YEAR]/[GRANULE]/[BLOCK_ID]/[YYYYMMDD_FORMAT].tif.
        """
        t_start = time.time()

        scale_range = [self.source_range[0], self.source_range[1],
                       self.dest_range[0], self.dest_range[1]]
        counter = 0

        for source_path, target_path in self.path_pairs:

            os.makedirs(os.path.dirname(target_path), exist_ok=True)
            block_id = os.path.split(os.path.dirname(target_path))[-1]

            try:

                result_warp = gdal.Warp(data_settings.TEMP_PATH,
                                        source_path,
                                        srcSRS=self.extract_block_projection(
                                            source_path),
                                        dstSRS='EPSG:3067',
                                        cutlineDSName=data_settings.BLOCK_SHAPEFILE_PATH,
                                        cutlineWhere="LOHKO = '{}'".format(
                                            block_id),
                                        cropToCutline=True,
                                        xRes=self.target_resolution,
                                        yRes=self.target_resolution,
                                        srcNodata=self.source_nodata,
                                        dstNodata=np.nan,
                                        outputType=gdal.GDT_CFloat32,
                                        multithread=True)

                arr = result_warp.ReadAsArray()
                assert ~np.isnan(arr).all(), "Warped image contains only NaNs"

            except (RuntimeError, AttributeError, AssertionError) as ex:

                print("{}\tError (warp): {}".format(block_id, ex))
                print("\t\tFrom\t{}".format(source_path))
                print("\t\tTo\t{}".format(data_settings.TEMP_PATH))

                continue

            finally:

                result_warp = None

            try:

                result_translate = gdal.Translate(target_path,
                                                  settings.TEMP_PATH,
                                                  outputType=gdal.GDT_Float32,
                                                  scaleParams=[scale_range],
                                                  noData=np.nan)

                arr = result_translate.ReadAsArray()

                assert ~np.isnan(arr).all(
                ), "Translated image contains only NaNs"
                assert np.nanmin(arr) >= self.dest_range[0], "Translated values below lower destination range {}, min={}".format(
                    self.dest_range[0], np.nanmin(arr))
                assert np.nanmax(arr) <= self.dest_range[1], "Translated values above upper destination range {}, max={}".format(
                    self.dest_range[1], np.nanmax(arr))

                print("{}\tFrom\t{}".format(block_id, source_path))
                print("\t\tTo\t{}".format(target_path))

                plt.rcParams['figure.figsize'] = 1, 1

                if len(arr.shape) >= 3:

                    plt.imshow(arr[:3].transpose(1, 2, 0))

                else:

                    plt.imshow(arr, cmap='gray', vmin=0, vmax=1)

                plt.axis('off')
                plt.show()

            except (RuntimeError, AttributeError, AssertionError) as ex:

                print("{}\tError (translate): {}".format(block_id, ex))
                print("\t\tFrom\t{}".format(data_settings.TEMP_PATH))
                print("\t\tTo\t{}".format(target_path))

            finally:

                result_translate = None

            counter += 1

        t_delta = time.time() - t_start
        print("Processed {} field blocks in {:.0f}m {:.0f}s".format(
            counter, t_delta // 60, t_delta % 60))

    def assert_filenames(self):
        """
        Perform a final assertion of generated filenames dates and datatype indicators.
        """
        print("Asserting filenames: ", end="")
        error_files = []

        for data_dir in data_settings.BLOCK_DATA_DIRS:

            filenames = os.listdir(data_dir)

            for filename in filenames:

                if 'aux.xml' in filename or 'yield':

                    continue

                try:

                    filename_split = filename.split("_")
                    date = filename_split[0]
                    _, suffix = filename_split[-1].split(".")

                    assert suffix == 'tif', "Wrong file suffix"
                    assert len(date) == 8, "Wrong amount of numbers in date"
                    assert date[0:4] == '2017', "Year is wrong"
                    assert date[4] == '0', "No double digit months in dataset"
                    assert date[5] in ['4', '5', '6', '7', '8',
                                       '9'], "Month outside dataset range"
                    assert date[6] in ['0', '1', '2',
                                       '3'], "Ten-indicator for day is wrong"
                    assert date[7] in ['0', '1', '2', '3', '4', '5',
                                       '6', '7', '8', '9'], "Date is not a digit"
                    assert 'ndvi' in filename or 'drone_rgb' in filename or 'drone_ndvi' in filename, "Proper type is missing"

                    if 'sentinel_ndvi' in filename:

                        assert len(filename) == 26, "Filename wrong for {} in {}".format(
                            filename, data_dir)

                    if 'drone_ndvi' in filename:

                        assert len(filename) == 23, "Filename wrong for {} in {}".format(
                            filename, data_dir)

                    if 'drone_rgb' in filename:

                        assert len(filename) == 22, "Filename wrong for {} in {}".format(
                            filename, data_dir)

                except (AssertionError, ValueError) as ex:

                    error_files.append("{}: {}".format(
                        ex, os.path.join(data_dir, filename)))

        if not error_files:

            print("All generated block datasets named correctly!")

        else:

            print("There were some problems with the following files")

            for error_file in error_files:
                print("\t{}".format(error_file))

    def extract(self):
        """
        A wrapper function for building the path pairs, extracting the Block-wise datasets and asserting the generated filenames.
        """
        self.build_path_pairs()
        self.extract_field_blocks()
        self.assert_filenames()


class Sentinel2ANDVI(BlockExtractor):
    """
    A class for extracting resampled and rescaled Block-wise Sentinel-2A NDVI datasets from distinct source files.

    Example usage:

        sentinel_ndvi = Sentinel2ANDVI(target_resolution=10, source_range=(-1,1), 
                                dest_range=(-1,1),source_nodata=np.nan)
        sentinel_ndvi.extract()
    """

    def __init__(self, target_resolution, source_range, dest_range, source_nodata, block_id=None):
        """
        Args:

            target_resolution: The resolution in coordinate units per pixels to which the extracted images are to resampled.
            source_range: The absolute data range of source data.
            dest_range: The absolute data range of output data.
            source_nodata: The `NoData` value of the source data.
            source_paths: A list of source file paths.
            block_id: Optional. Perform operations only for a single Block if provided.
        """

        source_paths = [os.path.join(data_settings.GRANULE_DATA_DIR, p)
                        for p in os.listdir(data_settings.GRANULE_DATA_DIR)
                        if not os.path.isdir(os.path.join(data_settings.GRANULE_DATA_DIR, p))
                        and 'sentinel_ndvi' in p and 'aux' not in p]

        super().__init__(target_resolution, source_range,
                         dest_range, source_nodata, source_paths, block_id)


class DroneNDVI(BlockExtractor):
    """
    A class for extracting resampled and rescaled Block-wise Drone NDVI datasets from distinct source files.

    Example usage:

        drone_ndvi = DroneNDVI(target_resolution=10/32, source_range=(-1,1), 
                            dest_range=(-1,1), source_nodata=-10000)
        drone_ndvi.extract()
    """

    def __init__(self, target_resolution, source_range, dest_range, source_nodata, block_id=None):
        """
        Args:

            target_resolution: The resolution in coordinate units per pixels to which the extracted images are to resampled.
            source_range: The absolute data range of source data.
            dest_range: The absolute data range of output data.
            source_nodata: The `NoData` value of the source data.
            source_paths: A list of source file paths.
            block_id: Optional. Perform operations only for a single Block if provided.
        """
        source_paths = []
        for drone_dir in data_settings.DRONE_DATA_DIRS:
            [source_paths.append(os.path.join(drone_dir, p))
             for p in os.listdir(drone_dir)
             if 'drone_ndvi' in p and 'aux' not in p]

        super().__init__(target_resolution, source_range,
                         dest_range, source_nodata, source_paths, block_id)


class DroneRGB(BlockExtractor):
    """
    A class for extracting resampled and rescaled Block-wise Drone RGB datasets from distinct source files.

    Example usage:

        drone_rgb = DroneRGB(target_resolution=10/32, source_range=(0,255),   
                            dest_range=(0,1), source_nodata=0)
        drone_rgb.extract()
    """

    def __init__(self, target_resolution, source_range, dest_range, source_nodata, block_id=None):
        """
        Args:

            target_resolution: The resolution in coordinate units per pixels to which the extracted images are to resampled.
            source_range: The absolute data range of source data.
            dest_range: The absolute data range of output data.
            source_nodata: The `NoData` value of the source data.
            source_paths: A list of source file paths.
            block_id: Optional. Perform operations only for a single Block if provided.
        """
        source_paths = []
        for drone_dir in data_settings.DRONE_DATA_DIRS:
            [source_paths.append(os.path.join(drone_dir, p))
             for p in os.listdir(drone_dir)
             if 'drone_rgb' in p and 'aux' not in p]

        super().__init__(target_resolution, source_range,
                         dest_range, source_nodata, source_paths, block_id)
