from abc import ABC, abstractmethod
import os
import time

import numpy as np
from osgeo import gdal
from snappy import GPF, ProductIO, jpy

from ..settings import data as data_settings

gdal.UseExceptions()


class SNAPExtractor(ABC):
    """
    A base class for utilizing SNAP Graph Processing Framework (GPF) in Python.
    """

    def __init__(self):
        GPF.getDefaultInstance().getOperatorSpiRegistry().loadOperatorSpis()
        self.product = None

    def read(self, path):
        """
        Read the dataset to as a SNAP Product.

        Args:

            path: The file path to the source dataset.

        Returns:

            The current state of the `self.product`.
        """
        print("\tRead, path={}".format(path))

        self.product = ProductIO.readProduct(path)
        return self.product

    def resample(self):
        """
        Perform a SNAP GPF Resampling operation on the product by resizing every band to have spatial resolution of 10m.

        Example code:
        http://forum.step.esa.int/t/resample-all-bands-of-an-l2a-image/5032
        """
        print("\tResample, product={}".format(self.product.getName()))

        HashMap = jpy.get_type('java.util.HashMap')
        parameters = HashMap()
        parameters.put('targetResolution', 10)

        self.product = GPF.createProduct('Resample', parameters, self.product)
        return self.product

    def band_maths(self, expression):
        """
        Perform a SNAP GPF BandMath operation on the product, excluding non-vegetation and non-soil pixels.

        Args:

            expression: The band maths expression to execute
        """
        print("\tBandMaths, product={}".format(self.product.getName()))

        BandDescriptor = jpy.get_type(
            'org.esa.snap.core.gpf.common.BandMathsOp$BandDescriptor')

        band = BandDescriptor()
        band.name = 'band_maths'
        band.type = 'float32'
        band.expression = expression
        band.noDataValue = np.nan

        bands = jpy.array(
            'org.esa.snap.core.gpf.common.BandMathsOp$BandDescriptor', 1)
        bands[0] = band

        HashMap = jpy.get_type('java.util.HashMap')
        parameters = HashMap()
        parameters.put('targetBands', bands)

        self.product = GPF.createProduct('BandMaths', parameters, self.product)
        return self.product

    def reproject(self):
        """
        Perform a SNAP GPF Reprojection operation on the product by reprojecting from WGS-84 to EPSG:3067.
        """
        print("\tReproject, product={}".format(self.product.getName()))

        HashMap = jpy.get_type('java.util.HashMap')
        parameters = HashMap()
        parameters.put('crs', 'EPSG:3067')

        self.product = GPF.createProduct('Reproject', parameters, self.product)
        return self.product

    def write(self, file_path):
        """
        Write the processed product to a file with a file type and close the product.
        """
        print("\tWrite, file_path={}".format(file_path))
        ProductIO.writeProduct(self.product, file_path, 'GeoTIFF')
        self.product.closeIO()
        self.product = None
        return self.product

    @abstractmethod
    def extract(self):
        pass


class Sentinel2A(SNAPExtractor):
    """
    SNAP GPF NDVI extractor for Sentinel2A granules.
    """

    def __init__(self):
        super().__init__()

        self.granules = [p for p in os.listdir(data_settings.GRANULE_DATA_DIR)
                         if os.path.isdir(os.path.join(data_settings.GRANULE_DATA_DIR, p))]

        self.file_suffix = "_sentinel_ndvi.tif"

    def extract(self):
        """
        Perform required SNAP Graph Processing Framework (GPF) and GDAL operations on select granules. While every other operation is run according to `SNAPExtractor` function definitions, the band maths expression is formulated after this this example: 

        https://github.com/senbox-org/snap-engine/blob/master/snap-python/src/main/resources/snappy/examples/snappy_bmaths.py
        """
        print("Performing SNAP GPF ops on Sentinel2A Granules")

        t_main = time.time()

        for granule in self.granules:

            t_start = time.time()

            granule_path = os.path.join(
                data_settings.GRANULE_DATA_DIR,
                granule, data_settings.SENTINEL_FILE_NAME
            )
            target_file_path = os.path.join(
                data_settings.GRANULE_DATA_DIR,
                granule + self.file_suffix
            )

            print("Processing", granule, "from", granule_path)

            self.read(granule_path)
            self.resample()
            self.band_maths(
                '((quality_scene_classification==4 or quality_scene_classification==5)?(B8-B4)/(B8+B4):NaN)')
            self.reproject()
            self.write(target_file_path)

            t_delta = time.time() - t_start
            print("\tProcessing done in {:.0f}m {:.0f}s".format(
                t_delta // 60, t_delta % 60))

        t_delta_main = time.time() - t_main
        print("Done in {:.0f}m {:.0f}s".format(
            t_delta_main // 60, t_delta_main % 60))
