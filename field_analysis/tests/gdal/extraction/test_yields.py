import os
import unittest

import geopandas
import numpy as np
import osgeo

from ....settings import data as data_settings
from ....extraction import yields


class TestYieldExtractions(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_01_drop_null_points(self):
        yields.drop_null_points(test=True)
        shape_path = [os.path.join(data_settings.YIELD_SHAPES_DIR, f)
                      for f in os.listdir(data_settings.YIELD_SHAPES_DIR)
                      if '.shp' in f][0]
        try:
            gdf = geopandas.read_file(shape_path)
            self.assertFalse(gdf['Mass_Yield'].isnull().all())
        finally:
            gdf = None

    def test_02_rasterize_shapefiles(self):
        yields.rasterize_shapefiles(
            algorithm='invdist', short_side=64, test=True)
        raster_path = os.path.join(data_settings.YIELD_RASTERS_DIR,
                                   "grid_000.tif")
        try:
            raster = osgeo.gdal.Open(raster_path)
            arr = raster.ReadAsArray()
            self.assertFalse(np.isnan(arr).all())
        finally:
            raster = None

    def test_03_reproject(self):
        yields.reproject(source_srs='EPSG:4326', target_srs='EPSG:3067',
                         resolution=10/32, test=True)
        raster_path = os.path.join(data_settings.YIELD_RASTERS_DIR,
                                   "warp_000.tif")
        try:
            raster = osgeo.gdal.Open(raster_path)
            arr = raster.ReadAsArray()
            self.assertFalse(np.isnan(arr).all())
        finally:
            raster = None

    def test_04_extract_block_shapes(self):
        raster_path = yields.extract_block_shapes(test=True)
        try:
            raster = osgeo.gdal.Open(raster_path)
            arr = raster.ReadAsArray()
            self.assertFalse(np.isnan(arr).all())
        finally:
            raster = None
