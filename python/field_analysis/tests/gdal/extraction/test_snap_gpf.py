import os
import unittest

import snappy

from ....settings import data as data_settings
from ....extraction.snap_gpf import Sentinel2A


class TestSNAPGPFSentinel2A(unittest.TestCase):

    def setUp(self):
        self.snap = Sentinel2A()
        self.granule_path = os.path.join(
            data_settings.GRANULE_DATA_DIR,
            self.snap.granules[0],
            data_settings.SENTINEL_FILE_NAME)
        self.target_path = os.path.join(
            data_settings.GRANULE_DATA_DIR,
            self.snap.granules[0] + self.snap.file_suffix)
        self.product = None

    def tearDown(self):
        self.product = None

    def test_01_read(self):
        self.assertTrue(os.path.isfile(self.granule_path))
        self.product = self.snap.read(self.granule_path)
        print(type(self.product))
        self.assertIsNotNone(self.product)
        self.assertIsInstance(self.product, snappy.Product)

    def test_02_resample(self):
        self.snap.read(self.granule_path)
        self.product = self.snap.resample()
        self.assertIsNotNone(self.product)
        self.assertIsInstance(self.product, snappy.Product)

    def test_03_band_maths(self):
        self.snap.read(self.granule_path)
        self.snap.resample()
        self.product = self.snap.band_maths(
            '((quality_scene_classification==4 or quality_scene_classification==5)?(B8-B4)/(B8+B4):NaN)')
        self.assertIsNotNone(self.product)
        self.assertIsInstance(self.product, snappy.Product)

    def test_04_reproject(self):
        self.snap.read(self.granule_path)
        self.snap.resample()
        self.snap.band_maths(
            '((quality_scene_classification==4 or quality_scene_classification==5)?(B8-B4)/(B8+B4):NaN)')
        self.product = self.snap.reproject()
        self.assertIsNotNone(self.product)
        self.assertIsInstance(self.product, snappy.Product)

    def test_05_write(self):
        self.snap.read(self.granule_path)
        self.snap.resample()
        self.snap.band_maths(
            '((quality_scene_classification==4 or quality_scene_classification==5)?(B8-B4)/(B8+B4):NaN)')
        self.snap.reproject()
        self.product = self.snap.write(self.target_path)
        self.assertIsNone(self.product)
        self.assertTrue(os.path.isfile(self.target_path))
