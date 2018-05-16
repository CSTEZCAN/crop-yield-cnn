import os
import unittest

import numpy as np

from ....settings import data as data_settings
from ....extraction import blocks


class TestSentinel2ANDVI(unittest.TestCase):

    def setUp(self):
        self.extractor = blocks.Sentinel2ANDVI(
            target_resolution=10,
            source_range=(-1, 1),
            dest_range=(-1, 1),
            source_nodata=np.nan)

    def tearDown(self):
        pass

    def test_01_build_path_pairs(self):
        self.extractor.build_path_pairs()
        self.assertGreater(len(self.extractor.path_pairs), 0)
        self.assertIsInstance(self.extractor.path_pairs[0], tuple)

    def test_02_extract_field_block(self):
        self.extractor.build_path_pairs()
        self.extractor.path_pairs = self.extractor.path_pairs[0:1]
        self.extractor.extract_field_blocks()
        self.assertTrue(os.path.isfile(self.extractor.path_pairs[0][0]))
        self.assertTrue(os.path.isfile(data_settings.TEMP_PATH))
        self.assertTrue(os.path.isfile(self.extractor.path_pairs[0][1]))


class TestDroneNDVI(unittest.TestCase):

    def setUp(self):
        self.extractor = blocks.DroneNDVI(
            target_resolution=10/32,
            source_range=(-1, 1),
            dest_range=(-1, 1),
            source_nodata=np.nan)

    def tearDown(self):
        pass

    def test_01_build_path_pairs(self):
        self.extractor.build_path_pairs()
        self.assertGreater(len(self.extractor.path_pairs), 0)
        self.assertIsInstance(self.extractor.path_pairs[0], tuple)

    def test_02_extract_field_block(self):
        self.extractor.build_path_pairs()
        self.extractor.path_pairs = self.extractor.path_pairs[0:1]
        self.extractor.extract_field_blocks()
        self.assertTrue(os.path.isfile(self.extractor.path_pairs[0][0]))
        self.assertTrue(os.path.isfile(data_settings.TEMP_PATH))
        self.assertTrue(os.path.isfile(self.extractor.path_pairs[0][1]))


class TestDroneRGB(unittest.TestCase):

    def setUp(self):
        self.extractor = blocks.DroneRGB(
            target_resolution=10/32,
            source_range=(0, 255),
            dest_range=(0, 1),
            source_nodata=np.nan)

    def tearDown(self):
        pass

    def test_01_build_path_pairs(self):
        self.extractor.build_path_pairs()
        self.assertGreater(len(self.extractor.path_pairs), 0)
        self.assertIsInstance(self.extractor.path_pairs[0], tuple)

    def test_02_extract_field_block(self):
        self.extractor.build_path_pairs()
        self.extractor.path_pairs = self.extractor.path_pairs[0:1]
        self.extractor.extract_field_blocks()
        self.assertTrue(os.path.isfile(self.extractor.path_pairs[0][0]))
        self.assertTrue(os.path.isfile(data_settings.TEMP_PATH))
        self.assertTrue(os.path.isfile(self.extractor.path_pairs[0][1]))
