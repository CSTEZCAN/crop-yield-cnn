import os
import unittest

from ....db import dataset, models


class TestDatasetBuilder(unittest.TestCase):

    def setUp(self):
        self.builder = dataset.DatasetBuilder(
            initialize=True,
            area_size_meters=10,
            area_size_px=16)

    def tearDown(self):
        db_path = self.builder.db_path
        os.remove(db_path)

    def test_01_get_soil_data(self):
        soil_data = self.builder.get_soil_data()
        self.assertTrue(soil_data.size, 0)

    def test_02_get_weather_data(self):
        weather_data = self.builder.get_weather_data()
        self.assertTrue(weather_data.size, 0)

    def test_03_extract_area_grids(self):
        block_areas = self.builder.extract_area_grids(
            show_image=False, test=True)
        self.assertGreater(len(list(block_areas.values())), 0)

    def test_04_generate_areas(self):
        block_areas = self.builder.extract_area_grids(
            show_image=False, test=True)
        self.builder.generate_areas(block_id_only=list(block_areas.keys())[0])
        with models.db:
            self.assertGreater(models.Area.select().count(), 0)

    def test_05_generate_dataperiods(self):
        self.builder.get_soil_data()
        self.builder.get_weather_data()

        block_areas = self.builder.extract_area_grids(
            show_image=False, test=True)
        self.builder.generate_areas(block_id_only=list(block_areas.keys())[0])
        self.builder.generate_dataperiods()
        with models.db:
            self.assertGreater(models.DataPeriod.select().count(), 0)

    def test_06_generate_targets(self):
        block_areas = self.builder.extract_area_grids(
            show_image=False, test=True)
        self.builder.generate_areas(block_id_only=list(block_areas.keys())[0])
        self.builder.generate_targets()
        with models.db:
            self.assertGreater(models.Target.select().count(), 0)
