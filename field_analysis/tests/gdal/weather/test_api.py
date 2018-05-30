import datetime as dt
import os
import unittest

from ....weather import api


class TestWeatherAPI(unittest.TestCase):

    def setUp(self):
        self.api = api.FMIDetailed(
            place='Pori',
            start_date=dt.datetime(2017, 6, 1),
            end_date=dt.datetime(2017, 6, 10))

    def tearDown(self):
        pass

    def test_get_weather_data(self):
        dataset = self.api.get_weather_data()
        self.assertGreater(dataset.size, 0)
        self.assertTrue(os.path.isfile(self.api.filepath))
