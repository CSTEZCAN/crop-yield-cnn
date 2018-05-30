import unittest

from torch.utils.data import DataLoader

from ....model.dataset.dataperiod import (DroneNDVIEarlier, DroneNDVILater,
                                          DroneRGBEarlier, DroneRGBLater)

db_name1 = 'field_analysis_10m_32px.db'
db_name2 = 'field_analysis_20m_64px.db'


class TestDatasets(unittest.TestCase):

    def setUp(self):
        self.datasets = [DroneNDVIEarlier, DroneNDVILater,
                         DroneRGBEarlier, DroneRGBLater]

    def tearDown(self):
        pass

    def test_01_get(self):
        for dataset in self.datasets:
            x, y = dataset(db_name1)[0].values()
            self.assertGreaterEqual(len(x.size()), 3)
            self.assertIsInstance(y, float)

    def test_02_single_initialization(self):
        ds1 = self.datasets[0](db_name1)
        x1, y1 = ds1[0].values()
        self.assertEqual(x1.size()[2], 32)
        ds2 = self.datasets[1](db_name2)
        x2, y2 = ds2[0].values()
        self.assertEqual(x2.size()[2], 64)

    def test_03_multiple_initialization(self):
        ds1 = self.datasets[0](db_name1)
        ds2 = self.datasets[1](db_name2)
        x1, y1 = ds1[0].values()
        x2, y2 = ds2[0].values()
        self.assertEqual(x1.size()[2], 64)
        self.assertEqual(x2.size()[2], 64)

    def test_04_dataset_to_dataloader(self):
        for dataset in self.datasets:
            dataloader = DataLoader(dataset(db_name1), batch_size=16)
            for batch in dataloader:
                self.assertEqual(len(batch['x'].size()), 4)
                self.assertEqual(len(batch['y'].size()), 1)
                self.assertEqual(batch['x'].size()[0], 16)
                self.assertEqual(batch['y'].size()[0], 16)
                break

    def test_05_separate_datasets(self):
        for dataset in self.datasets:
            train, test = dataset(db_name1).separate_train_test(
                batch_size=32, train_ratio=0.8)
            self.assertGreater(len(train), len(test))
