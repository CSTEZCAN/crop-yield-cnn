import os
import unittest

from torch import optim

from ....model.dataset.dataperiod import DroneNDVIEarlier, DroneRGBEarlier
from ....model.nets.cnn import DroneYieldMeanCNN


class TestDroneYieldMeanCNN(unittest.TestCase):

    def setUp(self):
        db_name = 'field_analysis_10m_32px.db'
        self.ndvi = DroneNDVIEarlier(db_name)
        self.rgb = DroneRGBEarlier(db_name)
        self.model = DroneYieldMeanCNN

    def tearDown(self):
        pass

    def test_01_initialize(self):
        self.assertIsInstance(DroneYieldMeanCNN(
            source_bands=1, source_dim=32), DroneYieldMeanCNN)

    def test_02_train_separate_sets(self):
        model = DroneYieldMeanCNN(source_bands=1, source_dim=32)
        train, test = self.ndvi.separate_train_test(
            batch_size=64, train_ratio=0.8)
        losses = model.train(epochs=1, training_data=train,
                             test_data=test, visualize=False)
        self.assertGreater(len(losses[0]), 0)
        self.assertGreater(len(losses[1]), 0)

    def test_03_train_cv(self):
        model = DroneYieldMeanCNN(source_bands=1, source_dim=32)
        losses = model.train(
            epochs=1, training_data=self.ndvi, k_cv_folds=2, visualize=False)
        self.assertGreater(len(losses[0]), 0)
        self.assertGreater(len(losses[1]), 0)

    def test_04_train_multiband(self):
        model = DroneYieldMeanCNN(source_bands=3, source_dim=32)
        losses = model.train(epochs=1, training_data=self.rgb,
                             k_cv_folds=2, visualize=False)
        self.assertGreater(len(losses[0]), 0)
        self.assertGreater(len(losses[1]), 0)

    def test_05_model_persistence(self):
        model = DroneYieldMeanCNN(source_bands=3, source_dim=32)
        model.train(epochs=1, training_data=self.rgb,
                    k_cv_folds=2, visualize=False)
        self.assertTrue(os.path.isfile(model.model_path))

    def test_06_persisted_model_usage(self):
        model = DroneYieldMeanCNN(source_bands=3, source_dim=32)
        model.load_model()
        losses = model.train(epochs=1, training_data=self.rgb,
                             k_cv_folds=2, visualize=False)
        self.assertGreater(len(losses[0]), 0)
        self.assertGreater(len(losses[1]), 0)

    def test_07_custom_optimizer(self):
        model = DroneYieldMeanCNN(
            source_bands=1, source_dim=32, optimizer=optim.Adadelta)
        losses = model.train(
            epochs=1, training_data=self.ndvi, k_cv_folds=2, visualize=False)
        self.assertGreater(len(losses[0]), 0)
        self.assertGreater(len(losses[1]), 0)

    def test_08_custom_optimizer_params(self):
        model = DroneYieldMeanCNN(
            source_bands=1, source_dim=32, optimizer_parameters={'momentum': 0.1})
        losses = model.train(
            epochs=1, training_data=self.ndvi, k_cv_folds=2, visualize=False)
        self.assertGreater(len(losses[0]), 0)
        self.assertGreater(len(losses[1]), 0)
