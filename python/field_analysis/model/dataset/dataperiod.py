import datetime
import os
import pickle

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from ...settings import model as model_settings
from ...db import models
from ...db.models import Area, DataPeriod, Target
from . import utils

DATE_DIVIDER = datetime.date(2017, 7, 1)


class DataPeriodDataset(Dataset):
    """
    Abstract base class for retrieving input-target pairs for a single DataPeriod persisted as a database entries.
    """

    def __init__(self, db_name=None):
        """
        Args:

            db_name: Name of the dataset DB file to use.
        """
        self.dataperiods = None
        self.db_name = db_name

        if db_name is not None:
            models.initialize_db(os.path.join(
                model_settings.DATABASES_DIR, db_name))

    def __getitem__(self, idx):
        raise NotImplementedError

    def __len__(self):
        return self.dataperiods.count()

    def size(self):
        """
        Return the number of samples in the dataset.

        Returns:

            Sample count.
        """
        return self.__len__()

    def array_to_tensor(self, array):
        """
        Convert Numpy arrays of images to tensors. If the image has multiple bands, it is transformed from (rows, cols, bands) to (bands, rows, cols).

        Args:

            array: A Numpy image array with dimensions (rows, cols, bands).

        Returns:

            A Torch Tensor with image's bands as the first dimension.
        """
        if len(array.shape) == 3:

            array = array.transpose(2, 0, 1)

        else:

            array = np.expand_dims(array, 0)

        return torch.from_numpy(np.nan_to_num(array))

    def to_batches(self, batch_size):
        """
        Convert the sample-wise dataset to batches of samples.

        Args:

            batch_size: The number of samples in a batch.

        Returns:

            A `DataLoader` for iterating over generated batches of samples.
        """
        return DataLoader(self, batch_size=batch_size, shuffle=True)

    def pair_rgb_to_yield_mean(self, idx):
        """
        Retrieve a single RGB-Yield image pair from the dataset as Tensors. The input is stored with the key 'x' and the target with the key 'y'.

        Args:

            idx: The index of the DataPeriod to retrieve.

        Returns:

            A dictionary containing input and target samples for the DataPeriod.
        """
        sample = self.pair_rgb_to_yield_img(idx)

        sample['y'] = sample['y'].mean()

        return sample

    def pair_ndvi_to_yield_mean(self, idx):
        """
        Retrieve a single NDVI-Yield image pair from the dataset as Tensors. The input is stored with the key 'x' and the target with the key 'y'.

        Args:

            idx: The index of the DataPeriod to retrieve.

        Returns:

            A dictionary containing input and target samples for the DataPeriod.
        """
        sample = self.pair_ndvi_to_yield_img(idx)

        sample['y'] = sample['y'].mean()

        return sample

    def pair_rgb_to_yield_img(self, idx):
        """
        Retrieve a single RGB-Yield image-value pair from the dataset as Tensors. The input is stored with the key 'x' and the target with the key 'y'.

        Args:

            idx: The index of the DataPeriod to retrieve.

        Returns:

            A dictionary containing input and target samples for the DataPeriod.
        """
        dataperiod = self.dataperiods[idx]
        target = dataperiod.area.target

        arr_x = pickle.loads(dataperiod.area_drone_rgb)
        arr_y = pickle.loads(target.area_yield)

        sample = {'x': self.array_to_tensor(arr_x),
                  'y': self.array_to_tensor(arr_y)}

        return sample

    def pair_ndvi_to_yield_img(self, idx):
        """
        Retrieve a single NDVI-Yield image-value pair from the dataset as Tensors. The input is stored with the key 'x' and the target with the key 'y'.

        Args:

            idx: The index of the DataPeriod to retrieve.

        Returns:

            A dictionary containing input and target samples for the DataPeriod.
        """
        dataperiod = self.dataperiods[idx]
        target = dataperiod.area.target

        arr_x = pickle.loads(dataperiod.area_drone_ndvi)
        arr_y = pickle.loads(target.area_yield)

        sample = {'x': self.array_to_tensor(arr_x),
                  'y': self.array_to_tensor(arr_y)}

        return sample

    def separate_train_test(self, batch_size, train_ratio):
        """
        Create separate batches of samples for training and testing. The samples are shuffled before allocation to the one or the other.

        Args:

            batch_size: The number of samples in a batch.
            train_ratio: The ratio of samples from the dataset to be used as training samples.
        """

        shuffled_indices = torch.randperm(self.size())

        training_indices_list = list(shuffled_indices[:round(
            self.size()*train_ratio)].numpy())
        validation_indices_list = list(shuffled_indices[round(
            self.size()*train_ratio):].numpy())

        training_set = self[training_indices_list]
        validation_set = self[validation_indices_list]

        return (DataLoader(training_set, batch_size=batch_size),
                DataLoader(validation_set, batch_size=batch_size))


class DroneRGBEarlier(DataPeriodDataset):
    """
    Class providing training dataset for pre-July Drone RGB images and corresponding yields.

    Example usage:

        dataset = DroneRGBEarlier()
        x, y = dataset[0].values()

    The samples are contained in a dict with key 'x' corresponding to input dataset and 'y' to target dataset.
    """

    def __init__(self, db_name=None, image_target=False):
        """
        Args:

            image_target: A Boolean for whether to use images as targets or not. Defaults to False.
            db_name: Name of the dataset DB file to use.
        """
        super().__init__(db_name)

        self.dataperiods = (DataPeriod
                            .select(DataPeriod.date,
                                    DataPeriod.area_drone_rgb,
                                    Area.id,
                                    Target.area_yield)
                            .join(Area)
                            .join(Target)
                            .where(
                                (DataPeriod.area_drone_rgb.is_null(False)) &
                                (DataPeriod.date < DATE_DIVIDER) &
                                (Area.block_id != 8860095891) &
                                (Target.area_yield.is_null(False))))
        if image_target:
            self.get_function = self.pair_rgb_to_yield_img

        else:
            self.get_function = self.pair_rgb_to_yield_mean

    def __getitem__(self, idx):
        return utils.allow_multi_index(idx=idx, get_item_function=self.get_function)


class DroneRGBLater(DataPeriodDataset):
    """
    Class providing training dataset for post-July Drone RGB images and corresponding yields.

    Example usage:

        dataset = DroneRGBLater()
        x, y = dataset[0].values()

    The samples are contained in a dict with key 'x' corresponding to input dataset and 'y' to target dataset.
    """

    def __init__(self, db_name=None, image_target=False):
        """
        Args:

            image_target: A Boolean for whether to use images as targets or not. Defaults to False.
            db_name: Name of the dataset DB file to use.
        """
        super().__init__(db_name)

        self.dataperiods = (DataPeriod
                            .select(DataPeriod.date,
                                    DataPeriod.area_drone_rgb,
                                    Area.id,
                                    Target.area_yield)
                            .join(Area)
                            .join(Target)
                            .where(
                                (DataPeriod.area_drone_rgb.is_null(False)) &
                                (DataPeriod.date >= DATE_DIVIDER) &
                                (Area.block_id != 8860095891) &
                                (Target.area_yield.is_null(False))))
        if image_target:
            self.get_function = self.pair_rgb_to_yield_img

        else:
            self.get_function = self.pair_rgb_to_yield_mean

    def __getitem__(self, idx):
        return utils.allow_multi_index(idx=idx, get_item_function=self.get_function)


class DroneNDVIEarlier(DataPeriodDataset):
    """
    Class providing training dataset for pre-July Drone NDVI images and corresponding yields.

    Example usage:

        dataset = DroneNDVIEarlier()
        x, y = dataset[0].values()

    The samples are contained in a dict with key 'x' corresponding to input dataset and 'y' to target dataset.
    """

    def __init__(self, db_name=None, image_target=False):
        """
        Args:

            image_target: A Boolean for whether to use images as targets or not. Defaults to False.
            db_name: Name of the dataset DB file to use.
        """
        super().__init__(db_name)

        self.dataperiods = (DataPeriod
                            .select(DataPeriod.date,
                                    DataPeriod.area_drone_ndvi,
                                    Area.id,
                                    Target.area_yield)
                            .join(Area)
                            .join(Target)
                            .where(
                                (DataPeriod.area_drone_ndvi.is_null(False)) &
                                (DataPeriod.date < DATE_DIVIDER) &
                                (Area.block_id != 8860095891) &
                                (Target.area_yield.is_null(False))))
        if image_target:
            self.get_function = self.pair_ndvi_to_yield_img

        else:
            self.get_function = self.pair_ndvi_to_yield_mean

    def __getitem__(self, idx):
        return utils.allow_multi_index(idx=idx, get_item_function=self.get_function)


class DroneNDVILater(DataPeriodDataset):
    """
    Class providing training dataset for post-July Drone NDVI images and corresponding yields.

    Example usage:

        dataset = DroneNDVILater()
        x, y = dataset[0].values()

    The samples are contained in a dict with key 'x' corresponding to input dataset and 'y' to target dataset.
    """

    def __init__(self, db_name=None, image_target=False):
        """
        Args:

            image_target: A Boolean for whether to use images as targets or not. Defaults to False.
            db_name: Name of the dataset DB file to use.
        """
        super().__init__(db_name)
        self.dataperiods = (DataPeriod
                            .select(DataPeriod.date,
                                    DataPeriod.area_drone_ndvi,
                                    Area.id,
                                    Target.area_yield)
                            .join(Area)
                            .join(Target)
                            .where(
                                (DataPeriod.area_drone_ndvi.is_null(False)) &
                                (DataPeriod.date >= DATE_DIVIDER) &
                                (Area.block_id != 8860095891) &
                                (Target.area_yield.is_null(False))))

        if image_target:
            self.get_function = self.pair_ndvi_to_yield_img

        else:
            self.get_function = self.pair_ndvi_to_yield_mean

    def __getitem__(self, idx):
        return utils.allow_multi_index(idx=idx, get_item_function=self.get_function)
