import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec
import torch
from torch import optim

import field_analysis.model.dataset.dataperiod as dp
from field_analysis.model.nets.cnn import DroneYieldMeanCNN
from field_analysis.settings import model as model_settings

plt.switch_backend('agg')



RESULTS_DIR = os.path.join(model_settings.EXEC_ROOT, 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

def ml_iii_1_2():

    db_32 = 'field_analysis_10m_32px.db'
    db_64 = 'field_analysis_20m_64px.db'
    db_128 = 'field_analysis_40m_128px.db'
    dbs = [db_32, db_64, db_128]
    optimizers = [None, optim.Adadelta]

    for dataset in [dp.DroneNDVIEarlier,dp.DroneNDVILater,dp.DroneRGBEarlier,dp.DroneRGBLater]:
        models = pd.DataFrame()
        losses = pd.DataFrame()
        for i, db in enumerate(dbs):
            ds = dataset(db_name=db)
            dataset_name = ds.__class__.__name__
            source_bands = 1  # NDVI
            if 'RGB' in dataset_name:
                source_bands = 3
            for optimizer in optimizers:
                source_dim = 32*(2**i)
                if optimizer is not None:
                    optim_name = 'Adadelta'
                else:
                    optim_name = 'SGD'
                print("Dataset={}, Image={}x{}, Optimizer={}".format(
                    dataset_name, source_dim, source_dim, optim_name))
                cnn = DroneYieldMeanCNN(
                    source_bands=source_bands,
                    source_dim=source_dim,
                    cnn_layers=6,
                    fc_layers=2,
                    optimizer=optimizer)
                losses_dict = cnn.train(
                    epochs=50,
                    training_data=ds,
                    k_cv_folds=3,
                    suppress_output=True)
                cnn.save_model()
                best_loss = np.array(losses_dict['test_losses_mean_std'])[:, 0].min()
                losses.loc[source_dim,optim_name] = best_loss
                models.loc[source_dim,optim_name] = cnn.model_filename[:4]
        losses.to_csv(os.path.join(RESULTS_DIR,f'ml_iii_1_2_{dataset_name}_losses.csv'))
        models.to_csv(os.path.join(RESULTS_DIR,f'ml_iii_1_2_{dataset_name}_models.csv'))
