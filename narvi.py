import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec
from torch import optim

from field_analysis.model.dataset import dataperiod
from field_analysis.model.nets.cnn import DroneYieldMeanCNN
from field_analysis.settings import model as model_settings

plt.switch_backend('agg')

DB_32 = 'field_analysis_10m_32px.db'
DB_64 = 'field_analysis_20m_64px.db'
DB_128 = 'field_analysis_40m_128px.db'
DBS = [DB_32, DB_64, DB_128]
DATASETS = [dataperiod.DroneNDVIEarlier, dataperiod.DroneNDVILater,
            dataperiod.DroneRGBEarlier, dataperiod.DroneRGBLater]
RESULTS_DIR = os.path.join(model_settings.EXEC_ROOT, 'results')


def main():

    os.makedirs(RESULTS_DIR, exist_ok=True)

    parser = argparse.ArgumentParser(
        description='Perform CNN tuning component-wise utilizing Narvi batch execution and the $SLURM_ARRAY_TASK_ID variable.')
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--optimizer-batch',
                       dest='optimizer_batch',
                       type=int,
                       metavar='N',
                       default=None,
                       help="Perform optimizer-wise batch size tests.")
    args = parser.parse_args()

    if args.optimizer_batch is not None:

        test_optimizer_batch_size(args.optimizer_batch)


def test_optimizer_batch_size(batch_number):

    if batch_number == 0:

        optimizer = None
        optimizer_name = 'SGD'

    elif batch_number == 1:

        optimizer = optim.RMSprop
        optimizer_name = 'RMSProp'

    elif batch_number == 2:

        optimizer = optim.Adadelta
        optimizer_name = 'Adadelta'

    else:

        raise ValueError(f'Batch number out of allowed scope: {batch_number}')

    print(f'Performing optimizer batch tests with {optimizer_name}.')

    plt.rcParams['figure.figsize'] = 10, 6
    batch_sizes = [32 * 2**x for x in range(6)]

    df = pd.DataFrame()

    for i, source_dim in enumerate([32, 64, 128]):

        for j, dataset in enumerate(DATASETS):

            data_type = dataset.__name__

            gs = GridSpec(nrows=2, ncols=1, height_ratios=[3, 1])
            plt.subplot(gs[0])

            batch_losses_deltas = []

            for batch_size in batch_sizes:

                print("{} {}x{}, batch_size={}".format(
                    data_type, source_dim, source_dim, batch_size))

                source_bands = 1

                if 'RGB' in data_type:

                    source_bands = 3

                losses = []
                losses_deltas = []

                try:

                    for k in range(3):

                        print("\tInitialization {}".format(k + 1), end=" - ")

                        try:

                            train, test = dataset(DBS[i]).separate_train_test(
                                batch_size=batch_size,
                                train_ratio=0.8)

                            cnn = DroneYieldMeanCNN(
                                source_bands=source_bands,
                                source_dim=source_dim,
                                optimizer=optimizer)

                            losses_dict = cnn.train(
                                epochs=3,
                                training_data=train,
                                test_data=test,
                                visualize=False,
                                suppress_output=True)

                        except RuntimeError as ex:

                            print(ex)

                        row_data = pd.Series()
                        row_data['source_data'] = data_type
                        row_data['source_dim'] = source_dim
                        row_data['batch_size'] = batch_size
                        row_data['iteration'] = k+1
                        row_data['best_loss'] = np.array(
                            losses_dict['test_losses_mean_std'])[:, 0].min()

                        df = df.append(row_data, ignore_index=True)

                        losses.append(
                            np.array(losses_dict['test_losses_mean_std'])[:, 0].min())
                        losses_deltas.append(
                            1 - np.min(losses_dict['test_losses']) / np.max(losses_dict['training_losses']))

                except Exception as ex:

                    print("Exception:", ex)
                    raise ex

                if len(losses) > 0 and len(losses_deltas) > 0:

                    losses = np.array(losses)
                    plt.scatter([batch_size] * len(losses), losses, alpha=0.5)
                    plt.errorbar(batch_size, losses.mean(),
                                 losses.std(), capsize=6, marker='o')

                batch_losses_deltas.append(np.mean(losses_deltas))

            plt.title('Best Test Losses for {} {}x{}'.format(
                data_type, source_dim, source_dim))
            plt.xlabel('Batch Size')
            plt.ylabel('$\mu_{Loss}$')
            plt.xticks(batch_sizes)
            plt.ylim(ymin=0)
            plt.xlim(16, 1040)
            plt.grid()

            plt.subplot(gs[1])
            plt.bar(batch_sizes, batch_losses_deltas, 20)
            plt.title('Mean Loss Reduction Ratio')
            plt.xlabel('Batch Size')
            plt.ylabel('$1-(L_{min}/L_{max})$')
            plt.xticks(batch_sizes)
            plt.ylim(0, 1)
            plt.xlim(16, 1040)
            plt.grid()

            plt.tight_layout()

            img_filename = f'optimizer_batch_test_{optimizer_name}_{data_type}_{source_dim}.png'
            plt.savefig(
                fname=os.path.join(RESULTS_DIR, img_filename),
                dpi=300)

    csv_filename = f'optimizer_batch_test_{optimizer_name}.csv'
    df.to_csv(os.path.join(RESULTS_DIR, csv_filename))


if __name__ == '__main__':

    main()
