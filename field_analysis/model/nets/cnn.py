import os
import time
from datetime import datetime
from collections import OrderedDict

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import torch
from torch import nn, optim
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset, random_split

from ...settings import model as model_settings


class DroneYieldMeanCNN(nn.Module):
    """
    Class for predicting mean yield outputs from images with variable band counts. The high-level architecture of the network consists of Convolutional Neural Network and Linea fully-connected layers.

    Basic usage is the initialization and the training of the network:

        cnn = DroneYieldMeanCNN(source_bands=1, source_dim=32,
                    cnn_layers=2, learning_rate=1e-4,
                    momentum=0.7, debug=False)
        cnn.train(epochs=200,
                training_data=dataset,
                k_cv_folds=5)
    """

    CUDA = torch.cuda.is_available()
    FC_FEATURES = 1024
    MIN_CNN_CHANNELS = 64

    def __init__(self, source_bands, source_dim, cnn_layers=0, fc_layers=0, optimizer=None, optimizer_parameters=None, debug=False):
        """
        Args:

            source_bands: The number of bands in the source images.
            source_dim: The length of the side of the source images.
            cnn_layers: Optional. Number of total CNN layers to initialize the network with. Minimum is 2 and this is enforced in the implementation details. Default value is 0.
            fc_layers: Optional. Number of fully connected rectified linear layers to use. Minimum is 2 and this is enforced in the implementation details. Default value is 0.
            optimizer: Optional. An uninitialized `torch.optim` algorithm to be used during training. The optimizer parameters are to be provided separately.
            optimizer_parameters: Optional. The parameters for the optimizer to use. To see the optimizer, initialize the CNN and call its `optimizer` object. The paramters must be in dict with parameters as the keys. Refer to the optimizer-corresponding documentation found in http://pytorch.org/docs/0.4.1/optim.html.
            debug: Boolean for whether to print debug statements and follow debugging logic. Default is False.
        """

        super().__init__()

        self.source_bands = source_bands
        self.source_dim = source_dim
        self.debug = debug

        self.cnn_total_layers = max(2, cnn_layers)
        self.fc_total_layers = max(2, fc_layers)

        self.cnn_output = 0
        self.best_test_loss = 1e15
        self.best_model = None
        self.early_stopping_wait = 0

        self.log_debug("Creating layers:")
        self.cnn_layers = self.create_cnn_layers()
        self.fc_layers = self.create_fc_layers()

        self.log_debug("Initializing weights:")
        self.apply(self.initialize_weights)

        self.objective_loss = nn.L1Loss(reduction='none')

        if self.CUDA:
            self.cuda()

        self.optimizer = optim.SGD(
            params=self.parameters(), lr=1e-4, momentum=0.8)

        if optimizer is not None:

            self.optimizer = optimizer(params=self.parameters())

        if optimizer_parameters is not None:

            self.optimizer.defaults = {
                **self.optimizer.defaults,
                **optimizer_parameters
            }

        self.num_workers = 1 if self.debug else 8

        self.model_filename = "{}_{}_{}x{}x{}_{}cnn_{}fc.pkl".format(
            datetime.now().strftime("%Y%m%d_%H%M%S%f")[:-3],
            self.optimizer.__class__.__name__,
            self.source_bands,
            self.source_dim,
            self.source_dim,
            self.cnn_total_layers,
            self.fc_total_layers)
        self.model_path = os.path.join(
            model_settings.MODELS_DIR, self.model_filename)

        self.log_debug("Initialized {}".format(self))
        self.log_debug("Optimizer: {} {}".format(
            self.optimizer.__class__.__name__, self.optimizer.defaults))

    def log_debug(self, text):
        """
        Debug console logger.

        Args:

            text: A string of text.
        """
        if self.debug:
            print(text)

    def create_cnn_layers(self):
        """
        Create the CNN network structure by adding a Batch Normalized Convolution layer to layers with ReLU activation in-between possible MaxPooling. Doubles the channels from in to out when pooling is applied, but minimum out channels is still `self.MIN_CNN_CHANNELS`.

        The running dimension corresponding to the output array's side length is also calculated.

        Only the input and output layers are equipped with a max pooling operation. The first pooling's kernel size is calculated w.r.t. input image dimension to ensure consistent feature map dimensions for the remaining layers. This is to especially tackle a situation, where lasrge input image would result in +100k parameter input to the first FC layer and thus killing the training through memory overload.

        Returns:

            A `Sequential`-object containing the CNN layers.
        """
        layers_dict = OrderedDict()
        in_channels = self.source_bands
        in_dim = self.source_dim

        self.log_debug("\tCNN output dimensions:")
        if self.debug:
            dims = ["{}(in)".format(in_dim)]

        for i in range(self.cnn_total_layers):

            out_channels = in_channels
            out_dim = in_dim
            add_pooling = False

            if i == 0 or i == self.cnn_total_layers - 1:

                add_pooling = True
                out_channels = max(in_channels * 2, self.MIN_CNN_CHANNELS)
                pooling_kernel_size = 2

                if i == 0:
                    pooling_kernel_size = int(2/(32/in_dim))

            prefix = "conv{}_{}_{}".format(i, in_channels, out_channels)

            conv = prefix
            layers_dict[conv] = nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=5,
                stride=1,
                padding=2,
                bias=False)

            bnorm = prefix+"_bnorm"
            layers_dict[bnorm] = nn.BatchNorm2d(num_features=out_channels)

            relu = prefix + "_relu"
            layers_dict[relu] = nn.ReLU()

            if add_pooling:

                maxpool = prefix + "_maxpool"
                layers_dict[maxpool] = nn.MaxPool2d(
                    kernel_size=pooling_kernel_size)

                out_dim = out_dim // pooling_kernel_size

            in_channels = out_channels
            in_dim = out_dim

            if self.debug:
                dims.append(str(out_dim))

        if self.debug:
            self.log_debug("\t\t{}(out)".format(" -> ".join(dims)))

        self.cnn_output = out_channels * out_dim ** 2

        return nn.Sequential(layers_dict)

    def create_fc_layers(self):
        """
        Create the fully connected rectified linear network structure. The first layer's input corresponds to the output of the CNN. The last layer provides a single output

        Returns:

            A `Sequential`-object containing the FC layers.
        """
        layers_dict = OrderedDict()
        in_dim = self.cnn_output
        out_dim = self.FC_FEATURES

        self.log_debug("\tFC output dimensions:")
        if self.debug:
            dims = ["{}(in)".format(in_dim)]

        for i in range(self.fc_total_layers):

            if i == self.fc_total_layers - 1:
                out_dim = 1

            prefix = "fc{}_{}_{}".format(i, in_dim, out_dim)

            fc = prefix
            layers_dict[fc] = nn.Linear(
                in_features=in_dim,
                out_features=out_dim)

            relu = prefix+"_relu"
            layers_dict[relu] = nn.ReLU()

            in_dim = out_dim

            if self.debug:
                dims.append(str(out_dim))

        if self.debug:
            self.log_debug("\t\t{}(out)".format(" -> ".join(dims)))

        layers = nn.Sequential(layers_dict)

        return layers

    def initialize_weights(self, layer):
        """
        Initialize layers inplace with weights with Gaussian distribution. We will use zero mean and 0.2 standard deviation. AttributeError means that the layer has no weights.

        Args:

            layer: A layer of the network.
        """

        try:

            layer.weight.data.normal_(mean=0, std=0.2)
            weights = layer.weight.data.numpy()

            self.log_debug("\t{} weights: mean={:.4f}, std={:.4f}".format(
                layer,
                np.mean(weights),
                np.std(weights)))

        except AttributeError:

            pass

    def forward(self, x):
        """
        Perform forward pass with the network. After the CNN, the output is flattened to provide Linear fully connected layers their proper input.

        Args:

            x: A batch of input data.

        Returns:

            The output of the network with the corresponding batch of input data.
        """

        x = self.cnn_layers(x)

        self.log_debug("\t\tForward pass:")
        self.log_debug("\t\t\tPost-CNN shape={}".format(x.size()))

        x = x.view(-1, self.cnn_output)

        self.log_debug("\t\t\tPre-FC shape={}".format(x.size()))

        x = self.fc_layers(x)

        return x

    def process_batch(self, dataset, training):
        """
        Perform a forward pass and backprop for a batched dataset. When using a test set, the gradients are left out to speed the process of forward pass.

        Args:

            dataset: A batched dataset of instance `DataLoader`.
            training: Boolean for whether to treat the dataset as training or test set.

        Returns:

            A list of batch-wise losses.
        """

        losses = []
        dataset_name = 'Training'

        if not training:

            dataset_name = 'Test'

        if self.debug:

            i = 1

        for batch in dataset:

            with torch.set_grad_enabled(training):

                self.optimizer.zero_grad()

                x = Variable(batch['x'].float())
                y_true = Variable(batch['y'].float())

                if self.CUDA:

                    x = Variable(batch['x'].float().cuda())
                    y_true = Variable(batch['y'].float().cuda())

                self.log_debug("\t{} batch:".format(dataset_name))
                self.log_debug("\t\tInput type={}, shape={}"
                               .format(x.__class__.__name__, x.size()))
                self.log_debug("\t\tTargets type={}, shape={}"
                               .format(y_true.__class__.__name__, y_true.size()))
                self.log_debug("\t\tTargets={} ...".format(
                    y_true.view(-1).data[:3].tolist()))

                y_pred = self(x)
                self.log_debug("\t\tPredictions type={}, shape={}"
                               .format(y_pred.__class__.__name__, y_pred.size()))
                self.log_debug("\t\tPredictions={} ...".format(
                    y_pred.view(-1).data[:3].tolist()))

                loss = self.objective_loss(y_pred.view(-1), y_true)
                self.log_debug("\t\tLoss={}".format(
                    loss.tolist()[:3]))

                if training:

                    loss.mean().backward()
                    self.optimizer.step()

                losses.append(loss)

                del loss, y_pred, y_true, x

                if self.debug:

                    i += 1

                    if i > 2:

                        break

        losses = torch.cat(losses).tolist()
        self.log_debug("\t\tLosses={}".format(losses[:3]))

        return losses

    def calculate_model_mean_state(self, states):
        """Calculate the mean of trained model's module-wise parameters
        :param states: A list of model state dictionaries to ave
        """
        mean_state = {}
        for key in states[0].keys():
            try:
                values = []
                for state in states:
                    values.append(state[key])
                mean_state.update({key: sum(values)/len(values)})
            except RuntimeError as ex:
                print(f'Error with key: {key}')
                [print(value) for value in values]
                raise ex
        del states
        return mean_state

    def perform_cross_validation(self, k_cv_folds, dataset, cv_indices=None):
        """
        Perform cross-validation with the whole dataset by dividing it into `k_cv_folds` shuffled folds. The fold-wise training and test losses are kept track of. The batch_size is determined dynamically. This is to ensure large enough batches for higher utilization of the GPU and thus making the learning progress faster.

        Args:

            k_cv_folds: The number of CV folds to perform.
            dataset: A non-batch dataset of `Dataset` instance.
            cv_indices: Indices to use with CV from the Dataset. Workaround, assume pre-shuffled. Default is None.

        Returns:

            Tuple of lists of training and test losses.
        """
        if k_cv_folds is None:
            raise ValueError(
                "You need to specify `k_cv_folds` if no `test_data` is provided.")

        if not isinstance(dataset, Dataset):
            raise TypeError(
                "Only instances of `Dataset` allowed, as pre-batching messes CV fold-wise reindexing, inbound dataset type={}".format(type(dataset)))

        fold_samples = int(len(dataset)/k_cv_folds)
        fold_batch_size = 128

        self.log_debug("{}-fold CV, fold-wise samples={}, fold batch size={}".format(
            k_cv_folds,
            fold_samples,
            fold_batch_size))

        if cv_indices is None:
            shuffled_indices = (torch.randperm(
                len(dataset)).numpy().flatten().tolist())
        else:
            shuffled_indices = cv_indices

        model_initial_state = self.state_dict()
        model_cv_states = []
        training_losses = []
        validation_losses = []

        for fold in range(k_cv_folds):
            self.log_debug("Performing CV {}:".format(fold+1))
            self.load_state_dict(model_initial_state)
            if (fold + 1)*fold_samples < len(shuffled_indices):
                validation_set_indices = shuffled_indices[
                    fold * fold_samples: (fold+1)*fold_samples]
            else:
                validation_set_indices = shuffled_indices[fold*fold_samples:]
            training_set_indices = list(set(shuffled_indices)
                                        .difference(set(validation_set_indices)))
            self.log_debug("\tTraining set: samples={}"
                           .format(len(training_set_indices)))
            training_losses_cv = self.process_batch(
                dataset=DataLoader(
                    dataset[training_set_indices],
                    batch_size=fold_batch_size,
                    num_workers=self.num_workers),
                training=True)
            self.log_debug("\tTest set: samples={}, samples_idx_start={}"
                           .format(len(validation_set_indices),
                                   fold * fold_samples))
            validation_losses_cv = self.process_batch(
                dataset=DataLoader(
                    dataset[validation_set_indices],
                    batch_size=fold_batch_size,
                    num_workers=self.num_workers),
                training=False)
            training_losses += training_losses_cv
            validation_losses += validation_losses_cv
            model_cv_states.append(self.state_dict())

        self.load_state_dict(self.calculate_model_mean_state(model_cv_states))

        return training_losses, validation_losses

    def save_model(self, suppress_output=False):
        """
        Persist the CNN model state with a filename corresponding to network configuration.

        Args:

            suppress_output: Whether to suppress the output.
        """
        if not suppress_output:

            print("Saving the model to", self.model_path)

        torch.save(self.state_dict(), self.model_path)

    def load_model(self):
        """
        Load a persisted model with a filename corresponding to network configuration.
        """
        print("Reading the model from", self.model_path)
        self.load_state_dict(torch.load(self.model_path))

    def determine_early_stopping(self, patience, loss):
        """
        Measure calculated loss to `self.best_test_loss`. If the new loss is at least a percent lower, persist it and reset patience counter continuing the training. Else increase the counter until patience level is reached, persist the model state with best loss and send termination signal.

        Args:

            patience: The number of epochs to continue training without decrease in loss.
            loss: The calculated loss.

        Returns:

            Boolean for whether to terminate training or not.
        """
        terminate = False

        if loss < self.best_test_loss:

            self.best_test_loss = loss
            self.best_model = self.state_dict()
            self.early_stopping_wait = 0

        if patience is not None:

            if self.early_stopping_wait < patience:

                self.early_stopping_wait += 1

            else:

                self.load_state_dict(self.best_model)
                self.early_stopping_wait = 0
                terminate = True

        return terminate

    def visualize_training(self, training_losses, validation_losses, test_losses, save_path=None):
        """
        Visualize the training process with epoch-wise training and test loss means and standard deviations.

        Args:

            training_losses: List of [mean, std] pairs for training losses.
            test_losses: List of [mean, std] pairs for test losses.
            save_path: Path to save the image to. Default: None
        """
        min_losses = []

        train = np.array(training_losses)
        train_mean, train_std = train[:, 0], train[:, 1]
        min_losses.append(train_mean.min())

        if validation_losses:
            validation = np.array(validation_losses)
            validation_mean, validation_std = validation[:,
                                                         0], validation[:, 1]
            min_losses.append(validation_mean.min())

        test = np.array(test_losses)
        test_mean, test_std = test[:, 0], test[:, 1]
        min_losses.append(test_mean.min())

        min_value = min(min_losses)

        x = range(1, len(train)+1)
        lowest_test_error = [self.best_test_loss]*len(x)

        plt.rcParams['figure.figsize'] = 10, 4
        plt.plot(
            x, train_mean, label="CV Training Error", color='royalblue')
        plt.fill_between(
            x, train_mean+train_std, train_mean-train_std, alpha=0.15, color='royalblue')
        if validation_losses:
            plt.plot(
                x, validation_mean, label="CV Validation Error", color='forestgreen')
            plt.fill_between(
                x, validation_mean+validation_std, validation_mean-validation_std, alpha=0.15, color='forestgreen')
        plt.plot(
            x, test_mean, label="Validation Error", color='goldenrod')
        plt.fill_between(
            x, test_mean+test_std, test_mean-test_std, alpha=0.15, color='goldenrod')
        plt.plot(
            x, lowest_test_error, linestyle="--", alpha=0.6, color='goldenrod')
        plt.title("Epoch-wise loss averages over single/multi-fold batches")
        plt.xlabel("Epoch")
        plt.ylabel(self.objective_loss.__class__.__name__)
        plt.xlim(left=min(x), right=max(x))
        plt.ylim(bottom=min_value * 0.95, top=test_mean.mean()*1.5)
        plt.yscale('log')
        plt.xticks(x)
        plt.grid(which='minor', axis='y')
        plt.grid(which='major', axis='x')
        plt.gca().yaxis.set_major_formatter(ticker.NullFormatter())
        plt.gca().yaxis.set_minor_locator(ticker.MaxNLocator())
        plt.gca().yaxis.set_minor_formatter(ticker.ScalarFormatter())
        plt.gca().xaxis.set_major_locator(ticker.MaxNLocator())
        plt.gca().xaxis.set_major_formatter(ticker.ScalarFormatter())
        plt.legend()
        if save_path is not None:
            plt.savefig(save_path, dpi=200)
        plt.show()

    def split_test_train(self, dataset, test_split_ratio):
        """Exclude a test set from the whole dataset. The test set will never be used with the training. Negative split ratio is handled so that the training split contains all the samples and test split is left empty.
        :param dataset: An instance of a PyTorch Dataset class or an equivalent sequence.
        :param test_split_ratio: The ratio of test samples to split from the whole dataset.
        """
        split_ratio = min(max(0, test_split_ratio), 1)
        train_length = int((1-split_ratio)*len(dataset))
        shuffled_indices = (torch.randperm(
            len(dataset)).numpy().flatten().tolist())
        test_indices = shuffled_indices[train_length:]
        train_indices = shuffled_indices[:train_length]
        return test_indices, train_indices

    def train(self, epochs, training_data, test_data=None, k_cv_folds=None, early_stopping_patience=None, visualize=True, suppress_output=False, save_model=True):
        """
        Train the network for `epochs` number of full iterations with either CV or distinct training and test datasets.

        Args:

            epochs: The number of epochs to train the network for.
            training_data: Either a `Dataset` with samples or a `DataLoader` with batches depending on whether the plan is to use CV or a distinct test dataset.
            test_data: Optional. A `DataLoader` with samples to be used as the test dataset. Excludes the use of CV.
            k_cv_folds: Optional. The number of CV folds to perform. Required, if no test dataset is provided.
            early_stopping_patience: Optional. The number of training iterations (backprops) to tolerate where the test loss doesn't decrease before terminating the training.
            visualize: Optional. Whether to visualize the training or not. Defaults to True.
            suppress_output: Optional. Whether to suppress training time output. Defaults to False.

        Returns:

            A tuple of [mean, std] lists for both training and test losses.
        """
        t_start = time.time()
        if not suppress_output:
            print("Starting the training", end=" ")
            if self.CUDA:
                print("with GPU:")
            else:
                print("with CPU:")

        validation_split = 0.15
        if test_data is None:
            test_indices, training_indices = self.split_test_train(
                training_data, validation_split)

        training_losses = []
        validation_losses = []
        test_losses = []
        training_losses_mean_std = []
        validation_losses_mean_std = []
        test_losses_mean_std = []

        for epoch in range(1, epochs+1):

            if test_data is None:
                epoch_training_losses, epoch_validation_losses = self.perform_cross_validation(
                    k_cv_folds=k_cv_folds,
                    dataset=training_data,
                    cv_indices=training_indices)
                epoch_test_losses = self.process_batch(
                    dataset=DataLoader(
                        training_data[test_indices],
                        batch_size=128,
                        num_workers=self.num_workers),
                    training=False)

                validation_losses.append(epoch_validation_losses)
                validation_mean_loss = np.mean(epoch_validation_losses)
                validation_loss_std = np.std(epoch_validation_losses)
                validation_losses_mean_std.append(
                    [validation_mean_loss, validation_loss_std])

            else:
                epoch_training_losses = self.process_batch(
                    training_data,
                    training=True)
                epoch_test_losses = self.process_batch(
                    test_data,
                    training=False)

            training_losses.append(epoch_training_losses)
            training_mean_loss = np.mean(epoch_training_losses)
            training_loss_std = np.std(epoch_training_losses)
            training_losses_mean_std.append(
                [training_mean_loss, training_loss_std])

            test_losses.append(epoch_test_losses)
            test_mean_loss = np.mean(epoch_test_losses)
            test_loss_std = np.std(epoch_test_losses)
            test_losses_mean_std.append(
                [test_mean_loss, test_loss_std])

            terminate = self.determine_early_stopping(
                patience=early_stopping_patience,
                loss=test_mean_loss)

            if not suppress_output and (epoch % np.ceil(epochs/20) == 0 or terminate):
                t_delta = time.time() - t_start
                print("[{:4d}/{:4d}]".format(epoch, epochs), end=" ")
                print("({:.0f}m {:2.0f}s)".format(
                    t_delta // 60, t_delta % 60), end=" ")
                print("\tMean Loss:\tTrain={:4.2f} +-{:4.2f}\tTest={:4.2f} +-{:4.2f}".format(
                    training_mean_loss, training_loss_std, test_mean_loss, test_loss_std))

            if terminate:
                print("Early stopping criterion met, terminating training.")
                break

            if self.debug:
                break

        if save_model:
            self.save_model(suppress_output)
            print("Best Validation Loss: {:4.2f}".format(self.best_test_loss))

        if visualize and not self.debug:
            self.visualize_training(
                training_losses=training_losses_mean_std,
                validation_losses=validation_losses_mean_std,
                test_losses=test_losses_mean_std)

        return {
            'training_losses': training_losses,
            'validation_losses': validation_losses,
            'test_losses': test_losses,
            'training_losses_mean_std': training_losses_mean_std,
            'validation_losses_mean_std': validation_losses_mean_std,
            'test_losses_mean_std': test_losses_mean_std
        }

    def produce_test_errors(self, data):
        ids, dates = [], []
        corner_top_lefts, corner_bot_rights = [], []
        y_trues, y_preds = [], []
        for sample in data:
            ids.append(sample['id'])
            dates.append(sample['date'])
            corner_top_lefts.append(sample['corner_top_left'])
            corner_bot_rights.append(sample['corner_bot_right'])
            y_trues.append(sample['y'].item())
            y_preds.append(
                self(sample['x'].unsqueeze(0).float().cuda()).item())
        df = pd.DataFrame(columns=['y_true','y_pred','mae','block_id','date','corner_top_left','corner_bot_right'])
        df.loc[:,'y_true'] = y_trues
        df.loc[:,'y_pred'] = y_preds
        df.loc[:,'mae'] = (df.loc[:,'y_pred']-df.loc[:,'y_true']).abs()
        df.loc[:,'block_id'] = ids
        df.loc[:,'date'] = dates
        df.loc[:,'corner_top_left'] = corner_top_lefts
        df.loc[:,'corner_bot_right'] = corner_bot_rights
        df.to_csv(os.path.join(model_settings.RESULTS_DIR, f'{self.model_filename}.csv'))
        return {
            'mean_error':df.loc[:,'mae'].mean(),
            'errors':df.loc[:,'mae']
        }
        