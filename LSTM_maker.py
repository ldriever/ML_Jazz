'''
This file contains the class JazzModel, which is the ML model used for this project.
For more details and an explanation of the model architecture, please see the report on https://github.com/ldriever/ML_Jazz/
'''


# First import the necessary modules
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pytorch_lightning as lightning
import numpy as np
import pickle


class JazzModel(lightning.LightningModule):
    '''
    This class uses Pytorch Lighning to set up the machine learning model for the desired parameters.
    The model consists of a dense embedding layer, one or multiple LSTM layers, and a dense output layer.
    The applicable hyperparameters are input into the model using the dictionary Params.

    The code is adapted in parts from Sandro Luck: https://towardsdatascience.com/pytorch-lightning-machine-learning-zero-to-hero-in-75-lines-of-code-7892f3ba83c0

    INPUTS
        train_data:     An instance of the class JazzDataset for the training data

        val_data:       An instance of the class JazzDataset for the validation data

        test_data:      An instance of the class JazzDataset for the test data

        Params          A dictionary containing the desired hyperparameters for the model

    OUTPUTS
        when validating and testing the model, four files are automatically saved by this class. They...
        ... contain the targets and predictions for validation and test data. Furthermore, the class also...
        ... save log entries about the use of the model in a directory called lightning_logs

    '''
    def __init__(self, train_data, val_data, test_data, Params):
        super(JazzModel, self).__init__()
        self.output_size = train_data.out_vocab_size
        self.vocab_size = train_data.vocab_size
        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data
        self.lr = Params["learning_rate"]
        self.batch_size = Params["batch_size"]
        self.embedding_size = Params["embedding_size"]
        self.lstm_hidden_size = Params["lstm_hidden_size"]
        self.num_lstm_layers = Params["num_lstm_layers"]
        self.detailed_validation = False
        self.test_count = 0
        self.test_output = []
        self.test_targets = []
        self.weight_decay = Params['weight_decay']

        self.lstm = nn.LSTM(
            self.embedding_size,
            self.lstm_hidden_size,
            num_layers=self.num_lstm_layers,
            bidirectional=False,
            batch_first=True,
        )

        self.input = nn.Linear(self.vocab_size, self.embedding_size)
        self.output = nn.Linear(self.lstm_hidden_size, self.output_size)

    def forward(self, inputs, lengths):
        '''
        Defines how a forward pass is run for the model
        '''
        embedded_input = F.relu(self.input(inputs))
        packed = pack_padded_sequence(embedded_input, lengths.cpu(), enforce_sorted=False, batch_first=True)
        lstm_out_packed, _ = self.lstm(packed)
        lstm_out, _ = pad_packed_sequence(lstm_out_packed, batch_first=True)
        relu1 = F.relu(lstm_out)

        output = self.output(relu1)

        return output  

    def configure_optimizers(self):
        '''
        Defines that the Adam algorithm is used as optimizer
        '''
        return optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

    def train_dataloader(self):
        '''
        Allows the model to access the training data
        '''
        loader = DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True)
        return loader

    def training_step(self, batch, batch_nb):
        '''
        This function calculates the output and subsequently the loss for the data points in the training set
        '''

        inpt, lengths, target = batch["input"], batch["length"], batch["target"][:, :max(batch["length"])]
        outpt = self(inpt, lengths)

        loss = F.cross_entropy(outpt.transpose(1, 2), target, ignore_index=-1)
        
        self.log("training_loss", loss)
        return {'loss': loss, 'log': {'train_loss': loss}}

    
    def val_dataloader(self):
        '''
        Allows the model to access the validation data
        '''
        loader = DataLoader(self.val_data, batch_size=self.batch_size, shuffle=False)
        return loader

    def validation_step(self, batch, batch_nb):
        '''
        This function calculates the output and prediction accuracy for tunes in the validation set
        '''
        inpt, lengths, target, melids = batch["input"], batch["length"], batch["target"][:, :max(batch["length"])], batch["melid"]
        outpt = self(inpt, lengths)
        loss = F.cross_entropy(outpt.transpose(1, 2), target, ignore_index=-1)

        if self.detailed_validation:
            predictions = []
            target_list = []
            lengths = []
            accuracies = []
            for i in range(len(outpt)):
                accuracy_i = self.get_accuracy(outpt[i], target[i])
                self.log("validation_accuracy_%i"  % (melids[i]), accuracy_i)
                accuracies.append(accuracy_i)

                predict = outpt[i].argmax(dim=1).flatten()
                targets_local = target[i].flatten()
                mask = targets_local != -1

                predictions.append(predict[mask].tolist())
                target_list.append(targets_local[mask].tolist())
                lengths.append(sum(mask))

            with open('validation_predictions.lst', "wb") as fp:
                pickle.dump(predictions, fp)
            with open('validation_targets.lst', "wb") as fp:
                pickle.dump(target_list, fp)

            accuracy_all = sum(np.array(accuracies) * np.array(lengths)) / sum(lengths)
            self.log("validation_accuracy_all", accuracy_all)

        self.log("validation_loss", loss)
        return {'val_loss': loss, 'log': {'val_loss': loss}, 'predictions': [1,2,3]}

    def validation_epoch_end(self, outputs):
        '''
        This function documents the validation loss, which is necessary for early stopping
        '''
        val_loss_mean = sum([o['val_loss'] for o in outputs]) / len(outputs)
        # show val_acc in progress bar but only log val_loss
        results = {'progress_bar': {'val_loss': val_loss_mean.item()}, 'log': {'val_loss': val_loss_mean.item()},
                   'val_loss': val_loss_mean.item()}

        return results

    def test_dataloader(self):
        '''
        Allows the model to access the test data
        '''
        loader = DataLoader(self.test_data, shuffle=False)
        return loader
    
    def test_step(self, batch, batch_idx):
        '''
        This function calculates the test output and prediction accuracy for each tune in the test set
        '''
        inpt, lengths, target, melid = batch["input"], batch["length"], batch["target"][:, :max(batch["length"])], batch['melid']
        outpt = self(inpt, lengths)

        loss = F.cross_entropy(outpt.transpose(1, 2), target, ignore_index=-1)
        
        accuracy = self.get_accuracy(outpt[0], target)
        self.log("test_accuracy_%i"  % (melid), accuracy)

        predict = outpt[0].argmax(dim=1).flatten()
        targets_local = target.flatten()
        mask = targets_local != -1

        self.test_output.append(predict[mask].tolist())
        self.test_targets.append(targets_local[mask].tolist())

        self.test_count += 1

        return[len(outpt[0]), loss, accuracy]

    def test_epoch_end(self, results):
        '''
        This function calculates the total weighted test accuracy and documents the results
        '''
        test_song_lengths, test_song_losses, test_song_accuracies = np.hsplit(np.array(results).T, 1)[0]
        
        total_length = sum(test_song_lengths)
        accuracy_all = (test_song_lengths @ test_song_accuracies[..., None] / total_length)[0]
        self.log("test_accuracy_all", accuracy_all)

        loss_all = (test_song_lengths @ test_song_losses[..., None] / total_length)[0]
        self.log("test_loss", loss_all)

        with open('test_predictions.lst', "wb") as fp:
                pickle.dump(self.test_output, fp)
        with open('test_targets.lst', "wb") as fp:
                pickle.dump(self.test_targets, fp)
        
    def get_accuracy(self, outpt, target, type="one_song"):
        '''
        This function is used for finding the prediction accuracy for a matching pair of predictions (called outputs) and targets
        '''

        if type == "all":
            flat_outputs = outpt.argmax(dim=2).flatten()
        elif type == "one_song":
            flat_outputs = outpt.argmax(dim=1).flatten()
        else: raise ValueError("Invalid type specified for accuracy function")
        flat_targets = target.flatten()

        # Mask the outputs and targets
        mask = flat_targets != -1

        return 100 * (flat_outputs[mask] == flat_targets[mask]).sum().item() / sum(mask).item()
