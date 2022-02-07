# This file brings together the code necessary for running the baseline and melody models using k-fold cross-validation. It thus supports...
#   ... the code in the file ML_main.ipynb

# First import the necessary libraries and helper functions
import torch
import pytorch_lightning as lightning

from NN_data_helpers import JazzDataset
from LSTM_maker import JazzModel


def divide_data(cyclic_dataset, k, k_fold, output_options):
    '''
    This function is used to get the desired data split for the k'th fold of a k-fold cross-validation. It ensures that each data point is...
    ...present at least and and only once in the test set. The locally created validation and test sets each consist of one of the k data...
    ...blocks, the training set consists of the remaining k-2 data blocks.

    INPUTS
        cyclic_dataset:     The relevant dataset D as a numpy array and in a format where it is appended to itself horizontally (i.e. [D, D])

        k:                  The current iteration of the k-fold cross-validation

        k_fold:             The total number of iterations used for cross-validation

        output_options:     The possible output chords, necessary for initializing the class JazzDataset

    OUTPUTS
        Three instances of the class JazzDataset for the training, validation, and test data
    '''

    # Determine the length of each data block as well as the overshoot that arises from the number of data points not being a multiple of k_fold
    base_length = int((len(cyclic_dataset) / 2) // k_fold)
    overshoot = int((len(cyclic_dataset) / 2) % k_fold)

    # Define corrector terms that ensure that the overshoot is handled such that (considered over all k) all points occur in the test set once but not more than once
    if k < overshoot:
        position_corrector = 0
        val_corrector = 0
    else:
        position_corrector = k % overshoot
        val_corrector = 1

    if k >= overshoot:
        test_corrector = 1
    else:
        test_corrector = 0

    # Then split the data into the suitable ratios. The first k-2 blocks are for training, the k-1'th block is for validation, and the k'th block is for testing
    start_pos = k * (base_length + 1) - position_corrector
    train_end_pos = start_pos + (k_fold - 2) * base_length + overshoot - 1

    train_data = cyclic_dataset[start_pos : train_end_pos]
    val_data = cyclic_dataset[train_end_pos : train_end_pos + base_length + val_corrector]
    test_data = cyclic_dataset[train_end_pos + base_length + val_corrector - test_corrector: train_end_pos + 2 * base_length + 1 - test_corrector]

    # Finally create and return and instance of JazzDataset for each of the three data arrays
    return JazzDataset(train_data, output_options), JazzDataset(val_data, output_options), JazzDataset(test_data, output_options)


def run_model(train_data, val_data, test_data, Params, with_melody=False, counter=0):
    '''
    This function is used for running the ML model for chord prediction. It is possible to select the type of 
    model, i.e. with or without melody information with the parameter 'with_melody'.

    INPUTS
        train_data:             Instance of the class JazzDataset containing the training data for this iteration of the cross validation

        val-data:               Instance of the class JazzDataset containing the validation data for this iteration of the cross validation

        test_data:              Instance of the class JazzDataset containing the test data for this iteration of the cross validation

        Params:                 Dictionary containing the relevant parameters

        with_melody:            Bool specifying whether to run the baseline model (False) or the melody model (True)

        counter:                integer indicating the current iteration of the k-fold cross validation

    OUTPUTS
        test_predictions:       list containing the predictions made for the test set, as well as the associated melids

    '''
    
    # Modify certain parameters in Params based on which model is being used
    if with_melody:
        Params['num_lstm_layers'] = 2
        Params['weight_decay'] = 0.00008
        if Params['model_path_save']:
            Params['model_path_save'] = f"model_with_melody_{counter}.nn"

    else:
        Params['num_lstm_layers'] = 1
        Params['weight_decay'] = 0.001
        if Params['model_path_save']:
            Params['model_path_save'] = f"model_without_melody_{counter}.nn"

    # The early stopping callback is necessary to allow early stopping - the model stops training once the validation...
    #   ... error no longer decreases for a number "patience" of epochs. This reduces the amount of overfitting
    # The seed is set such that the study is repeatable

    lightning.seed_everything(1)
    early_stop_callback = lightning.callbacks.EarlyStopping(monitor='validation_loss', min_delta=0.00, patience=Params["patience"], verbose=False, mode='min')

    # This part initializes the ML model and the trainer, a Pytorch Lightning wrapper for Pytorch training (here with early stopping)

    if Params["train_on_gpu"]:
        trainer = lightning.Trainer(max_epochs=500, min_epochs=1, auto_lr_find=False, auto_scale_batch_size=True,
                                progress_bar_refresh_rate=10, callbacks=[early_stop_callback], gpus=1)
    else:
        trainer = lightning.Trainer(max_epochs=500, min_epochs=1, auto_lr_find=False, auto_scale_batch_size=True,
                                progress_bar_refresh_rate=10, callbacks=[early_stop_callback])

    model = JazzModel(train_data, val_data, test_data, Params = Params)

    # Next the training of the model is done. This part is the most time-intensive, especially if not using a GPU

    trainer.fit(model)

    if Params['model_path_save']:
        torch.save(model.state_dict(), Params['model_path_save'])

    # Then perform testing and return the results
    trainer.test(model)
    test_predictions = model.test_output

    return test_predictions

