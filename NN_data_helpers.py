'''
The code in this file consists of auxilliary code that places the loaded data arrays in a format suitable to be handled by the ML model.
Key to this functionality is the class JazzDataset.
'''


# First import the necessary modules
from torch.utils.data import Dataset
import copy
import torch
import numpy as np


class JazzDataset(Dataset):
  '''
  This class take the input data and stores it in a suitable format.
  Importantly, the class also pads the data points, thereby allowing for data sequences of varying lengths.
  Related to this, the class also records the lengths of the different input tunes.

  The code is adapted from a demonstration by Andrew McLeod (EPFL)

INPUTS
    data:             A suitably formatted data array, containing dictionaries with entries "input" and "target" for each data point

    output_options    An iterable containing all possible output options that the model must be able to predict

OUTPUTS
    -   an initialized instance of the JazzDataset class

  '''
  def __init__(self, data, output_options):
    self.data = copy.deepcopy(data) # Essential to create a deepcopy, as otherwise the original is modified which can lead to errors if operations are repeated
    self.out_ops = output_options
    self.out_vocab_size = len(output_options)
    self.num_sequences = len(data)
    self.vocab_size = len(data[0]["input"][0])
    self.sequence_lengths = np.zeros(self.num_sequences, dtype=int)
    for i in range(len(data)): self.sequence_lengths[i] = (len(self.data[i]['input']))
    self.max_sequence_length = max(self.sequence_lengths)
    self.pad()

  def pad(self):
    '''
    This function handles the padding of the data points. For short data sequences, the targets are padded with -1s, an invalid output value for this problem
    '''
    for i in range(self.num_sequences):
      dat = self.data[i]

      # Storing the length of the original unpadded sequence
      dat["length"] = self.sequence_lengths[i]

      # Pad input with 0  arrays of vocab size, giving a total input length of max-sequence_length
      inpt_store = dat["input"]
      dat["input"] = torch.zeros((self.max_sequence_length, self.vocab_size))
      dat["input"][ : dat["length"]] = inpt_store

      # Pad target with some INVALID value (-1)
      trgt_store = dat["target"]
      dat["target"] = torch.ones(self.max_sequence_length, dtype=int) * -1
      dat["target"][ : dat["length"]] = trgt_store

  def __len__(self):
    return self.num_sequences

  def __getitem__(self, i):
    # Loading the relevant data point
    return self.data[i]


def split_data(data_array, output_options, Params, seed=1):
    '''
    This function acts as an easy interface between the loaded data array and the above class JazzDataset.
    It splits the data into training, validation, and testing data according to the split prescribed in the dictionary Params.
    For each of these sub-datasets an instance of JazzDataset is then created and returned

    INPUTS
      data_array        A suitably formatted data array containing all of the input data for the ML model

      output options    An iterable containing all possible output options that the model must be able to predict

      Params            Dictionary with entries "fraction_learning_data", "fraction_validation_data", and "fraction_testing_data", which must add up to 1
      
      seed              Optional seed to initialize the random shuffling of the input data

    OUTPUTS
      -   three instances of the class JazzDataset

    '''

    assert(Params["fraction_learning_data"] + Params["fraction_validation_data"] + Params["fraction_testing_data"] == 1)

    np.random.seed(1) # Essential to seed such that always the same data will end up in the different gorups
    data_array_copy = copy.deepcopy(data_array) # Shuffle a copy, not the original array
    np.random.shuffle(data_array_copy)

    split_learn = int(len(data_array_copy) * Params["fraction_learning_data"])
    split_validate = int(len(data_array_copy) * (Params["fraction_learning_data"] + Params["fraction_validation_data"]))

    train_data = JazzDataset(data_array_copy[ : split_learn], output_options)
    val_data = JazzDataset(data_array_copy[split_learn : split_validate], output_options)
    test_data = JazzDataset(data_array_copy[split_validate : ], output_options)

    return train_data, val_data, test_data