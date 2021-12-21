"""
This file contains the code used for running the permutation test on the output data.
It uses as inputs the targets for the test set as well as the predictions of both model on this set
"""

# First import the necessary modules
import pickle
import numpy as np

# Next read the applicable files into the program
with open("test_targets_melody.lst", 'rb') as f:
    targets = pickle.load(f)

with open("test_predictions_melody.lst", 'rb') as f:
    melody_predictions = pickle.load(f)


with open("test_predictions_chords_only.lst", 'rb') as f:
    chords_predictions = pickle.load(f)

# Then calculate the accuracy for each song in the test set for both models
melody_accuracies = []
chord_accuracies = []
lengths = []

for i in range(len(targets)):
    lengths.append(len(targets[i]))
    count_mel = 0
    count_chord = 0
    for j in range(len(targets[i])):
        if targets[i][j] == melody_predictions[i][j]: count_mel +=1
        if targets[i][j] == chords_predictions[i][j]: count_chord +=1
    melody_accuracies.append(count_mel / lengths[-1])
    chord_accuracies.append(count_chord / lengths[-1])

# Next determine the weighted averages for both models and also the difference between the averages
mel_avg = np.average(melody_accuracies, weights=lengths)
chord_avg = np.average(chord_accuracies, weights=lengths)
observed_mean = mel_avg - chord_avg

# Next perform the permutation test, making sure to always split the data in half and to use weighted averages
np.random.seed(1)

data = np.array([chord_accuracies + melody_accuracies, lengths + lengths]).T

mean_diff = []
for i in range(50000):

    np.random.shuffle(data)
    array_split = np.split(data, [int(len(data)/2)], axis=0)
    array_1 = array_split[0]
    array_2 = array_split[1]
    mean_diff.append(np.average(array_1[:, 0], weights=array_1[:, 1]) - np.average(array_2[:, 0], weights=array_2[:, 1]))

# Then use the results to calculate and output the p vale
sum = 0
for item in mean_diff:
    if abs(item) >= observed_mean:
        sum +=1

p_value = sum / 50000

print("The p value as determined by the permutation test is: ", p_value)




