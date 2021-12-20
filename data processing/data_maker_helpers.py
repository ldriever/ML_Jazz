'''
This file contains the helper functions for data_maker.py for processing the "beats" and "melody" tables from the Weimar Jazz Database (WJazzD). 

For more details and an explanation of the model architecture, please see the report on https://github.com/ldriever/ML_Jazz/

'''

# First import the necessary modules

import pandas as pd
import numpy as np
import torch
from mingus.core import chords
import re

from music_data_functions import chord2vec, midi2vec, chord2quality


'''
This function will return a list containing within each element a dictionary containing the keys "input", "target" and "melid". This way for every unique melody (under the constraint that at least 2 chords are present within the melid) a unique entry in the list will be created.

For the input arguments we have:
- inputs       : array of two column; melids and multi-hot vectors
- targets      : array of two coluns; melids and targets
- num_sequence : number of unique melids in total

'''

def transform_data(inputs, targets, num_sequences):
    
    #Initialze list for storing the data dictionaries
    DS = [0] * num_sequences
    
    #Initialzie empty string for collecting melids containing only 0 or 1 chord.
    missing_melids = []
    
    for i in range(len(DS)):
        
        #Constraint for checking if current melid has at least 2 chords
        if len(inputs[:,1][inputs[:,0] == i+1]) > 1:
            
            #Skip to the moment the first chord if being played.
            try:
                first_idx = np.where(targets[:, 1][targets[:,0] == i+1] != -1)[0][0]
            except:
                first_idx = 0
        
            input_melid = inputs[:,1][inputs[:,0] == i+1][first_idx : ]
            target_melid = targets[:,1][targets[:,0] == i+1][first_idx : ]
            
            #Stack inputs and targets associated to a unique melid
            sequence_inputs = np.stack(input_melid, axis=0)
            sequence_targets = np.stack(target_melid, axis=0) 
            
            #For melody_info = True, check if another chord is played after the first.
            #This code was neccesary in addition to the previous contstraint as the first
            #constraint does not take into account the melody_info = True variant where 
            #a melid with pitches only can still have len() > 1.
            
            if np.any(sequence_targets[1:] != -1):
                DS[i] = {   "input": torch.from_numpy(sequence_inputs[:-1]),
                            "target": torch.from_numpy(sequence_targets[1:]),
                            "melid": i+1}
            else:
                missing_melids.append(i+1)
        else:
            missing_melids.append(i+1)
                
    
    return np.array(DS), missing_melids

'''

The function below performs the chord reduction. Here we remove chord extensions beyond the 7th and we remove the bass note. Furthermore, we change some notations (e.g. o --> dim) to be compatible with the mingus library that we use to convert chord notations to their respective notes representation.

input:
- all_chord       : All unique chords to be transformed
- remove_slashes  : boolean to indicate wether to include bass note or not

'''

def transform_chords(all_chords, remove_slashes=True):
    
    #Initialize dataset for storing the newly tranformed chords
    dataset = ['0'] * len(all_chords)
    
    for i, chord in enumerate(all_chords):
        
        #Extract bass note of chord if present
        slash = re.findall("/.+", chord) 
        if len(slash) == 0:
            slash = ""
        else:
            slash = slash[0]
        
        #Remove bass note if present
        chord = chord.replace(slash, "")

        #Replace NC by empty string
        if chord == "NC":
            chord = ''
        
        #Remove "alt" notation
        if "alt" in chord:
            chord = chord.replace("alt", "")
        
        #Remove extensions beyond 7th's
        match = re.findall("9.*", chord)
        if match:
            chord = chord.replace(match[0], "")
        
        #Rename sus to sus4 (all the sus in the database are sus4's)
        match = re.findall("sus", chord)
        if match: 
            chord = chord.replace(match[0], "sus4")
            
        #Rename o to dim
        if "o" in chord:
            chord = chord.replace("o", "dim")
        
        #Rename the augmented major 7th
        if "+j7" in chord:
            chord = chord.replace("+j7", "7+")
        
        #Rename the major 7th
        if "j7" in chord:
            chord = chord.replace("j7", "maj7") 
            
        #Rename augmented minor 7th 
        if "+7" in chord:
            chord = chord.replace("+7", "7#5") 

        #Slashes are removed to reduce the total number of chords
        if remove_slashes:
            dataset[i] = chord
        else:
            #Concat the slash back in, if there was one
            dataset[i] = chord + slash
            
    return dataset


'''

The function below is the main function that coordinates the formation of the desired output to be fed into the model.

input:
- melody         : melody table from WJazzD
- beats          : beats tabe from WJazzD
- melody_info    : boolean to indicate wether to include melody info or not
- remove_slashes : boolean to indicate wether to include bass note or not

'''

def make_data_set(melody, beats, melody_info, remove_slashes):
    
    #Get all chords
    all_chords = beats['chord'].unique()
    
    #Reduce and rename the chords in the beats table according to the transform_chords() function
    for chord in all_chords:
        beats["chord"] = beats["chord"].replace(chord, transform_chords([chord])[0])

    #Create a seperate independent list of unique transformed chords
    all_chords = transform_chords(all_chords, remove_slashes)

    #Make all chords unique. After transformation some previously unique chords are mapped to the same chord (e.g. C79 and C711 --> C7)
    all_chords = sorted(list(set(all_chords)))
    
    #Create dictionary of all chords and their corresponding chord quality
    all_chords_quality = chord2quality(all_chords)

    #Get vector representation of every vector
    chord_to_notes_dict = chord2vec(all_chords)
    
    #Empty chords will be encoded by a zero vector
    chord_to_notes_dict[''] = [0]*24

    # Dictionary for every unique chord to a unique index (used for targets later)
    positions = np.arange(len(all_chords))
    chord_to_position = dict(zip(all_chords, positions-1))
    position_to_chord = dict(zip(positions-1, all_chords))

  
    #Extract relevant columns of the beats table   
    data_beats = beats[['melid', 'bar', 'beat', 'chord']]
    data_melody = np.array(list(zip(melody['melid'], 
                                    melody['bar'], 
                                    melody['beat'],
                                    melody['pitch'],
                                    melody['duration']/melody['beatdur'])), dtype=object)
    
    #Initialize inputs and targets by empty strings
    inputs = []
    targets = []
    
    #Initialize melid
    melid = 0
    
    '''
    
    In the for loop below, every unique beat will be located by their coordinate (melid, bar, beat). Then
    for every distinct beats the associated chords and/or pitch(es) will be tracked.
    
    '''
    for index, row in data_beats.iterrows():
        
        #Print statement to keep track of progress        
        if melody_info:
            if index == 0:
                print("\nmelody_info = True:")
            if index%10000 == 0:
                print("Current melid: {}".format(index+1))
        else:
            if index == 0:
                print("\nmelody_info = False:")
            if index%10000 == 0:
                print("Current melid: {}".format(index+1))
            
        
        #Look in melody table for the specific coordinate (melid, bar, beat) and their associated pitch(es)
        data_melody_filter = data_melody[data_melody[:,0] == row['melid']]
        data_melody_filter = data_melody_filter[data_melody_filter[:,1] == row['bar']]
        data_melody_filter = data_melody_filter[data_melody_filter[:,2] == row['beat']]

        #Find chord for current coordinate (melid, bar, beat)
        chord = row['chord']
        
        #If a new melid is reached, update melid tracker. Also define new_chord = ''. This will reset the
        #chord reference when we enter a new solo (melid)
        if row['melid'] != melid:
            new_chord = ''
            melid += 1
        
        #If a new chord appears, update chord reference
        if chord != '':
            new_chord = chord
        
        #If there are no associated pitch(es) to a coordinate (melid, bar, beat)
        if data_melody_filter.size == 0:
            pitch = midi2vec("-1")
            
            #Including melody information
            if melody_info:
                
                #Create input vector
                inputs.append([int(row['melid']), np.array(chord_to_notes_dict[new_chord] + pitch + [0])])
                
                #Create associated target value
                if chord == '':
                    targets.append([int(row['melid']), -1])
                else:
                    targets.append([int(row['melid']), chord_to_position[chord]])
                    
            #Excluding melody information     
            elif chord != '':
                #Create input vector and associated target value
                inputs.append([int(row['melid']), np.array(chord_to_notes_dict[chord])])
                targets.append([int(row['melid']), chord_to_position[chord]])
                
        #If at least 1 or more pitches are associated with the current coordinate (melid, bar, beat)
        else:
            
            #We only want predict the chrod based on the last pitch before a new chord appears. This can be ensured using counter.
            counter = 0
            for element in data_melody_filter:
                pitch = midi2vec(element[3])
                
                #Including melody information
                if melody_info:
                    
                    #Create associated target value
                    inputs.append([int(row['melid']), np.array(chord_to_notes_dict[new_chord] + pitch + [element[4]])])
                    
                    #Create associated target value
                    if chord != '' and counter == 0:
                        targets.append([int(row['melid']), chord_to_position[chord]])
                        counter += 1                 
                    else:
                        targets.append([int(row['melid']), -1])
                
                #Excluding melody information
                elif chord != '':
                    if counter == 0:
                        #Create input vector and associated target value
                        inputs.append([int(row['melid']), np.array(chord_to_notes_dict[chord])])
                        targets.append([int(row['melid']), chord_to_position[chord]]) 
                        counter += 1

    #Convert inputs and targets to numpy format. Extract num_sequences based on last appearing melid
    inputs = np.array(inputs, dtype = object)
    num_sequences = inputs[-1, 0]
    targets = np.array(targets, dtype = object)
    
    #Create data_array in correct format
    data_array, missing_melids = transform_data(inputs, targets, num_sequences)
    
    #Remove empty entries corresponding to melids with only 0 or 1 chord
    data_array = np.delete(data_array, np.argwhere(data_array == [0]).squeeze())

    return data_array, all_chords, all_chords_quality, missing_melids