'''
This file contains supporting functions for data_maker_helpers.py 

For more details and an explanation of the model architecture, please see the report on https://github.com/ldriever/ML_Jazz/


'''

#First import the necessary modules

import numpy as np
from mingus.core import chords

#Initiliaze a dictionary where index the position of every note in an octave starting at C/B#)
octave_index_dict = {"B#":0, 
                     "C": 0, 
                     "C#": 1, 
                     "Db": 1, 
                     "D": 2, 
                     "D#": 3, 
                     "Eb": 3, 
                     "E": 4, 
                     "Fb": 4, 
                     "E#": 5, 
                     "F": 5, 
                     "F#": 6, 
                     "Gb": 6, 
                     "G": 7, 
                     "G#": 8, 
                     "Ab": 8, 
                     "A": 9, 
                     "A#":10, 
                     "Bb":10, 
                     "B":11, 
                     "Cb": 11}

#Function to extract the root not of a chord notation
def root_note(chord):
  
    try:
        if chord[1] == "b" or chord[1] == "#":
            rootnote = chord[:2]
        else:
            rootnote = chord[:1]
    except:
        rootnote = chord[0]
    
    return rootnote

#Function to encode a chord to a multi-hot vector of size 12.
def chord2vec(all_chords):
    
    #initiliaze dictionary to store chords and their associated vector representation later
    chord_to_notes_dict = {}
    
    for chord in all_chords:
        
        # Transpose root to 'C'    
        chord_t = chord
        if chord != '':
            chord_t = chord_t.replace(root_note(chord_t), 'C')
        
        #Convert chord to note representation according to the mingus library
        # e.g., C7 = ['C', 'E', 'G', 'Bb']
        try:
            chord_to_notes_dict[chord] = chords.from_shorthand(chord_t)
        except:
            if chord != '':
                print('Could not find note representation for: {}'.format(chord))

     # Convert notes to vector representation, without taking base notes into account
    for chord in chord_to_notes_dict:
        
        #Initiliaze two vector for storing the transposed "chord" encoding and their associated root note encoding
        vector_representation = [0]*12
        root_note_vector = [0]*12

        for note in chord_to_notes_dict[chord]:

            if note[-2:] == "##":
                    note_position = (octave_index_dict[note[0]]+2)%12
            if note[-2:] == "bb":
                    note_position = (octave_index_dict[note[0]]-2)%12
            if (note[-2:] != "##") and (note[-2:] != "bb"):
                    note_position = octave_index_dict[note]

            vector_representation[note_position] = 1
        
        #If bass note is present, it will not be encoded in the vector
        if '/' in chord:
            bass_note = chord_to_notes_dict[chord][0]

            if bass_note not in chord_to_notes_dict[chord][1:]:
                note_position = octave_index_dict[bass_note]
                vector_representation[note_position] = 0
        
        #Create 1-hot vector for root note
        root_note_position = octave_index_dict[root_note(chord)]
        root_note_vector[root_note_position] = 1
        
        #Concatenote "chord" and "root note"
        chord_to_notes_dict[chord] = vector_representation + root_note_vector
    
    return chord_to_notes_dict

#Convert midi pitch the vector encoding
def midi2vec(midi):
    '''
    Defined for C3 as middle C
    '''
    
    #Initialize note and octave vector
    note = [0]*12
    octave = [0]*8
    
    #We define midi = -1 when no pitch was present at a unique coordinate (melid, bar, beat) 
    #If we do find a pitch, then convert it to their vector representation accordingly
    if midi != '-1':
        
        note_position = int(midi%12)
        octave_position = int(midi//12 - 2)

        note[note_position] = 1
        octave[octave_position] = 1
    
    return note+octave

#Function to map all chords to their chord quality
def chord2quality(all_chords):
    
    #Initialize dictionary to link chord to their chrod quality
    chord_to_group_dict = {}
    
    for chord in all_chords:
        
        if chord != '':
        
            original_chord = chord
            
            #Processing of chrod names so we can easily define chord quality based on suffix later.
            if chord[-1] == "7" or chord[-1] == "6":
                chord = chord[:-1]

            if "maj" in chord:
                chord = chord.replace("maj", "")

            if "m7b5" in chord:
                chord = chord.replace("m7b5", "dim")

            if "7+" in chord:
                chord = chord.replace("7+", "+")

            if "7#5" in chord:
                chord = chord.replace("7#5", "+")
            
            #Find chord quality based on suffix
            if chord[-1] == "-":
                category = "min"
            elif chord[-1] == "+":
                category = "aug"
            elif "dim" in chord:
                category = "dim"
            elif "sus4" in chord:
                category = "sus4"
            else:
                category = "maj"         
            
            #Store chord and its associated chord quality in the dictionary
            chord_to_group_dict[original_chord] = category
            
    return chord_to_group_dict