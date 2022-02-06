'''
This file contain the function for filling the "bar", "bar" and "pitches" columns in the final output csv file.

For more details and an overview of the whole project, please see the full repo at https://github.com/ldriever/ML_Jazz/
'''

# First import the necessary modules

import os
import csv

import numpy as np
import torch
from sqlalchemy import create_engine
import pandas as pd

from data_maker_helpers import transform_chords

# Import and load relevant datasets

current_path = os.path.dirname(os.path.realpath(__file__))
path = os.path.join(current_path, 'datasets', 'wjazzd.db')
engine = create_engine(f"sqlite:///{path}")

beats = pd.read_sql("beats", engine)
melody = pd.read_sql("melody", engine)

df = pd.read_csv(current_path + '/datasets/output_data_array.csv').drop(labels=["Unnamed: 0", "Unnamed: 0.1"], axis=1)
df = df.astype({"notes": str})

#Get all chords
all_chords = beats['chord'].unique()

#Reduce and rename the chords in the beats table according to the transform_chords() function
for chord in all_chords:
    beats["chord"] = beats["chord"].replace(chord, transform_chords([chord])[0])

#Filter our first chord of every melid
data_beats = beats[["melid","bar", "beat", "chord"]][(beats['chord'] != "") & (beats['chord'] != "NC")].reset_index(drop=True)

def mask_first(x):
    result = np.ones_like(x)
    result[0] = 0
    return result

mask = data_beats.groupby(['melid'])['melid'].transform(mask_first).astype(bool)
data_beats = data_beats.loc[mask].reset_index(drop=True)

#Update "bar" column in dataframe
df["bar"] = data_beats["bar"]

#Update "beat" column in dataframe
df["beat"] = data_beats["beat"]

#Get relevant columns of melody dataset
data_melody = melody[['melid', 'bar', 'beat', 'pitch']]

#Function for filtering all rows in data_melody dataset between 2 coordinates defined by (bar, beat, melid) corresponding to 2 succesive chords
def chord_interval(min_melid, min_bar, min_beat, max_melid, max_bar, max_beat, dataset):

    proceed = True
    first = True
    while proceed:
        
        if not ((min_bar >= max_bar) and (min_beat >= max_beat)):

            dataset_filter = dataset[dataset['melid'] == min_melid]
            dataset_filter = dataset_filter[dataset_filter['bar'] == min_bar]
            dataset_filter = dataset_filter[dataset_filter['beat'] == min_beat]
            
            if dataset_filter.empty:
                if min_beat == 4:                    
                    min_bar += 1
                    min_beat = 1
                else:
                    min_beat += 1
            else:
                if first:
                    collection = dataset_filter.to_numpy()
                    first = False
                else:
                    collection = np.vstack((collection, dataset_filter.to_numpy()))

                if min_beat == 4:  
                    min_bar += 1
                    min_beat = 1
                else:
                    min_beat += 1

        else:
            proceed = False

    try:
        return pd.DataFrame(collection, columns = ['melid','bar','beat', 'pitch']).astype("int64")
    except: 
        return pd.DataFrame([[0, 0, 0, ""]], columns = ['melid','bar','beat', 'pitch'])

for row_index, row in df.iterrows():
    
    if row_index < 29701:
        min_row = df.iloc[row_index]
        min_melid = min_row['melid']
        min_bar = min_row['bar']
        min_beat = min_row['beat']

        max_row = df.iloc[row_index+1]
        max_melid = max_row['melid']
        max_bar = max_row['bar']
        max_beat = max_row['beat']

        if min_melid == max_melid:    
            df_interval = chord_interval(min_melid, 
                                         min_bar,
                                         min_beat,
                                         max_melid,
                                         max_bar,
                                         max_beat,
                                         data_melody)
        else:

            last_note = data_melody[data_melody["melid"] == min_melid].iloc[-1]
            max_melid = last_note["melid"]
            max_bar = last_note["bar"]
            max_beat = last_note["beat"]
            
            if max_beat == 4:
                max_bar += 1
                max_beat = 1
            else:
                max_beat += 1

            df_interval = chord_interval(min_melid, 
                                         min_bar,
                                         min_beat,
                                         max_melid,
                                         max_bar,
                                         max_beat,
                                         data_melody)
    else:
        min_row = df.iloc[row_index]
        min_melid = min_row['melid']
        min_bar = min_row['bar']
        min_beat = min_row['beat']
        
        last_note = data_melody[data_melody["melid"] == min_melid].iloc[-1]
        max_melid = last_note["melid"]
        max_bar = last_note["bar"]
        max_beat = last_note["beat"]
        
        if max_beat == 4:
            max_bar += 1
            max_beat = 1
        else:
            max_beat += 1

        df_interval = chord_interval(min_melid, 
                                     min_bar,
                                     min_beat,
                                     max_melid,
                                     max_bar,
                                     max_beat,
                                     data_melody)
        
    pitches = ','.join(str(i) for i in df_interval["pitch"].tolist())
    
    df.at[row_index, "notes"] = pitches
    
df.to_csv(current_path + "/output_csv.csv", index = False)
    

        
        