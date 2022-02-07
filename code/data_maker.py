'''
This file contains the functional implementation processing the "beats" and "melody" tables from the Weimar Jazz Database (WJazzD). 

For more details and an explanation of the model architecture, please see the report on https://github.com/ldriever/ML_Jazz/

'''

# First import the necessary modules

import os

from sqlalchemy import create_engine
import pandas as pd

from data_maker_helpers import make_data_set


'''

This function has 4 arguments indicating:
- directory_name : name of data folder containing the WJazzD
- database_name  : name of the WJazzD
- melody_info    : boolean to indicate wether to include melody info or not
- remove_slashes : boolean to indicate wether to include bass note or not

'''

def make_data_array(directory_name, database_name, melody_info=True, remove_slashes=True):
    path = os.path.join(os.getcwd(), directory_name, database_name)
    engine = create_engine(f"sqlite:///{path}")

    beats = pd.read_sql("beats", engine)
    melody = pd.read_sql("melody", engine)

    return make_data_set(melody, beats, melody_info, remove_slashes)