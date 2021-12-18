# ML_Jazz

***Authors: Leonhard Driever, Mels Loe Jagt and Kuan Lon Vu***

The code in this repository is aimed at chord prediction using machine learning. The aim of the project is to investigate whether including melody information in the input data is able to improve prediction accuracy compared to a model that only takes chord progressions as input. This is investigated using the [Weimar Jazz Database](https://jazzomat.hfm-weimar.de/dbformat/dboverview.html) and LSTM neural networks.

All of the results obtained for this project can be replicated using the code in this repository. Overall, the code can be divided into two parts: data handling and machine learning tasks. The data handling files have as ultimate output the data files *output_options.pt*, *data_array_without_melody.pt* and *ata_array_with_melody.pt*. The machine learning files take these data files as input and produce the desired chord prediction accuracies.

The list below indicates in which group all of the files in this repository belong to. Not that there are also files corresponding to the final report of this project.

### Data Handling Files
- *wjazzd.db - the database file for the Weiar Jazz Database
- FILE
- FILE
- FILE

### Machine Learning Files
- *NN_data_helpers.py* - a python file with helper functions that convert the processed data into the format suitable to be handled by the ML model
- *LSTM_maker.py* - a python file in which the neural network is defined
- *ML_main.ipynb* - a python notebook that is used to run the ML model

### Report Files
- *Improving Chord Prediction in Jazz Music using Melody Information.pdf* - the final report documenting the project
- *loss_comparison_for_chords_only_model.png* - a plot of training and validation loss for the chords-only model. This is referred to in the report
- *loss_comparison_for_model_with_melody* - a plot of training and validation loss for the model with melody information. This is referred to in the report

# Running the Data Handling Files
STELLA WRITE HERE

# Running the ML Model
This can be done using the file *ML_main.ipynb*. This file can be run locally on a regular computer, but it is highly recommended to instead set it up on [Google Colab](https://colab.research.google.com/). This will require the user to either upload the files directly to Google Colab or to their Google Drive and to then mount their Google Drive in Colab. Also, dependencies such as Pytorch LIghtning may need to be installed. However, a full detailed explanation of how to use Google Colab is beyond the scope of this project.

When running the notebook it is important to ensure that all files (data and helper python files) are in the same relative locations as they are in this repository. Otherwise, if a different file tree is used, please make sure to adapt the file paths accordingly.

It is then possible to run the model by running the different cells consecutively in their original order in the notebook. More information on which parameters can be changed to get different outputs can be found in the notebook.