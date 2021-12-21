# ML_Jazz

***Authors: Leonhard Driever, Mels Loe Jagt and Kuan Lon Vu***

The code in this repository is aimed at chord prediction using machine learning. The aim of the project is to investigate whether including melody information in the input data is able to improve prediction accuracy compared to a model that only takes chord progressions as input. This is investigated using the [Weimar Jazz Database](https://jazzomat.hfm-weimar.de/dbformat/dboverview.html) and LSTM neural networks.

All of the results obtained for this project can be replicated using the code in this repository. Overall, the majority of the code can be divided into two parts: data handling and machine learning tasks. The data handling files have as ultimate output the data files `output_options.pt`, `data_array_without_melody.pt` and `data_array_with_melody.pt`. The machine learning files take these data files as input and produce the desired chord prediction accuracies. Therefore, it is important to note that there is no need to run the data handling files to generate the input data files as they are already provided as `.pt` files. Notice `data_array_with_melody.pt` is uploaded in compressed forma, therefore please unzip before using it. In addition, there is also a group of files linked to reporting. The only code in this group is the file `permutation_test.py` used to assess the statistical significance of the results for the report.

### Directory structure
```
ML_Jazz:
│   Improving Chord Prediction in Jazz Music using Melody Information.pdf
│   README.md
│
├───data processing
│   │   Data_Processing.ipynb
│   │   data_maker.py
│   │   data_maker_helpers.py
│   │   music_data_functions.py
│   │
│   └───datasets
│           wjazzd.db
│
├───figures
│       loss_comparison_for_chords_only_model.png
│       loss_comparison_for_model_with_melody.png
│
└───model
        LSTM_maker.py
        data_array_with_melody.pt.zip (Please unzip before using it)
        data_array_without_melody.pt
        output_options.pt
        permutation_test.py
        ML_main.ipynb
        NN_data_helpers.py
```

### Data Handling Files
- `data processing/datasets/wjazzd.db` - the database file for the Weiar Jazz Database. The file is larger than GitHubs file size limit and is thus uploaded in compressed format. Please unzip before use.
- `data processing/Data_Processing.ipynb` - a python notebook that is used to produce the input data for the model
- `data processing/data_maker.py` - a python file containing the function that is the entry point to generate the data for the model
- `data processing/data_maker_helpers.py` - a python file containing the helper functions that process the data provided in the WJazzD database for `data_maker.py`
- `data processing/music_data_functions` - a python file containing the helper functions for `data_maker_helpers.py`
- `model/output_options.pt` - the data file containing the set of possible output chords
- `model/data_array_without_melody.pt` - the created data file containing the inputs for the model using chords only
- `model/data_array_with_melody.pt` - the created data file containing the inputs for the model with melody information. The file is larger than GitHubs file size limit and is thus uploaded in compressed format. Please unzip before use.

### Machine Learning Files 
##### *(these can be found in the `model` folder)*
- `NN_data_helpers.py` - a python file with helper functions that convert the processed data into the format suitable to be handled by the ML model
- `LSTM_maker.py` - a python file in which the neural network is defined
- `ML_main.ipynb` - a python notebook that is used to run the ML model

### Report Files
- `Improving Chord Prediction in Jazz Music using Melody Information.pdf` - the final report documenting the project
- `figures/loss_comparison_for_chords_only_model.png` - a plot of training and validation loss for the chords-only model. This is referred to in the report
- `figures/loss_comparison_for_model_with_melody.png` - a plot of training and validation loss for the model with melody information. This is referred to in the report
- `model/permutation-test.py` - code used for testing the statistical significnace of the results

### Dependencies
Please note that, in order to run the code for this project, it is necessary to have the following Python libraries installed. Click the hyperlinks to visit the respective websites and find out more about the different dependencies.
- [Numpy](https://numpy.org/)
- [SQLAlchemy](https://www.sqlalchemy.org/)
- [Pandas](https://pandas.pydata.org/)
- [mingus](https://bspaans.github.io/python-mingus/)
- [Pytorch](https://pytorch.org/)
- [Pytorch Lightning](https://www.pytorchlightning.ai/)

# Running the Data Handling Files
To produce the data used for training the model, please run the `Data_processing.ipynb` in the `data processing` folder. The code is currently configured to produce the data for model with meoldy information. Therefore, for chord-only data, please set the `melody_info` to `False` in cell 3 of the notebook file. Please note that the computation time can be very long.

# Running the ML Model
This can be done using the file `ML_main.ipynb` in the `model` folder. This file can be run locally on a regular computer, but it is highly recommended to instead set it up on [Google Colab](https://colab.research.google.com/). This will require the user to either upload the files directly to Google Colab or to their Google Drive and to then mount their Google Drive in Colab. Also, dependencies such as Pytorch LIghtning may need to be installed. However, a full detailed explanation of how to use Google Colab is beyond the scope of this project.

When running the notebook it is important to ensure that all files (data and helper python files) are in the same relative locations as they are in this repository. Otherwise, if a different file tree is used, please make sure to adapt the file paths accordingly.

It is then possible to run the model by running the different cells consecutively in their original order in the notebook. More information on which parameters can be changed to get different outputs can be found in the notebook.

# Running the Permutation Test
When running the file `permutation_test.py` please first run the ML model as this will create the necessary files `test_targets_melody.lst`, `test_predictions_melody.lst`, and `test_predictions_chords_only.lst`. Then, after ensuring that these files are in the same directory as the `permutation_test.py` file, simply run the file like a normal python file.

# Citation for external libraries
- Harris, C.R., Millman, K.J., van der Walt, S.J. et al. Array programming with NumPy. Nature 585, 357–362 (2020). DOI: 10.1038/s41586-020-2649-2. (Publisher link).
- M. Bayer, “Sqlalchemy,” in The Architecture of Open Source Applications Volume II: Structure, Scale, and a Few More Fearless Hacks, A. Brown and G. Wilson, Eds. aosabook.org, 2012.
- Wes  McKinney,  "Data  Structures  for  Statistical  Computing  in Python," in *Proceedings of the 9th Python in Science Conference*, Stefan  van  der  Walt  and  Jarrod  Millman,  Eds.,  2010,  pp.  56  –61.
- A. Paszke, S. Gross, F. Massa, A. Lerer, J. Bradbury, G. Chanan,T.  Killeen,  Z.  Lin,  N.  Gimelshein,  L.  Antiga,  A.  Desmaison,A.    Kopf,    E.    Yang,    Z.    DeVito,    M.    Raison,    A.    Tejani, S.  Chilamkurthy,  B.  Steiner,  L.  Fang,  J.  Bai,  and  S.  Chintala,“Pytorch:  An  imperative  style,  high-performance  deep  learning library,"  in *Advances in Neural Information Processing Systems 32*,  H.  Wallach,  H.  Larochelle,  A.  Beygelzimer,  F.  d'Alche-Buc, E. Fox, and R. Garnett, Eds.    Curran Associates, Inc., 2019, pp. 8024–8035.   [Online].   Available:   http://papers.neurips.cc/paper/9015-pytorch-an-imperative-style-high-performance-deep-learning-library.pd
- T. Killeen, et.al. “Pytorch: An imperative style, high-performance deep learning library,” in Advances in Neural Information Processing Systems 32, H. Wallach, H. Larochelle, A. Beygelzimer, F. d'Alch ́e-Buc, E. Fox, and R. Garnett, Eds. Curran Associates, Inc., 2019, pp. 8024–8035.
- Falcon, W., & The PyTorch Lightning team. (2019). PyTorch Lightning (Version 1.4) [Computer software]. https://doi.org/10.5281/zenodo.3828935
- Spaans, B. (2020). Mingus (Version 0.6.1) [Computer software]. http://bspaans.github.io/python-mingus/