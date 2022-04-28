# ML_Jazz

***Authors: Leonhard Driever, Mels Loe Jagt, Kuan Lon Vu, Daniel Harasim, Andrew McLeod, and Martin Rohrmeier***

The code in this repository is aimed at chord prediction using machine learning. The aim of the project is to investigate whether including melody information in the input data is able to improve prediction accuracy compared to a model that only takes chord progressions as input. This is investigated using the [Weimar Jazz Database](https://jazzomat.hfm-weimar.de/dbformat/dboverview.html) and LSTM neural networks. The outcomes of the study are discussed in the article `Improving Chord Prediction in Jazz Music using Melody Information` (in `SMC_2022_Chords.pdf`) which is submitted to the 2022 SMC and can also be found in this repository.

All of the results obtained for this project and discussed in the report can be replicated using the code in this repository. Overall, the code can be divided into three parts: data handling, machine learning, and data analysis tasks. The data handling files have as ultimate output the data files `output_options.pt`, `data_array_without_melody.pt` and `data_array_with_melody.pt`. The machine learning files take these data files as input and produce the desired chord predictions, storing them in 'output_data_array.csv'. The data analysis files are then used to assess these predictions. It is important to note that there is no need to run the data handling files to generate the input data files as they are already provided as `.pt` files. Notice `data_array_with_melody.pt` is uploaded in compressed format, therefore please unzip before using it. Similarly, if only the results are of interest, the ML model does not have to be rerun, as the file including the chord predictions is also provided in this repository.

### Directory structure
```
ML_Jazz:
│   SMC_2022_Chords.pdf
│   README.md
│   output_data_array
│
└───code
    │   csv_output.py
    │   data_array_with_melody.pt.zip
    │   data_array-without_melody.pt
    │   data_processing.ipynb
    │   data_maker.py
    │   data_maker_helpers.py
    │   dataframe_helpers.py
    │   evaluation.ipynb
    │   LSTM_maker.py
    │   ML_core_functions.py
    │   ML_main.ipynb
    │   music_data_functions.py
    │   NN_data_helpers.py
    │   output_options.pt
    │   statistic_tests.ipynb
    │
    └───datasets
            wjazzd.db
```

### Data Handling Files
- `datasets/wjazzd.db` - the database file for the Weiar Jazz Database. The file is larger than GitHubs file size limit and is thus uploaded in compressed format. Please unzip before use.
- `data_processing.ipynb` - a python notebook that is used to produce the input data for the model
- `data_maker.py` - a python file containing the function that is the entry point to generate the data for the model
- `data_maker_helpers.py` - a python file containing the helper functions that process the data provided in the WJazzD database for `data_maker.py`
- `music_data_functions` - a python file containing the helper functions for `data_maker_helpers.py`
- `output_options.pt` - the data file containing the set of possible output chords
- `data_array_without_melody.pt` - the created data file containing the inputs for the model using chords only
- `data_array_with_melody.pt` - the created data file containing the inputs for the model with melody information. The file is larger than GitHubs file size limit and is thus uploaded in compressed format. Please unzip before use.
- `csv_output.py`- a python file with functions used for filling in melody and tune information into the output csv file.

### Machine Learning Files 
- `NN_data_helpers.py` - a python file with helper functions that convert the processed data into the format suitable to be handled by the ML model
- `LSTM_maker.py` - a python file in which the neural network is defined
- `ML_core_functions.py`- a python file containing wrapper functions for the tasks involved in training and running the ML model
- `ML_main.ipynb` - a python notebook that is used to run the ML model

### Data Analysis Files
- `evaluation.ipynb` - a python notebook used for evaluating the performance of the two models using the data stored in òutput_data_array.csv`
- `statistic_tests.ipynb` - code used for assessing the sstatistical significance of the results

### Dependencies
Please note that, in order to run the code for this project, it is necessary to have the following Python libraries installed. Click the hyperlinks to visit the respective websites and find out more about the different dependencies.
- [Numpy](https://numpy.org/)
- [SQLAlchemy](https://www.sqlalchemy.org/)
- [Pandas](https://pandas.pydata.org/)
- [mingus](https://bspaans.github.io/python-mingus/)
- [Pytorch](https://pytorch.org/)
- [Pytorch Lightning](https://www.pytorchlightning.ai/)
- [regex](https://pypi.org/project/regex/)
- [bambi](https://bambinos.github.io/bambi/main/index.html)
- [arviz](https://arviz-devs.github.io/arviz/)

# Running the Data Handling Files
To produce the data used for training the model, please run the notebook `Data_processing.ipynb`. The code is currently configured to produce the data for model with melody information. Therefore, for chord-only data, please set the `melody_info` to `False` in cell 3 of the notebook file. Please note that the computation time can be long.

# Running the ML Model
This can be done using the file `ML_main.ipynb`. This file can be run locally on a regular computer, but it is highly recommended to instead set it up on [Google Colab](https://colab.research.google.com/) as that allows the free use of GPUs. This will require the user to either upload the files directly to Google Colab or to their Google Drive and to then mount their Google Drive in Colab. Also, dependencies such as Pytorch LIghtning may need to be installed. However, a full detailed explanation of how to use Google Colab is beyond the scope of this project.

When running the notebook it is important to ensure that all files (data and helper python files) are in the same relative locations as they are in this repository. Otherwise, if a different file tree is used, please make sure to adapt the file paths accordingly.

It is then possible to run the model by running the different cells consecutively in their original order in the notebook. The output is the file `output_data_array.csv`, which is also already provided in the repository.

# Running the Data Analysis Files
This can be done by running the files `evaluation.ipynb` and `statistics_tests.ipynb`. The former can be used to extract information on how the baseline and melody models perform on different tunes and relative to one another. The latter is used to determine the statistical significance of the results. To run the notebooks, make sure that the file `output_data_array.csv` is present one directory level above the notebooks, or adjust the filepaths in notebooks accordingly.

# Citation for external libraries
- Harris, C.R., Millman, K.J., van der Walt, S.J. et al. Array programming with NumPy. Nature 585, 357–362 (2020). DOI: 10.1038/s41586-020-2649-2. (Publisher link).
- M. Bayer, “Sqlalchemy,” in The Architecture of Open Source Applications Volume II: Structure, Scale, and a Few More Fearless Hacks, A. Brown and G. Wilson, Eds. aosabook.org, 2012.
- Wes  McKinney,  "Data  Structures  for  Statistical  Computing  in Python," in *Proceedings of the 9th Python in Science Conference*, Stefan  van  der  Walt  and  Jarrod  Millman,  Eds.,  2010,  pp.  56  –61.
- A. Paszke, S. Gross, F. Massa, A. Lerer, J. Bradbury, G. Chanan,T.  Killeen,  Z.  Lin,  N.  Gimelshein,  L.  Antiga,  A.  Desmaison,A.    Kopf,    E.    Yang,    Z.    DeVito,    M.    Raison,    A.    Tejani, S.  Chilamkurthy,  B.  Steiner,  L.  Fang,  J.  Bai,  and  S.  Chintala,“Pytorch:  An  imperative  style,  high-performance  deep  learning library,"  in *Advances in Neural Information Processing Systems 32*,  H.  Wallach,  H.  Larochelle,  A.  Beygelzimer,  F.  d'Alche-Buc, E. Fox, and R. Garnett, Eds.    Curran Associates, Inc., 2019, pp. 8024–8035.   [Online].   Available:   http://papers.neurips.cc/paper/9015-pytorch-an-imperative-style-high-performance-deep-learning-library.pd
- T. Killeen, et.al. “Pytorch: An imperative style, high-performance deep learning library,” in Advances in Neural Information Processing Systems 32, H. Wallach, H. Larochelle, A. Beygelzimer, F. d'Alch ́e-Buc, E. Fox, and R. Garnett, Eds. Curran Associates, Inc., 2019, pp. 8024–8035.
- Falcon, W., & The PyTorch Lightning team. (2019). PyTorch Lightning (Version 1.4) [Computer software]. https://doi.org/10.5281/zenodo.3828935
- Spaans, B. (2020). Mingus (Version 0.6.1) [Computer software]. http://bspaans.github.io/python-mingus/
- Matthew Barnett (2022). Alternative regular expression module, to replace re. PyPi. https://github.com/mrabarnett/mrab-regex
- Tomás Capretto, Camen Piho, Ravin Kumar, Jacob Westfall, Tal Yarkoni, & Osvaldo A. Martin. (2020). Bambi: A simple interface for fitting Bayesian linear models in Python.
- Ravin Kumar, Colin Carroll, Ari Hartikainen, & Osvaldo Martin (2019). ArviZ a unified library for exploratory analysis of Bayesian models in Python. Journal of Open Source Software, 4(33), 1143.
