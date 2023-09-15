# Project : Twitter Stock Prediction

## Dataset
- DataSet is taken from Kaggle - https://www.kaggle.com/datasets/maharshipandya/twitter-stocks-dataset?select=twitter-stocks.csv 
- This dataset contains nine years of Twitter stock price data, spanning from November 2013 to October 2022, in CSV format. It is suitable for time series analysis using Recurrent Neural Networks (RNNs) to predict future Twitter stock prices, making it valuable for financial forecasting and investment decision-making.

## Column Description
There are 7 columns in this dataset.

Note: The currency is in USD ($)

- Date: The date for which the stock data is considered.
- Open: The stock's opening price on that day.
- High: The stock's highest price on that day.
- Low: The stock's lowest price on that day.
- Close: The stock's closing price on that day. The close price is adjusted for splits.
- Adj Close: Adjusted close price adjusted for splits and dividend and/or capital gain distributions.
- Volume: Volume measures the number of shares traded in a stock or contracts traded in futures or options.

## SKills

- Python
- Pandas
- Numpy
- Matplotlib
- scikit-learn
- Data visualization
- Data Preprocessing
- Treating missing values in dataset
- Splitting the Data
- Feature Scaling
- Create the Datasets
- Tensorflow and Keras to built model
- Evaluate the Model
- Visualizing Loss vs Epochs

## Installation
```bash
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

from sklearn.preprocessing import MinMaxScaler

import tensorflow as tf
tf.random.set_seed(42)

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam
import keras.backend as K
```
    

