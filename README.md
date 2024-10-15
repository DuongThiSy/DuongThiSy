!pip install matplotlib==3.4.3
!pip install numpy==1.21.5
!pip install pandas==1.3.5
!pip install scikit_learn==1.0.2
!pip install seaborn==0.11.2
!pip install torch==1.10.2
!pip install torchvision==0.11.3
#importing necessary libraries
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
import sklearn
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from warnings import filterwarnings
filterwarnings('ignore')
#reading data
df = pd.read_csv("../Input/data.csv")
df.tail(8)  # Visualizing some of the rows of our data 
