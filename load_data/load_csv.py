import csv
import numpy as np
import pandas as pd

path = r"C:\Users\marce\Documents\GitHub\DataScience\data\\"
file = "train.csv"
startRow = 2;

data = pd.read_csv(path+file, delimiter=',');
