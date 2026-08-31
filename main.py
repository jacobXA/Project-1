import pandas as pd
import numpy as np
import sklearn
data1 = sklearn.datasets.load_breast_cancer() # Data set for Part 1
data2 = sklearn.datasets.fetch_california_housing() # Data set for Part 2
df = pd.read_csv("Mall_Customers.csv") # Data set for Part 3
df2 = pd.read_csv("Air_Quality.csv") # Data set for Part 4