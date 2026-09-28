import pandas as pd
import numpy as np
import sklearn
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

#=================Part 1=================
def part1():
    data1 = sklearn.datasets.load_breast_cancer() # Data set for Part 1

#=================Part 2=================
def part2():
    data2 = sklearn.datasets.fetch_california_housing() # Data set for Part 2

#=================Part 3=================
def part3():
    df = pd.read_csv("Mall_Customers.csv") # Data set for Part 3

    # CustomerID is just a row label, not a real feature, so we drop it
    df = df.drop(columns=["CustomerID"])

    # Two competing feature combinations, per the assignment:
    # Option A: the "classic" 2D view every mall-customer tutorial uses
    features_a = df[["Annual Income (k$)", "Spending Score (1-100)"]]

    # Option B: adds Age, which pulls the clustering in a different direction
    features_b = df[["Age", "Annual Income (k$)", "Spending Score (1-100)"]]

    # scale to standardize features before clustering
    scaler_a = StandardScaler()
    scaled_a = scaler_a.fit_transform(features_a)

    scaler_b = StandardScaler()
    scaled_b = scaler_b.fit_transform(features_b)


#=================Part 4=================
def part4():
    df2 = pd.read_csv("Air_Quality.csv") # Data set for Part 4

#=================Main=================
def main():
    part1()
    part2()
    part3()
    part4()

if __name__ == "__main__":
    main()