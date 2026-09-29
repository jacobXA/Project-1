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

    # scale for a
    scaler_a = StandardScaler()
    scaled_a = scaler_a.fit_transform(features_a)

    # scale for b
    scaler_b = StandardScaler()
    scaled_b = scaler_b.fit_transform(features_b)

    # function for computing k metrics for a and b
    def compute_k_metrics(scaled_data):
        # lists to hold inertias and silhouette scores
        inertias = []
        silhouette_scores = []

        # k values from 2-11
        k_values = range(2, 11)

        # go through k values and get KMeans to group data into clusters
        for k in k_values:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(scaled_data)

            # add inertia to list
            #   --> inertia is how tightly data points are grouped in clusters
            inertias.append(kmeans.inertia_)

            # add silhouette score to list
            #   --> silhouette score looks at how close one cluster is to neighboring clusters
            silhouette_scores.append(silhouette_score(scaled_data, labels))

        return inertias, silhouette_scores

    inertias_a, silhouette_a = compute_k_metrics(scaled_a)
    inertias_b, silhouette_b = compute_k_metrics(scaled_b)



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