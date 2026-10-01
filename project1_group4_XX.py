import pandas as pd
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, recall_score, accuracy_score, confusion_matrix, precision_recall_curve, make_scorer

#=================Part 1=================
def part1():

# Load the breast cancer dataset
    data = load_breast_cancer(as_frame=True)
    X = data.data
    y = data.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=43)

    # Grid search for k that maximizes recall on the malignant class (0), with feature scaling
    pipeline = Pipeline([('scaler', StandardScaler()), ('knn', KNeighborsClassifier())])
    recall_malignant = make_scorer(recall_score, pos_label=0)
    grid = GridSearchCV(pipeline, {'knn__n_neighbors': range(1, 8, 2)}, scoring=recall_malignant, cv=5)
    grid.fit(X_train, y_train)
    model = grid.best_estimator_
    print(f'Best k: {grid.best_params_["knn__n_neighbors"]}, CV recall(malignant): {grid.best_score_:.3f}')

    # Evaluate the KNN classifier
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Accuracy: {accuracy:.3f}')
    conf_matrix = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:")
    print(conf_matrix)
    probs = model.predict_proba(X_test)[:, 1]  # P(benign)

    # Sweep thresholds to see the FN/FP tradeoff for lowering benign threshold
    print(f"\n{'Threshold':>10} {'FN':>6} {'FP':>6} {'Accuracy':>10}")
    for threshold in np.arange(0.1, 0.6, 0.05):
        y_pred_t = (probs >= threshold).astype(int)
    cm = confusion_matrix(y_test, y_pred_t)
    fn, fp = cm[0, 1], cm[1, 0]
    acc = accuracy_score(y_test, y_pred_t)
    print(f'{threshold:>10.2f} {fn:>6} {fp:>6} {acc:>10.3f}')

    precision, recall_vals, thresholds = precision_recall_curve(y_test, probs)

    # Plot the precision-recall curve
    plt.plot(recall_vals, precision, marker='X')
    plt.xlabel("Recall")
    plt.ylabel("Precision")

    # thresholds has one fewer element than precision/recall, so pad it for a clean table
    print(f"\n{'Threshold':>10} {'Precision':>10} {'Recall':>10}")
    for thresh, prec, rec in zip(np.append(thresholds, np.nan), precision, recall_vals):
        thresh_str = f'{thresh:.3f}' if not np.isnan(thresh) else 'N/A'
    print(f"{thresh_str:>10} {prec:>10.3f} {rec:>10.3f}")

    plt.title("KNN Classifier Precision-Recall Curve")
    plt.show()

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