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
# Load the California housing data
housing = fetch_california_housing(as_frame=True)

# X is what we know about each area, y is the price we want to predict
X = housing.data
y = housing.target  # prices are in $100k, so 2.5 means $250k

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# ----- Baseline model -----

# Train a regular linear regression on the raw data
base_model = LinearRegression()
base_model.fit(X_train, y_train)

# Predict prices for the test houses
test_pred1 = base_model.predict(X_test)

# MSE: average squared error, lower is better
baseline_mse = mean_squared_error(y_test, test_pred1)
print(f"\nBaseline Test MSE: {baseline_mse:.4f}")

baseline_r2 = r2_score(y_test, test_pred1)
print(f"Baseline Test R²: {baseline_r2:.4f}")

cap = y_train.max()
print(f"\nCap value: {cap}")
print(f"Training homes stuck at the cap: {(y_train == cap).sum()} ({(y_train == cap).mean()*100:.1f}%)")
print("Most common prices:")
print(y_train.value_counts().head())


# ----- Clean the data -----

# Drop every house sitting at the cap, in both train and test
clean_train_rows = y_train < cap
X_train_clean = X_train[clean_train_rows]
y_train_clean = y_train[clean_train_rows]

clean_test_rows = y_test < cap
X_test_clean = X_test[clean_test_rows]
y_test_clean = y_test[clean_test_rows]


# ----- Cleaned model -----

# Train a second model on the cleaned data
cleaned_model = LinearRegression()
cleaned_model.fit(X_train_clean, y_train_clean)

# Test it on the cleaned test houses
test_pred2 = cleaned_model.predict(X_test_clean)
cleaned_mse = mean_squared_error(y_test_clean, test_pred2)
cleaned_r2 = r2_score(y_test_clean, test_pred2)


# test the baseline on the cleaned test houses
base_on_clean = base_model.predict(X_test_clean)
base_clean_mse = mean_squared_error(y_test_clean, base_on_clean)
base_clean_r2 = r2_score(y_test_clean, base_on_clean)

print(f"  Baseline (full test):  MSE {baseline_mse:.4f} | R² {baseline_r2:.4f}")
print(f"  Cleaned (clean test):  MSE {cleaned_mse:.4f} | R² {cleaned_r2:.4f}")

print("Both models on the same clean test set (the fair one):")
print(f"  Baseline:  MSE {base_clean_mse:.4f} | R² {base_clean_r2:.4f}")
print(f"  Cleaned:   MSE {cleaned_mse:.4f} | R² {cleaned_r2:.4f}")
print(f"  MSE improved by {((base_clean_mse - cleaned_mse) / base_clean_mse) * 100:.2f}%")

# Looking at homes under $200k and check the average error
# A positive number means the model guesses too high
low = y_test_clean < 2.0
base_low_error = (base_on_clean[low] - y_test_clean[low]).mean()
clean_low_error = (test_pred2[low] - y_test_clean[low]).mean()

print("\n--- Error on Homes Under $200k ---")
print(f"Baseline average error: {base_low_error:.4f}")
print(f"Cleaned average error:  {clean_low_error:.4f}")


# ----- Plot -----

# Each dot is a house: what we guessed vs what it really cost
plt.figure(figsize=(8, 6))
plt.scatter(test_pred1, y_test, alpha=0.3, color='blue')

# Red line at the $500k cap
plt.axhline(5.0, color='red', linestyle='--', label='$500k Cap')
plt.plot([0, 5], [0, 5], color='black', linestyle=':', label='Ideal (Actual = Predicted)')

# Point out the flat row of capped houses
plt.annotate('Horizontal Artifact:\nCapped at $500k',
             xy=(3.5, 5.0), xytext=(1.0, 4.2),
             arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=6),
             bbox=dict(boxstyle="round,pad=0.3", fc="yellow", alpha=0.5))

plt.title("Baseline Model: Predicted vs. Actual")
plt.xlabel("Predicted Price ($100k)")
plt.ylabel("Actual Price ($100k)")
plt.legend()

# Save the chart for the report before showing it
plt.savefig("baseline_pred_vs_actual.png", dpi=150, bbox_inches="tight")
plt.show()

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
