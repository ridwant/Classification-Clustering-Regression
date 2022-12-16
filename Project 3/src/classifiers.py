# import libraries

import argparse
import numpy as np
import pandas as pd
from numpy import mean

from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier

from sklearn.model_selection import (
    train_test_split,
    cross_val_score,
    RepeatedStratifiedKFold,
)
from sklearn.metrics import accuracy_score, f1_score

from imblearn.under_sampling import RandomUnderSampler
import matplotlib.pyplot as plt

train_data_url = (
    "https://raw.githubusercontent.com/ridwant/DataMinig/main/bank-additional-full.csv"
)


def data_description(train_data_url):
    # load all data using pandas data frame
    df = pd.read_csv(train_data_url, sep=";")
    # Chech Number of instances and features
    print(df.shape)
    # Chech for NAN or Null
    print(df.isnull().any())
    # Chech for duplicate values
    print(df.duplicated().sum())


def data_cleaning(train_data_url):
    # load all data using pandas data frame
    all_df = pd.read_csv(train_data_url, sep=";")
    all_df = all_df.drop_duplicates()
    all_df["y"] = all_df["y"].map({"yes": 1, "no": 0})

    # convert categorical label to numerical using pandas get dummies
    cat_item = [
        "job",
        "marital",
        "education",
        "default",
        "housing",
        "loan",
        "contact",
        "month",
        "day_of_week",
        "poutcome",
    ]
    converted_df = pd.get_dummies(all_df, columns=cat_item)
    return converted_df


def get_train_test_split(all_df):

    """
    Returns the splitted (3:1) train:test dataset with categorical features converted to numeric
    """
    # split data into testing and training data
    train_df, test_df = train_test_split(all_df, test_size=0.25, random_state=25)

    return train_df, test_df


def k_fold_cross_validation(X, Y, estimator):
    # cross validation with repeated stratified, with 5 split and 1 repeats, returns the mean score of the scoring metric
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=1, random_state=1)
    f1 = cross_val_score(estimator, X, Y, scoring="f1", cv=cv)
    acc = cross_val_score(estimator, X, Y, scoring="accuracy", cv=cv)
    return mean(f1), mean(acc)


# Experiment on Training dataset using stratified k-fold cross validation with K-nearnest neighbour classifier to find the best k value and distance metric
def knn(train_df, k=3, weights="distance", metric="euclidean", p=2):
    Y_train = train_df["y"].values
    X_train = train_df.drop(["y"], axis=1).values
    model = KNeighborsClassifier(n_neighbors=k, weights=weights, metric=metric, p=p)
    f1, acc = k_fold_cross_validation(X_train, Y_train, model)
    return f1, acc


# Experiments
def experiment_on_training_data_on_acc(train_data_url):
    converted_df = data_cleaning(train_data_url)
    train_df, test_df = get_train_test_split(converted_df)
    x = list()
    y = list()
    num_neighbours = list()
    for k in range(3, 50, 2):
        f1, acc = knn(train_df, k)
        x.append(acc)
        y.append(f1)
        num_neighbours.append(k)

    # plot lines
    plt.plot(num_neighbours, x, label="Accuracy")
    plt.plot(num_neighbours, y, label="F1-Score")
    plt.xlabel("Score")
    plt.xlabel("Number of Neighbours")
    plt.title("Experiment With KNN in Training Set to find the k-value")
    plt.legend()
    plt.show()


def experiment_on_training_data_dist_metric(train_data_url):
    converted_df = data_cleaning(train_data_url)
    train_df, test_df = get_train_test_split(converted_df)

    metrics = ["Manhattan", "Euclidean", "Minkowski"]
    score = list()

    _, acc_1 = knn(train_df, k=33, weights="uniform", p=1)
    score.append(acc_1)
    _, acc_2 = knn(train_df, k=33, weights="uniform", p=2)
    score.append(acc_2)
    _, acc_3 = knn(train_df, k=33, weights="uniform", p=3)
    score.append(acc_3)

    print(score)
    plt.bar(metrics, score)
    plt.xlabel("Metrics")
    plt.ylabel("Score")
    plt.title("Experiment With KNN in Training Set to find the distance metric")
    plt.legend()
    plt.show()


def experiment_on_training_data_voting(train_data_url):
    converted_df = data_cleaning(train_data_url)
    train_df, test_df = get_train_test_split(converted_df)

    metrics = ["Majority Voting", "Weighted Distance"]
    score = list()

    _, acc_1 = knn(train_df, k=33, weights="uniform", metric="euclidean")
    score.append(acc_1)
    _, acc_2 = knn(train_df, k=33, weights="distance", metric="euclidean")
    score.append(acc_2)

    print(score)
    plt.bar(metrics, score)
    plt.xlabel("Metrics")
    plt.ylabel("Score")
    plt.title(
        "Experiment With KNN in Training Set to find the voting and distance metric"
    )
    plt.legend()
    plt.show()


# experiment_on_training_data_on_acc(train_data_url)
# experiment_on_training_data_dist_metric(train_data_url)
# experiment_on_training_data_voting


def prediction_using_knn_on_test_data(train_data_url):
    converted_df = data_cleaning(train_data_url)
    train_df, test_df = get_train_test_split(converted_df)

    Y_train = train_df["y"].values
    X_train = train_df.drop(["y"], axis=1).values
    y_test = test_df["y"].values
    x_test = test_df.drop(["y"], axis=1).values

    metrics = ["accuracy", "f1-score"]
    score = list()

    model = KNeighborsClassifier(n_neighbors=33, weights="distance", metric="euclidean")
    model.fit(X_train, Y_train)
    predictions = model.predict(x_test)

    score.append(accuracy_score(y_test, predictions))
    score.append(f1_score(y_test, predictions))

    print(score)
    plt.bar(metrics, score, color="teal")
    plt.xlabel("Metrics")
    plt.ylabel("Score")
    plt.title("Prediction on Test Data")
    plt.legend()
    plt.show()


# prediction_using_knn_on_test_data(train_data_url)


def prediction_using_dt(train_data_url):
    converted_df = data_cleaning(train_data_url)
    train_df, test_df = get_train_test_split(converted_df)

    Y_train = train_df["y"].values
    X_train = train_df.drop(["y"], axis=1).values
    y_test = test_df["y"].values
    x_test = test_df.drop(["y"], axis=1).values

    metrics = ["accuracy", "f1-score"]
    score = list()

    model = DecisionTreeClassifier(random_state=123)
    model.fit(X_train, Y_train)
    predictions = model.predict(x_test)

    score.append(accuracy_score(y_test, predictions))
    score.append(f1_score(y_test, predictions))

    print(score)
    plt.bar(metrics, score, color="teal")
    plt.xlabel("Metrics")
    plt.ylabel("Score")
    plt.title("Prediction on Test Data")
    plt.legend()
    plt.show()


def prediction_using_rf(train_data_url):
    converted_df = data_cleaning(train_data_url)
    train_df, test_df = get_train_test_split(converted_df)

    Y_train = train_df["y"].values
    X_train = train_df.drop(["y"], axis=1).values
    y_test = test_df["y"].values
    x_test = test_df.drop(["y"], axis=1).values

    metrics = ["accuracy", "f1-score"]
    score = list()

    model = RandomForestClassifier(max_depth=11, random_state=123)
    model.fit(X_train, Y_train)
    predictions = model.predict(x_test)

    score.append(accuracy_score(y_test, predictions))
    score.append(f1_score(y_test, predictions))

    print(score)
    plt.bar(metrics, score, color="teal")
    plt.xlabel("Metrics")
    plt.ylabel("Score")
    plt.title("Prediction on Test Data")
    plt.legend()
    plt.show()


def prediction_using_adaboost_on_test_data(train_data_url):
    converted_df = data_cleaning(train_data_url)
    train_df, test_df = get_train_test_split(converted_df)

    Y_train = train_df["y"].values
    X_train = train_df.drop(["y"], axis=1).values
    y_test = test_df["y"].values
    x_test = test_df.drop(["y"], axis=1).values

    under = RandomUnderSampler(random_state=1)
    X_resampled, y_resampled = under.fit_resample(X_train, Y_train)

    metrics = ["accuracy", "f1-score"]
    score = list()

    clf = DecisionTreeClassifier(random_state=123, max_depth=1)
    model = AdaBoostClassifier(base_estimator=clf, n_estimators=650, random_state=1)
    model.fit(X_resampled, y_resampled)
    predictions = model.predict(x_test)

    score.append(accuracy_score(y_test, predictions))
    score.append(f1_score(y_test, predictions))

    print(score)
    plt.bar(metrics, score, color="orange")
    plt.xlabel("Metrics")
    plt.ylabel("Score")
    plt.title("Prediction on Test Data")
    plt.legend()
    plt.show()


def prediction_using_all_three_classifiers(train_data_url):
    # Final Prediction using the test data reserved for validation
    converted_df = data_cleaning(train_data_url)
    train_df, test_df = get_train_test_split(converted_df)

    Y_train = train_df["y"].values
    X_train = train_df.drop(["y"], axis=1).values
    y_test = test_df["y"].values
    x_test = test_df.drop(["y"], axis=1).values

    knn_clf = KNeighborsClassifier(
        n_neighbors=33, weights="distance", metric="euclidean"
    )
    dt_clf = DecisionTreeClassifier(random_state=123)
    rf_clf = RandomForestClassifier(max_depth=11, random_state=123)

    models = [knn_clf, dt_clf, rf_clf]
    model_names = ["knn", "decision_tree", "random_forest"]
    f1_scr = list()
    acc_score = list()

    for model in models:
        model.fit(X_train, Y_train)
        predictions = model.predict(x_test)
        f1_scr.append(f1_score(y_test, predictions))
        acc_score.append(accuracy_score(y_test, predictions))

    print(f1_scr)
    print(acc_score)
    # set width of bar
    barWidth = 0.35
    fig = plt.subplots(figsize=(12, 8))

    # Set position of bar on X axis
    br1 = np.arange(len(f1_scr))
    br2 = [x + barWidth for x in br1]

    # Make the plot
    plt.bar(br1, f1_scr, width=barWidth, edgecolor="grey", label="f1-score")
    plt.bar(br2, acc_score, width=barWidth, edgecolor="grey", label="accuracy")
    # Adding Xticks
    plt.xlabel("Performance evaluation metrics comparison", fontsize=15)
    plt.xticks([r + barWidth for r in range(len(model_names))], model_names)

    plt.legend()
    plt.show()


# prediction_using_all_three_classifiers(train_data_url)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("classifierName", type=str, default=None)
    args = parser.parse_args()

    classifierName = args.classifierName

    if classifierName.lower().strip() == "knn":
        prediction_using_knn_on_test_data(train_data_url)
    if classifierName.lower().strip() == "dt":
        prediction_using_dt(train_data_url)
    if classifierName.lower().strip() == "rf":
        prediction_using_rf(train_data_url)
    if classifierName.lower().strip() == "other":
        prediction_using_adaboost_on_test_data(train_data_url)
    if classifierName.lower().strip() == "all":
        prediction_using_all_three_classifiers(train_data_url)


if __name__ == "__main__":
    main()
