# import libraries

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from numpy import mean

from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression


from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import VotingRegressor

train_data_url = "https://raw.githubusercontent.com/ridwant/DataMinig/main/bug.txt"


def data_description(train_data_url):
    df = pd.read_csv(train_data_url, sep=" ")
    # Chech Number of instances and features
    print(df.shape)
    # Chech for NAN or Null
    print(df.isnull().any())
    # Chech for duplicate values
    print(df.duplicated().sum())


def data_cleaning(train_data_url):
    # load all data using pandas data frame
    all_df = pd.read_csv(train_data_url, sep=" ")
    all_df = all_df.drop_duplicates()

    return all_df


def get_train_test_split(all_df):

    """
    Returns the splitted (3:1) train:test dataset with categorical features converted to numeric
    """
    # split data into testing and training data
    train_df, test_df = train_test_split(all_df, test_size=0.25, random_state=25)

    return train_df, test_df


def data_preprocess(df):
    """
    Returns the dataframe with continuous values scaled to 0 to 1
    """
    Y = df["d"].values
    X = df.drop(["d"], axis=1).values
    scaler = MinMaxScaler(feature_range=(0, 1))
    x_train_scaled = scaler.fit_transform(X)
    X = pd.DataFrame(x_train_scaled)

    return X, Y


def k_fold_validation(X, Y, estimator):
    accuracy = cross_val_score(estimator, X, Y, scoring="r2", cv=5)
    return mean(accuracy)


def knn(train_df, k=5, weights="distance", p=2):
    model = KNeighborsRegressor(n_neighbors=k, weights=weights, metric="euclidean", p=p)
    X_train, Y_train = data_preprocess(train_df)
    evaluation_metrics = k_fold_validation(X_train, Y_train, model)
    return evaluation_metrics


def experiment_on_training_data_k_value():
    all_df = data_cleaning(train_data_url)
    train_df, test_df = get_train_test_split(all_df)
    mse = list()
    num_neighbours = list()

    for k in range(1, 50, 1):
        acc = knn(train_df, k=k)
        mse.append(acc)
        num_neighbours.append(k)

    # plot lines
    plt.plot(num_neighbours, mse, label="R2")
    plt.xlabel("Score")
    plt.xlabel("Number of Neighbours")
    plt.title("Experiment With KNN in Training Set to find the k-value")
    plt.legend()
    plt.show()


def experiment_on_training_data_dist_metric(train_data_url):
    all_df = data_cleaning(train_data_url)
    train_df, test_df = get_train_test_split(all_df)

    metrics = ["Manhattan", "Euclidean", "Minkowski"]
    score = list()

    acc_1 = knn(train_df, k=6, weights="uniform", p=1)
    score.append(acc_1)
    acc_2 = knn(train_df, k=6, weights="uniform", p=2)
    score.append(acc_2)
    acc_3 = knn(train_df, k=6, weights="uniform", p=3)
    score.append(acc_3)

    print(score)
    plt.bar(metrics, score)
    plt.xlabel("Metrics")
    plt.ylabel("Score")
    plt.title("Experiment With KNN in Training Set to find the distance metric")
    plt.legend()
    plt.show()


def prediction_on_test_data(train_data_url, model):
    all_df = data_cleaning(train_data_url)
    train_df, test_df = get_train_test_split(all_df)

    Y_train = train_df["d"].values
    X_train = train_df.drop(["d"], axis=1).values
    y_test = test_df["d"].values
    x_test = test_df.drop(["d"], axis=1).values

    scaler = MinMaxScaler(feature_range=(0, 1))
    x_train_scaled = scaler.fit_transform(X_train)
    x_test_scaled = scaler.fit_transform(x_test)
    X_train = pd.DataFrame(x_train_scaled)
    x_test = pd.DataFrame(x_test_scaled)

    metrics = ["r2-score"]
    score = list()

    model.fit(X_train, Y_train)
    predictions = model.predict(x_test)
    mse = r2_score(y_test, predictions)
    score.append(mse)

    print(score)
    plt.bar(metrics, score, color="teal", width=0.5)
    plt.xlabel("Metrics")
    plt.ylabel("Score")
    plt.title("Prediction on Test Data")
    plt.legend()
    plt.show()


# Final Prediction using the test data reserved for validation


def final_prediction(train_data_url):
    all_df = data_cleaning(train_data_url)
    train_df, test_df = get_train_test_split(all_df)

    Y_train = train_df["d"].values
    X_train = train_df.drop(["d"], axis=1).values
    y_test = test_df["d"].values
    x_test = test_df.drop(["d"], axis=1).values

    scaler = MinMaxScaler(feature_range=(0, 1))
    x_train_scaled = scaler.fit_transform(X_train)
    x_test_scaled = scaler.fit_transform(x_test)
    X_train = pd.DataFrame(x_train_scaled)
    x_test = pd.DataFrame(x_test_scaled)

    knn_reg = KNeighborsRegressor(n_neighbors=6, weights="distance", metric="euclidean")
    dt_reg = DecisionTreeRegressor(random_state=123)
    rf_reg = RandomForestRegressor(random_state=123)
    voting_reg = VotingRegressor(estimators=[("gb", knn_reg), ("lr", rf_reg)])

    models = [knn_reg, dt_reg, rf_reg, voting_reg]
    model_names = ["knn", "decision_tree", "random_forest", "voting"]
    mse_score = list()

    for model in models:
        model.fit(X_train, Y_train)
        predictions = model.predict(x_test)
        mse = r2_score(y_test, predictions)
        mse_score.append(mse)

    print(mse_score)
    # set width of bar
    barWidth = 0.5
    fig = plt.subplots(figsize=(6, 4))

    # Set position of bar on X axis
    br1 = np.arange(len(mse_score))
    # Make the plot
    plt.bar(br1, mse_score, width=barWidth, edgecolor="grey", label="R2")
    # Adding Xticks
    plt.xlabel("Performance evaluation metrics comparison", fontsize=15)
    plt.xticks([r for r in range(len(model_names))], model_names)

    plt.legend()
    plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("regressorName", type=str, default=None)
    args = parser.parse_args()
    regressorName = args.regressorName
    model = DecisionTreeRegressor(random_state=123)
    if regressorName.lower().strip() == "knn":
        model = KNeighborsRegressor(
            n_neighbors=6, weights="distance", metric="euclidean"
        )
        prediction_on_test_data(train_data_url, model)
    if regressorName.lower().strip() == "dt":
        model = DecisionTreeRegressor(random_state=123)
        prediction_on_test_data(train_data_url, model)
    if regressorName.lower().strip() == "rf":
        model = RandomForestRegressor(random_state=123)
        prediction_on_test_data(train_data_url, model)
    if regressorName.lower().strip() == "other":
        knn_reg = KNeighborsRegressor(
            n_neighbors=6, weights="distance", metric="euclidean"
        )
        rf_reg = RandomForestRegressor(random_state=123)
        model = VotingRegressor(estimators=[("gb", knn_reg), ("lr", rf_reg)])
        prediction_on_test_data(train_data_url, model)
    if regressorName.lower().strip() == "all":
        final_prediction(train_data_url)


if __name__ == "__main__":
    main()
