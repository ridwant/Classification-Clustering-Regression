# import libraries

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score

from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.preprocessing import LabelEncoder

import time

train_data_url = "https://raw.githubusercontent.com/ridwant/DataMinig/main/train.csv"
test_data_url = (
    "https://raw.githubusercontent.com/ridwant/DataMinig/main/test%20(12).csv"
)


def data_cleaning(train_data_url):
    # load all data using pandas data frame
    df = pd.read_csv(train_data_url, header=None, sep=",")
    df = df.drop_duplicates()
    return df


def see_correlation():
    train_df = data_cleaning(train_data_url)
    train_df.corr()
    fig, ax = plt.subplots(figsize=(15, 15))
    sns.heatmap(train_df.corr(), cmap="YlGnBu", annot=True, linewidths=0.5)
    plt.show()


def random_forest_classifier(X_train, Y_train, x_test, y_true):
    clf = RandomForestClassifier(random_state=123).fit(X_train, Y_train)
    y_pred = clf.predict(x_test)
    return accuracy_score(y_true, y_pred)


def mlp_classifier(X_train, Y_train, x_test, y_true):
    clf = MLPClassifier(
        random_state=59,
        activation="tanh",
        max_iter=500,
        hidden_layer_sizes=(
            128,
            256,
        ),
    ).fit(X_train, Y_train)
    y_pred = clf.predict(x_test)
    return accuracy_score(y_true, y_pred)


def baseline_model():
    model = Sequential()
    model.add(Dense(128, input_dim=16, activation="tanh"))
    model.add(Dense(256, activation="tanh"))
    model.add(Dense(26, activation="softmax"))
    model.compile(
        loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"]
    )
    return model


def keras_neural_network(X_train, Y_train, x_test, y_true, epochs=100):
    estimator = KerasClassifier(
        build_fn=baseline_model, epochs=epochs, batch_size=32, verbose=1
    )
    estimator.fit(X_train, Y_train)
    prediction = estimator.predict(x_test)
    return accuracy_score(y_true, prediction)


def comparison_with_baseline_classifier(X_train, Y_train, x_test, y_true):

    model_names = [
        "RandomForest Classifier",
        "MLP Classifier",
        "Keras Sequential Classifier",
    ]
    time_scr = list()
    acc_score = list()

    for idx, model in enumerate(model_names):
        start_time = time.time()
        if idx == 0:
            acc = random_forest_classifier(X_train, Y_train, x_test, y_true)
            acc_score.append(acc * 100)
            time_scr.append(time.time() - start_time)
        elif idx == 1:
            acc = mlp_classifier(X_train, Y_train, x_test, y_true)
            acc_score.append(acc * 100)
            time_scr.append(time.time() - start_time)
        elif idx == 2:
            acc = keras_neural_network(X_train, Y_train, x_test, y_true)
            acc_score.append(acc * 100)
            time_scr.append(time.time() - start_time)

    print(time_scr)
    print(acc_score)
    # set width of bar
    barWidth = 0.35
    fig = plt.subplots(figsize=(12, 8))

    # Set position of bar on X axis
    br1 = np.arange(len(time_scr))
    br2 = [x + barWidth for x in br1]

    # Make the plot
    plt.bar(br2, acc_score, width=barWidth, edgecolor="grey", label="accuracy")
    plt.bar(br1, time_scr, width=barWidth, edgecolor="grey", label="time")
    # Adding Xticks
    plt.xlabel("Comparison with baseline (RandomForestClassifier)", fontsize=15)
    plt.xticks([r + barWidth for r in range(len(model_names))], model_names)

    plt.legend()
    plt.show()


def plot_on_dataset(X, y):
    params = [
        {
            "solver": "sgd",
            "learning_rate": "constant",
            "learning_rate_init": 0.001,
        },
        {
            "solver": "sgd",
            "learning_rate": "constant",
            "momentum": 0.9,
            "learning_rate_init": 0.2,
        },
        {
            "solver": "sgd",
            "learning_rate": "invscaling",
            "learning_rate_init": 0.2,
        },
        {
            "solver": "adam",
            "learning_rate": "constant",
            "learning_rate_init": 0.001,
        },
    ]

    labels = [
        "constant learning-rate sgd",
        "constant with momentum sgd",
        "inv-scaling learning-rate sgd",
        "constant learning-rate adam",
    ]

    plot_args = [
        {"c": "red", "linestyle": "dashed"},
        {"c": "blue", "linestyle": "dotted"},
        {"c": "orange", "linestyle": "solid"},
        {"c": "green"},
    ]
    plt.title("Digit Data")
    mlps = []
    for label, param in zip(labels, params):
        print("training: %s" % label)
        mlp = MLPClassifier(random_state=0, max_iter=500, **param)
        mlp.fit(X, y)
        mlps.append(mlp)
        print("Training set score: %f" % mlp.score(X, y))
        print("Training set loss: %f" % mlp.loss_)
    for mlp, label, args in zip(mlps, labels, plot_args):
        plt.plot(mlp.loss_curve_, label=label, **args)
    plt.legend(labels)
    plt.show()


def experiment_parameter_tuning():
    train_df = data_cleaning(train_data_url)
    Y_train = train_df[0].values
    X_train = train_df.drop(columns=[0], axis=1).values
    plot_on_dataset(X_train, Y_train)


def experiment_on_number_of_epocs():
    train_df = data_cleaning(train_data_url)
    test_df = data_cleaning(test_data_url)
    le = LabelEncoder()

    Y_train = train_df[0].values
    Y_train = le.fit_transform(Y_train)
    X_train = train_df.drop(columns=[0], axis=1).values

    y_true = test_df[0].values
    y_true = le.fit_transform(y_true)
    x_test = test_df.drop(columns=[0], axis=1).values

    k_array = list()
    y = list()

    for k in range(50, 201, 50):
        acc = keras_neural_network(X_train, Y_train, x_test, y_true, epochs=k)
        y.append(acc)
        k_array.append(k)

    plt.plot(k_array, y, label="Accuracy")
    plt.ylabel("Accuracy")
    plt.xlabel("Epocs")
    plt.title(
        "Experiment to determine number of epocs needed to converge in Keras Model"
    )
    plt.legend()
    plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("classifierName", type=str, default=None)
    args = parser.parse_args()

    classifierName = args.classifierName

    train_df = data_cleaning(train_data_url)
    test_df = data_cleaning(test_data_url)
    le = LabelEncoder()

    Y_train = train_df[0].values
    Y_train = le.fit_transform(Y_train)
    X_train = train_df.drop(columns=[0], axis=1).values

    y_true = test_df[0].values
    y_true = le.fit_transform(y_true)
    x_test = test_df.drop(columns=[0], axis=1).values
    if classifierName.lower().strip() == "mlp":
        acc = mlp_classifier(X_train, Y_train, x_test, y_true)
        print(f"Accuracy {acc} using MLP Classifier on Test Data")
    if classifierName.lower().strip() == "ksm":
        acc = keras_neural_network(X_train, Y_train, x_test, y_true)
        print(f"Accuracy {acc} using Keras Sequential Model on Test Data")
    if classifierName.lower().strip() == "rf":
        acc = random_forest_classifier(X_train, Y_train, x_test, y_true)
        print(f"Accuracy {acc} using RandomForest Classifier on Test Data")
    if classifierName.lower().strip() == "all":
        comparison_with_baseline_classifier(X_train, Y_train, x_test, y_true)


if __name__ == "__main__":
    main()
