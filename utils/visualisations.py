
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from utils.constants import NO_TRAIN, NO_TEST, NO_POINTS


def display_plots():
    """API call to Matplotlib to display all plots."""
    plt.show()


def line_plot_all(df: pd.DataFrame, row: int):
    """Creates line plot for all or selected row in dataframe."""
    if row is not None:
        plt.plot(range(0, NO_TRAIN + NO_TEST), df.iloc[row, 0:NO_POINTS], label="{}".format(df.loc[row, 'seed']))
    else:
        for r in range(0, df.shape[0]):
            plt.plot(range(0, NO_TRAIN+NO_TEST), df.iloc[r, 0:NO_POINTS], label="{}".format(df.loc[r, 'seed']))
    plt.axvline(x=NO_TRAIN, color='k', linestyle='--')
    plt.title("PRNG sequences")
    plt.xlabel('Index')
    plt.legend()


# TODO: ...
def binary_array_plot(df: pd.DataFrame, row: int):
    """Per PRNG seed data, plots each training sequential output as a binary visualisation, resulting in static TV-like
    image"""
    raise NotImplemented


def line_plot_with_predictions(row: np.array, data_predicted: np.array):
    """Plot the give seeds actual and predicted values."""
    fig, ax = plt.subplots(1, 1)
    train = np.array([range(0, NO_POINTS), row[0:NO_POINTS]]).T
    test = np.array([range(NO_TRAIN, NO_POINTS), list(data_predicted)]).T
    plt.plot(train[:, 0], train[:, 1], color='g', label='Actual')
    plt.plot(test[:, 0], test[:, 1], color='r', label='Predicted')
    plt.axvline(x=NO_TRAIN, color='k', linestyle='--')
    plt.title("Seed:{} PRNG prediction".format(str(row['seed'])))
    plt.xlabel('Index')
    plt.ylabel('Prediction')
    plt.legend()


def cross_correlation(row: np.array, data_predicted: np.array):
    fig, ax = plt.subplots(1, 1)
    train = np.array([range(0, NO_POINTS), row[0:NO_POINTS]]).T.astype(float)
    test = np.array([range(NO_TRAIN, NO_POINTS), list(data_predicted)]).T.astype(float)
    plt.xcorr(train[-NO_TEST:, 1], test[-NO_TEST:, 1], maxlags=8, usevlines=False)
    plt.axvline(x=0, color='k', linestyle='--')
    plt.title("Cross-correlation between the predicted and real values")
    plt.xlabel('Sequence lags')
    plt.legend()
