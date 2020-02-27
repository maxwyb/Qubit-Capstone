from os import sys
import csv
from collections import defaultdict
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.pyplot import subplots_adjust
from sklearn.model_selection import  KFold, StratifiedKFold, train_test_split


def log(message):
    print(message, file=sys.stderr)


BRIGHT_QUBITS_DATASETS = [
    'Data4Jens/BrightTimeTagSet1.csv',
    'Data4Jens/BrightTimeTagSet2.csv',
    'Data4Jens/BrightTimeTagSet3.csv',
    'Data4Jens/BrightTimeTagSet4.csv',
    'Data4Jens/BrightTimeTagSet5.csv',
]

DARK_QUBITS_DATASETS = [
    'Data4Jens/DarkTimeTagSet1.csv',
    'Data4Jens/DarkTimeTagSet2.csv',
    'Data4Jens/DarkTimeTagSet3.csv',
    'Data4Jens/DarkTimeTagSet4.csv',
    'Data4Jens/DarkTimeTagSet5.csv',
]


RANDOM_SEED = 42


def load_datasets():
    def load_datasets_with_ground_truth(qubits_datasets, ground_truth):
        qubits_measurements = []
        for dataset_filename in qubits_datasets:
            with open(dataset_filename, 'r') as dataset_file:
                log("Loading {}".format(dataset_filename))
                csv_reader = csv.reader(dataset_file)
                for line in csv_reader:
                    qubits_measurements.append(
                        np.array(list(map(lambda timestamp: float(timestamp), line)))
                    )
        qubits_ground_truths = [ground_truth for i in range(len(qubits_measurements))]
        return qubits_measurements, qubits_ground_truths
    
    bright_qubits_measurements, bright_qubits_ground_truths = load_datasets_with_ground_truth(BRIGHT_QUBITS_DATASETS, 0)
    dark_qubits_measurements, dark_qubits_ground_truths = load_datasets_with_ground_truth(DARK_QUBITS_DATASETS, 1)
    return (
        (bright_qubits_measurements + dark_qubits_measurements), 
        (bright_qubits_ground_truths + dark_qubits_ground_truths))


def draw_photon_histogram_plot(qubits_measurements, qubits_truths, split_indicies):
    log("Plotting photons arrival time histograms of data split.")

    fig = plt.figure()
    fig.suptitle("Photons' Arrival Times Histogram of Each Training/Testing Set")
    n_splits = len(split_indicies)
    _i = 0
    for train_index, test_index in split_indicies:
        log("Plotting histogram at {i}-th fold.".format(i=_i+1))

        qubits_measurements_train, qubits_measurements_test = \
            qubits_measurements[train_index], qubits_measurements[test_index]

        ax = plt.subplot(n_splits, 2, _i*2+1)  # training set histogram of the i-th fold
        ax.set_title("{i}-th Fold: Training".format(i=_i+1))
        ax.set_ylabel("Amount")
        ax.set_xlabel("Time")
        ax.hist([timestamp for measurement in qubits_measurements_train for timestamp in measurement], bins=int(1000/n_splits*(n_splits-1)))

        ax = plt.subplot(n_splits, 2, _i*2+2)  # testing set histogram of the i-th fold
        ax.set_title("{i}-th Fold: Testing".format(i=_i+1))
        ax.set_ylabel("Amount")
        ax.set_xlabel("Time")
        ax.hist([timestamp for measurement in qubits_measurements_test for timestamp in measurement], bins=int(1000/n_splits))

        _i = _i + 1
    
    # plt.tight_layout()
    plt.show()


MOST_NUMBER_OF_PHOTONS_CAPTURED = 77

def draw_photon_count_frequency_plot(qubits_measurements, qubits_truths, split_indicies):
    log("Plotting frequency bar charts by number of photons captured.")

    fig = plt.figure()
    fig.suptitle("Qubits Count per Number of Photons Captured of Each Training/Testing Set")
    n_splits = len(split_indicies)
    _i = 0
    for train_index, test_index in split_indicies:
        log("Plotting bar chart at {i}-th fold.".format(i=_i+1))

        qubits_measurements_train, qubits_measurements_test = \
            qubits_measurements[train_index], qubits_measurements[test_index]

        training_frequency = defaultdict(int)
        testing_frequency = defaultdict(int)
        for measurement in qubits_measurements_train:
            training_frequency[len(measurement)] += 1
        for measurement in qubits_measurements_test:
            testing_frequency[len(measurement)] += 1
    
        ax = plt.subplot(n_splits, 2, _i*2+1)  # training set bar chart of the i-th fold
        ax.set_title("{i}-th Fold: Training".format(i=_i+1))
        ax.set_ylabel("Qubits Count")
        ax.set_xlabel("Number of Photons")
        ax.bar(
            [n for n in range(0, MOST_NUMBER_OF_PHOTONS_CAPTURED+1)], 
            [training_frequency[i] for i in range(0, MOST_NUMBER_OF_PHOTONS_CAPTURED+1)])

        ax = plt.subplot(n_splits, 2, _i*2+2)  # testing set bar chart of the i-th fold
        ax.set_title("{i}-th Fold: Testing".format(i=_i+1))
        ax.set_ylabel("Qubits Count")
        ax.set_xlabel("Number of Photons")
        ax.bar(
            [n for n in range(0, MOST_NUMBER_OF_PHOTONS_CAPTURED+1)], 
            [testing_frequency[i] for i in range(0, MOST_NUMBER_OF_PHOTONS_CAPTURED+1)])

        _i = _i + 1
    
    # plt.tight_layout()
    left  = 0.125  # the left side of the subplots of the figure
    right = 0.9    # the right side of the subplots of the figure
    bottom = 0.05   # the bottom of the subplots of the figure
    top = 0.925      # the top of the subplots of the figure
    wspace = 0.2   # the amount of width reserved for blank space between subplots
    hspace = 0.6   # the amount of height reserved for white space between subplots
    subplots_adjust(left=left, bottom=bottom, right=right, top=top, wspace=wspace, hspace=hspace)
    plt.show()


def draw_photon_count_frequency_plot_split_by_truth(qubits_measurements, qubits_truths, split_indicies):
    log("Plotting frequency bar charts by number of photons captured (split by ground truth).")

    fig = plt.figure()
    fig.suptitle("Qubits Count per Number of Photons Captured of Training/Testing Set")

    # Plot the 1st training/testing split only
    assert(len(split_indicies) > 1)
    train_index, test_index = split_indicies[0]
    qubits_measurements_train, qubits_measurements_test, qubits_truths_train, qubits_truths_test = \
        qubits_measurements[train_index], qubits_measurements[test_index], qubits_truths[train_index], qubits_truths[test_index]

    # Split Bright/Dark qubits in training and testing set
    qubits_measurements_train_bright, qubits_measurements_train_dark = [], []
    qubits_measurements_test_bright, qubits_measurements_test_dark = [], []

    assert(len(qubits_measurements_train) == len(qubits_truths_train))
    assert(len(qubits_measurements_test) == len(qubits_truths_test))
    for index, measurement in enumerate(qubits_measurements_train):
        if qubits_truths_train[index] == 0:
            qubits_measurements_train_bright.append(measurement)
        elif qubits_truths_train[index] == 1:
            qubits_measurements_train_dark.append(measurement)
    for index, measurement in enumerate(qubits_measurements_test):
        if qubits_truths_test[index] == 0:
            qubits_measurements_test_bright.append(measurement)
        elif qubits_truths_test[index] == 1:
            qubits_measurements_test_dark.append(measurement)

    # Build frequency dict
    training_frequency_bright, training_frequency_dark = defaultdict(int), defaultdict(int)
    testing_frequency_bright, testing_frequency_dark = defaultdict(int), defaultdict(int)

    for measurement in qubits_measurements_train_bright:
        training_frequency_bright[len(measurement)] += 1
    for measurement in qubits_measurements_train_dark:
        training_frequency_dark[len(measurement)] += 1
    for measurement in qubits_measurements_test_bright:
        testing_frequency_bright[len(measurement)] += 1
    for measurement in qubits_measurements_test_dark:
        testing_frequency_dark[len(measurement)] += 1

    # Draw plots
    ax = plt.subplot(2, 1, 1)  # training set bar chart 
    ax.set_title("Training Set")
    ax.set_ylabel("Qubits Count")
    ax.set_xlabel("Number of Photons")
    ax.bar(
        [n for n in range(0, MOST_NUMBER_OF_PHOTONS_CAPTURED+1)], 
        [training_frequency_bright[i] for i in range(0, MOST_NUMBER_OF_PHOTONS_CAPTURED+1)],
        label="Bright Qubits")
    ax.bar(
        [n for n in range(0, MOST_NUMBER_OF_PHOTONS_CAPTURED+1)], 
        [training_frequency_dark[i] for i in range(0, MOST_NUMBER_OF_PHOTONS_CAPTURED+1)],
        label="Dark Qubits")

    ax.legend()

    ax = plt.subplot(2, 1, 2)  # testing set bar chart
    ax.set_title("Testing Set")
    ax.set_ylabel("Qubits Count")
    ax.set_xlabel("Number of Photons")
    ax.bar(
        [n for n in range(0, MOST_NUMBER_OF_PHOTONS_CAPTURED+1)], 
        [testing_frequency_bright[i] for i in range(0, MOST_NUMBER_OF_PHOTONS_CAPTURED+1)],
        label="Bright Qubits")
    ax.bar(
        [n for n in range(0, MOST_NUMBER_OF_PHOTONS_CAPTURED+1)], 
        [testing_frequency_dark[i] for i in range(0, MOST_NUMBER_OF_PHOTONS_CAPTURED+1)],
        label="Dark Qubits")
    
    ax.legend()
    
    # plt.tight_layout()
    left  = 0.125  # the left side of the subplots of the figure
    right = 0.9    # the right side of the subplots of the figure
    bottom = 0.05   # the bottom of the subplots of the figure
    top = 0.925      # the top of the subplots of the figure
    wspace = 0.2   # the amount of width reserved for blank space between subplots
    hspace = 0.6   # the amount of height reserved for white space between subplots
    subplots_adjust(left=left, bottom=bottom, right=right, top=top, wspace=wspace, hspace=hspace)
    plt.show()

if __name__ == '__main__':
    qubits_measurements, qubits_truths = tuple(map(lambda dataset: np.array(dataset), load_datasets()))

    # "Just-kidding" Split
    draw_photon_count_frequency_plot_split_by_truth(qubits_measurements, qubits_truths, 
        list(KFold(n_splits=5, shuffle=False).split(qubits_measurements)))

    # Old Split
    # draw_photon_histogram_plot(qubits_measurements, qubits_truths, 
    #     list(KFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED).split(qubits_measurements)))
    # draw_photon_count_frequency_plot(qubits_measurements, qubits_truths, 
    #     list(KFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED).split(qubits_measurements)))

    # Intermediate Split
    # draw_photon_histogram_plot(qubits_measurements, qubits_truths, 
    #     list(StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED).split(qubits_measurements, qubits_truths)))

    # New Split
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    qubits_class = []
    assert(len(qubits_measurements) == len(qubits_truths))
    for index in range(len(qubits_measurements)):
        qubits_class.append(qubits_truths[index] * 100 + len(qubits_measurements[index]))
    # draw_photon_count_frequency_plot(qubits_measurements, qubits_truths, 
    #     list(kf.split(qubits_measurements, qubits_class)))
    draw_photon_count_frequency_plot_split_by_truth(qubits_measurements, qubits_truths, 
        list(kf.split(qubits_measurements, qubits_class))) 

    log("Done.")
