from os import sys
import csv
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import  KFold, StratifiedKFold


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


def draw_plot(qubits_measurements, qubits_truths, iterator):
    log("Plotting histograms of iterator: {iterator}".format(iterator=iterator))

    fig = plt.figure()
    fig.suptitle("Photons' Arrival Times Histogram of Each Training/Testing Set")
    n_splits = iterator.get_n_splits()
    _i = 0
    for train_index, test_index in iterator.split(qubits_measurements, qubits_truths):
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

if __name__ == '__main__':
    qubits_measurements, qubits_truths = tuple(map(lambda dataset: np.array(dataset), load_datasets()))
    draw_plot(qubits_measurements, qubits_truths, StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED))
    draw_plot(qubits_measurements, qubits_truths, KFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED))
    log("Done.")
