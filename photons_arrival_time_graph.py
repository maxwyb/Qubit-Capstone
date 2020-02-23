from os import sys
import csv
import numpy as np
from matplotlib import pyplot as plt

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


def use_ramdisk(filenames):
    return list(map(lambda filename: "/Volumes/ramdisk/" + filename, filenames))


def load_datasets():
    qubits_measurements = []
    for dataset_filename in BRIGHT_QUBITS_DATASETS + DARK_QUBITS_DATASETS:
        with open(dataset_filename, 'r') as dataset_file:
            log("Loading {}".format(dataset_filename))
            csv_reader = csv.reader(dataset_file)
            for line in csv_reader:
                qubits_measurements.extend(
                    list(map(lambda timestamp: float(timestamp), line)))
    return qubits_measurements


def draw_plot(qubits_measurements):
    log("Plotting histogram graph.")
    fig, ax = plt.subplots()
    ax.set_title("Distribution of Photons' Arrival Times")
    ax.set_xlabel("Time")
    ax.set_ylabel("Number of Photons")
    ax.set_xticks(np.linspace(0, 0.006, 21))
    plt.hist(qubits_measurements, bins=2000)
    plt.show()


if __name__ == '__main__':
    qubits_measurements = load_datasets()
    draw_plot(qubits_measurements)
    log("Done.")
