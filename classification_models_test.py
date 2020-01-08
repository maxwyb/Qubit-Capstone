import argparse
from enum import Enum
import csv
from abc import ABC, abstractmethod
import sys
import numpy as np
import matplotlib.pyplot as plt

# Assume the given files are compiled by the ground truth (if the measured qubit is dark or bright)
BRIGHT_QUBITS_MEASUREMENTS = [
    'Data4Jens/BrightTimeTagSet1.csv',
    'Data4Jens/BrightTimeTagSet2.csv',
    'Data4Jens/BrightTimeTagSet3.csv',
    'Data4Jens/BrightTimeTagSet4.csv',
    'Data4Jens/BrightTimeTagSet5.csv',
]

DARK_QUBITS_MEASUREMENTS = [
    'Data4Jens/DarkTimeTagSet1.csv',
    'Data4Jens/DarkTimeTagSet2.csv',
    'Data4Jens/DarkTimeTagSet3.csv',
    'Data4Jens/DarkTimeTagSet4.csv',
    'Data4Jens/DarkTimeTagSet5.csv',
]


class Qubit(Enum):
    BRIGHT = 0
    DARK = 1


class QubitMeasurement():
    def __init__(self, photons, ground_truth):
        super().__init__()
        self.photons = photons
        self.ground_truth = ground_truth
        self.classified_result = None


class ClassificationModel(ABC):
    def __init__(self):
        super().__init__()
    
    @abstractmethod
    def classify(self, qubit_measurement):
        pass


class ThresholdCutoffModel(ClassificationModel):
    def __init__(self, threshold):
        super().__init__()
        self.threshold = threshold

    def __str__(self):
        return "Threshold Cutoff Model w/ threshold {}".format(self.threshold)

    def classify(self, qubit_measurement):
        return Qubit.BRIGHT if len(qubit_measurement.photons) > self.threshold else Qubit.DARK


def get_arguments():
    parser = argparse.ArgumentParser()
    return parser.parse_args


def log(message):
    sys.stderr.write(message + '\n')


def read_qubit_measurements():
    def read_from_files_with_ground_truth(filenames, ground_truth, qubit_measurements):
        for measurement_filename in filenames:
            log("Loading {}".format(measurement_filename))
            with open(measurement_filename, 'r') as measurement_file:
                reader = csv.reader(measurement_file)
                for photons in reader:
                    qubit_measurements.append(QubitMeasurement([float(photon) for photon in photons], ground_truth))
        return qubit_measurements

    qubit_measurements = []
    read_from_files_with_ground_truth(BRIGHT_QUBITS_MEASUREMENTS, Qubit.BRIGHT, qubit_measurements)
    read_from_files_with_ground_truth(DARK_QUBITS_MEASUREMENTS, Qubit.DARK, qubit_measurements)
    return qubit_measurements


def classify_qubits(model, qubit_measurements):
    log("Classifying qubit measurements with {}".format(model))
    for measurement in qubit_measurements:
        measurement.classified_result = model.classify(measurement)
    return


def gather_measurement_statistics(qubit_measurements):
    datapoints = len(qubit_measurements)
    false_positives = len(list(filter(
        lambda measurement: measurement.ground_truth == Qubit.BRIGHT and measurement.classified_result == Qubit.DARK, 
        qubit_measurements)))
    false_negatives = len(list(filter(
        lambda measurement: measurement.ground_truth == Qubit.DARK and measurement.classified_result == Qubit.BRIGHT, 
        qubit_measurements)))
    reliability = 1 - (false_positives + false_negatives) / datapoints

    print("Datapoints: {}\nFalse Positives : {}\nFalse Negatives: {}\nReliability: {}".format(
        datapoints, false_positives, false_negatives, reliability))
    return reliability


def threshold_cutoff_experiments():
    options = get_arguments()
    qubit_measurements = read_qubit_measurements()

    _most_photons_received = max(list(map(lambda measurement: len(measurement.photons), qubit_measurements)))
    print("Max number of photons captured for one qubit: {}".format(_most_photons_received))

    _accuracy_results = []
    # try to classify measurements with a range of cutoff values and look at their accuracy
    for threshold in range(0, _most_photons_received + 1):
        model = ThresholdCutoffModel(threshold)
        classify_qubits(model, qubit_measurements)
        reliability = gather_measurement_statistics(qubit_measurements)
        _accuracy_results.append((threshold, reliability))
    
    print("Threshold Cutoff Model Accuracy:")
    for threshold, reliability in _accuracy_results:
        print("{},{}".format(threshold, reliability))


def find_false_classifications_with_photon_histogram():
    """
    Classify qubits by the Threshold Cutoff Model with the optimal threshold, find all mis-classified qubits and
    print the histogram of each's measured photons (frequency of every arriving time interval)
    """
    options= get_arguments()
    qubit_measurements = read_qubit_measurements()
    model = ThresholdCutoffModel(12)
    classify_qubits(model, qubit_measurements)
    reliability = gather_measurement_statistics(qubit_measurements)

    misclassified_qubits = list(filter(
        lambda measurement: measurement.ground_truth != measurement.classified_result, qubit_measurements))
    false_positive_qubits = list(filter(
        lambda measurement: measurement.ground_truth == Qubit.BRIGHT and measurement.classified_result == Qubit.DARK,
        qubit_measurements))
    false_negative_qubits = list(filter(
        lambda measurement: measurement.ground_truth == Qubit.DARK and measurement.classified_result == Qubit.BRIGHT,
        qubit_measurements))

    measurement_photon_histograms = [np.histogram(qubit.photons) for qubit in misclassified_qubits]
    print("One historgram of measured photons in a mis-classifed qubit: \n{}".format(measurement_photon_histograms[0]))

    plt.figure(0, figsize=(9, 8))
    plt.title("Histogram of False Positive Qubits")
    for index in range(len(false_positive_qubits)):
        plt.hist(false_positive_qubits[index].photons)
    plt.figure(1, figsize=(9, 8))
    plt.title("Histogram of False Negative Qubits")
    for index in range(len(false_negative_qubits)):
        plt.hist(false_negative_qubits[index].photons)
    plt.show()


if __name__ == '__main__':
    find_false_classifications_with_photon_histogram()
