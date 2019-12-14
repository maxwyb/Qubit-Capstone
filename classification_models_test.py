import argparse
from enum import Enum
import csv
from abc import ABC, abstractmethod
import sys

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
                    qubit_measurements.append(QubitMeasurement(photons, ground_truth))
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


if __name__ == '__main__':
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
