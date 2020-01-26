import pandas as pd
import numpy as np
import csv
import pickle
from os import path
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix

from classification_models_test import log


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


def pickle_db_path(data_name):
    return "pickle_data/{}.pickle".format(data_name)


def pickle_load_store(db_filename, overwrite=False):
    def decorator(function):
        def wrapper(*args, **kwargs):
            if (not overwrite) and path.exists(db_filename):
                log("Loading data from Pickle database {}".format(db_filename))
                with open(db_filename, 'rb') as db_file:
                    return pickle.load(db_file)
            else:
                ret = function(*args, **kwargs)
                with open(db_filename, 'wb') as db_file:
                    pickle.dump(ret, db_file)
                return ret
        return wrapper
    return decorator

# Load datasets
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


# qubits_measurements, qubits_truths = pickle_load_store(pickle_db_path("qubits_measurements_truths"))(load_datasets)()
qubits_measurements, qubits_truths = load_datasets()

# Data pre-processing
ARRIVAL_TIME_THRESHOLD = 0.00529914
HISTOGRAM_NUM_BUCKETS = 5

def data_preprocessing():
    log("Pre-processing measurement data into histograms.")
    histogram_bins = np.linspace(0, ARRIVAL_TIME_THRESHOLD, num=(HISTOGRAM_NUM_BUCKETS + 1), endpoint=True)
    qubits_measurements_histogram = list(map(
        lambda measurement: np.histogram(measurement, bins=histogram_bins)[0], qubits_measurements))
    return qubits_measurements_histogram

qubits_measurements_histogram = pickle_load_store(pickle_db_path('qubits_measurements_histogram'))(data_preprocessing)()
qubits_measurements_train, qubits_measurements_test, qubits_truths_train, qubits_truths_test = train_test_split(
    qubits_measurements_histogram, qubits_truths, test_size=0.30, random_state=42)

# MLP Classifier
def train_mlp_classifier():
    mlp = MLPClassifier(hidden_layer_sizes=(2), activation='relu', solver='adam')
    log("Start MLP Classifier training.")
    mlp.fit(qubits_measurements_train, qubits_truths_train)
    return mlp

mlp = pickle_load_store(pickle_db_path("mlp"))(train_mlp_classifier)()

qubits_predict_train = mlp.predict(qubits_measurements_train)
qubits_predict_test = mlp.predict(qubits_measurements_test)

print("Classification Report on Train Data:")
print(confusion_matrix(qubits_truths_train, qubits_predict_train))
print(classification_report(qubits_truths_train, qubits_predict_train))

print("Classification Report on Test Data:")
print(confusion_matrix(qubits_truths_test, qubits_predict_test))
print(classification_report(qubits_truths_test, qubits_predict_test))
