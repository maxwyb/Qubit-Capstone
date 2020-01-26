import pandas as pd
import numpy as np
import csv
import pickle
from os import path
from sklearn.model_selection import train_test_split, GridSearchCV
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


def picklize(db_id, overwrite=False):
    def decorator(function):
        def wrapper(*args, **kwargs):
            def _pickle_db_path(db_id):
                return "pickle_data/{}.pickle".format(db_id)

            db_filename = _pickle_db_path(db_id)
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


# Data pre-processing
ARRIVAL_TIME_THRESHOLD = 0.00529914


def data_preprocessing(histogram_num_buckets=5):
    log("Pre-processing measurement data into histograms with {} buckets.".format(histogram_num_buckets))
    histogram_bins = np.linspace(0, ARRIVAL_TIME_THRESHOLD, num=(histogram_num_buckets + 1), endpoint=True)
    qubits_measurements_histogram = list(map(
        lambda measurement: np.histogram(measurement, bins=histogram_bins)[0], qubits_measurements))
    return qubits_measurements_histogram


def data_train_test_split(qubits_measurements, qubits_truths):
    return train_test_split(qubits_measurements, qubits_truths, test_size=0.20, random_state=42)


# MLP Classifier
def mlp_classifier_train(qubits_measurements_train, qubits_truths_train):
    mlp = MLPClassifier(hidden_layer_sizes=(8, 8), activation='relu', solver='adam')
    log("Start MLP Classifier training.")
    mlp.fit(qubits_measurements_train, qubits_truths_train)
    return mlp


def mlp_classifier_test(qubits_measurements_train, qubits_measurements_test, qubits_truths_train, qubits_truths_test):
    qubits_predict_train = mlp.predict(qubits_measurements_train)
    qubits_predict_test = mlp.predict(qubits_measurements_test)

    print("Classification Report on Train Data:")
    print(confusion_matrix(qubits_truths_train, qubits_predict_train))
    print(classification_report(qubits_truths_train, qubits_predict_train))

    print("Classification Report on Test Data:")
    print(confusion_matrix(qubits_truths_test, qubits_predict_test))
    print(classification_report(qubits_truths_test, qubits_predict_test))


def mlp_grid_search_cross_validation(qubits_measurements_train, qubits_truths_train):
    log("Starting Grid Search with Cross Validation on MLC Classifier.")

    mlp = MLPClassifier(activation='relu', solver='adam', learning_rate='constant')

    mlp_param_grid = {
        'hidden_layer_sizes': [(n, n) for n in range(8, 44, 4)],  # keep at 2 layers
        'learning_rate_init': [0.001, 0.0005],
        'max_iter': [200, 500]
    }

    mlp_grid = GridSearchCV(mlp, cv=4, n_jobs=-1, param_grid=mlp_param_grid, scoring="accuracy")
    mlp_grid.fit(qubits_measurements_train, qubits_truths_train)

    return pd.DataFrame(mlp_grid.cv_results_).sort_values('mean_test_score', ascending=False)


if __name__ == '__main__':
    qubits_measurements, qubits_truths = load_datasets()
    # qubits_measurements_histogram = picklize('qubits_measurements_histogram')(data_preprocessing)(5)
    # qubits_measurements_train, qubits_measurements_test, qubits_truths_train, qubits_truths_test = data_train_test_split(
    #     qubits_measurements_histogram, qubits_truths)
    # mlp = picklize("mlp", overwrite=True)(mlp_classifier_train)(qubits_measurements_train, qubits_truths_train)
    # mlp_classifier_test(qubits_measurements_train, qubits_measurements_test, qubits_truths_train, qubits_truths_test)

    for histogram_num_buckets in range(2, 11):
        qubits_measurements_histogram = picklize('qubits_measurements_histogram_{}'.format(histogram_num_buckets))(
            data_preprocessing)(histogram_num_buckets)
        qubits_measurements_train, qubits_measurements_test, qubits_truths_train, qubits_truths_test = data_train_test_split(
            qubits_measurements_histogram, qubits_truths)
        
        grid_search_result = mlp_grid_search_cross_validation(qubits_measurements_train, qubits_truths_train)
        grid_search_result.to_csv("mlp_grid_search_{}.csv".format(histogram_num_buckets))
