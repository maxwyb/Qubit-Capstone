import pandas as pd
import numpy as np
import csv
import pickle
from os import path
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

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
                log("Pickle: Loading from database {}".format(db_filename))
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
BEST_ARRIVAL_TIME_THRESHOLD = 0.00529914


class Histogramize(BaseEstimator, TransformerMixin):
    def __init__(self, arrival_time_threshold=BEST_ARRIVAL_TIME_THRESHOLD, num_buckets=5):
        self.arrival_time_threshold = arrival_time_threshold
        self.num_buckets = num_buckets
    
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        histogram_bins = np.linspace(0, self.arrival_time_threshold, num=(self.num_buckets + 1), endpoint=True)
        return list(map(
            lambda measurement: np.histogram(measurement, bins=histogram_bins)[0], X))


# Classifiers
def classifier_train(classifier, qubits_measurements_train, qubits_truths_train):
    log("Training Classifier: {}".format(classifier))
    classifier.fit(qubits_measurements_train, qubits_truths_train)
    return classifier


def classifier_test(classifier, qubits_measurements_train, qubits_measurements_test, 
        qubits_truths_train, qubits_truths_test):
    log("Testing classifier: {}".format(classifier))

    qubits_predict_train = classifier.predict(qubits_measurements_train)
    qubits_predict_test = classifier.predict(qubits_measurements_test)

    print("Classification Report on Training Data:")
    print(confusion_matrix(qubits_truths_train, qubits_predict_train))
    print(classification_report(qubits_truths_train, qubits_predict_train, digits=8))

    print("Classification Report on Testing Data:")
    print(confusion_matrix(qubits_truths_test, qubits_predict_test))
    print(classification_report(qubits_truths_test, qubits_predict_test, digits=8))


# MLP Classifier
def mlp_grid_search_cv(qubits_measurements_train, qubits_truths_train):
    log("Starting Grid Search with Cross Validation on MLP Classifier.")
    
    mlp_pipeline = Pipeline([
        ('hstgm', Histogramize(num_buckets=6)),
        ('clf', MLPClassifier(activation='relu', solver='adam'))
    ])

    mlp_param_grid = {
        # 'hstgm__num_buckets': range(2, 33),
        'clf__hidden_layer_sizes': [(n, n) for n in range(8, 41)]
        # 'clf__learning_rate_init': [0.001, 0.0005],
        # 'clf__max_iter': [200, 500]
    }

    mlp_grid = GridSearchCV(mlp_pipeline, cv=4, n_jobs=-1, param_grid=mlp_param_grid, scoring="accuracy", verbose=2)
    mlp_grid.fit(qubits_measurements_train, qubits_truths_train)
    return mlp_grid


# Logistic Regression
def logistic_regression_grid_search_cv(qubits_measurements_train, qubits_truths_train):
    log("Starting Grid Search with Cross Validation on Logistic Regression models.")

    lg_pipeline = Pipeline([
        ('hstgm', Histogramize(arrival_time_threshold=BEST_ARRIVAL_TIME_THRESHOLD, num_buckets=6)),
        ('clf', LogisticRegression(solver='liblinear', random_state=42))
    ])

    lg_param_grid = {
        # 'histogram__num_buckets': range(2, 33),
        'clf__penalty': ['none', 'l1', 'l2'],
        'clf__C': [10**-3, 10**-2, 10**-1, 10**0, 10**1, 10**2, 10**3]
    }

    lg_grid = GridSearchCV(lg_pipeline, cv=4, n_jobs=-1, param_grid=lg_param_grid, scoring="accuracy", refit=True, verbose=2)
    lg_grid.fit(qubits_measurements_train, qubits_truths_train)
    return lg_grid


# Random Forest
def random_forest_grid_search_cv(qubits_measurements_train, qubits_truths_train):
    log("Starting Grid Search with Cross Validation on Random Forest Classifier.")
    
    rf_pipeline = Pipeline([
        ('hstgm', Histogramize(num_buckets=6)),
        ('clf', RandomForestClassifier())
    ])

    rf_param_grid = {}

    rf_grid = GridSearchCV(rf_pipeline, cv=4, n_jobs=-1, param_grid=rf_param_grid, scoring="accuracy", verbose=2)
    rf_grid.fit(qubits_measurements_train, qubits_truths_train)
    return rf_grid


# Main Tasks
def run_mlp_classifier_in_paper():
    qubits_measurements, qubits_truths = load_datasets()
    qubits_measurements_train, qubits_measurements_test, qubits_truths_train, qubits_truths_test = \
        train_test_split(qubits_measurements, qubits_truths, test_size=0.20, random_state=42)

    log("Histogramizing training and testing data.")
    histogramizer = Histogramize(num_buckets=6)
    qubits_measurements_train_histogram = histogramizer.transform(qubits_measurements_train)
    qubits_measurements_test_histogram = histogramizer.transform(qubits_measurements_test)

    mlp = picklize("mlp", overwrite=True)(classifier_train)(
        MLPClassifier(hidden_layer_sizes=(8, 8), activation='relu', solver='adam'),  # 2-layer feed-forward neural network used in the paper
        qubits_measurements_train_histogram, qubits_truths_train)
    classifier_test(mlp, qubits_measurements_train_histogram, qubits_measurements_test_histogram, 
        qubits_truths_train, qubits_truths_test)


def run_mlp():
    qubits_measurements, qubits_truths = load_datasets()
    qubits_measurements_train, qubits_measurements_test, qubits_truths_train, qubits_truths_test = \
        train_test_split(qubits_measurements, qubits_truths, test_size=0.20, random_state=42)
        
    mlp_grid = picklize('mlp_grid_search_cv') \
        (mlp_grid_search_cv)(qubits_measurements_train, qubits_truths_train)
    log(pd.DataFrame(mlp_grid.cv_results_))

    classifier_test(mlp_grid, qubits_measurements_train, qubits_measurements_test, 
        qubits_truths_train, qubits_truths_test)


def run_logistic_regression():
    qubits_measurements, qubits_truths = load_datasets()
    qubits_measurements_train, qubits_measurements_test, qubits_truths_train, qubits_truths_test = \
        train_test_split(qubits_measurements, qubits_truths, test_size=0.20, random_state=42)

    lg_grid = picklize('logistic_regression_grid_search_cv') \
        (logistic_regression_grid_search_cv) \
        (qubits_measurements_train, qubits_truths_train)

    classifier_test(lg_grid, qubits_measurements_train, qubits_measurements_test, 
        qubits_truths_train, qubits_truths_test)


def run_random_forest():
    qubits_measurements, qubits_truths = load_datasets()
    qubits_measurements_train, qubits_measurements_test, qubits_truths_train, qubits_truths_test = \
        train_test_split(qubits_measurements, qubits_truths, test_size=0.20, random_state=42)
        
    rf_grid = picklize('random_forest_grid_search_cv') \
        (random_forest_grid_search_cv)(qubits_measurements_train, qubits_truths_train)
    log(pd.DataFrame(rf_grid.cv_results_))

    classifier_test(rf_grid, qubits_measurements_train, qubits_measurements_test, 
        qubits_truths_train, qubits_truths_test)


if __name__ == '__main__':
    # run_mlp_classifier_in_paper()
    # run_mlp()
    run_logistic_regression()
    # run_random_forest()
