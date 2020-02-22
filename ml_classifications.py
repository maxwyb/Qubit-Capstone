import pandas as pd
import numpy as np
import csv
import pickle
from os import path
from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, VotingClassifier

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

RANDOM_SEED = 42

# Data pre-processing
BEST_ARRIVAL_TIME_THRESHOLD = 0.00529914

PRE_ARRIVAL_TIME_THRESHOLD = 0.000722906  # from "Distribution of Photons Arrival Times" graph
POST_ARRIVAL_TIME_THRSHOLD = 0.00522625


class Histogramize(BaseEstimator, TransformerMixin):
    def __init__(self, arrival_time_threshold=(0, BEST_ARRIVAL_TIME_THRESHOLD), num_buckets=6):
        self.arrival_time_threshold = arrival_time_threshold
        self.num_buckets = num_buckets
    
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        histogram_bins = np.linspace(
            self.arrival_time_threshold[0], self.arrival_time_threshold[1], 
            num=(self.num_buckets+1), endpoint=True)
        return list(map(
            lambda measurement: np.histogram(measurement, bins=histogram_bins)[0], X))


# Classifiers
BEST_PHOTON_COUNT_THRESHOLD = 12

class ThresholdCutoffClassifier(BaseEstimator, ClassifierMixin):
    """
    Given a histogram of photons' arrival times as the data instance, 
    classify the qubit by counting the number of captured photons
    """
    def __init__(self, threshold=BEST_PHOTON_COUNT_THRESHOLD):
        self.threshold = threshold

    def fit(self, X, y):
        return self

    def predict(self, X):
        return list(map(lambda instance: 0 if sum(instance) > self.threshold else 1, X))


def classifier_train(classifier, qubits_measurements_train, qubits_truths_train):
    log("Training Classifier: {}".format(classifier))
    classifier.fit(qubits_measurements_train, qubits_truths_train)
    return classifier


_classifier_test_counter = 0
CLASSIFIER_TEST_OUTPUT_FILENAME_BASE = 'classifier_test_result'

def classifier_test(classifier, qubits_measurements_train, qubits_measurements_test, 
        qubits_truths_train, qubits_truths_test):
    global _classifier_test_counter
    log("Testing classifier: {}".format(classifier))

    # qubits_predict_train = classifier.predict(qubits_measurements_train)
    qubits_predict_test = classifier.predict(qubits_measurements_test)

    # print("Classification Report on Training Data:")
    # print(confusion_matrix(qubits_truths_train, qubits_predict_train))
    # print(classification_report(qubits_truths_train, qubits_predict_train, digits=8))

    print("Classification Report on Testing Data:")
    print(confusion_matrix(qubits_truths_test, qubits_predict_test))
    print(classification_report(qubits_truths_test, qubits_predict_test, digits=8))

    # Print out all instances of false positives and false negatives in the test set
    # assert(len(qubits_measurements_train) == len(qubits_truths_train) == len(qubits_predict_train))
    assert(len(qubits_measurements_test) == len(qubits_truths_test) == len(qubits_predict_test))
    false_positives_test = list(map(
        lambda index: qubits_measurements_test[index], 
        list(filter(
            lambda index: qubits_truths_test[index] == 0 and qubits_predict_test[index] == 1, 
            range(len(qubits_measurements_test))))))
    false_negatives_test = list(map(
        lambda index: qubits_measurements_test[index], 
        list(filter(
            lambda index: qubits_truths_test[index] == 1 and qubits_predict_test[index] == 0, 
            range(len(qubits_measurements_test))))))

    output_filename = "{base}_{counter}.txt".format(
        base=CLASSIFIER_TEST_OUTPUT_FILENAME_BASE, counter=_classifier_test_counter)
    with open(output_filename, 'w') as file:
        file.write("Classifier: \n")
        file.write(str(classifier))
        file.write("\nFalse Positives: \n")
        for instance in false_positives_test:
            file.write(str(list(instance)) + '\n')
        file.write("\nFalse Negatives: \n")
        for instance in false_negatives_test:
            file.write(str(list(instance)) + '\n')

    _classifier_test_counter += 1
    log("Falsely-classified instances written to the report file.")

    return accuracy_score(qubits_truths_test, qubits_predict_test)


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
        ('hstgm', Histogramize(num_buckets=6)),
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


# Majority Vote with improved Feed-forward Neural Network
def majority_vote_grid_search_cv(qubits_measurements_train, qubits_truths_train):
    log("Starting Grid Search with Cross Validation on Majority Vote Classifier.")

    mv_pipeline = Pipeline([
        ('hstgm', Histogramize(
            arrival_time_threshold=(PRE_ARRIVAL_TIME_THRESHOLD, POST_ARRIVAL_TIME_THRSHOLD))),
        ('clf', VotingClassifier([
            ('mlp', 
                MLPClassifier(activation='relu', solver='adam', hidden_layer_sizes=(32, 32), random_state=RANDOM_SEED)),
            ('lg', 
                LogisticRegression(solver='liblinear', penalty='l2', C=10**-3, random_state=RANDOM_SEED)),
            ('tc', 
                ThresholdCutoffClassifier(threshold=BEST_PHOTON_COUNT_THRESHOLD))
        ]))
    ])

    mv_param_grid = {
        'hstgm__num_buckets': range(2, 33)
    }

    mv_grid = GridSearchCV(mv_pipeline, cv=4, n_jobs=-1, param_grid=mv_param_grid, scoring="accuracy", refit=True, verbose=2)
    mv_grid.fit(qubits_measurements_train, qubits_truths_train)
    return mv_grid


# Main Tasks
def run_mlp_classifier_in_paper():
    qubits_measurements, qubits_truths = load_datasets()
    qubits_measurements_train, qubits_measurements_test, qubits_truths_train, qubits_truths_test = \
        train_test_split(qubits_measurements, qubits_truths, test_size=0.20, random_state=42)

    log("Histogramizing training and testing data.")
    histogramizer = Histogramize(num_buckets=6)
    qubits_measurements_train_histogram = histogramizer.transform(qubits_measurements_train)
    qubits_measurements_test_histogram = histogramizer.transform(qubits_measurements_test)

    mlp = picklize("mlp")(classifier_train)(
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
    print("Best parameters found in Grid Search:")
    print(mlp_grid.best_params_)

    classifier_test(mlp_grid, qubits_measurements_train, qubits_measurements_test, 
        qubits_truths_train, qubits_truths_test)


def run_mlp_with_kfold_data_split():
    """
    Run the best model gotten from "run_mlp", but using 5-fold training/testing dataset split
    """
    qubits_measurements, qubits_truths = tuple(map(lambda dataset: np.array(dataset), load_datasets()))

    kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    clf_accuracies = []
    _i_fold = 0
    for train_index, test_index in kf.split(qubits_measurements):
        _i_fold += 1
        log("Train/Test data split {fold}-th fold.".format(fold=_i_fold))

        qubits_measurements_train, qubits_measurements_test, qubits_truths_train, qubits_truths_test = \
            qubits_measurements[train_index], qubits_measurements[test_index], \
            qubits_truths[train_index], qubits_truths[test_index]

        mlp_pipeline = classifier_train(Pipeline([
                ('hstgm', Histogramize(num_buckets=6, arrival_time_threshold=(0, BEST_ARRIVAL_TIME_THRESHOLD))),
                ('clf', MLPClassifier(hidden_layer_sizes=(32, 32), activation='relu', solver='adam'))
            ]), qubits_measurements_train, qubits_truths_train)

        curr_accuracy = classifier_test(mlp_pipeline, qubits_measurements_train, qubits_measurements_test, 
                qubits_truths_train, qubits_truths_test)
        print("Train/test split {fold}-th fold accuracy: {accuracy}".format(fold=_i_fold, accuracy=curr_accuracy))
        clf_accuracies.append(curr_accuracy)
    
    avg_accuracy = sum(clf_accuracies) / len(clf_accuracies)
    print("Majority Vote with KFold Data Split: Average Accuracy = {accuracy}".format(accuracy=avg_accuracy))


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


def run_majority_vote_with_kfold_data_split():
    """
    Split the dataset 5 times into 80% training and 20% testing, 
    perform Majority Vote classification on each set, and report
    the average accuracy across the 5 folds.
    """
    qubits_measurements, qubits_truths = load_datasets()
    qubits_measurements = np.array(qubits_measurements)
    qubits_truths = np.array(qubits_truths)

    kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    clf_accuracies = []
    _i_fold = 0
    for train_index, test_index in kf.split(qubits_measurements):
        _i_fold += 1
        log("Train/Test data split {fold}-th fold.".format(fold=_i_fold))

        qubits_measurements_train, qubits_measurements_test, qubits_truths_train, qubits_truths_test = \
            qubits_measurements[train_index], qubits_measurements[test_index], \
            qubits_truths[train_index], qubits_truths[test_index]

        mv_grid = picklize("majority_vote_grid_search_cv_{fold}".format(fold=_i_fold)) \
            (majority_vote_grid_search_cv)(qubits_measurements_train, qubits_truths_train)
        print(mv_grid.best_params_)

        curr_accuracy = classifier_test(mv_grid, qubits_measurements_train, qubits_measurements_test, 
                qubits_truths_train, qubits_truths_test)
        print("Train/test split {fold}-th fold accuracy: {accuracy}".format(fold=_i_fold, accuracy=curr_accuracy))
        clf_accuracies.append(curr_accuracy)
    
    avg_accuracy = sum(clf_accuracies) / len(clf_accuracies)
    print("Majority Vote with KFold Data Split: Average Accuracy = {accuracy}".format(accuracy=avg_accuracy))
    

if __name__ == '__main__':
    # run_mlp_classifier_in_paper()
    # run_mlp()
    # run_logistic_regression()
    # run_random_forest()
    # run_majority_vote_with_kfold_data_split()
    run_mlp_with_kfold_data_split()
    log("Done.")
