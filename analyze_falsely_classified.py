from os import sys
import csv
from collections import defaultdict
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


def load_datasets():
    qubits_measurements = []
    for dataset_filename in BRIGHT_QUBITS_DATASETS + DARK_QUBITS_DATASETS:
        with open(dataset_filename, 'r') as dataset_file:
            log("Loading {}".format(dataset_filename))
            csv_reader = csv.reader(dataset_file)
            for line in csv_reader:
                qubits_measurements.append(
                    list(map(lambda timestamp: float(timestamp), line)))
    return qubits_measurements


def load_classifier_test_results(filenames):
    fp_instances = []  # false positives
    fn_instances = []  # false negatives
    for result_filename in filenames:
        with open(result_filename, 'r') as result_file:
            log("Loading {}".format(result_filename))
            csv_reader = csv.reader(result_file)
            for line in csv_reader:
                if line[0] == 'FALSE_POSITIVE':
                    fp_instances.append(
                        list(map(lambda timestamp: float(timestamp), line[1:])))
                if line[0] == 'FALSE_NEGATIVE':
                    fn_instances.append(
                        list(map(lambda timestamp: float(timestamp), line[1:])))
    return fp_instances, fn_instances


def find_instances_indices(dataset, instances):
    def index_approximate(source, target):
        """
        list.find() but account for precision in floating point numbers
        """
        PRECISION = 0.000000001
        for index, candidate in enumerate(source):
            if len(target) == len(candidate):
                _cond = sum(
                    [(True if target[i] < candidate[i] + PRECISION and target[i] > candidate[i] - PRECISION else False)
                        for i in len(target)]) == len(target)
                if _cond:
                    return index
        raise ValueError('No index found.')

    log("Mapping instances to indices in the dataset.")
    # return list(map(lambda instance: index_approximate(dataset, instance), instances))
    return list(map(lambda instance: dataset.index(instance), instances))


def draw_plot(fp_indices, fn_indices):
    log("Plotting instances frequency plot.")
    fp_index_frequency = defaultdict(int)
    fn_index_frequency = defaultdict(int)
    for index in fp_indices:
        fp_index_frequency[index] += 1
    for index in fn_indices:
        fn_index_frequency[index] += 1

    print("False Positives: ")
    print(dict(fp_index_frequency))
    print("False Negatives: ")
    print(dict(fn_index_frequency))

    # plt.tight_layout()  # prevent overlapping of xlabel and subplot title
    fig = plt.figure()
    fig.suptitle("Frequency of Falsely-classified Instances")
    ax = plt.subplot(2, 1, 1)
    ax.set_title("False Positive Instances")
    ax.set_xlabel("Instance Index")
    ax.set_ylabel("Occurrences")
    ax.bar(fp_index_frequency.keys(), fp_index_frequency.values())

    ax = plt.subplot(2, 1, 2)
    ax.set_title("False Negative Instances")
    ax.set_xlabel("Instance Index")
    ax.set_ylabel("Occurrences")
    ax.bar(fn_index_frequency.keys(), fn_index_frequency.values())
    
    plt.show()


if __name__ == '__main__':
    # draw_plot([1, 1, 1, 5, 5, 2], [1, 1, 1, 5, 5, 2])
    dataset = load_datasets()
    fp_instances, fn_instances = \
        load_classifier_test_results(
            ['classifier_test_result_mlp_{}.csv'.format(n) for n in range(0, 5)]
            + ['classifier_test_result_mlp_kfold_{}.csv'.format(n) for n in range(0, 5)])
    fp_indices = find_instances_indices(dataset, fp_instances)
    fn_indices = find_instances_indices(dataset, fn_instances)
    draw_plot(fp_indices, fn_indices)
    log("Done.")
