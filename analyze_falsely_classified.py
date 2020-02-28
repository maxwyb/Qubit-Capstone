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

MOST_NUMBER_OF_PHOTONS_CAPTURED = 77


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


def write_instances_to_file(dataset, fp_indices, fn_indices):
    def _write(dataset, indices, filename):
        with open(filename, 'w') as file:
            csv_writer = csv.writer(file)
            for index in set(indices):
                csv_writer.writerow(dataset[index])
    _write(dataset, fp_indices, 'false_positive_instances.csv')
    _write(dataset, fn_indices, 'false_negative_instances.csv')


def draw_plot_misclassified_indices(fp_indices, fn_indices):
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

    # max_overlap_count = max(list(fp_index_frequency.values()) + list(fn_index_frequency.values()))
    max_overlap_count = 3  # TODO: frequency value currently can be higher than the number of classifiers, use hardcoded value for now
    overlap_percentage = len(
            list(filter(lambda index: fp_index_frequency[index] == max_overlap_count, list(fp_index_frequency.keys())))
                + list(filter(lambda index: fn_index_frequency[index] == max_overlap_count, list(fn_index_frequency.keys())))
        ) / len(list(fp_index_frequency.keys()) + list(fn_index_frequency.keys()))
    print("Overlap Percentage - instances where most number of classifiers get wrong: {}".format(overlap_percentage))

    # plt.tight_layout()  # prevent overlapping of xlabel and subplot title

    # fig = plt.figure()
    # fig.suptitle("Frequency of Falsely-classified Instances")
    # ax = plt.subplot(2, 1, 1)
    # ax.set_title("False Positive Instances")
    # ax.set_xlabel("Instance Index")
    # ax.set_ylabel("Occurrences")
    # ax.bar(list(map(lambda index: str(index), fp_index_frequency.keys())), fp_index_frequency.values())

    # ax = plt.subplot(2, 1, 2)
    # ax.set_title("False Negative Instances")
    # ax.set_xlabel("Instance Index")
    # ax.set_ylabel("Occurrences")
    # ax.bar(list(map(lambda index: str(index), fn_index_frequency.keys())), fn_index_frequency.values())
    
    # plt.show()


def draw_plot_photons_count(fp_instances, fn_instances):
    bright_qubits_frequency = defaultdict(int)
    dark_qubits_frequency = defaultdict(int)
    for instance in fp_instances:  # NOTE: instance mis-classified multiple times by 2+ classifiers will be counted multiple times here
        bright_qubits_frequency[len(instance)] += 1
    for instance in fn_instances:
        dark_qubits_frequency[len(instance)] += 1
    
    ax = plt.subplot(1, 1, 1)
    ax.set_title("Falsely-classified Qubits with Number of Photons Captured")
    ax.set_ylabel("Qubits Count")
    ax.set_xlabel("Number of Photons")
    ax.bar(
        [n for n in range(0, MOST_NUMBER_OF_PHOTONS_CAPTURED+1)], 
        [bright_qubits_frequency[i] for i in range(0, MOST_NUMBER_OF_PHOTONS_CAPTURED+1)],
        label="Bright Qubits")
    ax.bar(
        [n for n in range(0, MOST_NUMBER_OF_PHOTONS_CAPTURED+1)], 
        [dark_qubits_frequency[i] for i in range(0, MOST_NUMBER_OF_PHOTONS_CAPTURED+1)],
        label="Dark Qubits")
    
    ax.legend()
    plt.show()
    

if __name__ == '__main__':
    # draw_plot([1, 1, 1, 5, 5, 2], [1, 1, 1, 5, 5, 2])
    dataset = load_datasets()
    fp_instances, fn_instances = load_classifier_test_results(
            # ['classifier_test_result_mlp_{}.csv'.format(n) for n in range(0, 5)]
            # + ['classifier_test_result_mlp_kfold_{}.csv'.format(n) for n in range(0, 5)])
            ['Results/falsely-classified-instances-mlp-lg-rf/classifier_test_result_{}.csv'.format(n) for n in range(0, 15)])
    fp_indices = find_instances_indices(dataset, fp_instances)
    fn_indices = find_instances_indices(dataset, fn_instances)
    # draw_plot_misclassified_indices(fp_indices, fn_indices)
    draw_plot_photons_count(fp_instances, fn_instances)
    # write_instances_to_file(dataset, fp_indices, fn_indices)
    log("Done.")
