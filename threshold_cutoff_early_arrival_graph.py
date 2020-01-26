import csv
import matplotlib.pyplot as plt
# This import registers the 3D projection, but is otherwise unused.
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import numpy as np
from collections import defaultdict

def draw_plot():
    with open('threshold_cutoff_early_arrival_experiment.csv', 'r') as data_file:
        reader = csv.reader(data_file)
        number_threshold, arrival_time_threshold = set(), set()
        reliability = defaultdict(dict)
        for line in reader:
            number_threshold.add(int(line[1]))
            arrival_time_threshold.add(float(line[0]))
            reliability[int(line[1])][float(line[0])] = float(line[2])

        _number_threshold = np.array(sorted(list(number_threshold)))
        _arrival_time_threshold = np.array(sorted(list(arrival_time_threshold)))
        # meshgrid function See: https://www.geeksforgeeks.org/numpy-meshgrid-function/
        _number_threshold_plot, _arrival_time_threshold_plot = np.meshgrid(_number_threshold, _arrival_time_threshold)

        _reliability_plot = np.empty([len(_number_threshold_plot), len(_number_threshold_plot[0])], dtype='float')
        for row in range(len(_reliability_plot)):
            for col in range(len(_reliability_plot[0])):
                _reliability_plot[row][col] = reliability[_number_threshold_plot[row][col]][_arrival_time_threshold_plot[row][col]]

        fig = plt.figure(0)
        ax = fig.gca(projection='3d')
        surf = ax.plot_surface(_number_threshold_plot, _arrival_time_threshold_plot, _reliability_plot, 
            cmap=cm.coolwarm, linewidth=0, antialiased=False)
        ax.set_title("Reliability of Threshold Cutoff Model with Early Arrival Photons")
        ax.set_xlabel("Photon number threshold")
        ax.set_ylabel("Photon arrival time threshold")
        ax.set_zlabel("Reliability")
        fig.colorbar(surf, shrink=0.5, aspect=20)

        # 2D plot: Fixed photon number threshold to a number (Ex. 12)
        arrival_time_2d = sorted(list(arrival_time_threshold))
        reliability_2d = [reliability[12][arrival_time] for arrival_time in arrival_time_2d]

        fig = plt.figure(1)
        plt.plot(arrival_time_2d, reliability_2d)
        plt.show()


if __name__ == '__main__':
    draw_plot()
