import numpy as np
import histograms as dp
import matplotlib.pyplot as plt


def numpy_histogram(array_input):
    # We can find the distribution of ages, as determined by
    # ten equally-spaced bins calculated by histogram.
    hist, bins = np.histogram(a=array_input, bins=10)
    hist = hist / hist.sum()
    return hist, bins


def dp_histogram(array_input, bins_range, random_state=np.random.randint(0, 1234), epsilon=1.0):
    dp_hist, dp_bins = dp.histogram(array_input, epsilon=epsilon, range=bins_range, random_state=random_state)
    dp_hist = dp_hist / dp_hist.sum()
    return dp_hist, dp_bins


if __name__ == '__main__':
    # Read in the list of ages in the Adult UCI dataset (the first column).
    ages_adult = np.loadtxt("https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data",
                            usecols=0, delimiter=",")
    print(ages_adult, "len:", len(ages_adult))

    # Using numpy's native histogram function.
    numpy_hist, numpy_bins = numpy_histogram(ages_adult)

    # Using matplotlib.pyplot, to plot a barchart of the histogram distribution.
    # plt.bar(numpy_bins[:-1], numpy_hist, width=(numpy_bins[1] - numpy_bins[0]) * 0.9)
    # plt.show()

    # Differentially private histograms.
    # For this example, we use the following settings:
    # epsilon is 1.0,
    # bins_range is (0, 100), we know that there are people in the dataset aged from 0 to 100.
    # The range can be specified independently of the data (i.e., using domain knowledge).
    # As of 2019, less than 0.005% of the world's population is aged over 100, so this is an appropriate simplification.
    # Values in the dataset above 100 will be excluded from calculations.
    dp_hist1, dp_bins1 = dp_histogram(ages_adult, bins_range=(0, 100), epsilon=1.0, random_state=1234)
    print(dp_bins1[0], dp_bins1[-1])

    # plt.bar(dp_bins1[:-1], dp_hist1, width=(dp_bins1[1] - dp_bins1[0]) * 0.9)
    # plt.show()

    # Effect of epsilon: If we decrease epsilon (i.e. increase the privacy guarantee), the error will increase.
    dp_hist2, dp_bins2 = dp_histogram(ages_adult, bins_range=(0, 100), epsilon=0.001, random_state=1234)
    print("Total histogram error: %f" % np.abs(dp_hist2 - dp_hist1).sum())

    plt.bar(dp_bins2[:-1], dp_hist2, width=(dp_bins2[1] - dp_bins2[0]) * 0.9)
    plt.show()

    # By setting epsilon = float("inf").
    # This should give the exact same result as running numpy.histogram.
    dp_hist3, dp_bins3 = dp_histogram(ages_adult, bins_range=(17, 90), epsilon=float("inf"))
    print("Total histogram error: %f" % np.abs(dp_hist3 - numpy_hist).sum())
