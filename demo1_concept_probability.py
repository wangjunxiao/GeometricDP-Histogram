import numpy as np
import histograms as dp
import matplotlib.pyplot as plt


def dp_histogram(array_input, bins_range, random_state=np.random.randint(0, 1234), epsilon=1.0):
    dp_hist, dp_bins = dp.histogram(array_input, epsilon=epsilon, range=bins_range, random_state=random_state)
    dp_hist = dp_hist / dp_hist.sum()
    return dp_hist, dp_bins


if __name__ == '__main__':
    # Read in the list of probabilities in the dataset.
    # A distinct sample is denoted by each row, while each column denotes a distinct concept.
    # For this example, 20 samples were run on the client,
    # and the resulting probabilities were associated with 2 distinct concepts.
    probabilities = np.loadtxt("probabilities.data", delimiter=",").T

    print("sample-wise probabilities with concept1:", probabilities[0],
          "len:", len(probabilities[0]))
    print("sample-wise probabilities with concept2:", probabilities[1],
          "len:", len(probabilities[1]))

    # Differentially private histograms.
    # For this example, we use the following settings:
    # distribution of sample-wise probabilities with concept1 is published,
    # epsilon is 1.0,
    # bins_range is (0, 1), we know that there are probabilities in the dataset range from 0 to 1.
    # random_state is for reproducibility.
    dp_hist1, dp_bins1 = dp_histogram(probabilities[0],
                                      bins_range=(0, 1), epsilon=1.0, random_state=1234)

    # Ranges of probabilities, as determined by 10 equally-spaced bins.
    print(dp_bins1)

    # Sample percentages in different ranges.
    print(dp_hist1)

    # Using matplotlib.pyplot, we can plot a barchart of the probability distribution.
    plt.bar(dp_bins1[:-1], dp_hist1, width=(dp_bins1[1] - dp_bins1[0]) * 0.9)
    plt.show()
