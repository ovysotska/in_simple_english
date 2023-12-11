import numpy as np
import scipy.stats
import matplotlib.pyplot as plt

from matplotlib.colors import ListedColormap

import time


def visualize(ax, sigma, associations, x, mu):
    num_colors = 256
    color_class_0 = "steelblue"
    color_class_1 = "orange"
    # Create a list of colors, making sure to include the midpoint color
    colors = [
        color_class_1,
        "white",
        color_class_0,
    ]

    # Create the colormap using ListedColormap
    custom_cmap = ListedColormap(colors)

    y = np.zeros((len(x), 1))
    ax.scatter(x, y, c=associations[:, 0], cmap=custom_cmap)
    ax.plot(mu[0], 1, "d", color=color_class_0)
    ax.plot(mu[1], 1, "d", color=color_class_1)
    ax.plot([mu[0] - sigma[0], mu[0] + sigma[0]], [1, 1], "-", color=color_class_0)
    ax.plot([mu[1] - sigma[1], mu[1] + sigma[1]], [1, 1], "-", color=color_class_1)

    gaussian_0 = scipy.stats.norm(mu[0], sigma[0])
    gaussian_1 = scipy.stats.norm(mu[1], sigma[1])
    class0_x = np.linspace(gaussian_0.ppf(0.01), gaussian_0.ppf(0.99), 100)
    class1_x = np.linspace(gaussian_1.ppf(0.01), gaussian_1.ppf(0.99), 100)

    ax.plot(
        class0_x,
        gaussian_0.pdf(class0_x),
        "-",
        color=color_class_0,
        lw=5,
        alpha=0.6,
        label="norm pdf",
    )
    ax.plot(
        class1_x,
        gaussian_1.pdf(class1_x),
        "-",
        color=color_class_1,
        lw=5,
        alpha=0.6,
        label="norm pdf",
    )

    # ax.set_xlim([0, 1])


# Assumes that class 0 means is close to 0 and class 1 means close to 1
def initializeAssociations(x):
    associations = np.zeros((len(x), 2))
    associations[:, 1] = x
    associations[:, 0] = np.ones(len(x)) - x
    return associations


def fitGaussians(X):
    N = len(X)
    classes_num = 2

    associations = np.zeros((N, classes_num)) + 0.5
    # associations = initializeAssociations(x)

    phi = np.zeros((classes_num)) + 1.0 / classes_num  # class weight

    associations_sum = np.sum(associations, axis=0)

    phi = associations_sum / N

    mu = np.zeros((classes_num))
    mu[0] = 0
    mu[1] = 1

    sigma2 = np.zeros((classes_num)) + 1

    mu_prev = np.copy(mu)

    fig3 = plt.figure(3)
    visualize(fig3.gca(), np.sqrt(sigma2), associations, X, mu)
    fig3.gca().set_title("Initial")
    plt.show()

    for i in range(100):
        associations[:, 0] = phi[0] * scipy.stats.norm(mu[0], np.sqrt(sigma2[0])).pdf(X)
        associations[:, 1] = phi[1] * scipy.stats.norm(mu[1], np.sqrt(sigma2[1])).pdf(X)
        # normalize
        associations = associations / associations.sum(axis=1, keepdims=True)

        # maximization step
        # The importance of the class is average of associations probabilities
        associations_sum_0 = np.sum(associations[:, 0])
        associations_sum_1 = np.sum(associations[:, 1])

        phi[0] = associations_sum_0 / N
        phi[1] = associations_sum_1 / N

        mu[0] = np.dot(associations[:, 0], X) / associations_sum_0
        mu[1] = np.dot(associations[:, 1], X) / associations_sum_1

        sigma2[0] = (
            np.dot(associations[:, 0], (X - mu[0]) * (X - mu[0])) / associations_sum_0
        )
        sigma2[1] = (
            np.dot(associations[:, 1], (X - mu[1]) * (X - mu[1])) / associations_sum_1
        )
        if np.linalg.norm(mu_prev - mu) < 1e-12:
            print(
                "Change between mu prev and mu is too small",
                np.linalg.norm(mu_prev - mu),
            )
            break

        mu_prev = np.copy(mu)

        if i % 10 == 0:
            print("iteration", i)
            # print("sample associations to class 0", associations[:, 0].transpose())
            # print("sample associations to class 1", associations[:, 1].transpose())

            print("class importance weights", phi)
            print("class, means", mu)
            print("class variance2", sigma2)
            print("======================")

            sigma = np.sqrt(sigma2)
            fig = plt.figure(1)
            visualize(fig.gca(), sigma, associations, X, mu)
            plt.show()

    print("======================")
    print("======================")
    print("Final result at iteration", i)
    print("sample associations to class 0", associations[:, 0].transpose())
    print("sample associations to class 1", associations[:, 1].transpose())

    print("class importance weights", phi)
    print("class, means", mu)
    print("class variance2", np.sqrt(sigma2))

    fig2 = plt.figure(2)
    visualize(fig2.gca(), np.sqrt(sigma2), associations, X, mu)
    fig2.gca().set_title("Final result")
    plt.show()
    return mu, np.sqrt(sigma2)


if __name__ == "__main__":
    # x = [0.1, 0.12, 0.24, 0.3, 0.21, 0.23, 0.6, 0.62, 0.7]
    x = [0.1, 0.12, 0.24, 0.7]
    start = time.time()
    fitGaussians(x)
    end = time.time()
    print("Elapsed time", end - start, "seconds")
