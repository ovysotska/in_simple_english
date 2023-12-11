import numpy as np
import scipy.stats
import matplotlib.pyplot as plt


def NormDensity(mu, sigma, value):
    return scipy.stats.norm(mu, sigma).pdf(value)


def visualize(ax, sigma, associations, x, mu):

    ax.scatter(x, np.zeros((len(x), 1)), c=associations[:, 0], cmap="viridis")
    ax.scatter(x, np.zeros((len(x), 1)) + 2, c=associations[:, 1], cmap="viridis")
    ax.plot(mu[0], 1, "rd")
    ax.plot(mu[1], 1, "bd")
    ax.plot([mu[0] - sigma[0], mu[0] + sigma[0]], [1, 1], "r-")
    ax.plot([mu[1] - sigma[1], mu[1] + sigma[1]], [1, 1], "b-")


def fitGaussians(X):

    N = len(X)
    classes_num = 2

    associations = np.zeros((N, classes_num)) + 0.5

    phi = np.zeros((classes_num)) + 1.0 / classes_num  # class weight
    mu = np.zeros((classes_num))
    mu[0] = 0
    mu[1] = 1

    sigma2 = np.zeros((classes_num)) + 1

    mu_prev = np.copy(mu)
    fig = plt.figure(1)

    for i in range(100):

        for sample_idx in range(N):
            for k in range(classes_num):
                associations[sample_idx, k] = phi[k] * NormDensity(
                    mu[k], np.sqrt(sigma2[k]), X[sample_idx]
                )

        # normalize
        for sample_id in range(associations.shape[0]):
            associations[sample_id, :] = associations[sample_id, :] / np.sum(
                associations[sample_id, :]
            )

        # maximization step

        phi[0] = np.sum(associations[:, 0]) / N
        phi[1] = np.sum(associations[:, 1]) / N

        mu[0] = np.dot(associations[:, 0], X) / np.sum(associations[:, 0])
        mu[1] = np.dot(associations[:, 1], X) / np.sum(associations[:, 1])

        sigma2[0] = np.dot(associations[:, 0], (X - mu[0]) * (X - mu[0])) / np.sum(
            associations[:, 0]
        )
        sigma2[1] = np.dot(associations[:, 1], (X - mu[1]) * (X - mu[1])) / np.sum(
            associations[:, 1]
        )
        # print("Mu", mu, "mu_prev", mu_prev, "iteration", i)
        if np.linalg.norm(mu_prev - mu) < 1e-12:
            print(
                "Change between mu prev and mu is too small",
                np.linalg.norm(mu_prev - mu),
            )
            break

        mu_prev = np.copy(mu)

    #     if i % 10 == 0:

    #         # print("sample associations to class 0", associations[:, 0].transpose())
    #         # print("sample associations to class 1", associations[:, 1].transpose())

    #         print("class importance weights", phi)
    #         print("class, means", mu)
    #         print("class variance2", sigma2)
    #         print("======================")

    #         sigma = np.sqrt(sigma2)
    #         visualize(fig.gca(), sigma, associations, X, mu)
    #         plt.show()

    print("======================")
    print("======================")
    print("Final result at iteration", i)
    # # print("sample associations to class 0", associations[:, 0].transpose())
    # # print("sample associations to class 1", associations[:, 1].transpose())

    print("class importance weights", phi)
    print("class, means", mu)
    print("class variance2", np.sqrt(sigma2))

    # fig2 = plt.figure(2)
    # visualize(fig2.gca(), np.sqrt(sigma2), associations, X, mu)
    # fig2.gca().set_title("Final result")
    # plt.show()
    return mu, np.sqrt(sigma2)


if __name__ == "__main__":
    x = [0.1, 0.12, 0.24, 0.3, 0.21, 0.23, 0.6, 0.62, 0.7]
    fitGaussians(x)


# Make sure this runs faster and has proper visualization for understanding what is hapening
