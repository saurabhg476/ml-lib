import numpy as np
import random
import matplotlib.pyplot as plt


def model(X,k):

    # number of examples
    m = X.shape[0]

    # number of features
    n = X.shape[1]

    if k > m:
        raise ValueError("k should be less than or equal to number of points in training set")

    centroids = initialize(X, k)

    # array holding indexes of centroid with minimum distance from the point
    c = np.zeros((m, 1))

    for p in range(10):
        # cluster assignment step
        for i,example in enumerate(X):
            min_dist = np.inf
            for j,centroid in enumerate(centroids):
                if dist(example,centroid) < min_dist:
                    min_dist = dist(example,centroid)
                    c[i] = j

        # move centroid step
        for i in range(k):
            centroids[i] = X[np.squeeze(c) == i,:].mean(axis = 0,keepdims=True)


    plt.figure(2)
    plt.title("After clustering:")
    plt.scatter(X[:,0], X[:,1], alpha=0.5,c=np.squeeze(c))
    plt.show()

    return centroids



def dist(a,b):

    return np.linalg.norm(a-b,ord=2)


def initialize(X, k):

    m = X.shape[0]
    random_indexes = random.sample(range(m),k)
    centroids = X[random_indexes]
    return centroids


