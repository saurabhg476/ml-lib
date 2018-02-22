import numpy as np
import random
import matplotlib.pyplot as plt


def initialize(X,k):

    m = X.shape[0]
    random_indexes = random.sample(range(m), k)
    centroids = X[random_indexes]
    return centroids


def dist(a,b):

    return np.linalg.norm(a-b,ord=2)


def model(X, k,iterations=100):

    m = X.shape[0]
    k = k

    if k > m:
        raise ValueError("k should be less than or equal to number of points in training set")

    centroids = initialize(X,k)

    # array holding indexes of centroid with minimum distance from the point
    c = np.zeros((m, 1))

    for p in range(iterations):
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


    return centroids,c

class KMeans:

    def __init__(self):
        self.k = 0
        self.centroids = None
        self.iterations = 0
        self.belongs = None

    def model(self, X, k, iterations=100):
        self.k = k
        self.iterations = iterations

        centroids , c  = model(X,k)

        self.centroids = centroids
        self.belongs = c












