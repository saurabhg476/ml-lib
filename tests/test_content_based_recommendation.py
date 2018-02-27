from algorithms import content_based_recommendation
import numpy as np

R = np.array([
    [1,1,1,1],
    [1,0,0,1],
    [0,1,1,0],
    [1,1,1,1],
    [1,1,1,0]])

Y = np.array([
    [5,5,1,1],
    [5,0,0,1],
    [0,4,1,0],
    [1,1,5,4],
    [1,1,5,0]
])

X = np.array()