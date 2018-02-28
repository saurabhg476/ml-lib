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

X = np.array([
    [1,0],
    [0.9,0],
    [0.9,0.1],
    [0.1,0.9],
    [0,1]
])

clf = content_based_recommendation.ContentBasedRecommendation()
theta  = clf.model(R,Y,X,200,0.01)

X = np.c_[np.ones(X.shape[0]), X]
print(np.dot(X,theta.T))
