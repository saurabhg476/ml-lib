import numpy as np
from algorithms import collaborative_filtering


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

print(R.shape)
print(Y.shape)

model = collaborative_filtering.CollaborativeFiltering(num_features=3)
model.train(Y,R)

print(model.X.shape)
print(model.theta.shape)

print(np.dot(model.X,model.theta.T))