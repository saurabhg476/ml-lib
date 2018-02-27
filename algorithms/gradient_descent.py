import numpy as np
import matplotlib.pyplot as plt

def optimize_parameters(X,y,cost_function,initial_parameters,num_iter,learning_rate):

    parameters = initial_parameters
    costs = []

    for i in range(num_iter):
        cost,grads = cost_function(X,y,parameters)
        costs.append(cost)
        parameters = parameters - learning_rate * grads


    return cost,parameters






