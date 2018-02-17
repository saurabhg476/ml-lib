import numpy as np
import matplotlib.pyplot as plt


def optimize_parameters(X,y,parameters,gradient_function,learning_rate=0.01, iterations=100, plot=False):
    cost = []
    for i in range(0, iterations):
        dparameters = gradient_function(X,y,parameters)
        parameters = parameters - learning_rate * dparameters

    return parameters


