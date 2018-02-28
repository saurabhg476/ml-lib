import numpy as np
from utility_functions import sigmoid


def model(X, y):
    y = y.reshape(y.shape[0],1)
    initial_parameters = initialize_parameters(X.shape[1])
    parameters = optimize_parameters(X, y, initial_parameters, compute_cost)
    predictions = predict(X, parameters,0.5)

    return predictions,parameters

def optimize_parameters(X, y, parameters, cost_function, learning_rate=0.08, iterations=500, plot=False):

    for i in range(0, iterations):
        cost, dparameters = cost_function(X, y, parameters)
        print("iteration =",i,"cost =",cost)

        w = parameters["w"]
        b = parameters["b"]
        dw = dparameters["dw"]
        db = dparameters["db"]

        w = w - learning_rate*dw
        b = b - learning_rate*db

        parameters = {
            "w":w,
            "b":b
        }

    return parameters

def initialize_parameters(n):
    '''
    Initializes parameters for logistic regression
    :param n: number of features
    :return: parameters: dictionary containing w and b
    '''
    w = np.zeros((1, n))
    b = 0

    parameters = {
        "w":w,
        "b":b
    }

    return parameters

def predict_proba(X, parameters):
    '''
    Predicts the output for given input and paramters
    :param X: shape: (m,n)
    :param parameters: shape: (n,1)
    :return: yhat: the prediction probabilities
    '''
    m = X.shape[0]
    w = parameters["w"]
    b = parameters["b"]

    Z = np.dot(X, w.T) + b
    yhat = sigmoid(Z)
    return yhat.reshape(m, 1)

def predict(X, parameters, threshold):
    return (predict_proba(X, parameters) > threshold).astype(int)


def compute_cost(X,y,parameters):
    '''
    Computes the cost and gradients for logistic regression
    :param yhat: shape: (m,1)
    :param y: shape: (m,1)
    :return: cost,gradients
    '''
    m = y.shape[0]
    n = X.shape[1]

    yhat = predict_proba(X, parameters)
    cost = -(1/m)*np.sum(y*np.log(yhat) + (1-y)*np.log(1-yhat))

    dw = (1/m)*np.dot((yhat - y).T,X)
    dw = dw.reshape(1,n)
    db = (1/m)*np.sum(yhat-y)

    dparameters = {
        "dw":dw,
        "db":db
    }
    return cost, dparameters


class LogisticRegression:

    def __init__(self):
        self.parameters = None
        pass

    def model(self):
        pass

