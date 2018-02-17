import numpy as np
from utility_functions import sigmoid
from gradient_descent import optimize_parameters

def model(X,y):

    initial_parameters = initialize_parameters(X.shape[1])
    parameters = optimize_parameters(X,y,initial_parameters,compute_cost())
    print(predict(X,parameters,0.5))

def initialize_parameters(n):
    '''
    Initializes parameters for logistic regression
    :param n: number of features
    :return: parameters: dictionary containing w and b
    '''
    w = np.zeros((1,n))
    b = 0
    parameters = {
        "w":w,
        "b":b
    }

    return parameters

def predict_proba(X,parameters):
    '''
    Predicts the output for given input and paramters
    :param X: shape: (m,n)
    :param parameters: shape: (n,1)
    :return: yhat: the prediction probabilities
    '''
    m = X.shape[0]
    w = parameters["w"]
    b = parameters["b"]

    Z = np.dot(X,w) + b
    yhat = sigmoid(Z)
    return yhat.reshape(m,1)

def predict(X,parameters,threshold):
    return predict_proba(X,parameters) > threshold


def compute_cost(X,y,parameters):
    '''
    Computes the cost and gradients for logistic regression
    :param yhat: shape: (m,1)
    :param y: shape: (m,1)
    :return: cost,gradients
    '''
    m = y.shape[0]

    yhat = predict_proba(X, parameters)
    cost = -(1/m)*np.sum(y*np.log(yhat) + (1-y)*np.log(1-yhat))

    dw = np.dot(X.T, (yhat - y))
    db = np.sum(yhat-y)
    return cost,dparameters

def compute_gradients(X,y,parameters):
    '''
    :param X: shape(m,n)
    :param y: shape(m,1)
    :param yhat: shape(m,1)
    :param parameters: shape(1,n)
    :return: dparameters: shape(1,n)
    '''
    yhat = predict_proba(X,parameters)

    return dparameters