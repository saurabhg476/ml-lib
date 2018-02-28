import numpy as np


def initialize(num_movies,num_users,num_features):
    '''
    X : shape: (num_movies,num_features)
    :return: theta: shape: (num_users,num_features)
    '''
    initial_theta = np.random.rand(num_users,num_features)
    initial_X = np.random.rand(num_movies,num_features)
    return initial_X,initial_theta

def optimize_parameters(initial_X,initial_theta,Y,
                        R,num_iter,learning_rate,lambd):

    theta = initial_theta
    #print("theta=",theta.shape)
    X = initial_X
    #print("X=",X.shape)

    for iter in range(num_iter):

        predictions = np.dot(X,theta.T)
        #print("predictions=",predictions.shape)
        grad = np.dot((predictions - Y)*R,theta)
        X = X*(1-lambd) - learning_rate*grad

        predictions = np.dot(X,theta.T)
        grad = np.dot(((predictions - Y)*R).T, X)
        theta = theta*(1-lambd) - learning_rate * grad

    return X,theta


class CollaborativeFiltering:

    def __init__(self,num_features=10,num_iter=200,learning_rate=0.01,lambd=0):
        self.num_users = None
        self.num_movies = None
        self.num_features = num_features
        self.num_iter = num_iter
        self.learning_rate = learning_rate
        self.lambd = lambd
        self.X = None
        self.theta = None
        pass

    def train(self,Y,R):
        self.num_users = Y.shape[1]
        self.num_movies = Y.shape[0]

        initial_X, initial_theta = initialize(self.num_movies,self.num_users,self.num_features)

        X,theta = optimize_parameters(initial_X,initial_theta,Y,R,self.num_iter,
                                      self.learning_rate,self.lambd)
        self.X = X
        self.theta = theta



