import numpy as np

def initialize(num_users,num_features):
    initial_theta = np.zeros((num_users,num_features))
    return initial_theta

def optimize(R,Y,X,initial_theta,num_iter,learning_rate):
    '''
    :param R:
    :param Y:
    :param X: shape (num_movies,num_features)
    :param initial_theta: shape (num_users,num_features)
    :param num_iter:
    :param learning_rate:
    :return: optimized parameters
    '''
    #TODO: insert regularization
    num_users = R.shape[1]
    theta = initial_theta

    for j in range(num_users):
        for iterations in range(num_iter):
            predictions = np.dot(X,theta[j,:])
            grad = np.dot((predictions - Y[:,j])*R[:,j],X)
            theta[j,:] = theta[j,:] - learning_rate*grad

    return theta


class ContentBasedRecommendation:

    def __init__(self):
        self.num_users = None
        self.num_features = None
        self.num_movies = None

    def model(self,R,Y,X,max_iter,learning_rate):
        self.num_users = Y.shape[1]
        self.num_movies = X.shape[0]

        #adding intercept feature
        X = np.c_[np.ones(self.num_movies),X]
        self.num_features = X.shape[1]
        initial_theta = initialize(self.num_users,self.num_features)

        theta = optimize(R,Y,X,initial_theta,max_iter,learning_rate)
        return theta

