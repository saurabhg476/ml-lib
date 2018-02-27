import numpy as np

def model(R,Y,X):
    '''

    :param R: r(i,j) == 1 if user j has rated movie i otherwise 0
    :param Y: y(i,j) is the rating of movie i by user j if r(i,j) =1
    :param X: X(i,:) is feature vector for movie i
    :return:
    '''

    pass


def initialize(num_users,num_features):
    initial_theta = np.zeros(num_users,num_features)
    return initial_theta

def optimize(R,Y,X,initial_theta,num_iter,learning_rate):
    '''
    :param R:
    :param Y:
    :param X: shape (num_movies,num_features)
    :param initial_theta: shape (num_users,num_features)
    :param num_iter:
    :param learning_rate:
    :return:
    '''
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


    def model(self,R,Y,X):
        self.num_users = Y.shape[1]
        self.num_features = X.shape[1]

        #adding intercept feature
        X = np.c_[np.ones(self.num_users),X]
        initial_theta = initialize(self.num_users,self.num_features)

        theta = optimize(R,Y,X,initial_theta,100,0.01)
        return theta

