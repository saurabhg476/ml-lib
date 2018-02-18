import numpy as np


def model(X, variance):
    '''

    :param variance(float): amount of variance in data to be kept after applying PCA
    :param X: shape (m,n)
    :return:
    '''

    m = X.shape[0]
    n = X.shape[1]

    #mean normalization
    mean = X.mean(axis=0,keepdims=True)
    range = X.ptp(axis=0).reshape(1,n)

    X = (X - mean)/range

    sigma = (1/m) *np.dot(X.T,X)
    (u,s,v) = np.linalg.svd(sigma)

    #selecting k i.e. number of principal components
    total_sum = np.sum(s)
    running_sum = 0.0

    for i, value in enumerate(s,start=1):
        running_sum += value
        if running_sum/total_sum >= variance:
            break

    k = i
    u_reduce = u[:,:k]

    Z = np.dot(X,u_reduce)


    ###printing k
    print(k)
    ###printing approximations
    X_approx = np.dot(Z,u_reduce.T)
    print(X_approx*range + mean)

    return Z, k


a = np.array([[1,2,3],[4,5,6],[7,8,9]])
model(a, 0.90)







