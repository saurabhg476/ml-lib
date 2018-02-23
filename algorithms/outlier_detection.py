import numpy as np

def prob(X,mean,var):
    probs =  (1/(np.sqrt(2*np.pi)*var)) * np.exp(-np.power(X-mean,2)/2*var*var)
    return probs.prod(axis=1)

def model_fit(X):

    mean = X.mean(axis = 0)
    variance = X.var(axis=0)
    return mean,variance


class OutlierDetection:

    def __init__(self):
        self.mean = None
        self.variance = None


    def model_fit(self,X):
        mean,variance = model_fit(X)
        self.mean = mean
        self.variance = variance


    def predict_proba(self,X):
        return prob(X,self.mean,self.variance)




