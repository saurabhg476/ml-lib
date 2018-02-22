import matplotlib.pyplot as plt
from sklearn.datasets.samples_generator import make_blobs
from algorithms import K_means_clustering as k_means
import sklearn

X,y = make_blobs(n_samples=100,centers=3,n_features=2,random_state=0)
print(X.shape)
print(y.shape)

plt.figure(1)
plt.scatter(X[:,0], X[:,1],c=y, alpha=0.5)
plt.title("original")


centroids = k_means.model(X,3)
print(sklearn.__version__)

# plt.figure(3)
# plt.scatter(centroids[:,0],centroids[:,1],c='g',alpha=0.9,marker='+')
# plt.title("centroids")