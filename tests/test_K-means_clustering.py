import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets.samples_generator import make_blobs
from algorithms import K_means_clustering as k_means


X,y = make_blobs(n_samples=100,centers=3,n_features=2,random_state=0)
print(X.shape)
print(y.shape)

plt.figure(1)
plt.title("Original")
plt.scatter(X[:,0], X[:,1],c=y, alpha=0.5)
plt.title("original")


clf = k_means.KMeans()
clf.model(X,3,10)
print(clf.centroids)
print(clf.belongs)


plt.figure(2)
plt.title("After applying clustering")
plt.scatter(X[:,0],X[:,1],c=np.squeeze(clf.belongs),alpha=0.5)


plt.figure(2)
plt.scatter(clf.centroids[:,0],clf.centroids[:,1],alpha=0.5,marker='+')
plt.show()