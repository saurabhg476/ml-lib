from algorithms import outlier_detection
import numpy as np
import matplotlib.pyplot as plt

x1 = np.random.normal(2,1,300)
x2 = np.random.normal(3,2,300)

# plt.figure(1)
# plt.scatter(x1,x2,marker='.')
# plt.axis([-5,7.5,-5,10])
# plt.title("Generated Data")
# plt.show()

#generating X
X = np.array([x1,x2]).T
#print(X.shape)


clf = outlier_detection.OutlierDetection()
clf.model_fit(X)
prob = clf.predict_proba(X)

print(prob)



