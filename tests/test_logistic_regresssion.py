from algorithms import logistic_regression
from sklearn import datasets
from sklearn import linear_model

iris = datasets.load_iris()

X = iris.data
y = iris.target

X = X[y != 0]
y = y[y!=0]
y = y-1

print(y)
logistic_regression.model(X, y)



#testing using sklearn
model = linear_model.LogisticRegression()
model.fit(X,y)
print(model.score(X,y))
print(model.predict(X))


