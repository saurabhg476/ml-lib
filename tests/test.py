# class for general purpose testing

from sklearn.neural_network import MLPClassifier

model = MLPClassifier(solver='lbfgs',alpha= 1e-5,hidden_layer_sizes=(4,4,4),  )