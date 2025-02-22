import numpy as np

class NearestNeighbor:
    def __init__(self):
        pass

    def train(self,X,y):
        self.Xtr=X
        self.ytr=y

    def predict(self,X):
        number_of_predictions = X.shape[0]
        y_predictions = np.zeros(number_of_predictions)
        for i in range(number_of_predictions):
            distances = np.sum(np.abs(self.Xtr-X[i,:]),axis=1)
            min_index = np.argmin(distances)
            y_predictions[i] = self.ytr[min_index]
        return y_predictions

x_train = np.array([[1,2],[3,4],[5,6],[7,8]])
y_train = np.array([1,2,3,4])
x_test = np.array([[1,2],[3,4],[5,6],[7,8]])

nn = NearestNeighbor()
nn.train(x_train,y_train)
y_pred = nn.predict(x_test)
print(y_pred)