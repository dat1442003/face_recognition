import numpy as np

class KNN:
    def __init__(self, n_neighbors=5):
        self.n_neighbors = n_neighbors

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def euclidean_distance(self, x1, x2):
        return np.sqrt(np.sum((x1 - x2) ** 2))

    def find_neighbors(self, x):
        distances = []
        for i in range(len(self.X_train)):
            dist = self.euclidean_distance(x, self.X_train[i])
            distances.append((dist, self.y_train[i]))
        distances.sort(key=lambda x: x[0])
        neighbors = [distances[i][1] for i in range(self.n_neighbors)]
        return neighbors

    def predict(self, X):
        predictions = []
        for x in X:
            neighbors = self.find_neighbors(x)
            counts = np.bincount(neighbors)
            prediction = np.argmax(counts)
            predictions.append(prediction)
        return predictions
    def predict_proba(self, X):
        probabilities = []
        for x in X:
            neighbors = self.find_neighbors(x)
            counts = np.bincount(neighbors)
            probabilities.append(counts / self.n_neighbors)
        return np.array(probabilities)