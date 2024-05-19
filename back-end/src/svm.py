import numpy as np

class SVM:
    def __init__(self, C=1.0):
        self.C = C
        self.weights = None
        self.bias = None

    def train(self, X, y):
        # X: mảng numpy có kích thước (n_samples, n_features)
        # y: mảng numpy chứa nhãn tương ứng của mỗi dữ liệu trong X

        n_samples, n_features = X.shape
        n_classes = np.max(y) + 1

        self.weights = np.zeros((n_classes, n_features))
        self.bias = np.zeros(n_classes)

        for cls in range(n_classes):
            # Tạo mảng nhãn nhị phân cho lớp cls
            binary_labels = np.where(y == cls, 1, -1)

            # Huấn luyện bộ phân loại SVM cho lớp cls
            weights, bias = self._train_binary_classifier(X, binary_labels)

            # Lưu trữ trọng số và bias cho lớp cls
            self.weights[cls] = weights
            self.bias[cls] = bias

    def _train_binary_classifier(self, X, y):
        n_samples, n_features = X.shape

        # Tính ma trận Gram (dot product) của X
        gram_matrix = np.dot(X, X.T)

        # Khởi tạo các tham số tối ưu
        alpha = np.zeros(n_samples)
        bias = 0.0
        learning_rate = 1e-3
        num_iterations = 1000

        # Huấn luyện bộ phân loại nhị phân
        for _ in range(num_iterations):
            for i in range(n_samples):
                # Tính đầu ra của bộ phân loại hiện tại
                prediction = np.dot(gram_matrix[i], alpha * y) + bias

                # Kiểm tra điều kiện tối ưu
                if y[i] * prediction < 1:
                    # Cập nhật các tham số
                    alpha[i] += learning_rate
                    bias += learning_rate * y[i]

        # Tính trọng số dự đoán
        weights = np.dot(alpha * y, X)

        return weights, bias

    def predict(self, X):
        # X: mảng numpy có kích thước (n_samples, n_features)

        n_samples = X.shape[0]
        scores = np.dot(X, self.weights.T) + self.bias.reshape(1, -1)
        predicted_labels = np.argmax(scores, axis=1)

        return predicted_labels
    def predict_proba(self, X):
        # X: mảng numpy có kích thước (n_samples, n_features)

        n_samples = X.shape[0]
        scores = np.dot(X, self.weights.T) + self.bias.reshape(1, -1)
        probabilities = self._softmax(scores)

        return probabilities

    def _softmax(self, X):
        # X: ma trận numpy

        exp_scores = np.exp(X)
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

        return probs