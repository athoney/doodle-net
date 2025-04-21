from sklearn.utils import shuffle
import numpy as np

def get_data(file_paths, n_train=10000, n_test=2000, n_valid=2000):
    X_train, y_train = [], []
    X_valid, y_valid = [], []
    X_test, y_test = [], []

    for label, path in enumerate(file_paths):
        data = shuffle(np.load(path), random_state=10)

        X_train.append(data[:n_train]) # grab only n_train samples
        y_train.extend([label] * n_train)

        X_test.append(data[n_train:n_train + n_test]) # grab next n_test samples for testing
        y_test.extend([label] * n_test)

        X_valid.append(data[n_train + n_test:n_train + n_test + n_valid]) # grab next n_valid for validation
        y_valid.extend([label] * n_valid)

    def preprocess(X):
        return np.concatenate(X).astype('float32') / 255.0

    def reshape(X):
        return X.reshape(-1, 1, 28, 28)

    return (
        reshape(preprocess(X_train)), np.array(y_train),
        reshape(preprocess(X_test)),  np.array(y_test),
        reshape(preprocess(X_valid)), np.array(y_valid)
    )
