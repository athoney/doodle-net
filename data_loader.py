from sklearn.utils import shuffle
import numpy as np

def get_data(file_paths):
    X_train = []
    y_train = []
    X_valid = []
    y_valid = []
    X_test = []
    y_test = []

    for i, path in enumerate(file_paths):
        data = np.load(path)
        data = shuffle(data, random_state=10)
        
        X_train.append(data[:10000])
        y_train.extend([i]*10000)

        X_valid.append(data[10000:11000])
        y_valid.extend([i]*1000)

        X_test.append(data[11000:12000])
        y_test.extend([i]*1000)

    X_train = np.concatenate(X_train).astype('float32') / 255
    X_valid = np.concatenate(X_valid).astype('float32') / 255
    X_test = np.concatenate(X_test).astype('float32') / 255
    y_train = np.array(y_train)
    y_valid = np.array(y_valid)
    y_test = np.array(y_test)

    X_train = X_train.reshape(-1, 28, 28)
    X_valid = X_valid.reshape(-1, 28, 28)
    X_test = X_test.reshape(-1, 28, 28)

    X_train = X_train[:, np.newaxis, :, :]
    X_valid = X_valid[:, np.newaxis, :, :]
    X_test = X_test[:, np.newaxis, :, :]

    return X_train, y_train, X_valid, y_valid, X_test, y_test