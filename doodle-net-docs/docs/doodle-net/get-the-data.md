---
sidebar_position: 1
---

# 💾 Get the Data


### Download the Quick, Draw! Dataset
- Head over to [The Quick, Draw! Github](https://github.com/googlecreativelab/quickdraw-dataset), realize you are in the wrong place, and then go to the [Cloud Storage](https://console.cloud.google.com/storage/browser/quickdraw_dataset/full/numpy_bitmap;tab=objects?prefix=&forceOnObjectsSortingFiltering=false) bucket
- The authors provide the files in multiple formats, but the NumPy format is the most convenient for this project. (They look like the 28x28 MNIST digits)
- Download the files you want to use. I recommend starting with a few classes (e.g., `cat`, `dog`, `car`, `house`, etc.) to get a feel for the data. You can always add more later
- The files should download as `.npy` files. You can use the helper function below to load them into a list of NumPy arrays.

```python
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
```

### Add import statements
Before we move on, let's import everything we need for this project:
<details>
  <summary>Show code</summary>
<pre><code class="language-python">
```
python
%pip install tqdm
%pip install torch
%pip install numpy
%pip install scikit-learn
%pip install seaborn
%pip install matplotlib

import numpy as np
from tqdm import tqdm
import torch
from sklearn.utils import shuffle
from data_loader import get_data
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import classification_report
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
```
</code></pre>
</details>

### Load the data
Now let's load the data
```python
# Match these to the names of the files you uploaded (without `.npy`)
class_names = ['bicycle', 'cactus', 'couch', 'diving board', 'flamingo', 'mushroom', 'ocean', 'octopus', 'shark', 'sweater']

file_paths = [f"./data/full_numpy_bitmap_{cls}.npy" for cls in class_names]

X_train, y_train, X_valid, y_valid, X_test, y_test = get_data(file_paths)
```

### Check the data
Finally, let's visualize some of the data
```python
def show_samples(X, y, classes, n=5):
    plt.figure(figsize=(n, n))
    for i in range(n * n):
        plt.subplot(n, n, i + 1)
        plt.imshow(X[i][0], cmap='gray')
        plt.title(classes[y[i]], fontsize=8)
        plt.axis('off')
    plt.tight_layout()
    plt.show()

show_samples(X_train, y_train, class_names, n=5)
```
![Bicycles](/img/bikes.png)