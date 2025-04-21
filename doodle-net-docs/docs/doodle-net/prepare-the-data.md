---
sidebar_position: 2
---

# Prepare the Data

🗂️ Now that we’ve loaded and visualized the doodle data, it’s time to prepare it for training in PyTorch.

### 🔄 Step 1: Convert to Tensors

PyTorch models work with `torch.Tensor` objects, not NumPy arrays. So the first thing we’ll do is convert our training, validation, and test sets:

```python
# Convert numpy arrays to torch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)
X_valid_tensor = torch.tensor(X_valid, dtype=torch.float32)
y_valid_tensor = torch.tensor(y_valid, dtype=torch.long)

print(X_test.shape)
print(y_test.shape)
```
We use:
- float32 for image tensors so that they can be processed by the neural network
- long for labels because PyTorch’s classification loss functions (like CrossEntropyLoss) expect class labels in that format

### 📦 Step 2: Wrap in TensorDatasets

PyTorch provides a handy class called TensorDataset that allows us to treat our tensors like a dataset. This is useful for batching and shuffling during training:
```python
# Create PyTorch datasets
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
valid_dataset = TensorDataset(X_valid_tensor, y_valid_tensor)
```

### 🚚 Step 3: Set Up DataLoaders
Next, we create DataLoader objects that handle the batching and shuffling automatically. These will be used in our training loop:

```python
# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64)
valid_loader = DataLoader(valid_dataset, batch_size=64)
``` 
- batch_size=64 means we’ll train on 64 samples at a time
- shuffle=True randomizes the training data order each epoch for better generalization

✅ That’s it! Your data is now ready to be used in a PyTorch training loop. Let’s move on to defining the model. Here is all the code in one place:

<details>
  <summary>Show code</summary>
<pre><code class="language-python">

```python

# Convert numpy arrays to torch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)
X_valid_tensor = torch.tensor(X_valid, dtype=torch.float32)
y_valid_tensor = torch.tensor(y_valid, dtype=torch.long)

print(X_test.shape)
print(y_test.shape)

# Create PyTorch datasets
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
valid_dataset = TensorDataset(X_valid_tensor, y_valid_tensor)

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64)
valid_loader = DataLoader(valid_dataset, batch_size=64)
```
</code></pre>
</details>