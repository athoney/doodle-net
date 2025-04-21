---
sidebar_position: 4
---

# Training Loop

🌀 Before we dive into the training loop code, take a deep breath. There’s a lot going on here, but this is where the magic happens. This is the heart of the deep learning workflow: the part where your model actually learns from data.

---

### 🔁 Training vs. Validation

In each epoch, we go through **two main phases**:

- **Training Phase**: The model adjusts its internal weights using backpropagation.
- **Validation Phase**: The model makes predictions on unseen data to check its generalization performance — but without learning from it.

---

### 🧪 Training Loop Breakdown

Let’s walk through what this function does:

```python
def train_with_validation(model, train_loader, valid_loader, optimizer, criterion, device, epochs=5, save_best=False):
```
We pass in:
- **model**: your CNN
- **train_loader**, **valid_loader**: DataLoaders for training/validation
- **optimizer**: like Adam or SGD
- **criterion**: the loss function (e.g. CrossEntropyLoss)
- **device**: CPU or CUDA
- **epochs**: how many times we loop through the data
- **save_best**: whether to save the model with the best validation accuracy

### 🔄 The Epoch Loop
Each epoch:
- Trains the model on batches of training data
- Evaluates performance on validation data
- (Optionally) Saves the best model so far
- We also use tqdm to visualize progress bars for both training and validation.

```python
for epoch in range(epochs):
  # See full code below
```

### 📉 Loss & Accuracy
We track:
- **Training loss**: How well the model fits the training data
- **Validation loss**: How well the model fits new, unseen data
- **Validation accuracy**: The percent of correct predictions on the validation set
```python
avg_train_loss = total_loss / len(train_loader)
train_losses.append(avg_train_loss)
...
avg_valid_loss = valid_loss / len(valid_loader)
val_accuracy = correct / total
val_losses.append(avg_valid_loss)
val_accuracies.append(val_accuracy)
```
All of this is stored in lists so we can plot the metrics later.

### 💾 Saving the Best Model
```python
if save_best and val_accuracy > best_val_acc:
    best_val_acc = val_accuracy
    torch.save(model.state_dict(), 'best_model.pth')
    print("Best model saved!")
```
Turn on `save_best=True` to save the best-performing model during training.

### ✅ Run the Function
Now that the function is defined, run it like this:
```python
train_losses, val_losses, val_accuracies = train_with_validation(
    model,
    train_loader,
    valid_loader,
    optimizer,
    criterion,
    device,
    epochs=5,
    save_best=False
)
```

📊 In the next section, we’ll use the collected training history to visualize learning curves and evaluate the model’s performance!

<details>
  <summary>Show Full Code</summary>
<pre><code class="language-python">

```python
def train_with_validation(model, train_loader, valid_loader, optimizer, criterion, device, epochs=5, save_best=False):
    model.to(device)
    best_val_acc = 0.0

    # Track history for plotting
    train_losses = []
    val_losses = []
    val_accuracies = []

    for epoch in range(epochs):
        # ----- TRAIN -----
        model.train()
        total_loss = 0
        train_loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]", leave=False)

        for X_batch, y_batch in train_loop:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            output = model(X_batch)
            loss = criterion(output, y_batch)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            train_loop.set_postfix(loss=loss.item())

        avg_train_loss = total_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # ----- VALIDATE -----
        model.eval()
        valid_loss = 0
        correct = 0
        total = 0
        val_loop = tqdm(valid_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]", leave=False)

        with torch.no_grad():
            for X_val, y_val in val_loop:
                X_val, y_val = X_val.to(device), y_val.to(device)
                output = model(X_val)
                loss = criterion(output, y_val)
                valid_loss += loss.item()

                preds = output.argmax(dim=1)
                correct += (preds == y_val).sum().item()
                total += y_val.size(0)

                val_loop.set_postfix(loss=loss.item())

        avg_valid_loss = valid_loss / len(valid_loader)
        val_accuracy = correct / total

        val_losses.append(avg_valid_loss)
        val_accuracies.append(val_accuracy)

        print(f"\nEpoch {epoch+1}/{epochs} | "
              f"Train Loss: {avg_train_loss:.4f} | "
              f"Val Loss: {avg_valid_loss:.4f} | "
              f"Val Acc: {val_accuracy:.2%}")

        # Save best model if requested
        if save_best and val_accuracy > best_val_acc:
            best_val_acc = val_accuracy
            torch.save(model.state_dict(), 'best_model.pth')
            print("model saved!")

    print("✅ Training complete.")
    return train_losses, val_losses, val_accuracies

train_losses, val_losses, val_accuracies = train_with_validation(model, train_loader, valid_loader, optimizer, criterion, device, epochs=5, save_best=False)
```
</code></pre>
</details>
