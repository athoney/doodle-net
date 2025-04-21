---
sidebar_position: 5
---

# Evaluate the Model

📊 Now that training is complete, it’s time to evaluate how well the model performs on the test set — data it has never seen before. This is where we assess **generalization**: does the model just memorize the training data, or can it correctly classify new doodles?

---

### 🧪 Final Evaluation on Test Data

We start by collecting all predictions from the model on the test set.

```python
all_preds = []
all_labels = []

model.eval()
with torch.no_grad():
    for X_batch, y_batch in test_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        outputs = model(X_batch)
        preds = outputs.argmax(dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(y_batch.cpu().numpy())

# Convert to numpy arrays
all_preds = np.array(all_preds)
all_labels = np.array(all_labels)

print(classification_report(all_labels, all_preds, target_names=class_names, digits=4))
```
This will print precision, recall, and F1-score for each class. It's a great way to see which classes are performing well — and which might need more data or model tuning.

### 🔍 Confusion Matrix
To get a better understanding of what mistakes the model is making, let’s visualize the confusion matrix:

```python
cm = confusion_matrix(all_labels, all_preds)

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()
```
![Confusion Matrix](/img/confusion_matrix.png)
Each row is the actual class, and each column is the predicted class. Diagonal values represent correct predictions, off-diagonal values are misclassifications.

### 📈 Learning Curves
If you saved the training history, you can plot how the model improved over time:
```python
epochs_range = range(len(train_losses))

plt.plot(epochs_range, train_losses, label="Training Loss")
plt.plot(epochs_range, val_losses, label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training & Validation Loss Over Time")
plt.legend()
plt.show()

plt.plot(epochs_range, val_accuracies, label="Validation Accuracy", color='green')
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Validation Accuracy Over Time")
plt.legend()
plt.show()
```
![Training & Validation Loss](/img/train_validate.png)
![Validation Accuracy](/img/validation_acc.png)

Use this to check for:
- **Overfitting** (if training loss goes down but validation loss goes up)
- **Underfitting** (if both losses remain high)

### 🏆 Top-k Accuracy (k=3)
Top-1 accuracy is great, but sometimes it’s useful to see if the correct label was among the top-k predictions. Let’s calculate top-3 accuracy:

```python
def compute_top_k_accuracy(model, loader, k=3):
    correct = 0
    total = 0

    model.eval()
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            topk_preds = torch.topk(outputs, k, dim=1).indices
            correct += sum([y_batch[i] in topk_preds[i] for i in range(len(y_batch))])
            total += y_batch.size(0)

    return correct / total

top3_acc = compute_top_k_accuracy(model, test_loader, k=3)
print(f"Top-3 Accuracy: {top3_acc:.2%}")
``` 
This tells you how often the correct label was in the model’s top 3 guesses — useful for multi-class tasks like doodle recognition.