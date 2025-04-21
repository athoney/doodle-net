---
sidebar_position: 3
---

# Prepare the Model

🧠 Now that we have our data ready, it’s time to define the model. We’ll be using a simple Convolutional Neural Network (CNN) architecture to classify doodles from the Quick, Draw! dataset.

CNNs are particularly well-suited for image classification tasks because they are able to learn spatial hierarchies in images, starting from edges and textures in early layers, all the way to more abstract shapes in deeper layers.

---

### 🏗️ Define the Model Architecture

Here’s the architecture we’ll use:

```python
class QuickDrawCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(QuickDrawCNN, self).__init__()

        # Convolutional layer 1: 1 input channel (grayscale), 16 output channels, 3x3 kernel
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1)  # keeps size = 28x28
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # halves to 14x14

        # Convolutional layer 2: 16 input, 32 output
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)  # 14x14 → 14x14
        self.pool2 = nn.MaxPool2d(2, 2)  # 14x14 → 7x7

        # Fully connected layers
        self.fc1 = nn.Linear(32 * 7 * 7, 128)  # Flattened input → dense
        self.fc2 = nn.Linear(128, num_classes)  # Output layer

        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))  # (1,28,28) → (16,14,14)
        x = self.pool2(F.relu(self.conv2(x)))  # (16,14,14) → (32,7,7)
        x = x.view(-1, 32 * 7 * 7)  # Flatten for dense layer
        x = self.dropout(F.relu(self.fc1(x)))  # Dense with dropout
        x = self.fc2(x)  # No activation; will use CrossEntropyLoss
        return x
```

### 🧪 What This Model Does
- **Two convolutional** layers: Learn spatial features from the input image.
- **Max pooling**: Downsamples the image and reduces computation.
- **Dropout layer**: Helps prevent overfitting.
- **Two fully** connected layers: Learn non-linear combinations of the extracted features.
- **Output layer**: Produces class scores for each of the 10 categories.

### ⚙️ Set Up the Model
Once the model is defined, we configure the device, loss function, and optimizer:
```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = QuickDrawCNN(num_classes=10).to(device) # Change num_classes to the number of classes you have
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
```

🎉 Now for the real work, training!