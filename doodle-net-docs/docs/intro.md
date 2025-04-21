---
sidebar_position: 1
---

# 🧠 Building a QuickDraw CNN

Welcome to my machine learning learning journey! This site documents my beginner-friendly, slightly chaotic, and often joyful path toward building a Convolutional Neural Network (CNN) to classify doodles using the [Quick, Draw!](https://quickdraw.withgoogle.com/data) dataset.

---

## ✨ Inspiration and Learning

### 🎥 [3Blue1Brown: What is a Neural Network?](https://www.youtube.com/watch?v=aircAruvnKk)
This was *the* video that finally made neural networks click. It’s a beautiful visual walkthrough of what’s actually happening in a neural net—no code needed. If you're just starting, this is a must-watch.

---

### ✍️ Medium Post (wish I remembered the link)
One of the funniest and most approachable intros I read—until I got lost in matrix shape conventions used for backprop math proofs. 

Still, the **concept** of backpropagation from 10,000 feet up finally clicked. I’ve saved the 3Blue1Brown backpropagation series for when I’m ready to dive deeper into the math.

---

## 🛠️ Setting Up

### 📘 Spun up this blog using Docusaurus
Okay, this isn’t about machine learning, but if you’re also documenting your journey: Docusaurus is great. I’ll probably host this on GitHub Pages later.

---

## 🔬 Learning Deep Learning

### 📺 [Intro to Deep Learning with PyTorch – FreeCodeCamp Playlist](https://www.youtube.com/watch?v=GIsg-ZUy0MY)
Started following along with this playlist. It’s been a surprisingly useful mix of theory and tooling:

- **Episode 1**: Taught me about GitHub Actions and how to link Colab notebooks to GitHub—wasn’t expecting that!
- **Episode 2**: Great intro to tensors. Moved a bit slow, but perfect if you're new.
- **Episode 3**: Introduced `shape` vs `view`, and I learned about tensor **strides**—an underrated concept.
- **Episode 4**: Accidentally wrote XOR logic while messing with tensor math. This led me down a delightful tangent about the **AI winter**, single-layer networks, and the historical importance of non-linearities.

```python
import torch
import torch.nn as nn

class XORNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 2),
            nn.ReLU(),
            nn.Linear(2, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)
```
The XOR function is often used as the "hello world" for why shallow neural nets need non-linearity to do anything useful. Shoutout to Minsky and Papert for that historical rabbit hole.

### All of this and no code?!
The code deserves it's very own page, let's go check it out!