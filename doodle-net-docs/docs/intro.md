---
sidebar_position: 1
---

# 🧠 Building a QuickDraw CNN
Welcome to my machine learning journey! This site documents my beginner-friendly (albeit slightly chaotic) path toward building a Convolutional Neural Network (CNN) to classify doodles using the [Quick, Draw!](https://quickdraw.withgoogle.com/data) dataset. Playing the game at least once is a required prerequisite for this project!


You know how when you find a recipe on the internet and it takes you 20 minutes to scroll through the author's life story before you get to the actual recipe? Boy, do I have bad news for you! **JK** I’m not going to make you scroll through my life story, but I do want to share some of the resources that helped me along the way.

---

## ✨ Inspiration and Learning

### 🎥 [3Blue1Brown: What is a Neural Network?](https://www.youtube.com/watch?v=aircAruvnKk)
This was *the* video that finally made neural networks click. It’s a great visual walkthrough of what’s actually happening in a neural net without overwhelming the viewer with math and code. If you're just starting, this is a must-watch.

---

### ✍️ Medium Post
[Learn to Build a Neural Network From Scratch — Yes, Really.](https://medium.com/@waadlingaadil/learn-to-build-a-neural-network-from-scratch-yes-really-cac4ca457efc)

This post is a GEM. Truly the most enjoyable and approachable medium post I've ever read. Personally, I got hung up on matrix shapes and ordering like `A0 @ W1.T` vs `W1 @ A0.T`, but after some googling, I learned that the author writes their matrix math through the lens of backprop math proofs. Key takeaway:  
   - **If you’re following a math-heavy or academic source** it will look like: Z = W @ A + b (inputs are column vectors and shape: input_dim × batch_size)
   - **If you’re writing code (NumPy, PyTorch, etc.)** it will look like: Z = A @ W.T + b (inputs are row vectors and shape: batch_size × input_dim) 


Personally, I find the code-style notation more intuitive. That said, if you're following along with the backpropagation derivation, I’d recommend sticking with the mathematical notation. Admittedly, I skipped over some of the proof-heavy sections because I was eager to start coding. Still, this post gave me a solid conceptual grasp of backprop, and I’ve bookmarked the 3Blue1Brown series on backpropagation for when I’m ready to circle back to the math in more depth. You can find it [here](https://www.youtube.com/watch?v=IHZwWFHWa-w&list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi&index=2).

---

## 🛠️ Setting Up

### 📘 Spun up this blog using Docusaurus
Okay, this isn’t about machine learning, but if you’re also documenting *literally* anything: Docusaurus is great. I have used Docusaurus for many projects because it is so easy to spin up and deploy. I would also turn to vercel for hosting, but I tried out Github Pages for this project...and guys, I have been missing out! Code, build, and deploy all in one place. LOVE. Shoutout to Daniel for this tutorial on [Github Pages Deployment](https://www.youtube.com/watch?v=9iVNf0T09dE). A few edits to `docusaurus.config.js`, a quick `npm run deploy`, and I was up and running.

---

## 🔬 Learning Deep Learning (get it?)

### 📺 [Intro to Deep Learning with PyTorch](https://www.youtube.com/watch?v=kY14KfZQ1TI&list=PLCC34OHNcOtpcgR9LEYSdi9r7XIbpkpK1)
Whenever I attempt to learn something new, I start small and equip myself with resources. My resource of choice this time was following along with this playlist. It’s been a surprisingly useful mix of theory and tooling:

- **Episode 1**: Taught me about GitHub Actions and how to link Colab notebooks to Github (cool tidbit)!
- **Episode 2**: Great intro to tensors. Moved a bit slow, but perfect if you're new.
- **Episode 3**: Introduced `shape` vs `view`, and I learned about tensor **strides**, neat tangent.
- **Episode 4**: Accidentally wrote XOR logic while messing with tensor math. This led me down a delightful rabbit hole about the **AI winter**, single-layer networks, and the historical importance of non-linearities. Turns out the XOR function is often used as the "hello world" for why shallow neural nets need non-linearity to do anything useful. Shoutout to Minsky and Papert for that.
- **Episode [5:-1]**: Learned all sorts of things about programming neural nets, cnns, and pytorch. Truthfully, I was so excited about the programming I forgot to take notes. 

### 😱 All of this and no code?!
This is a **journey**! But don't worry, the destination is up ahead.