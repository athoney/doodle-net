🔥 PyTorch Reshaping Quiz
1. What is the main requirement for using .view() on a tensor?
A) The tensor must be 2D
B) The tensor must have the same total number of elements
C) The tensor must be contiguous in memory
D) The tensor must be on the GPU

2. Which method is guaranteed to work on non-contiguous tensors?
A) .reshape()
B) .view()
C) .resize()
D) .flatten()

3. If you're reshaping a tensor and want to avoid unnecessary memory copying, which should you prefer (when safe)?
A) .reshape()
B) .clone()
C) .view()
D) .unsqueeze()

4. What will happen if you try to use .view() on a non-contiguous tensor?
A) It will reshape the tensor but with a warning
B) It will automatically make the tensor contiguous
C) It will throw a RuntimeError
D) It will reshape but slow down your code

5. Code Check: What will happen here?
python
Copy
Edit
x = torch.randn(3, 4)
y = x.t()            # Transpose
z = y.view(12)
A) z will be a reshaped tensor
B) It will error because y is not contiguous
C) It will silently return wrong values
D) It will automatically fix the layout

6. Code Check: Which of the following ensures view() will work after a transpose?
A)

python
Copy
Edit
x = torch.randn(3, 4).t()
x = x.view(12)
B)

python
Copy
Edit
x = torch.randn(3, 4).t()
x = x.contiguous()
x = x.view(12)
C)

python
Copy
Edit
x = torch.randn(3, 4).t()
x = x.reshape(12)