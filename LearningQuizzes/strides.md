🧠 Strides & Memory Quiz (Intermediate Level)
1. What do “strides” represent in a PyTorch tensor?
A) The number of non-zero elements
B) The byte size of each element
C) The number of elements to jump in memory to move to the next dimension
D) The shape of the tensor

2. If a tensor has shape (3, 4) and strides (4, 1), what does it mean?
A) It's stored in row-major order (C-style)
B) It's stored in column-major order (Fortran-style)
C) It has been permuted
D) It's non-contiguous

3. Which operation will most likely make a tensor non-contiguous?
A) .clone()
B) .reshape()
C) .transpose()
D) .contiguous()

4. What do you expect the stride to be for a contiguous tensor of shape (2, 3, 4)?
A) (1, 2, 3)
B) (12, 4, 1)
C) (4, 3, 2)
D) (1, 3, 12)

5. Code Check: What will be the output?
python
Copy
Edit
x = torch.randn(2, 3)
x_t = x.t()
print(x_t.is_contiguous())
print(x_t.stride())
A) True, (3, 1)
B) False, (1, 2)
C) False, (1, 2)
D) False, (1, 2) (this is a trick 👀)

6. If you call .view() on a non-contiguous tensor, how can you make it work first?
A) Call .permute()
B) Call .contiguous()
C) Call .flatten()
D) Call .clone()

