# Autograd Basics

The power of Pytorch is not necessarily that it is a flexible
tool for deep learning. At its heart, Pytorch is a tensor library,
similar to Numpy. Unlike Numpy, however, it has built in automatic 
differentiation (AD) and can operate on both CPUs and GPUs.

### Example 1.

```python
import torch

x = torch.tensor(5., requires_grad=True)
w = torch.randn(1, requires_grad=True)
b = torch.tensor(1., requires_grad=True)

# linear model
y = x * w + b

y.backward()

# Check out the gradients.
print(x.grad)    # x.grad = some small number
print(w.grad)    # w.grad = 5.
print(b.grad)    # b.grad = 1.
```

### Example 2.

```python
import torch
import torch.nn as nn

# Create tensors of shape (5, 5) and (5, 2).
x = torch.randn(5, 5)
y = torch.randn(5, 3)

# Build a fully connected layer.
model = nn.Linear(5, 2)
print (f'w: {model.weight}')
print (f'b: {model.bias}')

# Build loss function and optimizer.
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# Forward pass.
pred = model(x)

# Compute loss.
loss = criterion(pred, y)
print(f'loss: {loss.item()}')

# Backward pass.
loss.backward()

# Print out the gradients.
print (f'dL/dw: {model.weight.grad}')
print (f'dL/db: {model.bias.grad}')

# One step of gradient descent.
optimizer.step()

# Print out the loss after one step of gradient descent.
pred = model(x)
loss = criterion(pred, y)
print(f'loss after one iteration: {loss.item()}')
```
