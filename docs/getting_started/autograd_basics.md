# Autograd Basics

The power of Pytorch is not necessarily that it is a flexible
tool for deep learning. At its heart, Pytorch is a tensor library,
similar to Numpy. Unlike Numpy, however, it has built in automatic 
differentiation (AD) and can operate on both CPUs and GPUs.

## Autograd Example 1.

```python
import torch
import torch.nn as nn

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
