# Models

Models in Pytorch typically inherit from the 
[`torch.nn.Module`](https://pytorch.org/docs/stable/nn.html#module) class. The 
nn.Module class has a lot going on underneath the hood, but we only need to 
implement one method to get us started, the `forward` function.

Here is the basic setup. We start by defining a class that inherits from `nn.Module`.
In our class's `__init__` method, we compose some number of member variables which
define our model's graph. These member variables are often also `nn.Module`s.
[Composition](https://en.wikipedia.org/wiki/Object_composition) is incredibly valuable 
here and can be used to express large, complex models in a straightforward manner. Once
we have defined our model's structure, we then implement the `forward` method, which 
defines how data passes through the graph. 

## Model Template

```python
import torch.nn as nn

class Model(nn.Module):
    """Template for Pytorch models."""
    def __init__(self):
        super(Model, self).__init__()
        # ToDo
        # Compose model components (e.g. nn.Linear, nn.Conv2d, custom nn.Module)
    
    def forward(self, x):
        # ToDo
        # Pass our batched data, x, through our model components.
        # Optionally, do something crazy (e.g. print your intermediate representations, etc.)
```

You have probably heard of [backpropagation](https://www.youtube.com/watch?v=Ilg3gGewQ5U), the 
special case of automatic differentiation that makes deep learning tractable. You may then wonder 
why we only define the `forward` function and not the `backward` function in our model class.
So long as we define a differentiable function, Pytorch will take care of the backward pass for us.
This is where Pytorch's design around automatic differentiation comes into play. Next up, we'll go 
from our model template to an implementation of a classic neural network, the mulit-layer perceptron.

## Example 1: Multi-Layer Perceptron

```python
import torch.nn as nn

class MultiLayerPerceptron(nn.Module):
    """Fully connected network with one hidden layer."""
    def __init__(self, input_size, hidden_size, num_classes):
        super(MultiLayerPerceptron, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size) 
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(hidden_size, num_classes)  
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        out = self.fc2(x)
        return out
```