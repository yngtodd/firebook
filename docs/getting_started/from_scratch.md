# Illustration: A Neural Network from Scratch

In the [models](getting_started/models.md) section, we laid out a basic
template for neural network models in Pytorch. But quite a bit of the work 
handled by the model is abstracted away by the `nn.Module` class. Let's 
take a look at a simple example to illustrate what is going on when we 
train a neural network. I am going to keep the general structure on 
Pytorch's `nn.Module` class, but we will add a `backward` method to 
illustrate backpropagation for a simple fully connected network. You can
imagine that this is what Pytorch is doing in the background, though their
implementation is not quite this simple.

```python
import torch
import torch.nn as nn


class Sigmoid:
    """Standard Sigmoid function.
    
    Our forward function is the normal sigmoid()
    Our backward function is the functions' derivative.

    These names are just used to clarify when we use them 
    in our neural network.
    """
    def forward(self, s):
        return 1 / (1 + torch.exp(-s))
    
    def backward(self, s):
        """Derivative of the Sigmoid function."""
        return s * (1 - s)


class MultiLayerPerceptron(nn.Module):
    
    def __init__(self, input_size, hidden_size, num_classes):
        super(MultiLayerPerceptron, self).__init__()
        
        # Weights
        self.W1 = torch.randn(input_size, hidden_size)
        self.W2 = torch.randn(hidden_size, num_classes)
        # Activation function
        self.sigmoid = Sigmoid()
        
    def forward(self, x):
        self.z = torch.matmul(x, self.W1)
        self.z2 = self.sigmoid.forward(self.z)
        self.z3 = torch.matmul(self.z2, self.W2)
        out = self.sigmoid.forward(self.z3)
        return out
    
    def backward(self, x, y, logits):
        self.error = y - logits
        self.logits_delta = self.error * self.sigmoid.backward(logits)
        self.z2_error = torch.matmul(self.logits_delta, torch.t(self.W2))
        self.z2_delta = self.z2_error * self.sigmoid.backward(self.z2)
        self.W1 += torch.matmul(torch.t(x), self.z2_delta)
        self.W2 += torch.matmul(torch.t(self.z2), self.logits_delta)
        
    def train(self, x, y):
        logits = self.forward(x)
        self.backward(x, y, logits)
```

Above we have defined a fully connected neural network with one hidden layer. This 
network is composed of two layers of weights `W1` and `W2` which have shapes
`(input_size, hidden_size)` and `(hidden_size, num_classes)` respectively. When we 
instantiate our models, these weights will be initialized to be random normal vectors.

Connecting our two linear layers is the `sigmoid` function. The final layer of the network 
is similarly followed by the sigmoid function, which ensures all of our model's predictions 
lie within the interval [0,1].

### Example: XOR

 There is an early example problem for machine learning algorithms called the 'exclusive or'
 (XOR) problem. Given two binary variables, $x_{1}$ and $x_{2}$, the XOR function is true when
 either $x_{1}$ or $x_{2}$ is true, but not both. This can be summarized by the following truth
 table:

 ```
 | <code>x_{1}</code> | <code>x_{2}</code>   | label     |
 |   :---:            |   :---:              |   :---:   |
 | 1                  | 0                    | 1         |
 | 1                  | 1                    | 0         |
 | 0                  | 1                    | 1         |
 | 0                  | 0                    | 0         |
 ```