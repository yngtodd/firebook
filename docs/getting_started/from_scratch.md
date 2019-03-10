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

 | $x_1$ | $x_2$ | label |
 | :---: | :---: | :---: |
 | 1     | 0     | 1     |
 | 1     | 1     | 0     |
 | 0     | 1     | 1     |
 | 0     | 0     | 0     |

 The XOR problem is interesting in that it is not 
 [linearly separable](https://en.wikipedia.org/wiki/Linear_separability). As such, a single
 layer neural network (perceptron) cannot learn a decision boundary that will correctly 
 classify this problem. Our multi-layer perceptron, however, can.

```python
def print_predictions(model):
    """Print the probability for each example.
    
    This is just a helper function to check in our model.
    """
    pred00 = model.forward(torch.tensor([0., 0.]))
    pred10 = model.forward(torch.tensor([1., 0.]))
    pred01 = model.forward(torch.tensor([0., 1.]))
    pred11 = model.forward(torch.tensor([1., 1.]))
    print(f"Prediction (0, 0): {pred00.item():.4f}")
    print(f"Prediction (1, 0): {pred10.item():.4f}")
    print(f"Prediction (0, 1): {pred01.item():.4f}")
    print(f"Prediction (1, 1): {pred11.item():.4f}")


# Use for replicability.
torch.manual_seed(42)
    
# Training data for XOR.
x = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float)
y = torch.tensor([[0], [1], [1], [0]], dtype=torch.float)

# Instantiate our model
# Note: num_classes can be set to 1 since we can get the logits of the 
# negative class by taking 1 - logits.
model = MultiLayerPerceptron(input_size=2, hidden_size=3, num_classes=1)
 
# Binary cross entropy loss
criterion = nn.BCELoss()
```

 Above we have defined a helper function, `print_predictions` to give us the 
 predicted probability of each class by our model. We then define two Pytorch
 tensors, `x` and `y` for our training data and labels respectively. We then 
 instantiate our model, telling it that our input data has two dimensions, and
 that we would like the hidden dimension of our model to have three dimensions.
 Finally, we will use [`nn.BCELoss`](https://pytorch.org/docs/stable/nn.html#torch.nn.BCELoss)
 as our loss function. Since we randomly initialize the weights of our neural network,
 we can first see what our model will predict for each of the classes before being
 trained:

```python
print_predictions(model)

>>> Prediction (0, 0): 0.7342
>>> Prediction (1, 0): 0.7697
>>> Prediction (0, 1): 0.7830
>>> Prediction (1, 1): 0.8135
```

 At first, the model gives a high likelihood of each class. To train our model, 
 we simply run

```python
# Train the model for 2000 epochs.
# Each epoch will see every sample of data.
for i in range(2000):
    if i % 100 == 0:
        loss = criterion(m(x), y)
        print (f"Epoch {i} Loss: {loss}")
    model.train(x, y)
```

Finally, we can print our model predictions again, and we find that the model has
learned the XOR function:

```python
print_predictions(model)

>>> Prediction (0, 0): 0.0581
>>> Prediction (1, 0): 0.9210
>>> Prediction (0, 1): 0.9029
>>> Prediction (1, 1): 0.0781
```