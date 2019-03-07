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
# TODO: implement a model class with Pytorch tensors
```