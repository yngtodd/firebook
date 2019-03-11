# Model Composition

In our [models](getting_started/models) section, we touched on the
value of [composition](https://en.wikipedia.org/wiki/Object_composition). 
When working with deep learning models, you will often find that the models 
have regular, repeating structures. A common pattern is a series of 
convolution layers along with batch normalization. It is generally best not to 
repeat ourselves when writing software, and this holds true in machine learning.
Let's take a look at an example deep network that used for 
[image segmentation](https://en.wikipedia.org/wiki/Image_segmentation), the U-Net,
and how the composition can help simplify our code.

## Example: U-Net

U-Nets are an example of what's known as fully-convolution neural networks. They
have repeating blocks of convolution layers and no fully connected layers. The name
comes from the distinct "U" shaped pattern of the network, where the first half of 
the model downsamples its input to some reduced dimension, while the second half of 
the network upsamples its input to return to the original data dimensions. The 
downsampling and upsampling sections are generally composed of multiple convolution 
layers along with bach normalization. We can then write these components of the model 
as `nn.Module`s, to be used repeatedly in our U-Net model.

```python
import torch

from torch import nn
import torch.nn.functional as F


class DoubleBlock(nn.Module):
    '''(conv => BN => ReLU) * 2'''
    def __init__(self, in_ch, out_ch):
        super(DoubleBlock, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class Down(nn.Module):

    def __init__(self, in_ch, out_ch):
        super(Down, self).__init__()
        self.pool = nn.MaxPool2d(2, return_indices=True)
        self.block = DoubleBlock(in_ch, out_ch)

    def forward(self, x):
        x = self.block(x)
        x, indices = self.pool(x)
        return x, indices


class Up(nn.Module):

    def __init__(self, in_ch, out_ch):
        super(Up, self).__init__()
        self.unpool = nn.MaxUnpool2d(2)
        self.block = DoubleBlock(in_ch, out_ch)

    def forward(self, x, indices, output_shape):
        x = self.unpool(x, indices, output_shape)
        x = self.block(x)
        return x
```
 