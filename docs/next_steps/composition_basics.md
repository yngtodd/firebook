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
import torch.nn as nn


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
Our basic building block here is the `DoubleBlock` which is made up of 
two layers of convolution followed by ReLU activation functions and batch
normalization. This building block is used in both the `Down` and `Up` 
modules. The `Down` and `Up` modules are then used repeatedly in our U-Net
model.

```python
class InConv(nn.Module):

    def __init__(self, in_ch, out_ch):
        super(InConv, self).__init__()
        self.conv = DoubleBlock(in_ch, out_ch)

    def forward(self, x):
        return self.conv(x)


class OutConv(nn.Module):

    def __init__(self, in_ch, out_ch):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        return self.conv(x)


class UNet2D(nn.Module):

    def __init__(self, n_channels, n_classes):
        super(UNet2D, self).__init__()
        self.inconv = InConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024)
        self.up1 = Up(1024, 512)
        self.up2 = Up(512, 256)
        self.up3 = Up(256, 128)
        self.up4 = Up(128, 64)
        self.outconv = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inconv(x)
        x2, indices1 = self.down1(x1)
        x3, indices2 = self.down2(x2)
        x4, indices3 = self.down3(x3)
        x5, indices4 = self.down4(x4)
        x = self.up1(x5, indices4, x4.shape) 
        x = self.up2(x, indices3, x3.shape)
        x = self.up3(x, indices2, x2.shape)
        x = self.up4(x, indices1, x1.shape)
        x = self.outconv(x)
        x = torch.sigmoid(x)
        return x
```

The model we defined above has 19 layers of 2D convolution, along with all
of the associated batch normalization and activation functions. If we were to 
write out each of those layers, we would be bound to make numerous mistakes.
With composition, we can easily structure our code to make it both more 
readable and easier to experiment with.