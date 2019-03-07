# Dataset Classes

Data classes give us the ability to conveniently contain our data. When we
inherit from `torch.utils.data.Dataset`, we need to define two methods,
`__getitem__` and `__len__`. The first method, `__getitem__`, loads the data
at a given batch `index`, preprocesses it, and returns, in the supervised 
learning case, a tuple (data, label). The `__len__` method returns the number
of data samples in our dataset. This is used internally by Pytorch's dataloader
class.

## Dataset Template

```python
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

# Template for a dataset. 
class CustomDataset(Dataset):
    def __init__(self):
        # TODO
        # 1. Initialize file paths or a list of file names.
        pass
    def __getitem__(self, index):
        # TODO
        # 1. Read one data from file (e.g. using numpy.fromfile, PIL.Image.open).
        # 2. Preprocess the data (e.g. torchvision.Transform).
        # 3. Return a data pair (e.g. image and label).
        pass
    def __len__(self):
        # You should change 0 to the total size of your dataset.
        return 1 

# You can then use the prebuilt data loader.
dataset = CustomDataset()

train_loader = DataLoader(
    dataset=dataset,
    batch_size=64,
    shuffle=True
)
```

That last bit shows the `torch.utils.data.Dataloader` class. This class wraps 
our dataset, determines the batch size handled by each gradient descent step, 
and controls any shuffling of the between epochs.
