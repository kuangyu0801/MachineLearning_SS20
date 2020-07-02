# MachineLearning_SS20

## Reference Book
[Python Data Science Handbook: Essential Tools for Working with Data](https://jakevdp.github.io/PythonDataScienceHandbook/)

# Assignment 09
time-series data: training dataset and testing dataset should be also seperated in interval
no training data is the history of testing data, vice versa.

[pytorch](https://pytorch.org/docs/stable/nn.html)
we only need to implement FF and pytorch will do the rest!

torch.nn.Module
```
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5)
        self.conv2 = nn.Conv2d(20, 20, 5)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        return F.relu(self.conv2(x))
```
torch uses float64, need to convert it into float32

```
linear.forward(torch.from_numpy(data_train[0]['x']).float())
```

torch can remember the last execution result
in the way, it can directly compute backward prog without explicitly buffering loss result