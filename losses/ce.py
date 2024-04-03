import torch
import torch.nn as nn
class CE(nn.Module):
    def __init__(self):
        super(CE, self).__init__()
        self.fn = nn. CrossEntropyLoss()

    def forward(self, output, target):
        return self.fn(output, target)
