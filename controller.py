import torch
from torch import nn
import torch.nn.functional as F

class Controller(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(Controller, self).__init__()

        print("--- Initialize Controller")
        self.fc1 = nn.Linear(num_inputs, num_outputs).cuda()
        self.reset_parameters()

    def reset_parameters(self):
        # Initialize the linear layers
        nn.init.normal_(self.fc1.weight, std=1)
        nn.init.normal_(self.fc1.bias, std=0.01)

    def forward(self, x):
        x = F.sigmoid(self.fc1(x.cuda()))
        return x
