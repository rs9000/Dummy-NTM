import torch
from torch import nn

class Controller(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(Controller, self).__init__()

        print("--- Initialize Controller")
        self.fc1 = nn.Linear(num_inputs, num_outputs).cuda()
        self.reset_parameters()

    def reset_parameters(self):
        # Initialize the linear layers
        nn.init.xavier_uniform_(self.fc1.weight, gain=1.4)
        nn.init.normal_(self.fc1.bias, std=0.01)

    def forward(self, x, program):
        x = torch.cat((x, program.cuda()), dim=1)
        x = self.fc1(x)
        return x
