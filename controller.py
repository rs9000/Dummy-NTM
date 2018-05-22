import torch
from torch import nn
import torch.nn.functional as F

torch.manual_seed(1)

class Controller(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(Controller, self).__init__()

        print("--- Initialize Controller")
        self.fc1 = nn.Linear(num_inputs, num_outputs)
        self.reset_parameters()

    def reset_parameters(self):
        # Initialize the linear layers
        nn.init.normal_(self.fc1.weight, std=1)
        nn.init.normal_(self.fc1.bias, std=0.01)

    def forward(self, x):
        x = F.sigmoid(self.fc1(x))
        return x


class RNNController(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(RNNController, self).__init__()
        self.num_outputs = num_outputs
        self.num_inputs = num_inputs
        self.rnn = nn.LSTM(num_inputs, num_outputs, 1)

    def forward(self, inputs, hidden):
        if hidden is None:
            hidden = (torch.randn(1, 1, self.num_outputs).cuda(),
                      torch.randn(1, 1, self.num_outputs).cuda())

        output, hidden = self.rnn(inputs, hidden)
        output = F.sigmoid(output)
        return output, hidden


class RNNseqtoseq(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(RNNseqtoseq, self).__init__()
        self.num_outputs = num_outputs
        self.num_inputs = num_inputs


class Encoder(RNNseqtoseq):

    def __init__(self, num_inputs, num_outputs):
        super(Encoder, self).__init__(num_inputs, num_outputs)
        self.rnn = nn.LSTM(num_inputs, num_outputs, 1)

    def forward(self, inputs, hidden):
        if hidden is None:
            hidden = (torch.randn(1, 1, self.num_outputs).cuda(),
                      torch.randn(1, 1, self.num_outputs).cuda())

        output, hidden = self.rnn(inputs, hidden)
        return output, hidden


class Decoder(RNNseqtoseq):

    def __init__(self, num_inputs, num_outputs):
        super(Decoder, self).__init__(num_inputs, num_outputs)
        self.rnn = nn.LSTM(num_inputs, num_outputs, 1)

    def forward(self, inputs, hidden):
        output, hidden = self.rnn(inputs, hidden)
        return output, hidden