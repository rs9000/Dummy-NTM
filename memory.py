import torch
from torch import nn
import torch.nn.functional as F


class Memory(nn.Module):
    def __init__(self, M, N):
        super(Memory, self).__init__()

        self.N = N
        self.M = M
        self.w_last = []
        self.reset_memory()

    def get_weights(self):
        return self.w_last

    def reset_memory(self):
        self.w_last = []

    def addressing(self, program):
        w = F.softmax(program)
        return w


class ReadHead(Memory):

    def __init__(self, M, N, function_vector_size):
        super(ReadHead, self).__init__(M, N)

        print("--- Initialize Memory: ReadHead")
        self.fc_read1 = nn.Linear(self.N, function_vector_size*function_vector_size).cuda()
        self.reset_parameters()

    def reset_parameters(self):
        # Initialize the linear layers
        nn.init.normal(self.fc_read1.weight, std=1)
        nn.init.normal_(self.fc_read1.bias, std=0.01)

    def read(self, memory, program):

        w = self.addressing(program)
        self.w_last.append(w)
        read = torch.matmul(w.cuda(), memory)
        read = F.logsigmoid(self.fc_read1(read))
        return read, w


class WriteHead(Memory):

    def __init__(self, M, N, controller_dim):
        super(WriteHead, self).__init__(M, N)

        print("--- Initialize Memory: WriteHead")
        self.fc_write1 = nn.Linear(controller_dim,  self.N).cuda()
        self.reset_parameters()

    def reset_parameters(self):
        # Initialize the linear layers
        nn.init.normal_(self.fc_write1.weight, std=1)
        nn.init.normal_(self.fc_write1.bias, std=0.01)

    def write(self, memory, w, a):
        a = torch.squeeze(a)
        add = torch.ger(w.cuda(), a)
        memory_update = memory.cuda() + add
        return memory_update

    def forward(self, x, memory, program):
        param = self.fc_write1(x)
        a = F.tanh(param)

        w = self.addressing(program)
        self.w_last.append(w)
        mem = self.write(memory, w, a)
        return mem, w