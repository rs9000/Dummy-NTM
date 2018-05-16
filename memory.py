import torch
from torch import nn
import torch.nn.functional as F


class Memory(nn.Module):
    def __init__(self, M, N):
        super(Memory, self).__init__()

        self.N = N
        self.M = M
        self.reset_memory()
        self.rw_addressing = []

    def get_weights(self):
        return self.rw_addressing

    def reset_memory(self):
        self.rw_addressing = []

    def addressing(self, k, b, memory):
        wc = self._similarity(k, b, memory)
        return wc

    def _similarity(self, k, β, memory):
        # Similarità coseno
        w = F.cosine_similarity(memory.cuda(), k, -1, 1e-16)
        w = F.softmax(β * w, dim=-1)
        return w


class ReadHead(Memory):

    def __init__(self, M, N, controller_dim, function_vector_size):
        super(ReadHead, self).__init__(M, N)

        print("--- Initialize Memory: ReadHead")
        self.fc_read1 = nn.Linear(controller_dim, self.N+1).cuda()
        # TODO: move decode layer in NTM class
        self.fc_decode = nn.Linear(self.N, function_vector_size*function_vector_size).cuda()
        self.reset_parameters()

    def reset_parameters(self):
        # Initialize linear layers read1
        nn.init.normal_(self.fc_read1.weight, std=1)
        nn.init.normal_(self.fc_read1.bias, std=0.01)
        # Initialize linear layers decoder
        nn.init.normal_(self.fc_decode.weight, std=1)
        nn.init.normal_(self.fc_decode.bias, std=0.01)

    def read(self, controller_out, memory):
        # Genera parametri
        param = self.fc_read1(controller_out)
        k, b = torch.split(param, [self.N, 1], dim=1)
        k = F.tanh(k)
        b = F.softplus(b)
        # Addressing
        w = self.addressing(k,b, memory)
        self.rw_addressing.append(w)
        # Read
        read = torch.matmul(w, memory)
        # Decode
        read = F.tanh(self.fc_decode(read))
        return read, w


class WriteHead(Memory):

    def __init__(self, M, N, controller_dim):
        super(WriteHead, self).__init__(M, N)

        print("--- Initialize Memory: WriteHead")
        self.fc_write1 = nn.Linear(controller_dim,  self.N + 1 + self.N).cuda()
        self.reset_parameters()

    def reset_parameters(self):
        # Initialize the linear layers
        nn.init.normal_(self.fc_write1.weight, std=1)
        nn.init.normal_(self.fc_write1.bias, std=0.01)

    def write(self, memory, w, a):
        #Write function (as NTM without erase vector)
        a = torch.squeeze(a)
        w = torch.squeeze(w)
        add = torch.ger(w, a)
        memory_update = memory.cuda() + add
        return memory_update

    def forward(self, controller_out, memory):
        # Genera parametri
        param = self.fc_write1(controller_out)
        k, b, a = torch.split(param, [self.N, 1, self.N], dim=1)
        k = F.tanh(k)
        b = F.softplus(b)
        a = F.tanh(a)
        # Addressing
        w = self.addressing(k, b, memory)
        self.rw_addressing.append(w)
        # Write
        mem = self.write(memory, w, a)
        return mem, w