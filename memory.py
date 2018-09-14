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
        w = F.cosine_similarity(memory, k, -1, 1e-16)
        w = F.softmax(β * w, dim=-1)
        return w


class ReadHead(Memory):

    def __init__(self, M, N, controller_dim, function_vector_size):
        super(ReadHead, self).__init__(M, N)

        print("--- Initialize Memory: ReadHead")
        self.fc_read1 = nn.Linear(controller_dim, self.N+1)
        self.reset_parameters()

    def reset_parameters(self):
        # Initialize linear layers read1
        nn.init.normal_(self.fc_read1.weight, std=1)
        nn.init.normal_(self.fc_read1.bias, std=0.01)

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
        return read, w
