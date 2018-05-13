import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from memory import ReadHead, WriteHead
from executioner import Executioner
from controller import Controller


class NTM(nn.Module):
    def __init__(self, M, N, num_inputs, num_outputs, controller_dim, function_vector_size, max_program_length, input_embedding, output_embedding):
        super(NTM, self).__init__()

        print("----------- Build Neural Turing machine -----------")
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.M = M
        self.N = N

        self.memory = torch.zeros(self.M, self.N)
        self.last_read = torch.zeros(1, self.N)

        self.function_vector_size = function_vector_size
        self.input_embedding = torch.from_numpy(input_embedding)
        self.output_embedding = torch.from_numpy(output_embedding)

        self.controller = Controller(function_vector_size + max_program_length, controller_dim)
        self.read_head = ReadHead(self.M, self.N, function_vector_size)
        self.write_head = WriteHead(self.M, self.N, controller_dim)
        self.executioner = Executioner()

        self.programList = []

    def forward(self, X, program):
        # STEP 1 Embedding input
        X = self._embed_input(X)

        for program_i in program[0]:
            # STEP 2 Controller
            X2 = self.controller(X, torch.unsqueeze(program_i,0))
            # STEP 3 Write/Read head
            self._read_write(X2, program_i)
            reshap = self.function_vector_size
            # STEP 4 Execute functions
            X = self.executioner(X, self.last_read.view(reshap, reshap))

        # STEP 5 Embedding output
        out = self._embed_output(X)
        return out

    def _read_write(self, X, program):
        # WRITE
        mem, w = self.write_head(X, self.memory, program)
        self.memory = mem

        # READ
        read, w = self.read_head.read(self.memory, program)
        self.last_read = read
        self.programList.append(read)

    def initalize_state(self):
        #Initialize stuff
        stdev = 1 / (np.sqrt(self.N + self.M))
        self.memory = nn.init.uniform_((torch.Tensor(self.M, self.N)), -stdev, stdev)
        self.last_read = F.tanh(torch.randn(1, self.N))
        self.write_head.reset_memory()
        self.read_head.reset_memory()

    def get_memory_info(self):
        #Get info for Tensorboard
        return self.memory, self.read_head.get_weights(), self.last_read, self.programList

    def calculate_num_params(self):
        """Returns the total number of parameters."""
        num_params = 0
        for p in self.parameters():
            num_params += p.data.view(-1).size(0)
        return num_params

    def _embed_input(self, input_vec):
        return torch.mm(input_vec, self.input_embedding.cuda())

    def _embed_output(self, activations):
        return torch.mm(activations, self.output_embedding.cuda())