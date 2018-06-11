import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from memory import ReadHead
from executioner import Executioner
from controller import Controller, RNNController, Encoder, Decoder


class NTM(nn.Module):
    def __init__(self, M, N, num_inputs, num_outputs, use_RnnController, controller_dim, function_vector_size, n_functions, input_embedding, output_embedding):
        super(NTM, self).__init__()

        print("----------- Build Neural Turing machine -----------")
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.controller_dim = controller_dim
        self.use_RnnController = use_RnnController
        self.M = M
        self.N = N

        self.memory = torch.nn.Parameter(torch.zeros(self.M, self.N))
        self.last_read = torch.zeros(1, self.N)

        self.function_vector_size = function_vector_size
        self.input_embedding = input_embedding.cuda()
        self.output_embedding = output_embedding.cuda()

        if self.use_RnnController == "rnn":
            self.controller = RNNController(n_functions, controller_dim)
        elif self.use_RnnController == "rnn_seq2seq":
            self.RNN_encoder = Encoder(n_functions, controller_dim)
            self.RNN_decoder = Decoder(controller_dim, controller_dim)
        else:
            self.controller = Controller(n_functions, controller_dim)

        self.read_head = ReadHead(self.M, self.N, controller_dim, function_vector_size)
        self.executioner = Executioner()

        self.programList = []
        self.initalize_state()

    def forward(self, X, program):

        self.programList = []
        self.read_head.reset_memory()
        self.rnn_hidden = None
        X2 = None
        out = None

        # STEP 1 Embedding input
        X = self._embed_input(X)

        # STEP 2 Controller
        # ----------------------------------------------------------------------
        # - Controller type=Seq2seq -> Encoder
        if self.use_RnnController == "rnn_seq2seq":
            for program_i in program[0]:
                program_i = program_i.view(1, 1, -1)
                _, self.rnn_hidden = self.RNN_encoder(program_i, self.rnn_hidden)

        for program_i in program[0]:
            # - Controller type=Seq2seq -> Decoder
            if self.use_RnnController == "rnn":
                program_i = program_i.view(1, 1, -1)
                out, self.rnn_hidden = self.controller(program_i, self.rnn_hidden)
            # - Controller type=RNN
            elif self.use_RnnController == "rnn_seq2seq":
                if out is None:
                    out = torch.ones(1, 1, self.controller_dim).cuda()
                out, self.rnn_hidden = self.RNN_decoder(out, self.rnn_hidden)
            # - Controller type=FeedForward
            else:
                out = self.controller(torch.unsqueeze(program_i, 0))
            # -----------------------------------------------------------------------

            # STEP 3 Read head
            self._read_write(out.view(1, -1))
            reshap = self.function_vector_size
            # STEP 4 Execute functions
            X = self.executioner(X, self.last_read.view(reshap, reshap))

        # STEP 5 Embedding output
        out = self._embed_output(X)
        return out

    def _read_write(self, controller_out):
        # READ
        read, w = self.read_head.read(controller_out, self.memory)
        self.last_read = read
        self.programList.append(read)

    def initalize_state(self):
        # Initialize stuff
        stdev = 1 / (np.sqrt(self.N + self.M))
        self.memory = nn.Parameter(nn.init.uniform_((torch.Tensor(self.M, self.N)), -stdev, stdev))
        self.last_read = F.tanh(torch.randn(1, self.N))
        self.read_head.reset_memory()
        self.programList = []

    def get_memory_info(self):
        # Get info for Tensorboard
        return self.memory, self.read_head.get_weights(), self.programList

    def calculate_num_params(self):
        """Returns the total number of parameters."""
        num_params = 0
        for p in self.parameters():
            num_params += p.data.view(-1).size(0)
        return num_params

    def _embed_input(self, input_vec):
        return torch.mm(input_vec, self.input_embedding)

    def _embed_output(self, activations):
        return torch.mm(activations, self.output_embedding)