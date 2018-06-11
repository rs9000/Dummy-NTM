import numpy as np
from torch.utils.data import Dataset
from utils.generic import np_funct_categorical
import torch
import torch.nn.functional as F

#np.random.seed(30101990)
#torch.manual_seed(30101990)


class FunctionDataset(Dataset):
    """
    This class provides examples of compositional functions.
    Each example is composed of:
        Input: a vector having length `input_vector_size`.
        Functions: a series of one-hot vectors indexing the functions applied.
        Output: the input processed with the functions.
    """

    def __init__(self, input_vector_size, function_vector_size, output_vector_size,
                 n_functions, max_program_length, samples_per_epoch, use_curriculum,
                 train_transforms=None):
        """
        Dataset class constructor.

        Parameters
        ----------
        input_vector_size: int
            dimensionality of the input.
        function_vector_size: int
            dimensionality of the function.
        output_vector_size: int
            dimensionality of the output.
        n_functions: int
            the number of different functions to be applied.
        max_program_length: int
            the maximum number of functions that can be applied to each input.
        samples_per_epoch: int
            number of samples generated each epoch.
        use_curriculum: bool
            whether or not to increase the program length each epoch.
        train_transforms: callable
            callable object to be applied to samples.
        """

        assert max_program_length > 1, 'max_program_length should be > 1 (provided: {}).'.format(max_program_length)

        self._input_vector_size = input_vector_size
        self._function_vector_size = function_vector_size
        self._output_vector_size = output_vector_size
        self._n_functions = n_functions
        self._max_program_length = max_program_length
        self._samples_per_epoch = samples_per_epoch
        self._use_curriculum = use_curriculum
        self._train_transforms = train_transforms

        self._cur_max_program_length = max_program_length if not use_curriculum else 1

        # random parameters
        self._input_embedding = torch.FloatTensor(input_vector_size, function_vector_size).uniform_(-1, 1)

        self._functions = torch.FloatTensor(n_functions, function_vector_size, function_vector_size).uniform_(-1, 1)

        self._output_embedding = torch.FloatTensor(function_vector_size, output_vector_size).uniform_(-1, 1)


    def __len__(self):
        """
        __len__ method.

        Returns
        -------
        int
            the number of examples in the dataset.
        """
        return self._samples_per_epoch

    def __getitem__(self, idx):
        """

        Returns
        -------

        """

        # initialize input_vector
        input_vec = torch.FloatTensor(1, self._input_vector_size).uniform_(-1, 1)

        # sample a random number of functions
        program_length = np.random.randint(1, self._cur_max_program_length + 1)
        sampled_functions_idx = np.random.choice(np.arange(0, self._n_functions), size=program_length)
        sampled_functions = self._functions[sampled_functions_idx, :]

        # apply functions
        h = torch.mm(input_vec, self._input_embedding)  # input embedding
        for func in sampled_functions:
            h = F.sigmoid(torch.mm(h, func))
        output_vec = torch.mm(h, self._output_embedding)

        # get one-hot representation of functions applied.
        one_hot_functions = np_funct_categorical(functions_idx=sampled_functions_idx, n_functions=self._n_functions)

        # check if I need to update cur_max_program_length
        if self._use_curriculum and idx % 10000 == 0:
            # at the end of the epoch increase max program length
            self._cur_max_program_length = min(self._cur_max_program_length + 1, self._max_program_length)

        return input_vec, one_hot_functions, output_vec

    @property
    def input_embedding(self):
        """
        Returns parameters of the input to program embedding.

        Returns
        -------
        np.array()
            dense embedding parameters having shape (input_vector_size, function_vector_size).
        """
        return self._input_embedding

    @property
    def output_embedding(self):
        """
        Returns parameters of the function to output embedding.

        Returns
        -------
        np.array()
            dense embedding parameters having shape (function_vector_size, output_vector_size).
        """
        return self._output_embedding

    def program_list(self):
        return self._functions
