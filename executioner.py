import torch
import torch.nn as nn
import torch.nn.functional as F


class Executioner(nn.Module):
    """
    This class implements the execution engine of the FTM
    Takes as input a vector and some functions to be applied to it.
    Applies such functions.
    """

    def __init__(self):
        """ Class constructor. """
        super(Executioner, self).__init__()

    def forward(self, input, functions):
        """
        Forward propagation function, applies functions.
        
        Parameters
        ----------
        input: Variable
            the input vector to be processes.
        functions: Variable
            the functions to be applied (the program).

        Returns
        -------
        Variable
            the processed vector.
        """

        # simply apply dot products
        h = input
        h = F.sigmoid(torch.mm(h, functions))
        out = h

        return out
