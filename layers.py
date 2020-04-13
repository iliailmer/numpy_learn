"""Collection of neural network classes: linear layer, activations, etc."""
from functional import sigmoid, sigmoid_prime
import numpy as np
from datatype import Tensor


class Linear:
    """A linear layer."""

    def __init__(self, in_features: int, out_features: int):
        """Initialize a linear layer with weights and biases."""
        self.W = np.random.randn(in_features, out_features)

        self.b = np.random.randn(out_features)

    def forward(self, x: Tensor) -> Tensor:
        """Compute forward pass, return W @ x + b.

        Arguments:
            W: the weight Tensor of shape (in_featuers, out_features)
            b: the bias vector of shape (out_features,)
            x: the input of shape (batch_size, in_features)

        Returns:
            A tensor of shape (batch_size, out_features)

        """
        self.input = x
        return x @ self.W + self.b

    def backward(self, grad: Tensor) -> Tensor:
        """Propagate the gradient from the l+1 layer to l-1 layer.

        Arguments:
            grad: the tensor gradients from the l+1 layer to be
                  propagated, shape: (batch_size, out_features).

        References:
            http://home.agh.edu.pl/~vlsi/AI/backp_t_en/backprop.html

        """
        # in_feat by batch_size @ batch_size by out_feat
        self.dydw = self.input.T @ grad
        # we sum across batches and get shape (out_features)
        self.dydb = grad.sum(axis=0)
        # output must be of shape (batch_size, output_features)
        return grad @ self.W.T

    def __call__(self, x: Tensor) -> Tensor:
        """Peform forward pass on `__call__`."""
        return self.forward(x)

    def __repr__(self) -> str:
        """Print a representation for Jupyter/IPython."""
        return f"""Linear Layer:\n\tWeight: {self.W.shape}"""\
            + f"""\n\tBias: {self.b.shape}"""


class ReLU:
    """ReLU class."""

    def __init__(self):
        """Initialize the ReLU instance."""

    def forward(self, x: Tensor) -> Tensor:
        """Compute the activation in the forward pass.

        Arguments:
            x: Tensor of inputs, shape (batch_size, in_features)

        Returns:
            Tensor of shape (batch_size, in_features)

        """
        return np.maximum(x, 0)

    def backward(self, grad: Tensor) -> Tensor:
        """Compute the gradient and pass it backwards.

        Arguments:
            grad: Tensor of gradients of shape (batch_size, out_features)

        Returns:
            Tensor of shape (batch_size, out_features)

        """
        return np.maximum(grad, 0)

    def __call__(self, x: Tensor) -> Tensor:
        """Peform forward pass on `__call__`."""
        return self.forward(x)

    def __repr__(self) -> str:
        """Print a representation of ReLU for Jupyter/IPython."""
        return """ReLU()"""


class Sigmoid:
    """Sigmoid class."""

    def __init__(self):
        """Initialize the instance.

        We add the main function for activation and its derivative function.
        """
        self.sigmoid = sigmoid
        self.sigmoid_prime = sigmoid_prime

    def forward(self, x: Tensor) -> Tensor:
        """Compute the activation in the forward pass.

        Arguments:
            x: Tensor of inputs with shape(batch_size, in_features)

        Returns:
            Tensor of shape(batch_size, in_features)

        """
        self.input = x
        return self.sigmoid(x)

    def backward(self, grad: Tensor):
        """Compute the gradient and pass it backwards.

        Arguments:
            grad: Tensor of gradients with shape(batch_size, out_features)

        Returns:
            Tensor of shape(in_features, out_features)

        """
        return self.sigmoid_prime(self.input) * grad

    def __call__(self, x: Tensor) -> Tensor:
        """Peform forward pass on `__call__`."""
        return self.forward(x)

    def __repr__(self) -> str:
        """Print a representation of Sigmoid for Jupyter/IPython."""
        return """Sigmoid()"""
