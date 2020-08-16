"""Neural network wrapper."""

from datatype import Tensor
from typing import List, Union
from layers import Linear, ReLU, Sigmoid

Layer = Union[Linear, ReLU, Sigmoid]


class Network:
    """Basic Neural Network Class."""

    def __init__(self, layers: List[Layer]):
        """Initialize the Netowrk with a list of layers."""
        self.layers = layers[:]

    def forward(self, x: Tensor):
        """Run the forward pass."""
        for layer in self.layers:
            x = layer(x)
        return x

    def backward(self, grad: Tensor):
        """Run the backward pass."""
        for layer in self.layers[::-1]:
            grad = layer.backward(grad)
        return grad

    def __call__(self, x: Tensor):
        """Run the forward pass on __call__."""
        return self.forward(x)

    def __repr__(self) -> str:
        """Print the representation for the network."""
        return "\n".join(layer.__repr__() for layer in self.layers)
