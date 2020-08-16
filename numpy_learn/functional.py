"""Basic functions needed for the module."""
import numpy as np
from scipy.special import softmax as s
from datatype import Tensor


def sigmoid(x: Tensor) -> Tensor:
    """Calculate the sigmoid function of x."""
    return 1 / (1 + np.exp(-x))


def sigmoid_prime(x: Tensor) -> Tensor:
    """Calculate the d/dx of sigmoid function of x."""
    return sigmoid(x) * (1 - sigmoid(x))


def softmax(x: Tensor) -> Tensor:
    """Calculate softmax using scipy."""
    return s(x, axis=1)
