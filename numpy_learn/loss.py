"""MSE Loss."""
from functional import softmax
from datatype import Tensor
import numpy as np


class Loss:
    """Placeholder class for losses."""

    def __init__(self):
        """Initialize the class with 0 gradient."""
        self.grad = 0.

    def grad_fn(self, pred: Tensor, true: Tensor) -> Tensor:
        """Create placeholder for the gradient funtion."""
        pass

    def loss_fn(self, pred: Tensor, true: Tensor) -> Tensor:
        """Create placeholder for the loss funtion."""
        pass

    def __call__(self, pred: Tensor, true: Tensor):
        """Calculate gradient and loss on call."""
        self.grad = self.grad_fn(pred, true)
        return self.loss_fn(pred, true)


class MSE(Loss):
    """Mean squared error loss."""

    def __init__(self):
        """Initialize via superclass."""
        super().__init__()

    def grad_fn(self, pred: Tensor, true: Tensor) -> Tensor:
        """Calculate the gradient of MSE.

        Args:
            pred: Tensor of predictions (raw output),
            shape (batch, )
            true: Tensor of true labels,
            shape (batch, )

        """
        return (pred - true)/true.shape[0]

    def loss_fn(self, pred: Tensor, true: Tensor) -> Tensor:
        """Calculate the MSE.

        Args:
            pred: Tensor of predictions (raw output),
            shape (batch,)
            true: Tensor of true labels (raw output),
            shape (batch,)

        """
        return 0.5*np.sum((pred - true)**2)/true.shape[0]

    def __repr__(self):
        """Put pretty representation in Jupyter/IPython."""
        return """Mean Squared Error loss (pred: Tensor, true: Tensor)"""


class CrossEntropyLoss(Loss):
    """CrossEntropyLoss class."""

    def __init__(self) -> None:
        """Initialize via superclass."""
        super().__init__()

    def loss_fn(self, logits: Tensor, true: Tensor) -> Tensor:
        """Calculate loss.

        Args:
            logits: Tensor of shape (batch size, number of classes),
            raw output of a neural network

            true: Tensor of shape (batch size,),
            a one-hot encoded vector

        """
        p = softmax(logits)
        return -np.sum(true * np.log(p))

    def grad_fn(self, logits: Tensor, true: Tensor) -> Tensor:
        """Calculate the gradient.

        Args:
            logits: Tensor of shape (batch size, number of classes),
            raw output of a neural network

            true: Tensor of shape (batch size, number of classes),
            a one-hot encoded vector

        """
        self.probabilities = softmax(logits)
        # one_hot = np.zeros((true.shape[0], true.max()+1),)
        # one_hot[np.arange(true.shape[0]), true] = 1
        return self.probabilities - true
