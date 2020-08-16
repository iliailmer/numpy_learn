"""Simple optimizer algorithm."""

from network import Network


class SGD:
    """Stochastic Gradient Descent class."""

    def __init__(self, lr: float, l2: float = 0.0):
        """Initialize with learning rate and l2-regularization parameter."""
        self.lr = lr
        self.l2 = l2

    def step(self, net: Network):
        """Perform optimization step."""
        for layer in net.layers:
            if hasattr(layer, "dydw"):
                layer.W = layer.W - (self.lr * layer.dydw) - 2 * self.l2 * layer.W
            if hasattr(layer, "dydb"):
                layer.b = layer.b - self.lr * layer.dydb - 2 * self.l2 * layer.b


# TODO: Add Adam?
