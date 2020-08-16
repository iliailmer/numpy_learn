"""Simple optimizer algorithm."""

from network import Network


class SGD:
    """Stochastic Gradient Descent class."""

    def __init__(self, lr: float, l2: float = 0.):
        """Initialize with learning rate and l2-regularization parameter."""
        self.lr = lr
        self.l2 = l2

    def step(self, net: Network):
        """Perform optimization step."""
        for l in net.layers:
            if hasattr(l, 'dydw'):
                l.W = l.W - self.lr*l.dydw - 2 * self.l2 * l.W
            if hasattr(l, 'dydb'):
                l.b = l.b - self.lr*l.dydb - 2 * self.l2 * l.b

# TODO: Add Adam?
