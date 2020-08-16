"""Module for data iteration abstraction."""
from datatype import Tensor


class DataSet:
    """Base class for iterating through datasets."""

    def __init__(self, X: Tensor, y: Tensor):
        """Construct the instance of the class."""
        self.X = X
        self.y = y

    def get_data(self, batch_size: int):
        """Generate iterator for the dataset."""
        offset = 0
        while (offset+batch_size <= len(self.X)):
            yield (self.X[offset:offset+batch_size],
                   self.y[offset:offset+batch_size])
            offset += batch_size
