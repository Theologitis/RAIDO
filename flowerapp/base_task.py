# task/base.py
from abc import ABC, abstractmethod

class BaseTask(ABC):
    """Abstract base class for all tasks."""

    @abstractmethod
    def get_model(self):
        """Return the initialized model (torch.nn.Module, keras.Model, etc.)."""
        pass

    @abstractmethod
    def train(self, model, train_loader, config):
        """Train the model on the given data loader and config."""
        pass

    @abstractmethod
    def test(self, model, test_loader):
        """Evaluate the model on the test loader. Return (loss, metrics_dict)."""
        pass

    @abstractmethod
    def get_data_loaders(self, client_id=None):
        """Return (train_loader, test_loader) for a specific client if needed."""
        pass

    @abstractmethod
    def get_input_shape(self):
        """Return the shape of a single input (e.g., for initializing models)."""
        pass

    @abstractmethod
    def get_output_size(self):
        """Return the number of output classes or targets."""
        pass

