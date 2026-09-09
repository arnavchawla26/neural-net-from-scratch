"""nn_scratch: a small multilayer perceptron with hand-derived NumPy
backprop -- no autodiff framework. See README.md for scope and usage.
"""
from .layers import Dense
from .activations import ReLU, Sigmoid, Tanh, Softmax
from .losses import MSELoss, CrossEntropyLoss
from .optim import SGD, SGDMomentum
from .network import MLP

__all__ = [
    "Dense",
    "ReLU",
    "Sigmoid",
    "Tanh",
    "Softmax",
    "MSELoss",
    "CrossEntropyLoss",
    "SGD",
    "SGDMomentum",
    "MLP",
]

__version__ = "0.1.0"
