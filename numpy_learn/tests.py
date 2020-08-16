"""Test module."""
from datatype import Tensor  # noqa F401
from layers import Linear, ReLU, Sigmoid, MSE
from network import Network
from optimizer import SGD
import numpy as np


def testLinear():  # noqa D103
    linear = Linear(in_features=10, out_features=64)
    x = np.random.randn(32, 10)  # batch size by in_features
    output = linear(x)
    assert output.shape == (32, 64)


def testReLU():  # noqa D103
    relu = ReLU()
    x = np.random.randn(32, 10)  # batch size by in_features
    output = relu(x)
    assert output.shape == (32, 10)


def testSigmoid():  # noqa D103
    s = Sigmoid()
    x = np.random.randn(32, 10)  # batch size by in_features
    output = s(x)
    assert output.shape == (32, 10)


def testMSE():  # noqa D103
    mse = MSE()
    y_true = np.random.randn(32, 10)  # batch size by in_features
    y_pred = np.random.randn(32, 10)  # batch size by in_features
    err = mse(y_pred, y_true)
    assert isinstance(err, float)
    assert mse.grad.shape == (32, 10)


def testNetwork():  # noqa D103
    net = Network([Linear(10, 64),
                   ReLU(),
                   Linear(64, 2),
                   Sigmoid()])
    x = np.random.randn(32, 10)
    y = np.random.randn(32, 2)
    mse = MSE()
    optim = SGD(0.001, 0.001)
    pred = net(x)
    _ = mse(pred, y)
    _ = net.backward(mse.grad)
    optim.step(net)
