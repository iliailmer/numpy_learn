from tqdm import auto
from loss import MSE, CrossEntropyLoss
from layers import Linear, ReLU, Sigmoid
from network import Network
from optimizer import SGD
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, KFold
from datatype import Tensor


def to_one_hot(vector: Tensor) -> Tensor:
    """Create one hot encoding of a vector."""
    oh = np.zeros((vector.shape[0], vector.max()+1))
    oh[np.arange(vector.shape[0]), vector] = 1
    return oh


train = pd.read_csv('mnist_train.csv', header=None).values[:, 1:]
train_label = pd.read_csv(
    'mnist_train.csv', header=None).values[:, 0]

net = Network(layers=[
    Linear(784, 128),
    ReLU(),
    Linear(128, 1),
])

loss = MSE()
optim = SGD(1e-4, 0.)
x_train, x_val, y_train, y_val = train_test_split(
    train.astype(np.float32) / 255,
    train_label.astype(np.int32),
    test_size=0.2, random_state=42)  # to_one_hot

batch_size = 100
progress_bar = auto.tqdm(range(10))
for epoch in progress_bar:
    offset = 0
    val_err = 0
    err = 0
    while (offset+batch_size <= len(x_train)):
        data = x_train[offset:offset+batch_size, :]
        label = y_train[offset:offset+batch_size]
        pred = net(data)
        err += loss(pred, label.reshape(-1, 1))/(len(x_train)/batch_size)
        g = net.backward(loss.grad)
        optim.step(net)
        offset += batch_size
    offset = 0
    while (offset+batch_size <= len(x_val)):
        val_data = x_val[offset:offset+batch_size, :]
        val_label = y_val[offset:offset+batch_size]
        pred = net(val_data)
        val_err += loss(pred, val_label.reshape(-1, 1))/(len(x_val)/batch_size)
        offset += batch_size
    if (epoch) % 2 == 0:
        progress_bar.set_postfix({"Mean_loss_train": err,
                                  "Mean_loss_val": val_err})


test = pd.read_csv('mnist_test.csv', header=None).values[:, 1:]
test_label = pd.get_dummies(pd.read_csv(
    'mnist_test.csv', header=None).values[:, 0])

offset = 0
test_err = 0.
while (offset+batch_size <= len(test)):
    data = test[offset:offset+batch_size, :]
    label = test_label[offset:offset+batch_size]
    pred = net(data)
    test_err += loss(pred, label.reshape(-1, 1))/(len(x_val)/batch_size)
    offset += batch_size


print(f"Test Error is {test_err}")
