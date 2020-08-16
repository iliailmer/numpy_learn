"""Training example for a simple network with MNIST Dataset."""
from tqdm import auto
from loss import CrossEntropyLoss
from layers import Linear, Sigmoid
from network import Network
from optimizer import SGD
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
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
    Sigmoid(),
    Linear(128, 10),
])

loss = CrossEntropyLoss()
optim = SGD(5e-2, 0.001)
x_train, x_val, y_train, y_val = train_test_split(
    train.astype(np.float32) / 255,
    train_label.astype(np.int32),
    test_size=0.2, random_state=42)  # to_one_hot

y_train = to_one_hot(y_train)
y_val = to_one_hot(y_val)
batch_size = 100
progress_bar = auto.tqdm(range(200))
accuracies: dict = {"train": [],
                    "val": [],
                    "test": []}
acc_train: list = []
acc_val: list = []

for epoch in progress_bar:
    offset = 0
    val_err = 0
    err = 0
    while (offset+batch_size <= len(x_train)):
        data = x_train[offset:offset+batch_size, :]
        label = y_train[offset:offset+batch_size, :]
        try:
            pred = net(data)
        except RuntimeWarning:
            print(f"Runtime warning on {offset}")
        err += loss(pred, label)/(len(x_train)/batch_size)
        g = net.backward(loss.grad)
        optim.step(net)
        offset += batch_size
        acc_train.append(accuracy_score(
            label.argmax(axis=1),
            pred.argmax(axis=1)
        ))
    offset = 0
    while (offset+batch_size <= len(x_val)):
        val_data = x_val[offset:offset+batch_size, :]
        val_label = y_val[offset:offset+batch_size]
        pred = net(val_data)
        val_err += loss(pred, val_label)/(len(x_val)/batch_size)
        offset += batch_size
        acc_val.append(accuracy_score(
            val_label.argmax(axis=1),
            pred.argmax(axis=1)
        ))
    if (epoch) % 2 == 0:
        progress_bar.set_postfix({"loss_train": err,
                                  "loss_val": val_err,
                                  "acc_val": np.mean(acc_val)})
    accuracies['train'].append(np.mean(acc_train))
    accuracies['val'].append(np.mean(acc_val))
    acc_train = []
    acc_val = []


test = pd.read_csv('mnist_test.csv', header=None).values[:, 1:]
test_label = to_one_hot(pd.read_csv(
    'mnist_test.csv',
    header=None).values[:, 0])


offset = 0
test_err = 0.
while (offset+batch_size <= len(test)):
    data = test[offset:offset+batch_size, :]
    label = test_label[offset:offset+batch_size]
    pred = net(data)
    test_err += loss(pred, label)/(len(test)/batch_size)
    offset += batch_size
    accuracies['test'].append(accuracy_score(
        label.argmax(axis=1),
        pred.argmax(axis=1)
    ))

print(f"Average Test Accuracy: {np.mean(accuracies['test']):.2f}")
