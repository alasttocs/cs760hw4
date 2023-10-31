
# Package imports
import numpy as np
import matplotlib.pyplot as plt
# here planar_utils.py can be found on its github repo
from planar_utils import plot_decision_boundary, sigmoid, load_planar_dataset
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import datasets
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt


def ce_loss(y, y_hat):
    ce_loss = -np.sum(y * np.log(y_hat))
    return ce_loss


# https://shaktiwadekar.medium.com/how-to-avoid-numerical-overflow-in-sigmoid-function-numerically-stable-sigmoid-function-5298b14720f6
def sigmoid(z):
    sigmoid = np.where(z >= 0, 1 / (1 + np.exp(-z)),
                       np.exp(z) / (1 + np.exp(z)))
    return sigmoid


def softmax(z):
    # for numerical stability https://www.turing.com/kb/softmax-activation-function-with-python
    max_z = np.max(z, axis=1, keepdims=True)
    e_z = np.exp(z - max_z)
    softmax = e_z / np.sum(e_z, axis=1, keepdims=True)
    return softmax


def forward_pass(x, w1, w2):
    z1 = np.dot(x, w1.T)
    a1 = sigmoid(z1)
    z2 = np.dot(a1, w2.T)
    y_hat = softmax(z2)
    return z1, a1, z2, y_hat


def back_pass(x, y, y_hat, a1, w1, w2):
    global learning_rate
    dz2 = y_hat - y
    dw2 = np.dot(dz2.T, a1)
    dz1 = np.dot(dz2, w2) * a1 * (1 - a1)
    dw1 = np.dot(dz1.T, x)
    w1 = w1 - (learning_rate * dw1)
    w2 = w2 - (learning_rate * dw2)
    return w1, w2


def predict(x, w1, w2):
    _, _, _, y_hat = forward_pass(x, w1, w2)
    y_pred = np.argmax(y_hat, axis=1)
    return y_pred


# load the MNIST dataset https://www.datascienceweekly.org/tutorials/pytorch-mnist-load-mnist-dataset-from-pytorch-torchvision
mnist_trainset = datasets.MNIST(
    root='./data', train=True, download=True, transform=transforms.ToTensor())
mnist_testset = datasets.MNIST(
    root='./data', train=False, download=True, transform=transforms.ToTensor())

batch_size = 32

learning_rate = 0.0001
w1 = np.random.uniform(-1, 1, size=(300, 784))
w2 = np.random.uniform(-1, 1, size=(10, 300))
epochs = 25


train_loader = DataLoader(mnist_trainset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(
    mnist_testset, batch_size=10000, shuffle=False)

for epoch in range(epochs):
    cost = 0
    loss = []
    for i, (x_batch, y_batch) in enumerate(train_loader):
        X = x_batch.numpy().reshape(batch_size, 784)
        Y = np.eye(10)[y_batch.numpy()]
        z1, a1, z2, y_hat = forward_pass(X, w1, w2)

        cost = ce_loss(Y, y_hat)
        w1, w2 = back_pass(X, Y, y_hat, a1, w1, w2)
    loss.append(cost)
    print(f"loss: {loss}")

correct = 0
total = 0
for i, (x, y) in enumerate(test_loader):
    x = x.numpy().reshape(10000, 784)
    y = y.numpy()
    y_pred = predict(x, w1, w2)
    for i in range(len(y)):
        if y_pred[i] == y[i]:
            correct += 1
        total += 1
print(f"correct, total & accuracy: {correct} / {total} = {correct/total}")
print(f"error rate: {(1 - correct/total) * 100 }%")
print("set to zero")
