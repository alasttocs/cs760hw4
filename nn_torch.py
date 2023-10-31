import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import datasets
import torchvision.transforms as transforms


batch_size = 100

# load the MNIST dataset https://www.datascienceweekly.org/tutorials/pytorch-mnist-load-mnist-dataset-from-pytorch-torchvision
mnist_trainset = datasets.MNIST(
    root='./data', train=True, download=True, transform=transforms.ToTensor())
mnist_testset = datasets.MNIST(
    root='./data', train=False, download=True, transform=transforms.ToTensor())

# set up DataLoader for training set
train_loader = DataLoader(mnist_trainset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(
    mnist_testset, batch_size=len(mnist_testset), shuffle=False)

# create model
model = nn.Sequential(
    nn.Linear(784, 300),
    nn.Sigmoid(),
    nn.Linear(300, 10),
    nn.Softmax(dim=1)
)

# Train the model
epochs = 10
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)
model.train()
for epoch in range(epochs):
    for X_batch, y_batch in train_loader:
        X_batch = X_batch.view(X_batch.size(0), -1)
        y_pred = model(X_batch)
        loss = loss_fn(y_pred, y_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f"Epoch: {epoch}, Loss: {loss.item()}")

# evaluate accuracy after training
model.eval()
correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_loader:
        # Assuming 'images' and 'labels' are the batches
        # Process 'images' and 'labels' if needed
        images = images.view(images.size(0), -1)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()


print('Accuracy of the network on the test images: %d %%' %
      (100 * correct / total))
