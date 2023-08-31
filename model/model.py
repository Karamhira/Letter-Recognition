import torch.optim as op
import cv2
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import TensorDataset
from sklearn.model_selection import train_test_split
import copy
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torchvision
from torchvision import transforms
import torch
import matplotlib
matplotlib.use('TkAgg')


def emnistData():
    dataSet = torchvision.datasets.EMNIST(
        root='./emnist',
        split='letters',
        download=True,
    )

    letterCategories = dataSet.classes[1:]
    labels = copy.deepcopy(dataSet.targets) - 1
    torch.unique(labels)

    images = dataSet.data.view([124800, 1, 28, 28]).float()

    images /= 255.0

    batch_size = 32
    loader = torch.utils.data.DataLoader(
        dataSet, batch_size=batch_size, shuffle=True)
    trainData, testData, trainLabels, testLabels = train_test_split(
        images, labels, test_size=0.2)

    trainData = TensorDataset(trainData, trainLabels)
    testData = TensorDataset(testData, testLabels)

    trainLoader = torch.utils.data.DataLoader(
        trainData, batch_size=batch_size, shuffle=True)
    testLoader = torch.utils.data.DataLoader(
        testData, batch_size=batch_size, shuffle=True)

    return trainLoader, testLoader


class emnistNet(nn.Module):
    def __init__(self):
        super(emnistNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.conv3 = nn.Conv2d(64, 128, 3, 1)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(1152, 128)
        self.fc2 = nn.Linear(128, 26)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


def trainModel(model, trainLoader, testLoader, numEpochs=10, learning_rate=0.001):
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using {device} device")

    model.to(device)
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(numEpochs):
        model.train()
        for batch_idx, (features, targets) in enumerate(trainLoader):
            features, targets = features.to(device), targets.to(device)
            optimizer.zero_grad()
            forwardPassOutputs = model(features)
            loss = loss_fn(forwardPassOutputs, targets)
            loss.backward()
            optimizer.step()

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for features, targets in testLoader:
                features, targets = features.to(device), targets.to(device)
                outputs = model(features)
                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
        accuracy = correct / total
        print(
            f"Epoch [{epoch+1}/{numEpochs}] Loss: {loss:.4f} Accuracy: {accuracy:.4f}")


if __name__ == "__main__":
    trainLoader, testLoader = emnistData()
    model = emnistNet()
    trainModel(model, trainLoader, testLoader)
    torch.save(model.state_dict(), 'emnistModel.pt')
