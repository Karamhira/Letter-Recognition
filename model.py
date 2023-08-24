import torch.nn.functional as F
import copy
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torchvision
from torchvision import transforms
import torch
import matplotlib
matplotlib.use('TkAgg')
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset
import torch.nn as nn



dataSet = torchvision.datasets.EMNIST(
    root='./emnist', 
    split='letters', 
    download=True,
)

letterCategories = dataSet.classes[1:]
dataSet.classes = letterCategories
labels = copy.deepcopy(dataSet.targets) - 1
dataSet.targets = labels

print(str(len(dataSet.classes)) + ' classes')
print('\n datasize')
print(dataSet.data.shape) 
print(dataSet.classes)

for x in range(5):
    img = dataSet.data[x].numpy()
    print(repr(img))
    img = img.T
    plt.imshow(img, cmap='gray')
    plt.axis('off')
    plt.show()

#plt.hist(images[:10].view(-1).detach(), bins=40)
#plt.title("bottle")
#plt.show()

images /= 255.0

#plt.hist(images[:10].view(-1).detach(), bins=40)
#plt.title("bottle")
#plt.show()




batch_size = 32
loader = torch.utils.data.DataLoader(
    dataSet, batch_size=batch_size, shuffle=True)
train_data, test_data, train_labels, test_labels = train_test_split(
    images, labels, test_size=0.2)

train_data = TensorDataset(train_data, train_labels)
test_data = TensorDataset(test_data, test_labels)


train_loader = torch.utils.data.DataLoader(
    train_data, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(
    test_data, batch_size=batch_size, shuffle=True)

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # 1 input channel, 32 output channels, 3x3 kernel, stride 1
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        # 32 input channels, 64 output channels, 3x3 kernel, stride 1
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        
        self.conv3 = nn.Conv2d(64, 128, 3, 1)
        
        self.pool1 = nn.MaxPool2d(2, 2)
        self.pool2 = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

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

model = Net()
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=.001)
 
num_epochs = 20
for epoch in range(num_epochs):
    for batch_idx, (features, targets) in enumerate(train_loader):

        forward_pass_outputs = model(features)
        loss = loss_fn(forward_pass_outputs, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
