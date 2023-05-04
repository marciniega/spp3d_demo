#!/usr/bin/env python
import numpy as np
import sys
import torch
import torch.nn.functional as F  # Parameterless functions, like (some) activation functions
from torch import optim  # For optimizers like SGD, Adam, etc.
from torch import nn  # All neural network modules
from torch.utils.data import Dataset, DataLoader  # Gives easier dataset managment by creating mini batches etc.
from tqdm import tqdm  # For nice progress bar!
import math
from torchsummary import summary

path_to_data = "PATH_TO_DATA"
learning_rate = 0.001
num_epochs = 10
params = {'dim': (50,40,60),
          'batch_size': 50,
          'n_classes': 5,
          'n_channels': 1,
          'shuffle': True}

class Spp3D(nn.Module):
    """
    Args:
        levels (tuple): Length of side in the pooling results of each pyramid layer.

    Inputs:
        - `input`: the input Tensor to invert ([batch, channel, width, height])
    """

    def __init__(self, levels, mode='max'):
        super(Spp3D, self).__init__()
        self.levels = levels
        self.mode = mode

    def forward(self, x):
        out = None
        for n in self.levels:
            d_r, w_r, h_r = map(lambda s: math.ceil(s / n), x.size()[2:])  # Receptive Field Size
            s_d, s_w, s_h = map(lambda s: math.floor(s / n), x.size()[2:])  # Stride
            if self.mode == 'max':
               pool = nn.MaxPool3d(kernel_size=(d_r, w_r, h_r), stride=(s_d, s_w, s_h))
            elif self.mode == 'avg':
               pool = nn.AvgPool3d(kernel_size=(d_r, w_r, h_r), stride=(s_d, s_w, s_h))
            else:
               raise RuntimeError("Unknown pooling type: %s, please use \"max\" or \"avg\".")
            y = pool(x)
            if out is None:
                out = y.view(y.size()[0], -1)
            else:
                out = torch.cat((out, y.view(y.size()[0], -1)), 1)
        return out

    def get_output_size(self, filters):
        out = 0
        for level in self.levels:
            out += filters * level * level * level
        return out

class CNN(nn.Module):
    def __init__(self, in_channels=1, num_classes=5):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv3d(
            in_channels=in_channels,
            out_channels=64, #32
            kernel_size=(3, 3, 3),
            stride=(1, 1, 1),
            padding=(0, 0, 0),
        )
        self.pool1 = nn.MaxPool3d(kernel_size=(3, 3 ,3 ), stride=(2, 2, 2))
        self.conv2 = nn.Conv3d(
            in_channels=64,
            out_channels=128,
            kernel_size=(3, 3, 3),
            stride=(1, 1, 1),
            padding=(0, 0, 0),
        )
        self.pool2 = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=(2, 2, 2))
        self.conv3 = nn.Conv3d(
            in_channels=128,
            out_channels=64, #128
            kernel_size=(3, 3, 3),
            stride=(1, 1, 1),
            padding=(0, 0, 0),
        )
        self.pool3 = Spp3D([1,2,4],mode='max')
        self.dropout1 = nn.Dropout3d(0.5)
        self.fc1 = nn.Linear(4672,32) #32
        #self.fc1 = nn.Linear(16*(1+8+64),64)
        self.dropout2 = nn.Dropout3d(0.3)
        self.fc2 = nn.Linear(32,num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))   #First convolutional layer and ReLU applied
        x = self.pool1(x)           #Max Pooling 3D
        x = F.relu(self.conv2(x))   #Second convolutional layer and ReLU applied
        x = self.pool2(x)           #Max Pooling 3D
        x = F.relu(self.conv3(x))   #Third convolutional layer and ReLu applied
        x = self.pool3(x)           #SPP max
        #x = self.dropout1(x)        #First Dropout
        x = torch.flatten(x,1)        #Flatten tensor
        x = F.relu(self.fc1(x))     #First fully connected layer
        #x = self.dropout2(x)        #Second Dropout
        x = self.fc2(x)             #Second fully connected layer
        output = F.log_softmax(x, dim=1) #Softmax
        return output

def check_accuracy(loader, model):
    num_correct = 0
    num_samples = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device)
            y = y.to(device=device)

            scores = model(x.float())
            _, predictions = scores.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)

    model.train()
    return num_correct/num_samples

def load_image(apath):
    with np.load(apath+'.npz','rb') as datfil:
         a = datfil['arr_0']
         return a.reshape((params['n_channels'],params['dim'][0],params['dim'][1],params['dim'][2]))

def load_label(filename):
    yt = int(filename.split('_')[0])-1
    return yt

def read_annotations(annotations_file):
    annts= [] 
    for line in open(annotations_file).readlines():
        line = line.split('\n')[0]
        annts.append(line)
    return annts

class CustomDataset(Dataset):
    def __init__(self, annotations_file, img_dir):
        self.img_labels = read_annotations(annotations_file)
        self.img_dir = img_dir

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_name = self.img_labels[idx]
        img_path = self.img_dir+img_name
        image = load_image(img_path)
        label = load_label(img_name)
        return image, label

training_data = CustomDataset("train_list.txt",path_to_data)
test_data = CustomDataset("test_list.txt",path_to_data)

train_loader = DataLoader(training_data, 
                          batch_size=params['batch_size'],
                          shuffle=params['shuffle'],
                          num_workers=8)
test_loader = DataLoader(test_data, 
                         batch_size=params['batch_size'], 
                         shuffle=params['shuffle'],
                         num_workers=8)

device = torch.device("cuda")

# Initialize network
model=CNN(in_channels=params['n_channels'],num_classes=params['n_classes']).to(device)

try:
        model.load_state_dict(torch.load('model_weights_spp3d.pth'))
        model.eval()
        print("Model weights were loaded.")
except FileNotFoundError:
        print("No model weights were found.")
        print("New trainning is about to start.")

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Train Network
for epoch in range(num_epochs):
    #for batch_idx, (data, targets) in enumerate(tqdm(train_loader)):
    for batch_idx, (data, targets) in enumerate(train_loader):
        # Get data to cuda if possible
        data = data.to(device=device)
        targets = targets.to(device=device)

        # forward
        scores = model(data.float())
        loss = criterion(scores, targets)

        # backward
        optimizer.zero_grad()
        loss.backward()

        # gradient descent or adam step
        optimizer.step()
    acc_train = check_accuracy(train_loader, model)*100
    acc_test = check_accuracy(test_loader, model)*100
    #print(epoch, ".2f"%acc_train ,acc_test )
    print(f"Accuracy on training set: {check_accuracy(train_loader, model)*100:.2f}")
    print(f"Accuracy on test set: {check_accuracy(test_loader, model)*100:.2f}")
torch.save(model.state_dict(), 'model_1.pth')
