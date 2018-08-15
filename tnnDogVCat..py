import torch.nn as nn
import torch.nn.functional as F


class DogVsCat(nn):

    def __init__(self):
        super(DogVsCat, self).__init__()
        
        self.conv1 = nn.Conv2d(1, 32, 3)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.conv3 = nn.Conv2d(64, 64, 3)     
        self.fc1 = nn.Linear(64*3*3 ,64)
        self.fc2 = nn.Linear(64 ,2)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.dropout(F.max_pool2d(F.relu(self.conv3(F.relu(self.conv2))), 2))
        x = x.view(-1, 64*3*3)
        x = F.relu(self.fc1(x))
        x = F.dropout(x)
        x = F.softmax(self.fc2(x))
        return x
