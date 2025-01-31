import torch
import torch.nn as nn
from torch.nn import functional

# Stop training if 99.5% accurate
EPOCH_BREAK_ACCURACY = .995

# How many images tested at a time
TEST_BATCH_SIZE = 1000

class CNN (nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        # Creating the convultional layers.  
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)

        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)

        # Fully connected layers
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)
    
    def forward(self, x):
        # input is shape of (28,28,1)
        # 28 pixels by 28pixels, one as we only run once, since we only have one color channel, white/black
        # if multiple color channels you would want to run more times, to compare for each color.

        x = self.conv1(x)

        # ReLU introduces non-linearity to the model
        x = functional.relu(x)
        x = self.conv2(x)
        x = functional.relu(x)
        x = functional.max_pool2d(x, 2)

        x = self.dropout1(x)
        x = torch.flatten(x, 1) # Flatten the layers to reduce the data from multidimensional to 10
        x = self.fc1(x)
        x = functional.relu(x)
        x = self.dropout2(x)
        self.fc2(x)

        return x


