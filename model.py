import torch.nn as nn

class AlexNet(nn.Module):
    def __init__(self, num_classes = 10, dropout_rate = 0.5):
        super().__init__()

        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(3, stride=2)
        self.dropout = nn.Dropout(dropout_rate)

        self.conv1 = nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=0)
        self.norm1 = nn.BatchNorm2d(96)

        self.conv2 = nn.Conv2d(96, 256, kernel_size=5, stride=1, padding=2)
        self.norm2 = nn.BatchNorm2d(256)

        self.conv3 = nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1)
        self.norm3 = nn.BatchNorm2d(384)

        self.conv4 = nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=1)
        self.norm4 = nn.BatchNorm2d(384)

        self.conv5 = nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1)
        self.norm5 = nn.BatchNorm2d(256)

        self.fc1 = nn.Linear(9216, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, num_classes)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu(x)
        x = self.pool(x)

        x = self.conv2(x)
        x = self.norm2(x)
        x = self.relu(x)
        x = self.pool(x)

        x = self.conv3(x)
        x = self.norm3(x)
        x = self.relu(x)

        x = self.conv4(x)
        x = self.norm4(x)
        x = self.relu(x)

        x = self.conv5(x)
        x = self.norm5(x)
        x = self.relu(x)
        x = self.pool(x)

        x = x.reshape(x.size(0), -1)

        x = self.dropout(x)
        x = self.fc1(x)
        x = self.relu(x)

        x = self.dropout(x)
        x = self.fc2(x)
        x = self.relu(x)

        x = self.fc3(x)

        return x