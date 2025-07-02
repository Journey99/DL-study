import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5, stride=1, padding=2)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=2)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, stride=1, padding=2)

        self.fc1 = nn.Linear(128 * 4 * 4, 1024)
        self.fc2 = nn.Linear(1024, 10)  # 10개 클래스 
        
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        # Conv1 -> ReLU -> MaxPool
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        
        # Conv2 -> ReLU -> MaxPool
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        
        # Conv3 -> ReLU -> MaxPool
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, 2, 2)

        # Flatten the tensor 
        x = x.view(-1, 128 * 4 * 4)
        
        # Fully Connected Layer 1 -> ReLU -> Dropout
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        
        # Fully Connected Layer 2 (output layer)
        x = self.fc2(x)
        
        return x

model = CNN()
# print(model)

input_data = torch.randn(16, 1, 28, 28)  
output = model(input_data)
print("출력 크기:", output.shape) 