import torch
import torch.nn as nn

class MalariaCNN(nn.Module):
    def __init__(self):
        super(MalariaCNN, self).__init__()
        # Première couche convolutive avec batch normalization
        self.conv1 = nn.Conv2d(3, 32, 3)
        self.bn1 = nn.BatchNorm2d(32)
        
        # Deuxième couche convolutive avec batch normalization
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.bn2 = nn.BatchNorm2d(64)
        
        # Troisième couche convolutive avec batch normalization
        self.conv3 = nn.Conv2d(64, 128, 3)
        self.bn3 = nn.BatchNorm2d(128)
        
        # Couches fully connected avec les bonnes dimensions
        self.fc1 = nn.Linear(128, 128)
        self.fc2 = nn.Linear(128, 2)
        
        # Autres couches
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(2)
        self.dropout = nn.Dropout(0.5)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        # Premier bloc
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        # Deuxième bloc
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        # Troisième bloc
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        # Adaptive pooling pour obtenir une taille fixe
        x = self.adaptive_pool(x)
        
        # Partie fully connected
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x

def load_model(model_path):
    model = MalariaCNN()
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model_state_dict'])
    return model 