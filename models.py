import torch
import torch.nn as nn

import torch
import torch.nn as nn
from torchvision.models import resnet18,resnet50,googlenet,vgg16

class Cifar10CNN(nn.Module):
    def __init__(self):
        super(Cifar10CNN, self).__init__()
        
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1), 
            nn.ReLU(),
            nn.MaxPool2d(2), 
            
            nn.Conv2d(16, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2), 
            
            nn.Flatten(), 
            nn.Linear(64 * 8 * 8, 512), 
            nn.ReLU(),
        )

        
        self.classifier = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):

        projection_head = self.feature_extractor(x)

        x = self.classifier[0](projection_head) 
        x = self.classifier[1](x) 
        intermediate_mapping = x 
   
        output = self.classifier[2](x) 

        return intermediate_mapping, output
    
    def forward_classifier(self, features):

        x = self.classifier[0](features)  
        x = self.classifier[1](x)  
        intermediate_mapping = x 

        output = self.classifier[2](x)  

        return intermediate_mapping, output
    
class GeneratorCifar10(nn.Module):
    def __init__(self, latent_dim=110, output_dim=512):
        super(GeneratorCifar10, self).__init__()
        self.fc1 = nn.Linear(latent_dim, 256)
        self.fc2 = nn.Linear(256, 512) 

    def forward(self, z):
        x = torch.relu(self.fc1(z))
        x = torch.relu(self.fc2(x))  
        return x
    
class Cifar100CNN(nn.Module):
    def __init__(self, num_classes=100):

        super(Cifar100CNN, self).__init__()

        self.feature_extractor = googlenet(weights=True) 

        self.classifier = nn.Sequential(
            nn.Linear(1000, 1000),
            nn.ReLU(),
            nn.Linear(1000, 512),
            nn.ReLU(),
            nn.Linear(512, num_classes)
        )


    def forward(self, x):

        x = self.feature_extractor(x)

        x = self.classifier[0](x)
        x = self.classifier[1](x) 

        intermediate_mapping = x

        x = self.classifier[2](intermediate_mapping) 
        x = self.classifier[3](x) 

        output = self.classifier[4](x) 
        return intermediate_mapping,output

    def forward_classifier(self, features):

        x = self.classifier[0](features)
        x = self.classifier[1](x)
        intermediate_mapping = x 

        x = self.classifier[2](intermediate_mapping)  
        x = self.classifier[3](x) 

        output = self.classifier[4](x) 

        return intermediate_mapping, output 
    
class GeneratorCifar100(nn.Module):
    
    def __init__(self, latent_dim=200):
        super(GeneratorCifar100, self).__init__()
        self.fc1 = nn.Linear(latent_dim, 512)
        self.fc2 = nn.Linear(512, 1000) 

    def forward(self, z):
        x = torch.relu(self.fc1(z))
        x = torch.relu(self.fc2(x)) 
        return x
    
class TinyImageNetCNN(nn.Module):
    def __init__(self, num_classes=200):

        super(TinyImageNetCNN, self).__init__()

        self.feature_extractor = googlenet(weights=True) 

        self.classifier = nn.Sequential(
            nn.Linear(1000, 1000),
            nn.ReLU(),
            nn.Linear(1000, 512),
            nn.ReLU(),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):

        x = self.feature_extractor(x)

        x = self.classifier[0](x)  
        x = self.classifier[1](x) 
        intermediate_mapping = x  

        x = self.classifier[2](x) 
        x = self.classifier[3](x)  

        output = self.classifier[4](x)  
        return intermediate_mapping,output
    
    def forward_classifier(self, features):
        
        x = self.classifier[0](features) 
        x = self.classifier[1](x) 
        intermediate_mapping = x 

        x = self.classifier[2](x) 
        x = self.classifier[3](x)  

        output = self.classifier[4](x)  

        return intermediate_mapping, output


class TinyImageNetGenerator(nn.Module):
     
    def __init__(self, latent_dim=300):
        super(TinyImageNetGenerator, self).__init__()
        self.fc1 = nn.Linear(latent_dim, 512)
        self.fc2 = nn.Linear(512, 1000) 

    def forward(self, z):
        x = torch.relu(self.fc1(z))
        x = torch.relu(self.fc2(x)) 
        return x
    
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN,self).__init__()

        self.feature_extractor = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1), 
            nn.ReLU(),
            nn.MaxPool2d(2), 
            
            nn.Conv2d(16, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2), 
            
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 512),
            nn.ReLU(),
        )

        
        self.classifier = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):

        x = self.feature_extractor(x)

        x = self.classifier[0](x) 
        x = self.classifier[1](x) 
        intermediate_mapping = x 
        output = self.classifier[2](x) 

        return intermediate_mapping, output
    
    def forward_classifier(self, features):

        x = self.classifier[0](features) 
        x = self.classifier[1](x) 
        intermediate_mapping = x 

        output = self.classifier[2](x) 

        return intermediate_mapping, output

class SimpleGenerator(nn.Module):
    
    def __init__(self,latent_dim=110):
        super(SimpleGenerator, self).__init__()
        self.fc1 = nn.Linear(latent_dim, 256)
        self.fc2 = nn.Linear(256, 512) 

    def forward(self, z):
        x = torch.relu(self.fc1(z))
        x = torch.relu(self.fc2(x))
        return x
