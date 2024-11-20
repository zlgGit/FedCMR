import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset,Dataset
import numpy as np
from collections import defaultdict
from PIL import Image
import os
import pandas as pd

def load_train_data_CIFAR10():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True,transform =transform)
    return dataset
def load_test_loader_CIFAR10():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    return test_loader

def load_train_data_CIFAR100():
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2675, 0.2565, 0.2761))
    ])
    dataset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
    return dataset

def load_test_loader_CIFAR100():
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2675, 0.2565, 0.2761))
    ])
    test_dataset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    return test_loader

def load_train_data_FashionMNIST():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    train_dataset = torchvision.datasets.FashionMNIST(
        root='./data', train=True, download=True, transform=transform)
    return train_dataset

def load_test_loader_FashionMNIST():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    test_dataset = torchvision.datasets.FashionMNIST(root='./data', train=False, download=True,transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    return test_loader

def load_train_data_TinyImageNet():

    transform = transforms.Compose([
        transforms.Resize((224, 224)), 
        transforms.ToTensor(),        
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  
    ])


    train_dataset = torchvision.datasets.ImageFolder(root='./data/tiny-imagenet-200/train', transform=transform)


    return train_dataset, train_dataset.class_to_idx

def load_test_loader_TinyImageNet(class_to_idx, batch_size=64):

    transform = transforms.Compose([
        transforms.Resize((224, 224)), 
        transforms.ToTensor(),        
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])


    val_annotations_file='./data/tiny-imagenet-200/val/val_annotations.txt'
    val_labels = pd.read_csv(val_annotations_file, sep='\t', header=None,names=['filename','label','A','B','C','D'])
    val_labels['label'] = val_labels['label'].map(class_to_idx)


    class ValDataset(Dataset):
        def __init__(self, labels, val_images_dir, transform=None):
            self.labels = labels
            self.val_images_dir = val_images_dir
            self.transform = transform

        def __len__(self):
            return len(self.labels)

        def __getitem__(self, idx):
            img_name = os.path.join(self.val_images_dir, self.labels.iloc[idx, 0])
            image = Image.open(img_name).convert("RGB") 
            label = self.labels.iloc[idx, 1]

            if self.transform:
                image = self.transform(image)

            return image, label

    val_dataset = ValDataset(val_labels, './data/tiny-imagenet-200/val/images', transform=transform)

    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return val_loader


def create_client_loaders_dirichlet(dataset, num_clients, alpha=0.5, batch_size=64,num_class=10):
    client_loaders = []
    data_indices = defaultdict(list)


    for idx, (_, label) in enumerate(dataset):
        data_indices[label].append(idx)
    np.random.seed(44)

    dirichlet_distribution = np.random.dirichlet([alpha] * num_clients,size=num_class)  # 10 是类别数
    np.random.seed()


    client_data_indices = [[] for _ in range(num_clients)]
    for category, proportions in enumerate(dirichlet_distribution):
        indices = data_indices[category]
        np.random.shuffle(indices)
        split_indices = np.split(indices, (proportions[:-1] * len(indices)).astype(int).cumsum())
        
        for client_id, client_indices in enumerate(split_indices):
            client_data_indices[client_id].extend(client_indices)
    
    for client_indices in client_data_indices:
        client_subset = Subset(dataset, client_indices)
        client_loader = DataLoader(client_subset, batch_size=batch_size, shuffle=True)
        client_loaders.append(client_loader)

    return client_loaders

def aggregate_weights(weights):
    total_weight = sum(len(w) for w in weights)
    average_weights = {}
    for key in weights[0].keys():
        average_weights[key] = sum(weights[i][key] * len(weights[i]) for i in range(len(weights))) / total_weight
    return average_weights