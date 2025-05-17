import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def cifar10_dataset(data_dir='./data'):
    #define transform
    transform = transforms.Compose([
        transforms.ToTensor(), #from [0, 255] -> [0,1]
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) #normalize the rgb img to (2*pixel - 1), [0,1] -> [-1,1]
    ])

    #load datasets
    train_dataset = datasets.CIFAR10(
        root=data_dir,
        train=True,
        transform= transform,
        download= True
        )
    
    test_dataset = datasets.CIFAR10(
        root=data_dir,
        train=False,
        transform= transform,
        download= True
        )
    
    return train_dataset, test_dataset


def cifar10_dataloader(train_dataset, test_dataset, batch_size=32, num_workers=2):
    #get dataloaders
    train_loader = DataLoader(
        dataset= train_dataset,
        batch_size= batch_size,
        shuffle= True,
        num_workers= num_workers
    )

    test_loader = DataLoader(
        dataset= test_dataset,
        batch_size= batch_size,
        shuffle= False,
        num_workers= num_workers
    )

    return train_loader, test_loader