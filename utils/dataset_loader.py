import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

def split_train_val(data, val_ratio=0.1):
    total_len = len(data)
    val_len = int(total_len*val_ratio)
    train_len = total_len - val_len

    train_set, val_set = random_split(data, [train_len, val_len], generator=torch.Generator().manual_seed(42))

    return train_set, val_set

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


def cifar10_dataloader(train_dataset, test_dataset, val_ratio=0.1, batch_size=32, num_workers=2):
    train_dataset, val_dataset = split_train_val(train_dataset, val_ratio=val_ratio)

    #get dataloaders
    train_loader = DataLoader(
        dataset= train_dataset,
        batch_size= batch_size,
        shuffle= True,
        num_workers= num_workers
    )

    val_dataloader = DataLoader(
        dataset=val_dataset,
        batch_size= batch_size,
        shuffle= False,
        num_workers= num_workers
    )

    test_loader = DataLoader(
        dataset= test_dataset,
        batch_size= batch_size,
        shuffle= False,
        num_workers= num_workers
    )

    return train_loader, test_loader, val_dataloader