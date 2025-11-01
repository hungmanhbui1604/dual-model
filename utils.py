import os
import torch
from torch.utils.data import Dataset
from PIL import Image


class IntraSensorBinaryDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.samples = []
        
        live_dir = os.path.join(data_dir, 'Live')
        for filename in os.listdir(live_dir):
            if filename.lower().endswith(('.png', '.bmp')):
                self.samples.append((os.path.join(live_dir, filename), 0))
        
        spoof_dir = os.path.join(data_dir, 'Spoof')
        for spoof_subdir in os.listdir(spoof_dir):
            spoof_subdir_path = os.path.join(spoof_dir, spoof_subdir)
            if os.path.isdir(spoof_subdir_path):
                for filename in os.listdir(spoof_subdir_path):
                    if filename.lower().endswith(('.png', '.bmp')):
                        self.samples.append((os.path.join(spoof_subdir_path, filename), 1))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, label


def get_train_validation_split(dataset, validation_split):    
    total_size = len(dataset)
    validation_size = int(validation_split * total_size)
    train_size = total_size - validation_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, 
        [train_size, validation_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    return train_dataset, val_dataset


class IntraSensorMulticlassDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.samples = []
        live_dir = os.path.join(data_dir, 'Live')
        spoof_dir = os.path.join(data_dir, 'Spoof')
        self.num_classes = len(os.listdir(spoof_dir)) + 1

        for filename in os.listdir(live_dir):
            if filename.lower().endswith(('.png', '.bmp')):
                self.samples.append((os.path.join(live_dir, filename), 0))
        
        for i, spoof_subdir in enumerate(os.listdir(spoof_dir), start=1):
            spoof_subdir_path = os.path.join(spoof_dir, spoof_subdir)
            if os.path.isdir(spoof_subdir_path):
                for filename in os.listdir(spoof_subdir_path):
                    if filename.lower().endswith(('.png', '.bmp')):
                        self.samples.append((os.path.join(spoof_subdir_path, filename), i))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, label