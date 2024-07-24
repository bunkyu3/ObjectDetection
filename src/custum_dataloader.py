from torchvision.datasets import VOCDetection
from torch.utils.data import DataLoader
from torchvision import transforms
from custum_dataset import *
from utils import *


def collate_fn(batch):
    images = []
    targets = []
    for item in batch:
        images.append(item[0])
        targets.append(item[1])
    return images, targets


def batch_to_device(images, targets, device):
    processed_images = [image.to(device) for image in images]
    processed_targets = []
    for target in targets:
        processed_target = {}
        processed_target["boxes"] = target["boxes"].to(device)
        processed_target["labels"] = target["labels"].to(device)
        processed_targets.append(processed_target)    
    return processed_images, processed_targets

def create_train_val_dataloader(cfg):
    transform = transforms.Compose([
        transforms.Resize((100, 100)),
        transforms.ToTensor(),
    ])
    full_dataset = CustomVOCDetection(root='./raw_data', 
                                      year='2012', 
                                      image_set='train', 
                                      download=False, 
                                      transform=transform)
    small_dataset = create_subset(full_dataset, cfg.dataset.subset_size_ratio)
    train_dataset, val_dataset = split_dataset(small_dataset, cfg.dataset.train_size_ratio)
    train_loader = DataLoader(dataset=train_dataset, 
                              batch_size=cfg.train.batch_size, 
                              shuffle=True,
                              collate_fn=collate_fn)
    val_loader = DataLoader(dataset=val_dataset, 
                            batch_size=cfg.train.batch_size, 
                            shuffle=False, 
                            collate_fn=collate_fn)
    return train_loader, val_loader
