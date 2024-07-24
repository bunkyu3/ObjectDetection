import torch
import random
import numpy as np
from torch.utils.data import DataLoader, Subset, random_split


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def create_subset(full_dataset, subset_ratio):
    """データセットの一部を作成する関数"""
    subset_size = int(subset_ratio * len(full_dataset))
    indices = list(range(len(full_dataset)))
    subset_indices = indices[:subset_size]
    small_dataset = Subset(full_dataset, subset_indices)
    print("len(full_dataset): ", len(full_dataset))
    print("len(small_dataset): ", len(small_dataset))    
    return small_dataset


def split_dataset(dataset, train_ratio):
    """データセットを学習用と検証用に分割する関数"""
    train_size = int(train_ratio * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    print("len(train_dataset): ", len(train_dataset))
    print("len(val_dataset): ", len(val_dataset))
    return train_dataset, val_dataset