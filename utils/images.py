## Saving & Loading FashionMNIST Dataset from file

from typing import Optional
import torch

from torch.utils.data import Subset
from torchvision import datasets
from torchvision.transforms import ToTensor
from model import classes

sample_images_path = 'data/sample_images.pt'
full_set_images_path = 'data/whole_dataset.pt'

def save_dataset(*, sample=True):
    """By default saves sample dataset with the first five images"""
    SAMPLE_SZ = 5
    dataset = datasets.FashionMNIST(
        root='../main_intro_pytorch/data',
        train=False,
        download=True,
        transform=ToTensor(),
    )
    if sample:
        torch.save(Subset(dataset, range(SAMPLE_SZ)), sample_images_path)
        return 
    torch.save(dataset, full_set_images_path)

def load_dataset(*, sample=True):
    if sample:
        return classes, torch.load(sample_images_path)
    return classes, torch.load(full_set_images_path)

if __name__ == "__main__":
    save_dataset(sample=False)
    classes, ds = load_dataset()