## Saving & Loading FashionMNIST Dataset from file

from typing import Optional
import torch

from torch.utils.data import Subset
from torchvision import datasets
from torchvision.transforms import ToTensor
from model import classes

full_set_images_path = 'data/whole_dataset.pt'

def save_dataset():
    """By default saves sample dataset with the first five images"""
    dataset = datasets.FashionMNIST(
        root='../main_intro_pytorch/data',
        train=False,
        download=True,
        transform=ToTensor(),
    )
    torch.save(dataset, full_set_images_path)

def load_dataset(*, sample=True):
    SAMPLE_SZ = 5
    ds = torch.load(full_set_images_path)
    if sample:
        ds = Subset(ds, range(SAMPLE_SZ))
    return classes, ds

if __name__ == "__main__":
    # save_dataset(sample=False)
    classes, ds = load_dataset()