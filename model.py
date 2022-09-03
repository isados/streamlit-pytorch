import torch
import os
import streamlit as st
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from torch import nn


model_path = './models/model_flatten_li784x512_relu_(li512x512_relu)x2_li512x10_lr0.01.pth'

# Build the Model with 
class NN(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
        nn.Linear(28*28, 512), #since size of image is 28 x 28, and we are aiming for 512 neurons for the first layer
        nn.ReLU(),
        nn.Linear(512, 512),
        nn.ReLU(),
        nn.Linear(512, 512),
        nn.ReLU(),
        nn.Linear(512, 10),
        )
    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits
    def summary(self):
        return 'flatten_li784x512_relu_(li512x512_relu)x2_li512x10'


classes = (
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
)

def predict(X, y=None):
    global model_path
    model = NN()

    if not os.path.exists(model_path):
        return 'Opps! No model found :('

    model.load_state_dict(torch.load(model_path))
    model.eval()

    loss_func = nn.CrossEntropyLoss()
    pred = model(X)
    pred_label = pred.argmax(1).item()
    if y:
        y = (torch.tensor([y]))
        loss = loss_func(pred, y).item()
        return pred_label, loss
    return pred_label

def get_model():
    model = NN()
    model.load_state_dict(torch.load(model_path))
    return model

    




