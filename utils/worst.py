import torch
import torchvision.transforms.functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import streamlit as st
import numpy as np
from model import NN, get_model, classes
from torch import nn


# confident in wrong answers
def get_top10_images(ds):
    model = get_model()
    model.eval()
    # No reduction or summing for losses
    loss_func = nn.CrossEntropyLoss(reduce=False)
    dl = DataLoader(ds, batch_size=64)

    incorrect_classified = []
    for batch_X, batch_y in dl:
        with torch.no_grad():
            batch_preds = model(batch_X)
            batch_loss = loss_func(batch_preds, batch_y)
            batch_preds = batch_preds.argmax(1)
            for loss, pred, x, y in zip(batch_loss, batch_preds, batch_X, batch_y):
                if pred != y:
                    incorrect_classified.append((x, loss, pred, y))
            # break 

    sort_by_loss = lambda x: x[1]
    incorrect_classified.sort(key=sort_by_loss, reverse=True)
    return incorrect_classified[:10]

@st.cache(allow_output_mutation=True)
def display_images(ds):
    imgs = get_top10_images(ds)
    rows,cols = 2, len(imgs)//2
    fig, axs = plt.subplots(ncols=len(imgs)//2, nrows=2)
    for r in range(rows):
        for c in range(cols):
            index = r*cols + c
            if index >= len(imgs):
                return fig
            img,loss, pred, y = imgs[index]
            img = img.detach()
            img = F.to_pil_image(img)
            axs[r, c].imshow(np.asarray(img))
            title = f"Loss {round(loss.item(), 2)}\nP: {classes[pred]}\nA: {classes[y]}"
            axs[r, c].set(title=title, xticklabels=[], yticklabels=[], xticks=[], yticks=[])
    return fig


if __name__ == '__main__':
    pass
