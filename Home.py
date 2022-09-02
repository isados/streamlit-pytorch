from model import predict
import streamlit as st

import numpy as np
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader, Subset
from torchvision import datasets
from torchvision.transforms import ToTensor
import torchvision.transforms.functional as F
from utils import images, worst

CLASSES = [
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
]

@st.cache(allow_output_mutation=True)
def get_dataset(**kwargs):
    _, ds = images.load_dataset(**kwargs)
    return ds
        
def display_images():
    plt.rcParams["savefig.bbox"] = 'tight'

    imgs = get_dataset()
    fig, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    for i, (img,label) in enumerate(imgs):
        img = img.detach()
        img = F.to_pil_image(img)
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(title=CLASSES[label], xticklabels=[], yticklabels=[], xticks=[], yticks=[])
    st.pyplot(fig)
    return

def get_images():
    num_of_times_labelused = dict({})
    images = {}
    for img, label_index in get_dataset():
        num_of_times_labelused[label_index] = num_of_times_labelused.get(label_index, 0) + 1
        image_name = f"{CLASSES[label_index]}_{num_of_times_labelused[label_index]}"
        images[image_name] = {'image': img, 'class_index': label_index}
    return images

st.markdown('## Intro')
st.write('Here are examples of images from the Fashion MNIST dataset')
# Add images here
display_images()

# st.markdown('## Predict on Sample Images')
st.write("Below, you'll find our trained model we're using to make predictions.")
sample_images = get_images()
sel_image_name = st.selectbox(
    'Select an image to predict',
    sample_images.keys()
)
sel_image = sample_images[sel_image_name]
actual_label = sel_image['class_index']
img = sel_image['image']

pred, loss = predict(img, actual_label)
pred_is_correct = '✅' if pred==actual_label else '❌'
st.write(f"Actual: {CLASSES[actual_label]}")
st.write(f'Predicted: {CLASSES[pred]} {pred_is_correct}')
st.write(f'Loss: {loss}')

st.write("Now we'll look at some images that proved to be too difficult for our model to classify")

st.pyplot(worst.display_images(get_dataset(sample=False)))
