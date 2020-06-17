import torch
from PIL import Image
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt


def get_device():
    device = ("cuda" if torch.cuda.is_available() else "cpu")
    return device


def load_image(image_path: str,
               max_size: int = 400,
               shape: list = None) -> Image:
    image = Image.open(image_path).convert('RGB')
    print(f"Image shape before applying transform: {image.size}")
    size = max(image.size)
    if max(image.size) > max_size:
        size = max_size
    if shape is not None:
        size = shape
    transform_image = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    image = transform_image(image)[:3, :, :].unsqueeze(0)
    print(f"Image shape after applying transform: {image.size()}")
    return image


def convert_im_to_numpy_im(tensor_list: list):
    numpy_image_list: list = []
    for tensor in tensor_list:
        image = tensor.to("cpu").clone().detach()
        image = image.numpy().squeeze()
        image = image.transpose(1, 2, 0)
        image = image * np.array((0.5, 0.5, 0.5)) + np.array((0.5, 0.5, 0.5))
        image = image.clip(0, 1)
        print(f"image shape: {image.shape}")
        numpy_image_list.append(image)
    return numpy_image_list


def display_image(*images):  # ['a', 'b']
    fig = plt.figure()
    axes: list = []

    image_list = [image for image in images[0]]
    for i in range(len(image_list)):
        axes.append(fig.add_subplot(len(image_list), 1, i + 1))
        plt.imshow(image_list[i])
    plt.show()


def extract_features(image, model, layers=None):

    if layers is None:
        # we need only conv 1st conv layers from each stack
        # to get gram matrices
        # and one layer for content representation
        layers = {
            '0': 'conv1_1',
            '5': 'conv2_1',
            '10': 'conv3_1',
            '19': 'conv4_1',
            '21': 'conv4_2',  # content representation
            '28': 'conv5_1'
        }
    features: dict = {}
    x: torch.tensor = image

    # pass the image from the layer to get the feature
    for name, layer in model._modules.items():
        x = layer(x)
        if name in layers:
            features[layers[name]] = x
    return features


def calculate_gram_matrix(tensor):
    _, d, h, w = tensor.shape
    tensor = tensor.view(d, h * w)
    gram_matrix = torch.mm(tensor, tensor.t())
    return gram_matrix
