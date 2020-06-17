from torchvision import models
import torch
import matplotlib.pyplot as plt
from src.helper import (get_device, load_image, convert_im_to_numpy_im,
                        display_image, extract_features, calculate_gram_matrix)
from src import constants
from torch import optim
import os


def main():

    CONTENT_IMAGE_PATH = os.path.join(constants.IMAGE_DIR,
                                      constants.CONTENT_IMAGE_NAME)
    STYLE_IMAGE_PATH = os.path.join(constants.IMAGE_DIR,
                                    constants.STYLE_IMAGE_NAME)
    DISPLAY_INTERVAL = constants.DISPLAY_INTERVAL

    # As we are interested in getting content & style
    # We only need convolutional stack not FC layers.
    # so here fetaures will give us all pollimng and convolution layers
    model = models.vgg19(True).features

    # as we are not going to train we can freeze the params
    for param in model.parameters():
        # in pytorch '_' in last means in place operation True
        # means it'll change the original tensor
        param.requires_grad_(False)

    # Using cuda as device if available or go with CPU
    model.to(get_device())

    # Load content image
    content_image = load_image(CONTENT_IMAGE_PATH).to(get_device())
    print(f"shape of content image {content_image.size()}")

    # Load Style image
    style_image = load_image(STYLE_IMAGE_PATH,
                             shape=content_image.shape[-2:]).to(get_device())
    print(f"shape of style image {style_image.size()}")

    # We need to create our target image as well
    # In which content will be from content image and style will be from style

    target_image = content_image.clone().requires_grad_(True).to(get_device())
    # TODO:Please uncomment this in case we need to display the image
    # image_for_plot = convert_im_to_numpy_im([content_image, style_image])

    # claculate the content and style features
    content_features: dict = extract_features(content_image, model)
    style_features: dict = extract_features(style_image, model)

    # calculate gram matrices for all style features
    style_grams: dict = {
        layer: calculate_gram_matrix(style_features[layer])
        for layer in style_features
    }

    style_weights = {
        'conv1_1': 1.,
        'conv2_1': 0.8,
        'conv3_1': 0.5,
        'conv4_1': 0.3,
        'conv5_1': 0.1
    }

    content_weight = 1  # alpha
    style_weight = 1e6  # beta

    optimizer = optim.Adam([target_image])
    STEPS = constants.STEPS
    for i in range(1, STEPS + 1):
        target_features: dict = extract_features(target_image, model)
        content_loss = torch.mean(
            (target_features['conv4_2'] - content_features['conv4_2'])**2)
        style_loss: float = float(0)
        for layer in style_weights:
            target_feature = target_features[layer]
            target_gram = calculate_gram_matrix(target_feature)
            _, d, h, w = target_feature.shape
            style_gram = style_grams[layer]
            layer_style_loss = style_weights[layer] * torch.mean(
                (target_gram - style_gram)**2)
            style_loss += layer_style_loss / (d * h * w)
        print("Iteration index: ", i)
        total_loss: float = (content_weight * content_loss +
                             style_weight * style_loss)
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        if i % DISPLAY_INTERVAL == 0:
            print('Total loss: ', total_loss.item())
            plt.imshow(convert_im_to_numpy_im([target_image])[0])
            plt.show()


if __name__ == '__main__':
    main()
