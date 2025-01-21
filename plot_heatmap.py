"""This script plots a heatmap of SV as a heatmap onto a given image."""

import os

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

import shapiq
from shapiq.plot._config import BLUE, RED

PLOT_DIR = os.path.join("results", "examples", "plots")


def value_to_color(value: float) -> tuple[float, float, float]:
    """Converts a value to a color."""
    color = RED
    if value < 0:
        color = BLUE
    return color.get_red(), color.get_green(), color.get_blue()


def own_resize(img: np.ndarray, multiples: int = 12) -> np.ndarray:
    """Resizes the image to the given size by repeating the pixels in both directions:

    4x4 -> 8x8
    """
    # get the shape of the image
    height, width, n_channels = img.shape
    assert height == width, "The image must be square."

    # get the new size
    new_size = height * multiples

    # create the new image
    new_img = np.zeros((new_size, new_size, n_channels))

    # fill the new image with the pixels of the old image
    for i in range(new_size):
        for j in range(new_size):
            new_img[i, j] = img[i // multiples, j // multiples]

    return new_img


def plot_heatmap(img: Image.Image, sv: shapiq.InteractionValues) -> None:
    """Plots the Shapley values as a heatmap onto the image."""
    n_patches_per_row = 12
    patch_size = 32
    n_patches = n_patches_per_row**2

    # img = feature_extractor(images=img, return_tensors="np")["pixel_values"][0]
    # img = np.moveaxis(img, 0, -1)  # shape is 3, 384, 384 and must be reshaped to 384, 384, 3

    # make img into a square
    img = img.resize((384, 384))

    all_values_not_baseline = np.array([sv.values[(i,)] for i in range(n_patches)])
    max_abs_value = float(np.max(np.abs(all_values_not_baseline)))

    # create sv "image" for each patch
    sv_image = np.zeros((n_patches_per_row, n_patches_per_row, 4))
    for i in range(n_patches):
        row = i // n_patches_per_row
        column = i % n_patches_per_row

        # get the value of the Shapley value
        sv_patch = float(sv[(i,)])
        sv_abs_patch = abs(sv_patch)

        # get the color of the Shapley value
        color_patch = value_to_color(sv_patch)
        # get numeric values from Colour object
        alpha = sv_abs_patch / max_abs_value  # alpha is the intensity of the color
        color = (*color_patch, alpha)
        sv_image[row, column] = color

    sv_image = own_resize(sv_image, multiples=patch_size)
    sv_image = Image.fromarray((sv_image * 255).astype(np.uint8))

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(img, alpha=0.9)
    ax.imshow(sv_image)
    ax.axis("off")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":

    os.makedirs(PLOT_DIR, exist_ok=True)

    image_path = os.path.join("images", "dog_example.jpg")
    image = Image.open(image_path)

    # Load the Shapley values
    # sv_name = "dog_example_budget=1000000_seed=42.npz"

    sv_name = "dog_example_plectrum_budget=1000000_seed=42.npz"
    sv_path = os.path.join("results", "examples", "interaction_values", sv_name)
    shapley_values = shapiq.InteractionValues.load(path=sv_path)

    # Plot the heatmap
    plot_heatmap(image, shapley_values)
