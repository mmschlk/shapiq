import numpy as np

def is_valid_image_shape(image: np.ndarray) -> bool:
    """Checks if the input image has a valid shape (H, W, C) where C is 1, 3."""
    if image.ndim != 3 or image.shape[2] not in (1, 3):
        return False
    return True
