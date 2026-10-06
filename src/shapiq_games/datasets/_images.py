"""ImageNet validation images used as examples for the vision games.

The images are downloaded on first use from a pinned commit of the shapiq repository and verified
by their SHA-256 checksum.
"""

from __future__ import annotations

import numpy as np
from PIL import Image

from ._cache import RemoteFile, fetch

__all__ = ["list_example_images", "load_example_image"]

_EXAMPLE_IMAGES: dict[str, str] = {
    "ILSVRC2012_val_00000014.JPEG": "3898d81fa0345b9f308106198c69b1d1d44ea8d20d67e9ff141186ea92aadf9b",
    "ILSVRC2012_val_00000048.JPEG": "8bb300ac4a10f0c08d359ee0342de48dcc0934476c0b7a36d4fa69f2d04cb9c4",
    "ILSVRC2012_val_00000115.JPEG": "fc876c9c7993b1ceac7c01029794a37c71f52d6ecbc17d42da3ca83044dbb85a",
    "ILSVRC2012_val_00000138.JPEG": "3f78b2d639012f75320ee139fe4d58689728ec43d5c228ad26f3dc74b7567d88",
    "ILSVRC2012_val_00000150.JPEG": "c0506c72689545abbb4308d3930ef5d5536e9c0b527a35ee78a42e0618e6ce90",
    "ILSVRC2012_val_00000154.JPEG": "ca229cb2cb6afb78015e76484adc69c64b7d1f79e3dc8f78d0f6584dd3910345",
    "ILSVRC2012_val_00000178.JPEG": "84cc03052f99608f7928492bcb1223ca9ee949d791e29fb3cb0432d32c5a9016",
    "ILSVRC2012_val_00000204.JPEG": "e396c9519f84ea709479b4e3ee978b2863227fae4b5aaf348d53831ae0f052ac",
    "ILSVRC2012_val_00000206.JPEG": "9a93c44e45d1f65b4cb8c04ec9bcbfdba48b26ddfbc73e1418bbc4d09f4e1df1",
    "ILSVRC2012_val_00000212.JPEG": "39d44e3638c3f18d814880d264a57e350083445d6e463d47ef20dfc0f6c098bb",
    "ILSVRC2012_val_00000220.JPEG": "64b00c59072b3735aa38a45f4c90dca91a517a66f86aa18c43404186044c57c8",
    "ILSVRC2012_val_00000232.JPEG": "9fa40aac071480d3b08e1a8f7c5f9775d13a653b052cd0dfe1e6aafee74a2092",
    "ILSVRC2012_val_00000242.JPEG": "9f82408eecb8b368e46b818257614e6869d7d405ab2bd173069851ba9d709c7e",
    "ILSVRC2012_val_00000253.JPEG": "94e2041834b4a689933fddc7189fc00fb4377fef9dae97704c77dbe28f8e0252",
    "ILSVRC2012_val_00000270.JPEG": "9013f22e2039f1df020fb794971d8561c29f85c041ae3e7fa58d47586330c222",
    "ILSVRC2012_val_00000286.JPEG": "01d88534b224bde312dd04545e5d64edd96e0fecfc6d57e174c3af7460c7a373",
    "ILSVRC2012_val_00000294.JPEG": "6fd784e43a97e5e81adcb4bdf0f25d8a373c2739f56f5c9ba28d8445752ff175",
    "ILSVRC2012_val_00000299.JPEG": "c51587914ec2425000080ebe192e48c7f2da2b715341b5afab52f560c2b225c2",
    "ILSVRC2012_val_00000325.JPEG": "5c5f28a00eed3e41829e5000d7229b094a28ada245feacedc8dd5e174d053db3",
    "ILSVRC2012_val_00000330.JPEG": "2661794555ce02cf0fe59fe4eda55f47031d12c5db4b33c224f8219b970627b8",
    "ILSVRC2012_val_00000343.JPEG": "a6dbb88a474db77369f048ce6bdab3a9356590beef0039355bc0faf308ae15e4",
    "ILSVRC2012_val_00000356.JPEG": "83987c096d43e56d4241eabb83cc83abd930b6a59e7e522a5384a4846c6bf06c",
    "ILSVRC2012_val_00000367.JPEG": "e8b75e4558d56dfcc34e392c47b506fe17428fe689904a52891493eb436446a9",
    "ILSVRC2012_val_00000994.jpg": "8dd77d156ddd19a3c767cee1526d50d617f6c08a229d1296d65bea82a6600582",
    "ILSVRC2012_val_00001143.JPEG": "5eabd605a2dc718e7fcac372130a9a7c59d35ce9d70b408fe047e706ec0cb0cb",
    "ILSVRC2012_val_00001915.JPEG": "f65f173358cc3a83d1e7cc88578a13a0bf5028e213e96910fbbf072bc757a1a4",
    "ILSVRC2012_val_00002541.JPEG": "4fcd9c7716e7e82c32a9c33b2a2f72fc983c3f6a496c54a51a9be324d16ba9d9",
    "ILSVRC2012_val_00005815.JPEG": "038af929bd4a0df2ef08839d9826010feab56c0c1ed47a4329a1156ee1f64f74",
    "ILSVRC2012_val_00010860.JPEG": "5d79e1650f250d2515f3bcf9fdbca595b7b20c8c5b662450f0ef9e24a5a87f99",
    "ILSVRC2012_val_00010863.JPEG": "1293577ea2d4c12d9982e6ea7eed6796d890031cf2f4633bc735a009658dab55",
    "ILSVRC2012_val_00028489.JPEG": "12796e2bac81a9589fbc913d4b92fd9f262e9c202a6b73653d576e0871374eaa",
}


def list_example_images() -> list[str]:
    """Return the file names of the available example images, sorted."""
    return sorted(_EXAMPLE_IMAGES)


def load_example_image(image: int | str = 0) -> np.ndarray:
    """Load an ImageNet example image as an RGB array.

    Args:
        image: The position in :func:`list_example_images` or the file name. Defaults to ``0``.

    Returns:
        The image as a ``uint8`` array of shape ``(height, width, 3)``.

    Raises:
        ValueError: If the image is unknown.
    """
    names = list_example_images()
    if isinstance(image, int):
        if not 0 <= image < len(names):
            msg = f"image={image} is out of range for {len(names)} example images."
            raise ValueError(msg)
        name = names[image]
    elif image in _EXAMPLE_IMAGES:
        name = image
    else:
        msg = f"Unknown example image '{image}'. Available images: {', '.join(names)}."
        raise ValueError(msg)
    remote = RemoteFile.pinned(
        f"benchmark/imagenet_examples/{name}", _EXAMPLE_IMAGES[name], subdir="images"
    )
    with Image.open(fetch(remote)) as img:
        return np.asarray(img.convert("RGB"))
