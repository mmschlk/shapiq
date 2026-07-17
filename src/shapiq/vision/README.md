# 🏞️ Vision for shapiq - Starter README

Explain image classifiers with Shapley values and Shapley interactions.
The idea is the same as everywhere else in `shapiq`: split the input into players, hide subsets of them, and watch what the prediction does.
For images a player is a region, either a group of pixels for CNNs or a group of patch tokens for Vision Transformers.

## Install

The vision code needs `torch` and `scikit-image`, which are not part of the base install:

```sh
pip install shapiq[vision]
```

## Quickstart Example

```python
import numpy as np
from PIL import Image
from torchvision.models import resnet18
from shapiq.vision import ClassificationArchitecture, ImageExplainer

image = np.asarray(Image.open("your_image.png").convert("RGB"))
model = resnet18(weights="IMAGENET1K_V1").eval()
architecture = ClassificationArchitecture(model=model)
explainer = ImageExplainer(model=architecture, data=image, index="k-SII", max_order=2)
values = explainer.explain(budget=256)
values.plot_image_attributions(image=image, player_masks=explainer.imputer.player_masks)
```

`data` can be a PIL image, an `(H, W, C)` numpy array, or a torch tensor.
Pass one image, not a batch.
The plotting helpers are stricter than the explainer and want a numpy array, so it is easiest to convert once up front.

`ClassificationArchitecture` is the standard architecture for CNNs.
As above, it can be called with just a model and no `processor`, in which case masking happens on the raw pixels you pass in.

By default the image is cut into roughly 10 SLIC superpixels and absent regions are filled with the mean colour.
The explained class defaults to whatever the model predicts, so pass `class_index=...` if you want a different one.

## Vision Transformers

ViTs get their own architecture, because masking happens in token space rather than on pixels:

```python
from transformers import ViTForImageClassification, ViTImageProcessor
from shapiq.vision import ImageExplainer, ViTClassificationArchitecture

name = "google/vit-base-patch32-384"
model = ViTForImageClassification.from_pretrained(name).eval()
architecture = ViTClassificationArchitecture(
    model=model,
    vit_processor=ViTImageProcessor.from_pretrained(name),
)
explainer = ImageExplainer(model=architecture, data=image)
```

Players are patch groups derived from the model's token grid, and absent players are hidden with the model's `mask_token`.
This needs a model that accepts `bool_masked_pos`, which only some Hugging Face ViTs do.
Plenty of ViT-family models are missing `bool_masked_pos`, `hidden_size` or `patch_size` altogether, and some of the rest return outputs we cannot read a class score of, so the token path is narrower than the name "ViT" suggests.

`ViTClassificationArchitecture` currently only supports models that expose `bool_masked_pos`, `hidden_size` and `patch_size` in their Hugging Face config, because the masking and player strategies are built directly on top of these.
Models that miss any of them (no `vit.embeddings.mask_token`, or no ViT-style `image_size`/`patch_size` config) have to go through `ClassificationArchitecture` instead.
See the compatibility table below for what that means in practice.


## Things worth knowing

**Players and masking have to agree on a domain.**
Pixel-space players (`SuperpixelStrategy`, `GridStrategy`, `CustomPlayerStrategy`) go with pixel-space masking (`MeanColorMasking`, `ZeroMasking`), and token-space players (`PatchStrategy`) go with token-space masking (`MaskTokenStrategy`, `BoolMaskedPosStrategy`).
Mixing them raises a `TypeError` when you build the architecture rather than producing quiet nonsense later.

**Models without `bool_masked_pos` still work.**
Swin, BEiT, and friends can go through `ClassificationArchitecture` with their processor attached, which masks pixels before preprocessing:

```python
ClassificationArchitecture(model=model, processor=processor)
```

You lose token-space masking that way, but you keep the explanation.

**The two architectures return different scales.**
`ClassificationArchitecture` reports the raw logit of the explained class, `ViTClassificationArchitecture` reports the softmax probability.
Don't compare the numbers across the two without keeping that in mind.

**SLIC does not always return the number of segments you asked for.**
`n_players` reflects what you actually got, so read it off the imputer rather than assuming.

## Examples

Runnable versions of all of this live in `examples/vision`, including custom player layouts and the interaction network plots.


## Model compatibility
We report the models we have tested with the two architectures below:

| Model | Type | Pixel path (`ClassificationArchitecture`) | Token path (`ViTClassificationArchitecture`) |
|---|---|---|---|
| `resnet18` | torchvision CNN | ✓ | — |
| `resnet50` | torchvision CNN | ✓ | — |
| `wide_resnet50_2` | torchvision CNN | ✓ | — |
| `resnext50_32x4d` | torchvision CNN | ✓ | — |
| `densenet121` | torchvision CNN | ✓ | — |
| `vgg16` | torchvision CNN | ✓ | — |
| `alexnet` | torchvision CNN | ✓ | — |
| `googlenet` | torchvision CNN | ✓ | — |
| `inception_v3` | torchvision CNN | ✓ | — |
| `mnasnet1_0` | torchvision CNN | ✓ | — |
| `mobilenet_v3_small` | torchvision CNN | ✓ | — |
| `efficientnet_b0` | torchvision CNN | ✓ | — |
| `squeezenet1_0` | torchvision CNN | ✓ | — |
| `shufflenet_v2_x0_5` | torchvision CNN | ✓ | — |
| `regnet_y_400mf` | torchvision CNN | ✓ | — |
| `convnext_tiny` | torchvision CNN | ✓ | — |
| `vit_b_16` | torchvision ViT | ✓ | — |
| `vit_b_32` | torchvision ViT | ✓ | — |
| `swin_t` | torchvision ViT | ✓ | — |
| `swin_v2_t` | torchvision ViT | ✓ | — |
| `maxvit_t` | torchvision ViT | ✓ | — |
| `microsoft/resnet-50` | HF CNN | ✓ | ✗ no ViT config (image_size/patch_size) |
| `facebook/convnext-tiny-224` | HF CNN | ✓ | ✗ no vit.embeddings.mask_token |
| `facebook/convnextv2-tiny-22k-224` | HF CNN | ✓ | ✗ no vit.embeddings.mask_token |
| `google/mobilenet_v2_1.0_224` | HF CNN | ✓ | ✗ no ViT config (image_size/patch_size) |
| `facebook/regnet-y-040` | HF CNN | ✓ | ✗ no ViT config (image_size/patch_size) |
| `google/vit-base-patch16-224` | HF ViT-family | ✓ | ✓ |
| `facebook/deit-base-patch16-224` | HF ViT-family | ✓ | ✓ |
| `facebook/deit-base-distilled-patch16-224` | HF ViT-family | ✓ | ✗ no vit.embeddings.mask_token |
| `microsoft/beit-base-patch16-224-pt22k-ft22k` | HF ViT-family | ✓ | ✗ no vit.embeddings.mask_token |
| `microsoft/swin-tiny-patch4-window7-224` | HF ViT-family | ✓ | ✗ no vit.embeddings.mask_token |
| `microsoft/swinv2-tiny-patch4-window8-256` | HF ViT-family | ✓ | ✗ no vit.embeddings.mask_token |
| `apple/mobilevit-small` | HF ViT-family | ✓ | ✗ no vit.embeddings.mask_token |
| `facebook/levit-128S` | HF ViT-family | ✓ | ✗ no vit.embeddings.mask_token |
| `microsoft/cvt-13` | HF ViT-family | ✓ | ✗ no ViT config (image_size/patch_size) |
| `microsoft/focalnet-tiny` | HF ViT-family | ✓ | ✗ no vit.embeddings.mask_token |
| `facebook/dinov2-small-imagenet1k-1-layer` | HF ViT-family | ✓ | ✗ no vit.embeddings.mask_token |
| `nvidia/mit-b0` | HF ViT-family | ✓ | ✗ no ViT config (image_size/patch_size) |
| `google/vit-base-patch16-224-in21k` | HF masked-image ViT | ✗ no classification logits | ✗ no classification logits |
| `facebook/vit-mae-base` | HF encoder-only | ✗ no classification logits | ✗ no vit.embeddings.mask_token |
| `openai/clip-vit-base-patch32` | HF encoder-only | ✗ needs text input | ✗ no ViT config (image_size/patch_size) |
| `nvidia/segformer-b0-finetuned-ade-512-512` | HF dense-prediction | ✗ dense-prediction output | ✗ no ViT config (image_size/patch_size) |
| `facebook/detr-resnet-50` | HF dense-prediction | ✗ dense-prediction output | ✗ no ViT config (image_size/patch_size) |
