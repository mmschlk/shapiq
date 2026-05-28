from __future__ import annotations

import numpy as np
import torch
from PIL import Image
from torchvision.models import ResNet18_Weights, resnet18

from shapiq.approximator.proxy.proxyshap import ProxySHAP
from shapiq_benchmark.image_bench import ImageBench
from shapiq_benchmark.metrics import get_all_metrics

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

weights = ResNet18_Weights.DEFAULT
model = resnet18(weights=weights).to(device).eval()
preprocess = weights.transforms()


def my_model(image: np.ndarray) -> np.ndarray:
    # image: HWC uint8
    img = Image.fromarray(image)
    tensor = preprocess(img).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=-1)
    return probs.squeeze(0).cpu().numpy()


print("Testing custom ResNet model with 14 superpixels...")
bench = ImageBench(
    data="tests/shapiq/data",  # folder or single image path
    model=my_model,
    x_explain=0,
)

values = bench.exact_values(index="SII", order=2)
print(values)

game = bench.game
print(type(game))
approximator = ProxySHAP(n=game.n_players, random_state=42)
approx_values = approximator.approximate(game=game, budget=1000)

metrics = get_all_metrics(
    values,
    approx_values,
    game,
    #save_path="C:/Users/isabe/OneDrive/Dokumente/Uni/LMU/Semester 10/shapiq/src/shapiq_benchmark/tests/results_image.json",
)
print(metrics)

print("Testing ViT model with 16 superpixels...")
bench = ImageBench(
    data="tests/shapiq/data",
    model="vit_16_patches",
    x_explain=0,
)

values = bench.exact_values(index="SII", order=2)
print(values)

game = bench.game
print(type(game))

approximator = ProxySHAP(n=game.n_players, random_state=42)
approx_values = approximator.approximate(game=game, budget=1000)

metrics = get_all_metrics(values, approx_values, game)
print(metrics)


print("Testing ResNet model with 14 superpixels...")
bench = ImageBench(
    data="tests/shapiq/data",
    model="resnet_18",
    x_explain=0,
)

values = bench.exact_values(index="SII", order=2)
print(values)

game = bench.game
print(type(game))

approximator = ProxySHAP(n=game.n_players, random_state=42)
approx_values = approximator.approximate(game=game, budget=1000)

metrics = get_all_metrics(values, approx_values, game)
print(metrics)
