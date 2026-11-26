"""FIxLIP games: image patches and text tokens of CLIP and SigLIP, removed from the inputs."""

import jax.numpy as jnp
import numpy as np
import pytest

from shapiq.players import FinitePlayers

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
from shapiq.xai.fixlip import FIxLIPGame  # noqa: E402

LAYERS = {
    "hidden_size": 32,
    "intermediate_size": 37,
    "num_attention_heads": 4,
    "num_hidden_layers": 2,
}
TEXT = {
    **LAYERS,
    "vocab_size": 99,
    "max_position_embeddings": 16,
    "bos_token_id": 0,
    "eos_token_id": 98,
}
VISION = {**LAYERS, "image_size": 32, "patch_size": 8}  # a 4 x 4 grid of patches
N_PATCHES = 16


def tiny_model(kind):
    """Return a small random CLIP or SigLIP model."""
    torch.manual_seed(0)
    if kind == "clip":
        config = transformers.CLIPConfig(text_config=TEXT, vision_config=VISION, projection_dim=16)
        return transformers.CLIPModel(config).eval()
    config = transformers.SiglipConfig(text_config=TEXT, vision_config=VISION)
    return transformers.SiglipModel(config).eval()


def expected(model, pixel_values, input_ids, first, coalitions):
    """Return ``logits_per_image`` of every coalition's masked image and text, one at a time."""
    n_tokens = coalitions.shape[-1] - N_PATCHES
    values = []
    for coalition in coalitions:
        patches = torch.as_tensor(coalition[:N_PATCHES]).reshape(4, 4)
        mask = patches.repeat_interleave(8, 0).repeat_interleave(8, 1)
        attention = torch.ones_like(input_ids)
        attention[:, first : first + n_tokens] = torch.as_tensor(coalition[N_PATCHES:])
        with torch.no_grad():
            outputs = model(
                pixel_values=pixel_values * mask, input_ids=input_ids, attention_mask=attention
            )
        values.append(float(outputs.logits_per_image[0, 0]))
    return np.array(values)


@pytest.mark.parametrize(("kind", "first"), [("clip", 1), ("siglip", 0)])
def test_values_are_masked_logits(kind, first):
    """Is the value the model's logit with absent patches zeroed and absent tokens unattended?"""
    model = tiny_model(kind)
    torch.manual_seed(1)
    pixel_values = torch.randn(1, 3, 32, 32)
    input_ids = torch.tensor([[0, 5, 17, 23, 42, 7, 98]])
    n_tokens = 5 if kind == "clip" else 6  # CLIP explains the tokens between start and end
    game = FIxLIPGame(
        model, FinitePlayers(N_PATCHES + n_tokens), pixel_values, input_ids, first, batch_size=3
    )
    coalitions = np.random.default_rng(0).random((7, game.n_players)) < 0.5
    coalitions[0] = True  # the full coalition
    values = game(jnp.asarray(coalitions.reshape(7, 1, -1)))
    assert game.dtype == values.dtype == jnp.float32
    assert not game.weak_type
    assert values.shape == (7, 1)
    want = expected(model, pixel_values, input_ids, first, coalitions)
    np.testing.assert_allclose(values[:, 0], want, rtol=1e-4, atol=1e-5)


def test_from_processor():
    """Does a processor's output give one player per patch and per token between CLIP's ends?"""
    model = tiny_model("clip")

    def processor(images, text, padding, return_tensors):
        return {
            "pixel_values": torch.zeros(1, 3, 32, 32),
            "input_ids": torch.tensor([[0, 5, 6, 98]]),
        }

    game = FIxLIPGame.from_processor(model, processor, image=None, text="a cat")
    assert game.n_players == N_PATCHES + 2
    assert game.first_token == 1
