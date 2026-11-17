"""Explain the demo's photos for CLIP with the three baselines of FIxLIP's Figure 2.

The page's figure on the methods (``methods`` in ``../assets/figures.js``) shows, for the photo
chosen in the demo, the explanations of CLIP (ViT-B/32, 7 x 7 patches) by GAME, Grad-ECLIP, exCLIP
and FIxLIP. FIxLIP's explanation is the demo's (``../assets/data.js``). This script computes the
other three, with the settings that FIxLIP's repository used for the paper
(github.com/hbaniecki/fixlip, folders ``gradeclip`` and ``exclip``):

- GAME (Chefer et al., 2021): the relevance of the last attention block of each encoder, from the
  gradient of CLIP's logit on the attention maps. A value per patch and per token, none negative.
- Grad-ECLIP (Zhao et al., 2024): the gradient of the cosine similarity on the output of the last
  attention block of the vision encoder, and of the last eight blocks of the text encoder, weighted
  by the values and by the similarity of the queries and the keys. A value per patch and per token,
  none negative.
- exCLIP (Moeller et al., 2025): the cross-modal interactions of integrated gradients on the input
  of the last block of both encoders, 20 steps from a reference (the image of the normalization
  mean, an empty text). A value per patch and token.

The functions take the Hugging Face ``CLIPModel``, the pixels and the token ids, and run the
encoders block by block, so that they can see and change what the methods need. The pixels are
those the demo's explanation saw (``explain.py``, through ``shapiq_games``), and the caption is
tokenized as in ``explain.py``, so the tokens are those of ``data.js``. As in FIxLIP's repository
(OpenAI's ``clip.tokenize``), the ids are padded with zeros to CLIP's 77 positions.

Usage::

    python baselines.py

The script writes the part ``baselines`` of ``../assets/figures-data.js``. It takes about 45
seconds for each photo on 4 CPU cores, most of it for exCLIP.
"""

from __future__ import annotations

import time
import tomllib
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from explain import Network, caption_tokens, load_photo
from pack import read_data, read_figures, write_figures

from shapiq_games.vision import _image_text_model as image_text

if TYPE_CHECKING:
    from collections.abc import Callable

    from transformers import CLIPModel

HERE = Path(__file__).resolve().parent
MODEL = "clip"
CONTEXT = 77  # CLIP's text positions, padded with the id 0 as by OpenAI's clip.tokenize
START, END = 49406, 49407  # <|startoftext|> and <|endoftext|>
GRADECLIP_TEXT_BLOCKS = 8  # the last blocks of the text encoder that Grad-ECLIP reads
EXCLIP_STEPS = 20  # the steps of exCLIP's integrated gradients
EXCLIP_BLOCK = 11  # the block of both encoders whose input exCLIP interpolates
EXCLIP_REFERENCE_TOKENS = 3  # the reference text: <|startoftext|>, two zeros, <|endoftext|>
DIGITS = 3  # significant digits kept in figures-data.js


# -- CLIP, block by block ----------------------------------------------------------------------------


def block(
    layer: torch.nn.Module,
    hidden: torch.Tensor,
    heads: int,
    *,
    causal: bool = False,
    keep: dict[str, Any] | None = None,
) -> torch.Tensor:
    """Run one block of a CLIP encoder (a Hugging Face ``CLIPEncoderLayer``) on ``hidden``.

    The block is CLIP's: attention with ``heads`` heads and an MLP, each after a layer norm and
    added to its input. With ``keep``, it holds the parts that the methods read: the queries, keys
    and values (``q``, ``k``, ``v``, before they are split into heads), the attention maps
    (``attention``, one per head) and the output of the attention before its output projection
    (``output``).
    """
    attn = layer.self_attn
    x = layer.layer_norm1(hidden)
    q, k, v = attn.q_proj(x), attn.k_proj(x), attn.v_proj(x)
    batch, length, width = q.shape

    def split(t: torch.Tensor) -> torch.Tensor:
        return t.view(batch, length, heads, width // heads).transpose(1, 2)

    scores = split(q) @ split(k).transpose(-1, -2) * (width // heads) ** -0.5
    if causal:
        mask = torch.full((length, length), float("-inf"), device=hidden.device).triu(1)
        scores = scores + mask
    attention = scores.softmax(dim=-1)
    output = (attention @ split(v)).transpose(1, 2).reshape(batch, length, width)
    if keep is not None:
        keep.update(q=q, k=k, v=v, attention=attention, output=output)
    hidden = hidden + attn.out_proj(output)
    return hidden + layer.mlp(layer.layer_norm2(hidden))


def image_tokens(model: CLIPModel, pixels: torch.Tensor) -> torch.Tensor:
    """Return the tokens that enter the first block of CLIP's vision encoder."""
    vision = model.vision_model
    return vision.pre_layrnorm(vision.embeddings(pixels))


def image_embedding(model: CLIPModel, hidden: torch.Tensor) -> torch.Tensor:
    """Return CLIP's image embedding (not normalized) from the output of the last block."""
    return model.visual_projection(model.vision_model.post_layernorm(hidden[:, 0]))


def text_embedding(model: CLIPModel, hidden: torch.Tensor, ends: torch.Tensor) -> torch.Tensor:
    """Return CLIP's text embedding (not normalized) at the position ``ends`` of each text."""
    hidden = model.text_model.final_layer_norm(hidden)
    return model.text_projection(hidden[torch.arange(len(hidden)), ends])


def run_image(
    model: CLIPModel,
    hidden: torch.Tensor,
    blocks: range,
    *,
    heads: dict[int, int] | None = None,
    keep: dict[int, dict[str, Any]] | None = None,
) -> torch.Tensor:
    """Run the blocks ``blocks`` of the vision encoder, with other heads for some (``heads``)."""
    config = model.config.vision_config
    for i in blocks:
        layer = model.vision_model.encoder.layers[i]
        n_heads = (heads or {}).get(i, config.num_attention_heads)
        hidden = block(
            layer, hidden, n_heads, keep=None if keep is None else keep.setdefault(i, {})
        )
    return hidden


def run_text(
    model: CLIPModel,
    hidden: torch.Tensor,
    blocks: range,
    *,
    heads: dict[int, int] | None = None,
    keep: dict[int, dict[str, Any]] | None = None,
) -> torch.Tensor:
    """Run the blocks ``blocks`` of the text encoder (causal), as :func:`run_image`."""
    config = model.config.text_config
    for i in blocks:
        layer = model.text_model.encoder.layers[i]
        n_heads = (heads or {}).get(i, config.num_attention_heads)
        parts = None if keep is None else keep.setdefault(i, {})
        hidden = block(layer, hidden, n_heads, causal=True, keep=parts)
    return hidden


def padded(ids: list[int]) -> torch.Tensor:
    """Return the token ids of a caption padded with zeros to CLIP's 77 positions, as a batch of 1."""
    if len(ids) > CONTEXT or ids[0] != START or ids[-1] != END:
        message = f"expected <|startoftext|>, at most 75 tokens and <|endoftext|>, got {ids}"
        raise ValueError(message)
    return torch.tensor([[*ids, *[0] * (CONTEXT - len(ids))]])


def tracked(tokens: torch.Tensor) -> torch.Tensor:
    """Return input tokens that autograd follows: with frozen weights, the graph starts there."""
    return tokens.detach().requires_grad_(requires_grad=True)


def embeddings(
    model: CLIPModel, pixels: torch.Tensor, text: torch.Tensor
) -> tuple[torch.Tensor, ...]:
    """Return the unit-length image and text embeddings of CLIP itself, without gradients."""
    n_image = model.config.vision_config.num_hidden_layers
    n_text = model.config.text_config.num_hidden_layers
    with torch.no_grad():
        image = image_embedding(
            model, run_image(model, image_tokens(model, pixels), range(n_image))
        )
        hidden = run_text(model, model.text_model.embeddings(input_ids=text), range(n_text))
        words = text_embedding(model, hidden, text.argmax(dim=-1))
    return image / image.norm(dim=-1, keepdim=True), words / words.norm(dim=-1, keepdim=True)


# -- the methods -------------------------------------------------------------------------------------


def game(
    model: CLIPModel, pixels: torch.Tensor, text: torch.Tensor
) -> tuple[np.ndarray, np.ndarray]:
    """Explain CLIP with GAME, the generic attention-model explainability of Chefer et al. (2021).

    As ``mm_interpret`` of FIxLIP's repository (``gradeclip/generate_emap.py``), with its defaults,
    which the paper used: the relevance starts at the last block of each encoder
    (``start_layer=-1``), so it is ``I + mean over heads of max(0, grad * A)`` for the attention
    maps ``A`` of the last block and their gradient ``grad``. What is differentiated is CLIP's logit
    of the pair, the cosine similarity times ``exp(logit_scale)``. A patch's value is the relevance
    of the class token for it, a token's value is the relevance of <|endoftext|> for it. This
    follows Chefer et al.'s CLIP code (github.com/hila-chefer/Transformer-MM-Explainability, MIT).

    Args:
        model: The Hugging Face ``CLIPModel``.
        pixels: The normalized image, of shape ``(1, 3, 224, 224)``.
        text: The padded token ids, of shape ``(1, 77)`` (:func:`padded`).

    Returns:
        The values of the patches, row by row, and of the caption's tokens.
    """
    n_image = model.config.vision_config.num_hidden_layers
    n_text = model.config.text_config.num_hidden_layers
    end = int(text.argmax())
    keep_image: dict[int, dict[str, Any]] = {}
    keep_text: dict[int, dict[str, Any]] = {}
    with torch.enable_grad():
        hidden = run_image(
            model, tracked(image_tokens(model, pixels)), range(n_image), keep=keep_image
        )
        image = image_embedding(model, hidden)
        hidden = run_text(
            model,
            tracked(model.text_model.embeddings(input_ids=text)),
            range(n_text),
            keep=keep_text,
        )
        words = text_embedding(model, hidden, text.argmax(dim=-1))
        cosine = torch.nn.functional.cosine_similarity(image, words)[0]
        logit = model.logit_scale.exp() * cosine
        maps = [keep_image[n_image - 1]["attention"], keep_text[n_text - 1]["attention"]]
        grads = torch.autograd.grad(logit, maps)
    relevance = []
    for attention, grad in zip(maps, grads, strict=True):
        cam = (grad * attention).clamp(min=0).mean(dim=1)[0].detach()  # mean over the heads
        relevance.append(torch.eye(len(cam)) + cam)  # R = I + cam @ I, the last block only
    patches = relevance[0][0, 1:]
    tokens = relevance[1][end, 1:end]
    return patches.numpy(), tokens.numpy()


def _qk_similarity(q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    """Return the cosine similarity of a query with every key, scaled to [0, 1] (Grad-ECLIP)."""
    similarity = torch.nn.functional.cosine_similarity(q[None], k, dim=-1)
    return (similarity - similarity.min()) / (similarity.max() - similarity.min())


def grad_eclip(
    model: CLIPModel, pixels: torch.Tensor, text: torch.Tensor
) -> tuple[np.ndarray, np.ndarray]:
    """Explain CLIP with Grad-ECLIP, the gradient-based explanation of Zhao et al. (2024).

    As ``gradeclip`` and ``gradeclip_text`` of FIxLIP's repository (``gradeclip/generate_emap.py``,
    adapted from github.com/Cyang-Zhao/Grad-Eclip, with ``withksim=True``) and their use in
    ``gradeclip/experiments.ipynb``. What is differentiated is the cosine similarity of the pair.

    - The image map reads the last block of the vision encoder. The gradient of the similarity on
      the output of its attention (before the output projection), at the class token, is multiplied
      with the values of the patches and weighted by the similarity of the class token's query with
      each patch's key, scaled to [0, 1]. Its sum over the channels, at least 0, is a patch's value.
      The text embedding is CLIP's.
    - The text map reads the last eight blocks of the text encoder in the same way, at the position
      of <|endoftext|>, with the query-key similarity scaled over all 77 positions. The maps of the
      blocks are added, and their sum, at least 0, is a token's value. The image embedding is CLIP's.

    As in the reference, the blocks that are read compute their attention with one head over all
    channels, and the queries and keys of the similarity are passed through the attention's output
    projection (with its bias). The other blocks are CLIP's.

    Args:
        model: The Hugging Face ``CLIPModel``.
        pixels: The normalized image, of shape ``(1, 3, 224, 224)``.
        text: The padded token ids, of shape ``(1, 77)`` (:func:`padded`).

    Returns:
        The values of the patches, row by row, and of the caption's tokens.
    """
    n_image = model.config.vision_config.num_hidden_layers
    n_text = model.config.text_config.num_hidden_layers
    end = int(text.argmax())
    image_fixed, text_fixed = embeddings(model, pixels, text)
    with torch.enable_grad():
        # the image map: the last block of the vision encoder with one head
        keep: dict[int, dict[str, Any]] = {}
        last = n_image - 1
        hidden = run_image(
            model, tracked(image_tokens(model, pixels)), range(n_image), heads={last: 1}, keep=keep
        )
        image = image_embedding(model, hidden)
        cosine = torch.nn.functional.cosine_similarity(image, text_fixed)[0]
        parts = keep[last]
        grad = torch.autograd.grad(cosine, parts["output"])[0][0, 0]  # at the class token
        out = model.vision_model.encoder.layers[last].self_attn.out_proj
        with torch.no_grad():
            weight = _qk_similarity(out(parts["q"][0, 0]), out(parts["k"][0, 1:]))
            patches = (grad * parts["v"][0, 1:] * weight[:, None]).sum(-1).clamp(min=0)

        # the text map: the last eight blocks of the text encoder with one head each
        read = range(n_text - GRADECLIP_TEXT_BLOCKS, n_text)
        keep = {}
        hidden = run_text(
            model,
            tracked(model.text_model.embeddings(input_ids=text)),
            range(n_text),
            heads=dict.fromkeys(read, 1),
            keep=keep,
        )
        words = text_embedding(model, hidden, text.argmax(dim=-1))
        cosine = torch.nn.functional.cosine_similarity(image_fixed, words)[0]
        grads = torch.autograd.grad(cosine, [keep[i]["output"] for i in read])
        total = torch.zeros(text.shape[1])
        with torch.no_grad():
            for i, grad_i in zip(read, grads, strict=True):
                parts = keep[i]
                out = model.text_model.encoder.layers[i].self_attn.out_proj
                weight = _qk_similarity(out(parts["q"][0, end]), out(parts["k"][0]))
                total += (grad_i[0, end] * parts["v"][0] * weight[:, None]).sum(-1)
        tokens = total.clamp(min=0)[1:end]
    return patches.detach().numpy(), tokens.numpy()


def _integrated_projection(
    embed: Callable[..., torch.Tensor],
    start: torch.Tensor,
    reference: torch.Tensor,
    steps: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return exCLIP's integrated gradients of an embedding on the input of a block, per token.

    The input ``start`` (tokens x channels) is interpolated to the ``reference`` on ``steps``
    points, from the input itself (1) to 1/steps, and the embedding ``embed`` (a function of a
    batch of inputs and of whether it is the reference) is differentiated there. For every channel of the embedding, the gradients are
    averaged over the points and multiplied with the difference of input and reference, and summed
    over the channels of every token. Returns these integrated gradients (embedding channels x
    tokens) and the embedding of the input minus that of the reference.
    """
    alphas = torch.arange(1, 0, -1 / steps)[:steps, None, None]
    points = (reference + alphas * (start - reference)).requires_grad_(requires_grad=True)
    with torch.enable_grad():
        embedded = embed(points, reference=False)  # (steps, channels)
        rows = []
        for d in range(embedded.shape[1]):
            (grad,) = torch.autograd.grad(
                embedded[:, d].sum(), points, retain_graph=d < embedded.shape[1] - 1
            )
            rows.append((grad.sum(0) / steps * (start - reference)).sum(-1))
    with torch.no_grad():
        shift = embedded[0] - embed(reference[None], reference=True)[0]  # point 1 is the input
    return torch.stack(rows), shift


def exclip(
    model: CLIPModel,
    pixels: torch.Tensor,
    text: torch.Tensor,
    steps: int = EXCLIP_STEPS,
    layer: int = EXCLIP_BLOCK,
) -> np.ndarray:
    """Explain CLIP with exCLIP, the cross-modal interactions of Moeller et al. (2025).

    As the ``exCLIP`` wrapper of FIxLIP's repository (``exclip/wrapper.py`` and ``hooks.py``,
    adapted from the supplementary material of the exCLIP paper) with the settings of
    ``exclip/experiments.ipynb``: the inputs of block 11 of both encoders are interpolated to those
    of a reference on 20 points, ``1, 0.95, ..., 0.05``. The reference image is all zeros after
    normalization, the reference text is <|startoftext|>, the id 0 twice and <|endoftext|>, padded
    with zeros. The embeddings are shifted by the reference's (``shift_embeddings``), and the
    interaction of a text token ``s`` and an image token ``t`` is the sum over the channels of the
    integrated Jacobians of both embeddings, times the differences of their inputs to their
    references, divided by the norms of the two shifted embeddings, without the logit scale. The
    sum over the channels of both tokens is computed per embedding channel first, which gives the
    same values with less memory. Returns the interactions of the patches (rows) and the caption's
    tokens (columns), without the class token, <|startoftext|> and <|endoftext|>.

    Args:
        model: The Hugging Face ``CLIPModel``.
        pixels: The normalized image, of shape ``(1, 3, 224, 224)``.
        text: The padded token ids, of shape ``(1, 77)`` (:func:`padded`).
        steps: The points of the integration. Defaults to 20, as for the paper.
        layer: The block whose input is interpolated. Defaults to 11, as for the paper.
    """
    n_image = model.config.vision_config.num_hidden_layers
    n_text = model.config.text_config.num_hidden_layers
    end = int(text.argmax())
    reference_text = torch.zeros_like(text)
    reference_text[0, 0], reference_text[0, EXCLIP_REFERENCE_TOKENS] = START, END

    with torch.no_grad():
        inputs = model.text_model.embeddings(input_ids=torch.cat([text, reference_text]))
        inputs = run_text(model, inputs, range(layer))
        images = image_tokens(model, torch.cat([pixels, torch.zeros_like(pixels)]))
        images = run_image(model, images, range(layer))

    def embed_text(points: torch.Tensor, *, reference: bool) -> torch.Tensor:
        # the reference ends at its own <|endoftext|>, an interpolation at the caption's
        at = EXCLIP_REFERENCE_TOKENS if reference else end
        hidden = run_text(model, points, range(layer, n_text))
        return text_embedding(model, hidden, torch.full((len(points),), at))

    def embed_image(points: torch.Tensor, *, reference: bool) -> torch.Tensor:  # noqa: ARG001
        return image_embedding(model, run_image(model, points, range(layer, n_image)))

    words, words_shift = _integrated_projection(embed_text, inputs[0], inputs[1], steps)
    patches, image_shift = _integrated_projection(embed_image, images[0], images[1], steps)
    interactions = words[:, : end + 1].T @ patches / image_shift.norm() / words_shift.norm()
    return interactions[1:end, 1:].T.detach().numpy()


# -- the photos of the demo ----------------------------------------------------------------------------


def rounded(values: np.ndarray) -> list[Any]:
    """Return values with ``DIGITS`` significant digits, as nested lists."""
    return np.vectorize(lambda v: float(f"{v:.{DIGITS}g}"), otypes=[float])(values).tolist()


def explain(model: CLIPModel, pixels: torch.Tensor, ids: list[int]) -> dict[str, Any]:
    """Explain one image and caption with the three methods, rounded for ``figures-data.js``."""
    text = padded(ids)
    game_patches, game_tokens = game(model, pixels, text)
    eclip_patches, eclip_tokens = grad_eclip(model, pixels, text)
    return {
        "game": {"patches": rounded(game_patches), "tokens": rounded(game_tokens)},
        "gradeclip": {"patches": rounded(eclip_patches), "tokens": rounded(eclip_tokens)},
        "exclip": rounded(exclip(model, pixels, text)),  # patches x tokens
    }


def write(results: dict[str, dict[str, Any]], model: dict[str, Any]) -> None:
    """Write the explanations into the part ``baselines`` of ``figures-data.js``, in the demo's order."""
    figures = read_figures()
    figures["baselines"] = {
        "model": model["id"],
        "checkpoint": model["checkpoint"],
        "grid": model["grid"],
        "images": results,
    }
    write_figures(figures)


def main() -> None:
    """Explain every photo of config.toml for CLIP and write the explanations."""
    torch.set_num_threads(4)
    config = tomllib.loads((HERE / "config.toml").read_text(encoding="utf-8"))
    model = next(m for m in config["models"] if m["id"] == MODEL)
    data = read_data()
    results_dir = HERE / "results"
    results_dir.mkdir(exist_ok=True)
    name = f"{model['id']}_demo"  # as explain.py
    image_text.IMAGE_TEXT_MODEL_IDS[name] = model["checkpoint"]
    grid = (model["grid"], model["grid"])
    results = {}
    for photo in config["photos"]:
        start = time.time()
        embedder = image_text.ImageTextTokenModel(
            np.asarray(load_photo(photo, results_dir)), grid, model=name, mask_strategy="mask"
        )
        network = Network(embedder)
        clip = network.model.eval().requires_grad_(requires_grad=False)
        ids = network.tokenizer([photo["caption"]])["input_ids"][0]
        tokens = [t["text"] for t in caption_tokens(network, photo["caption"])]
        shown = [text for text, _ in data["explanations"][f"{photo['id']}/{MODEL}"]["tokens"]]
        if tokens != shown or len(ids) != len(tokens) + 2:
            message = f"{photo['id']}: the tokens {tokens} are not those of data.js, {shown}"
            raise ValueError(message)
        results[photo["id"]] = explain(clip, embedder._pixels, ids)  # noqa: SLF001
        print(f"{photo['id']}: {len(tokens)} tokens, {time.time() - start:.0f} s", flush=True)
    write(results, model)


if __name__ == "__main__":
    main()
