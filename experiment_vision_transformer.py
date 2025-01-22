"""This script computes "ground truth" Shapley values for the vision transformer by running
KernelSHAP on the vision transformer model with a budget of 1_000_000 estimations."""

import copy
import os
from pathlib import Path
from warnings import warn

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

import shapiq
from experiment_utils import make_file_paths, pre_compute_model_values
from shapiq.approximator.regression.shapleygax import ExplanationBasisGenerator, ShapleyGAX
from shapiq.benchmark.metrics import get_all_metrics

RANDOM_SEED = 42


class MaskedDataset(torch.utils.data.Dataset):
    """A Dataset that returns image embeddings for masked images.

    The masking is applied to the embedding layer of the Vision Transformer model. Out-of-coalition
    patches are set to a mask token in the embedding layer.

    Args:
        coalitions: A boolean matrix where each row corresponds to a coalition of masked patches.
        original_embedding: The embedding of the original image.
        background_embedding: The embedding of a gray image.
    """

    def __init__(
        self,
        coalitions: np.ndarray,
        original_embedding: torch.Tensor,
        background_embedding: torch.Tensor,
    ):
        self.coalitions = coalitions
        self.original_embedding = original_embedding
        self.background_embedding = background_embedding

    def __len__(self):
        return self.coalitions.shape[0]

    def __getitem__(self, idx):
        coalition = self.coalitions[idx]
        # add an additional True value for the bias term in the beginning of the coalition
        coalition_array = np.array([True] + coalition.tolist(), dtype=bool)
        embedding = self.background_embedding.clone()
        embedding[0, coalition_array] = self.original_embedding[0, coalition_array]
        return embedding[0]


class VisionTransformerGame(shapiq.Game):

    def __init__(self, x_explain_path: str, verbose: bool = True, class_id: int = None):
        from transformers import ViTForImageClassification, ViTImageProcessor

        from shapiq.games.benchmark._setup._vit_setup import NORM_BIAS, NORM_WEIGHT

        self.input_image = Image.open(x_explain_path)
        self.verbose = verbose

        # setup device for model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Using device:", self.device)

        # get model
        feature_extractor = ViTImageProcessor.from_pretrained("google/vit-base-patch32-384")
        model = ViTForImageClassification.from_pretrained("google/vit-base-patch32-384")
        model = torch.compile(model, fullgraph=True)  # try removing fullgraph=True first
        self.n_patches_per_row = model.vit.config.image_size // model.vit.config.patch_size
        self.n_patches = self.n_patches_per_row**2
        self.patch_size = model.vit.config.patch_size

        # get model components
        self.model = model

        # move model to device
        self.model.to(self.device)
        self.model.eval()

        self._classifier = self.model.classifier
        self._embedding_layer = self.model.vit.embeddings
        self._encoder = self.model.vit.encoder

        self._norm_weight = NORM_WEIGHT
        self._norm_bias = NORM_BIAS
        self._norm_eps = 1e-12
        self._norm_shape = (768,)

        # run input image through the feature extractor
        self.transformed_image = feature_extractor(images=self.input_image, return_tensors="pt")
        self.transformed_image = {k: v.to(self.device) for k, v in self.transformed_image.items()}
        original_embeddings = self._embedding_layer(**self.transformed_image)
        original_embeddings.to(self.device)
        self.original_embeddings = original_embeddings

        # set mask token of embedding layer to zeros to use `bool_masked_pos` parameter for masking
        grey_image = Image.new("RGB", (384, 384), (128, 128, 128))
        grey_image = feature_extractor(images=grey_image, return_tensors="pt")
        grey_image = {k: v.to(self.device) for k, v in grey_image.items()}
        background_embedding = self._embedding_layer(**grey_image)
        background_embedding.to(self.device)
        self.background_embedding = background_embedding

        # get class id
        out_logit = model(**self.transformed_image).logits
        out_proba = F.softmax(out_logit, dim=1).cpu().detach().numpy()
        self.class_id = class_id
        if class_id is None:
            self.class_id = int(np.argmax(out_proba))

        # store values
        self.out_proba = float(out_proba[0, self.class_id])
        self.out_logit = float(out_logit.cpu().detach().numpy()[0, self.class_id])
        self.original_class_name = str(model.config.id2label[self.class_id])
        print(
            f"Explaining class: {self.original_class_name} (proba={self.out_proba:.4f}, "
            f"and logit={self.out_logit:.4f})"
        )

        #  call the model with no information to get empty prediction
        empty_output = self.value_function(np.zeros(self.n_patches, dtype=bool))
        self.empty_value = float(empty_output[0])

        full_output = self.value_function(np.ones(self.n_patches, dtype=bool))
        self.full_value = float(full_output[0])
        close = np.allclose(self.full_value, self.out_logit)
        print(f"Are value function and model output equal?: {close}")
        if not close:
            warn("Value function and model output are not equal.")
            print(f"Value function: {self.full_value}, Model output: {self.out_logit}")

        super().__init__(n_players=self.n_patches, normalize=False)

    def value_function(self, coalitions: np.ndarray, batch_size: int = 2_000) -> np.ndarray:
        """Runs the Model on a dataset of masked images and returns the logits."""

        if len(coalitions.shape) == 1:
            coalitions = coalitions.reshape(1, -1)

        dataset = MaskedDataset(
            coalitions=coalitions,
            original_embedding=self.original_embeddings,
            background_embedding=self.background_embedding,
        )
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)

        logits = []
        with torch.no_grad():
            with torch.autocast("cuda"):
                for batch in tqdm(data_loader):
                    encodings = self._encoder(batch)
                    norm_encodings = F.layer_norm(
                        encodings.last_hidden_state[:, 0],
                        self._norm_shape,
                        self._norm_weight.to(self.device),
                        self._norm_bias.to(self.device),
                        self._norm_eps,
                    )
                    logit_output = self._classifier(norm_encodings)
                    logit_output = logit_output.cpu().detach().numpy()[:, self.class_id]
                    logits.append(logit_output)
        logits = np.concatenate(logits)
        return logits


def validate_embeddings_are_positional():
    """This test script checks if the embeddings of the vision transformer models only contain
    patch and positional embeddings.

    This check works as follows:
        1. Create a gray image and a normal image.
        2. Run both images through the feature extractor.
        3. Set the first patch of the normal image to gray. (Replace the first patch of the normal
            image with the first patch of the gray image.)
        4. Compute the embeddings of both images.

    If the embeddings are only including positional encodings and patch embeddings.
        - The bias term should be the same for both images (first "token").
        - The embeddings of the first patch should be the same for both images (second "token").
        - The embeddings of the remaining patches should be different for both images.
    """
    from transformers import ViTForImageClassification, ViTImageProcessor

    # get the images
    test_image = os.path.join("images", "dog_example.jpg")
    test_image = Image.open(test_image)
    gray_image = Image.new("RGB", (384, 384), (128, 128, 128))
    random_image = np.array(np.random.randint(0, 255, (384, 384, 3)), dtype=np.uint8)
    random_image = Image.fromarray(random_image)

    feature_extractor = ViTImageProcessor.from_pretrained("google/vit-base-patch32-384")
    model = ViTForImageClassification.from_pretrained("google/vit-base-patch32-384")
    patch_size = model.vit.config.patch_size
    n_patches = model.vit.config.image_size // model.vit.config.patch_size

    # run images through the feature extractor
    gray_image = feature_extractor(images=gray_image, return_tensors="pt")
    original_image = feature_extractor(images=test_image, return_tensors="pt")
    random_image = feature_extractor(images=random_image, return_tensors="pt")

    # replace the first patch of the original image with the first patch of the gray image
    original_image["pixel_values"][0, :, 0:patch_size, 0:patch_size] = gray_image["pixel_values"][
        0, :, 0:patch_size, 0:patch_size
    ]

    # compute the embeddings
    gray_embedding = model.vit.embeddings(**gray_image)
    original_embedding = model.vit.embeddings(**original_image)
    random_embedding = model.vit.embeddings(**random_image)

    # check if the embeddings are only including positional encodings and patch embeddings
    for i in range(0, n_patches + 1):  # the first patch is the bias term
        gray_patch_embedding = gray_embedding[0, i, :].detach().cpu().numpy()
        original_patch_embedding = original_embedding[0, i, :].detach().cpu().numpy()
        random_patch_embedding = random_embedding[0, i, :].detach().cpu().numpy()
        if i == 0:  # bias term should be the same
            assert np.allclose(gray_patch_embedding, original_patch_embedding)
            assert np.allclose(gray_patch_embedding, random_patch_embedding)
            assert np.allclose(original_patch_embedding, random_patch_embedding)
        elif i == 1:  # first patch should be the same (grey and grey in the original)
            print(gray_patch_embedding)
            print(original_patch_embedding)
            assert np.allclose(gray_patch_embedding, original_patch_embedding)
            assert not np.allclose(gray_patch_embedding, random_patch_embedding)
            assert not np.allclose(original_patch_embedding, random_patch_embedding)
        else:  # the remaining patches should be different
            assert not np.allclose(gray_patch_embedding, original_patch_embedding)
            assert not np.allclose(gray_patch_embedding, random_patch_embedding)
            assert not np.allclose(original_patch_embedding, random_patch_embedding)


def _run_approximation(
    approximator: shapiq.approximator._base.Approximator,
    game: shapiq.Game,
    gt_sv: shapiq.InteractionValues,
    budget: int,
    name: str,
    image_name: str,
    print_estimate: bool = False,
) -> dict:
    """Run the approximation and return the metrics."""
    estimate = approximator.approximate(budget=budget, game=game)
    if print_estimate:
        print(estimate)
    gt_sv, estimate = gt_sv.get_n_order(order=1), estimate.get_n_order(order=1)
    name = name if name is not None else approximator.__class__.__name__
    metrics: dict = get_all_metrics(ground_truth=gt_sv, estimated=estimate)
    metrics["approximator"] = name
    metrics["budget"] = budget
    metrics["image_name"] = image_name
    print(metrics)
    return metrics


def approximate(
    approx_to_use: list[str],
    budgets: list[int],
    n_samples: int,
) -> None:
    # get all available images
    image_dir = Path(__file__).parent / "images"
    available_images = list(image_dir.glob("*ILSVRC2012_val*"))
    print("Available images:", available_images)

    results_file_name = "".join(approx_to_use) + "_results_vit.csv"

    # select a subset of images
    image_names = [f"{image.stem}{image.suffix}" for image in available_images]
    image_names = image_names[:n_samples]

    results = []
    for i, image_name in enumerate(image_names, start=1):
        print(f"Processing image {i}/{len(image_names)}: {image_name}\n")
        _, ground_truth_name = make_file_paths(image_name, budget=1_000_000, experiment="vit")
        gt_sv = shapiq.InteractionValues.load_interaction_values(path=ground_truth_name)
        print("Ground Truth", gt_sv)

        # setup the game
        game = VisionTransformerGame(
            x_explain_path=os.path.join("images", image_name),
            verbose=True,
        )

        for budget in budgets:

            # approximate with kernel shap ---------------------------------------------------------
            if "KernelSHAP" in approx_to_use:
                name = "KernelSHAP"
                basis_gen = ExplanationBasisGenerator(N=set(range(game.n_players)))
                explanation_basis = basis_gen.generate_kadd_explanation_basis(1)
                approximator = ShapleyGAX(
                    n=game.n_players, random_state=RANDOM_SEED, explanation_basis=explanation_basis
                )
                result = _run_approximation(
                    approximator, game, gt_sv, budget, name, image_name, print_estimate=False
                )
                results.append(copy.deepcopy(result))

            # approximate with shapley gax with 500 conjugate --------------------------------------
            if "ShapleyGAX (400)" in approx_to_use:
                name = "ShapleyGAX (400)"
                basis_gen = ExplanationBasisGenerator(N=set(range(game.n_players)))
                explanation_basis = basis_gen.generate_stochastic_explanation_basis(
                    400, conjugate=False
                )
                approximator = ShapleyGAX(
                    n=game.n_players, random_state=RANDOM_SEED, explanation_basis=explanation_basis
                )
                result = _run_approximation(
                    approximator, game, gt_sv, budget, name, image_name, print_estimate=False
                )
                results.append(copy.deepcopy(result))

            # approximate with shapley gax with 500 ------------------------------------------------
            if "ShapleyGAX (500)" in approx_to_use:
                name = "ShapleyGAX (500)"
                basis_gen = ExplanationBasisGenerator(N=set(range(game.n_players)))
                explanation_basis = basis_gen.generate_stochastic_explanation_basis(
                    500, conjugate=False
                )
                approximator = ShapleyGAX(
                    n=game.n_players, random_state=RANDOM_SEED, explanation_basis=explanation_basis
                )
                result = _run_approximation(
                    approximator, game, gt_sv, budget, name, image_name, print_estimate=False
                )
                results.append(copy.deepcopy(result))

            # approximate with permutation sampling ------------------------------------------------
            if "PermutationSamplingSV" in approx_to_use:
                name = "PermutationSamplingSV"
                approximator = shapiq.PermutationSamplingSV(
                    n=game.n_players, random_state=RANDOM_SEED
                )
                result = _run_approximation(
                    approximator, game, gt_sv, budget, name, image_name, print_estimate=False
                )
                results.append(copy.deepcopy(result))

            # approximate with SVARM ---------------------------------------------------------------
            if "SVARM" in approx_to_use:
                name = "SVARM"
                approximator = shapiq.SVARM(n=game.n_players, random_state=RANDOM_SEED)
                result = _run_approximation(
                    approximator, game, gt_sv, budget, name, image_name, print_estimate=False
                )
                results.append(copy.deepcopy(result))

            results_df = pd.DataFrame(results)
            results_df.to_csv(results_file_name, index=False)

    # save the results
    results_df = pd.DataFrame(results)
    results_df.to_csv(results_file_name, index=False)


if __name__ == "__main__":

    os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # set to 0 or 1 for selecting the GPU
    print("CUDA device:", os.environ["CUDA_VISIBLE_DEVICES"])

    validate_embeddings = False  # set to True to run the validation test
    pre_compute_ground_truth = False  # set to True to pre-compute the ground truth values
    approximate_values = True

    if validate_embeddings:
        validate_embeddings_are_positional()
        print("Validation test passed.")

    if pre_compute_ground_truth:
        images_to_compute_dir = Path(__file__).parent / "images"
        # grab all files jpg, JPG, jpeg, JPEG, png, PNG
        images_to_compute = list(images_to_compute_dir.glob("*.[jJ][pP][gG]*"))
        images_to_compute += list(images_to_compute_dir.glob("*.[pP][nN][gG]*"))
        images_to_compute += list(images_to_compute_dir.glob("*.[jJ][pP][eE][gG]*"))

        # for cuda 1 reverse the order of the images
        if os.environ["CUDA_VISIBLE_DEVICES"] == "1":
            images_to_compute = images_to_compute[::-1]

        print("Images to compute:", images_to_compute)
        print(f"Cuda device: {os.environ['CUDA_VISIBLE_DEVICES']}")

        # pre-compute all that do not exist yet in the results directory
        for image_path in images_to_compute:
            image_name = image_path.stem
            file_extension = image_path.suffix
            if image_name == "dog_example":
                class_id = 207
            elif image_name == "dog_example_guitar":
                class_id = 402
            elif image_name == "dog_example_plectrum":
                class_id = 714
            else:
                class_id = None
            image_name = f"{image_name}{file_extension}"
            pre_compute_model_values(
                image_name=image_name,
                experiment="vit",
                class_id=class_id,
                recompute_if_exists=False,
            )

    if approximate_values:
        approximate(
            approx_to_use=[
                "KernelSHAP",
                "ShapleyGAX (500)",
                "ShapleyGAX (400)",
                "ShapleyGAX (600)",
                "PermutationSamplingSV",
                # "SVARM",
            ],
            n_samples=20,
            budgets=[10_000, 15_000, 20_000, 25_000, 30_000],
        )
