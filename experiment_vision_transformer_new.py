"""This script computes "ground truth" Shapley values for the vision transformer by running
KernelSHAP on the vision transformer model with a budget of 1_000_000 estimations."""

import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from transformers import ViTForImageClassification, ViTImageProcessor

import shapiq

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
        original_image: torch.Tensor,
        background_image: torch.Tensor,
        n_patches_per_row: int = 12,
        patch_size: int = 32,
    ):
        self.coalitions = coalitions
        self.original_image = original_image.clone()
        self.background_image = background_image.clone()
        self.n_patches_per_row = n_patches_per_row
        self.patch_size = patch_size

        # create a dictionary mapping from patch ids to pixels
        self.patches_positions = {}
        for i in range(self.coalitions.shape[1]):
            row = i // self.n_patches_per_row
            column = i % self.n_patches_per_row
            self.patches_positions[i] = {
                "row": row,
                "column": column,
                "start_row": row * self.patch_size,
                "end_row": (row + 1) * self.patch_size,
                "start_column": column * self.patch_size,
                "end_column": (column + 1) * self.patch_size,
            }

    def __len__(self):
        return self.coalitions.shape[0]

    def __getitem__(self, idx):
        coalition = self.coalitions[idx]
        # add an additional True value for the bias term in the beginning of the coalition
        image = self.background_image.clone()
        for i, is_present in enumerate(coalition):
            if not is_present:
                continue
            start_row = self.patches_positions[i]["start_row"]
            end_row = self.patches_positions[i]["end_row"]
            start_column = self.patches_positions[i]["start_column"]
            end_column = self.patches_positions[i]["end_column"]
            image[:, start_row:end_row, start_column:end_column] = self.original_image[
                :, start_row:end_row, start_column:end_column
            ]
        return image[0]


class VisionTransformerGame(shapiq.Game):

    def __init__(self, x_explain_path: str, verbose: bool = True, class_id: int = None):
        self.input_image = Image.open(x_explain_path)
        self.verbose = verbose

        # setup device for model
        # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = "cpu"
        print("Using device:", self.device)

        # get model
        feature_extractor = ViTImageProcessor.from_pretrained("google/vit-base-patch32-384")
        model = ViTForImageClassification.from_pretrained("google/vit-base-patch32-384")
        # model = torch.compile(model, fullgraph=True)  # TODO: maybe remove fullgraph=True
        self.n_patches_per_row = model.vit.config.image_size // model.vit.config.patch_size
        self.n_patches = self.n_patches_per_row**2
        self.patch_size = model.vit.config.patch_size

        # get model components
        self.model = model

        # move model to device
        self.model.to(self.device)
        self.model.eval()

        # get the original image
        original_image = feature_extractor(images=self.input_image, return_tensors="pt")[
            "pixel_values"
        ]
        original_image.to(self.device)
        self.original_image = original_image
        self.original_image.to(self.device)

        # get the background image
        grey_image = Image.new("RGB", (384, 384), (128, 128, 128))
        grey_image = feature_extractor(images=grey_image, return_tensors="pt")["pixel_values"]
        grey_image.to(self.device)
        self.background_image = grey_image

        # get class id
        out_logit = self.model(self.original_image.to(self.device)).logits
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

        super().__init__(n_players=self.n_patches, normalize=False)

    def test_equal_full(self):
        assert np.allclose(self.full_value, self.out_logit)

    def value_function(self, coalitions: np.ndarray, batch_size: int = 1_000) -> np.ndarray:
        """Runs the Model on a dataset of masked images and returns the logits."""

        if len(coalitions.shape) == 1:
            coalitions = coalitions.reshape(1, -1)

        dataset = MaskedDataset(
            coalitions=coalitions,
            original_image=self.original_image,
            background_image=self.background_image,
            n_patches_per_row=self.n_patches_per_row,
            patch_size=self.patch_size,
        )
        data_loader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=5
        )

        logits = []
        with torch.no_grad():
            # with torch.autocast("cuda"):
            for batch in tqdm(data_loader, desc="Computing"):
                batch = batch.to(self.device)
                logit_output = self.model(batch).logits
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
    # get the images
    test_image = os.path.join("images", "dog_example.jpg")
    test_image = Image.open(test_image)
    gray_image = Image.new("RGB", (384, 384), (128, 128, 128))

    feature_extractor = ViTImageProcessor.from_pretrained("google/vit-base-patch32-384")
    model = ViTForImageClassification.from_pretrained("google/vit-base-patch32-384")
    patch_size = model.vit.config.patch_size
    n_patches = model.vit.config.image_size // model.vit.config.patch_size

    # run images through the feature extractor
    gray_image = feature_extractor(images=gray_image, return_tensors="pt")
    original_image = feature_extractor(images=test_image, return_tensors="pt")

    # replace the first patch of the original image with the first patch of the gray image

    original_image["pixel_values"][0, :, 0:patch_size, 0:patch_size] = gray_image["pixel_values"][
        0, :, 0:patch_size, 0:patch_size
    ]

    # compute the embeddings
    gray_embedding = model.vit.embeddings(**gray_image)
    original_embedding = model.vit.embeddings(**original_image)

    # check if the embeddings are only including positional encodings and patch embeddings
    for i in range(0, n_patches + 1):  # the first patch is the bias term
        gray_patch_embedding = gray_embedding[0, i, :].detach().cpu().numpy()
        original_patch_embedding = original_embedding[0, i, :].detach().cpu().numpy()
        if i == 0:  # bias term should be the same
            assert np.allclose(gray_patch_embedding, original_patch_embedding)
        elif i == 1:  # first patch should be the same (grey and grey in the original)
            print(gray_patch_embedding)
            print(original_patch_embedding)
            assert np.allclose(gray_patch_embedding, original_patch_embedding)
        else:  # the remaining patches should be different
            assert not np.allclose(gray_patch_embedding, original_patch_embedding)


if __name__ == "__main__":

    import torch

    # set visible devices to 0
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # set to 0 for the first GPU

    results_dir = Path(__file__).parent / "results"
    experiment = "vit"
    os.makedirs(results_dir, exist_ok=True)

    # pre_compute_model_values(image_name="dog_example.jpg", experiment=experiment, class_id=207)
    # pre_compute_model_values(image_name="dog_example_guitar.jpg", experiment=experiment, class_id=402)

    #  dog = 207, acoustic guitar = 402, pick, plectrum, plectron 714
    game = VisionTransformerGame(x_explain_path="dog_example.jpg", class_id=207)
    coalition = np.zeros(game.n_patches, dtype=bool)
    coalition[2] = True
    coalition[3] = True
    game.value_function(coalition)
