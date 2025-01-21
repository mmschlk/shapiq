"""This script computes "ground truth" Shapley values for the vision transformer by running
KernelSHAP on the vision transformer model with a budget of 1_000_000 estimations."""

import os
from pathlib import Path
from warnings import warn

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from transformers import ViTForImageClassification, ViTImageProcessor

import shapiq
from experiment_utils import pre_compute_model_values

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


if __name__ == "__main__":

    os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # set to 0 or 1 for selecting the GPU

    validate_embeddings = False  # set to True to run the validation test

    if validate_embeddings:
        validate_embeddings_are_positional()
        print("Validation test passed.")

    images_to_compute_dir = Path(__file__).parent / "images"
    # grab all files jpg, JPG, jpeg, JPEG, png, PNG
    images_to_compute = list(images_to_compute_dir.glob("*.[jJ][pP][gG]*"))
    images_to_compute += list(images_to_compute_dir.glob("*.[pP][nN][gG]*"))
    images_to_compute += list(images_to_compute_dir.glob("*.[jJ][pP][eE][gG]*"))

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
            image_name=image_name, experiment="vit", class_id=class_id, recompute_if_exists=False
        )
