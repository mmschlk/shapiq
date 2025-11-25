import numpy as np
from shapiq import Game, InteractionValues

import torch
from transformers import AutoFeatureExtractor, AutoModelForImageClassification
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor, ToPILImage
from torch.utils.data import DataLoader
from tqdm import tqdm
from PIL import Image
import random

from shapiq import ExactComputer, InteractionValues

class LocalXAICIFAR10Game(Game):
    def __init__(self,id_explain,random_state: int = 42, use_model: bool = False):
        # Load CIFAR-10 test set
        test_dataset = CIFAR10(root="./data", train=False, download=True)
        self.label_names = [
            "airplane", "automobile", "bird", "cat", "deer",
            "dog", "frog", "horse", "ship", "truck"
        ]

        self.grid_size = 4 # 4x4 grid of superpixels

        self.random_state = random_state
        # Create deterministic random generator
        rng = random.Random(random_state)
        # Create and shuffle indices
        indices = list(range(len(test_dataset)))
        rng.shuffle(indices)
        # Select the instance to explain
        self.id_explain = id_explain
        self.x_explain, self.true_label = test_dataset[indices[id_explain]]

        n_players = 16  # 4x4 superpixels
        if use_model:
            # Device
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

            # Load model and feature extractor
            model_name = "aaraki/vit-base-patch16-224-in21k-finetuned-cifar10"
            self.model = AutoModelForImageClassification.from_pretrained(model_name).to(self.device)
            self.feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
            self.model.eval()
            self.BATCH_SIZE = 16  # Batch size for processing masked images
            self.baseline_value = 128  # Gray baseline for masking

            self.x_baseline = self.batch_mask_superpixels_pil(np.zeros((1,n_players),dtype=bool))[0]
            inputs = self.feature_extractor(images=self.x_explain, return_tensors="pt").to(self.device)
            inputs_baseline = self.feature_extractor(images=self.x_baseline, return_tensors="pt").to(self.device)
            with torch.no_grad():
                outputs = self.model(**inputs)
                self.explained_class = torch.argmax(outputs.logits, dim=-1).item()
                outputs_baseline = self.model(**inputs_baseline)
                self.baseline_prediction = outputs_baseline.logits[0, self.explained_class].item()

        #     super().__init__(n_players=n_players,normalize=True,normalization_value=self.baseline_prediction)
        # else:
        super().__init__(n_players=n_players,normalize=False) # no normalization - use pre-loaded game values


    def summary(self):
        self.x_explain.show()
        print(f"True label: { self.label_names[self.true_label]}")
        print(f"Predicted label: { self.label_names[self.explained_class]}")
        print(f"Explained instance logits for class { self.label_names[self.explained_class]}: {self.grand_coalition_value:.4f}")
        print(f"Baseline logits (all superpixels masked): {self.baseline_prediction:.4f}")
        print(f"Explained class: { self.label_names[self.explained_class]}")
        print(f"Number of players (superpixels): {self.n_players}")
        print(f"Predicted correctly?: {self.true_label == self.explained_class}")

    def batch_mask_superpixels_pil(self, coalitions):
        """
        coalitions: numpy array of shape (N, num_superpixels)
        returns: list of masked PIL images length N
        """
        img_np = np.array(self.x_explain)
        H, W, C = img_np.shape
        h_step = H // self.grid_size
        w_step = W // self.grid_size

        # Precompute masks for each superpixel
        masks = []
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                mask = np.zeros((H, W), dtype=bool)
                mask[i * h_step:(i + 1) * h_step, j * w_step:(j + 1) * w_step] = True
                masks.append(mask)
        masks = np.stack(masks)  # shape = (num_superpixels, H, W)

        # Batch masking
        masked_imgs = []
        for coalition in coalitions:
            keep = coalition.astype(bool)  # shape = (num_superpixels,)
            mask_keep = masks[keep].any(0)  # OR over kept masks → shape = (H, W)

            img = img_np.copy()
            img[~mask_keep] = self.baseline_value
            masked_imgs.append(Image.fromarray(img))

        return masked_imgs

    def value_function(self, coalitions: np.ndarray) -> np.ndarray:
        # Create all masked images at once
        masked_images = self.batch_mask_superpixels_pil(
            coalitions
        )

        all_logits = []
        # Process in smaller batches
        for i in tqdm(range(0, len(masked_images), self.BATCH_SIZE)):
            batch_images = masked_images[i:i + self.BATCH_SIZE]

            # Encode batch
            inputs = self.feature_extractor(images=batch_images, return_tensors="pt", padding=True).to(self.device)

            # Inference
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits[:, self.explained_class].cpu().numpy()

            all_logits.append(logits)

        # Concatenate all results
        all_logits = np.concatenate(all_logits, axis=0)
        return all_logits



    def exact_values(self, index: str, order: int) -> InteractionValues:
        if index=="SV" and order==1:
            shap_save_path = f"cifar10_precomputed/cifar10_{self.id_explain}_sv"
            shapley_values = InteractionValues.load_interaction_values(shap_save_path)
        return shapley_values

if __name__ == "__main__":
    import os
    def compute_game(id_explain):
        game = LocalXAICIFAR10Game(id_explain=id_explain)
        save_path = f"experiments/cifar10_precomputed/cifar10_{id_explain}_game_values.npz"
        # game.summary()
        if os.path.exists(save_path):
            game.load_values(save_path)
            print(f"Game values for explanation id {id_explain} already computed, skipping.")
        else:
            game.save_values(save_path)

        exact_computer = ExactComputer(n_players=game.n_players,game=game)
        shapley_value = exact_computer(index="SV", order=1)
        shap_save_path = f"experiments/cifar10_precomputed/cifar10_{id_explain}_sv"
        shapley_value.save(shap_save_path)

    for id_explain in range(30):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        compute_game(id_explain)



