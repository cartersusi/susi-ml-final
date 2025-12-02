import os
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.metrics import (
    confusion_matrix,
    precision_recall_fscore_support,
)
from torch.utils.data import DataLoader
from torchvision import transforms

from data import Dataset
from model import CatDogCNN, device
from train import _val_step


class Visualizer:
    def __init__(
        self,
        model: CatDogCNN,
        dataset: Dataset,
        results: dict[str, list[Any]],
        visuals_dir: str,
    ):
        self.model = model
        self.dataset = dataset
        self.results = results
        self.class_names = ["Cat", "Dog"]
        self.visuals_dir = visuals_dir

    def plot_training_curves(self) -> None:
        _, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

        epochs = range(1, len(self.results["train_loss"]) + 1)

        ax1.plot(
            epochs, self.results["train_loss"], "b-", label="Training Loss", linewidth=2
        )
        ax1.plot(
            epochs, self.results["val_loss"], "r-", label="Validation Loss", linewidth=2
        )
        ax1.set_title("Training and Validation Loss", fontsize=14, fontweight="bold")
        ax1.set_xlabel("Epoch", fontsize=12)
        ax1.set_ylabel("Loss", fontsize=12)
        ax1.legend(fontsize=11)
        ax1.grid(True, alpha=0.3)

        ax2.plot(
            epochs,
            self.results["train_acc"],
            "b-",
            label="Training Accuracy",
            linewidth=2,
        )
        ax2.plot(
            epochs,
            self.results["val_acc"],
            "r-",
            label="Validation Accuracy",
            linewidth=2,
        )
        ax2.set_title(
            "Training and Validation Accuracy", fontsize=14, fontweight="bold"
        )
        ax2.set_xlabel("Epoch", fontsize=12)
        ax2.set_ylabel("Accuracy", fontsize=12)
        ax2.legend(fontsize=11)
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(
            os.path.join(self.visuals_dir, "training_curves.png"),
            dpi=300,
            bbox_inches="tight",
        )
        print(f"Training curves saved to {self.visuals_dir}/training_curves.png")
        plt.close()

    def plot_confusion_matrix(self):
        test_loader = DataLoader(
            self.dataset.test_data,
            batch_size=32,
            shuffle=False,
            num_workers=os.cpu_count(),
        )

        loss_fn = nn.CrossEntropyLoss()
        _, _, predictions, true_labels = _val_step(
            self.model, test_loader, loss_fn, return_preds=True
        )

        cm = confusion_matrix(true_labels, predictions)

        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=self.class_names,
            yticklabels=self.class_names,
            cbar_kws={"label": "Count"},
        )
        plt.title("Confusion Matrix - Test Set", fontsize=16, fontweight="bold", pad=20)
        plt.ylabel("True Label", fontsize=13)
        plt.xlabel("Predicted Label", fontsize=13)
        plt.tight_layout()
        plt.savefig(
            os.path.join(self.visuals_dir, "confusion_matrix.png"),
            dpi=300,
            bbox_inches="tight",
        )
        print(f"Confusion matrix saved to {self.visuals_dir}/confusion_matrix.png")
        plt.close()

        return cm, predictions, true_labels

    def generate_classification_report(
        self, predictions: list[int], true_labels: list[int]
    ) -> None:
        precision, recall, f1, support = precision_recall_fscore_support(
            true_labels, predictions, average=None
        )

        # Visualize metrics
        fig, ax = plt.subplots(figsize=(10, 6))
        x = np.arange(len(self.class_names))
        width = 0.25

        ax.bar(x - width, precision, width, label="Precision", color="skyblue")
        ax.bar(x, recall, width, label="Recall", color="lightcoral")
        ax.bar(x + width, f1, width, label="F1-Score", color="lightgreen")

        ax.set_ylabel("Score", fontsize=12)
        ax.set_title(
            "Precision, Recall, and F1-Score by Class", fontsize=14, fontweight="bold"
        )
        ax.set_xticks(x)
        ax.set_xticklabels(self.class_names)
        ax.legend()
        ax.grid(True, alpha=0.3, axis="y")
        ax.set_ylim([0, 1.1])

        plt.tight_layout()
        plt.savefig(
            os.path.join(self.visuals_dir, "metrics_comparison.png"),
            dpi=300,
            bbox_inches="tight",
        )
        print(f"Metrics comparison saved to {self.visuals_dir}/metrics_comparison.png")
        plt.close()

    def show_sample_predictions(self, num_samples: int = 16) -> None:
        test_loader = DataLoader(
            self.dataset.test_data, batch_size=num_samples, shuffle=True, num_workers=1
        )

        self.model.eval()
        images, labels = next(iter(test_loader))
        images, labels = images.to(device), labels.to(device)

        with torch.inference_mode():
            outputs = self.model(images)
            probabilities = torch.softmax(outputs, dim=1)
            predictions = torch.argmax(probabilities, dim=1)
            confidences = torch.max(probabilities, dim=1)[0]

        # Plot
        rows = 4
        cols = 4
        fig, axes = plt.subplots(rows, cols, figsize=(16, 16))
        fig.suptitle("Sample Predictions on Test Set", fontsize=18, fontweight="bold")

        for idx, ax in enumerate(axes.flat):
            if idx >= len(images):
                break

            img = images[idx].cpu().permute(1, 2, 0).numpy()
            true_label = self.class_names[labels[idx].item()]
            pred_label = self.class_names[predictions[idx].item()]
            confidence = confidences[idx].item()

            ax.imshow(img)
            color = "green" if true_label == pred_label else "red"
            ax.set_title(
                f"True: {true_label}\nPred: {pred_label}\nConf: {confidence:.2%}",
                color=color,
                fontsize=10,
                fontweight="bold",
            )
            ax.axis("off")

        plt.tight_layout()
        plt.savefig(
            os.path.join(self.visuals_dir, "sample_predictions.png"),
            dpi=300,
            bbox_inches="tight",
        )
        print(f"Sample predictions saved to {self.visuals_dir}/sample_predictions.png")
        plt.close()

    def analyze_misclassifications(self, num_samples: int = 12) -> None:
        test_loader = DataLoader(
            self.dataset.test_data,
            batch_size=100,
            shuffle=False,
            num_workers=os.cpu_count(),
        )

        self.model.eval()
        misclassified_images = []
        misclassified_labels = []
        misclassified_preds = []
        misclassified_confs = []

        with torch.inference_mode():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = self.model(images)
                probabilities = torch.softmax(outputs, dim=1)
                predictions = torch.argmax(probabilities, dim=1)
                confidences = torch.max(probabilities, dim=1)[0]

                mask = predictions != labels
                if mask.any():
                    misclassified_images.extend(images[mask].cpu())
                    misclassified_labels.extend(labels[mask].cpu())
                    misclassified_preds.extend(predictions[mask].cpu())
                    misclassified_confs.extend(confidences[mask].cpu())

                if len(misclassified_images) >= num_samples:
                    break

        if len(misclassified_images) == 0:
            print("No misclassifications found!")
            return

        num_to_show = min(num_samples, len(misclassified_images))
        rows = 3
        cols = 4
        fig, axes = plt.subplots(rows, cols, figsize=(16, 12))
        fig.suptitle(
            "Misclassified Examples - Analysis",
            fontsize=18,
            fontweight="bold",
            color="red",
        )

        for idx, ax in enumerate(axes.flat):
            if idx >= num_to_show:
                ax.axis("off")
                continue

            img = misclassified_images[idx].permute(1, 2, 0).numpy()
            true_label = self.class_names[misclassified_labels[idx].item()]
            pred_label = self.class_names[misclassified_preds[idx].item()]
            confidence = misclassified_confs[idx].item()

            ax.imshow(img)
            ax.set_title(
                f"True: {true_label}\nPredicted: {pred_label}\nConfidence: {confidence:.2%}",
                fontsize=10,
                fontweight="bold",
                color="darkred",
            )
            ax.axis("off")

        plt.tight_layout()
        plt.savefig(
            os.path.join(self.visuals_dir, "misclassifications.png"),
            dpi=300,
            bbox_inches="tight",
        )
        print(
            f"Misclassifications analysis saved to {self.visuals_dir}/misclassifications.png"
        )
        plt.close()

    def visualize_feature_maps(self, sample_image_path: str = "cat.jpg") -> None:
        if sample_image_path and os.path.exists(sample_image_path):
            from PIL import Image

            img = Image.open(sample_image_path).convert("RGB")
            transform = transforms.Compose(
                [transforms.Resize(self.model.imsize), transforms.ToTensor()]
            )
            image_tensor = transform(img).unsqueeze(0).to(device)
        else:
            test_loader = DataLoader(self.dataset.test_data, batch_size=1, shuffle=True)
            image_tensor, _ = next(iter(test_loader))
            image_tensor = image_tensor.to(device)

        self.model.eval()

        feature_maps = {}

        def get_features(name):
            def hook(model, input, output):
                feature_maps[name] = output.detach()

            return hook

        self.model.conv_layer_1.register_forward_hook(get_features("conv1"))
        self.model.conv_layer_2.register_forward_hook(get_features("conv2"))
        self.model.conv_layer_3.register_forward_hook(get_features("conv3"))

        with torch.inference_mode():
            _ = self.model(image_tensor)

        fig = plt.figure(figsize=(20, 12))
        fig.suptitle("Feature Maps Visualization", fontsize=18, fontweight="bold")

        # Original image in top-left
        ax = plt.subplot(3, 9, 1)
        img = image_tensor[0].cpu().permute(1, 2, 0).numpy()
        ax.imshow(img)
        ax.set_title("Original Image", fontsize=10, fontweight="bold")
        ax.axis("off")

        layers = [("conv1", 2), ("conv2", 11), ("conv3", 20)]
        for layer_idx, (layer_name, start_pos) in enumerate(layers):
            fmaps = feature_maps[layer_name][0].cpu()
            num_features = min(8, fmaps.shape[0])

            for i in range(num_features):
                subplot_idx = start_pos + i
                ax = plt.subplot(3, 9, subplot_idx)
                fmap = fmaps[i].numpy()
                ax.imshow(fmap, cmap="viridis")
                if i == 0:
                    ax.set_ylabel(
                        f"Layer {layer_idx + 1}",
                        fontsize=11,
                        fontweight="bold",
                        rotation=0,
                        labelpad=40,
                    )
                ax.set_title(f"Filter {i + 1}", fontsize=8)
                ax.axis("off")

        plt.tight_layout()
        plt.savefig(
            os.path.join(self.visuals_dir, "feature_maps.png"),
            dpi=300,
            bbox_inches="tight",
        )
        print(f"Feature maps saved to {self.visuals_dir}/feature_maps.png")
        plt.close()

    def generate_full_report(self) -> None:
        print("GENERATING COMPREHENSIVE VISUALIZATION REPORT")

        print("1. Creating training curves...")
        self.plot_training_curves()

        print("\n2. Generating confusion matrix...")
        cm, predictions, true_labels = self.plot_confusion_matrix()

        print("\n3. Creating classification report...")
        self.generate_classification_report(predictions, true_labels)

        print("\n4. Displaying sample predictions...")
        self.show_sample_predictions()

        print("\n5. Analyzing misclassifications...")
        self.analyze_misclassifications()

        print("\n6. Visualizing feature maps...")
        self.visualize_feature_maps()
