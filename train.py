import os
from timeit import default_timer as timer
from typing import Any, Literal, overload

import torch
import torch.nn as nn
from tqdm import tqdm

from data import Dataset
from model import device


def train(
    model: torch.nn.Module, dataset: Dataset, models_dir: str = "./models"
) -> dict[str, list[Any]]:
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    train_dataloader_augmented = dataset.augment("train")
    val_dataloader_augmented = dataset.augment("val")

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=1e-3)

    start_time = timer()

    model_results = _train(
        model=model,
        train_dataloader=train_dataloader_augmented,
        val_dataloader=val_dataloader_augmented,
        optimizer=optimizer,
        loss_fn=loss_fn,
        models_dir=models_dir,
        epochs=25,
    )

    end_time = timer()
    print(f"Total training time: {end_time - start_time:.3f} seconds")

    return model_results


def _train(
    model: torch.nn.Module,
    train_dataloader: torch.utils.data.DataLoader,
    val_dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: torch.nn.Module = nn.CrossEntropyLoss(),
    models_dir: str = "./models",
    epochs: int = 25,
) -> dict[str, list[Any]]:
    results = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    for epoch in tqdm(range(epochs)):
        train_loss, train_acc = _train_step(
            model=model,
            dataloader=train_dataloader,
            loss_fn=loss_fn,
            optimizer=optimizer,
        )
        val_loss, val_acc = _val_step(
            model=model, dataloader=val_dataloader, loss_fn=loss_fn, return_preds=False
        )

        tqdm.write(
            f"Epoch: {epoch + 1} | "
            f"train_loss: {train_loss:.4f} | "
            f"train_acc: {train_acc:.4f} | "
            f"val_loss: {val_loss:.4f} | "
            f"val_acc: {val_acc:.4f}"
        )

        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["val_loss"].append(val_loss)
        results["val_acc"].append(val_acc)

        torch.save(
            model.state_dict(), os.path.join(models_dir, f"cat_dog_cnn_{epoch + 1}.pth")
        )

    return results


def _train_step(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    loss_fn: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> tuple[float, float]:
    model.train()

    train_loss, train_acc = 0, 0
    for _, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # 1. Forward pass
        y_pred = model(X)

        # 2. Calculate  and accumulate loss
        loss = loss_fn(y_pred, y)
        train_loss += loss.item()

        # 3. Optimizer zero grad
        optimizer.zero_grad()

        # 4. Loss backward
        loss.backward()

        # 5. Optimizer step
        optimizer.step()

        y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
        train_acc += (y_pred_class == y).sum().item() / len(y_pred)

    train_loss = train_loss / len(dataloader)
    train_acc = train_acc / len(dataloader)
    return train_loss, train_acc


@overload
def _val_step(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    loss_fn: torch.nn.Module,
    return_preds: Literal[False] = False,
) -> tuple[float, float]: ...


@overload
def _val_step(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    loss_fn: torch.nn.Module,
    return_preds: Literal[True],
) -> tuple[float, float, list[int], list[int]]: ...


def _val_step(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    loss_fn: torch.nn.Module,
    return_preds: bool = False,
) -> tuple[float, float] | tuple[float, float, list[int], list[int]]:
    model.eval()
    val_loss, val_acc = 0, 0
    all_preds, all_labels = [], []

    with torch.inference_mode():
        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)
            val_pred_logits = model(X)
            loss = loss_fn(val_pred_logits, y)
            val_loss += loss.item()
            val_pred_labels = val_pred_logits.argmax(dim=1)
            val_acc += (val_pred_labels == y).sum().item() / len(val_pred_labels)

            if return_preds:
                all_preds.extend(val_pred_labels.cpu().numpy())
                all_labels.extend(y.cpu().numpy())

    val_loss = val_loss / len(dataloader)
    val_acc = val_acc / len(dataloader)

    if return_preds:
        return val_loss, val_acc, all_preds, all_labels
    return val_loss, val_acc
