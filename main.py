import argparse
import asyncio
import json
import os
import sys
import warnings

import torch
from PIL import ImageFile

warnings.filterwarnings("ignore", category=UserWarning, module="PIL")
warnings.filterwarnings("ignore", ".*Truncated File Read.*")
ImageFile.LOAD_TRUNCATED_IMAGES = True

from data import Dataset, cat_dog_download
from inference import hostModel, isFile, isPort, runModel
from model import CatDogCNN, device
from train import train
from visualizer import Visualizer


def handleData(args: argparse.Namespace) -> Dataset:
    dataset_path = cat_dog_download(args.kaggle_creds)
    if dataset_path == "":
        print("Unable to find or download datset. Exiting...")
        sys.exit(1)

    dataset = Dataset(dataset_path, imsize=args.image_size)

    if args.visualize:
        dataset.show_samples(
            os.path.join(args.models_dir, "dataset_samples_before.png")
        )

    dataset.preprocess()

    if args.visualize:
        dataset.show_samples(
            os.path.join(args.models_dir, "dataset_samples_after.png"),
            train_data=True,
        )

    return dataset


def handleModel(args: argparse.Namespace) -> CatDogCNN:
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("Unable to find torch compatible AMD or NVIDIA GPU. Exiting...")
        sys.exit(1)

    model = CatDogCNN(imsize=args.image_size)

    if args.visualize:
        model.summary()

    return model.to(device)


def handleInference(args: argparse.Namespace, conf: dict | None = None) -> None:
    if conf is None or conf["model_path"] is None:
        print("'model_path' not set in conf.json")
        return

    model = CatDogCNN(imsize=args.image_size)
    model.load_state_dict(torch.load(conf["model_path"], map_location=device))
    model.to(device)
    model.eval()

    inference_value = args.inference
    if isPort(inference_value):
        asyncio.run(hostModel(model, int(inference_value)))
    elif isFile(inference_value):
        runModel(model, inference_value)

        if args.visualize:
            with open(os.path.join(args.models_dir, "res.json"), "r") as file:
                res = json.load(file)

            dataset_path = cat_dog_download(args.kaggle_creds)
            if dataset_path == "":
                print("Unable to find or download datset. Exiting...")
                sys.exit(1)

            dataset = Dataset(dataset_path, imsize=args.image_size)
            dataset.preprocess()

            visualizer = Visualizer(model, dataset, res, args.models_dir)
            visualizer.generate_full_report()

    else:
        print(
            "Unable to inference model since --inference [value] is neither a port or file."
        )


def main(args: argparse.Namespace, conf: dict) -> None:
    if args.inference:
        handleInference(args, conf)
        return
    dataset = handleData(args)
    model = handleModel(args)

    res = train(model, dataset, models_dir=args.models_dir)
    with open(os.path.join(args.models_dir, "res.json"), "w") as json_file:
        json.dump(res, json_file, indent=4)

    if args.visualize:
        visualizer = Visualizer(model, dataset, res, args.models_dir)
        visualizer.generate_full_report()


if __name__ == "__main__":
    with open("conf.json", "r") as file:
        conf = json.load(file)

    try:
        import google.colab

        image_size = (64, 64)
        print(f"In Colab, using image size: '{image_size[0]}x{image_size[1]}px'")
    except ImportError:
        vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        if vram < 10.0:  # Needs ~8.5 Gb for 256x256
            image_size = (128, 128)
        else:
            image_size = (256, 256)

        print(
            f"Not in Colab with {vram} GB VRAM, default image size: '{image_size[0]}x{image_size[1]}px'"
        )

    parser = argparse.ArgumentParser(
        prog="CAP4613 Project",
        description="Cat Dog CNN Model Training",
    )
    parser.add_argument(
        "-v", "--visualize", action="store_true", help="Enable visualization"
    )
    parser.add_argument(
        "-i",
        "--inference",
        type=str,
        help="Inference the model from conf.model_path with an image '-i <image_path>' or host with -i <port>",
    )
    parser.add_argument(
        "-k",
        "--kaggle_creds",
        type=str,
        default=conf["kaggle_creds"],
        help="Input Kaggle Credentials",
    )
    parser.add_argument(
        "-o",
        "--models_dir",
        type=str,
        default=conf["models_dir"],
        help="Output directory for the trained models",
    )
    parser.add_argument(
        "-im",
        "--image_size",
        type=lambda s: tuple(map(int, s.split("x"))),
        default=image_size,
        help="Image size as WIDTHxHEIGHT (e.g., '256x256')",
    )

    args = parser.parse_args()

    if args.visualize:
        print("Visualization enabled.")

    if args.inference:
        print(f"Inferencing the conf['model_path'] model with/on {args.inference}")

    if args.kaggle_creds != conf["kaggle_creds"]:
        print(
            f"Kaggle credentials override: {conf['kaggle_creds']} -> {args.kaggle_creds}"
        )

    if args.models_dir != conf["models_dir"]:
        print(f"Models directory override: {conf['models_dir']} -> {args.models_dir}")

    if args.image_size[0] != image_size[0]:
        print(f"Image size override: {image_size} -> {args.image_size}")

    os.makedirs(args.models_dir, exist_ok=True)

    main(args, conf=conf)
