import io
import os

import torch
import uvicorn
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image
from torchvision import transforms

from data import Dataset
from model import CatDogCNN, device


def runModel(model: CatDogCNN, image_path: str) -> None:
    try:
        image = Image.open(image_path).convert("RGB")
        image_tensor = transform_image(image, model.imsize)

        prediction = predict_image(model, image_tensor)

        print(f"Image: {image_path}")
        print(f"Prediction: {prediction['label']}")
        print(f"Confidence: {prediction['confidence']:.2%}")
        print(
            f"Probabilities - Cat: {prediction['probabilities'][0, 0]:.2%}, Dog: {prediction['probabilities'][0, 1]:.2%}"
        )

    except FileNotFoundError:
        print(f"Error: Image file not found at {image_path}")
    except Exception as e:
        print(f"Error processing image: {str(e)}")


async def hostModel(model: CatDogCNN, port: int) -> None:
    app = FastAPI(title="Cat/Dog Classifier API")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.state.model = model
    app.state.device = device

    @app.post("/predict")
    async def predict(file: UploadFile = File(...)):
        """Predict whether the uploaded image is a cat or dog."""
        print(f"Received prediction request for: {file.filename}")

        try:
            # Read image data
            image_data = await file.read()

            if len(image_data) == 0:
                return JSONResponse({"error": "Empty file received"}, status_code=400)

            # Open and convert image
            try:
                image = Image.open(io.BytesIO(image_data)).convert("RGB")
                print(f"Image loaded: {image.size}")
            except Exception as img_error:
                return JSONResponse(
                    {"error": f"Invalid image file: {str(img_error)}"}, status_code=400
                )

            image_tensor = transform_image(image, app.state.model.imsize)
            prediction = predict_image(app.state.model, image_tensor)

            response_data = {
                "prediction": prediction["label"].lower(),
                "confidence": float(prediction["confidence"]),
                "probabilities": {
                    "cat": float(prediction["probabilities"][0, 0]),
                    "dog": float(prediction["probabilities"][0, 1]),
                },
            }

            print(
                f"Prediction: {prediction['label']}, Confidence: {prediction['confidence']:.4f}"
            )
            return JSONResponse(response_data, status_code=200)

        except Exception as e:
            print(f"Error during prediction: {str(e)}")
            return JSONResponse(
                {"error": f"Prediction failed: {str(e)}"}, status_code=500
            )

    print(f"Starting server on http://0.0.0.0:{port}")
    print(f"Using device: {device}")
    print(f"Image size: {model.imsize}")

    config = uvicorn.Config(app, host="0.0.0.0", port=port, log_level="info")
    server = uvicorn.Server(config)
    await server.serve()


def predict_image(model: CatDogCNN, im_tensor: torch.Tensor) -> dict:
    model.eval()
    with torch.inference_mode():
        logits = model(im_tensor)
        probs = torch.softmax(logits, dim=1)
        predicted_class = torch.argmax(probs, dim=1).item()

    class_names = ["Cat", "Dog"]

    return {
        "label": class_names[predicted_class],
        "confidence": probs[0, predicted_class].item(),
        "probabilities": torch.softmax(logits, dim=1),
    }


def transform_image(im: Image.Image, size: tuple[int, int]) -> torch.Tensor:
    transform = transforms.Compose(
        [
            transforms.Resize(size),
            transforms.ToTensor(),
        ]
    )

    image_tensor = transform(im)

    return image_tensor.unsqueeze(0).to(device)


def isPort(s: str) -> bool:
    if not s or not isinstance(s, str):
        return False

    s = s.strip()

    try:
        port = int(s)
        if 1 <= port <= 65535:
            return True
    except ValueError:
        pass

    return False


def isFile(s: str) -> bool:
    if not isPort(s):
        return os.path.exists(s)

    return False
