"""Reusable prediction helpers for the Student Emotion Recognition project."""
import numpy as np
from PIL import Image, ImageOps

CLASS_NAMES = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

def preprocess_image(image):
    """Convert a PIL image to the 48x48 grayscale input expected by the CNN."""
    image = ImageOps.exif_transpose(image).convert("L")
    image = ImageOps.fit(image, (48, 48), method=Image.Resampling.LANCZOS)
    arr = np.asarray(image, dtype=np.float32) / 255.0
    return arr.reshape(1, 48, 48, 1)

def predict_expression(model, image):
    batch = preprocess_image(image)
    probabilities = model.predict(batch, verbose=0)[0]
    order = np.argsort(probabilities)[::-1]
    return {
        "label": CLASS_NAMES[int(order[0])],
        "confidence": float(probabilities[order[0]]),
        "probabilities": {CLASS_NAMES[int(i)]: float(probabilities[i]) for i in order}
    }
