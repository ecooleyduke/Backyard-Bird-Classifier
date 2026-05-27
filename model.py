import numpy as np
from PIL import Image
from keras.applications.efficientnet import preprocess_input
from keras.models import load_model


IMG_SIZE = 224
MODEL_PATH = "weights/bird_model.keras"


def load_model():
    """
    Loads fine-tuned EfficientNet model.
    """
    model = load_model(MODEL_PATH)
    print("Successfully loaded weights")
    return {"model": model, "backend": "trained"}


def preprocess(img: Image.Image):
    img = img.convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    arr = np.array(img, dtype=np.float32)
    arr = preprocess_input(arr)
    return np.expand_dims(arr, axis=0)


def predict(model_dict, img: Image.Image):
    model = model_dict["model"]
    x = preprocess(img)
    probs = model.predict(x, verbose=0)[0]
    return probs.astype(np.float32)