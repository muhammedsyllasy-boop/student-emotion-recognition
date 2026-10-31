"""
Train a CNN using either:
1) the original FER-2013 CSV at data/fer2013.csv, or
2) the image-folder FER dataset ZIP at data/fer2013_images.zip.
Run from the project root: python train.py
"""
import os
import json
import random
import zipfile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

DATA_DIR = "data"
CSV_PATH = os.path.join(DATA_DIR, "fer2013.csv")
IMAGE_ZIP_PATH = os.path.join(DATA_DIR, "fer2013_images.zip")
IMAGE_ROOT = os.path.join(DATA_DIR, "fer2013_images")
MODEL_DIR = "models"
OUTPUT_DIR = "outputs"
IMG_SIZE = 48
NUM_CLASSES = 7
CLASS_NAMES = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
CLASS_TO_ID = {name.lower(): i for i, name in enumerate(CLASS_NAMES)}

def parse_pixels(pixel_string):
    values = np.fromstring(str(pixel_string), sep=" ", dtype=np.float32)
    if values.size != IMG_SIZE * IMG_SIZE:
        raise ValueError(f"Expected 2304 pixel values, got {values.size}.")
    return values.reshape(IMG_SIZE, IMG_SIZE, 1)

def load_csv_dataset(path):
    df = pd.read_csv(path)
    columns = {c.strip().lower(): c for c in df.columns}
    for required in ("emotion", "pixels", "usage"):
        if required not in columns:
            raise ValueError(f"CSV must contain emotion, pixels, and Usage columns. Found: {list(df.columns)}")
    emotion_col, pixels_col, usage_col = columns["emotion"], columns["pixels"], columns["usage"]
    df = df.dropna(subset=[emotion_col, pixels_col, usage_col]).copy()
    df[emotion_col] = pd.to_numeric(df[emotion_col], errors="coerce")
    df = df.dropna(subset=[emotion_col])
    df[emotion_col] = df[emotion_col].astype(int)
    df = df[df[emotion_col].between(0, 6)]
    images, labels, usages = [], [], []
    for _, row in df.iterrows():
        try:
            images.append(parse_pixels(row[pixels_col]))
            labels.append(int(row[emotion_col]))
            usages.append(str(row[usage_col]).strip().lower())
        except ValueError:
            continue
    X = np.asarray(images, dtype=np.float32) / 255.0
    y = np.asarray(labels, dtype=np.int64)
    usage = np.asarray(usages)
    train_mask = usage == "training"
    test_mask = usage == "privatetest"
    if not test_mask.any():
        test_mask = usage == "publictest"
    X_train_all, y_train_all = X[train_mask], y[train_mask]
    if len(y_train_all) == 0:
        raise ValueError("No rows with Usage='Training' found in CSV.")
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_all, y_train_all, test_size=0.10, random_state=SEED, stratify=y_train_all
    )
    if not test_mask.any():
        raise ValueError("No PrivateTest or PublicTest rows found in CSV.")
    return X_train, y_train, X_val, y_val, X[test_mask], y[test_mask]

def ensure_image_dataset():
    if os.path.isdir(os.path.join(IMAGE_ROOT, "train")) and os.path.isdir(os.path.join(IMAGE_ROOT, "test")):
        return
    if not os.path.isfile(IMAGE_ZIP_PATH):
        raise FileNotFoundError(
            "Dataset missing. Add either data/fer2013.csv or the image-folder dataset ZIP "
            "as data/fer2013_images.zip."
        )
    os.makedirs(IMAGE_ROOT, exist_ok=True)
    print("Extracting image-folder dataset ZIP (this may take a minute)...")
    with zipfile.ZipFile(IMAGE_ZIP_PATH, "r") as archive:
        archive.extractall(IMAGE_ROOT)
    if not (os.path.isdir(os.path.join(IMAGE_ROOT, "train")) and os.path.isdir(os.path.join(IMAGE_ROOT, "test"))):
        raise ValueError("The ZIP must contain train/<emotion>/ and test/<emotion>/ folders.")

def read_image_folder(split):
    split_dir = os.path.join(IMAGE_ROOT, split)
    images, labels = [], []
    for folder in os.listdir(split_dir):
        label = CLASS_TO_ID.get(folder.lower())
        folder_path = os.path.join(split_dir, folder)
        if label is None or not os.path.isdir(folder_path):
            continue
        for filename in os.listdir(folder_path):
            if not filename.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                continue
            path = os.path.join(folder_path, filename)
            try:
                with Image.open(path) as image:
                    image = image.convert("L").resize((IMG_SIZE, IMG_SIZE))
                    arr = np.asarray(image, dtype=np.float32) / 255.0
                images.append(arr[..., np.newaxis])
                labels.append(label)
            except (OSError, ValueError):
                print(f"Skipping unreadable image: {path}")
    if not images:
        raise ValueError(f"No supported images found in {split_dir}.")
    return np.asarray(images, dtype=np.float32), np.asarray(labels, dtype=np.int64)

def load_image_folder_dataset():
    ensure_image_dataset()
    X_train_all, y_train_all = read_image_folder("train")
    X_test, y_test = read_image_folder("test")
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_all, y_train_all, test_size=0.10, random_state=SEED, stratify=y_train_all
    )
    return X_train, y_train, X_val, y_val, X_test, y_test

def load_dataset():
    if os.path.isfile(CSV_PATH):
        print("Using FER-2013 CSV dataset.")
        return load_csv_dataset(CSV_PATH)
    print("CSV not found; using the supplied image-folder dataset.")
    return load_image_folder_dataset()

def build_model():
    inputs = tf.keras.Input(shape=(48, 48, 1), name="face_image")
    x = tf.keras.layers.Conv2D(32, 3, padding="same", activation="relu")(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv2D(32, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.MaxPooling2D()(x)
    x = tf.keras.layers.Dropout(0.25)(x)
    x = tf.keras.layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.MaxPooling2D()(x)
    x = tf.keras.layers.Dropout(0.30)(x)
    x = tf.keras.layers.Conv2D(128, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling2D()(x)
    x = tf.keras.layers.Dropout(0.35)(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.40)(x)
    outputs = tf.keras.layers.Dense(NUM_CLASSES, activation="softmax", name="expression_probabilities")(x)
    model = tf.keras.Model(inputs, outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model

def main():
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    X_train, y_train, X_val, y_val, X_test, y_test = load_dataset()
    print(f"Training: {len(y_train)} | Validation: {len(y_val)} | Test: {len(y_test)}")
    model = build_model()
    model.summary()
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint(os.path.join(MODEL_DIR, "best_model.keras"),
                                           monitor="val_loss", save_best_only=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6)
    ]
    history = model.fit(X_train, y_train, validation_data=(X_val, y_val),
                        epochs=25, batch_size=64, callbacks=callbacks, verbose=1)
    model = tf.keras.models.load_model(os.path.join(MODEL_DIR, "best_model.keras"))
    test_loss, test_accuracy = model.evaluate(X_test, y_test, batch_size=64, verbose=0)
    probabilities = model.predict(X_test, batch_size=64, verbose=0)
    y_pred = np.argmax(probabilities, axis=1)
    report = classification_report(y_test, y_pred, labels=list(range(7)),
                                   target_names=CLASS_NAMES, output_dict=True, zero_division=0)
    with open(os.path.join(OUTPUT_DIR, "classification_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    with open(os.path.join(OUTPUT_DIR, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump({"test_loss": float(test_loss), "test_accuracy": float(test_accuracy),
                   "test_samples": int(len(y_test))}, f, indent=2)
    print("\nFINAL TEST RESULTS")
    print(f"Test loss: {test_loss:.4f}")
    print(f"Test accuracy: {test_accuracy:.4f}")
    print(classification_report(y_test, y_pred, labels=list(range(7)),
                                target_names=CLASS_NAMES, zero_division=0))
    cm = confusion_matrix(y_test, y_pred, labels=list(range(7)))
    plt.figure(figsize=(9, 7))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.xlabel("Predicted label"); plt.ylabel("True label")
    plt.title("FER-2013 Test Confusion Matrix"); plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "confusion_matrix.png"), dpi=160); plt.close()
    for key, val_key, title, filename in [
        ("accuracy", "val_accuracy", "Training and Validation Accuracy", "accuracy_curve.png"),
        ("loss", "val_loss", "Training and Validation Loss", "loss_curve.png")
    ]:
        plt.figure(figsize=(10, 4))
        plt.plot(history.history[key], label="Training " + key)
        plt.plot(history.history[val_key], label="Validation " + key)
        plt.xlabel("Epoch"); plt.ylabel(key.title()); plt.title(title); plt.legend()
        plt.tight_layout(); plt.savefig(os.path.join(OUTPUT_DIR, filename), dpi=160); plt.close()
    print("\nSaved model and evaluation outputs in models/ and outputs/.")

if __name__ == "__main__":
    main()
