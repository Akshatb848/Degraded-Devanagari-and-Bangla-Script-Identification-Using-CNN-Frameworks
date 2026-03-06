"""
Training script for CNN script classifiers.
Trains multiple architectures and saves the best model.

Architectures:
- Custom CNN (from research notebook)
- VGG16 (transfer learning)
- DenseNet121 (transfer learning)
- ResNet50 (transfer learning)

Usage:
    python scripts/train_model.py --data-dir /path/to/data --model custom_cnn
    python scripts/train_model.py --data-dir /path/to/data --model all
"""
import argparse
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import structlog

logger = structlog.get_logger()

IMAGE_SIZE = (64, 64)
NUM_CLASSES = 2
BATCH_SIZE = 32
EPOCHS = 20
SCRIPT_LABELS = ["bangla", "devanagari"]


def build_custom_cnn(input_shape=(64, 64, 3)) -> keras.Model:
    """
    Custom CNN architecture from the research notebook.
    3x Conv2D blocks with MaxPooling → Dense → Dropout → Softmax
    Achieves ~99% accuracy on the Ekush dataset.
    """
    model = keras.Sequential([
        layers.Input(shape=input_shape),
        layers.Conv2D(32, (3, 3), activation="relu"),
        layers.MaxPooling2D(2, 2),
        layers.Conv2D(64, (3, 3), activation="relu"),
        layers.MaxPooling2D(2, 2),
        layers.Conv2D(128, (3, 3), activation="relu"),
        layers.MaxPooling2D(2, 2),
        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(NUM_CLASSES, activation="softmax"),
    ], name="custom_cnn")

    return model


def build_vgg16(input_shape=(64, 64, 3)) -> keras.Model:
    """VGG16 with frozen base + fine-tuned top layers."""
    base = keras.applications.VGG16(
        weights="imagenet",
        include_top=False,
        input_shape=input_shape,
    )
    # Freeze all base layers
    base.trainable = False

    model = keras.Sequential([
        base,
        layers.GlobalAveragePooling2D(),
        layers.Dense(256, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(NUM_CLASSES, activation="softmax"),
    ], name="vgg16")

    return model


def build_densenet121(input_shape=(64, 64, 3)) -> keras.Model:
    """DenseNet121 with frozen base + fine-tuned classifier."""
    base = keras.applications.DenseNet121(
        weights="imagenet",
        include_top=False,
        input_shape=input_shape,
    )
    base.trainable = False

    model = keras.Sequential([
        base,
        layers.GlobalAveragePooling2D(),
        layers.Dense(256, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(NUM_CLASSES, activation="softmax"),
    ], name="densenet121")

    return model


def build_resnet50(input_shape=(64, 64, 3)) -> keras.Model:
    """ResNet50 with frozen base + fine-tuned classifier."""
    base = keras.applications.ResNet50(
        weights="imagenet",
        include_top=False,
        input_shape=input_shape,
    )
    base.trainable = False

    model = keras.Sequential([
        base,
        layers.GlobalAveragePooling2D(),
        layers.Dense(256, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(NUM_CLASSES, activation="softmax"),
    ], name="resnet50")

    return model


def get_data_generators(data_dir: str, val_split: float = 0.2):
    """Create training and validation data generators with augmentation."""
    train_gen = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode="nearest",
        validation_split=val_split,
    )

    val_gen = ImageDataGenerator(
        rescale=1.0 / 255,
        validation_split=val_split,
    )

    train_data = train_gen.flow_from_directory(
        data_dir,
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        subset="training",
        shuffle=True,
    )

    val_data = val_gen.flow_from_directory(
        data_dir,
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        subset="validation",
        shuffle=False,
    )

    return train_data, val_data


def train_model(
    model: keras.Model,
    train_data,
    val_data,
    model_name: str,
    output_dir: str,
) -> dict:
    """Compile, train, and save a model. Returns training history."""
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    logger.info("model_summary", model=model_name, params=model.count_params())
    model.summary()

    save_path = os.path.join(output_dir, f"{model_name}_script.keras")
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True),
        ModelCheckpoint(save_path, monitor="val_accuracy", save_best_only=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6),
    ]

    history = model.fit(
        train_data,
        validation_data=val_data,
        epochs=EPOCHS,
        callbacks=callbacks,
        verbose=1,
    )

    # Evaluate
    val_loss, val_acc = model.evaluate(val_data, verbose=0)
    logger.info("training_complete", model=model_name, val_accuracy=val_acc, val_loss=val_loss)

    # Also save best model as default classifier
    if model_name == "custom_cnn":
        default_path = os.path.join(output_dir, "script_classifier.keras")
        model.save(default_path)
        logger.info("default_model_saved", path=default_path)

    return {
        "model": model_name,
        "val_accuracy": float(val_acc),
        "val_loss": float(val_loss),
        "saved_to": save_path,
    }


def main():
    parser = argparse.ArgumentParser(description="Train AIOCR script classifier models")
    parser.add_argument("--data-dir", required=True, help="Path to data directory with Bangla/ and Devanagari/ subdirs")
    parser.add_argument("--output-dir", default="saved_models", help="Directory to save trained models")
    parser.add_argument(
        "--model",
        choices=["custom_cnn", "vgg16", "densenet121", "resnet50", "all"],
        default="custom_cnn",
        help="Model architecture to train",
    )
    parser.add_argument("--epochs", type=int, default=EPOCHS, help="Maximum training epochs")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="Training batch size")

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Update global training config
    global EPOCHS, BATCH_SIZE
    EPOCHS = args.epochs
    BATCH_SIZE = args.batch_size

    logger.info("training_start", data_dir=args.data_dir, model=args.model)

    train_data, val_data = get_data_generators(args.data_dir)
    logger.info("data_loaded", train_samples=train_data.samples, val_samples=val_data.samples)

    model_builders = {
        "custom_cnn": build_custom_cnn,
        "vgg16": build_vgg16,
        "densenet121": build_densenet121,
        "resnet50": build_resnet50,
    }

    models_to_train = list(model_builders.keys()) if args.model == "all" else [args.model]
    results = []

    for model_name in models_to_train:
        logger.info("building_model", model=model_name)
        model = model_builders[model_name]()
        result = train_model(model, train_data, val_data, model_name, args.output_dir)
        results.append(result)

    print("\n=== Training Results ===")
    for r in results:
        print(f"{r['model']}: val_accuracy={r['val_accuracy']:.4f}, saved_to={r['saved_to']}")


if __name__ == "__main__":
    main()
