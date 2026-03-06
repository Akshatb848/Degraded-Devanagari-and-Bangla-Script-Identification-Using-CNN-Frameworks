"""
CNN-based Script Classifier.
Wraps the trained Devanagari/Bangla script identification models.

Supports:
- Custom CNN (from notebook - default)
- VGG16 transfer learning
- DenseNet121 transfer learning
- ResNet50 transfer learning
- AlexNet-style CNN
- Ensemble (majority voting)
"""
import io
import os
import threading
from typing import Optional, Tuple, Dict
import numpy as np
from PIL import Image
import structlog

logger = structlog.get_logger()

# Script labels matching the training data directory names
SCRIPT_LABELS = ["bangla", "devanagari"]
IMAGE_SIZE = (64, 64)


class ScriptClassifier:
    """
    Singleton CNN-based script classifier.
    Loads models lazily and caches them for reuse.
    """

    _instance: Optional["ScriptClassifier"] = None
    _lock = threading.Lock()
    _loaded = False

    def __init__(self):
        self.models: Dict[str, object] = {}
        self._try_load_models()

    @classmethod
    def get_instance(cls) -> "ScriptClassifier":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    @classmethod
    def is_loaded(cls) -> bool:
        return cls._loaded

    def _try_load_models(self) -> None:
        """Attempt to load available trained models."""
        from app.core.config import settings

        model_dir = settings.model_dir

        # Try loading Keras/TF models
        keras_paths = {
            "custom_cnn": os.path.join(model_dir, "script_classifier.keras"),
            "custom_cnn_h5": os.path.join(model_dir, "script_classifier.h5"),
            "vgg16": os.path.join(model_dir, "vgg16_script.keras"),
            "densenet121": os.path.join(model_dir, "densenet121_script.keras"),
            "resnet50": os.path.join(model_dir, "resnet50_script.keras"),
        }

        loaded_any = False
        for name, path in keras_paths.items():
            if os.path.exists(path):
                try:
                    import tensorflow as tf
                    model = tf.keras.models.load_model(path)
                    self.models[name] = ("keras", model)
                    logger.info("model_loaded", name=name, path=path)
                    loaded_any = True
                except Exception as e:
                    logger.warning("model_load_failed", name=name, error=str(e))

        if not loaded_any:
            logger.warning(
                "no_saved_models_found",
                model_dir=model_dir,
                message="Using fallback inference. Train models first with scripts/train_model.py",
            )
            # Build an untrained placeholder model for demo purposes
            self.models["custom_cnn"] = ("keras_untrained", self._build_default_model())

        ScriptClassifier._loaded = True

    def _build_default_model(self):
        """
        Build the custom CNN architecture from the research notebook.
        Architecture: 3x Conv2D → MaxPool → Dense → Dropout → Softmax
        """
        try:
            import tensorflow as tf

            model = tf.keras.Sequential([
                tf.keras.layers.Conv2D(32, (3, 3), activation="relu", input_shape=(64, 64, 3)),
                tf.keras.layers.MaxPooling2D(2, 2),
                tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
                tf.keras.layers.MaxPooling2D(2, 2),
                tf.keras.layers.Conv2D(128, (3, 3), activation="relu"),
                tf.keras.layers.MaxPooling2D(2, 2),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(128, activation="relu"),
                tf.keras.layers.Dropout(0.5),
                tf.keras.layers.Dense(2, activation="softmax"),
            ])
            model.compile(
                optimizer="adam",
                loss="categorical_crossentropy",
                metrics=["accuracy"],
            )
            logger.info("default_cnn_architecture_built")
            return model
        except ImportError:
            logger.warning("tensorflow_not_available")
            return None

    def _preprocess_image(self, image_bytes: bytes) -> np.ndarray:
        """Preprocess image to model input format: 64x64 RGB normalized."""
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        image = image.resize(IMAGE_SIZE, Image.LANCZOS)
        img_array = np.array(image, dtype=np.float32) / 255.0
        return np.expand_dims(img_array, axis=0)  # (1, 64, 64, 3)

    def _predict_with_keras(self, model, img_array: np.ndarray) -> Tuple[str, float]:
        """Run inference with a Keras model."""
        predictions = model.predict(img_array, verbose=0)
        class_idx = int(np.argmax(predictions[0]))
        confidence = float(predictions[0][class_idx])
        return SCRIPT_LABELS[class_idx], confidence

    def predict(
        self,
        image_bytes: bytes,
        model_name: str = "ensemble",
    ) -> Tuple[str, float, str]:
        """
        Classify script in the given image.

        Args:
            image_bytes: Raw image bytes
            model_name: 'ensemble', 'custom_cnn', 'vgg16', 'densenet121', 'resnet50'

        Returns:
            Tuple of (script_label, confidence, model_used)
        """
        img_array = self._preprocess_image(image_bytes)

        if model_name == "ensemble":
            return self._ensemble_predict(img_array)

        if model_name not in self.models:
            available = list(self.models.keys())
            logger.warning(
                "model_not_found",
                requested=model_name,
                available=available,
                fallback=available[0],
            )
            model_name = available[0]

        model_type, model = self.models[model_name]

        if model_type in ("keras", "keras_untrained"):
            if model is None:
                return "unknown", 0.0, model_name
            script, confidence = self._predict_with_keras(model, img_array)
            return script, confidence, model_name

        return "unknown", 0.0, model_name

    def _ensemble_predict(self, img_array: np.ndarray) -> Tuple[str, float, str]:
        """
        Ensemble prediction: majority vote across all loaded models.
        Confidence = mean confidence of all models for the winning class.
        """
        votes: Dict[str, list] = {"bangla": [], "devanagari": []}

        for name, (model_type, model) in self.models.items():
            if model_type in ("keras", "keras_untrained") and model is not None:
                try:
                    script, confidence = self._predict_with_keras(model, img_array)
                    votes[script].append(confidence)
                except Exception as e:
                    logger.warning("ensemble_model_skip", model=name, error=str(e))

        if not any(votes.values()):
            return "unknown", 0.0, "ensemble"

        # Pick the script with more votes; break ties by confidence
        bangla_score = (len(votes["bangla"]), np.mean(votes["bangla"]) if votes["bangla"] else 0.0)
        devanagari_score = (
            len(votes["devanagari"]),
            np.mean(votes["devanagari"]) if votes["devanagari"] else 0.0,
        )

        if bangla_score >= devanagari_score:
            winner = "bangla"
            confidence = float(bangla_score[1])
        else:
            winner = "devanagari"
            confidence = float(devanagari_score[1])

        return winner, confidence, "ensemble"

    @property
    def available_models(self) -> list:
        return list(self.models.keys())
