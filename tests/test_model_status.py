"""
Model status: the classifier must only report 'trained' when real weights load.
The TensorFlow-dependent test is skipped when TensorFlow is not installed
(it lives in requirements-train.txt, not the runtime requirements).
"""
import io

import pytest
from PIL import Image


def _png_bytes() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (64, 64), color=(200, 200, 200)).save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture
def fresh_classifier(monkeypatch, tmp_path):
    from app.core.config import settings
    from models.cnn_classifier import ScriptClassifier

    monkeypatch.setattr(settings, "model_dir", str(tmp_path))
    monkeypatch.setattr(ScriptClassifier, "_instance", None)
    monkeypatch.setattr(ScriptClassifier, "_loaded", False)
    return tmp_path


def test_empty_model_dir_is_untrained_fallback(fresh_classifier):
    from models.cnn_classifier import ScriptClassifier, UNTRAINED_MODEL_NAME

    clf = ScriptClassifier.get_instance()
    assert ScriptClassifier.model_status() == "untrained_fallback"
    assert clf.predict(_png_bytes()) == ("unknown", 0.0, UNTRAINED_MODEL_NAME)


def test_saved_weights_report_trained(fresh_classifier):
    pytest.importorskip("tensorflow", reason="TensorFlow not installed (see requirements-train.txt)")
    from scripts.train_model import build_custom_cnn
    from models.cnn_classifier import ScriptClassifier

    build_custom_cnn().save(str(fresh_classifier / "script_classifier.keras"))

    clf = ScriptClassifier.get_instance()
    assert ScriptClassifier.model_status() == "trained"
    script, conf, model_used = clf.predict(_png_bytes(), model_name="custom_cnn")
    assert script in ("bangla", "devanagari")
    assert 0.0 <= conf <= 1.0
    assert model_used == "custom_cnn"
