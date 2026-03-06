"""
Pytest configuration and shared fixtures.
"""
import pytest
import os

# Set test environment variables before importing app
os.environ["APP_ENV"] = "test"
os.environ["DEBUG"] = "true"
os.environ["DATABASE_URL"] = "postgresql://test:test@localhost:5432/test_db"
os.environ["REDIS_URL"] = "redis://localhost:6379/15"
os.environ["CELERY_BROKER_URL"] = "redis://localhost:6379/14"
os.environ["CELERY_RESULT_BACKEND"] = "redis://localhost:6379/13"
os.environ["MODEL_DIR"] = "/tmp/test_models"
os.environ["CNN_MODEL_PATH"] = "/tmp/test_models/script_classifier.keras"


@pytest.fixture(autouse=True)
def create_test_model_dir():
    """Create the test model directory."""
    os.makedirs("/tmp/test_models", exist_ok=True)
