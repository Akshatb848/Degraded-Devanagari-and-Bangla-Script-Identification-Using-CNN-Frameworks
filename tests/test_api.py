"""
API integration tests using TestClient.
"""
import io
import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    """Create test client with mocked model loading."""
    with patch("models.cnn_classifier.ScriptClassifier._try_load_models"):
        from app.main import app
        with TestClient(app) as c:
            yield c


@pytest.fixture
def sample_image_bytes():
    """Generate a minimal valid PNG image for testing."""
    from PIL import Image
    import io
    img = Image.new("RGB", (64, 64), color=(200, 200, 200))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


class TestHealthEndpoint:
    def test_health_returns_200(self, client):
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_response_structure(self, client):
        response = client.get("/health")
        data = response.json()
        assert "status" in data
        assert "models_loaded" in data
        assert "timestamp" in data


class TestRootEndpoint:
    def test_root_returns_200(self, client):
        response = client.get("/")
        assert response.status_code == 200

    def test_root_contains_version(self, client):
        response = client.get("/")
        data = response.json()
        assert "version" in data
        assert data["version"] == "1.0.0"


class TestScriptDetectionEndpoint:
    @patch("models.cnn_classifier.ScriptClassifier.predict")
    def test_detect_script_success(self, mock_predict, client, sample_image_bytes):
        mock_predict.return_value = ("devanagari", 0.99, "custom_cnn")

        response = client.post(
            "/api/v1/detect-script/",
            files={"file": ("test.png", sample_image_bytes, "image/png")},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["script"] == "devanagari"
        assert data["confidence"] == 0.99
        assert "request_id" in data
        assert "processing_time_ms" in data

    def test_detect_script_rejects_non_image(self, client):
        response = client.post(
            "/api/v1/detect-script/",
            files={"file": ("test.txt", b"hello world", "text/plain")},
        )
        assert response.status_code == 400

    def test_detect_script_missing_file(self, client):
        response = client.post("/api/v1/detect-script/")
        assert response.status_code == 422


class TestSchemas:
    def test_script_type_enum_values(self):
        from app.api.models.schemas import ScriptType
        assert ScriptType.DEVANAGARI == "devanagari"
        assert ScriptType.BANGLA == "bangla"
        assert ScriptType.UNKNOWN == "unknown"

    def test_bounding_box_validation(self):
        from app.api.models.schemas import BoundingBox
        bb = BoundingBox(x=10, y=20, width=100, height=50, confidence=0.95)
        assert bb.x == 10
        assert bb.confidence == 0.95

    def test_bounding_box_confidence_validation(self):
        from app.api.models.schemas import BoundingBox
        import pydantic
        with pytest.raises(pydantic.ValidationError):
            BoundingBox(x=0, y=0, width=10, height=10, confidence=1.5)  # confidence > 1.0

    def test_pipeline_response_structure(self):
        from app.api.models.schemas import PipelineResponse, ScriptType, ProcessingStatus
        response = PipelineResponse(
            status=ProcessingStatus.COMPLETED,
            script=ScriptType.DEVANAGARI,
            raw_text="भारत",
            corrected_text="भारत",
            overall_confidence=0.95,
            text_regions=[],
            bounding_boxes=[],
            language="Hindi/Sanskrit/Marathi",
            reasoning="Test reasoning",
            corrections_made=[],
            agent_statuses=[],
            processing_time_ms=100.0,
        )
        assert response.script == ScriptType.DEVANAGARI
        assert response.overall_confidence == 0.95
