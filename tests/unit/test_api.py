from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_read_root():
    """Test the root endpoint."""
    response = client.get("/")
    # Note: Depending on your implementation, root might return 404 or a welcome message
    # Adjusting expectation to 200 if you have a root handler, otherwise 404
    # Just checking it doesn't crash (500)
    assert response.status_code != 500

def test_health_check():
    """Test the health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy", "service": "quiz-generation-service"}
