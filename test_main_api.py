
"""Integration tests for the main ICU FastAPI application."""

from fastapi.testclient import TestClient

from src.api.main import app


def main():
    print("\n========== MAIN FASTAPI APPLICATION TEST ==========\n")

    client = TestClient(app)

    # Test 1: Root endpoint
    print("Testing root endpoint...")

    response = client.get("/")

    assert response.status_code == 200, response.text

    result = response.json()

    assert result["version"] == "1.0.0"
    assert result["status"] == "running"
    assert result["documentation"] == "/docs"

    print("Root endpoint: SUCCESS")

    # Test 2: Main health endpoint
    print("\nTesting main health endpoint...")

    response = client.get("/health")

    assert response.status_code == 200, response.text

    result = response.json()

    assert result["status"] == "healthy"

    print("Main health endpoint: SUCCESS")

    # Test 3: Monitoring health endpoint
    print("\nTesting monitoring health endpoint...")

    response = client.get("/monitoring/health")

    assert response.status_code == 200, response.text

    result = response.json()

    assert result["status"] == "healthy"
    assert result["service"] == "ICU Monitoring API"

    print("Monitoring health endpoint: SUCCESS")

    # Test 4: Swagger documentation
    print("\nTesting Swagger documentation...")

    response = client.get("/docs")

    assert response.status_code == 200

    print("Swagger documentation: SUCCESS")

    # Test 5: OpenAPI schema
    print("\nTesting OpenAPI schema...")

    response = client.get("/openapi.json")

    assert response.status_code == 200

    schema = response.json()

    assert "openapi" in schema
    assert "paths" in schema

    print("OpenAPI schema: SUCCESS")

    # Test 6: Verify all API endpoints
    print("\nChecking registered API routes...")

    expected_routes = [
        "/",
        "/health",
        "/monitoring/health",
        "/monitoring/history",
        "/monitoring/trends",
    ]

    # Use the OpenAPI schema as the authoritative
    # list of documented application endpoints.
    registered_routes = schema["paths"]

    for route in expected_routes:
        assert route in registered_routes, (
            f"Missing API route: {route}"
        )

        print(f"{route}: REGISTERED")

    # Test 7: Verify HTTP methods
    print("\nChecking API HTTP methods...")

    assert "get" in registered_routes["/"]
    assert "get" in registered_routes["/health"]
    assert "get" in registered_routes["/monitoring/health"]
    assert "get" in registered_routes["/monitoring/history"]
    assert "get" in registered_routes["/monitoring/trends"]

    print("HTTP methods: SUCCESS")

    print(
        "\n========== MAIN FASTAPI TEST PASSED =========="
    )


if __name__ == "__main__":
    main()
