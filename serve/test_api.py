import json
import time
from fastapi.testclient import TestClient
from serve.app import app, REQ_LOG_FILE

def test_deployment():
    print("Testing FastAPI app endpoints with TestClient...")
    with TestClient(app) as client:
        # 1. Health check
        res = client.get("/health")
        print("GET /health ->", res.status_code, res.json())
        assert res.status_code == 200

        # 2. Safe request
        res = client.post("/generate", json={"request": "list all files including hidden ones"})
        print("POST /generate (safe) ->", res.status_code, res.json())
        assert res.status_code == 200
        assert res.json()["dangerous"] == False

        # 3. Dangerous request
        res = client.post("/generate", json={"request": "delete all files recursively in root"})
        print("POST /generate (risky) ->", res.status_code, res.json())
        assert res.status_code == 200

        # 4. Metrics endpoint
        res = client.get("/metrics")
        print("GET /metrics ->", res.status_code, json.dumps(res.json(), indent=2))
        assert res.status_code == 200
        assert res.json()["request_count"] >= 2

    print("All API endpoints tested successfully!")

if __name__ == "__main__":
    test_deployment()

