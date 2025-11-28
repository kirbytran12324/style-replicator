# test_api_local_fix_auth.py
from fastapi.testclient import TestClient
import run_modal

# -------------------------------
# Override dependencies for TestClient
# -------------------------------
def override_get_current_user_id():
    return "test_user_local"

run_modal.api.dependency_overrides[run_modal.get_current_user_id] = override_get_current_user_id

# -------------------------------
# Monkeypatch remote_generate for local test
# -------------------------------
import types
from PIL import Image
import io

async def fake_remote_generate(prompt, num_samples=1, lora_path=None):
    images = []
    for i in range(num_samples):
        img = Image.new("RGB", (256, 256), color=(i*20 % 256, 100, 150))
        with io.BytesIO() as buf:
            img.save(buf, format="PNG")
            images.append(buf.getvalue())
    return images

run_modal.remote_generate_func = types.SimpleNamespace(remote=types.SimpleNamespace(aio=fake_remote_generate))

# -------------------------------
# Use TestClient
# -------------------------------
client = TestClient(run_modal.api)

def test_root():
    resp = client.get("/api")
    print("Root endpoint:", resp.json())

def test_generate():
    request_data = {
        "prompt": "A cute robot sitting on a bench",
        "num_samples": 2,
        "lora_path": None
    }
    resp = client.post("/api/generate", json=request_data)
    print("/api/generate response keys:", resp.json().keys())
    images_b64 = resp.json().get("images", [])
    print(f"Generated {len(images_b64)} dummy images.")

def test_train():
    request_data = {
        "config": {"dummy": "value"},
        "recover": False,
        "name": "test_job_local"
    }
    resp = client.post("/api/train", json=request_data)
    print("/api/train response:", resp.json())

def test_jobs():
    resp = client.get("/api/jobs")
    print("All user jobs:", resp.json())

# -------------------------------
# Run tests
# -------------------------------
if __name__ == "__main__":
    test_root()
    test_generate()
    test_train()
    test_jobs()
