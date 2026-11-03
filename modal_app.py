import os
import modal

# Pre-cache base model weights in image layer to eliminate download time during cold-starts
def download_base_model():
    from transformers import AutoTokenizer, AutoModelForCausalLM
    model_id = "Qwen/Qwen2.5-0.5B-Instruct"
    AutoTokenizer.from_pretrained(model_id)
    AutoModelForCausalLM.from_pretrained(model_id)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch==2.6.0",
        "transformers==5.22.1",
        "peft==0.21.2",
        "accelerate==1.15.0",
        "bashlex==0.18",
        "fastapi==0.162.0",
        "uvicorn==0.54.0",
        "pydantic==2.14.2",
        "numpy==2.5.4",
        "requests==2.34.2",
    )
    .run_function(download_base_model)
    .add_local_dir("serve", remote_path="/home/shivsharan/nl2shell-project/serve")
    .add_local_dir("adapters/lora_0.5b", remote_path="/home/shivsharan/nl2shell-project/adapters/lora_0.5b")
)

app = modal.App(name="nl2shell", image=image)

# Compute target: Set MODAL_GPU="T4" if payment method is linked on modal.com
# Otherwise runs on 4-core 8GB CPU (Starter tier compatible without card)
GPU_CONFIG = os.environ.get("MODAL_GPU", None)

@app.function(
    gpu=GPU_CONFIG,
    cpu=4.0 if not GPU_CONFIG else None,
    memory=8192 if not GPU_CONFIG else None,
    scaledown_window=60,
    timeout=600,
)
@modal.asgi_app()
def fastapi_app():
    import sys
    sys.path.insert(0, "/home/shivsharan/nl2shell-project")
    
    from serve.app import app as web_app, load_model, model
    
    if model is None:
        load_model()
        
    return web_app
