"""Shared helpers for the diffuser CLI.

Houses model/device selection, image loading, and IP-Adapter / FaceID
plumbing. Both the interactive and batch entry points build on these.
"""

import json
import os
import urllib.request
from enum import Enum
from io import BytesIO

import cv2
import numpy as np
import torch
from diffusers import DDIMScheduler, DiffusionPipeline, EulerDiscreteScheduler
from PIL import Image

# SD1.5 FaceID LoRA was trained at 512; higher res corrupts output.
FACEID_EMBED_DIM = 512
FACEID_MAX_SIDE = 512

SCHEDULER_MAP = {
    "ddim": DDIMScheduler,
    "euler": EulerDiscreteScheduler,
}

DTYPE_MAP = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}


class ModelType(Enum):
    TEXT_TO_IMAGE = "text-to-image"
    IMAGE_TO_IMAGE = "image-to-image"


def load_models():
    models_path = os.path.join(os.path.dirname(__file__), "models.json")
    if not os.path.exists(models_path):
        print("models.json not found. Copy models.json.example to models.json to get started.")
        raise SystemExit(1)
    with open(models_path) as f:
        raw = json.load(f)
    for model in raw:
        model["dtype"] = DTYPE_MAP[model["dtype"]]
        model["type"] = ModelType(model["type"])
    return raw


def load_image(source: str) -> Image.Image:
    if source.startswith("http://") or source.startswith("https://"):
        with urllib.request.urlopen(source) as response:
            return Image.open(BytesIO(response.read())).convert("RGB")
    return Image.open(os.path.expanduser(source)).convert("RGB")


def get_device():
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def make_face_app(device: str):
    from insightface.app import FaceAnalysis

    if device == "cuda":
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        ctx_id = 0
    else:
        providers = ["CPUExecutionProvider"]
        ctx_id = -1
    app = FaceAnalysis(name="buffalo_l", providers=providers)
    app.prepare(ctx_id=ctx_id, det_size=(640, 640))
    return app


def faceid_embeds_from_image(image: Image.Image, app, dtype, device):
    # InsightFace expects BGR; PIL gives RGB, so swap channels.
    arr = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
    faces = app.get(arr)
    if not faces:
        raise ValueError("No face detected in IP-Adapter reference image")
    embed = torch.from_numpy(faces[0].normed_embedding).unsqueeze(0)
    ref = torch.stack([embed], dim=0).unsqueeze(0)
    neg = torch.zeros_like(ref)
    return torch.cat([neg, ref]).to(dtype=dtype, device=device)


def blank_faceid_embeds(dtype, device):
    return torch.zeros((2, 1, 1, FACEID_EMBED_DIM), dtype=dtype, device=device)


def downscale_for_faceid(width: int, height: int) -> tuple[int, int]:
    max_side = max(width, height)
    if max_side <= FACEID_MAX_SIDE:
        return width, height
    scale = FACEID_MAX_SIDE / max_side
    def round8(x: float) -> int:
        return max(64, int(round(x * scale / 8)) * 8)
    return round8(width), round8(height)


def load_ip_adapter_into(pipe, ip_adapter_config: dict):
    print("Loading IP-Adapter...")
    load_kwargs = {
        "subfolder": ip_adapter_config.get("subfolder"),
        "weight_name": ip_adapter_config["weight_name"],
    }
    if ip_adapter_config.get("variant") == "faceid":
        load_kwargs["image_encoder_folder"] = None
    pipe.load_ip_adapter(ip_adapter_config["repo"], **load_kwargs)


def load_pipeline(model_name: str, models: list) -> tuple:
    selected_model = next((m for m in models if m["name"] == model_name), None)
    if selected_model is None:
        print(f"Error: model '{model_name}' not found in models.json")
        print("Available models:", [m["name"] for m in models])
        raise SystemExit(1)
    device = get_device()
    print(f"Loading model '{model_name}' on device '{device}'...")
    pipe = DiffusionPipeline.from_pretrained(selected_model["name"], torch_dtype=selected_model["dtype"])
    scheduler_name = selected_model.get("scheduler")
    if scheduler_name:
        scheduler_cls = SCHEDULER_MAP[scheduler_name]
        pipe.scheduler = scheduler_cls.from_config(pipe.scheduler.config)
    pipe = pipe.to(device)
    return pipe, selected_model
