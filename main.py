import argparse
import datetime
import json
import os
import sys
import urllib.request
from io import BytesIO

import cv2
import numpy as np
import torch
from diffusers import DiffusionPipeline
from enum import Enum
from PIL import Image

FACEID_EMBED_DIM = 512


class ModelType(Enum):
    TEXT_TO_IMAGE = "text-to-image"
    IMAGE_TO_IMAGE = "image-to-image"


DTYPE_MAP = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}


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
    return Image.open(source).convert("RGB")


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
    pipe = pipe.to(device)
    return pipe, selected_model


def run_batch(batch_file: str):
    """
    Batch mode: read a JSON file and generate all images described in it.

    Expected batch JSON format:
    {
        "model_name": "xyn-ai/dreamlike-photoreal-2.0",
        "negative_prompt": "lowres, bad anatomy, ...",
        "steps": 28,
        "guidance": 7.0,
        "width": 640,
        "height": 896,
        "images": [
            {
                "persona": "akemi",
                "output": "/abs/path/to/akemi_base.png",
                "prompt": "anime-style waist-up portrait ...",
                "seed": 42
            },
            {
                "persona": "akemi",
                "output": "/abs/path/to/s01_c1.png",
                "prompt": "akemi smiling under cherry trees ...",
                "seed": 1043,
                "ip_adapter_image": "/abs/path/to/akemi_base.png",
                "ip_adapter_scale": 0.75
            }
        ]
    }

    Top-level negative_prompt/steps/guidance/seed/width/height act as
    defaults; any image may override them. Each image requires an `output`
    path (absolute or ~-expandable); existing non-empty files are skipped.
    Image-to-image models still read `input_image`. If the selected model
    has an ip_adapter config in models.json, IP-Adapter is loaded once;
    per-image `ip_adapter_image` and `ip_adapter_scale` are honored, and
    images that omit them fall back to a 224x224 black reference at
    scale 0.0 so the pipeline signature stays consistent.
    """
    batch_path = os.path.expanduser(batch_file)
    if not os.path.exists(batch_path):
        print(f"Batch file not found: {batch_path}")
        raise SystemExit(1)

    with open(batch_path) as f:
        batch = json.load(f)

    model_name = batch["model_name"]
    images = batch["images"]

    default_keys = ("negative_prompt", "steps", "guidance", "seed", "width", "height")
    defaults = {k: batch[k] for k in default_keys if k in batch}

    models = load_models()
    pipe, selected_model = load_pipeline(model_name, models)
    device = get_device()

    ip_adapter_config = selected_model.get("ip_adapter")
    is_faceid = bool(ip_adapter_config and ip_adapter_config.get("variant") == "faceid")
    face_app = None
    if ip_adapter_config:
        load_ip_adapter_into(pipe, ip_adapter_config)
        if is_faceid:
            face_app = make_face_app(device)

    total = len(images)
    print(f"\nBatch mode: generating {total} images\n")

    for i, item in enumerate(images, 1):
        output_path = os.path.expanduser(item["output"])
        persona = item.get("persona", "unknown")
        prompt = item["prompt"]

        if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
            print(f"[{i}/{total}] Skipping {output_path} (already exists)")
            continue

        parent = os.path.dirname(output_path)
        if parent:
            os.makedirs(parent, exist_ok=True)

        params = {**defaults, **{k: item[k] for k in default_keys if k in item}}

        pipe_kwargs = {"prompt": prompt}
        if "negative_prompt" in params:
            pipe_kwargs["negative_prompt"] = params["negative_prompt"]
        if "steps" in params:
            pipe_kwargs["num_inference_steps"] = params["steps"]
        if "guidance" in params:
            pipe_kwargs["guidance_scale"] = params["guidance"]
        if "width" in params:
            pipe_kwargs["width"] = params["width"]
        if "height" in params:
            pipe_kwargs["height"] = params["height"]
        if "seed" in params:
            pipe_kwargs["generator"] = torch.Generator(device).manual_seed(params["seed"])

        print(f"[{i}/{total}] {persona}: {prompt[:80]}...")
        try:
            if selected_model["type"] == ModelType.IMAGE_TO_IMAGE:
                if "input_image" not in item:
                    print(f"  Warning: image-to-image model requires 'input_image' field, skipping.")
                    continue
                pipe_kwargs["image"] = load_image(item["input_image"])
                image = pipe(**pipe_kwargs).images[0]
            elif ip_adapter_config:
                if "ip_adapter_image" in item:
                    scale = item.get("ip_adapter_scale", ip_adapter_config.get("scale", 0.5))
                    pipe.set_ip_adapter_scale(scale)
                    ref = load_image(item["ip_adapter_image"])
                    if is_faceid:
                        pipe_kwargs["ip_adapter_image_embeds"] = [
                            faceid_embeds_from_image(ref, face_app, selected_model["dtype"], device)
                        ]
                    else:
                        pipe_kwargs["ip_adapter_image"] = ref
                else:
                    pipe.set_ip_adapter_scale(0.0)
                    if is_faceid:
                        pipe_kwargs["ip_adapter_image_embeds"] = [
                            blank_faceid_embeds(selected_model["dtype"], device)
                        ]
                    else:
                        pipe_kwargs["ip_adapter_image"] = Image.new("RGB", (224, 224), (0, 0, 0))
                image = pipe(**pipe_kwargs).images[0]
            else:
                image = pipe(**pipe_kwargs).images[0]

            image.save(output_path)
            print(f"  ✓ Saved to {output_path}")
        except Exception as e:
            print(f"  ✗ Error generating image: {e}", file=sys.stderr)

    print(f"\nBatch complete. {total} images processed.")


def run_interactive():
    """Original interactive mode."""
    models = load_models()

    print("Select a model:")
    for i, model in enumerate(models):
        print(f"{i}: {model['name']} ({model['type'].value})")

    selected_model = models[int(input())]

    device = get_device()
    pipe = DiffusionPipeline.from_pretrained(selected_model["name"], torch_dtype=selected_model["dtype"])
    pipe = pipe.to(device)

    ip_adapter_config = selected_model.get("ip_adapter")
    is_faceid = bool(ip_adapter_config and ip_adapter_config.get("variant") == "faceid")
    face_app = None
    if ip_adapter_config:
        load_ip_adapter_into(pipe, ip_adapter_config)
        pipe.set_ip_adapter_scale(ip_adapter_config.get("scale", 0.5))
        if is_faceid:
            face_app = make_face_app(device)

    while True:
        try:
            input_image = None
            if selected_model["type"] == ModelType.IMAGE_TO_IMAGE:
                image_source = input("Input image (file path or URL): ").strip()
                input_image = load_image(image_source)
            elif ip_adapter_config:
                image_source = input("Reference image for IP-Adapter (file path/URL, or Enter to skip): ").strip()
                if image_source:
                    input_image = load_image(image_source)
            prompt = input("Image description: ").strip()
        except KeyboardInterrupt:
            return

        if selected_model["type"] == ModelType.IMAGE_TO_IMAGE:
            image = pipe(prompt=prompt, image=input_image).images[0]
        elif input_image is not None:
            pipe.set_ip_adapter_scale(ip_adapter_config.get("scale", 0.5))
            if is_faceid:
                embeds = faceid_embeds_from_image(input_image, face_app, selected_model["dtype"], device)
                image = pipe(prompt=prompt, ip_adapter_image_embeds=[embeds]).images[0]
            else:
                image = pipe(prompt=prompt, ip_adapter_image=input_image).images[0]
        elif ip_adapter_config:
            pipe.set_ip_adapter_scale(0.0)
            if is_faceid:
                embeds = blank_faceid_embeds(selected_model["dtype"], device)
                image = pipe(prompt=prompt, ip_adapter_image_embeds=[embeds]).images[0]
            else:
                blank = Image.new("RGB", (224, 224), (0, 0, 0))
                image = pipe(prompt=prompt, ip_adapter_image=blank).images[0]
        else:
            image = pipe(prompt).images[0]

        now = datetime.datetime.now(datetime.timezone.utc)
        filename = f"/Users/don/Pictures/diffusion-renders/{now.isoformat()}.png"
        image.save(filename)
        print(f"Image saved to {filename}\n")


def main():
    parser = argparse.ArgumentParser(description="Diffuser CLI — generate images from text using diffusion models")
    parser.add_argument(
        "--batch",
        metavar="BATCH_JSON",
        help="Run in batch mode using a JSON file of prompts (skips interactive mode)",
    )
    args = parser.parse_args()

    if args.batch:
        run_batch(args.batch)
    else:
        run_interactive()


if __name__ == "__main__":
    main()
