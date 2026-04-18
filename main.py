import argparse
import datetime
import json
import os
import sys
import urllib.request
from io import BytesIO

import torch
from diffusers import DiffusionPipeline
from enum import Enum
from PIL import Image


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
        "output_dir": "~/Pictures/dating-app-personas",
        "images": [
            {
                "persona": "alex_chen",
                "filename": "alex_chen_01.png",
                "prompt": "photo of a 28-year-old man on a mountain summit..."
            },
            ...
        ]
    }
    """
    batch_path = os.path.expanduser(batch_file)
    if not os.path.exists(batch_path):
        print(f"Batch file not found: {batch_path}")
        raise SystemExit(1)

    with open(batch_path) as f:
        batch = json.load(f)

    model_name = batch["model_name"]
    output_dir = os.path.expanduser(batch["output_dir"])
    images = batch["images"]

    os.makedirs(output_dir, exist_ok=True)

    models = load_models()
    pipe, selected_model = load_pipeline(model_name, models)

    total = len(images)
    print(f"\nBatch mode: generating {total} images → {output_dir}\n")

    for i, item in enumerate(images, 1):
        filename = os.path.join(output_dir, item["filename"])
        persona = item.get("persona", "unknown")
        prompt = item["prompt"]

        if os.path.exists(filename):
            print(f"[{i}/{total}] Skipping {item['filename']} (already exists)")
            continue

        print(f"[{i}/{total}] {persona}: {prompt[:80]}...")
        try:
            if selected_model["type"] == ModelType.IMAGE_TO_IMAGE:
                if "input_image" not in item:
                    print(f"  Warning: image-to-image model requires 'input_image' field, skipping.")
                    continue
                input_image = load_image(item["input_image"])
                image = pipe(prompt=prompt, image=input_image).images[0]
            else:
                image = pipe(prompt).images[0]

            image.save(filename)
            print(f"  ✓ Saved to {filename}")
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
    if ip_adapter_config:
        print("Loading IP-Adapter...")
        pipe.load_ip_adapter(
            ip_adapter_config["repo"],
            subfolder=ip_adapter_config.get("subfolder"),
            weight_name=ip_adapter_config["weight_name"],
        )
        pipe.set_ip_adapter_scale(ip_adapter_config.get("scale", 0.5))

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
            image = pipe(prompt=prompt, ip_adapter_image=input_image).images[0]
        elif ip_adapter_config:
            pipe.set_ip_adapter_scale(0.0)
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
