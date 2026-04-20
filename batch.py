"""Batch mode: render a list of prompts described by a JSON file."""

import json
import os
import sys

import torch
from PIL import Image

from utils import (
    FACEID_MAX_SIDE,
    ModelType,
    blank_faceid_embeds,
    downscale_for_faceid,
    faceid_embeds_from_image,
    get_device,
    load_image,
    load_ip_adapter_into,
    load_models,
    load_pipeline,
    make_face_app,
)


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
        target_w = params.get("width")
        target_h = params.get("height")
        if is_faceid:
            if target_w is None or target_h is None:
                pipe_kwargs["width"] = FACEID_MAX_SIDE
                pipe_kwargs["height"] = FACEID_MAX_SIDE
            else:
                gen_w, gen_h = downscale_for_faceid(target_w, target_h)
                pipe_kwargs["width"] = gen_w
                pipe_kwargs["height"] = gen_h
        else:
            if target_w is not None:
                pipe_kwargs["width"] = target_w
            if target_h is not None:
                pipe_kwargs["height"] = target_h
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

            if is_faceid and target_w and target_h and image.size != (target_w, target_h):
                image = image.resize((target_w, target_h), Image.LANCZOS)
            image.save(output_path)
            print(f"  ✓ Saved to {output_path}")
        except Exception as e:
            print(f"  ✗ Error generating image: {e}", file=sys.stderr)

    print(f"\nBatch complete. {total} images processed.")
