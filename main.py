import datetime
import urllib.request
from enum import Enum
from io import BytesIO

import torch
from diffusers import DiffusionPipeline
from PIL import Image


class ModelType(Enum):
    TEXT_TO_IMAGE = "text-to-image"
    IMAGE_TO_IMAGE = "image-to-image"


models = (
    {"name": "stabilityai/stable-diffusion-3.5-medium", "type": ModelType.TEXT_TO_IMAGE, "dtype": torch.float16},
    {"name": "xyn-ai/dreamlike-photoreal-2.0", "type": ModelType.TEXT_TO_IMAGE, "dtype": torch.float16},
    {"name": "black-forest-labs/FLUX.2-klein-9b-kv", "type": ModelType.IMAGE_TO_IMAGE, "dtype": torch.float16},
    {"name": "Qwen/Qwen-Image-Edit", "type": ModelType.IMAGE_TO_IMAGE, "dtype": torch.bfloat16},
)


def load_image(source: str) -> Image.Image:
    if source.startswith("http://") or source.startswith("https://"):
        with urllib.request.urlopen(source) as response:
            return Image.open(BytesIO(response.read())).convert("RGB")
    return Image.open(source).convert("RGB")


def main():
    print("Select a model:")
    for i, model in enumerate(models):
        print(f"{i}: {model['name']} ({model['type'].value})")

    selected_model = models[int(input())]

    # switch to "mps" for apple devices
    pipe = DiffusionPipeline.from_pretrained(selected_model["name"], torch_dtype=selected_model["dtype"])
    pipe = pipe.to("mps")

    while True:
        # prompt = "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k"
        try:
            if selected_model["type"] == ModelType.IMAGE_TO_IMAGE:
                image_source = input("Input image (file path or URL): ")
                input_image = load_image(image_source)
            prompt = input("Image description: ")
        except KeyboardInterrupt:
            return

        if selected_model["type"] == ModelType.IMAGE_TO_IMAGE:
            image = pipe(prompt=prompt, image=input_image).images[0]
        else:
            image = pipe(prompt).images[0]

        now = datetime.datetime.now(datetime.timezone.utc)
        filename = f"/Users/don/Pictures/diffusion-renders/{now.isoformat()}.png"
        image.save(filename)
        print(f"Image saved to {filename}\n")


if __name__ == "__main__":
    main()
