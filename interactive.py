"""Interactive mode: prompt the user at the terminal, one image at a time."""

import datetime

from PIL import Image

from utils import (
    FACEID_MAX_SIDE,
    ModelType,
    blank_faceid_embeds,
    faceid_embeds_from_image,
    get_device,
    load_image,
    load_ip_adapter_into,
    load_models,
    load_pipeline,
    make_face_app,
)


def run_interactive():
    """Original interactive mode."""
    models = load_models()

    print("Select a model:")
    for i, model in enumerate(models):
        print(f"{i}: {model['name']} ({model['type'].value})")

    selected_model = models[int(input())]
    pipe, selected_model = load_pipeline(selected_model["name"], models)
    device = get_device()

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

        faceid_size_kwargs = {"width": FACEID_MAX_SIDE, "height": FACEID_MAX_SIDE} if is_faceid else {}
        if selected_model["type"] == ModelType.IMAGE_TO_IMAGE:
            image = pipe(prompt=prompt, image=input_image).images[0]
        elif input_image is not None:
            pipe.set_ip_adapter_scale(ip_adapter_config.get("scale", 0.5))
            if is_faceid:
                embeds = faceid_embeds_from_image(input_image, face_app, selected_model["dtype"], device)
                image = pipe(prompt=prompt, ip_adapter_image_embeds=[embeds], **faceid_size_kwargs).images[0]
            else:
                image = pipe(prompt=prompt, ip_adapter_image=input_image).images[0]
        elif ip_adapter_config:
            pipe.set_ip_adapter_scale(0.0)
            if is_faceid:
                embeds = blank_faceid_embeds(selected_model["dtype"], device)
                image = pipe(prompt=prompt, ip_adapter_image_embeds=[embeds], **faceid_size_kwargs).images[0]
            else:
                blank = Image.new("RGB", (224, 224), (0, 0, 0))
                image = pipe(prompt=prompt, ip_adapter_image=blank).images[0]
        else:
            image = pipe(prompt).images[0]

        now = datetime.datetime.now(datetime.timezone.utc)
        filename = f"/Users/don/Pictures/diffusion-renders/{now.isoformat()}.png"
        image.save(filename)
        print(f"Image saved to {filename}\n")
