# diffuser-cli

Small wrapper around 🤗 `diffusers` for generating images from text, with
optional IP-Adapter character conditioning. Models are declared in
`models.json` (copy `models.json.example` to get started).

## Interactive mode

```
uv run main.py
```

Pick a model from the list, then answer the prompts. Image-to-image models
ask for an input image; text-to-image models with an IP-Adapter configured
in `models.json` optionally ask for a reference image for character
conditioning.

## Batch mode

```
uv run main.py --batch path/to/batch.json
```

Designed to be called once per run and generate many images without paying
pipeline load cost per image. Skips any image whose `output` path already
exists and is non-empty.

### Batch JSON schema

```json
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
```

Top-level keys:

| Key | Required | Notes |
|---|---|---|
| `model_name` | yes | Must match a `name` in `models.json`. |
| `images` | yes | Array of image objects (below). |
| `negative_prompt` | no | Default applied to every image. |
| `steps` | no | Default `num_inference_steps`. |
| `guidance` | no | Default CFG `guidance_scale`. |
| `seed` | no | Default seed (usually you set per image). |
| `width`, `height` | no | Default output dimensions. |

Per-image keys:

| Key | Required | Notes |
|---|---|---|
| `output` | yes | Absolute path (or `~`-expandable). Parent dirs created on demand. |
| `prompt` | yes | |
| `persona` | no | Shown in log output only. |
| `negative_prompt`, `steps`, `guidance`, `seed`, `width`, `height` | no | Override top-level defaults. |
| `input_image` | image-to-image only | File path or URL. |
| `ip_adapter_image` | no | File path or URL; used when the model has an `ip_adapter` entry in `models.json`. Falls back to the model's configured scale if `ip_adapter_scale` is omitted. |
| `ip_adapter_scale` | no | Per-image override of IP-Adapter strength. |

When a model has IP-Adapter configured in `models.json` but an image
omits `ip_adapter_image`, the pipeline is called with a 224×224 black
reference at scale `0.0` — i.e., IP-Adapter is effectively disabled for
that image while keeping the pipeline signature consistent.
