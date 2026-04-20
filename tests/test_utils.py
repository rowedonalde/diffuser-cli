"""Tests for the pure/helper functions in utils."""

import json
from unittest.mock import MagicMock

import pytest
import torch
from PIL import Image

import utils as target


def test_dtype_map_covers_expected_names():
    assert target.DTYPE_MAP["float16"] is torch.float16
    assert target.DTYPE_MAP["bfloat16"] is torch.bfloat16
    assert target.DTYPE_MAP["float32"] is torch.float32


def test_scheduler_map_contains_expected_keys():
    assert set(target.SCHEDULER_MAP) == {"ddim", "euler"}


def test_model_type_values():
    assert target.ModelType("text-to-image") is target.ModelType.TEXT_TO_IMAGE
    assert target.ModelType("image-to-image") is target.ModelType.IMAGE_TO_IMAGE


class TestDownscaleForFaceid:
    def test_no_scaling_when_within_limit(self):
        assert target.downscale_for_faceid(512, 512) == (512, 512)
        assert target.downscale_for_faceid(256, 384) == (256, 384)

    def test_downscales_large_square(self):
        w, h = target.downscale_for_faceid(1024, 1024)
        assert (w, h) == (target.FACEID_MAX_SIDE, target.FACEID_MAX_SIDE)

    def test_preserves_aspect_ratio_roughly(self):
        w, h = target.downscale_for_faceid(1024, 1536)
        assert max(w, h) <= target.FACEID_MAX_SIDE
        # Taller than wide; ratio preserved within 8-px rounding tolerance.
        assert h > w
        ratio = h / w
        assert abs(ratio - 1.5) < 0.1

    def test_output_is_multiple_of_8_and_min_64(self):
        w, h = target.downscale_for_faceid(10000, 10000)
        assert w % 8 == 0 and h % 8 == 0
        assert w >= 64 and h >= 64


def test_blank_faceid_embeds_shape_and_dtype():
    embeds = target.blank_faceid_embeds(torch.float32, "cpu")
    assert embeds.shape == (2, 1, 1, target.FACEID_EMBED_DIM)
    assert embeds.dtype == torch.float32
    assert torch.all(embeds == 0)


def test_load_models_parses_and_converts_types(tmp_path, monkeypatch):
    fake_models = [
        {"name": "some/model", "type": "text-to-image", "dtype": "float16"},
        {
            "name": "other/model",
            "type": "image-to-image",
            "dtype": "bfloat16",
            "scheduler": "ddim",
        },
    ]
    fake_main = tmp_path / "main.py"
    fake_main.write_text("")
    (tmp_path / "models.json").write_text(json.dumps(fake_models))

    monkeypatch.setattr(target, "__file__", str(fake_main))
    result = target.load_models()

    assert len(result) == 2
    assert result[0]["dtype"] is torch.float16
    assert result[0]["type"] is target.ModelType.TEXT_TO_IMAGE
    assert result[1]["dtype"] is torch.bfloat16
    assert result[1]["type"] is target.ModelType.IMAGE_TO_IMAGE
    assert result[1]["scheduler"] == "ddim"


def test_load_models_errors_when_missing(tmp_path, monkeypatch):
    fake_main = tmp_path / "main.py"
    fake_main.write_text("")
    monkeypatch.setattr(target, "__file__", str(fake_main))
    with pytest.raises(SystemExit):
        target.load_models()


def test_load_image_from_local_file(tmp_path):
    path = tmp_path / "green.png"
    Image.new("RGB", (8, 8), (0, 255, 0)).save(path)
    img = target.load_image(str(path))
    assert img.size == (8, 8)
    assert img.mode == "RGB"


def test_load_image_converts_non_rgb_to_rgb(tmp_path):
    path = tmp_path / "gray.png"
    Image.new("L", (4, 4), 128).save(path)
    img = target.load_image(str(path))
    assert img.mode == "RGB"


def test_load_image_expands_user_home(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    path = tmp_path / "tilde.png"
    Image.new("RGB", (4, 4), (1, 2, 3)).save(path)
    img = target.load_image("~/tilde.png")
    assert img.size == (4, 4)


def test_load_ip_adapter_into_passes_standard_kwargs():
    pipe = MagicMock()
    cfg = {
        "repo": "h94/IP-Adapter",
        "subfolder": "models",
        "weight_name": "ip-adapter.bin",
    }
    target.load_ip_adapter_into(pipe, cfg)
    pipe.load_ip_adapter.assert_called_once_with(
        "h94/IP-Adapter", subfolder="models", weight_name="ip-adapter.bin"
    )


def test_load_ip_adapter_into_faceid_passes_none_encoder():
    pipe = MagicMock()
    cfg = {
        "repo": "h94/IP-Adapter-FaceID",
        "weight_name": "ip-adapter-faceid_sd15.bin",
        "variant": "faceid",
    }
    target.load_ip_adapter_into(pipe, cfg)
    _, kwargs = pipe.load_ip_adapter.call_args
    assert kwargs["image_encoder_folder"] is None
    assert kwargs["weight_name"] == "ip-adapter-faceid_sd15.bin"
    assert kwargs["subfolder"] is None


def test_get_device_returns_a_known_value():
    assert target.get_device() in {"cpu", "cuda", "mps"}
