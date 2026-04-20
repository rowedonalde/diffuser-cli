"""Tests for batch-mode dispatch. The real DiffusionPipeline is mocked so
we can assert argument wiring without loading any model weights.
"""

import json
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

import batch as target
import utils


def _fake_pipe_output(size=(64, 64), color=(10, 20, 30)):
    img = Image.new("RGB", size, color)
    result = MagicMock()
    result.images = [img]
    return result


def _setup_pipe(mock_pipe_factory=None):
    pipe = MagicMock()
    pipe.side_effect = lambda **kwargs: _fake_pipe_output()
    return pipe


def _write_batch(tmp_path, batch_doc):
    path = tmp_path / "batch.json"
    path.write_text(json.dumps(batch_doc))
    return str(path)


@pytest.fixture
def patched_helpers(monkeypatch):
    """Stub out all heavyweight helpers so run_batch can execute quickly."""
    pipe = MagicMock()
    pipe.side_effect = lambda **kwargs: _fake_pipe_output()

    selected_model = {
        "name": "fake/model",
        "type": target.ModelType.TEXT_TO_IMAGE,
        "dtype": utils.DTYPE_MAP["float16"],
    }

    monkeypatch.setattr(target, "load_models", lambda: [selected_model])
    monkeypatch.setattr(
        target, "load_pipeline", lambda name, models: (pipe, selected_model)
    )
    monkeypatch.setattr(target, "get_device", lambda: "cpu")
    monkeypatch.setattr(target, "make_face_app", lambda device: MagicMock())
    monkeypatch.setattr(target, "load_ip_adapter_into", lambda pipe, cfg: None)

    return {"pipe": pipe, "model": selected_model}


def test_batch_exits_on_missing_file(tmp_path):
    missing = str(tmp_path / "does-not-exist.json")
    with pytest.raises(SystemExit):
        target.run_batch(missing)


def test_batch_skips_existing_nonempty_outputs(tmp_path, patched_helpers, capsys):
    existing = tmp_path / "already.png"
    existing.write_bytes(b"not-empty")

    batch_path = _write_batch(
        tmp_path,
        {
            "model_name": "fake/model",
            "images": [
                {"persona": "x", "output": str(existing), "prompt": "unused"},
            ],
        },
    )
    target.run_batch(batch_path)

    patched_helpers["pipe"].assert_not_called()
    out = capsys.readouterr().out
    assert "Skipping" in out


def test_batch_applies_defaults_and_overrides(tmp_path, patched_helpers):
    out_path = tmp_path / "out.png"
    batch_path = _write_batch(
        tmp_path,
        {
            "model_name": "fake/model",
            "negative_prompt": "blurry",
            "steps": 20,
            "guidance": 5.0,
            "width": 512,
            "height": 768,
            "images": [
                {
                    "persona": "a",
                    "output": str(out_path),
                    "prompt": "hello world",
                    "steps": 40,  # override
                    "seed": 7,
                }
            ],
        },
    )
    target.run_batch(batch_path)

    patched_helpers["pipe"].assert_called_once()
    _, kwargs = patched_helpers["pipe"].call_args
    assert kwargs["prompt"] == "hello world"
    assert kwargs["negative_prompt"] == "blurry"
    assert kwargs["num_inference_steps"] == 40  # overridden
    assert kwargs["guidance_scale"] == 5.0  # from defaults
    assert kwargs["width"] == 512
    assert kwargs["height"] == 768
    assert "generator" in kwargs
    assert out_path.exists() and out_path.stat().st_size > 0


def test_batch_image_to_image_requires_input_image(tmp_path, monkeypatch, capsys):
    pipe = MagicMock()
    pipe.side_effect = lambda **kwargs: _fake_pipe_output()
    i2i_model = {
        "name": "fake/i2i",
        "type": target.ModelType.IMAGE_TO_IMAGE,
        "dtype": utils.DTYPE_MAP["float16"],
    }
    monkeypatch.setattr(target, "load_models", lambda: [i2i_model])
    monkeypatch.setattr(target, "load_pipeline", lambda n, m: (pipe, i2i_model))
    monkeypatch.setattr(target, "get_device", lambda: "cpu")
    monkeypatch.setattr(target, "load_ip_adapter_into", lambda p, c: None)

    out_path = tmp_path / "img.png"
    batch_path = _write_batch(
        tmp_path,
        {
            "model_name": "fake/i2i",
            "images": [
                {"persona": "a", "output": str(out_path), "prompt": "p"},
            ],
        },
    )

    target.run_batch(batch_path)

    pipe.assert_not_called()
    assert not out_path.exists()
    out = capsys.readouterr().out
    assert "image-to-image" in out


def test_batch_ip_adapter_blank_reference_when_missing(tmp_path, monkeypatch):
    pipe = MagicMock()
    pipe.side_effect = lambda **kwargs: _fake_pipe_output()
    ip_model = {
        "name": "fake/ip",
        "type": target.ModelType.TEXT_TO_IMAGE,
        "dtype": utils.DTYPE_MAP["float16"],
        "ip_adapter": {
            "repo": "h94/IP-Adapter",
            "weight_name": "foo.bin",
            "scale": 0.6,
        },
    }
    monkeypatch.setattr(target, "load_models", lambda: [ip_model])
    monkeypatch.setattr(target, "load_pipeline", lambda n, m: (pipe, ip_model))
    monkeypatch.setattr(target, "get_device", lambda: "cpu")
    monkeypatch.setattr(target, "load_ip_adapter_into", lambda p, c: None)

    out_path = tmp_path / "blank_ref.png"
    batch_path = _write_batch(
        tmp_path,
        {
            "model_name": "fake/ip",
            "images": [
                {"persona": "a", "output": str(out_path), "prompt": "hello"},
            ],
        },
    )
    target.run_batch(batch_path)

    pipe.set_ip_adapter_scale.assert_called_with(0.0)
    _, kwargs = pipe.call_args
    assert "ip_adapter_image" in kwargs
    assert kwargs["ip_adapter_image"].size == (224, 224)
    assert out_path.exists()


def test_batch_ip_adapter_uses_reference_image(tmp_path, monkeypatch):
    pipe = MagicMock()
    pipe.side_effect = lambda **kwargs: _fake_pipe_output()
    ip_model = {
        "name": "fake/ip",
        "type": target.ModelType.TEXT_TO_IMAGE,
        "dtype": utils.DTYPE_MAP["float16"],
        "ip_adapter": {
            "repo": "h94/IP-Adapter",
            "weight_name": "foo.bin",
            "scale": 0.5,
        },
    }
    ref_path = tmp_path / "ref.png"
    Image.new("RGB", (32, 32), (200, 100, 50)).save(ref_path)

    monkeypatch.setattr(target, "load_models", lambda: [ip_model])
    monkeypatch.setattr(target, "load_pipeline", lambda n, m: (pipe, ip_model))
    monkeypatch.setattr(target, "get_device", lambda: "cpu")
    monkeypatch.setattr(target, "load_ip_adapter_into", lambda p, c: None)

    out_path = tmp_path / "with_ref.png"
    batch_path = _write_batch(
        tmp_path,
        {
            "model_name": "fake/ip",
            "images": [
                {
                    "persona": "a",
                    "output": str(out_path),
                    "prompt": "hello",
                    "ip_adapter_image": str(ref_path),
                    "ip_adapter_scale": 0.9,
                },
            ],
        },
    )
    target.run_batch(batch_path)

    pipe.set_ip_adapter_scale.assert_called_with(0.9)
    _, kwargs = pipe.call_args
    assert kwargs["ip_adapter_image"].size == (32, 32)


def test_batch_faceid_downscales_and_resizes_back(tmp_path, monkeypatch):
    pipe = MagicMock()
    # Return a FaceID-sized image so we can observe the upscale back to target.
    pipe.side_effect = lambda **kwargs: _fake_pipe_output(
        size=(target.FACEID_MAX_SIDE, target.FACEID_MAX_SIDE)
    )
    faceid_model = {
        "name": "fake/faceid",
        "type": target.ModelType.TEXT_TO_IMAGE,
        "dtype": utils.DTYPE_MAP["float16"],
        "ip_adapter": {
            "repo": "h94/IP-Adapter-FaceID",
            "weight_name": "ip-adapter-faceid_sd15.bin",
            "scale": 0.5,
            "variant": "faceid",
        },
    }
    monkeypatch.setattr(target, "load_models", lambda: [faceid_model])
    monkeypatch.setattr(target, "load_pipeline", lambda n, m: (pipe, faceid_model))
    monkeypatch.setattr(target, "get_device", lambda: "cpu")
    monkeypatch.setattr(target, "make_face_app", lambda d: MagicMock())
    monkeypatch.setattr(target, "load_ip_adapter_into", lambda p, c: None)

    out_path = tmp_path / "face.png"
    batch_path = _write_batch(
        tmp_path,
        {
            "model_name": "fake/faceid",
            "width": 1024,
            "height": 1024,
            "images": [
                {"persona": "a", "output": str(out_path), "prompt": "a person"},
            ],
        },
    )
    target.run_batch(batch_path)

    _, kwargs = pipe.call_args
    assert kwargs["width"] == target.FACEID_MAX_SIDE
    assert kwargs["height"] == target.FACEID_MAX_SIDE

    saved = Image.open(out_path)
    assert saved.size == (1024, 1024)
