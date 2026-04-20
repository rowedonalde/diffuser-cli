# CLAUDE.md

Guidance for Claude Code sessions on this repo.

## What this is

A small CLI around 🤗 `diffusers`. Two entry modes: interactive REPL and
JSON-driven batch. Supports IP-Adapter (including FaceID) for character
conditioning.

## Module layout

Keep each file focused. Don't re-inline helpers back into `main.py`.

- `main.py` — argparse + dispatch only. Should stay tiny.
- `interactive.py` — `run_interactive()`.
- `batch.py` — `run_batch()`.
- `utils.py` — shared helpers, constants, `ModelType`. If a helper is used
  by both modes (or is a pure function worth testing on its own), it
  belongs here.
- `tests/` — pytest. `DiffusionPipeline` and heavy helpers are mocked.

## Running

```
uv run main.py                          # interactive
uv run main.py --batch <file.json>      # batch
uv run pytest tests/                    # tests
```

`uv add <pkg>` for deps; `uv add --dev <pkg>` for dev deps. `uv.lock` is
tracked.

## Config files

- `models.json` — gitignored. User copies from `models.json.example` (no
  FaceID) or `models.json.faceid` (FaceID variant). Keep both examples
  in sync when adding new model fields.
- Interactive mode saves to a hardcoded path
  (`/Users/don/Pictures/diffusion-renders/`). Don't change without asking.

## Non-obvious invariants

Read these before editing generation code:

1. **FaceID max side is 512.** The SD1.5 FaceID LoRA was trained at 512;
   generating larger corrupts output. `batch.py` downscales requested
   `width`/`height` to fit (`downscale_for_faceid`), generates at that
   size, then resizes the output back up to the requested dimensions with
   LANCZOS. Don't remove this dance.
2. **FaceID uses embeddings, not images.** Non-FaceID IP-Adapter is fed
   `ip_adapter_image=<PIL image>`. FaceID is fed
   `ip_adapter_image_embeds=[<tensor>]` produced by `insightface`. The
   two paths are not interchangeable.
3. **Blank-reference fallback.** When a model has `ip_adapter` configured
   but a batch item omits `ip_adapter_image`, the pipe is still called
   with an `ip_adapter_image` (or embeds) argument — a 224×224 black
   image or zeroed embeds — and scale set to `0.0`. This keeps the pipe
   call signature consistent across items. Dropping the arg can break the
   pipeline.
4. **`load_models` transforms in place.** String values for `type` and
   `dtype` are converted to `ModelType` / `torch.dtype` on load. Callers
   (and tests) should expect the enum/tensor types, not strings.
5. **Device picked once via `get_device()`** — mps → cuda → cpu. Any new
   helper that allocates tensors should accept a device argument rather
   than re-detecting.

## Testing philosophy

- Pipelines are mocked; tests run in seconds with no network/GPU.
- `run_batch` tests patch `load_models`, `load_pipeline`, `get_device`,
  `make_face_app`, `load_ip_adapter_into` on the `batch` module (the
  names the function looks up). See `tests/test_batch.py` for the
  pattern.
- Prefer adding a test alongside any new behavior — this module is small
  enough that the suite stays fast.

## Style / tooling

- Python 3.13 (`.python-version`). No type-checker configured; add type
  hints when they clarify intent, don't chase 100% coverage.
- No linter/formatter wired up. Match surrounding style.
- Keep `main.py` small — new flags/subcommands are fine, but put their
  implementation in a module, not inline.
