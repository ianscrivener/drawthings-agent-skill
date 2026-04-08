---
name: drawthings
description: "Generate, render, or create images using Draw Things — a Mac-based Stable Diffusion compatible AI image renderer — via its gRPC API. Use this skill when asked to generate images, create artwork, render a scene, produce an image, do text-to-image, img2img, image-to-image, or when the user wants to use DrawThings, Draw Things, or stable diffusion locally. Also use to list available models, LoRAs, or ControlNets on the Draw Things server."
argument-hint: "describe what image you want to generate"
---

# DrawThings Image Generation Skill

Generates images via the Draw Things gRPC API running locally on macOS.
Supports text-to-image, image-to-image, and server introspection.

## Prerequisites

**Draw Things** must be running on macOS with the gRPC server enabled (default port `7859`).
Enable it in Draw Things → Settings → API Server.

## Environment Setup (agent-managed)

Before running any skill script, the agent must check and prepare the environment.
All setup scripts run with the system Python and have no dependencies on the drawthings package.

### Step 1: Check environment

```bash
python src/drawthings/check_env.py --host localhost:7859
```

Returns JSON with `ready: true/false`, a `checks` object, `missing` list, and `venv_python` path.
If `ready` is `true`, skip to running scripts. Use the `venv_python` value from the response to invoke skill scripts.

### Step 2: Setup (only if check reports not ready)

```bash
python src/drawthings/setup_env.py
```

Creates the `.venv`, installs all dependencies via `uv sync`, and verifies the package is importable.
Returns JSON with `success: true/false` and the `venv_python` path.
This is idempotent — safe to run repeatedly.

### Step 3: Run skill scripts

Use the `venv_python` path from check/setup (typically `.venv/bin/python`) to run scripts:

```bash
.venv/bin/python src/drawthings/generate.py --prompt "..."
.venv/bin/python src/drawthings/list_models.py
.venv/bin/python src/drawthings/img2img.py --input /path/to/image.png --prompt "..."
```

## Available Scripts

| Script | Purpose |
|--------|----------|
| [src/drawthings/check_env.py](src/drawthings/check_env.py) | Validate environment (zero-dependency) |
| [src/drawthings/setup_env.py](src/drawthings/setup_env.py) | Create venv and install deps (zero-dependency) |
| [src/drawthings/generate.py](src/drawthings/generate.py) | Text-to-image generation |
| [src/drawthings/img2img.py](src/drawthings/img2img.py) | Image-to-image (modify an existing image) |
| [src/drawthings/list_models.py](src/drawthings/list_models.py) | List available models, LoRAs, ControlNets |

All scripts output JSON to stdout. Errors are written to stderr with a non-zero exit code.

---

## Procedure: Generate an image from a text prompt

1. Optionally run `list_models.py` to find an available model name
2. Run `generate.py` with the prompt and model
3. The saved file path is returned in the JSON output as `output`

```bash
python src/drawthings/generate.py \
  --prompt "a golden retriever on a beach at sunset" \
  --model "flux_qwen_srpo_v1.0_f16.ckpt" \
  --width 512 --height 512 \
  --output /tmp/result.png
```

**All flags for generate.py:**

| Flag | Default | Description |
|------|---------|-------------|
| `--prompt` | *(required)* | Positive prompt text |
| `--negative` | `""` | Negative prompt text |
| `--model` | server default | Model filename (e.g. `flux_qwen_srpo_v1.0_f16.ckpt`) |
| `--width` | `512` | Image width in pixels (rounded to nearest 64) |
| `--height` | `512` | Image height in pixels (rounded to nearest 64) |
| `--steps` | `20` | Diffusion steps |
| `--guidance` | server default | CFG guidance scale (e.g. `7.5`) |
| `--seed` | random | Seed for reproducibility |
| `--output` | `./output.png` | Output file path (must end in `.png` or `.jpg`) |
| `--host` | `localhost:7859` | Draw Things gRPC server address |

**Output JSON:**
```json
{ "success": true, "output": "/absolute/path/to/output.png" }
```

---

## Procedure: Modify an existing image (img2img)

```bash
python src/drawthings/img2img.py \
  --input /path/to/source.png \
  --prompt "same scene but in winter with snow" \
  --strength 0.6 \
  --output /tmp/result.png
```

**All flags for img2img.py:**

Same flags as `generate.py`, plus:

| Flag | Default | Description |
|------|---------|-------------|
| `--input` | *(required)* | Path to source image |
| `--strength` | `0.6` | How much to change the image (0=none, 1=full) |

---

## Procedure: List available models

```bash
python src/drawthings/list_models.py --host localhost:7859
```

**Output JSON:**
```json
{
  "models": ["flux_qwen_srpo_v1.0_f16.ckpt", "sd_v1.5_f16.ckpt"],
  "loras": ["my_style.safetensors"],
  "control_nets": ["controlnet_depth_1.x_v1.1_f16.ckpt"],
  "upscalers": ["realesrgan_x2plus_f16.ckpt"],
  "textual_inversions": []
}
```

---

## Config Reference

See [assets/references/config-options.md](assets/references/config-options.md) for the full list of generation parameters.

## Troubleshooting

- **Connection refused**: Draw Things is not running, or the API server is not enabled in Settings
- **Module not found**: Run `python src/drawthings/check_env.py` — if not ready, run `python src/drawthings/setup_env.py`
- **Model not found**: Run `list_models.py` to see available model filenames; use the exact filename including extension
- **Image too dark/bright**: Adjust `--guidance` (typical range 5–12 for SD models, 1–4 for Flux)
