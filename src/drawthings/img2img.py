#!/usr/bin/env python3
"""img2img.py — Image-to-image generation via Draw Things gRPC API.

Usage:
    python scripts/img2img.py \\
        --input /path/to/source.png \\
        --prompt "same scene but in winter with snow" \\
        --strength 0.6 \\
        --output /tmp/result.png

Outputs JSON to stdout: { "success": true, "output": "/absolute/path/to/output.png" }
"""

import argparse
import json
import os
import sys


def main():
    parser = argparse.ArgumentParser(description="Modify an existing image with a prompt")
    parser.add_argument("--input", required=True, help="Path to source image")
    parser.add_argument("--prompt", required=True, help="Positive prompt text")
    parser.add_argument("--negative", default="", help="Negative prompt text")
    parser.add_argument("--model", default=None, help="Model filename")
    parser.add_argument("--width", type=int, default=512, help="Image width (rounded to 64)")
    parser.add_argument("--height", type=int, default=512, help="Image height (rounded to 64)")
    parser.add_argument("--steps", type=int, default=20, help="Diffusion steps")
    parser.add_argument("--strength", type=float, default=0.6, help="How much to change (0=none, 1=full)")
    parser.add_argument("--guidance", type=float, default=None, help="CFG guidance scale")
    parser.add_argument("--seed", type=int, default=None, help="Seed for reproducibility")
    parser.add_argument("--output", default="./output.png", help="Output file path")
    parser.add_argument("--host", default="localhost:7859", help="gRPC server address")
    args = parser.parse_args()

    try:
        from PIL import Image
        from drawthings.service import DTService

        def _progress(info):
            print(json.dumps({"progress": info}), file=sys.stderr, flush=True)

        config = {
            "width": args.width,
            "height": args.height,
            "steps": args.steps,
        }
        if args.model:
            config["model"] = args.model
        if args.seed is not None:
            config["seed"] = args.seed
        if args.guidance is not None:
            config["guidance_scale"] = args.guidance

        source = Image.open(os.path.abspath(args.input))
        svc = DTService(args.host)
        images = svc.img2img(source, args.prompt, args.negative,
                             strength=args.strength, config=config,
                             progress_callback=_progress)

        output_path = os.path.abspath(args.output)
        images[0].save(output_path)

        print(json.dumps({"success": True, "output": output_path}))
    except Exception as e:
        print(json.dumps({"success": False, "error": str(e)}), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
