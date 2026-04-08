#!/usr/bin/env python3
"""generate.py — Text-to-image generation via Draw Things gRPC API.

Usage:
    python scripts/generate.py \\
        --prompt "a golden retriever on a beach at sunset" \\
        --model "flux_qwen_srpo_v1.0_f16.ckpt" \\
        --output /tmp/result.png

Outputs JSON to stdout: { "success": true, "output": "/absolute/path/to/output.png" }
"""

import argparse
import json
import os
import sys

def main():
    parser = argparse.ArgumentParser(description="Generate an image from a text prompt")
    parser.add_argument("--prompt", required=True, help="Positive prompt text")
    parser.add_argument("--negative", default="", help="Negative prompt text")
    parser.add_argument("--model", default=None, help="Model filename")
    parser.add_argument("--width", type=int, default=512, help="Image width (rounded to 64)")
    parser.add_argument("--height", type=int, default=512, help="Image height (rounded to 64)")
    parser.add_argument("--steps", type=int, default=20, help="Diffusion steps")
    parser.add_argument("--guidance", type=float, default=None, help="CFG guidance scale")
    parser.add_argument("--seed", type=int, default=None, help="Seed for reproducibility")
    parser.add_argument("--output", default="./output.png", help="Output file path")
    parser.add_argument("--host", default="localhost:7859", help="gRPC server address")
    args = parser.parse_args()

    try:
        from drawthings.service import DTService

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

        svc = DTService(args.host)
        images = svc.generate(args.prompt, args.negative, config=config)

        output_path = os.path.abspath(args.output)
        images[0].save(output_path)

        print(json.dumps({"success": True, "output": output_path}))
    except Exception as e:
        print(json.dumps({"success": False, "error": str(e)}), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
