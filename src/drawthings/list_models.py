#!/usr/bin/env python3
"""list_models.py — List available models, LoRAs, ControlNets, etc.

Usage:
    python scripts/list_models.py
    python scripts/list_models.py --host localhost:7859
    python scripts/list_models.py --type models

Outputs JSON to stdout.
"""

import argparse
import json
import sys


VALID_TYPES = ("models", "loras", "control_nets", "upscalers", "textual_inversions")


def main():
    parser = argparse.ArgumentParser(description="List available assets on Draw Things server")
    parser.add_argument("--host", default="localhost:7859", help="gRPC server address")
    parser.add_argument("--type", default=None, choices=VALID_TYPES,
                        help="Filter to a single asset type")
    args = parser.parse_args()

    try:
        from drawthings.service import DTService

        svc = DTService(args.host)
        data = svc.list_models()

        # Extract just filenames for the summary view
        result = {}
        for key in VALID_TYPES:
            items = data.get(key, [])
            result[key] = [item.get("file", item) if isinstance(item, dict) else item
                           for item in items]

        if args.type:
            print(json.dumps({"success": True, "type": args.type, "items": result[args.type]}))
        else:
            print(json.dumps({"success": True, **result}))
    except Exception as e:
        print(json.dumps({"success": False, "error": str(e)}), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
