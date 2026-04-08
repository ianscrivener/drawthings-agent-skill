"""DTService — gRPC client for the Draw Things image generation server."""

import hashlib
import json
import socket

import grpc

from drawthings.cred import get_credentials
from drawthings.config import build_config_buffer, DEFAULT_CONFIG
from drawthings.image_helpers import convert_response_image, convert_image_for_request
from drawthings.generated import imageService_pb2 as pb
from drawthings.generated import imageService_pb2_grpc as pb_grpc


def _sha256(data):
    return hashlib.sha256(data).digest()


def _round64(v, minimum=64):
    return max(round(v / 64) * 64, minimum)


class DTService:
    """High-level client for the Draw Things gRPC API.

    Usage::

        svc = DTService("localhost:7859")
        models = svc.list_models()
        images = svc.generate("a cat on a beach", model="flux_qwen_srpo_v1.0_f16.ckpt")
        images[0].save("/tmp/cat.png")
    """

    def __init__(self, address="localhost:7859"):
        self.address = address
        creds = get_credentials()
        self.channel = grpc.secure_channel(
            address,
            creds,
            options=[
                ("grpc.max_receive_message_length", -1),
                ("grpc.max_send_message_length", -1),
            ],
        )
        self.stub = pb_grpc.ImageGenerationServiceStub(self.channel)

    # ── Echo / list models ─────────────────────────────────────────────

    def echo(self, name=None):
        """Call Echo RPC. Returns the raw EchoReply protobuf."""
        req = pb.EchoRequest(name=name or socket.gethostname())
        return self.stub.Echo(req)

    def list_models(self):
        """Query the server for available models, LoRAs, controlnets, etc.

        Returns a dict with keys: models, loras, control_nets, upscalers, textual_inversions.
        Each value is a list of dicts parsed from the override metadata.
        """
        reply = self.echo()
        override = reply.override

        def _decode(buf):
            if not buf:
                return []
            return json.loads(buf.decode("utf-8"))

        return {
            "models": _decode(override.models),
            "loras": _decode(override.loras),
            "control_nets": _decode(override.controlNets),
            "upscalers": _decode(override.upscalers),
            "textual_inversions": _decode(override.textualInversions),
        }

    # ── Generate (txt2img) ─────────────────────────────────────────────

    def generate(self, prompt, negative_prompt="", config=None,
                 image_bytes=None, mask_bytes=None):
        """Generate image(s) from a text prompt.

        Args:
            prompt: The positive prompt.
            negative_prompt: Negative prompt text.
            config: Dict of generation config overrides (snake_case keys).
            image_bytes: Optional source image as DTTensor bytes (for img2img).
            mask_bytes: Optional mask as DTTensor bytes (for inpainting).

        Returns:
            List of PIL Images.
        """
        c = {**DEFAULT_CONFIG, **(config or {})}
        cfg_buf = build_config_buffer(c)

        contents = []
        image_hash = None
        mask_hash = None

        if image_bytes is not None:
            image_hash = _sha256(image_bytes)
            contents.append(image_bytes)
        if mask_bytes is not None:
            mask_hash = _sha256(mask_bytes)
            contents.append(mask_bytes)

        req = pb.ImageGenerationRequest(
            scaleFactor=1,
            user=socket.gethostname(),
            device=pb.LAPTOP,
            configuration=cfg_buf,
            prompt=prompt,
            negativePrompt=negative_prompt,
            image=image_hash,
            mask=mask_hash,
            contents=contents,
        )

        images = []
        for response in self.stub.GenerateImage(req):
            for img_data in response.generatedImages:
                images.append(convert_response_image(bytes(img_data)))

        return images

    # ── img2img helper ─────────────────────────────────────────────────

    def img2img(self, source_image, prompt, negative_prompt="",
                strength=0.6, config=None):
        """Modify an existing image according to a prompt.

        Args:
            source_image: PIL Image (the source).
            prompt: Text prompt describing desired changes.
            negative_prompt: Negative prompt.
            strength: How much to change (0=none, 1=completely replace).
            config: Additional config overrides.

        Returns:
            List of PIL Images.
        """
        c = {**(config or {}), "strength": strength}
        width = _round64(c.get("width", source_image.width))
        height = _round64(c.get("height", source_image.height))
        c["width"] = width
        c["height"] = height

        tensor = convert_image_for_request(source_image, width, height)
        return self.generate(prompt, negative_prompt, config=c, image_bytes=tensor)
