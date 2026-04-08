"""Image tensor encoding/decoding for Draw Things DTTensor format.

DTTensor layout:
  - 68-byte header (17 × uint32): metadata flags, height, width, channels
  - Payload: float16 pixel data (width × height × channels × 2 bytes)

Response images: float16 → uint8 via clamp((f16 + 1) * 127)
Request images:  uint8 → float16 via (u8 / 127) - 1
"""

import struct
import numpy as np
from PIL import Image


def convert_response_image(tensor_bytes):
    """Decode a DTTensor (from GenerateImage response) into (PIL.Image, channels).

    Returns a PIL Image in RGB mode.
    """
    buf = memoryview(tensor_bytes)
    header = struct.unpack_from("<17I", buf, 0)
    height, width, channels = header[6], header[7], header[8]

    offset = 68
    f16 = np.frombuffer(buf[offset:], dtype=np.float16)
    expected = width * height * channels
    f16 = f16[:expected]

    # Convert float16 → uint8: clamp((val + 1) * 127, 0, 255)
    u8 = np.clip((f16.astype(np.float32) + 1.0) * 127.0, 0, 255).astype(np.uint8)

    if channels == 4:
        u8 = u8.reshape((height, width, 4))[:, :, :3]
    elif channels == 3:
        u8 = u8.reshape((height, width, 3))
    else:
        u8 = u8.reshape((height, width))

    return Image.fromarray(u8, "RGB" if u8.ndim == 3 else "L")


def save_response_image(tensor_bytes, path):
    """Decode a DTTensor and save to a file (PNG, JPEG, etc.)."""
    img = convert_response_image(tensor_bytes)
    img.save(path)
    return path


def convert_image_for_request(img, width=None, height=None):
    """Convert a PIL Image to DTTensor bytes for the gRPC request.

    If width/height are given the image is resized first.
    Returns bytes (68-byte header + float16 payload).
    """
    if width and height:
        img = img.resize((width, height), Image.LANCZOS)

    arr = np.array(img, dtype=np.float32)
    if arr.ndim == 2:
        arr = np.expand_dims(arr, -1)

    h, w, c = arr.shape

    # uint8 → float16: (val / 127) - 1
    f16 = ((arr / 127.0) - 1.0).astype(np.float16)

    header = bytearray(68)
    struct.pack_into("<I", header, 0 * 4, 0)       # type
    struct.pack_into("<I", header, 1 * 4, 1)       # format
    struct.pack_into("<I", header, 2 * 4, 2)       # datatype
    struct.pack_into("<I", header, 3 * 4, 131072)  # reserved
    struct.pack_into("<I", header, 5 * 4, 1)
    struct.pack_into("<I", header, 6 * 4, h)
    struct.pack_into("<I", header, 7 * 4, w)
    struct.pack_into("<I", header, 8 * 4, c)

    return bytes(header) + f16.tobytes()
