"""Tests for image_helpers (tensor encode/decode)."""

import struct
import numpy as np
from PIL import Image
from drawthings.image_helpers import convert_response_image, convert_image_for_request


def _make_test_tensor(width, height, channels=3):
    """Create a fake DTTensor with known pixel values."""
    header = bytearray(68)
    struct.pack_into("<I", header, 6 * 4, height)
    struct.pack_into("<I", header, 7 * 4, width)
    struct.pack_into("<I", header, 8 * 4, channels)

    # Use 0.0 float16 values → should decode to 127
    f16 = np.zeros(width * height * channels, dtype=np.float16)
    return bytes(header) + f16.tobytes()


def test_convert_response_image_dimensions():
    tensor = _make_test_tensor(64, 48, 3)
    img = convert_response_image(tensor)
    assert img.size == (64, 48)
    assert img.mode == "RGB"


def test_convert_response_image_pixel_values():
    tensor = _make_test_tensor(2, 2, 3)
    img = convert_response_image(tensor)
    arr = np.array(img)
    # float16 value 0.0 → (0 + 1) * 127 = 127
    assert np.all(arr == 127)


def test_convert_response_image_4channel():
    tensor = _make_test_tensor(4, 4, 4)
    img = convert_response_image(tensor)
    assert img.size == (4, 4)
    assert img.mode == "RGB"


def test_convert_image_for_request_roundtrip():
    # Create a simple test image
    arr = np.full((32, 32, 3), 127, dtype=np.uint8)
    img = Image.fromarray(arr, "RGB")

    tensor = convert_image_for_request(img)
    assert isinstance(tensor, bytes)
    assert len(tensor) == 68 + 32 * 32 * 3 * 2  # header + float16 payload

    # Verify header
    header = struct.unpack_from("<17I", tensor, 0)
    assert header[6] == 32  # height
    assert header[7] == 32  # width
    assert header[8] == 3   # channels


def test_convert_image_for_request_resize():
    arr = np.full((100, 200, 3), 127, dtype=np.uint8)
    img = Image.fromarray(arr, "RGB")

    tensor = convert_image_for_request(img, width=64, height=64)
    header = struct.unpack_from("<17I", tensor, 0)
    assert header[6] == 64
    assert header[7] == 64
