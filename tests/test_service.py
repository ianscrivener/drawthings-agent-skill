"""Tests for DTService (mocked gRPC)."""

import json
from unittest.mock import MagicMock, call, patch
from drawthings.service import DTService, _parse_signpost


def test_dtservice_init():
    with patch("drawthings.service.grpc.secure_channel") as mock_channel:
        mock_channel.return_value = MagicMock()
        svc = DTService("localhost:7859")
        assert svc.address == "localhost:7859"
        mock_channel.assert_called_once()


def test_list_models_decodes_json():
    with patch("drawthings.service.grpc.secure_channel"):
        svc = DTService("localhost:7859")

        mock_reply = MagicMock()
        mock_override = MagicMock()
        mock_override.models = json.dumps([{"file": "sd_v1.5_f16.ckpt", "name": "SD 1.5"}]).encode()
        mock_override.loras = b"[]"
        mock_override.controlNets = b"[]"
        mock_override.upscalers = b"[]"
        mock_override.textualInversions = b"[]"
        mock_reply.override = mock_override

        svc.stub = MagicMock()
        svc.stub.Echo.return_value = mock_reply

        result = svc.list_models()
        assert len(result["models"]) == 1
        assert result["models"][0]["file"] == "sd_v1.5_f16.ckpt"
        assert result["loras"] == []


def test_parse_signpost_sampling():
    """_parse_signpost extracts stage and step from a sampling signpost."""
    signpost = MagicMock()
    signpost.WhichOneof.return_value = "sampling"
    signpost.sampling.step = 5
    result = _parse_signpost(signpost)
    assert result == {"stage": "sampling", "step": 5}


def test_parse_signpost_no_step():
    """_parse_signpost returns stage only when inner message has no step."""
    signpost = MagicMock()
    signpost.WhichOneof.return_value = "textEncoded"
    inner = MagicMock(spec=[])  # no attributes at all
    signpost.textEncoded = inner
    result = _parse_signpost(signpost)
    assert result == {"stage": "textEncoded"}


def test_parse_signpost_none():
    """_parse_signpost returns None when no signpost field is set."""
    signpost = MagicMock()
    signpost.WhichOneof.return_value = None
    assert _parse_signpost(signpost) is None


def test_generate_calls_progress_callback():
    """generate() calls progress_callback for each signpost in the stream."""
    with patch("drawthings.service.grpc.secure_channel"):
        svc = DTService("localhost:7859")
        svc.stub = MagicMock()

        # Build two mock streaming responses: one signpost, one with image
        resp1 = MagicMock()
        resp1.HasField.return_value = True
        signpost1 = MagicMock()
        signpost1.WhichOneof.return_value = "sampling"
        signpost1.sampling.step = 3
        resp1.currentSignpost = signpost1
        resp1.generatedImages = []

        resp2 = MagicMock()
        resp2.HasField.return_value = False
        resp2.generatedImages = [b"\x00" * 100]

        svc.stub.GenerateImage.return_value = iter([resp1, resp2])

        cb = MagicMock()
        with patch("drawthings.service.convert_response_image") as mock_convert:
            mock_convert.return_value = MagicMock()
            images = svc.generate("test", progress_callback=cb)

        cb.assert_called_once_with({"stage": "sampling", "step": 3})
        assert len(images) == 1


def test_generate_no_callback_skips_progress():
    """generate() works without a progress_callback (no error)."""
    with patch("drawthings.service.grpc.secure_channel"):
        svc = DTService("localhost:7859")
        svc.stub = MagicMock()

        resp = MagicMock()
        resp.HasField.return_value = True
        signpost = MagicMock()
        signpost.WhichOneof.return_value = "textEncoded"
        inner = MagicMock(spec=[])
        signpost.textEncoded = inner
        resp.currentSignpost = signpost
        resp.generatedImages = [b"\x00" * 100]

        svc.stub.GenerateImage.return_value = iter([resp])

        with patch("drawthings.service.convert_response_image") as mock_convert:
            mock_convert.return_value = MagicMock()
            images = svc.generate("test")  # no callback — should not raise

        assert len(images) == 1
