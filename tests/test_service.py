"""Tests for DTService (mocked gRPC)."""

import json
from unittest.mock import MagicMock, patch
from drawthings.service import DTService


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
