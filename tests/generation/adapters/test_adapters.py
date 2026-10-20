# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the BYOM adapter layer (no live endpoints)."""

import base64
import hashlib
import inspect
import json
import logging
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import requests

from generation.adapters import (
    NIMAdapter,
    OpenAIChatAdapter,
    OpenAIImageEditAdapter,
    OpenAIVideoAsyncAdapter,
    OpenAIVideoSyncAdapter,
    PassthroughAdapter,
    Result,
)
from generation.adapters.base import redact_media, resolve_api_key

LOGGER = logging.getLogger("test")

PNG_BYTES = b"\x89PNG\r\n\x1a\nfake-png"
PNG_B64 = base64.b64encode(PNG_BYTES).decode()


def _endpoint(**kw):
    """Build a duck-typed endpoint object with sensible defaults."""
    defaults = dict(
        id="ep",
        role="image_edit",
        url="http://localhost:9999/v1",
        model="some/model",
        adapter=None,
        api_key_env=None,
        timeout=30.0,
    )
    defaults.update(kw)
    return SimpleNamespace(**defaults)


def _image_payload(prompt="edit this", with_media=True):
    payload = {
        "prompt": prompt,
        "media": {},
        "params": {"seed": 0, "num_inference_steps": 30},
    }
    if with_media:
        payload["media"] = {
            "rgb": {"bytes": PNG_BYTES, "mime": "image/png", "path": "in.png"}
        }
    return payload


# ---------------------------------------------------------------------------
# resolve_api_key
# ---------------------------------------------------------------------------
class TestResolveApiKey:
    def test_explicit_env_name(self, monkeypatch):
        monkeypatch.setenv("MY_KEY", "abc")
        ep = _endpoint(api_key_env="MY_KEY")
        assert resolve_api_key(ep) == "abc"

    def test_role_default_env(self, monkeypatch):
        monkeypatch.setenv("LLM_API_KEY", "llmkey")
        ep = _endpoint(role="llm")
        assert resolve_api_key(ep) == "llmkey"

    def test_none_when_unset(self, monkeypatch):
        monkeypatch.delenv("BUILD_NVIDIA_API_KEY", raising=False)
        ep = _endpoint(role="image_edit")
        assert resolve_api_key(ep) is None


# ---------------------------------------------------------------------------
# OpenAIChatAdapter
# ---------------------------------------------------------------------------
class TestOpenAIChatAdapter:
    def _make(self, fake_response):
        with patch("generation.adapters.openai_chat.OpenAI") as mock_openai:
            client = MagicMock()
            client.chat.completions.create.return_value = fake_response
            mock_openai.return_value = client
            adapter = OpenAIChatAdapter(_endpoint(), LOGGER)
        return adapter, client

    def test_text_out(self):
        resp = SimpleNamespace(
            id="r1",
            choices=[SimpleNamespace(message=SimpleNamespace(content="a caption"))],
        )
        adapter, _ = self._make(resp)
        result = adapter.invoke({"prompt": "describe", "media": {}, "params": {}})
        assert result.text == "a caption"
        assert result.media_bytes is None
        assert result.request_id == "r1"

    def test_image_out_and_params_passthrough(self):
        item = SimpleNamespace(
            image_url=SimpleNamespace(url=f"data:image/png;base64,{PNG_B64}")
        )
        resp = SimpleNamespace(
            id="r2", choices=[SimpleNamespace(message=SimpleNamespace(content=[item]))]
        )
        adapter, client = self._make(resp)
        result = adapter.invoke(_image_payload())
        assert result.media_bytes == PNG_BYTES
        # Params pass straight into the body (no adapter-imposed envelope).
        _, kwargs = client.chat.completions.create.call_args
        assert kwargs["extra_body"] == {"seed": 0, "num_inference_steps": 30}
        # The user message must carry the data-URL image.
        content = kwargs["messages"][0]["content"]
        assert any(c["type"] == "image_url" for c in content)

    def test_config_controls_envelope_shape(self):
        # A nested envelope in params (Qwen's `extra_body`, or any model-specific
        # shape) passes through verbatim — the adapter imposes nothing.
        item = SimpleNamespace(
            image_url=SimpleNamespace(url=f"data:image/png;base64,{PNG_B64}")
        )
        resp = SimpleNamespace(
            id="r2b", choices=[SimpleNamespace(message=SimpleNamespace(content=[item]))]
        )
        adapter, client = self._make(resp)
        payload = _image_payload()
        payload["params"] = {"extra_body": {"num_inference_steps": 30, "seed": 0}}
        adapter.invoke(payload)
        _, kwargs = client.chat.completions.create.call_args
        assert kwargs["extra_body"] == {
            "extra_body": {"num_inference_steps": 30, "seed": 0}
        }

    def test_image_out_dict_fallback(self):
        item = {"image_url": {"url": f"data:image/png;base64,{PNG_B64}"}}
        resp = SimpleNamespace(
            id="r3", choices=[SimpleNamespace(message=SimpleNamespace(content=[item]))]
        )
        adapter, _ = self._make(resp)
        assert adapter.invoke(_image_payload()).media_bytes == PNG_BYTES

    def test_image_out_scans_all_content_parts(self):
        # A leading text/refusal part must not hide an image in a later part.
        text_part = SimpleNamespace(type="text", text="here you go")
        image_part = SimpleNamespace(
            image_url=SimpleNamespace(url=f"data:image/png;base64,{PNG_B64}")
        )
        resp = SimpleNamespace(
            id="r4",
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content=[text_part, image_part])
                )
            ],
        )
        adapter, _ = self._make(resp)
        assert adapter.invoke(_image_payload()).media_bytes == PNG_BYTES

    def test_chat_passthrough_injects_model(self):
        resp = SimpleNamespace(id="r4", choices=[])
        adapter, client = self._make(resp)
        adapter.chat(messages=[{"role": "user", "content": "hi"}], temperature=0.1)
        _, kwargs = client.chat.completions.create.call_args
        assert kwargs["model"] == "some/model"
        assert kwargs["temperature"] == 0.1

    def test_for_chat_builds_inline_endpoint(self, monkeypatch):
        monkeypatch.setenv("VLM_API_KEY", "vlmkey")
        with patch("generation.adapters.openai_chat.OpenAI") as mock_openai:
            adapter = OpenAIChatAdapter.for_chat(
                "http://host:1/v1/", "m/x", LOGGER, role="vlm", timeout=7200
            )
        assert adapter.url == "http://host:1/v1"  # trailing slash stripped
        assert adapter.model == "m/x"
        assert adapter.role == "vlm"
        assert adapter.timeout == 7200.0
        # API key resolved via the vlm role default env var.
        assert adapter.api_key == "vlmkey"
        # Client built with the no-retry policy and inline timeout.
        _, kwargs = mock_openai.call_args
        assert kwargs["max_retries"] == 0
        assert kwargs["timeout"] == 7200.0


# ---------------------------------------------------------------------------
# NIMAdapter
# ---------------------------------------------------------------------------
class TestNIMAdapter:
    def test_infer_body_and_artifact_decode(self):
        fake_resp = MagicMock()
        fake_resp.json.return_value = {"id": "nim1", "artifacts": [{"base64": PNG_B64}]}
        fake_resp.raise_for_status.return_value = None
        with patch(
            "generation.adapters.nim.requests.post", return_value=fake_resp
        ) as post:
            adapter = NIMAdapter(_endpoint(timeout=None), LOGGER)
            result = adapter.invoke(_image_payload(prompt="make it night"))
        url, kwargs = post.call_args[0][0], post.call_args[1]
        assert url == "http://localhost:9999/v1/infer"
        body = kwargs["json"]
        assert body["prompt"] == "make it night"
        assert body["seed"] == 0
        # Image input -> "image" field (chosen by MIME), as a data URL.
        assert body["image"] == f"data:image/png;base64,{PNG_B64}"
        assert result.media_bytes == PNG_BYTES
        assert result.request_id == "nim1"

    def test_video_input_and_nested_params_and_b64_video(self):
        """Video NIM (Cosmos): input under 'video', nested controls passed
        through verbatim from params, and a ``b64_video`` response decoded."""
        mp4_bytes = b"FAKE-MP4-DATA"
        mp4_b64 = base64.b64encode(mp4_bytes).decode()
        out_bytes = b"OUTPUT-MP4"
        fake_resp = MagicMock()
        fake_resp.json.return_value = {
            "b64_video": base64.b64encode(out_bytes).decode()
        }
        fake_resp.raise_for_status.return_value = None
        payload = {
            "prompt": "make it snowy",
            "media": {
                "rgb": {"bytes": mp4_bytes, "mime": "video/mp4", "path": "in.mp4"}
            },
            "params": {"guidance": 3.0, "edge": {"control_weight": 1.0}},
        }
        with patch(
            "generation.adapters.nim.requests.post", return_value=fake_resp
        ) as post:
            adapter = NIMAdapter(_endpoint(), LOGGER)
            result = adapter.invoke(payload)
        body = post.call_args[1]["json"]
        # MIME video/* -> "video" field, RAW base64 (no data: prefix; Cosmos NIM).
        assert body["video"] == mp4_b64
        assert not body["video"].startswith("data:")
        assert "image" not in body
        # Nested control structure flows straight from config params (no
        # hardcoding in the adapter).
        assert body["edge"] == {"control_weight": 1.0}
        assert body["guidance"] == 3.0
        assert result.media_bytes == out_bytes

    def test_transfer_serializes_all_control_inputs(self):
        """Cosmos Transfer: primary video -> 'video'; each of the FOUR control
        modalities (edge/depth/seg/vis) is nested as {control_weight?, control}.
        A null/absent control is never sent, so the NIM computes it online."""
        rgb = b"RGB-MP4"
        controls = {
            "edge": b"EDGE-MP4",
            "depth": b"DEPTH-MP4",
            "seg": b"SEG-MP4",
            "vis": b"VIS-MP4",
        }
        fake_resp = MagicMock()
        fake_resp.json.return_value = {"b64_video": base64.b64encode(b"OUT").decode()}
        fake_resp.raise_for_status.return_value = None
        media = {
            "rgb": {
                "bytes": rgb,
                "mime": "video/mp4",
                "path": "in.mp4",
                "kind": "video",
            }
        }
        for name, data in controls.items():
            media[name] = {
                "bytes": data,
                "mime": "video/mp4",
                "path": f"{name}.mp4",
                "kind": "control",
            }
        payload = {
            "prompt": "snowy night",
            "media": media,
            # edge weight comes from config params; the others carry only the video.
            "params": {"edge": {"control_weight": 1.0}},
        }
        with patch(
            "generation.adapters.nim.requests.post", return_value=fake_resp
        ) as post:
            NIMAdapter(_endpoint(), LOGGER).invoke(payload)
        body = post.call_args[1]["json"]
        # primary input video -> raw base64 under "video".
        assert body["video"] == base64.b64encode(rgb).decode()
        # ALL FOUR control modalities' read+encoded videos are nested per modality.
        for name, data in controls.items():
            assert body[name]["control"] == base64.b64encode(data).decode()
        # a configured control_weight is preserved alongside the injected control.
        assert body["edge"]["control_weight"] == 1.0

    def test_control_input_with_nonobject_param_raises(self):
        # If params already set a modality to a non-object, fail loud rather than
        # clobbering that value when attaching the control input.
        payload = {
            "prompt": "p",
            "media": {
                "rgb": {
                    "bytes": b"R",
                    "mime": "video/mp4",
                    "path": "in.mp4",
                    "kind": "video",
                },
                "edge": {
                    "bytes": b"E",
                    "mime": "video/mp4",
                    "path": "e.mp4",
                    "kind": "control",
                },
            },
            "params": {"edge": True},  # not an object -> conflicts with edge control
        }
        adapter = NIMAdapter(_endpoint(), LOGGER)
        with pytest.raises(ValueError, match="must be an object"):
            adapter.invoke(payload)

    def test_default_timeout_when_unset(self):
        adapter = NIMAdapter(_endpoint(timeout=None), LOGGER)
        assert adapter.timeout == 300.0

    @pytest.mark.parametrize(
        ("configured", "expected"), [("120", "120"), ("1000", "300")]
    )
    def test_nvcf_poll_header_uses_env_and_gateway_cap(
        self, monkeypatch, configured, expected
    ):
        monkeypatch.setenv("NVCF_POLL_SECONDS", configured)
        response = MagicMock(status_code=200, ok=True)
        response.json.return_value = {"artifacts": [{"base64": PNG_B64}]}
        with patch(
            "generation.adapters.nim.requests.post", return_value=response
        ) as post:
            NIMAdapter(
                _endpoint(
                    url="https://test.invocation.api.nvcf.nvidia.com/v1",
                    timeout=600,
                ),
                LOGGER,
            ).invoke(_image_payload())

        assert post.call_args.kwargs["headers"]["NVCF-POLL-SECONDS"] == expected

    def test_non_nvcf_omits_poll_header_and_does_not_poll(self):
        accepted = MagicMock(status_code=202, ok=True, headers={})
        accepted.json.return_value = {"status": "pending"}
        with (
            patch(
                "generation.adapters.nim.requests.post", return_value=accepted
            ) as post,
            patch("generation.adapters.nim.requests.get") as get,
        ):
            with pytest.raises(ValueError, match="no artifact"):
                NIMAdapter(_endpoint(), LOGGER).invoke(_image_payload())

        assert "NVCF-POLL-SECONDS" not in post.call_args.kwargs["headers"]
        get.assert_not_called()

    @pytest.mark.parametrize("configured", ["0", "not-a-number"])
    def test_invalid_nvcf_poll_seconds_fails_at_adapter_init(
        self, monkeypatch, configured
    ):
        monkeypatch.setenv("NVCF_POLL_SECONDS", configured)
        with pytest.raises(ValueError, match="NVCF_POLL_SECONDS"):
            NIMAdapter(
                _endpoint(url="https://test.invocation.api.nvcf.nvidia.com/v1"),
                LOGGER,
            )

    def test_nvcf_202_polls_until_artifact(self, monkeypatch):
        monkeypatch.setenv("BUILD_NVIDIA_API_KEY", "secret")
        monkeypatch.setenv("NVCF_STATUS_BASE_URL", "https://status.internal.test/x")
        accepted = MagicMock(
            status_code=202, ok=True, headers={"NVCF-REQID": "request/1"}
        )
        pending = MagicMock(status_code=202, ok=True)
        complete = MagicMock(status_code=200, ok=True)
        complete.json.return_value = {"artifacts": [{"base64": PNG_B64}]}

        with (
            patch("generation.adapters.nim.requests.post", return_value=accepted),
            patch(
                "generation.adapters.nim.requests.get",
                side_effect=[pending, complete],
            ) as get,
            patch("generation.adapters.nim.time.sleep") as sleep,
        ):
            result = NIMAdapter(
                _endpoint(
                    url="https://test.invocation.api.nvcf.nvidia.com/v1",
                    timeout=600,
                ),
                LOGGER,
            ).invoke(_image_payload())

        assert result.media_bytes == PNG_BYTES
        assert result.request_id == "request/1"
        assert get.call_count == 2
        assert get.call_args.args[0] == "https://status.internal.test/x/request%2F1"
        assert get.call_args.kwargs["headers"]["Authorization"] == "Bearer secret"
        assert "Content-Type" not in get.call_args.kwargs["headers"]
        sleep.assert_called_once_with(1)

    def test_nvcf_202_without_request_id_fails(self):
        accepted = MagicMock(status_code=202, ok=True, headers={})
        accepted.json.return_value = {"status": "pending"}

        with patch("generation.adapters.nim.requests.post", return_value=accepted):
            with pytest.raises(ValueError, match="without an NVCF-REQID"):
                NIMAdapter(
                    _endpoint(url="https://test.invocation.api.nvcf.nvidia.com/v1"),
                    LOGGER,
                ).invoke(_image_payload())

    def test_nvcf_polling_stops_at_endpoint_timeout(self):
        accepted = MagicMock(
            status_code=202, ok=True, headers={"NVCF-REQID": "request-3"}
        )

        with (
            patch("generation.adapters.nim.requests.post", return_value=accepted),
            patch("generation.adapters.nim.time.monotonic", side_effect=[10, 41]),
        ):
            with pytest.raises(requests.Timeout, match="did not complete within 30s"):
                NIMAdapter(
                    _endpoint(
                        url="https://test.invocation.api.nvcf.nvidia.com/v1",
                        timeout=30,
                    ),
                    LOGGER,
                ).invoke(_image_payload())

    def test_missing_artifact_raises(self):
        fake_resp = MagicMock()
        fake_resp.json.return_value = {"id": "x"}
        fake_resp.raise_for_status.return_value = None
        with patch("generation.adapters.nim.requests.post", return_value=fake_resp):
            adapter = NIMAdapter(_endpoint(), LOGGER)
            with pytest.raises(ValueError, match="no artifact"):
                adapter.invoke(_image_payload())

    def test_error_surfaces_response_body(self):
        """A non-2xx surfaces the NIM's response body (the real reason),
        not just the bare status line."""
        fake_resp = MagicMock()
        fake_resp.ok = False
        fake_resp.status_code = 500
        fake_resp.text = (
            '{"detail":"UnsupportedVideoFormat: Unsupported H264 pixel format '
            "'yuvj420p'. Only 4:2:0 is supported.\"}"
        )
        with patch("generation.adapters.nim.requests.post", return_value=fake_resp):
            adapter = NIMAdapter(_endpoint(), LOGGER)
            with pytest.raises(Exception, match="yuvj420p") as exc:
                adapter.invoke(_image_payload())
        # status code + body detail both present in the surfaced message
        assert "500" in str(exc.value)


# ---------------------------------------------------------------------------
# OpenAIVideoSyncAdapter (image-to-video: multipart /v1/videos/sync -> MP4 bytes)
# ---------------------------------------------------------------------------
class TestOpenAIVideoSyncAdapter:
    MP4_BYTES = b"FAKE-MP4-VIDEO-BYTES"

    def _video_payload(self, prompt="a warehouse, camera pushes forward"):
        return {
            "prompt": prompt,
            "media": {
                "rgb": {
                    "bytes": PNG_BYTES,
                    "mime": "image/png",
                    "path": "warehouse.png",
                }
            },
            "params": {
                "seed": 42,
                "size": "832x480",
                "num_frames": 189,
                "guidance_scale": 6.0,
                # nested dict -> JSON string form field
                "extra_params": {"guardrails": True, "use_resolution_template": False},
            },
        }

    def _resp(self, *, status=200, content=MP4_BYTES, ctype="video/mp4", text=""):
        return SimpleNamespace(
            ok=200 <= status < 300,
            status_code=status,
            content=content,
            text=text,
            headers={"content-type": ctype, "x-request-id": "vid1"},
            json=lambda: {},
        )

    def test_multipart_request_and_raw_bytes(self):
        ep = _endpoint(role="image2video", url="http://host:8000", model="")
        with patch(
            "generation.adapters.openai_video_sync.requests.post",
            return_value=self._resp(),
        ) as post:
            adapter = OpenAIVideoSyncAdapter(ep, LOGGER)
            result = adapter.invoke(self._video_payload())

        url, kwargs = post.call_args[0][0], post.call_args[1]
        # No /v1 in the base URL -> route appended cleanly (no doubling).
        assert url == "http://host:8000/v1/videos/sync"
        data = kwargs["data"]
        # Prompt + scalar params are stringified form fields.
        assert data["prompt"].startswith("a warehouse")
        assert data["size"] == "832x480"
        assert data["num_frames"] == "189"
        assert data["seed"] == "42"
        # Nested dict param -> JSON string (booleans lowercased by json).
        assert json.loads(data["extra_params"]) == {
            "guardrails": True,
            "use_resolution_template": False,
        }
        # model="" -> no model field sent (proven single-model contract).
        assert "model" not in data
        # First frame uploaded as the input_reference file part.
        assert kwargs["files"]["input_reference"] == (
            "warehouse.png",
            PNG_BYTES,
            "image/png",
        )
        # Accept negotiates raw MP4.
        assert kwargs["headers"]["Accept"] == "video/mp4"
        assert "NVCF-POLL-SECONDS" not in kwargs["headers"]
        # Raw bytes returned verbatim as media.
        assert result.media_bytes == self.MP4_BYTES
        assert result.request_id == "vid1"

    def test_nvcf_accepted_response_is_polled(self, monkeypatch):
        monkeypatch.setenv("NVCF_API_KEY", "k")
        monkeypatch.setenv("NVCF_STATUS_BASE_URL", "https://status.internal.test/x/")
        accepted = self._resp(status=202, content=b"")
        accepted.headers["NVCF-REQID"] = "req/1"
        finished = self._resp()
        ep = _endpoint(
            role="image2video",
            url="https://test.invocation.api.nvcf.nvidia.com/v1",
            timeout=120,
            api_key_env="NVCF_API_KEY",
        )
        with (
            patch(
                "generation.adapters.openai_video_sync.requests.post",
                return_value=accepted,
            ) as post,
            patch(
                "generation.adapters.openai_video_sync.requests.get",
                side_effect=[accepted, finished],
            ) as get,
            patch("generation.adapters.openai_video_sync.time.sleep") as sleep,
        ):
            result = OpenAIVideoSyncAdapter(ep, LOGGER).invoke(self._video_payload())

        assert post.call_args.kwargs["headers"]["NVCF-POLL-SECONDS"] == "119"
        assert get.call_args_list[0].args[0] == "https://status.internal.test/x/req%2F1"
        assert get.call_args_list[0].kwargs["headers"]["Authorization"] == "Bearer k"
        sleep.assert_called_once_with(1)
        assert result.media_bytes == self.MP4_BYTES
        assert result.request_id == "req/1"

    def test_control_inputs_become_extra_params_control_path(self):
        # Keep the legacy path contract for non-WSM control inputs.
        ep = _endpoint(role="video_transfer", url="http://host:8000", model="")
        payload = {
            "prompt": "night snow",
            "media": {
                "rgb": {
                    "bytes": PNG_BYTES,
                    "mime": "video/mp4",
                    "path": "in.mp4",
                    "kind": "video",
                },
                "depth": {
                    "bytes": b"DEPTH",
                    "mime": "video/mp4",
                    "path": "/data/depth.mp4",
                    "kind": "control",
                },
            },
            # config auto-computes edge; depth should be folded in as a path.
            "params": {"extra_params": {"edge": True}},
        }
        with patch(
            "generation.adapters.openai_video_sync.requests.post",
            return_value=self._resp(),
        ) as post:
            OpenAIVideoSyncAdapter(ep, LOGGER).invoke(payload)
        kwargs = post.call_args[1]
        extra = json.loads(kwargs["data"]["extra_params"])
        # config edge:true preserved; depth folded in as a control_path reference.
        assert extra["edge"] is True
        assert extra["depth"] == {"control_path": "/data/depth.mp4"}
        # only the primary rgb is uploaded; control bytes are NOT sent.
        assert kwargs["files"]["input_reference"][1] == PNG_BYTES

    def test_wsm_control_is_uploaded_with_rgb(self):
        ep = _endpoint(role="video_transfer", url="http://host:8000", model="")
        payload = {
            "prompt": "night snow",
            "media": {
                "rgb": {
                    "bytes": b"RGB-MP4",
                    "mime": "video/mp4",
                    "path": "RGB.mp4",
                    "kind": "video",
                },
                "wsm": {
                    "bytes": b"WSM-MP4",
                    "mime": "video/mp4",
                    "path": "/data/WSM_RGB.mp4",
                    "kind": "control",
                },
            },
            "params": {
                "extra_params": {"wsm": {"control_weight": 0.75}},
            },
        }
        with patch(
            "generation.adapters.openai_video_sync.requests.post",
            return_value=self._resp(),
        ) as post:
            OpenAIVideoSyncAdapter(ep, LOGGER).invoke(payload)

        kwargs = post.call_args[1]
        assert kwargs["data"]["control_type"] == "wsm"
        assert json.loads(kwargs["data"]["extra_params"])["wsm"] == {
            "control_weight": 0.75
        }
        assert kwargs["files"]["input_reference"] == (
            "RGB.mp4",
            b"RGB-MP4",
            "video/mp4",
        )
        assert kwargs["files"]["control_reference"] == (
            "WSM_RGB.mp4",
            b"WSM-MP4",
            "video/mp4",
        )

    def test_text_only_forces_multipart(self):
        # No input frame: the request must still be multipart (data= alone would
        # be x-www-form-urlencoded), so form fields ride through `files`.
        ep = _endpoint(role="image2video", url="http://host:8000", model="")
        payload = {"prompt": "a warehouse", "media": {}, "params": {"size": "832x480"}}
        with patch(
            "generation.adapters.openai_video_sync.requests.post",
            return_value=self._resp(),
        ) as post:
            adapter = OpenAIVideoSyncAdapter(ep, LOGGER)
            adapter.invoke(payload)
        kwargs = post.call_args[1]
        assert "data" not in kwargs  # not form-urlencoded
        files = dict(kwargs["files"])
        assert files["prompt"] == (None, "a warehouse")
        assert files["size"] == (None, "832x480")

    def test_model_injected_when_set(self):
        ep = _endpoint(
            role="image2video", url="http://host:8000/v1", model="nvidia/Cosmos3"
        )
        with patch(
            "generation.adapters.openai_video_sync.requests.post",
            return_value=self._resp(),
        ) as post:
            # URL ends in /v1 and route starts with /v1 -> no /v1/v1 doubling.
            adapter = OpenAIVideoSyncAdapter(ep, LOGGER)
            assert adapter.gen_url == "http://host:8000/v1/videos/sync"
            adapter.invoke(self._video_payload())
        assert post.call_args[1]["data"]["model"] == "nvidia/Cosmos3"

    def test_json_envelope_fallback(self):
        out = b"WRAPPED-MP4"
        resp = self._resp(
            content=b"{}",
            ctype="application/json",
        )
        resp.json = lambda: {"b64_video": base64.b64encode(out).decode()}
        with patch(
            "generation.adapters.openai_video_sync.requests.post", return_value=resp
        ):
            adapter = OpenAIVideoSyncAdapter(_endpoint(role="image2video"), LOGGER)
            result = adapter.invoke(self._video_payload())
        assert result.media_bytes == out

    def test_default_timeout_when_unset(self):
        adapter = OpenAIVideoSyncAdapter(
            _endpoint(role="image2video", timeout=None), LOGGER
        )
        assert adapter.timeout == 900.0

    def test_error_surfaces_response_body(self):
        resp = self._resp(
            status=422,
            content=b"",
            ctype="application/json",
            text='{"detail":"num_frames must be <= 189"}',
        )
        with patch(
            "generation.adapters.openai_video_sync.requests.post", return_value=resp
        ):
            adapter = OpenAIVideoSyncAdapter(_endpoint(role="image2video"), LOGGER)
            with pytest.raises(Exception, match="num_frames must be") as exc:
                adapter.invoke(self._video_payload())
        assert "422" in str(exc.value)

    def test_empty_response_raises(self):
        resp = self._resp(content=b"", ctype="video/mp4")
        with patch(
            "generation.adapters.openai_video_sync.requests.post", return_value=resp
        ):
            adapter = OpenAIVideoSyncAdapter(_endpoint(role="image2video"), LOGGER)
            with pytest.raises(ValueError, match="empty/undecodable"):
                adapter.invoke(self._video_payload())


# ---------------------------------------------------------------------------
# OpenAIVideoAsyncAdapter (async create -> poll job -> download content)
# ---------------------------------------------------------------------------
class TestOpenAIVideoAsyncAdapter:
    MP4_BYTES = b"FAKE-ASYNC-MP4-BYTES"
    POST = "generation.adapters.openai_video_async.requests.post"
    GET = "generation.adapters.openai_video_async.requests.get"
    SLEEP = "generation.adapters.openai_video_async.time.sleep"

    def _endpoint(self, **kw):
        # For the async contract the endpoint URL *is* the create route
        # (e.g. .../videos), with no /v1/videos/sync suffix. Veo requires `model`.
        defaults = dict(
            role="image2video",
            url="https://host/videos",
            model="gcp/google/veo-3.1-generate-001",
            adapter="openai.video.async",
            timeout=900.0,
        )
        defaults.update(kw)
        return _endpoint(**defaults)

    def _payload(self, prompt="a warehouse, pallet collapse"):
        return {
            "prompt": prompt,
            "media": {
                "rgb": {
                    "bytes": PNG_BYTES,
                    "mime": "image/png",
                    "path": "warehouse.png",
                }
            },
            "params": {"seed": 42, "seconds": 8, "size": "1280x720"},
        }

    def _job(self, body, *, status=200, ctype="application/json", text=""):
        """A JSON job-descriptor response (create / poll)."""
        return SimpleNamespace(
            ok=200 <= status < 300,
            status_code=status,
            content=b"",
            text=text,
            url="https://host/videos",
            headers={"content-type": ctype},
            json=lambda: body,
        )

    def _video(self, content, *, status=200, ctype="video/mp4"):
        """A raw-bytes content response (download)."""
        return SimpleNamespace(
            ok=200 <= status < 300,
            status_code=status,
            content=content,
            text="",
            url="https://host/videos",
            headers={"content-type": ctype},
            json=lambda: {},
        )

    def test_create_poll_download_happy_path(self):
        created = self._job(
            {
                "id": "video_abc",
                "status": "processing",
                "usage": {"duration_seconds": 8.0},
            }
        )
        poll_processing = self._job({"id": "video_abc", "status": "processing"})
        poll_done = self._job({"id": "video_abc", "status": "completed"})
        download = self._video(self.MP4_BYTES)
        with (
            patch(self.POST, return_value=created) as post,
            patch(self.GET, side_effect=[poll_processing, poll_done, download]) as get,
            patch(self.SLEEP) as sleep,
        ):
            adapter = OpenAIVideoAsyncAdapter(self._endpoint(), LOGGER)
            result = adapter.invoke(self._payload())

        # create: multipart POST to the URL itself (no route suffix).
        create_url, create_kwargs = post.call_args[0][0], post.call_args[1]
        assert create_url == "https://host/videos"
        data = create_kwargs["data"]
        assert data["prompt"].startswith("a warehouse")
        assert data["seconds"] == "8"
        assert data["size"] == "1280x720"
        assert data["seed"] == "42"
        # model is REQUIRED on this gateway (unlike the single-model sync server).
        assert data["model"] == "gcp/google/veo-3.1-generate-001"
        # first frame uploaded as the input_reference file part.
        assert create_kwargs["files"]["input_reference"] == (
            "warehouse.png",
            PNG_BYTES,
            "image/png",
        )
        # poll GET {url}/{id}, then download GET {url}/{id}/content (Accept mp4).
        assert get.call_args_list[0][0][0] == "https://host/videos/video_abc"
        assert get.call_args_list[-1][0][0] == "https://host/videos/video_abc/content"
        assert get.call_args_list[-1][1]["headers"]["Accept"] == "video/mp4"
        # polled until completed: 2 status polls + 1 content GET, slept once.
        assert get.call_count == 3
        assert sleep.call_count == 1
        # raw MP4 returned verbatim; request_id is the job id.
        assert result.media_bytes == self.MP4_BYTES
        assert result.request_id == "video_abc"

    def test_text_only_create_forces_multipart(self):
        # Text-to-video (no input_reference): create must stay multipart.
        created = self._job({"id": "vT", "status": "processing"})
        poll_done = self._job({"id": "vT", "status": "completed"})
        download = self._video(self.MP4_BYTES)
        payload = {"prompt": "a warehouse", "media": {}, "params": {"size": "1280x720"}}
        with (
            patch(self.POST, return_value=created) as post,
            patch(self.GET, side_effect=[poll_done, download]),
            patch(self.SLEEP),
        ):
            adapter = OpenAIVideoAsyncAdapter(self._endpoint(), LOGGER)
            adapter.invoke(payload)
        kwargs = post.call_args[1]
        assert "data" not in kwargs
        files = dict(kwargs["files"])
        assert files["prompt"] == (None, "a warehouse")
        assert files["model"] == (None, "gcp/google/veo-3.1-generate-001")

    def test_control_media_rejected(self):
        # Veo (async) has no control-transfer concept: control media must fail loud.
        payload = self._payload()
        payload["media"]["depth"] = {
            "bytes": PNG_BYTES,
            "mime": "video/mp4",
            "path": "d.mp4",
            "kind": "control",
        }
        adapter = OpenAIVideoAsyncAdapter(self._endpoint(), LOGGER)
        with pytest.raises(ValueError, match=r"control inputs \['depth'\]"):
            adapter.invoke(payload)

    def test_create_defensively_rejects_non_wsm_control(self):
        payload = self._payload()
        payload["media"]["depth"] = {
            "bytes": PNG_BYTES,
            "mime": "video/mp4",
            "path": "d.mp4",
            "kind": "control",
        }
        adapter = OpenAIVideoAsyncAdapter(self._endpoint(role="video_transfer"), LOGGER)

        with pytest.raises(ValueError, match=r"control inputs \['depth'\]"):
            adapter._create(payload, float("inf"))

    def test_cosmos3_wsm_control_is_uploaded_with_rgb(self):
        created = self._job({"id": "wsm-job", "status": "processing"})
        done = self._job({"id": "wsm-job", "status": "completed"})
        download = self._video(self.MP4_BYTES)
        payload = {
            "prompt": "follow the WSM control",
            "media": {
                "rgb": {
                    "bytes": b"RGB-MP4",
                    "mime": "video/mp4",
                    "path": "RGB.mp4",
                    "kind": "video",
                },
                "wsm": {
                    "bytes": b"WSM-MP4",
                    "mime": "video/mp4",
                    "path": "/data/WSM_RGB.mp4",
                    "kind": "control",
                },
            },
            "params": {"extra_params": {"wsm": {"control_weight": 0.75}}},
        }
        with (
            patch(self.POST, return_value=created) as post,
            patch(self.GET, side_effect=[done, download]),
            patch(self.SLEEP),
        ):
            adapter = OpenAIVideoAsyncAdapter(
                self._endpoint(role="video_transfer", model="nvidia/Cosmos3-Super"),
                LOGGER,
            )
            adapter.invoke(payload)

        kwargs = post.call_args[1]
        assert kwargs["data"]["control_type"] == "wsm"
        assert json.loads(kwargs["data"]["extra_params"])["wsm"] == {
            "control_weight": 0.75
        }
        assert kwargs["files"]["input_reference"] == (
            "RGB.mp4",
            b"RGB-MP4",
            "video/mp4",
        )
        assert kwargs["files"]["control_reference"] == (
            "WSM_RGB.mp4",
            b"WSM-MP4",
            "video/mp4",
        )

    def test_bearer_auth_header_sent(self, monkeypatch):
        monkeypatch.setenv("ASYNC_KEY", "sk-xyz")
        created = self._job({"id": "v1", "status": "processing"})
        done = self._job({"id": "v1", "status": "completed"})
        download = self._video(self.MP4_BYTES)
        with (
            patch(self.POST, return_value=created) as post,
            patch(self.GET, side_effect=[done, download]),
            patch(self.SLEEP),
        ):
            adapter = OpenAIVideoAsyncAdapter(
                self._endpoint(api_key_env="ASYNC_KEY"), LOGGER
            )
            adapter.invoke(self._payload())
        assert post.call_args[1]["headers"]["Authorization"] == "Bearer sk-xyz"

    def test_poll_failure_raises_with_reason(self):
        created = self._job({"id": "vF", "status": "processing"})
        failed = self._job(
            {
                "id": "vF",
                "status": "failed",
                "error": {"message": "safety blocked the prompt"},
            }
        )
        with (
            patch(self.POST, return_value=created),
            patch(self.GET, side_effect=[failed]),
            patch(self.SLEEP),
        ):
            adapter = OpenAIVideoAsyncAdapter(self._endpoint(), LOGGER)
            with pytest.raises(RuntimeError, match="safety blocked"):
                adapter.invoke(self._payload())

    @pytest.mark.parametrize(
        "poll_error",
        [
            requests.ReadTimeout("status request stalled"),
            requests.ConnectionError("status connection dropped"),
        ],
    )
    def test_transient_poll_error_reuses_submitted_job(self, poll_error):
        created = self._job({"id": "v1", "status": "processing"})
        done = self._job({"id": "v1", "status": "completed"})
        download = self._video(self.MP4_BYTES)
        with (
            patch(self.POST, return_value=created) as post,
            patch(self.GET, side_effect=[poll_error, done, download]) as get,
            patch(self.SLEEP) as sleep,
        ):
            adapter = OpenAIVideoAsyncAdapter(self._endpoint(), LOGGER)
            result = adapter.invoke(self._payload())

        assert result.request_id == "v1"
        assert result.media_bytes == self.MP4_BYTES
        post.assert_called_once()
        assert [call.args[0] for call in get.call_args_list] == [
            "https://host/videos/v1",
            "https://host/videos/v1",
            "https://host/videos/v1/content",
        ]
        sleep.assert_called_once()

    def test_repeated_poll_timeouts_end_at_job_deadline(self):
        now = [0.0]

        def timed_out_get(_url, *, timeout, **_kwargs):
            now[0] += timeout
            raise requests.ReadTimeout("status request stalled")

        def advance_clock(seconds):
            now[0] += seconds

        with (
            patch(self.GET, side_effect=timed_out_get) as get,
            patch(self.SLEEP, side_effect=advance_clock) as sleep,
            patch(
                "generation.adapters.openai_video_async.time.monotonic",
                side_effect=lambda: now[0],
            ),
        ):
            adapter = OpenAIVideoAsyncAdapter(self._endpoint(timeout=250), LOGGER)
            with pytest.raises(TimeoutError, match="video job v1 did not complete"):
                adapter._poll("v1", deadline=250)

        assert get.call_count == 2
        assert [call.kwargs["timeout"] for call in get.call_args_list] == [120, 120]
        assert [call.args[0] for call in get.call_args_list] == [
            "https://host/videos/v1",
            "https://host/videos/v1",
        ]
        assert [call.args[0] for call in sleep.call_args_list] == [8, 2]

    def test_poll_http_error_does_not_retry(self):
        created = self._job({"id": "v1", "status": "processing"})
        denied = self._job({}, status=403, text="forbidden")
        with (
            patch(self.POST, return_value=created) as post,
            patch(self.GET, return_value=denied) as get,
            patch(self.SLEEP) as sleep,
        ):
            adapter = OpenAIVideoAsyncAdapter(self._endpoint(), LOGGER)
            with pytest.raises(requests.HTTPError, match="HTTP 403"):
                adapter.invoke(self._payload())

        post.assert_called_once()
        get.assert_called_once()
        sleep.assert_not_called()

    def test_create_error_surfaces_response_body(self):
        bad = self._job(
            {}, status=400, text='{"error":{"message":"Invalid model name"}}'
        )
        with patch(self.POST, return_value=bad):
            adapter = OpenAIVideoAsyncAdapter(self._endpoint(), LOGGER)
            with pytest.raises(Exception, match="Invalid model name") as exc:
                adapter.invoke(self._payload())
        assert "400" in str(exc.value)

    def test_empty_content_raises(self):
        created = self._job({"id": "vE", "status": "processing"})
        done = self._job({"id": "vE", "status": "completed"})
        empty = self._video(b"", ctype="video/mp4")
        with (
            patch(self.POST, return_value=created),
            patch(self.GET, side_effect=[done, empty]),
            patch(self.SLEEP),
        ):
            adapter = OpenAIVideoAsyncAdapter(self._endpoint(), LOGGER)
            with pytest.raises(ValueError, match="empty/undecodable"):
                adapter.invoke(self._payload())

    def test_default_timeout_when_unset(self):
        adapter = OpenAIVideoAsyncAdapter(self._endpoint(timeout=None), LOGGER)
        assert adapter.timeout == 900.0
        # A single request stays capped even when the overall budget is huge.
        assert adapter._remaining_timeout(1e18) == 120.0
        # ...and once the budget is spent, per-request resolution raises.
        with pytest.raises(TimeoutError, match="budget"):
            adapter._remaining_timeout(0.0)

    def test_optional_route_suffix(self):
        adapter = OpenAIVideoAsyncAdapter(
            self._endpoint(url="https://host", route="/v1/videos"), LOGGER
        )
        assert adapter.create_url == "https://host/v1/videos"


# ---------------------------------------------------------------------------
# OpenAIImageEditAdapter
# ---------------------------------------------------------------------------
class TestOpenAIImageEditAdapter:
    def _make(self, fake_response):
        with patch("generation.adapters.openai_images_edit.OpenAI") as mock_openai:
            client = MagicMock()
            client.images.edit.return_value = fake_response
            mock_openai.return_value = client
            adapter = OpenAIImageEditAdapter(_endpoint(), LOGGER)
        return adapter, client

    def test_edit_request_and_b64_decode(self):
        resp = SimpleNamespace(
            _request_id="img1", data=[SimpleNamespace(b64_json=PNG_B64, url=None)]
        )
        adapter, client = self._make(resp)
        result = adapter.invoke(_image_payload(prompt="make it night"))

        _, kwargs = client.images.edit.call_args
        # Image uploaded as a (filename, bytes, mime) multipart tuple.
        assert kwargs["image"] == ("image.png", PNG_BYTES, "image/png")
        assert kwargs["prompt"] == "make it night"
        assert kwargs["model"] == "some/model"
        # b64_json requested by default so we get bytes back.
        assert kwargs["response_format"] == "b64_json"
        # Non-native knobs ride through extra_body, not as top-level kwargs.
        assert kwargs["extra_body"] == {"seed": 0, "num_inference_steps": 30}
        assert "seed" not in kwargs
        assert result.media_bytes == PNG_BYTES
        assert result.request_id == "img1"

    def test_native_params_promoted_not_in_extra_body(self):
        resp = SimpleNamespace(data=[SimpleNamespace(b64_json=PNG_B64, url=None)])
        adapter, client = self._make(resp)
        payload = _image_payload()
        payload["params"] = {"size": "1024x1024", "seed": 7}
        adapter.invoke(payload)
        _, kwargs = client.images.edit.call_args
        assert kwargs["size"] == "1024x1024"  # native kwarg
        assert kwargs["extra_body"] == {"seed": 7}  # leftover only

    def test_url_response_is_fetched(self):
        # Same-origin URL (endpoint default is http://localhost:9999) -> fetched.
        resp = SimpleNamespace(
            data=[SimpleNamespace(b64_json=None, url="http://localhost:9999/gen/y.png")]
        )
        adapter, _ = self._make(resp)
        with patch("generation.adapters.openai_images_edit.requests.get") as get:
            get.return_value = SimpleNamespace(
                content=PNG_BYTES, raise_for_status=lambda: None
            )
            result = adapter.invoke(_image_payload())
        assert result.media_bytes == PNG_BYTES

    def _make_with(self, ep, fake_response):
        with patch("generation.adapters.openai_images_edit.OpenAI") as mock_openai:
            client = MagicMock()
            client.images.edit.return_value = fake_response
            mock_openai.return_value = client
            return OpenAIImageEditAdapter(ep, LOGGER)

    def test_offorigin_url_rejected(self):
        # An off-origin download URL is refused (SSRF guard); the b64_json path is
        # the intended one, so this is a rare fallback.
        ep = _endpoint(url="http://localhost:9999/v1")
        resp = SimpleNamespace(
            data=[SimpleNamespace(b64_json=None, url="http://evil.example/x.png")]
        )
        adapter = self._make_with(ep, resp)
        with pytest.raises(ValueError, match="off-origin"):
            adapter.invoke(_image_payload())

    def test_sameorigin_url_gets_bearer(self, monkeypatch):
        monkeypatch.setenv("EDIT_KEY", "sk-secret")
        ep = _endpoint(url="http://localhost:9999/v1", api_key_env="EDIT_KEY")
        resp = SimpleNamespace(
            data=[SimpleNamespace(b64_json=None, url="http://localhost:9999/gen/x.png")]
        )
        adapter = self._make_with(ep, resp)
        with patch("generation.adapters.openai_images_edit.requests.get") as get:
            get.return_value = SimpleNamespace(
                content=PNG_BYTES, raise_for_status=lambda: None
            )
            adapter.invoke(_image_payload())
        assert get.call_args[1]["headers"]["Authorization"] == "Bearer sk-secret"

    def test_missing_image_raises(self):
        resp = SimpleNamespace(data=[SimpleNamespace(b64_json=PNG_B64, url=None)])
        adapter, _ = self._make(resp)
        with pytest.raises(ValueError, match="input image is required"):
            adapter.invoke(_image_payload(with_media=False))

    def test_empty_response_raises(self):
        resp = SimpleNamespace(_request_id="img2", data=[])
        adapter, _ = self._make(resp)
        with pytest.raises(ValueError, match="no image in response"):
            adapter.invoke(_image_payload())


# ---------------------------------------------------------------------------
# PassthroughAdapter
# ---------------------------------------------------------------------------
class TestPassthroughAdapter:
    def test_echo_media(self):
        adapter = PassthroughAdapter(_endpoint(), LOGGER)
        assert adapter.invoke(_image_payload()).media_bytes == PNG_BYTES

    def test_echo_text(self):
        adapter = PassthroughAdapter(_endpoint(), LOGGER)
        result = adapter.invoke({"prompt": "hello", "media": {}, "params": {}})
        assert result.text == "hello"
        assert isinstance(result, Result)


# ---------------------------------------------------------------------------
# Request recording (provenance for output metadata)
# ---------------------------------------------------------------------------
# A payload big enough to trip the elision threshold, and pure base64 once
# encoded — the shape a real video/image input takes on the wire.
BIG_MP4 = b"\x00" * 2000
BIG_MP4_B64 = base64.b64encode(BIG_MP4).decode()

PROSE = (
    "game playing with bad crappy graphics, cartoonish frames, old outdated "
    "games, fake lighting, raw basic textures, primitive geometry, pixelated "
    "poor CG quality, subtitles, unrealistic, and then some more words to push "
    "this negative prompt comfortably past the elision threshold so the test "
    "proves length alone never elides prose. " * 3
)


class TestRedactMedia:
    def test_long_base64_elided_with_digest(self):
        out = redact_media({"video": BIG_MP4_B64})["video"]
        assert out["__elided__"] == "base64"
        assert out["chars"] == len(BIG_MP4_B64)
        assert out["sha256"] == hashlib.sha256(BIG_MP4_B64.encode()).hexdigest()

    def test_prose_survives_however_long(self):
        assert len(PROSE) > 512  # would be elided if length were the only test
        assert redact_media({"negative_prompt": PROSE})["negative_prompt"] == PROSE

    def test_short_values_untouched(self):
        body = {"seed": 42, "guidance": 7.0, "edge": {"control_weight": 1.0}}
        assert redact_media(body) == body

    def test_data_url_elided_and_mime_kept(self):
        out = redact_media(f"data:image/png;base64,{BIG_MP4_B64}")
        assert out["__elided__"] == "data-url"
        assert out["mime"] == "image/png"
        assert out["chars"] == len(BIG_MP4_B64)

    def test_raw_bytes_elided(self):
        out = redact_media({"files": {"video": ("in.mp4", BIG_MP4, "video/mp4")}})
        descriptor = out["files"]["video"][1]
        assert descriptor["__elided__"] == "bytes"
        assert descriptor["bytes"] == len(BIG_MP4)
        assert descriptor["sha256"] == hashlib.sha256(BIG_MP4).hexdigest()


class TestAdapterRecordsRequest:
    def test_nothing_recorded_before_any_call(self):
        adapter = NIMAdapter(_endpoint(), LOGGER)
        assert adapter.recorded_requests == []
        assert adapter.recorded_call_count == 0

    def test_nim_records_body_with_video_elided(self, monkeypatch):
        monkeypatch.setenv("BUILD_NVIDIA_API_KEY", "super-secret")
        fake_resp = MagicMock()
        fake_resp.json.return_value = {"b64_video": base64.b64encode(b"OUT").decode()}
        payload = {
            "prompt": "make it snowy",
            "media": {"rgb": {"bytes": BIG_MP4, "mime": "video/mp4", "path": "in.mp4"}},
            "params": {"seed": 7, "guidance": 3.0, "edge": {"control_weight": 1.0}},
        }
        with patch("generation.adapters.nim.requests.post", return_value=fake_resp):
            adapter = NIMAdapter(
                _endpoint(role="video_transfer", api_key_env="BUILD_NVIDIA_API_KEY"),
                LOGGER,
            )
            adapter.invoke(payload)

        recorded = adapter.recorded_requests[-1]
        # Knobs that explain the output are kept verbatim...
        assert recorded["prompt"] == "make it snowy"
        assert recorded["seed"] == 7
        assert recorded["edge"] == {"control_weight": 1.0}
        # ...the megabytes of input video are not.
        assert recorded["video"]["__elided__"] == "base64"
        assert (
            recorded["video"]["sha256"]
            == hashlib.sha256(BIG_MP4_B64.encode()).hexdigest()
        )
        # The key rides in a header, so it can never reach the record.
        assert "super-secret" not in json.dumps(recorded)

    def test_seed_recorded_is_the_one_the_retry_loop_set(self, tmp_path):
        """cli.py re-rolls generator.seed between attempts; the record must show
        the seed actually sent, which config alone cannot reconstruct."""
        from generation.executors import BaseExecutor
        from generation.utils import generate_with_retries

        source = tmp_path / "in.mp4"
        source.write_bytes(BIG_MP4)
        fake_resp = MagicMock()
        fake_resp.json.return_value = {"b64_video": base64.b64encode(b"OUT").decode()}
        with patch("generation.adapters.nim.requests.post", return_value=fake_resp):
            adapter = NIMAdapter(_endpoint(role="video_transfer"), LOGGER)
            executor = BaseExecutor(
                adapter=adapter,
                params={"seed": 42},
                inputs_spec={"rgb": {"required": True, "kind": "video"}},
                logger=LOGGER,
            )
            executor.seed = 43  # what the retry loop does on attempt 1
            success, _, _ = generate_with_retries(
                executor,
                "p",
                str(source),
                {},
                str(tmp_path / "out.mp4"),
                0,
                0.0,
                LOGGER,
            )
        assert success
        assert adapter.recorded_requests[-1]["seed"] == 43

    def test_failed_request_does_not_overwrite_a_good_record(self):
        """A later attempt that errors must not relabel the retained output."""
        ok = MagicMock()
        ok.json.return_value = {"b64_video": base64.b64encode(b"OUT").decode()}
        bad = MagicMock()
        bad.ok = False
        bad.status_code = 500
        bad.text = "boom"
        payload = {
            "prompt": "p",
            "media": {"rgb": {"bytes": BIG_MP4, "mime": "video/mp4", "path": "i.mp4"}},
            "params": {"seed": 42},
        }
        with patch("generation.adapters.nim.requests.post", return_value=ok):
            adapter = NIMAdapter(_endpoint(role="video_transfer"), LOGGER)
            adapter.invoke(payload)
        payload["params"] = {"seed": 43}
        with patch("generation.adapters.nim.requests.post", return_value=bad):
            with pytest.raises(Exception):
                adapter.invoke(payload)
        # Still the seed that actually produced output.
        assert adapter.recorded_requests[-1]["seed"] == 42

    def test_secret_shaped_params_are_masked(self):
        """augmentation.parameters is extra=allow, so a token can reach the body
        — it must not reach the metadata file on disk."""
        fake_resp = MagicMock()
        fake_resp.json.return_value = {"b64_video": base64.b64encode(b"O").decode()}
        payload = {
            "prompt": "p",
            "media": {"rgb": {"bytes": BIG_MP4, "mime": "video/mp4", "path": "i.mp4"}},
            "params": {"seed": 1, "access_token": "tok-abc123", "api_key": "sk-xyz"},
        }
        with patch("generation.adapters.nim.requests.post", return_value=fake_resp):
            adapter = NIMAdapter(_endpoint(role="video_transfer"), LOGGER)
            adapter.invoke(payload)
        recorded = json.dumps(adapter.recorded_requests[-1])
        assert "tok-abc123" not in recorded and "sk-xyz" not in recorded
        assert adapter.recorded_requests[-1]["seed"] == 1

    def test_unserializable_value_cannot_break_metadata(self):
        """A provenance field must never abort the metadata write."""
        from pathlib import Path

        record = redact_media({"path": Path("/tmp/x"), "when": object()})
        json.dumps(record)  # must not raise

    def test_every_registered_adapter_records_its_request(self):
        """Enumerates ADAPTERS so a newly registered contract cannot silently
        ship without provenance — the gap that shipped for chat/passthrough."""
        from generation.factory import ADAPTERS

        uninstrumented = [
            name
            for name, cls in ADAPTERS.items()
            if "self.record_request(" not in inspect.getsource(cls)
        ]
        assert uninstrumented == [], (
            f"adapters missing record_request(): {uninstrumented}"
        )

    def test_request_url_is_the_concrete_route(self):
        """Provenance's url field must mean the same thing for every contract."""
        nim = NIMAdapter(_endpoint(role="video_transfer"), LOGGER)
        assert nim.request_url.endswith("/v1/infer")
        with patch("generation.adapters.openai_chat.OpenAI"):
            chat = OpenAIChatAdapter(_endpoint(role="vlm"), LOGGER)
        # The SDK appends this route to base_url; provenance must not report a
        # bare base URL for one contract and a full route for another.
        assert chat.request_url.endswith("/v1/chat/completions")

    def test_url_credentials_never_reach_the_record(self):
        """metadata.json ships with the dataset; basic-auth-in-URL is legal."""
        ep = _endpoint(role="video_transfer", url="https://user:s3cr3t@host/v1")
        nim = NIMAdapter(ep, LOGGER)
        assert "s3cr3t" not in nim.request_url
        assert "user" not in nim.request_url
        assert nim.request_url == "https://host/v1/infer"

    def test_chat_path_records_prompts(self):
        """Captioners/verifiers call adapter.chat(), not invoke(); recording only
        in invoke() left every VLM/LLM stage with no prompts in metadata."""
        resp = SimpleNamespace(id="c1", choices=[])
        with patch("generation.adapters.openai_chat.OpenAI") as mock_openai:
            client = MagicMock()
            client.chat.completions.create.return_value = resp
            mock_openai.return_value = client
            adapter = OpenAIChatAdapter(_endpoint(role="vlm"), LOGGER)
            adapter.chat(
                messages=[
                    {"role": "system", "content": "You describe scenes."},
                    {"role": "user", "content": "Describe this."},
                ],
                temperature=0.3,
            )
        recorded = adapter.recorded_requests[-1]
        assert [m["role"] for m in recorded["messages"]] == ["system", "user"]
        assert recorded["messages"][0]["content"] == "You describe scenes."
        assert recorded["temperature"] == 0.3
        assert recorded["model"] == "some/model"

    def test_chat_path_elides_image_payloads(self):
        """A VLM user message carries the frame as a data URL — the prompts must
        survive, the megabytes must not."""
        resp = SimpleNamespace(id="c2", choices=[])
        big_url = f"data:image/png;base64,{BIG_MP4_B64}"
        with patch("generation.adapters.openai_chat.OpenAI") as mock_openai:
            client = MagicMock()
            client.chat.completions.create.return_value = resp
            mock_openai.return_value = client
            adapter = OpenAIChatAdapter(_endpoint(role="vlm"), LOGGER)
            adapter.chat(
                messages=[
                    {"role": "system", "content": "Answer with one letter."},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "What colour is the top?"},
                            {"type": "image_url", "image_url": {"url": big_url}},
                        ],
                    },
                ]
            )
        content = adapter.recorded_requests[-1]["messages"][1]["content"]
        assert content[0]["text"] == "What colour is the top?"
        assert content[1]["image_url"]["url"]["__elided__"] == "data-url"
        assert BIG_MP4_B64 not in json.dumps(adapter.recorded_requests[-1])

    def test_token_budget_is_not_mistaken_for_a_credential(self):
        """_is_secret_field matches substrings, so "max_tokens" contains "token";
        masking it would destroy a generation knob and protect nothing."""
        record = redact_media(
            {
                "max_tokens": 2048,
                "max_new_tokens": 512,
                "temperature": 0.2,
                "access_token": "tok-abc",
                "api_key": "sk-xyz",
            }
        )
        assert record["max_tokens"] == 2048
        assert record["max_new_tokens"] == 512
        assert record["temperature"] == 0.2
        assert record["access_token"] == "***"
        assert record["api_key"] == "***"

    def test_multiple_calls_are_all_recorded(self):
        resp = SimpleNamespace(id="c", choices=[])
        with patch("generation.adapters.openai_chat.OpenAI") as mock_openai:
            client = MagicMock()
            client.chat.completions.create.return_value = resp
            mock_openai.return_value = client
            adapter = OpenAIChatAdapter(_endpoint(role="llm"), LOGGER)
            for i in range(3):
                adapter.chat(messages=[{"role": "user", "content": f"q{i}"}])
        assert adapter.recorded_call_count == 3
        contents = [r["messages"][0]["content"] for r in adapter.recorded_requests]
        assert contents == ["q0", "q1", "q2"]
        assert adapter.recorded_requests[-1]["messages"][0]["content"] == "q2"

    def test_recording_is_capped_but_the_count_is_not(self):
        """A capped list must never read as the complete set."""
        from generation.adapters.base import MAX_RECORDED_REQUESTS

        resp = SimpleNamespace(id="c", choices=[])
        overshoot = MAX_RECORDED_REQUESTS + 5
        with patch("generation.adapters.openai_chat.OpenAI") as mock_openai:
            client = MagicMock()
            client.chat.completions.create.return_value = resp
            mock_openai.return_value = client
            adapter = OpenAIChatAdapter(_endpoint(role="llm"), LOGGER)
            for i in range(overshoot):
                adapter.chat(messages=[{"role": "user", "content": f"q{i}"}])
        assert len(adapter.recorded_requests) == MAX_RECORDED_REQUESTS
        assert adapter.recorded_call_count == overshoot

    def test_reset_clears_between_samples(self):
        """Adapters outlive a sample; without a reset, sample 2 would report
        sample 1's calls as its own."""
        resp = SimpleNamespace(id="c", choices=[])
        with patch("generation.adapters.openai_chat.OpenAI") as mock_openai:
            client = MagicMock()
            client.chat.completions.create.return_value = resp
            mock_openai.return_value = client
            adapter = OpenAIChatAdapter(_endpoint(role="llm"), LOGGER)
            adapter.chat(messages=[{"role": "user", "content": "sample-1"}])
            adapter.reset_requests()
            adapter.chat(messages=[{"role": "user", "content": "sample-2"}])
        assert adapter.recorded_call_count == 1
        assert [r["messages"][0]["content"] for r in adapter.recorded_requests] == [
            "sample-2"
        ]

    def test_generation_knobs_containing_token_are_not_masked(self):
        """vLLM sampling params trip a substring secret test; masking them would
        destroy the very knobs this record exists to preserve."""
        record = redact_media(
            {
                "max_tokens": 2048,
                "stop_token_ids": [151645],
                "skip_special_tokens": True,
                "truncate_prompt_tokens": 4096,
                "min_tokens": 8,
            }
        )
        assert record == {
            "max_tokens": 2048,
            "stop_token_ids": [151645],
            "skip_special_tokens": True,
            "truncate_prompt_tokens": 4096,
            "min_tokens": 8,
        }

    def test_credential_shaped_names_are_still_masked(self):
        record = redact_media(
            {
                "api_key": "sk-1",
                "access_token": "t",
                "hf_token": "t",
                "client_secret": "s",
                "password": "p",
                "authorization": "Bearer x",
                "api_key_env": "MY_KEY_VAR",
            }
        )
        assert record["api_key"] == "***"
        assert record["access_token"] == "***"
        assert record["hf_token"] == "***"
        assert record["client_secret"] == "***"
        assert record["password"] == "***"
        assert record["authorization"] == "***"
        # *_env names an env var, not the secret.
        assert record["api_key_env"] == "MY_KEY_VAR"

    def test_non_string_keys_do_not_raise(self):
        """An OmegaConf mapping can carry int keys; provenance must survive it."""
        assert redact_media({1: "x", "ok": 2}) == {"1": "x", "ok": 2}

    def test_recording_never_fails_a_successful_call(self):
        """record_request runs after the endpoint returned; an exception here
        would discard inference that already succeeded."""

        class Exploding:
            # redact_media's fallback for unknown types is repr(); make even
            # that fail, so the redaction itself raises.
            def __repr__(self):
                raise RuntimeError("boom")

        adapter = NIMAdapter(_endpoint(role="video_transfer"), LOGGER)
        adapter.record_request({"params": Exploding()})  # must not raise
        assert adapter.recorded_requests == []
        assert adapter.recorded_call_count == 0

    def test_call_with_no_artifact_is_not_recorded(self):
        """A 200 carrying no artifact still fails; it must not appear as a call
        that produced the retained output."""
        empty = MagicMock()
        empty.json.return_value = {}
        payload = {
            "prompt": "p",
            "media": {"rgb": {"bytes": BIG_MP4, "mime": "video/mp4", "path": "i.mp4"}},
            "params": {"seed": 1},
        }
        with patch("generation.adapters.nim.requests.post", return_value=empty):
            adapter = NIMAdapter(_endpoint(role="video_transfer"), LOGGER)
            with pytest.raises(ValueError, match="no artifact"):
                adapter.invoke(payload)
        assert adapter.recorded_requests == []
        assert adapter.recorded_call_count == 0
