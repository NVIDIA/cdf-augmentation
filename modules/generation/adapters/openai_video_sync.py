# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""OpenAI **synchronous** video adapter — contract ``openai.video.sync``.

The synchronous video contract (``POST /v1/videos/sync`` → MP4 bytes in one
response). For the *asynchronous* create→poll→download contract (Veo / Sora),
see :class:`OpenAIVideoAsyncAdapter` (``openai.video.async``).

The contract for image-to-video / text-to-video servers such as
``nvidia/Cosmos3-Super-Image2Video`` (vLLM-served, OpenAI ``/v1/videos`` family)
differs from the JSON ``/v1/infer`` NIM in **both** directions:

  * **Request** is ``multipart/form-data``, not JSON. The first-frame image is a
    *file* part (``input_reference``); every generation knob is a *form field*
    (``prompt``, ``size``, ``num_frames``, ...); and nested structures
    (``extra_params``) are sent as a **JSON string** form field. Cosmos3 WSM
    transfer adds a second file part (``control_reference``) and the form field
    ``control_type=wsm``.
  * **Response** is the **raw MP4 bytes** (``Accept: video/mp4``), not a JSON
    envelope with base64.

As with :class:`NIMAdapter`, the adapter imposes no body structure beyond what it
must place itself: the config's ``parameters`` are forwarded verbatim as form
fields (so a new model's own knob — ``flow_shift``, ``fps``, ... — is a config
entry, not code), and any dict/list value is JSON-encoded to the string the
multipart contract requires. The only things the adapter supplies are the input
image (the ``input_reference`` file part) and decoding the response (raw bytes,
with a JSON-envelope fallback).

Reference::

    POST {url}/v1/videos/sync          (multipart/form-data, Accept: video/mp4)
      file: input_reference=<first frame image>
      file: control_reference=<optional WSM control video>
      form: prompt, negative_prompt, size, num_frames, fps,
            control_type='wsm',
            num_inference_steps, guidance_scale, flow_shift,
            extra_params='{"guardrails": true, ...}'   # JSON string
      -> raw MP4 bytes
"""

from __future__ import annotations

import json
import time
from typing import Any, Dict, Optional
from urllib.parse import quote

import requests

from .base import (
    BaseAdapter,
    Result,
    control_media,
    primary_media,
    strip_url_credentials,
)
from .nim import _is_nvcf_url, _nvcf_poll_seconds, _nvcf_status_base_url

# Video generation blocks until the MP4 is ready; default well above the generic
# request timeout (a 189-frame / 50-step clip is minutes, not seconds).
VIDEO_DEFAULT_TIMEOUT = 900.0

# Default multipart route and field/header names for the OpenAI ``/v1/videos``
# family. Each is overridable on the endpoint for a sibling server that renames
# them, without a code change.
DEFAULT_ROUTE = "/v1/videos/sync"
DEFAULT_FILE_FIELD = "input_reference"
DEFAULT_CONTROL_FILE_FIELD = "control_reference"
DEFAULT_CONTROL_TYPE_FIELD = "control_type"
DEFAULT_ACCEPT = "video/mp4"


class OpenAIVideoSyncAdapter(BaseAdapter):
    """Synchronous video contract: multipart ``POST /v1/videos/sync`` -> MP4 bytes."""

    def __init__(self, endpoint, logger):
        super().__init__(endpoint, logger)
        # Video generation is slow; prefer the video default over the short
        # generic one unless the endpoint declares an explicit timeout.
        if getattr(endpoint, "timeout", None) is None:
            self.timeout = VIDEO_DEFAULT_TIMEOUT
        # Defensive getattr so a duck-typed endpoint may override any of these
        # without requiring a schema field.
        self.route = getattr(endpoint, "route", None) or DEFAULT_ROUTE
        self.file_field = getattr(endpoint, "input_field", None) or DEFAULT_FILE_FIELD
        self.accept = getattr(endpoint, "accept", None) or DEFAULT_ACCEPT
        # Tolerate an OpenAI-style base URL ending in ``/v1`` so we don't build a
        # doubled ``/v1/v1`` when the route already carries the ``/v1`` prefix.
        base = self.url
        if self.route.startswith("/v1") and base.endswith("/v1"):
            base = base[:-3]
        self.gen_url = f"{base}{self.route}"
        self.is_nvcf = _is_nvcf_url(self.gen_url)
        self.nvcf_poll_seconds = _nvcf_poll_seconds() if self.is_nvcf else None
        self.request_url = strip_url_credentials(self.gen_url)
        self.logger.info(
            "OpenAIVideoSyncAdapter ready: %s model=%s (timeout %.0fs)",
            self.gen_url,
            self.model,
            self.timeout,
        )

    def invoke(self, payload: Dict[str, Any]) -> Result:
        params = dict(payload.get("params") or {})
        controls = dict(control_media(payload))
        wsm = controls.pop("wsm", None)

        # Preserve the existing path-based behavior for control types other than
        # WSM. WSM uses vllm-omni's request-scoped multipart upload below, so the
        # client and generation server do not need a shared filesystem.
        # Fold each remaining control input into
        # ``extra_params.<modality> = {"control_path": <path>}``, without
        # overriding a modality the config already set (e.g. ``edge: true`` to
        # auto-compute).
        if controls:
            extra = dict(params.get("extra_params") or {})
            for name, item in controls.items():
                if name not in extra and item.get("path"):
                    extra[name] = {"control_path": item["path"]}
            params["extra_params"] = extra

        # Build the form fields from prompt + pass-through params. Multipart
        # fields are strings, so dict/list values (e.g. extra_params) are
        # JSON-encoded and bools are lowercased to match JSON/server expectation.
        data: Dict[str, str] = {}
        prompt = payload.get("prompt")
        if prompt:
            data["prompt"] = prompt
        for key, value in params.items():
            if value is None:
                continue
            data[key] = _form_value(value)
        # Only send a model field if one is configured; the proven Cosmos3
        # contract omits it (single-model server). A sibling server that needs
        # it gets it via endpoint.model without code changes.
        if self.model:
            data.setdefault("model", self.model)

        # Upload the primary input (rgb) and, when present, the pre-rendered WSM
        # control. vllm-omni persists the control for this request and injects its
        # temporary path into extra_params.wsm on the server side.
        files = {}
        media = primary_media(payload)
        if media is not None:
            filename = _basename(media.get("path")) or "input.png"
            mime = media.get("mime", "image/png")
            files[self.file_field] = (filename, media["bytes"], mime)
        if wsm is not None:
            filename = _basename(wsm.get("path")) or "wsm.mp4"
            mime = wsm.get("mime", "video/mp4")
            files[DEFAULT_CONTROL_FILE_FIELD] = (filename, wsm["bytes"], mime)
            data[DEFAULT_CONTROL_TYPE_FIELD] = "wsm"

        if files:
            post_kwargs = {"data": data, "files": files}
        else:
            # Text-to-video: no file part. Force multipart anyway (``data=`` alone
            # is application/x-www-form-urlencoded, which the /v1/videos/sync
            # contract rejects) by routing the form fields through ``files``.
            post_kwargs = {"files": [(k, (None, v)) for k, v in data.items()]}

        headers = {"Accept": self.accept}
        if self.is_nvcf:
            headers["NVCF-POLL-SECONDS"] = str(
                min(self.nvcf_poll_seconds, max(1, int(self.timeout) - 1))
            )
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        deadline = time.monotonic() + self.timeout
        resp = requests.post(
            self.gen_url,
            headers=headers,
            timeout=self.timeout,
            **post_kwargs,
        )
        nvcf_request_id = None
        if self.is_nvcf and resp.status_code == 202:
            resp, nvcf_request_id = self._poll_nvcf(resp, headers, deadline)
        if not resp.ok:
            # raise_for_status() drops the response body, but that body holds the
            # real reason (a 422 field error, a guardrail rejection, ...).
            # Surface it so the pipeline log shows the cause directly.
            detail = (resp.text or "").strip()
            raise requests.HTTPError(
                f"video NIM {self.gen_url} returned HTTP {resp.status_code}: "
                f"{detail[:2000] if detail else '<empty response body>'}",
                response=resp,
            )

        media_bytes = _decode_response(resp)
        if not media_bytes:
            ctype = resp.headers.get("content-type", "?")
            raise ValueError(
                f"OpenAIVideoSyncAdapter: empty/undecodable response from {self.gen_url} "
                f"(content-type={ctype}, {len(resp.content)} bytes)"
            )
        # Recorded only now: a 200 with an undecodable body still fails the call.
        self.record_request(post_kwargs)
        request_id = nvcf_request_id or resp.headers.get("x-request-id")
        return Result(media_bytes=media_bytes, request_id=request_id)

    def _poll_nvcf(self, response, headers, deadline: float):
        request_id = response.headers.get("NVCF-REQID")
        if not request_id:
            raise ValueError("NVCF returned HTTP 202 without an NVCF-REQID")

        status_url = (
            _nvcf_status_base_url().rstrip("/") + "/" + quote(str(request_id), safe="")
        )
        self.logger.info("NVCF request accepted request_id=%s; polling", request_id)
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise requests.Timeout(
                    f"NVCF request {request_id} did not complete within "
                    f"{self.timeout:.0f}s"
                )
            poll_headers = dict(headers)
            poll_headers["NVCF-POLL-SECONDS"] = str(
                min(self.nvcf_poll_seconds, max(1, int(remaining) - 1))
            )
            response = requests.get(status_url, headers=poll_headers, timeout=remaining)
            if response.status_code != 202:
                self.logger.info(
                    "NVCF request finished request_id=%s status=%s",
                    request_id,
                    response.status_code,
                )
                return response, str(request_id)
            time.sleep(1)


def _form_value(value: Any) -> str:
    """Coerce a param value to a multipart form-field string."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (dict, list)):
        return json.dumps(value)
    return str(value)


def _basename(path: Optional[str]) -> Optional[str]:
    """Return the final path component (the upload filename), or ``None``."""
    if not path:
        return None
    return path.replace("\\", "/").rstrip("/").split("/")[-1] or None


def _decode_response(resp: requests.Response) -> Optional[bytes]:
    """Extract the video bytes from a ``/v1/videos/sync`` response.

    The proven Cosmos3 contract returns the **raw MP4 bytes** with a
    ``video/*`` (or ``octet-stream``) content type. As a fallback for sibling
    servers that wrap the result, a JSON body is parsed for a base64 field
    (``b64_video`` / ``video`` / ``b64_json`` / ``data[0].b64_json``).
    """
    ctype = (resp.headers.get("content-type") or "").lower()
    if "application/json" in ctype:
        try:
            data = resp.json()
        except ValueError:
            return resp.content or None
        return _decode_json_video(data)
    # Raw bytes (video/mp4, application/octet-stream, or unlabeled).
    return resp.content or None


def _decode_json_video(data: Any) -> Optional[bytes]:
    """Pull base64 video bytes out of a JSON envelope (fallback path)."""
    import base64

    if not isinstance(data, dict):
        return None
    # OpenAI images-style nesting: {"data": [{"b64_json": ...}]}.
    items = data.get("data")
    if isinstance(items, list) and items and isinstance(items[0], dict):
        b64 = items[0].get("b64_json") or items[0].get("base64")
        if isinstance(b64, str) and b64:
            return base64.b64decode(b64)
    for key in ("b64_video", "video", "b64_json", "base64"):
        b64 = data.get(key)
        if isinstance(b64, str) and b64:
            return base64.b64decode(b64)
    return None
