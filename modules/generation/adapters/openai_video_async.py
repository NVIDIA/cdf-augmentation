# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""OpenAI **asynchronous** video adapter — contract ``openai.video.async``.

The async create -> poll -> download-content contract. This is the Sora-style
OpenAI Videos API, as exposed by hosted gateways such as
``https://inference-api.nvidia.com/videos`` (LiteLLM -> Vertex AI, serving
``gcp/google/veo-3.1-generate-001``). It differs from :class:`OpenAIVideoSyncAdapter`
(``openai.video.sync``, Cosmos3's *synchronous* ``/v1/videos/sync`` that streams the
MP4 back in one shot) in its **control flow**: the create call returns a *job*,
not the video, so the adapter must poll the job to completion and then fetch the
rendered bytes from a separate content route::

    POST {url}                         (multipart/form-data)
      file: input_reference=<first frame image>   # optional (text->video too)
      file: control_reference=<WSM video>         # optional (Cosmos3 transfer)
      form: model, prompt, seconds, size, ...      # knobs forwarded verbatim
      form: control_type=wsm                       # with control_reference
      -> {"id": "video_...", "status": "processing", ...}

    GET  {url}/{id}                    -> {"status": "completed" | "failed" | ...}
    GET  {url}/{id}/content            (Accept: video/mp4) -> raw MP4 bytes

As with the sync adapter and :class:`NIMAdapter`, the body is not structured by
the adapter: the config's ``parameters`` become form fields verbatim (a new Veo
knob is a config entry, not code). The adapter only supplies the things YAML
can't — the ``input_reference`` file part, the create/poll/download orchestration,
and decoding the final raw-bytes response. From the executor's point of view this
is still a single blocking ``invoke`` (the poll loop lives inside), so the CLI's
retry/timeout tiers wrap it exactly as they wrap the sync contract.
"""

from __future__ import annotations

import time
from typing import Any, Dict, Optional

import requests

from .base import (
    BaseAdapter,
    Result,
    control_media,
    primary_media,
    reject_control_media,
    strip_url_credentials,
)
from .openai_video_sync import (
    DEFAULT_CONTROL_FILE_FIELD,
    DEFAULT_CONTROL_TYPE_FIELD,
    VIDEO_DEFAULT_TIMEOUT,
    _basename,
    _decode_response,
    _form_value,
)

DEFAULT_FILE_FIELD = "input_reference"
DEFAULT_ACCEPT = "video/mp4"

# How long to wait between job polls, and a sane per-HTTP-request cap so a single
# stuck poll/download can't consume the whole wall budget. The overall
# create->poll->download is bounded by ``self.timeout`` (the endpoint timeout).
DEFAULT_POLL_INTERVAL = 8.0
PER_REQUEST_TIMEOUT_CAP = 120.0

# Terminal job states. LiteLLM/Vertex report "completed"; accept the OpenAI
# "succeeded" spelling too. Anything in the failure set raises.
_SUCCESS_STATES = frozenset({"completed", "succeeded", "success"})
_FAILURE_STATES = frozenset({"failed", "error", "errored", "cancelled", "canceled"})


class OpenAIVideoAsyncAdapter(BaseAdapter):
    """Async video contract: multipart create -> poll job -> download MP4."""

    def __init__(self, endpoint, logger):
        super().__init__(endpoint, logger)
        # Video generation is slow (Veo ~1.5-2 min); prefer the video default
        # over the short generic one unless the endpoint declares a timeout.
        if getattr(endpoint, "timeout", None) is None:
            self.timeout = VIDEO_DEFAULT_TIMEOUT
        self.file_field = getattr(endpoint, "input_field", None) or DEFAULT_FILE_FIELD
        self.accept = getattr(endpoint, "accept", None) or DEFAULT_ACCEPT
        self.poll_interval = float(
            getattr(endpoint, "poll_interval", None) or DEFAULT_POLL_INTERVAL
        )
        # The endpoint URL *is* the create route (e.g. ``.../videos``); an
        # optional ``route`` suffix supports a ``/v1/videos``-style deployment.
        route = getattr(endpoint, "route", None) or ""
        self.create_url = f"{self.url}{route}" if route else self.url
        self.request_url = strip_url_credentials(self.create_url)
        self.logger.info(
            "OpenAIVideoAsyncAdapter ready: %s model=%s (poll %.0fs, budget %.0fs)",
            self.create_url,
            self.model,
            self.poll_interval,
            self.timeout,
        )

    def invoke(self, payload: Dict[str, Any]) -> Result:
        # One budget spans the whole create -> poll -> download flow so the
        # advertised endpoint timeout actually bounds it (not just the poll loop).
        deadline = time.monotonic() + self.timeout
        job_id = self._create(payload, deadline)
        self._poll(job_id, deadline)
        return self._download(job_id, deadline)

    # -- create -------------------------------------------------------------

    def _create(self, payload: Dict[str, Any], deadline: float) -> str:
        """POST the multipart create request; return the job id."""
        controls = control_media(payload)
        if controls and not (
            self.role == "video_transfer"
            and len(controls) == 1
            and controls[0][0] == "wsm"
        ):
            reject_control_media(payload, "OpenAIVideoAsyncAdapter")
        wsm = controls[0][1] if controls else None

        data: Dict[str, str] = {}
        prompt = payload.get("prompt")
        if prompt:
            data["prompt"] = prompt
        for key, value in (payload.get("params") or {}).items():
            if value is None:
                continue
            data[key] = _form_value(value)
        # Unlike the single-model Cosmos3 server, this gateway *requires* model
        # (model=None -> HTTP 400), so always send it when configured.
        if self.model:
            data.setdefault("model", self.model)

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
            # Text-to-video: no file part. Send the form fields as multipart parts
            # anyway (``data=`` alone is application/x-www-form-urlencoded, which
            # the video gateways reject) by routing them through ``files``.
            post_kwargs = {"files": [(k, (None, v)) for k, v in data.items()]}

        resp = requests.post(
            self.create_url,
            headers=self._headers(),
            timeout=self._remaining_timeout(deadline),
            **post_kwargs,
        )
        self._raise_for_status(resp, "create", self.create_url)
        self.record_request(post_kwargs)
        job = self._json(resp, "create")
        job_id = job.get("id")
        if not job_id:
            raise ValueError(
                f"OpenAIVideoAsyncAdapter: create response from {self.create_url} "
                f"has no job id: {str(job)[:500]}"
            )
        self.logger.info(
            "Video job created id=%s status=%s usage=%s",
            job_id,
            job.get("status"),
            job.get("usage"),
        )
        return job_id

    # -- poll ---------------------------------------------------------------

    def _poll(self, job_id: str, deadline: float) -> None:
        """Poll ``GET {create_url}/{id}`` until the job reaches a terminal state."""
        poll_url = f"{self.create_url}/{job_id}"
        headers = self._headers()
        last_status: Optional[str] = None
        while True:
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"video job {job_id} did not complete within {self.timeout:.0f}s "
                    f"(last status={last_status!r})"
                )
            try:
                resp = requests.get(
                    poll_url, headers=headers, timeout=self._remaining_timeout(deadline)
                )
            except (requests.Timeout, requests.ConnectionError) as exc:
                # Status GET is safe to repeat. Keep polling the submitted job;
                # restarting invoke() would POST a second expensive video job.
                self.logger.warning(
                    "Video job %s status poll %s: %s; %.0fs remain before deadline",
                    job_id,
                    type(exc).__name__,
                    exc,
                    max(0.0, deadline - time.monotonic()),
                )
            else:
                self._raise_for_status(resp, "poll", poll_url)
                job = self._json(resp, "poll")
                status = (job.get("status") or "").lower()
                if status != last_status:
                    self.logger.info("Video job %s status=%s", job_id, status or "?")
                    last_status = status
                if status in _SUCCESS_STATES:
                    return
                if status in _FAILURE_STATES or job.get("error"):
                    raise RuntimeError(
                        f"video job {job_id} failed (status={status!r}): "
                        f"{str(job.get('error'))[:1000]}"
                    )
            # Don't let the fixed poll interval overshoot the remaining budget.
            time.sleep(min(self.poll_interval, max(0.0, deadline - time.monotonic())))

    # -- download -----------------------------------------------------------

    def _download(self, job_id: str, deadline: float) -> Result:
        """Fetch the rendered MP4 from ``GET {create_url}/{id}/content``."""
        content_url = f"{self.create_url}/{job_id}/content"
        headers = {**self._headers(), "Accept": self.accept}
        resp = requests.get(
            content_url, headers=headers, timeout=self._remaining_timeout(deadline)
        )
        self._raise_for_status(resp, "download", content_url)
        media_bytes = _decode_response(resp)
        if not media_bytes:
            ctype = resp.headers.get("content-type", "?")
            raise ValueError(
                f"OpenAIVideoAsyncAdapter: empty/undecodable content for job {job_id} "
                f"from {content_url} (content-type={ctype}, {len(resp.content)} bytes)"
            )
        return Result(media_bytes=media_bytes, request_id=job_id)

    # -- helpers ------------------------------------------------------------

    def _remaining_timeout(self, deadline: float) -> float:
        """Per-request timeout: the time left in the overall budget, capped so a
        single HTTP call can't hog it. Raises once the budget is spent."""
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError(
                f"video job exceeded its {self.timeout:.0f}s budget before completing"
            )
        return min(remaining, PER_REQUEST_TIMEOUT_CAP)

    def _headers(self) -> Dict[str, str]:
        headers: Dict[str, str] = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _raise_for_status(self, resp: requests.Response, phase: str, url: str) -> None:
        if resp.ok:
            return
        # Keep the response body: it carries the real reason (a 400 bad model, a
        # 422 field error, a safety rejection) that raise_for_status() discards.
        detail = (resp.text or "").strip()
        raise requests.HTTPError(
            f"video job {phase} {url} returned HTTP {resp.status_code}: "
            f"{detail[:2000] if detail else '<empty response body>'}",
            response=resp,
        )

    def _json(self, resp: requests.Response, phase: str) -> Dict[str, Any]:
        try:
            data = resp.json()
        except ValueError as exc:
            raise ValueError(
                f"OpenAIVideoAsyncAdapter: {phase} response from {resp.url} "
                f"was not JSON: {(resp.text or '')[:500]}"
            ) from exc
        if not isinstance(data, dict):
            raise ValueError(
                f"OpenAIVideoAsyncAdapter: {phase} response from {resp.url} "
                f"was not a JSON object: {str(data)[:500]}"
            )
        return data
