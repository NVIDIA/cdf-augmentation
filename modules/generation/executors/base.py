# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Executor base class for the BYOM multi-endpoint generation layer.

An *executor* is transport-agnostic: it validates inputs, assembles the
normalized request payload, hands it to an :class:`~generation.adapters.BaseAdapter`,
and writes the adapter's :class:`~generation.adapters.Result` to ``output_path``.
It owns no wire protocol — that lives in the adapter.

:class:`BaseExecutor` subclasses :class:`~generation.base.BaseGenerator` so the
existing CLI invariant keeps working unchanged: ``cli.py`` and
``generation.utils.generate_with_retries`` call
``generator.execute(prompt, rgb_path, control_inputs, output_media_path)``
(expecting a ``(bool, path|None)`` tuple and catching exceptions themselves) and
mutate ``generator.seed``.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, Optional, Tuple

import multistorageclient as msc

from generation.base import BaseGenerator

# Values that mean "no input was supplied" for an input slot.
_ABSENT_STRINGS = frozenset({"none", "null", ""})

# Extension -> MIME sniffing for media reads. Anything unrecognized is treated
# as a PNG image, matching the prior ImageEditGenerator default.
_MIME_BY_EXT: Dict[str, str] = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
    ".mp4": "video/mp4",
}

# Input kinds that carry binary media (read into the payload's ``media`` map).
_MEDIA_KINDS = frozenset({"image", "video", "control"})

# Roles whose endpoints must return media. A text-only reply from one of these
# is the server explaining why it produced nothing -- e.g. vllm-omni answers an
# empty diffusion output with HTTP 200 and "Image generation completed but no
# images were produced." -- so writing it would leave a text file at the
# .png/.mp4 output path and report success.
MEDIA_OUTPUT_ROLES = frozenset(
    {"image_edit", "image2video", "video_transfer", "video_predict"}
)


def _is_present(value: Any) -> bool:
    """True if ``value`` is a real input (not missing / None / a null-ish string)."""
    if value is None:
        return False
    if isinstance(value, str) and value.strip().lower() in _ABSENT_STRINGS:
        return False
    return True


def seed_holder(params: Dict[str, Any]) -> Dict[str, Any]:
    """Return the dict that holds (or should hold) the ``seed`` knob.

    With the pass-through chat contract the *config* controls the param shape, so
    ``seed`` may sit at the top level or one level inside a model-specific
    envelope dict (e.g. ``extra_body`` / ``extra_param``). The retry loop must
    find and re-roll it regardless of envelope name. Resolution: a top-level
    ``seed`` wins; otherwise the first immediate child dict that already has a
    ``seed`` key; otherwise the top level (where a generated seed is created).
    """
    if "seed" in params:
        return params
    for value in params.values():
        if isinstance(value, dict) and "seed" in value:
            return value
    return params


class BaseExecutor(BaseGenerator):
    """Transport-agnostic generation executor.

    Args:
        adapter: The API-contract adapter that performs the actual call.
        params: Mutable generation knobs (``seed`` is written here via the
            :attr:`seed` property; copied so the caller's dict is untouched).
        inputs_spec: ``{name: {"required": bool, "kind": str}}`` describing the
            inputs this role accepts.
        logger: Logger instance.
    """

    def __init__(
        self,
        *,
        adapter,
        params: dict,
        inputs_spec: dict,
        logger: logging.Logger,
    ):
        super().__init__(logger)
        self.adapter = adapter
        self.params = dict(params)
        self.inputs_spec = inputs_spec

    # -- seed -------------------------------------------------------------
    @property
    def seed(self) -> Optional[int]:
        """Current generation seed (located wherever the config placed it)."""
        return seed_holder(self.params).get("seed")

    @seed.setter
    def seed(self, value) -> None:
        seed_holder(self.params)["seed"] = int(value)

    # -- BaseGenerator API ------------------------------------------------
    def execute(
        self,
        prompt: str,
        input_media_path: str,
        control_inputs: Dict[str, str],
        output_path: str,
    ) -> Tuple[bool, Optional[str]]:
        """Run one generation request. Never raises — failures become ``(False, None)``."""
        inputs = {
            "prompt": prompt,
            "rgb": input_media_path,
            **(control_inputs or {}),
        }
        try:
            self.run(inputs, output_path)
            return True, output_path
        except Exception as e:  # noqa: BLE001 - executor must never raise
            self.logger.error(
                f"Generation failed -> output_path={output_path} error={e}",
                exc_info=True,
            )
            return False, None

    def get_required_inputs(self) -> Dict[str, bool]:
        """Derive ``{name: required}`` from :attr:`inputs_spec`."""
        return {
            name: bool(spec.get("required")) for name, spec in self.inputs_spec.items()
        }

    # -- pipeline ---------------------------------------------------------
    def run(self, inputs: Dict[str, Any], output_path: str) -> bool:
        """Validate, build, invoke, write. Raises on any failure."""
        self._validate(inputs)
        payload = self.build_request(inputs)
        result = self.adapter.invoke(payload)
        self._write(result, output_path)
        self.logger.info(
            f"Generation request complete request_id={result.request_id} "
            f"-> output_path={output_path}"
        )
        return True

    def _validate(self, inputs: Dict[str, Any]) -> None:
        """Raise ``ValueError`` if a required input is missing/null."""
        for name, spec in self.inputs_spec.items():
            if spec.get("required") and not _is_present(inputs.get(name)):
                raise ValueError(f"input '{name}' is required")

    def build_request(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Assemble the normalized payload (flat default).

        Reads bytes for every present media input and drops ``None`` params.
        """
        prompt = inputs.get("prompt")
        media: Dict[str, Any] = {}
        for name, spec in self.inputs_spec.items():
            if spec.get("kind") not in _MEDIA_KINDS:
                continue
            value = inputs.get(name)
            if not _is_present(value):
                continue
            item = self._read_media(value)
            # Tag the input kind so adapters can tell the primary input from a
            # control modality (edge/depth/seg/vis) without re-deriving it.
            item["kind"] = spec.get("kind")
            media[name] = item
        params = {k: v for k, v in self.params.items() if v is not None}
        return {"prompt": prompt, "media": media, "params": params}

    def _read_media(self, path: str) -> Dict[str, Any]:
        """Read ``path`` into a ``{bytes, mime, path}`` media dict."""
        with msc.open(path, "rb") as f:
            data = f.read()
        ext = os.path.splitext(path)[1].lower()
        mime = _MIME_BY_EXT.get(ext, "image/png")
        return {"bytes": data, "mime": mime, "path": path}

    def _write(self, result, output_path: str) -> None:
        """Write the adapter result (media bytes XOR text) to ``output_path``."""
        if result.media_bytes:
            with msc.open(output_path, "wb") as f:
                f.write(result.media_bytes)
        elif result.text is not None:
            role = getattr(getattr(self.adapter, "endpoint", None), "role", None)
            if role in MEDIA_OUTPUT_ROLES:
                raise ValueError(
                    f"{role} endpoint returned text instead of media "
                    f"(request_id={result.request_id}): {result.text!r}"
                )
            with msc.open(output_path, "w") as f:
                f.write(result.text)
        else:
            raise ValueError("adapter returned neither media nor text")
