# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Per-role input specifications for the BYOM generation layer.

``inputs_spec_for`` returns the ``{name: {"required": bool, "kind": str}}`` map
describing the inputs a role accepts. This mirrors the input-requirement matrix
the legacy ``BaseGenerator.get_required_inputs`` implementations encoded, but in
a single declarative place that both the executor (validation / payload build)
and the factory can share.
"""

from __future__ import annotations

from typing import Optional

# Fallback spec for unknown roles: a text prompt plus an optional image, which
# covers the generic "VLM-like" shape without raising.
_DEFAULT_SPEC = {
    "prompt": {"required": True, "kind": "text"},
    "rgb": {"required": False, "kind": "image"},
}


def inputs_spec_for(role: str, inference_type: Optional[str] = None) -> dict:
    """Return the inputs_spec for ``role``.

    Args:
        role: Endpoint role (``image_edit``, ``video_transfer``,
            ``video_predict``, ``vlm``, ``llm``, ...).
        inference_type: Only consulted for ``video_predict`` — when
            ``"text2world"`` the ``rgb`` input is optional rather than required.
    """
    if role == "image_edit":
        return {
            "rgb": {"required": True, "kind": "image"},
            "prompt": {"required": True, "kind": "text"},
        }

    if role == "video_transfer":
        # rgb is the primary input; the remaining entries are control-modality
        # videos (kind "control") so adapters can serialize them per contract.
        return {
            "rgb": {"required": True, "kind": "video"},
            "prompt": {"required": True, "kind": "text"},
            "edge": {"required": False, "kind": "control"},
            "depth": {"required": False, "kind": "control"},
            "seg": {"required": False, "kind": "control"},
            "vis": {"required": False, "kind": "control"},
            "wsm": {"required": False, "kind": "control"},
        }

    if role == "video_predict":
        rgb_required = inference_type != "text2world"
        return {
            "rgb": {"required": rgb_required, "kind": "video"},
            "prompt": {"required": True, "kind": "text"},
        }

    if role == "image2video":
        # Image-to-video (Cosmos3): a single first-frame image + a motion prompt.
        return {
            "rgb": {"required": True, "kind": "image"},
            "prompt": {"required": True, "kind": "text"},
        }

    if role == "vlm":
        return {
            "prompt": {"required": True, "kind": "text"},
            "rgb": {"required": False, "kind": "image"},
        }

    if role == "llm":
        return {
            "prompt": {"required": True, "kind": "text"},
        }

    # Unknown role: return a fresh copy of the generic default.
    return {name: dict(spec) for name, spec in _DEFAULT_SPEC.items()}
