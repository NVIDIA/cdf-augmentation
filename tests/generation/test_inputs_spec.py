# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the per-role inputs_spec mapping."""

from generation.inputs_spec import inputs_spec_for


def _required(spec):
    return {name: meta["required"] for name, meta in spec.items()}


def test_image_edit():
    spec = inputs_spec_for("image_edit")
    assert _required(spec) == {"rgb": True, "prompt": True}
    assert spec["rgb"]["kind"] == "image"


def test_video_transfer():
    spec = inputs_spec_for("video_transfer")
    assert _required(spec) == {
        "rgb": True,
        "prompt": True,
        "edge": False,
        "depth": False,
        "seg": False,
        "vis": False,
        "wsm": False,
    }
    assert spec["rgb"]["kind"] == "video"
    # Control modalities are tagged "control" so adapters can distinguish them
    # from the primary input (serialize in NIM, reject in single-input contracts).
    assert spec["edge"]["kind"] == "control"
    assert spec["depth"]["kind"] == "control"
    assert spec["wsm"]["kind"] == "control"


def test_video_predict_default_rgb_required():
    spec = inputs_spec_for("video_predict")
    assert spec["rgb"]["required"] is True
    assert spec["prompt"]["required"] is True


def test_video_predict_text2world_rgb_optional():
    spec = inputs_spec_for("video_predict", inference_type="text2world")
    assert spec["rgb"]["required"] is False
    assert spec["prompt"]["required"] is True


def test_video_predict_other_inference_type_rgb_required():
    spec = inputs_spec_for("video_predict", inference_type="image2world")
    assert spec["rgb"]["required"] is True


def test_vlm():
    spec = inputs_spec_for("vlm")
    assert _required(spec) == {"prompt": True, "rgb": False}


def test_llm():
    spec = inputs_spec_for("llm")
    assert _required(spec) == {"prompt": True}
    assert "rgb" not in spec


def test_unknown_role_defaults():
    spec = inputs_spec_for("totally-unknown")
    assert _required(spec) == {"prompt": True, "rgb": False}


def test_unknown_role_returns_independent_copy():
    a = inputs_spec_for("unknown-a")
    a["prompt"]["required"] = False
    b = inputs_spec_for("unknown-b")
    assert b["prompt"]["required"] is True
