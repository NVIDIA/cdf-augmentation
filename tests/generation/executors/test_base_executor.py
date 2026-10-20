# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for BaseExecutor (no live endpoints, no real adapters)."""

import logging
from types import SimpleNamespace

import pytest

from generation.adapters import BaseAdapter, Result
from generation.executors import BaseExecutor

LOGGER = logging.getLogger("test")

IMAGE_EDIT_SPEC = {
    "rgb": {"required": True, "kind": "image"},
    "prompt": {"required": True, "kind": "text"},
}


class FakeAdapter(BaseAdapter):
    """Records the payload it was handed and returns a canned Result."""

    def __init__(self, result=None, role="image_edit"):
        # Bypass BaseAdapter.__init__ wiring (no real endpoint needed here).
        self.endpoint = SimpleNamespace(id="fake", role=role)
        self.logger = LOGGER
        self.result = result if result is not None else Result(media_bytes=b"out")
        self.last_payload = None

    def invoke(self, payload):
        self.last_payload = payload
        return self.result


def _make_executor(adapter=None, params=None, inputs_spec=None):
    return BaseExecutor(
        adapter=adapter or FakeAdapter(),
        params=params if params is not None else {"seed": 0, "guidance_scale": 4.0},
        inputs_spec=inputs_spec or IMAGE_EDIT_SPEC,
        logger=LOGGER,
    )


def _write_input(tmp_path, name="in.png", data=b"\x89PNG\r\nfake"):
    p = tmp_path / name
    p.write_bytes(data)
    return str(p)


class TestExecute:
    def test_success_writes_bytes(self, tmp_path):
        adapter = FakeAdapter(Result(media_bytes=b"generated", request_id="r1"))
        ex = _make_executor(adapter=adapter)
        rgb = _write_input(tmp_path)
        out = str(tmp_path / "out.png")

        ok, path = ex.execute("edit this", rgb, {}, out)

        assert ok is True
        assert path == out
        assert (tmp_path / "out.png").read_bytes() == b"generated"

    def test_text_result_writes_text(self, tmp_path):
        adapter = FakeAdapter(Result(text="hi"), role="llm")
        ex = _make_executor(
            adapter=adapter, inputs_spec={"prompt": {"required": True, "kind": "text"}}
        )
        out = str(tmp_path / "out.txt")

        ok, path = ex.execute("say hi", None, {}, out)

        assert ok is True
        assert (tmp_path / "out.txt").read_text() == "hi"

    @pytest.mark.parametrize(
        "role", ["image_edit", "image2video", "video_transfer", "video_predict"]
    )
    def test_text_result_for_media_role_fails(self, tmp_path, role):
        # vllm-omni's reply when the diffusion engine yields no output: HTTP 200
        # with this string as content. It must fail generation, not be written
        # to the media path as a "successful" output.
        adapter = FakeAdapter(
            Result(text="Image generation completed but no images were produced."),
            role=role,
        )
        ex = _make_executor(adapter=adapter)
        rgb = _write_input(tmp_path)
        out = str(tmp_path / "out.png")

        ok, path = ex.execute("edit", rgb, {}, out)

        assert ok is False
        assert path is None
        assert not (tmp_path / "out.png").exists()

    def test_text_result_for_media_role_error_names_server_text(self, tmp_path):
        adapter = FakeAdapter(Result(text="no images were produced", request_id="r9"))
        ex = _make_executor(adapter=adapter)
        rgb = _write_input(tmp_path)

        with pytest.raises(ValueError, match="image_edit endpoint returned text") as e:
            ex.run({"prompt": "edit", "rgb": rgb}, str(tmp_path / "out.png"))

        assert "r9" in str(e.value)
        assert "no images were produced" in str(e.value)

    def test_empty_media_bytes_fails(self, tmp_path):
        adapter = FakeAdapter(Result(media_bytes=b""))
        ex = _make_executor(adapter=adapter)
        rgb = _write_input(tmp_path)
        out = str(tmp_path / "out.png")

        ok, path = ex.execute("edit", rgb, {}, out)

        assert ok is False
        assert path is None
        assert not (tmp_path / "out.png").exists()

    def test_missing_required_input_is_swallowed(self, tmp_path):
        ex = _make_executor()
        out = str(tmp_path / "out.png")

        # rgb is None -> _validate raises -> execute returns (False, None).
        ok, path = ex.execute("edit this", None, {}, out)

        assert ok is False
        assert path is None
        assert not (tmp_path / "out.png").exists()

    def test_null_string_rgb_treated_as_absent(self, tmp_path):
        ex = _make_executor()
        out = str(tmp_path / "out.png")

        ok, path = ex.execute("edit", "none", {}, out)

        assert ok is False
        assert path is None

    def test_neither_media_nor_text_fails(self, tmp_path):
        adapter = FakeAdapter(Result())  # no media, no text
        ex = _make_executor(adapter=adapter)
        rgb = _write_input(tmp_path)
        out = str(tmp_path / "out.png")

        ok, path = ex.execute("edit", rgb, {}, out)

        assert ok is False
        assert path is None


class TestValidate:
    def test_validate_raises_on_missing_required(self):
        ex = _make_executor()
        with pytest.raises(ValueError, match="rgb"):
            ex._validate({"prompt": "p", "rgb": None})

    def test_validate_passes_when_present(self, tmp_path):
        ex = _make_executor()
        ex._validate({"prompt": "p", "rgb": _write_input(tmp_path)})  # no raise


class TestSeed:
    def test_seed_getter(self):
        ex = _make_executor(params={"seed": 7})
        assert ex.seed == 7

    def test_seed_setter_writes_params_as_int(self):
        ex = _make_executor(params={"seed": 0})
        ex.seed = "42"
        assert ex.params["seed"] == 42
        assert isinstance(ex.params["seed"], int)

    def test_params_are_copied(self):
        original = {"seed": 1}
        ex = _make_executor(params=original)
        ex.seed = 99
        assert original["seed"] == 1  # caller dict untouched

    def test_seed_located_inside_envelope(self):
        # Qwen-style nested config: seed lives under `extra_body`. The getter/setter
        # find and re-roll it there, NOT at the top level.
        ex = _make_executor(
            params={"extra_body": {"num_inference_steps": 30, "seed": 7}}
        )
        assert ex.seed == 7
        ex.seed = 42
        assert ex.params["extra_body"]["seed"] == 42
        assert "seed" not in ex.params  # not duplicated at top level

    def test_top_level_seed_wins_over_envelope(self):
        ex = _make_executor(params={"seed": 1, "extra_body": {"seed": 9}})
        assert ex.seed == 1
        ex.seed = 5
        assert ex.params["seed"] == 5
        assert ex.params["extra_body"]["seed"] == 9  # untouched

    def test_seed_holder_defaults_top_level_when_absent(self):
        ex = _make_executor(params={"extra_body": {"num_inference_steps": 30}})
        assert ex.seed is None
        ex.seed = 8
        assert ex.params["seed"] == 8  # created at top level


class TestBuildRequest:
    def test_only_present_media_included(self, tmp_path):
        spec = {
            "rgb": {"required": True, "kind": "video"},
            "prompt": {"required": True, "kind": "text"},
            "edge": {"required": False, "kind": "video"},
            "depth": {"required": False, "kind": "video"},
        }
        ex = _make_executor(inputs_spec=spec, params={"seed": 1})
        rgb = _write_input(tmp_path, "rgb.mp4", b"vid")
        edge = _write_input(tmp_path, "edge.mp4", b"edge")

        payload = ex.build_request(
            {"prompt": "p", "rgb": rgb, "edge": edge, "depth": None}
        )

        assert payload["prompt"] == "p"
        assert set(payload["media"]) == {"rgb", "edge"}
        assert payload["media"]["rgb"]["bytes"] == b"vid"
        assert payload["media"]["rgb"]["mime"] == "video/mp4"
        assert payload["media"]["rgb"]["path"] == rgb
        # Each media item is tagged with its declared kind so adapters can tell
        # the primary input from a control modality.
        assert payload["media"]["rgb"]["kind"] == "video"
        assert payload["media"]["edge"]["kind"] == "video"

    def test_none_params_dropped(self, tmp_path):
        ex = _make_executor(params={"seed": 0, "guidance_scale": None, "steps": 30})
        rgb = _write_input(tmp_path)
        payload = ex.build_request({"prompt": "p", "rgb": rgb})
        assert payload["params"] == {"seed": 0, "steps": 30}

    def test_text_kind_not_in_media(self, tmp_path):
        ex = _make_executor()
        rgb = _write_input(tmp_path)
        payload = ex.build_request({"prompt": "p", "rgb": rgb})
        assert "prompt" not in payload["media"]
        assert "rgb" in payload["media"]

    def test_mime_sniffing(self, tmp_path):
        ex = _make_executor()
        jpg = _write_input(tmp_path, "x.jpg")
        payload = ex.build_request({"prompt": "p", "rgb": jpg})
        assert payload["media"]["rgb"]["mime"] == "image/jpeg"


class TestGetRequiredInputs:
    def test_matches_inputs_spec(self):
        spec = {
            "rgb": {"required": True, "kind": "video"},
            "prompt": {"required": True, "kind": "text"},
            "edge": {"required": False, "kind": "video"},
        }
        ex = _make_executor(inputs_spec=spec)
        assert ex.get_required_inputs() == {"rgb": True, "prompt": True, "edge": False}
