# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the unified pipeline configuration schema."""

from pathlib import Path

import pytest
from omegaconf import OmegaConf
from pydantic import ValidationError

from aug_utils.schema import (
    ModelNameEnum,
    PipelineConfig,
)

CONFIGS_DIR = Path(__file__).resolve().parents[2] / "configs"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_yaml(path: str) -> dict:
    config = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    assert isinstance(config, dict)
    return config


def _ep(id, role, url, model="test", adapter=None) -> dict:
    """Build an endpoint-registry list entry."""
    entry = {"id": id, "role": role, "url": url, "model": model}
    if adapter is not None:
        entry["adapter"] = adapter
    return entry


def _minimal_config(**overrides) -> dict:
    """Return a minimal valid config dict for cosmos-transfer2.5."""
    base = {
        "data": [
            {
                "inputs": {"rgb": "/tmp/test.mp4"},
                "output": {
                    "video": "/tmp/out.mp4",
                    "caption": "/tmp/cap.txt",
                    "metadata": "/tmp/meta.json",
                },
            }
        ],
        "endpoints": [
            _ep(
                "cosmos_transfer",
                "video_transfer",
                "http://localhost:30002/",
                adapter="nim",
            ),
        ],
        "augmentation": {
            "model": {"name": "cosmos-transfer2.5"},
        },
    }
    base.update(overrides)
    return base


def _minimal_config_with_cosmos(*, enabled: bool = True) -> dict:
    """Return a minimal config wired to the external Cosmos Evaluator."""
    config = _minimal_config()
    config["data"][0]["inputs"]["rgb"] = "s3://bucket/run/input.mp4"
    config["data"][0]["output"]["video"] = "s3://bucket/run/out.mp4"
    config["endpoints"].append(
        _ep(
            "cosmos-evaluator",
            "evaluator",
            "http://cosmos-evaluator:8000",
            adapter="cosmos.evaluator.arbitrator",
        )
    )
    evaluator = {
        "enabled": enabled,
        "endpoint_id": "cosmos-evaluator",
        "checks": [
            {
                "name": "future.vendor.arbitrary_checker",
                "gate": True,
                "metadata_key": "future_check",
            }
        ],
    }
    if enabled:
        evaluator["output_storage_prefix"] = "runs/test-001/"
    config["evaluators"] = [{"cosmos_evaluator": evaluator}]
    return config


def _minimal_config_with_captioning() -> dict:
    """Return a minimal config with VLM captioning."""
    config = _minimal_config()
    config["endpoints"].append(
        _ep("vlm", "vlm", "http://localhost:9001/v1", "test-vlm")
    )
    config["captioning"] = {
        "vlm": {"system_prompt": "test", "user_prompt": "test"},
    }
    return config


def _minimal_config_image_edit() -> dict:
    """Return a minimal valid config for image-edit."""
    return {
        "data": [
            {
                "inputs": {"rgb": "/tmp/test.png"},
                "output": {
                    "video": "/tmp/out.png",
                    "caption": "/tmp/cap.txt",
                    "metadata": "/tmp/meta.json",
                },
            }
        ],
        "endpoints": [
            _ep("image_edit", "image_edit", "http://localhost:8080/v1", "test-edit"),
        ],
        "augmentation": {
            "model": {"name": "image-edit"},
            "parameters": {"num_inference_steps": 50, "guidance_scale": 1.0},
        },
    }


def _minimal_config_predict() -> dict:
    """Return a minimal valid config for cosmos-predict."""
    return {
        "data": [
            {
                "inputs": {"rgb": "/tmp/test.mp4"},
                "output": {
                    "video": "/tmp/out.mp4",
                    "caption": "/tmp/cap.txt",
                    "metadata": "/tmp/meta.json",
                },
            }
        ],
        "endpoints": [
            _ep(
                "cosmos_predict",
                "video_predict",
                "http://localhost:30003/",
                "predict",
                adapter="nim",
            ),
        ],
        "augmentation": {
            "model": {"name": "cosmos-predict"},
            "parameters": {
                "inference_type": "video2world",
                "num_output_frames": 121,
                "num_steps": 35,
                "enable_autoregressive": True,
                "chunk_size": 49,
                "chunk_overlap": 17,
                "seed": 42,
                "guidance": 3.0,
            },
        },
    }


def _minimal_config_predict_text2world() -> dict:
    """Return a minimal valid config for cosmos-predict text2world (no rgb input)."""
    return {
        "data": [
            {
                "inputs": {},
                "output": {
                    "video": "/tmp/out.mp4",
                    "caption": "/tmp/cap.txt",
                    "metadata": "/tmp/meta.json",
                },
            }
        ],
        "endpoints": [
            _ep(
                "cosmos_predict",
                "video_predict",
                "http://localhost:30003/",
                "predict",
                adapter="nim",
            ),
        ],
        "augmentation": {
            "model": {"name": "cosmos-predict"},
            "parameters": {
                "inference_type": "text2world",
                "num_output_frames": 93,
                "num_steps": 35,
                "seed": 42,
                "guidance": 3.0,
            },
        },
    }


# ---------------------------------------------------------------------------
# Positive tests — valid configs load successfully
# ---------------------------------------------------------------------------


class TestValidConfigs:
    """Configs that should pass validation."""

    def test_minimal_cosmos_transfer(self):
        pc = PipelineConfig(**_minimal_config())
        assert pc.augmentation.model.name == ModelNameEnum.COSMOS_TRANSFER

    def test_cosmos_transfer_accepts_wsm_control(self):
        config = _minimal_config()
        config["data"][0]["inputs"]["controls"] = {"wsm": "/tmp/WSM_RGB.mp4"}
        pc = PipelineConfig(**config)
        assert pc.data[0].inputs.controls.wsm == "/tmp/WSM_RGB.mp4"

    def test_minimal_image_edit(self):
        pc = PipelineConfig(**_minimal_config_image_edit())
        assert pc.augmentation.model.name == ModelNameEnum.IMAGE_EDIT

    def test_minimal_cosmos_predict(self):
        pc = PipelineConfig(**_minimal_config_predict())
        assert pc.augmentation.model.name == ModelNameEnum.COSMOS_PREDICT
        assert pc.augmentation.parameters.inference_type.value == "video2world"
        assert pc.augmentation.parameters.num_output_frames == 121
        assert pc.augmentation.parameters.enable_autoregressive is True
        assert pc.augmentation.parameters.chunk_size == 49

    def test_cosmos_predict_text2world(self):
        pc = PipelineConfig(**_minimal_config_predict_text2world())
        assert pc.augmentation.model.name == ModelNameEnum.COSMOS_PREDICT
        assert pc.augmentation.parameters.inference_type.value == "text2world"
        assert pc.data[0].inputs.rgb is None

    def test_cosmos_predict_image2world(self):
        config = _minimal_config_predict()
        config["augmentation"]["parameters"]["inference_type"] = "image2world"
        pc = PipelineConfig(**config)
        assert pc.augmentation.parameters.inference_type.value == "image2world"

    def test_cosmos_predict_invalid_inference_type(self):
        config = _minimal_config_predict()
        config["augmentation"]["parameters"]["inference_type"] = "invalid"
        with pytest.raises(ValidationError):
            PipelineConfig(**config)

    def test_cosmos_predict_with_resolution(self):
        config = _minimal_config_predict()
        config["augmentation"]["parameters"]["resolution"] = "none"
        pc = PipelineConfig(**config)
        assert pc.augmentation.parameters.resolution == "none"

    def test_with_vlm_captioning(self):
        pc = PipelineConfig(**_minimal_config_with_captioning())
        assert pc.captioning.vlm is not None
        assert pc.captioning.llm is None

    def test_with_vlm_llm_captioning(self):
        config = _minimal_config_with_captioning()
        config["endpoints"].append(_ep("llm", "llm", "http://localhost:8001/v1", "llm"))
        config["captioning"]["llm"] = {
            "system_prompt": "test",
            "variables": {"weather": ["rain"]},
        }
        pc = PipelineConfig(**config)
        assert pc.captioning.vlm is not None
        assert pc.captioning.llm is not None
        assert pc.captioning.llm.variables == {"weather": ["rain"]}

    def test_with_evaluators(self):
        config = _minimal_config()
        # attribute_verification needs an 'llm' (question generation) and a 'vlm'
        # (its inline verifier) transport to resolve.
        config["endpoints"].append(_ep("llm", "llm", "http://localhost:9002/v1", "llm"))
        config["endpoints"].append(_ep("vlm", "vlm", "http://localhost:9001/v1", "vlm"))
        config["pipeline"] = {"retry": 1}
        config["evaluators"] = [
            {"hallucination_check": {"enabled": True, "threshold": 0.7}},
            {
                "attribute_verification": {
                    "enabled": True,
                    "vlm_verification": {"system_prompt": "test"},
                }
            },
        ]
        pc = PipelineConfig(**config)
        assert len(pc.evaluators) == 2
        assert pc.pipeline.retry == 1
        assert pc.evaluators[1].attribute_verification.enabled is True
        assert (
            pc.evaluators[1].attribute_verification.vlm_verification.system_prompt
            == "test"
        )

    def test_standalone_vlm_verification_is_rejected(self):
        config = _minimal_config()
        config["endpoints"].append(_ep("vlm", "vlm", "http://localhost:9001/v1", "vlm"))
        config["evaluators"] = [
            {"vlm_verification": {"system_prompt": "standalone test"}},
        ]
        with pytest.raises(
            ValidationError, match="standalone vlm_verification is not supported"
        ):
            PipelineConfig(**config)

    def test_attribute_verification_without_inline_vlm(self):
        config = _minimal_config()
        # An enabled attribute_verification needs both an 'llm' (question
        # generation) and a 'vlm' (the default verifier) transport.
        config["endpoints"].append(_ep("llm", "llm", "http://localhost:9002/v1", "llm"))
        config["endpoints"].append(_ep("vlm", "vlm", "http://localhost:9001/v1", "vlm"))
        config["evaluators"] = [
            {"attribute_verification": {"enabled": True}},
        ]
        pc = PipelineConfig(**config)
        assert pc.evaluators[0].attribute_verification.vlm_verification is None

    def test_with_modalities(self):
        config = _minimal_config()
        config["augmentation"]["modalities"] = {
            "edge": 0.4,
            "depth": 0.4,
            "seg": 0.3,
            "positive_prompt": "cinematic",
            "negative_prompt": "cartoon",
        }
        pc = PipelineConfig(**config)
        assert pc.augmentation.modalities.edge == 0.4
        assert pc.augmentation.modalities.seg == 0.3
        assert pc.augmentation.modalities.positive_prompt == "cinematic"

    def test_with_pipeline_evaluation(self):
        config = _minimal_config()
        config["pipeline"] = {
            "retry": 5,
            "evaluation": {"strict": False, "retain_failures": False},
            "logging": {"enabled": True, "level": "DEBUG"},
        }
        pc = PipelineConfig(**config)
        assert pc.pipeline.retry == 5
        assert pc.pipeline.evaluation.strict is False
        assert pc.pipeline.logging.level == "DEBUG"

    @pytest.mark.parametrize(
        "config_file",
        sorted(CONFIGS_DIR.rglob("config_*.yaml")),
        ids=lambda p: str(p.relative_to(CONFIGS_DIR)),
    )
    def test_load_all_config_files(self, config_file):
        """Every config_*.yaml under configs/ (incl. cookbook/) must validate."""
        config = _load_yaml(str(config_file))
        PipelineConfig(**config)


# ---------------------------------------------------------------------------
# External Cosmos Evaluator schema
# ---------------------------------------------------------------------------


class TestCosmosEvaluatorSchema:
    """Validate the external Arbitrator config and cross-section contract."""

    def test_valid_arbitrary_checker_and_opaque_inputs(self):
        config = _minimal_config_with_cosmos()
        cosmos = config["evaluators"][0]["cosmos_evaluator"]
        cosmos.update(
            {
                "poll_interval_seconds": 2.5,
                "stream_key": "  camera_front_wide_120fov  ",
                "merge_captioning_selections": True,
                "inputs": {
                    "original_video_urls": {
                        "camera_front_wide_120fov": "s3://bucket/original.mp4"
                    },
                    "world_model_video_urls": {
                        "camera_front_wide_120fov": "https://storage.test/world.mp4"
                    },
                    "rds_hq_url": "gs://bucket/rds.zip",
                    "future_arbitrator_input": {"nested": [1, "two"]},
                },
                "config": {"future_checker_setting": {"threshold": 0.42}},
            }
        )

        pc = PipelineConfig(**config)
        parsed = pc.evaluators[0].cosmos_evaluator
        assert parsed.enabled is True
        assert parsed.endpoint_id == "cosmos-evaluator"
        assert parsed.poll_interval_seconds == 2.5
        assert parsed.stream_key == "camera_front_wide_120fov"
        assert parsed.output_storage_prefix == "runs/test-001/"
        assert parsed.merge_captioning_selections is True
        assert parsed.checks[0].name == "future.vendor.arbitrary_checker"
        assert parsed.inputs["future_arbitrator_input"] == {"nested": [1, "two"]}
        assert parsed.config == {"future_checker_setting": {"threshold": 0.42}}

    def test_disabled_entry_needs_no_endpoint_prefix_or_remote_output(self):
        config = _minimal_config_with_cosmos(enabled=False)
        config["endpoints"] = [
            ep for ep in config["endpoints"] if ep["role"] != "evaluator"
        ]
        config["data"][0]["output"]["video"] = "/tmp/out.mp4"
        config["evaluators"][0]["cosmos_evaluator"]["inputs"] = {
            "rds_hq_url": "/tmp/rds.zip"
        }

        pc = PipelineConfig(**config)
        cosmos = pc.evaluators[0].cosmos_evaluator
        assert cosmos.enabled is False
        assert cosmos.output_storage_prefix is None

    def test_one_enabled_and_multiple_disabled_entries_are_valid(self):
        config = _minimal_config_with_cosmos()
        disabled = {
            "cosmos_evaluator": {
                "enabled": False,
                "endpoint_id": "unused",
                "checks": [{"name": "disabled.check", "gate": False}],
            }
        }
        config["evaluators"].extend([disabled, disabled])

        pc = PipelineConfig(**config)
        assert len(pc.evaluators) == 3

    def test_more_than_one_enabled_entry_is_rejected(self):
        config = _minimal_config_with_cosmos()
        second = {
            "cosmos_evaluator": {
                "enabled": True,
                "endpoint_id": "cosmos-evaluator",
                "output_storage_prefix": "runs/test-002",
                "checks": [{"name": "another.check", "gate": False}],
            }
        }
        config["evaluators"].append(second)

        with pytest.raises(ValidationError, match="at most one enabled"):
            PipelineConfig(**config)

    def test_cosmos_participates_in_exactly_one_entry_rule(self):
        config = _minimal_config_with_cosmos()
        config["evaluators"][0]["hallucination_check"] = {"enabled": False}

        with pytest.raises(ValidationError, match="exactly one"):
            PipelineConfig(**config)

    def test_endpoint_id_not_found_is_rejected(self):
        config = _minimal_config_with_cosmos()
        config["evaluators"][0]["cosmos_evaluator"]["endpoint_id"] = "missing"

        with pytest.raises(ValidationError, match="not found"):
            PipelineConfig(**config)

    @pytest.mark.parametrize("endpoint_id", ["", "   "])
    def test_blank_endpoint_id_is_rejected(self, endpoint_id):
        config = _minimal_config_with_cosmos()
        config["evaluators"][0]["cosmos_evaluator"]["endpoint_id"] = endpoint_id

        with pytest.raises(ValidationError, match="endpoint_id must not be blank"):
            PipelineConfig(**config)

    def test_endpoint_role_mismatch_is_rejected(self):
        config = _minimal_config_with_cosmos()
        config["evaluators"][0]["cosmos_evaluator"]["endpoint_id"] = "cosmos_transfer"

        with pytest.raises(ValidationError, match="has role 'video_transfer'"):
            PipelineConfig(**config)

    @pytest.mark.parametrize("adapter", [None, "nim", "openai.chat.completions"])
    def test_endpoint_requires_explicit_arbitrator_adapter(self, adapter):
        config = _minimal_config_with_cosmos()
        evaluator_endpoint = config["endpoints"][-1]
        if adapter is None:
            evaluator_endpoint.pop("adapter")
        else:
            evaluator_endpoint["adapter"] = adapter

        with pytest.raises(ValidationError, match="must explicitly use adapter"):
            PipelineConfig(**config)

    @pytest.mark.parametrize("checker_name", ["", "   "])
    def test_blank_checker_name_is_rejected(self, checker_name):
        config = _minimal_config_with_cosmos()
        config["evaluators"][0]["cosmos_evaluator"]["checks"][0]["name"] = checker_name

        with pytest.raises(ValidationError, match="name must not be blank"):
            PipelineConfig(**config)

    def test_empty_checker_list_is_rejected(self):
        config = _minimal_config_with_cosmos()
        config["evaluators"][0]["cosmos_evaluator"]["checks"] = []

        with pytest.raises(ValidationError, match="at least 1"):
            PipelineConfig(**config)

    def test_checker_gate_is_required(self):
        config = _minimal_config_with_cosmos()
        config["evaluators"][0]["cosmos_evaluator"]["checks"] = [
            {"name": "vendor.check"}
        ]

        with pytest.raises(ValidationError, match="gate"):
            PipelineConfig(**config)

    @pytest.mark.parametrize("poll_interval", [0, -0.1])
    def test_poll_interval_must_be_positive(self, poll_interval):
        config = _minimal_config_with_cosmos()
        config["evaluators"][0]["cosmos_evaluator"]["poll_interval_seconds"] = (
            poll_interval
        )

        with pytest.raises(ValidationError, match="greater than 0"):
            PipelineConfig(**config)

    def test_duplicate_checker_names_are_rejected_after_trimming(self):
        config = _minimal_config_with_cosmos()
        config["evaluators"][0]["cosmos_evaluator"]["checks"] = [
            {"name": "vendor.check", "gate": True},
            {"name": " vendor.check ", "gate": False},
        ]

        with pytest.raises(ValidationError, match="duplicate Cosmos checker names"):
            PipelineConfig(**config)

    def test_reserved_augmented_video_input_is_rejected(self):
        config = _minimal_config_with_cosmos()
        config["evaluators"][0]["cosmos_evaluator"]["inputs"] = {
            "augmented_video_urls": {"camera": "s3://bucket/out.mp4"}
        }

        with pytest.raises(ValidationError, match="augmented_video_urls is reserved"):
            PipelineConfig(**config)

    @pytest.mark.parametrize("stream_key", ["", "   "])
    def test_blank_explicit_stream_key_is_rejected(self, stream_key):
        config = _minimal_config_with_cosmos()
        config["evaluators"][0]["cosmos_evaluator"]["stream_key"] = stream_key

        with pytest.raises(ValidationError, match="stream_key must not be blank"):
            PipelineConfig(**config)

    @pytest.mark.parametrize("metadata_key", ["", "   "])
    def test_blank_metadata_key_is_rejected(self, metadata_key):
        config = _minimal_config_with_cosmos()
        config["evaluators"][0]["cosmos_evaluator"]["checks"][0]["metadata_key"] = (
            metadata_key
        )

        with pytest.raises(ValidationError, match="metadata_key must not be blank"):
            PipelineConfig(**config)

    @pytest.mark.parametrize("prefix", [None, "", "   "])
    def test_enabled_entry_requires_nonblank_output_prefix(self, prefix):
        config = _minimal_config_with_cosmos()
        cosmos = config["evaluators"][0]["cosmos_evaluator"]
        if prefix is None:
            cosmos.pop("output_storage_prefix")
        else:
            cosmos["output_storage_prefix"] = prefix

        with pytest.raises(ValidationError, match="output_storage_prefix is required"):
            PipelineConfig(**config)

    @pytest.mark.parametrize(
        "prefix",
        [
            "s3://bucket/prefix",
            "bucket:prefix",
            "//bucket/prefix",
            "prefix?token=secret",
            "prefix#fragment",
            "/prefix",
            r"prefix\child",
            "prefix/\x00child",
            "prefix//child",
            "./prefix",
            "prefix/./child",
            "prefix/../child",
        ],
    )
    def test_invalid_output_prefix_forms_are_rejected(self, prefix):
        config = _minimal_config_with_cosmos()
        config["evaluators"][0]["cosmos_evaluator"]["output_storage_prefix"] = prefix

        with pytest.raises(ValidationError, match="output_storage_prefix"):
            PipelineConfig(**config)

    def test_disabled_supplied_prefix_still_gets_full_validation(self):
        config = _minimal_config_with_cosmos(enabled=False)
        config["evaluators"][0]["cosmos_evaluator"]["output_storage_prefix"] = (
            "s3://bucket/not-a-key-prefix"
        )

        with pytest.raises(ValidationError, match="relative key prefix"):
            PipelineConfig(**config)

    @pytest.mark.parametrize(
        "prefix",
        [
            "runs/test-001",
            "runs/test-001/",
            "<OUTPUT_PREFIX>/ct3-omni/<RUN_ID>/",
        ],
    )
    def test_valid_output_prefix_preserves_trailing_slash(self, prefix):
        config = _minimal_config_with_cosmos()
        config["evaluators"][0]["cosmos_evaluator"]["output_storage_prefix"] = (
            f"  {prefix}  "
        )

        pc = PipelineConfig(**config)
        assert pc.evaluators[0].cosmos_evaluator.output_storage_prefix == prefix

    @pytest.mark.parametrize(
        "output_uri",
        [
            "s3://bucket/out.mp4",
            "msc://profile/out.mp4",
            "gs://bucket/out.mp4",
            "az://container/out.mp4",
            "http://storage.test/out.mp4",
            "https://storage.test/out.mp4?signature=secret",
        ],
    )
    def test_supported_remote_output_paths_are_valid(self, output_uri):
        config = _minimal_config_with_cosmos()
        config["data"][0]["output"]["video"] = output_uri

        PipelineConfig(**config)

    @pytest.mark.parametrize("output_path", ["/tmp/out.mp4", "relative/out.mp4"])
    def test_enabled_cosmos_requires_remote_generated_output(self, output_path):
        config = _minimal_config_with_cosmos()
        config["data"][0]["output"]["video"] = output_path

        with pytest.raises(ValidationError, match=r"data\[0\]\.output\.video"):
            PipelineConfig(**config)

    @pytest.mark.parametrize(
        ("input_field", "input_value", "error_path"),
        [
            ("rgb", "/tmp/input.mp4", r"data\[0\]\.inputs\.rgb"),
            (
                "controls",
                {"wsm": "relative/control.mp4"},
                r"data\[0\]\.inputs\.controls\.wsm",
            ),
        ],
    )
    def test_enabled_cosmos_requires_remote_sample_media(
        self, input_field, input_value, error_path
    ):
        config = _minimal_config_with_cosmos()
        config["data"][0]["inputs"][input_field] = input_value

        with pytest.raises(ValidationError, match=error_path):
            PipelineConfig(**config)

    @pytest.mark.parametrize(
        ("field_name", "value"),
        [
            ("original_video_urls", {"camera": "/tmp/original.mp4"}),
            ("world_model_video_urls", {"camera": "relative/world.mp4"}),
            ("rds_hq_url", "/tmp/rds.zip"),
        ],
    )
    def test_known_media_inputs_must_be_remote(self, field_name, value):
        config = _minimal_config_with_cosmos()
        config["evaluators"][0]["cosmos_evaluator"]["inputs"] = {field_name: value}

        with pytest.raises(ValidationError, match=field_name):
            PipelineConfig(**config)

    @pytest.mark.parametrize(
        "field_name", ["original_video_urls", "world_model_video_urls"]
    )
    def test_per_stream_media_inputs_must_be_mappings(self, field_name):
        config = _minimal_config_with_cosmos()
        config["evaluators"][0]["cosmos_evaluator"]["inputs"] = {
            field_name: "s3://bucket/media.mp4"
        }

        with pytest.raises(ValidationError, match="must be a mapping"):
            PipelineConfig(**config)

    def test_per_stream_media_inputs_reject_blank_stream_keys(self):
        config = _minimal_config_with_cosmos()
        config["evaluators"][0]["cosmos_evaluator"]["inputs"] = {
            "original_video_urls": {"   ": "s3://bucket/original.mp4"}
        }

        with pytest.raises(ValidationError, match="blank stream key"):
            PipelineConfig(**config)

    @pytest.mark.parametrize(
        "field_name", ["original_video_urls", "world_model_video_urls"]
    )
    def test_explicit_stream_key_must_be_present_in_companion_maps(self, field_name):
        config = _minimal_config_with_cosmos()
        cosmos = config["evaluators"][0]["cosmos_evaluator"]
        cosmos["stream_key"] = "camera_front"
        cosmos["inputs"] = {field_name: {"other": "s3://bucket/companion.mp4"}}

        with pytest.raises(
            ValidationError, match=r"lacks resolved stream key.*camera_front"
        ):
            PipelineConfig(**config)

    def test_derived_stream_key_can_match_single_sample_companion(self):
        config = _minimal_config_with_cosmos()
        config["evaluators"][0]["cosmos_evaluator"]["inputs"] = {
            "original_video_urls": {"out": "s3://bucket/original.mp4"}
        }

        PipelineConfig(**config)

    def test_derived_stream_key_must_match_single_sample_companion(self):
        config = _minimal_config_with_cosmos()
        config["evaluators"][0]["cosmos_evaluator"]["inputs"] = {
            "original_video_urls": {"other": "s3://bucket/original.mp4"}
        }

        with pytest.raises(ValidationError, match=r"lacks resolved stream key.*out"):
            PipelineConfig(**config)

    def test_companion_maps_require_every_derived_multi_sample_key(self):
        config = _minimal_config_with_cosmos()
        config["data"][0]["output"]["video"] = "s3://bucket/run/front.mp4"
        config["data"].append(
            {
                "inputs": {"rgb": "s3://bucket/run/rear-input.mp4"},
                "output": {
                    "video": "s3://bucket/run/rear.mp4",
                    "caption": "/tmp/rear.txt",
                    "metadata": "/tmp/rear.json",
                },
            }
        )
        companion_map = {"front": "s3://bucket/original-front.mp4"}
        config["evaluators"][0]["cosmos_evaluator"]["inputs"] = {
            "world_model_video_urls": companion_map
        }

        with pytest.raises(ValidationError, match=r"lacks resolved stream key.*rear"):
            PipelineConfig(**config)

        companion_map["rear"] = "s3://bucket/original-rear.mp4"
        PipelineConfig(**config)

    def test_explicit_stream_key_is_global_for_multiple_samples(self):
        config = _minimal_config_with_cosmos()
        config["data"][0]["output"]["video"] = "s3://bucket/run/front.mp4"
        config["data"].append(
            {
                "inputs": {"rgb": "s3://bucket/run/rear-input.mp4"},
                "output": {
                    "video": "s3://bucket/run/rear.mp4",
                    "caption": "/tmp/rear.txt",
                    "metadata": "/tmp/rear.json",
                },
            }
        )
        cosmos = config["evaluators"][0]["cosmos_evaluator"]
        cosmos["stream_key"] = "camera"
        cosmos["inputs"] = {
            "original_video_urls": {"camera": "s3://bucket/original.mp4"}
        }

        parsed = PipelineConfig(**config)
        assert parsed.evaluators[0].cosmos_evaluator.stream_key == "camera"

    @pytest.mark.parametrize(
        "metadata_key",
        ["prompt", "captioning", "natural_caption", "cosmos_evaluator"],
    )
    def test_alias_cannot_overwrite_reserved_root_metadata(self, metadata_key):
        config = _minimal_config_with_cosmos()
        config["evaluators"][0]["cosmos_evaluator"]["checks"][0]["metadata_key"] = (
            metadata_key
        )

        with pytest.raises(ValidationError, match="metadata_key"):
            PipelineConfig(**config)

    @pytest.mark.parametrize(
        ("builtin_entry", "metadata_key"),
        [
            ({"hallucination_check": {"enabled": True}}, "hallucination_check"),
            (
                {"attribute_verification": {"enabled": True}},
                "attribute_verification",
            ),
        ],
    )
    def test_alias_cannot_overwrite_enabled_builtin(self, builtin_entry, metadata_key):
        config = _minimal_config_with_cosmos()
        if "attribute_verification" in builtin_entry:
            config["endpoints"].append(_ep("llm", "llm", "http://llm", "llm"))
            config["endpoints"].append(_ep("vlm", "vlm", "http://vlm", "vlm"))
        config["evaluators"].append(builtin_entry)
        config["evaluators"][0]["cosmos_evaluator"]["checks"][0]["metadata_key"] = (
            metadata_key
        )

        with pytest.raises(ValidationError, match="enabled built-in evaluator"):
            PipelineConfig(**config)

    def test_alias_may_replace_disabled_builtin_key(self):
        config = _minimal_config_with_cosmos()
        config["evaluators"].append({"attribute_verification": {"enabled": False}})
        config["evaluators"][0]["cosmos_evaluator"]["checks"][0]["metadata_key"] = (
            "attribute_verification"
        )

        PipelineConfig(**config)

    def test_duplicate_aliases_are_rejected(self):
        config = _minimal_config_with_cosmos()
        config["evaluators"][0]["cosmos_evaluator"]["checks"] = [
            {"name": "first.check", "gate": True, "metadata_key": "compat"},
            {"name": "second.check", "gate": False, "metadata_key": " compat "},
        ]

        with pytest.raises(ValidationError, match="duplicate Cosmos metadata_key"):
            PipelineConfig(**config)


# ---------------------------------------------------------------------------
# Default value tests
# ---------------------------------------------------------------------------


class TestDefaults:
    """Verify that Pydantic defaults match the tuned values."""

    def test_default_pipeline_settings(self):
        pc = PipelineConfig(**_minimal_config())
        assert pc.pipeline.retry == 1
        assert pc.pipeline.regenerate_caption_on_retry is True
        assert pc.pipeline.logging.enabled is True
        assert pc.pipeline.logging.level == "INFO"
        assert pc.pipeline.evaluation.strict is True
        assert pc.pipeline.evaluation.retain_failures is True

    def test_default_inference_parameters(self):
        pc = PipelineConfig(**_minimal_config_with_captioning())
        vlm_params = pc.captioning.vlm.parameters
        assert vlm_params.temperature == 0.3
        assert vlm_params.top_p == 0.95
        assert vlm_params.frequency_penalty == 1.05
        assert vlm_params.max_tokens == 4096
        assert vlm_params.stream is False
        assert vlm_params.fps == 4.0
        assert vlm_params.max_pixels == 307200

    def test_default_augmentation_parameters(self):
        pc = PipelineConfig(**_minimal_config())
        params = pc.augmentation.parameters
        assert params.sigma == 90
        assert params.seed is None
        assert params.guidance == 3.0
        assert params.num_steps == 35
        # Default ' ' (single space) so Image-Edit-Plus runs with CFG on
        # by default. Set to null in YAML to disable.
        assert params.negative_prompt == " "

    def test_image_edit_negative_prompt_can_be_null(self):
        config = _minimal_config_image_edit()
        config["augmentation"]["parameters"]["negative_prompt"] = None
        pc = PipelineConfig(**config)
        assert pc.augmentation.parameters.negative_prompt is None

    def test_image_edit_negative_prompt_custom_value(self):
        config = _minimal_config_image_edit()
        config["augmentation"]["parameters"]["negative_prompt"] = "blurry, low quality"
        pc = PipelineConfig(**config)
        assert pc.augmentation.parameters.negative_prompt == "blurry, low quality"

    def test_default_hallucination_params(self):
        config = _minimal_config()
        config["evaluators"] = [{"hallucination_check": {}}]
        pc = PipelineConfig(**config)
        hc = pc.evaluators[0].hallucination_check
        assert hc.enabled is True
        assert hc.threshold == 0.682
        assert hc.params.grad_thresh == 10.0


# ---------------------------------------------------------------------------
# Cross-section validation tests
# ---------------------------------------------------------------------------


class TestCrossSectionValidation:
    """Endpoint requirements are enforced across sections."""

    def test_cosmos_transfer_requires_endpoint(self):
        """Every model now needs a resolvable endpoint (no local fallback)."""
        config = _minimal_config()
        config["endpoints"] = []
        with pytest.raises(ValidationError, match="requires a matching"):
            PipelineConfig(**config)

    def test_cosmos_predict_requires_endpoint(self):
        config = _minimal_config_predict()
        config["endpoints"] = []
        with pytest.raises(ValidationError, match="requires a matching"):
            PipelineConfig(**config)

    def test_image_edit_requires_endpoint(self):
        config = _minimal_config_image_edit()
        config["endpoints"] = []
        with pytest.raises(ValidationError, match="requires a matching"):
            PipelineConfig(**config)

    def test_unknown_adapter_rejected(self):
        config = _minimal_config_image_edit()
        config["endpoints"][0]["adapter"] = "not-a-real-adapter"
        with pytest.raises(ValidationError, match="unknown adapter"):
            PipelineConfig(**config)

    def test_vlm_captioning_requires_vlm_endpoint(self, monkeypatch):
        monkeypatch.delenv("VLM_ENDPOINT_URL", raising=False)
        config = _minimal_config_with_captioning()
        config["endpoints"] = [ep for ep in config["endpoints"] if ep["role"] != "vlm"]
        with pytest.raises(ValidationError, match="role 'vlm'"):
            PipelineConfig(**config)

    def test_llm_captioning_requires_llm_endpoint(self, monkeypatch):
        monkeypatch.delenv("LLM_ENDPOINT_URL", raising=False)
        monkeypatch.delenv("LLM_CAPTION_ENDPOINT_URL", raising=False)
        config = _minimal_config()
        config["endpoints"].append(_ep("vlm", "vlm", "http://x", "x"))
        config["captioning"] = {
            "vlm": {"system_prompt": "t", "user_prompt": "t"},
            "llm": {"system_prompt": "t", "variables": {"a": ["b"]}},
        }
        with pytest.raises(ValidationError, match="role 'llm'"):
            PipelineConfig(**config)

    def test_text_captioning_does_not_require_llm_endpoint(self):
        """'text' field set should work without endpoints.llm."""
        config = _minimal_config()
        config["captioning"] = {
            "llm": {"text": "a fixed prompt"},
        }
        pc = PipelineConfig(**config)
        assert pc.captioning.llm.text == "a fixed prompt"

    def test_file_captioning_does_not_require_llm_endpoint(self):
        """'file_path' field set should work without endpoints.llm."""
        config = _minimal_config()
        config["captioning"] = {
            "llm": {"file_path": "/path/to/prompts.txt"},
        }
        pc = PipelineConfig(**config)
        assert pc.captioning.llm.file_path == "/path/to/prompts.txt"

    def test_text_and_file_path_mutually_exclusive(self):
        config = _minimal_config()
        config["captioning"] = {
            "llm": {"text": "a prompt", "file_path": "/path/to/file.txt"},
        }
        with pytest.raises(ValidationError, match="mutually exclusive"):
            PipelineConfig(**config)

    def test_attribute_verification_requires_llm_endpoint(self, monkeypatch):
        monkeypatch.delenv("LLM_ENDPOINT_URL", raising=False)
        config = _minimal_config()  # only a video_transfer endpoint, no llm/vlm
        config["evaluators"] = [{"attribute_verification": {"enabled": True}}]
        with pytest.raises(ValidationError, match="role 'llm'"):
            PipelineConfig(**config)

    def test_attribute_verification_requires_vlm_endpoint(self, monkeypatch):
        # An enabled attribute_verification needs a VLM too, even with no inline
        # vlm_verification block (it silently no-ops otherwise).
        monkeypatch.delenv("VLM_ENDPOINT_URL", raising=False)
        config = _minimal_config()
        config["endpoints"].append(_ep("llm", "llm", "http://x", "x"))
        config["evaluators"] = [{"attribute_verification": {"enabled": True}}]
        with pytest.raises(ValidationError, match="role 'vlm'"):
            PipelineConfig(**config)

    def test_disabled_attribute_verification_skips_endpoint_check(self, monkeypatch):
        monkeypatch.delenv("LLM_ENDPOINT_URL", raising=False)
        monkeypatch.delenv("VLM_ENDPOINT_URL", raising=False)
        config = _minimal_config()  # no llm/vlm endpoint
        config["evaluators"] = [{"attribute_verification": {"enabled": False}}]
        pc = PipelineConfig(**config)  # disabled -> no transport required
        assert pc.evaluators[0].attribute_verification.enabled is False

    def test_two_same_role_endpoints_require_id_even_with_env(self, monkeypatch):
        # An env URL override does NOT disambiguate 2+ same-role endpoints (it
        # can't say which one), and runtime select_endpoint() raises on the
        # ambiguous match — so this must fail validation, not crash at startup.
        monkeypatch.setenv("VLM_ENDPOINT_URL", "http://env/v1")
        config = _minimal_config()
        config["endpoints"].append(_ep("llm", "llm", "http://llm/v1", "llm"))
        config["endpoints"].append(_ep("vlm-a", "vlm", "http://a/v1", "a"))
        config["endpoints"].append(_ep("vlm-b", "vlm", "http://b/v1", "b"))
        config["evaluators"] = [
            {"attribute_verification": {"vlm_verification": {"system_prompt": "t"}}}
        ]
        with pytest.raises(ValidationError, match="set endpoint_id"):
            PipelineConfig(**config)

    def test_two_same_role_endpoints_ok_when_id_pins_one(self, monkeypatch):
        monkeypatch.delenv("VLM_ENDPOINT_URL", raising=False)
        config = _minimal_config()
        config["endpoints"].append(_ep("llm", "llm", "http://llm/v1", "llm"))
        config["endpoints"].append(_ep("vlm-a", "vlm", "http://a/v1", "a"))
        config["endpoints"].append(_ep("vlm-b", "vlm", "http://b/v1", "b"))
        config["evaluators"] = [
            {
                "attribute_verification": {
                    "vlm_verification": {
                        "system_prompt": "t",
                        "endpoint_id": "vlm-b",
                    }
                }
            }
        ]
        pc = PipelineConfig(**config)
        assert (
            pc.evaluators[0].attribute_verification.vlm_verification.endpoint_id
            == "vlm-b"
        )


class TestEndpointSelection:
    """BYOM endpoint_id selection guards at PipelineConfig construction."""

    def test_captioning_llm_endpoint_id_not_found(self, monkeypatch):
        monkeypatch.delenv("LLM_ENDPOINT_URL", raising=False)
        monkeypatch.delenv("LLM_CAPTION_ENDPOINT_URL", raising=False)
        config = _minimal_config()
        config["endpoints"].append(_ep("llm-a", "llm", "http://x", "x"))
        config["captioning"] = {
            "llm": {
                "system_prompt": "t",
                "variables": {"a": ["b"]},
                "endpoint_id": "does-not-exist",
            }
        }
        with pytest.raises(ValidationError, match="not found"):
            PipelineConfig(**config)

    def test_endpoint_id_wrong_role_rejected(self, monkeypatch):
        monkeypatch.delenv("LLM_ENDPOINT_URL", raising=False)
        monkeypatch.delenv("LLM_CAPTION_ENDPOINT_URL", raising=False)
        config = _minimal_config()
        config["endpoints"].append(_ep("vlm-a", "vlm", "http://x", "x"))
        config["endpoints"].append(_ep("llm-a", "llm", "http://x", "x"))
        config["captioning"] = {
            "llm": {
                "system_prompt": "t",
                "variables": {"a": ["b"]},
                "endpoint_id": "vlm-a",
            }
        }
        with pytest.raises(ValidationError, match="has role 'vlm'"):
            PipelineConfig(**config)

    def test_two_llm_endpoints_distinct_selectors_ok(self, monkeypatch):
        monkeypatch.delenv("LLM_ENDPOINT_URL", raising=False)
        monkeypatch.delenv("LLM_CAPTION_ENDPOINT_URL", raising=False)
        config = _minimal_config()
        config["endpoints"].append(_ep("llm-cap", "llm", "http://a", "a"))
        config["endpoints"].append(_ep("llm-qg", "llm", "http://b", "b"))
        config["endpoints"].append(_ep("vlm", "vlm", "http://c", "c"))
        config["captioning"] = {
            "llm": {
                "system_prompt": "t",
                "variables": {"a": ["b"]},
                "endpoint_id": "llm-cap",
            }
        }
        config["evaluators"] = [
            {
                "attribute_verification": {
                    "question_generation": {"endpoint_id": "llm-qg"},
                }
            }
        ]
        pc = PipelineConfig(**config)
        assert pc.captioning.llm.endpoint_id == "llm-cap"
        qg = pc.evaluators[0].attribute_verification.question_generation
        assert qg.endpoint_id == "llm-qg"

    def test_ambiguous_vlm_role_without_endpoint_id_rejected(self, monkeypatch):
        monkeypatch.delenv("VLM_ENDPOINT_URL", raising=False)
        config = _minimal_config()
        config["endpoints"].append(_ep("vlm-a", "vlm", "http://a", "a"))
        config["endpoints"].append(_ep("vlm-b", "vlm", "http://b", "b"))
        config["captioning"] = {
            "vlm": {"system_prompt": "t", "user_prompt": "t"},
        }
        with pytest.raises(ValidationError, match="endpoint_id"):
            PipelineConfig(**config)

    def test_duplicate_ids_rejected(self):
        config = _minimal_config()
        config["endpoints"].append(_ep("dup", "vlm", "http://a", "a"))
        config["endpoints"].append(_ep("dup", "llm", "http://b", "b"))
        with pytest.raises(ValidationError, match="duplicate endpoint ids"):
            PipelineConfig(**config)

    def test_id_omitted_resolves_by_role(self, monkeypatch):
        monkeypatch.delenv("VLM_ENDPOINT_URL", raising=False)
        config = _minimal_config()
        # An endpoint with no id at all (now optional) plus captioning.vlm.
        config["endpoints"].append(
            {"role": "vlm", "url": "http://localhost:9001/v1", "model": "test-vlm"}
        )
        config["captioning"] = {
            "vlm": {"system_prompt": "t", "user_prompt": "t"},
        }
        pc = PipelineConfig(**config)
        vlm_eps = [ep for ep in pc.endpoints if ep.role == "vlm"]
        assert len(vlm_eps) == 1
        assert vlm_eps[0].id is None


# ---------------------------------------------------------------------------
# Negative tests — invalid configs are rejected
# ---------------------------------------------------------------------------


class TestInvalidConfigs:
    """Configs that must be rejected with clear errors."""

    def test_unknown_model_without_endpoint_rejected(self):
        """BYOM model names are free-form, but must resolve to an endpoint."""
        config = _minimal_config()
        config["augmentation"]["model"]["name"] = "my-unmapped-model"
        with pytest.raises(ValidationError, match="requires a matching"):
            PipelineConfig(**config)

    def test_byom_model_name_with_matching_endpoint_ok(self):
        """A free-form model name resolves by endpoint id."""
        config = _minimal_config()
        config["augmentation"]["model"]["name"] = "my-byom-model"
        config["endpoints"] = [
            _ep("my-byom-model", "image_edit", "http://x", adapter="nim"),
        ]
        # rgb input is supplied by _minimal_config, role image_edit -> rgb ok.
        pc = PipelineConfig(**config)
        assert pc.augmentation.model.name == "my-byom-model"

    def test_missing_data_section(self):
        with pytest.raises(ValidationError):
            PipelineConfig(endpoints=[])

    def test_empty_data_list(self):
        config = _minimal_config()
        config["data"] = []
        with pytest.raises(ValidationError):
            PipelineConfig(**config)

    def test_missing_rgb_rejected_for_non_text2world(self):
        """rgb is required for models that need input media."""
        config = _minimal_config()
        del config["data"][0]["inputs"]["rgb"]
        with pytest.raises(ValidationError, match=r"inputs\.rgb is required"):
            PipelineConfig(**config)

    def test_missing_rgb_allowed_for_text2world(self):
        """text2world allows null/missing rgb since no media is needed."""
        pc = PipelineConfig(**_minimal_config_predict_text2world())
        rgb = pc.data[0].inputs.rgb if pc.data[0].inputs else None
        assert rgb is None

    def test_missing_video_in_output(self):
        config = _minimal_config()
        del config["data"][0]["output"]["video"]
        with pytest.raises(ValidationError):
            PipelineConfig(**config)

    def test_invalid_logging_level(self):
        config = _minimal_config()
        config["pipeline"] = {"logging": {"level": "TRACE"}}
        with pytest.raises(ValidationError):
            PipelineConfig(**config)

    def test_evaluator_retries_are_rejected(self):
        config = _minimal_config()
        config["evaluators"] = [{"hallucination_check": {"retries": 1}}]
        with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
            PipelineConfig(**config)

    def test_evaluator_entry_with_zero_types(self):
        config = _minimal_config()
        config["evaluators"] = [{}]
        with pytest.raises(ValidationError, match="exactly one"):
            PipelineConfig(**config)

    def test_evaluator_entry_with_two_types(self):
        config = _minimal_config()
        config["evaluators"] = [
            {
                "hallucination_check": {"enabled": True},
                "vlm_verification": {"system_prompt": "t"},
            }
        ]
        with pytest.raises(ValidationError, match="exactly one"):
            PipelineConfig(**config)

    def test_modality_weight_out_of_range(self):
        config = _minimal_config()
        config["augmentation"]["modalities"] = {"edge": 1.5}
        with pytest.raises(ValidationError):
            PipelineConfig(**config)


# ---------------------------------------------------------------------------
# Extra question tests
# ---------------------------------------------------------------------------


class TestExtraQuestions:
    """Verify extra_questions validation on evaluators."""

    def test_valid_extra_question(self):
        config = _minimal_config()
        config["endpoints"].append(_ep("llm", "llm", "http://localhost:9002/v1", "llm"))
        config["endpoints"].append(_ep("vlm", "vlm", "http://localhost:9001/v1", "vlm"))
        config["evaluators"] = [
            {
                "attribute_verification": {
                    "extra_questions": [
                        {
                            "question": "Is it consistent?",
                            "options": {"A": "Yes", "B": "No"},
                            "correct_answer": "A",
                        }
                    ]
                }
            }
        ]
        pc = PipelineConfig(**config)
        eq = pc.evaluators[0].attribute_verification.extra_questions[0]
        assert eq.question == "Is it consistent?"
        assert eq.correct_answer == "A"

    def test_extra_question_with_reasoning(self):
        config = _minimal_config()
        config["endpoints"].append(_ep("llm", "llm", "http://localhost:9002/v1", "llm"))
        config["endpoints"].append(_ep("vlm", "vlm", "http://localhost:9001/v1", "vlm"))
        config["evaluators"] = [
            {
                "attribute_verification": {
                    "extra_questions": [
                        {
                            "variable": "color",
                            "question": "What color?",
                            "options": {"A": "Red", "B": "Blue"},
                            "correct_answer": "B",
                            "request_reasoning": True,
                        }
                    ]
                }
            }
        ]
        pc = PipelineConfig(**config)
        eq = pc.evaluators[0].attribute_verification.extra_questions[0]
        assert eq.variable == "color"
        assert eq.request_reasoning is True
