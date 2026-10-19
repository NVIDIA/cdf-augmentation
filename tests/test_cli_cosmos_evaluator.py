# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Mocked CLI integration coverage for external Cosmos evaluation."""

import json
import logging
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
import sys

import pytest
import yaml

from aug_utils.schema import PipelineConfig
from modules import cli
from verification import CosmosEvaluationOutcome
from verification.cosmos_evaluator import CosmosEvaluatorServiceError


LOGGER = logging.getLogger("test-cli-cosmos-evaluator")


class FakeCaptioner:
    def get_caption(self, _media_path):
        return "Generate the selected scene."


def test_validate_only_stops_after_schema_validation(tmp_path, monkeypatch):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump({"data": [{"output": {}}]}))
    events = []

    monkeypatch.setattr(cli, "load_secrets", lambda _logger: None)
    monkeypatch.setattr(cli, "validate_environment", lambda _logger: None)
    monkeypatch.setattr(
        cli,
        "validate_config_structure",
        lambda _config, _logger: events.append("validated") or object(),
    )
    monkeypatch.setattr(
        cli,
        "_init_cosmos_evaluator",
        lambda *_args: pytest.fail("validate-only must not initialize endpoints"),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["cli.py", "--config", str(config_path), "--validate-only"],
    )

    cli.main()

    assert events == ["validated"]


class FakeGenerator:
    def __init__(self, seed=10):
        self.seed = seed


class FakeTracker:
    def __init__(self):
        self.updates = []
        self.completed = None

    def update(self, **kwargs):
        self.updates.append(kwargs)

    def complete(self, **kwargs):
        self.completed = kwargs


class FakeAttributeVerifier:
    def __init__(self, passed):
        self.passed = passed
        self.outcomes = iter(passed) if isinstance(passed, (list, tuple)) else None
        self.calls = []

    def verify_video_attributes(
        self, video_path, selected_variables, variable_options, extra_questions
    ):
        self.calls.append(
            {
                "video_path": video_path,
                "selected_variables": selected_variables,
                "variable_options": variable_options,
                "extra_questions": extra_questions,
            }
        )
        passed = next(self.outcomes) if self.outcomes is not None else self.passed
        return passed, {"opaque": "built-in-result"}


class FakeCosmosEvaluator:
    def __init__(
        self,
        outcomes,
        events,
        *,
        merge_captioning_selections=False,
        preflight_error=None,
    ):
        self.outcomes = list(outcomes)
        self.events = events
        self.merge_captioning_selections = merge_captioning_selections
        self.preflight_error = preflight_error
        self.preflight_calls = 0
        self.calls = []

    def preflight(self):
        self.preflight_calls += 1
        self.events.append("preflight")
        if self.preflight_error is not None:
            raise self.preflight_error
        return SimpleNamespace(version="1.1.0", commit_sha="abc123")

    def evaluate_candidate(self, candidate_video_uri, **kwargs):
        self.events.append("cosmos_evaluate")
        self.calls.append({"candidate_video_uri": candidate_video_uri, **kwargs})
        if not self.outcomes:
            raise AssertionError("unexpected external evaluation call")
        return self.outcomes.pop(0)


def _outcome(
    quality_passed=True,
    *,
    operational_success=True,
    provider=None,
    aliases=None,
    error=None,
):
    if provider is None:
        provider = {
            "request_id": 41,
            "status": "success" if operational_success else "service_failure",
            "version": "1.1.0",
            "commit_sha": "abc123",
            "stream_key": "candidate",
            "checks": {
                "arbitrary.future.checker": {
                    "candidate": {"passed": quality_passed, "result": {"x": 1}}
                }
            },
        }
    return CosmosEvaluationOutcome(
        operational_success=operational_success,
        quality_passed=quality_passed if operational_success else None,
        provider_metadata=provider,
        compatibility_aliases=aliases or {},
        error=error,
    )


def _typed_config(enabled):
    evaluator = {
        "enabled": enabled,
        "endpoint_id": "cosmos",
        "checks": [{"name": "arbitrary.future.checker", "gate": True}],
    }
    endpoints = []
    input_video = "/tmp/input.mp4"
    output_video = "/tmp/output.mp4"
    if enabled:
        evaluator["output_storage_prefix"] = "runs/test/"
        endpoints.append(
            {
                "id": "cosmos",
                "role": "evaluator",
                "url": "https://evaluator.example.test",
                "adapter": "cosmos.evaluator.arbitrator",
            }
        )
        input_video = "s3://bucket/input.mp4"
        output_video = "s3://bucket/output.mp4"
    return PipelineConfig.model_validate(
        {
            "data": [
                {
                    "inputs": {"rgb": input_video},
                    "output": {
                        "video": output_video,
                        "caption": "/tmp/caption.txt",
                        "metadata": "/tmp/metadata.json",
                    },
                }
            ],
            "endpoints": endpoints,
            "evaluators": [{"cosmos_evaluator": evaluator}],
        }
    )


@pytest.mark.parametrize("enabled", [True, False])
def test_init_cosmos_evaluator_respects_enabled_without_network(monkeypatch, enabled):
    pipeline_config = _typed_config(enabled)
    sentinel = object()
    factory = Mock(return_value=sentinel)
    monkeypatch.setattr(cli.CosmosEvaluatorClient, "from_config", factory)

    result = cli._init_cosmos_evaluator(pipeline_config, LOGGER)

    if enabled:
        assert result is sentinel
        factory.assert_called_once_with(
            pipeline_config.endpoints,
            pipeline_config.evaluators[0].cosmos_evaluator,
            logger=LOGGER,
        )
    else:
        assert result is None
        factory.assert_not_called()


def _run_cli(
    tmp_path,
    monkeypatch,
    outcomes,
    *,
    retry=0,
    strict=True,
    retain_failures=True,
    merge_captioning_selections=False,
    built_in_passed=None,
    cosmos_enabled=True,
    preflight_error=None,
    generation_successes=None,
):
    input_path = tmp_path / "input.mp4"
    caption_path = tmp_path / "caption.txt"
    video_path = tmp_path / "candidate.mp4"
    metadata_path = tmp_path / "metadata.json"
    evaluation_path = tmp_path / "evaluation.json"
    input_path.write_bytes(b"mock-input")

    config = {
        "data": [
            {
                "inputs": {"rgb": str(input_path)},
                "output": {
                    "video": str(video_path),
                    "caption": str(caption_path),
                    "metadata": str(metadata_path),
                    "evaluation": str(evaluation_path),
                },
            }
        ],
        "pipeline": {
            "retry": retry,
            "regenerate_caption_on_retry": False,
            "evaluation": {
                "strict": strict,
                "retain_failures": retain_failures,
            },
        },
        "captioning": {
            "llm": {
                "text": "Generate the selected scene.",
                "variables": {
                    "weather": ["sunny", "snowy"],
                    "lighting": ["day", "night"],
                },
                "verification_values": {
                    "weather": ["snowy"],
                    "lighting": ["night"],
                },
                "verification_options": {
                    "weather": ["sunny", "snowy"],
                    "lighting": ["day", "night"],
                },
            }
        },
        "augmentation": {
            "model": {"name": "mock-generator"},
            "parameters": {"seed": 10},
        },
        "evaluators": [
            {
                "cosmos_evaluator": {
                    "enabled": cosmos_enabled,
                    "merge_captioning_selections": merge_captioning_selections,
                    "checks": [
                        {
                            "name": "arbitrary.future.checker",
                            "gate": True,
                            "metadata_key": "future_check",
                        }
                    ],
                }
            }
        ],
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))

    evaluation_settings = SimpleNamespace(
        strict=strict,
        retain_failures=retain_failures,
    )
    pipeline_config = SimpleNamespace(
        evaluators=[],
        endpoints=[],
        pipeline=SimpleNamespace(
            retry=retry,
            regenerate_caption_on_retry=False,
            request_timeout=0.01,
            evaluation=evaluation_settings,
        ),
        data_processing=None,
    )
    events = []
    cosmos = None
    if cosmos_enabled:
        cosmos = FakeCosmosEvaluator(
            outcomes,
            events,
            merge_captioning_selections=merge_captioning_selections,
            preflight_error=preflight_error,
        )
    generator = FakeGenerator()
    tracker = FakeTracker()
    attribute_verifier = (
        FakeAttributeVerifier(built_in_passed) if built_in_passed is not None else None
    )

    monkeypatch.setattr(cli, "load_secrets", lambda _logger: None)
    monkeypatch.setattr(cli, "validate_environment", lambda _logger: None)

    def validate_config(_config, _logger):
        events.append("config_validation")
        return pipeline_config

    monkeypatch.setattr(
        cli,
        "validate_config_structure",
        validate_config,
    )
    monkeypatch.setattr(cli, "_init_cosmos_evaluator", lambda *_args: cosmos)

    def create_captioner(*_args, **_kwargs):
        events.append("captioner_init")
        return FakeCaptioner()

    def create_generator(*_args, **_kwargs):
        events.append("generator_init")
        return generator

    monkeypatch.setattr(cli, "create_captioner", create_captioner)
    monkeypatch.setattr(cli, "create_generator", create_generator)
    monkeypatch.setattr(
        cli,
        "_init_evaluators",
        lambda *_args: (
            None,
            attribute_verifier,
            {"enabled": True} if attribute_verifier else None,
            None,
        ),
    )
    monkeypatch.setattr(cli, "validate_sample_data_availability", lambda *_args: True)
    monkeypatch.setattr(cli, "NVCFProgressTracker", lambda logger=None: tracker)

    generation_seeds = []
    generation_successes_iter = (
        None if generation_successes is None else iter(generation_successes)
    )

    def generate(
        current_generator,
        _prompt,
        _input_path,
        _controls,
        output_path,
        _max_retries,
        _request_timeout,
        _logger,
    ):
        events.append("generation")
        generation_seeds.append(current_generator.seed)
        if generation_successes_iter is None:
            success = True
        else:
            try:
                success = next(generation_successes_iter)
            except StopIteration:
                pytest.fail("Unexpected extra call to generate_with_retries")
        if success:
            Path(output_path).write_bytes(
                f"mock-candidate-{current_generator.seed}".encode()
            )
        return success, output_path, 0.1

    monkeypatch.setattr(cli, "generate_with_retries", generate)
    monkeypatch.setattr(sys, "argv", ["cli.py", "--config", str(config_path)])
    monkeypatch.setenv("OVERWRITE_CAPTION", "true")

    exit_code = None
    try:
        cli.main()
    except SystemExit as exc:
        exit_code = exc.code

    return SimpleNamespace(
        exit_code=exit_code,
        cosmos=cosmos,
        generator=generator,
        attribute_verifier=attribute_verifier,
        tracker=tracker,
        events=events,
        generation_seeds=generation_seeds,
        video_path=video_path,
        caption_path=caption_path,
        metadata_path=metadata_path,
        evaluation_path=evaluation_path,
    )


def test_preflight_runs_before_captioning_and_generation(tmp_path, monkeypatch):
    run = _run_cli(tmp_path, monkeypatch, [_outcome()])

    assert run.exit_code is None
    assert run.cosmos.preflight_calls == 1
    assert run.events.index("config_validation") < run.events.index("preflight")
    assert run.events.index("preflight") < run.events.index("captioner_init")
    assert run.events.index("preflight") < run.events.index("generation")


def test_preflight_failure_stops_before_captioning_or_generation(tmp_path, monkeypatch):
    run = _run_cli(
        tmp_path,
        monkeypatch,
        [],
        preflight_error=RuntimeError("unavailable"),
    )

    assert run.exit_code == 1
    assert run.events == ["config_validation", "preflight"]
    assert not run.video_path.exists()


def test_expected_preflight_failure_logs_clear_safe_diagnostic(
    tmp_path, monkeypatch, caplog
):
    with caplog.at_level(logging.ERROR):
        run = _run_cli(
            tmp_path,
            monkeypatch,
            [],
            preflight_error=CosmosEvaluatorServiceError(
                "configured Cosmos checker is not healthy"
            ),
        )

    assert run.exit_code == 1
    assert "configured Cosmos checker is not healthy" in caplog.text
    assert run.events == ["config_validation", "preflight"]
    assert not run.video_path.exists()


def test_external_only_pass_writes_provider_alias_and_overlay(tmp_path, monkeypatch):
    provider = {
        "request_id": 77,
        "status": "success",
        "version": "1.1.0",
        "commit_sha": "def456",
        "stream_key": "candidate",
        "checks": {
            "arbitrary.future.checker": {
                "candidate": {
                    "passed": True,
                    "result": {"opaque_future_field": [1, 2, 3]},
                }
            }
        },
    }
    alias = {"passed": True, "result": {"opaque_future_field": [1, 2, 3]}}
    run = _run_cli(
        tmp_path,
        monkeypatch,
        [_outcome(provider=provider, aliases={"future_check": alias})],
        merge_captioning_selections=True,
    )

    assert run.exit_code is None
    assert run.cosmos.calls == [
        {
            "candidate_video_uri": str(run.video_path),
            "selected_variables": {"weather": "snowy", "lighting": "night"},
            "variable_options": {
                "weather": ["sunny", "snowy"],
                "lighting": ["day", "night"],
            },
        }
    ]
    metadata = json.loads(run.metadata_path.read_text())
    evaluation = json.loads(run.evaluation_path.read_text())
    assert metadata["cosmos_evaluator"]["request_id"] == 77
    assert metadata["cosmos_evaluator"]["operational_success"] is True
    assert metadata["cosmos_evaluator"]["quality_passed"] is True
    assert metadata["cosmos_evaluator"]["attempt"] == 1
    assert metadata["cosmos_evaluator"]["seed_used"] == 10
    assert metadata["cosmos_evaluator"]["checks"] == provider["checks"]
    assert metadata["future_check"] == alias
    assert evaluation["cosmos_evaluator"] == metadata["cosmos_evaluator"]
    assert evaluation["future_check"] == alias
    assert run.tracker.completed == {"metadata": {"totalSamplesProcessed": 1}}


@pytest.mark.parametrize(
    "checks",
    [
        {
            "vendor.new_checker": {
                "candidate": {"passed": True, "result": {"layout": "per-stream"}}
            }
        },
        {
            "vendor.new_checker": {
                "passed": True,
                "result": {"layout": "checker-level"},
            }
        },
    ],
)
def test_arbitrary_checker_result_layout_is_preserved_generically(
    tmp_path, monkeypatch, checks
):
    provider = {
        "request_id": 78,
        "status": "success",
        "version": "future",
        "commit_sha": "future-sha",
        "stream_key": "candidate",
        "checks": checks,
    }
    run = _run_cli(tmp_path, monkeypatch, [_outcome(provider=provider)])

    assert run.exit_code is None
    metadata = json.loads(run.metadata_path.read_text())
    assert metadata["cosmos_evaluator"]["checks"] == checks
    assert len(run.cosmos.calls) == 1


@pytest.mark.parametrize(
    ("strict", "retain_failures", "expected_exit", "artifacts_retained"),
    [
        (False, False, None, True),
        (True, True, 1, True),
        (True, False, 1, False),
    ],
)
def test_quality_failure_obeys_strict_and_retention(
    tmp_path,
    monkeypatch,
    caplog,
    strict,
    retain_failures,
    expected_exit,
    artifacts_retained,
):
    with caplog.at_level(logging.WARNING):
        run = _run_cli(
            tmp_path,
            monkeypatch,
            [_outcome(False)],
            strict=strict,
            retain_failures=retain_failures,
        )

    assert run.exit_code == expected_exit
    assert "failed checker(s): arbitrary.future.checker" in caplog.text
    assert len(run.generation_seeds) == 1
    assert len(run.cosmos.calls) == 1
    assert run.video_path.exists() is artifacts_retained
    assert run.caption_path.exists() is artifacts_retained
    assert run.metadata_path.exists() is artifacts_retained
    assert run.evaluation_path.exists() is artifacts_retained
    if artifacts_retained:
        metadata = json.loads(run.metadata_path.read_text())
        assert metadata["cosmos_evaluator"]["quality_passed"] is False
        assert metadata["cosmos_evaluator"]["operational_success"] is True


def test_quality_failure_reuses_outer_retry_and_increments_seed(tmp_path, monkeypatch):
    run = _run_cli(
        tmp_path,
        monkeypatch,
        [_outcome(False), _outcome(True)],
        retry=1,
    )

    assert run.exit_code is None
    assert run.generation_seeds == [10, 11]
    assert len(run.cosmos.calls) == 2
    assert run.generator.seed == 10
    metadata = json.loads(run.metadata_path.read_text())
    assert metadata["cosmos_evaluator"]["quality_passed"] is True
    assert metadata["cosmos_evaluator"]["attempt"] == 2
    assert metadata["cosmos_evaluator"]["seed_used"] == 11


def test_new_candidate_does_not_retain_prior_cosmos_state(tmp_path, monkeypatch):
    stale_alias = {"passed": False, "result": {"source": "attempt-1"}}
    run = _run_cli(
        tmp_path,
        monkeypatch,
        [_outcome(False, aliases={"future_check": stale_alias})],
        retry=1,
        built_in_passed=[True, False],
    )

    assert run.exit_code == 1
    assert run.generation_seeds == [10, 11]
    assert len(run.cosmos.calls) == 1
    assert run.video_path.read_bytes() == b"mock-candidate-11"
    metadata = json.loads(run.metadata_path.read_text())
    evaluation = json.loads(run.evaluation_path.read_text())
    assert metadata["attribute_verification"]["attempt"] == 2
    assert metadata["attribute_verification"]["seed_used"] == 11
    assert "cosmos_evaluator" not in metadata
    assert "future_check" not in metadata
    assert "cosmos_evaluator" not in evaluation
    assert "future_check" not in evaluation


def test_failed_later_generation_preserves_retained_candidate_cosmos_state(
    tmp_path, monkeypatch
):
    run = _run_cli(
        tmp_path,
        monkeypatch,
        [_outcome(False)],
        retry=1,
        generation_successes=[True, False],
    )

    assert run.exit_code == 1
    assert run.generation_seeds == [10, 11]
    assert len(run.cosmos.calls) == 1
    assert run.video_path.read_bytes() == b"mock-candidate-10"
    metadata = json.loads(run.metadata_path.read_text())
    evaluation = json.loads(run.evaluation_path.read_text())
    assert metadata["cosmos_evaluator"]["attempt"] == 1
    assert metadata["cosmos_evaluator"]["seed_used"] == 10
    assert evaluation["cosmos_evaluator"] == metadata["cosmos_evaluator"]


def test_explicit_generation_outcomes_fail_when_exhausted(tmp_path, monkeypatch):
    with pytest.raises(
        pytest.fail.Exception, match="Unexpected extra call to generate_with_retries"
    ):
        _run_cli(
            tmp_path,
            monkeypatch,
            [_outcome(False), _outcome(True)],
            retry=1,
            generation_successes=[True],
        )


def test_service_failure_stops_without_generating_another_candidate(
    tmp_path, monkeypatch, caplog
):
    with caplog.at_level(logging.ERROR):
        run = _run_cli(
            tmp_path,
            monkeypatch,
            [
                _outcome(
                    operational_success=False,
                    provider={
                        "request_id": 81,
                        "status": "service_failure",
                        "checks": {
                            "metropolis.attribute_verification": {
                                "candidate": {"passed": False}
                            }
                        },
                        "error": "poll contract failed",
                    },
                    error="poll contract failed",
                )
            ],
            retry=2,
            strict=True,
            retain_failures=True,
        )

    assert run.exit_code == 1
    assert run.generation_seeds == [10]
    assert len(run.cosmos.calls) == 1
    metadata = json.loads(run.metadata_path.read_text())
    assert metadata["cosmos_evaluator"]["operational_success"] is False
    assert metadata["cosmos_evaluator"]["quality_passed"] is None
    assert metadata["cosmos_evaluator"]["error"] == "poll contract failed"
    assert "failed checker(s): metropolis.attribute_verification" in caplog.text


def test_service_failure_is_reported_but_not_fatal_when_strict_is_false(
    tmp_path, monkeypatch
):
    run = _run_cli(
        tmp_path,
        monkeypatch,
        [
            _outcome(
                operational_success=False,
                provider={
                    "request_id": 82,
                    "status": "service_failure",
                    "checks": {},
                },
                error="service unavailable",
            )
        ],
        retry=2,
        strict=False,
        retain_failures=False,
    )

    assert run.exit_code is None
    assert run.generation_seeds == [10]
    assert run.video_path.exists()
    metadata = json.loads(run.metadata_path.read_text())
    assert metadata["cosmos_evaluator"]["operational_success"] is False
    assert metadata["cosmos_evaluator"]["quality_passed"] is None


def test_observe_only_failure_does_not_reject_candidate(tmp_path, monkeypatch):
    provider = {
        "request_id": 83,
        "status": "success",
        "version": "1.1.0",
        "commit_sha": "observe-sha",
        "stream_key": "candidate",
        "checks": {
            "vendor.observe_only": {
                "candidate": {
                    "passed": False,
                    "result": {"passed": False, "detail": "observed"},
                }
            }
        },
    }
    # The transport/parser computes the aggregate quality verdict from gating
    # checks. The CLI must not reinterpret an opaque observe-only result.
    run = _run_cli(tmp_path, monkeypatch, [_outcome(True, provider=provider)])

    assert run.exit_code is None
    assert run.generation_seeds == [10]
    metadata = json.loads(run.metadata_path.read_text())
    assert metadata["cosmos_evaluator"]["quality_passed"] is True
    assert (
        metadata["cosmos_evaluator"]["checks"]["vendor.observe_only"]["candidate"][
            "passed"
        ]
        is False
    )


@pytest.mark.parametrize(
    ("built_in_passed", "cosmos_passed", "expected_exit", "cosmos_calls"),
    [
        (False, True, 1, 0),
        (True, False, 1, 1),
        (True, True, None, 1),
    ],
)
def test_builtin_and_cosmos_gates_are_conjoined(
    tmp_path,
    monkeypatch,
    built_in_passed,
    cosmos_passed,
    expected_exit,
    cosmos_calls,
):
    run = _run_cli(
        tmp_path,
        monkeypatch,
        [_outcome(cosmos_passed)],
        built_in_passed=built_in_passed,
    )

    assert run.exit_code == expected_exit
    assert len(run.cosmos.calls) == cosmos_calls
    assert len(run.attribute_verifier.calls) == 1


def test_disabled_cosmos_pipeline_makes_no_external_calls(tmp_path, monkeypatch):
    run = _run_cli(tmp_path, monkeypatch, [], cosmos_enabled=False)

    assert run.exit_code is None
    assert run.cosmos is None
    assert "preflight" not in run.events
    assert "cosmos_evaluate" not in run.events
    metadata = json.loads(run.metadata_path.read_text())
    assert "cosmos_evaluator" not in metadata


def test_builtin_initializer_keeps_four_value_interface_for_cosmos_entry():
    result = cli._init_evaluators(
        {
            "evaluators": [
                {
                    "cosmos_evaluator": {
                        "enabled": True,
                        "endpoint_id": "unused-by-builtins",
                    }
                }
            ]
        },
        LOGGER,
    )

    assert result == (None, None, None, None)
    assert len(result) == 4


def test_environment_diagnostics_include_key_name_but_never_value(
    monkeypatch,
):
    logger = Mock()
    monkeypatch.delenv("IMAGE_EDIT_API_KEY", raising=False)
    cli.validate_environment(logger)
    logger.debug.assert_any_call("Environment variable IMAGE_EDIT_API_KEY is not set")

    logger.reset_mock()
    monkeypatch.setenv("IMAGE_EDIT_API_KEY", "super-secret-key")
    cli.validate_environment(logger)
    assert "super-secret-key" not in str(logger.mock_calls)
