# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Offline contract tests for the generic Cosmos Evaluator client."""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy

import pytest
import requests
from verification.cosmos_evaluator import (
    CosmosEvaluatorClient,
    CosmosEvaluatorInitializationError,
    CosmosEvaluatorServiceError,
    redact_sensitive_data,
    resolve_stream_key,
)

TIMESTAMP = "2026-08-30T00:00:00+00:00"


class StubResponse:
    def __init__(self, status_code, payload):
        self.status_code = status_code
        self._payload = payload

    def json(self):
        if isinstance(self._payload, Exception):
            raise self._payload
        return deepcopy(self._payload)


class StubSession:
    def __init__(self, *, get=(), post=()):
        self.get_responses = list(get)
        self.post_responses = list(post)
        self.get_calls = []
        self.post_calls = []
        self.events = []

    def get(self, url, **kwargs):
        self.events.append("get")
        self.get_calls.append((url, deepcopy(kwargs)))
        response = self.get_responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response

    def post(self, url, **kwargs):
        self.events.append("post")
        self.post_calls.append((url, deepcopy(kwargs)))
        response = self.post_responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response

    def close(self):
        self.events.append("close")


class FakeClock:
    def __init__(self):
        self.now = 0.0
        self.sleeps = []

    def monotonic(self):
        return self.now

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.now += seconds


def envelope(data, *, success=True):
    return {"success": success, "data": data, "timestamp": TIMESTAMP}


def response(status, data):
    return StubResponse(status, envelope(data))


def endpoint(**overrides):
    value = {
        "id": "cosmos",
        "role": "evaluator",
        "url": "https://arbitrator.example.test/root/",
        "adapter": "cosmos.evaluator.arbitrator",
        "timeout": 30,
    }
    value.update(overrides)
    return value


def evaluator(**overrides):
    value = {
        "enabled": True,
        "endpoint_id": "cosmos",
        "poll_interval_seconds": 0.25,
        "output_storage_prefix": "evaluations/run-1/",
        "checks": [
            {
                "name": "example.checker",
                "gate": True,
                "metadata_key": "example_result",
            }
        ],
        "inputs": {},
        "config": {},
    }
    value.update(overrides)
    return value


def execution_snapshot(
    response_json,
    *,
    status="success",
    terminal=True,
    request_id=41,
    checker_items=None,
):
    return {
        "execution": {
            "request_id": request_id,
            "status": status,
            "is_terminal": terminal,
            "version": "1.1.0",
            "commit_sha": "e901afe",
            "response_json": response_json,
        },
        "checkers": {
            "completed": 1 if terminal else 0,
            "total": 1,
            "items": checker_items
            if checker_items is not None
            else [
                {
                    "checker_name": "example.checker",
                    "camera_name": "candidate",
                    "status": "success" if terminal else "in_progress",
                }
            ],
        },
    }


def outcome_session(result, **snapshot_overrides):
    accepted = response(202, {"request_id": 41})
    snapshot = execution_snapshot(result, **snapshot_overrides)
    return StubSession(post=[accepted], get=[response(200, snapshot)])


def client(*, session=None, endpoint_value=None, evaluator_value=None, clock=None):
    clock = clock or FakeClock()
    return CosmosEvaluatorClient(
        [endpoint_value or endpoint()],
        evaluator_value or evaluator(),
        session=session or StubSession(),
        monotonic=clock.monotonic,
        sleep=clock.sleep,
    )


class TestInitializationAndPreflight:
    def test_bearer_token_is_read_from_named_environment_variable(self, monkeypatch):
        monkeypatch.setenv("COSMOS_TEST_KEY", "top-secret")
        session = StubSession()
        evaluator_client = client(
            session=session,
            endpoint_value=endpoint(api_key_env="COSMOS_TEST_KEY"),
        )

        assert evaluator_client.base_url == "https://arbitrator.example.test/root"
        assert evaluator_client._headers == {"Authorization": "Bearer top-secret"}

    def test_no_header_is_sent_without_key_configuration(self):
        evaluator_client = client()
        assert evaluator_client._headers == {}

    def test_explicit_unset_key_fails_without_a_key_value(self, monkeypatch):
        monkeypatch.delenv("COSMOS_MISSING_KEY", raising=False)
        with pytest.raises(
            CosmosEvaluatorInitializationError,
            match="COSMOS_MISSING_KEY.*unset",
        ) as error:
            client(endpoint_value=endpoint(api_key_env="COSMOS_MISSING_KEY"))
        assert "top-secret" not in str(error.value)

    @pytest.mark.parametrize(
        ("endpoint_override", "message"),
        [
            ({"role": "llm"}, "role 'evaluator'"),
            ({"adapter": "nim"}, "cosmos.evaluator.arbitrator"),
            ({"timeout": 0}, "timeout must be positive"),
            ({"url": "arbitrator.local"}, "absolute HTTP"),
            ({"url": "https://user:secret@host"}, "must not contain userinfo"),
        ],
    )
    def test_invalid_endpoint_contract_is_rejected(self, endpoint_override, message):
        with pytest.raises(CosmosEvaluatorInitializationError, match=message):
            client(endpoint_value=endpoint(**endpoint_override))

    def test_preflight_validates_all_routes_and_sends_auth(self, monkeypatch):
        monkeypatch.setenv("COSMOS_TEST_KEY", "top-secret")
        session = StubSession(
            get=[
                response(
                    200,
                    {
                        "status": "healthy",
                        "version": "1.1.0",
                        "git_sha": "e901afe",
                    },
                ),
                response(200, {"checkers": ["example.checker"]}),
                response(
                    200,
                    {"checkers": {"example.checker": {"deps": [], "priority": 0}}},
                ),
            ]
        )
        evaluator_client = client(
            session=session,
            endpoint_value=endpoint(api_key_env="COSMOS_TEST_KEY"),
        )

        result = evaluator_client.preflight()

        assert result.version == "1.1.0"
        assert result.commit_sha == "e901afe"
        assert result.healthy_checkers == ("example.checker",)
        assert [call[0] for call in session.get_calls] == [
            "https://arbitrator.example.test/root/health",
            "https://arbitrator.example.test/root/checkers",
            "https://arbitrator.example.test/root/dependency-graph",
        ]
        assert all(
            call[1]["headers"] == {"Authorization": "Bearer top-secret"}
            for call in session.get_calls
        )

    @pytest.mark.parametrize("missing_from", ["healthy", "graph"])
    def test_preflight_rejects_discovery_mismatch(self, missing_from):
        healthy = [] if missing_from == "healthy" else ["example.checker"]
        graph = {} if missing_from == "graph" else {"example.checker": {}}
        session = StubSession(
            get=[
                response(
                    200,
                    {"status": "healthy", "version": "1", "git_sha": "abc"},
                ),
                response(200, {"checkers": healthy}),
                response(200, {"checkers": graph}),
            ]
        )
        with pytest.raises(CosmosEvaluatorServiceError):
            client(session=session).preflight()

    def test_preflight_requires_standard_success_envelope(self):
        bad = StubResponse(200, {"success": True, "data": {"status": "healthy"}})
        with pytest.raises(CosmosEvaluatorServiceError, match="timestamp"):
            client(session=StubSession(get=[bad])).preflight()

    def test_preflight_retries_and_normalizes_other_request_exceptions(self):
        clock = FakeClock()
        session = StubSession(
            get=[requests.TooManyRedirects("redirect loop") for _ in range(3)]
        )

        with pytest.raises(
            CosmosEvaluatorServiceError,
            match=r"Cosmos Evaluator GET /health failed after 3 attempts",
        ):
            client(session=session, clock=clock).preflight()

        assert len(session.get_calls) == 3
        assert clock.sleeps == [0.5, 1.0]


class TestRequestBuildingAndStreamKeys:
    @pytest.mark.parametrize(
        ("candidate_uri", "configured_stream_key", "message"),
        [
            ("", None, "generated candidate URI is empty"),
            ("s3://bucket/candidate.mp4", "  ", "configured stream_key is empty"),
        ],
    )
    def test_public_stream_key_resolver_preserves_service_errors(
        self, candidate_uri, configured_stream_key, message
    ):
        with pytest.raises(CosmosEvaluatorServiceError, match=message):
            resolve_stream_key(candidate_uri, configured_stream_key)

    def test_exact_request_preserves_future_inputs_and_does_not_mutate_config(self):
        evaluator_value = evaluator(
            stream_key="  camera_front_wide_120fov  ",
            inputs={
                "original_video_urls": {
                    "camera_front_wide_120fov": "s3://bucket/original.mp4"
                },
                "future_input": {"nested": [1, 2]},
            },
            config={"configured": {"keep": True}},
        )
        original = deepcopy(evaluator_value)
        evaluator_client = client(evaluator_value=evaluator_value)

        stream_key, payload = evaluator_client.build_request(
            "s3://bucket/augmented.mp4?X-Amz-Signature=needed",
        )

        assert stream_key == "camera_front_wide_120fov"
        assert payload == {
            "inputs": {
                "original_video_urls": {
                    "camera_front_wide_120fov": "s3://bucket/original.mp4"
                },
                "future_input": {"nested": [1, 2]},
                "augmented_video_urls": {
                    "camera_front_wide_120fov": (
                        "s3://bucket/augmented.mp4?X-Amz-Signature=needed"
                    )
                },
                "config": {"configured": {"keep": True}},
            },
            "checks": ["example.checker"],
            "output_storage_prefix": "evaluations/run-1/",
        }
        assert evaluator_value == original

    def test_captioning_selection_merge_is_opt_in_and_runtime_wins(self):
        evaluator_value = evaluator(
            merge_captioning_selections=True,
            config={
                "selected_variables": {"weather": "snowy", "shared": "old"},
                "variable_options": {
                    "weather": ["sunny", "snowy"],
                    "shared": ["old"],
                },
            },
        )
        evaluator_client = client(evaluator_value=evaluator_value)

        _, payload = evaluator_client.build_request(
            "s3://bucket/candidate.mp4",
            selected_variables={"shared": "new", "lighting": "night"},
            variable_options={"shared": ["new"], "lighting": ["day", "night"]},
        )

        assert payload["inputs"]["config"]["selected_variables"] == {
            "weather": "snowy",
            "shared": "new",
            "lighting": "night",
        }
        assert payload["inputs"]["config"]["variable_options"] == {
            "weather": ["sunny", "snowy"],
            "shared": ["new"],
            "lighting": ["day", "night"],
        }
        assert evaluator_value["config"]["selected_variables"]["shared"] == "old"

    def test_captioning_values_are_ignored_without_opt_in(self):
        evaluator_client = client(evaluator_value=evaluator(config={"fixed": True}))
        _, payload = evaluator_client.build_request(
            "s3://bucket/candidate.mp4",
            selected_variables={"lighting": "night"},
            variable_options={"lighting": ["day", "night"]},
        )
        assert payload["inputs"]["config"] == {"fixed": True}

    @pytest.mark.parametrize(
        ("uri", "expected"),
        [
            ("s3://bucket/run/candidate.mp4", "candidate"),
            (
                "https://host/run/My%20Clip.final.mp4?token=secret#frame",
                "My_Clip.final",
            ),
            ("s3://bucket/run/%EF%BC%A1%EF%BC%A2%EF%BC%A3.mp4", "ABC"),
            ("s3://bucket/run/Caf%C3%A9.mp4", "Caf"),
        ],
    )
    def test_stream_key_algorithm(self, uri, expected):
        assert resolve_stream_key(uri) == expected

    def test_empty_normalized_stem_uses_query_free_digest(self):
        query_free = "https://host/run/%E9%9B%AA.mp4"
        expected_digest = hashlib.sha256(query_free.encode()).hexdigest()[:12]
        with_query = f"{query_free}?token=secret#frame"
        assert resolve_stream_key(with_query) == f"stream-{expected_digest}"

    def test_companion_stream_map_must_contain_the_resolved_key(self):
        evaluator_client = client(
            evaluator_value=evaluator(
                inputs={"original_video_urls": {"another": "s3://bucket/a.mp4"}}
            )
        )
        with pytest.raises(CosmosEvaluatorServiceError, match="lacks resolved"):
            evaluator_client.build_request("s3://bucket/candidate.mp4")


class TestSubmitPollAndResults:
    def test_each_submission_uses_fresh_owned_session_after_preflight(
        self, monkeypatch
    ):
        result = {
            "checker_results": {"example.checker": {"candidate": {"passed": True}}}
        }
        preflight_session = StubSession(
            get=[
                response(
                    200,
                    {"status": "healthy", "version": "1.1.0", "git_sha": "e901afe"},
                ),
                response(200, {"checkers": ["example.checker"]}),
                response(
                    200,
                    {"checkers": {"example.checker": {"deps": [], "priority": 0}}},
                ),
            ]
        )
        first_submission_session = outcome_session(result)
        second_submission_session = outcome_session(result)
        sessions = iter(
            [
                preflight_session,
                first_submission_session,
                second_submission_session,
            ]
        )
        monkeypatch.setattr(requests, "Session", lambda: next(sessions))
        clock = FakeClock()
        evaluator_client = CosmosEvaluatorClient(
            [endpoint()],
            evaluator(),
            monotonic=clock.monotonic,
            sleep=clock.sleep,
        )

        evaluator_client.preflight()
        first_outcome = evaluator_client.evaluate("s3://bucket/candidate.mp4")
        second_outcome = evaluator_client.evaluate("s3://bucket/candidate.mp4")

        assert first_outcome.operational_success is True
        assert second_outcome.operational_success is True
        assert preflight_session.events == ["get", "get", "get", "close"]
        assert first_submission_session.events == ["post", "get", "close"]
        assert second_submission_session.events == ["post", "get"]

    @pytest.mark.parametrize("as_string", [True, False])
    def test_terminal_pass_supports_pinned_string_and_mapping_response(self, as_string):
        checker_results = {
            "checker_results": {
                "example.checker": {
                    "candidate": {
                        "passed": True,
                        "score": 0.91,
                        "artifact": "https://user:pw@host/a?X-Amz-Signature=secret",
                    }
                }
            }
        }
        wire_result = json.dumps(checker_results) if as_string else checker_results
        session = outcome_session(wire_result)

        outcome = client(session=session).evaluate("s3://bucket/candidate.mp4")

        assert outcome.operational_success is True
        assert outcome.quality_passed is True
        assert outcome.provider_metadata["request_id"] == 41
        assert outcome.provider_metadata["status"] == "success"
        assert outcome.provider_metadata["version"] == "1.1.0"
        assert outcome.provider_metadata["commit_sha"] == "e901afe"
        assert outcome.provider_metadata["stream_key"] == "candidate"
        check_view = outcome.provider_metadata["checks"]["example.checker"]["candidate"]
        assert check_view["passed"] is True
        assert check_view["result"]["score"] == 0.91
        assert check_view["result"]["artifact"] == "https://host/a"
        assert outcome.compatibility_aliases["example_result"] == check_view
        assert len(session.post_calls) == 1
        assert session.post_calls[0][1]["json"]["inputs"]["augmented_video_urls"] == {
            "candidate": "s3://bucket/candidate.mp4"
        }

    def test_checker_level_result_is_supported_and_normalized_to_stream(self):
        result = {
            "checker_results": {"example.checker": {"passed": True, "summary": "flat"}}
        }
        outcome = client(session=outcome_session(result)).evaluate(
            "s3://bucket/candidate.mp4"
        )
        normalized = outcome.provider_metadata["checks"]["example.checker"][
            "candidate"
        ]["result"]
        assert outcome.quality_passed is True
        assert normalized == {"passed": True, "summary": "flat"}

    def test_quality_false_is_not_an_operational_failure(self):
        result = {
            "checker_results": {
                "example.checker": {"candidate": {"passed": False, "score": 0.1}}
            }
        }
        outcome = client(session=outcome_session(result)).evaluate(
            "s3://bucket/candidate.mp4"
        )
        assert outcome.operational_success is True
        assert outcome.quality_passed is False
        assert outcome.is_quality_failure is True
        assert outcome.error is None
        assert outcome.failed_checkers == ("example.checker",)

    def test_all_errored_checks_are_an_operational_failure(self):
        checker_name = "metropolis.attribute_verification"
        checker_result = {
            "passed": False,
            "summary": {
                "total_checks": 4,
                "passed_checks": 0,
                "failed_checks": 0,
                "errored_checks": 4,
            },
            "checks": [{"variable": "color", "error": "VLM unavailable"}],
        }
        result = {"checker_results": {checker_name: {"candidate": checker_result}}}

        outcome = client(
            session=outcome_session(result),
            evaluator_value=evaluator(checks=[{"name": checker_name, "gate": True}]),
        ).evaluate("s3://bucket/candidate.mp4")

        assert outcome.operational_success is False
        assert outcome.quality_passed is None
        assert outcome.is_service_failure is True
        assert outcome.failed_checkers == (checker_name,)
        assert "all 4 checks errored" in outcome.error
        assert outcome.provider_metadata["status"] == "success"
        assert (
            outcome.provider_metadata["checks"][checker_name]["candidate"]["result"]
            == checker_result
        )

    def test_zero_checks_are_not_reported_as_all_errored(self):
        result = {
            "checker_results": {
                "example.checker": {
                    "candidate": {
                        "passed": False,
                        "summary": {"total_checks": 0, "errored_checks": 0},
                    }
                }
            }
        }

        outcome = client(session=outcome_session(result)).evaluate(
            "s3://bucket/candidate.mp4"
        )

        assert outcome.operational_success is True
        assert outcome.quality_passed is False

    def test_observe_only_result_needs_no_passed_field(self):
        evaluator_client = client(
            session=outcome_session(
                {"checker_results": {"example.checker": {"score": 0.4}}}
            ),
            evaluator_value=evaluator(
                checks=[{"name": "example.checker", "gate": False}]
            ),
        )
        outcome = evaluator_client.evaluate("s3://bucket/candidate.mp4")
        assert outcome.operational_success is True
        assert outcome.quality_passed is True
        assert outcome.provider_metadata["checks"]["example.checker"]["candidate"][
            "result"
        ] == {"score": 0.4}

    @pytest.mark.parametrize(
        "result",
        [
            {"checker_results": {}},
            {"checker_results": {"example.checker": {"candidate": {"score": 1}}}},
            {"checker_results": {"example.checker": {"candidate": {"passed": 1}}}},
            {
                "checker_results": {
                    "example.checker": {
                        "candidate": {"error": "Skipped: dependency failed"}
                    }
                }
            },
        ],
    )
    def test_missing_skipped_or_invalid_gating_result_is_service_failure(self, result):
        outcome = client(session=outcome_session(result)).evaluate(
            "s3://bucket/candidate.mp4"
        )
        assert outcome.is_service_failure is True
        assert outcome.quality_passed is None
        assert outcome.compatibility_aliases == {}

    def test_failed_parent_execution_is_service_failure(self):
        session = outcome_session(
            {"error": "checker failed"},
            status="fail",
            checker_items=[{"checker_name": "example.checker", "status": "fail"}],
        )
        outcome = client(session=session).evaluate("s3://bucket/candidate.mp4")
        assert outcome.is_service_failure is True
        assert outcome.provider_metadata["status"] == "fail"
        assert outcome.provider_metadata["version"] == "1.1.0"

    def test_skipped_child_row_is_service_failure_even_with_parent_success(self):
        result = {
            "checker_results": {"example.checker": {"candidate": {"passed": True}}}
        }
        session = outcome_session(
            result,
            checker_items=[{"checker_name": "example.checker", "status": "skipped"}],
        )
        outcome = client(session=session).evaluate("s3://bucket/candidate.mp4")
        assert outcome.is_service_failure is True
        assert "skipped" in outcome.error

    def test_poll_retries_transient_get_failures_within_one_deadline(self):
        clock = FakeClock()
        result = {
            "checker_results": {"example.checker": {"candidate": {"passed": True}}}
        }
        session = StubSession(
            post=[response(202, {"request_id": 41})],
            get=[
                StubResponse(503, envelope({})),
                requests.ConnectionError("temporary"),
                response(200, execution_snapshot(result)),
            ],
        )
        outcome = client(session=session, clock=clock).evaluate(
            "s3://bucket/candidate.mp4"
        )
        assert outcome.operational_success is True
        assert len(session.get_calls) == 3
        assert clock.sleeps == [0.5, 1.0]

    def test_ambiguous_post_timeout_is_never_retried(self):
        session = StubSession(post=[requests.Timeout("ambiguous secret body")])
        outcome = client(session=session).evaluate("s3://bucket/candidate.mp4")
        assert outcome.is_service_failure is True
        assert len(session.post_calls) == 1
        assert session.get_calls == []
        assert "not resubmitted" in outcome.error
        assert outcome.provider_metadata["error_code"] == "ambiguous_post"
        assert outcome.provider_metadata["request_id"] is None

    @pytest.mark.parametrize(
        "accepted_response",
        [
            StubResponse(202, ValueError("invalid json")),
            response(202, {}),
            response(202, {"request_id": None}),
        ],
    )
    def test_accepted_post_without_usable_request_id_is_ambiguous(
        self, accepted_response
    ):
        outcome = client(session=StubSession(post=[accepted_response])).evaluate(
            "s3://bucket/candidate.mp4"
        )

        assert outcome.is_service_failure is True
        assert outcome.provider_metadata["error_code"] == "ambiguous_post"
        assert outcome.provider_metadata["request_id"] is None

    def test_rejected_post_is_a_clear_service_failure(self):
        outcome = client(session=StubSession(post=[response(400, {})])).evaluate(
            "s3://bucket/candidate.mp4"
        )

        assert outcome.is_service_failure is True
        assert outcome.provider_metadata["error_code"] == "service_failure"
        assert outcome.provider_metadata["request_id"] is None

    def test_total_timeout_covers_submission_polling_and_sleep(self):
        clock = FakeClock()
        pending = execution_snapshot(None, status="in_progress", terminal=False)
        session = StubSession(
            post=[response(202, {"request_id": 41})],
            get=[response(200, pending)],
        )
        evaluator_client = client(
            session=session,
            endpoint_value=endpoint(timeout=0.2),
            evaluator_value=evaluator(poll_interval_seconds=1),
            clock=clock,
        )
        outcome = evaluator_client.evaluate("s3://bucket/candidate.mp4")
        assert outcome.is_service_failure is True
        assert "total timeout" in outcome.error
        assert len(session.post_calls) == 1
        assert len(session.get_calls) == 1


def test_recursive_redaction_removes_credentials_userinfo_queries_and_key_value():
    secret = "actual-api-key"
    original = {
        "authorization": f"Bearer {secret}",
        "nested": [
            {
                "url": (
                    "https://user:password@example.test/a.mp4"
                    "?X-Amz-Credential=name&X-Amz-Signature=signed#frame"
                ),
                "message": f"failed at https://user:pw@host/b?token={secret}",
                "opaque": f"prefix-{secret}-suffix",
            }
        ],
    }

    redacted = redact_sensitive_data(original, secrets=(secret,))

    serialized = json.dumps(redacted)
    assert secret not in serialized
    assert "password" not in serialized
    assert "X-Amz" not in serialized
    assert "?" not in serialized
    assert redacted["authorization"] == "[REDACTED]"
    assert redacted["nested"][0]["url"] == "https://example.test/a.mp4"
    assert redacted["nested"][0]["message"] == "failed at https://host/b"
