# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Generic client for the external Cosmos Evaluator Arbitrator service.

The client deliberately knows nothing about individual checker algorithms.  It
discovers checker names from Arbitrator, forwards their opaque inputs and
configuration, and interprets only the common top-level ``passed: bool`` field
for checks configured as quality gates.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from collections.abc import Callable, Mapping, MutableMapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlsplit, urlunsplit

import requests
from aug_utils.stream_keys import resolve_stream_key as _resolve_stream_key

ARBITRATOR_ADAPTER = "cosmos.evaluator.arbitrator"
DEFAULT_TIMEOUT_SECONDS = 1800.0
MAX_GET_ATTEMPTS = 3
_GET_BACKOFF_SECONDS = 0.5
_TRANSIENT_GET_STATUSES = frozenset({429, *range(500, 600)})
_FAILED_CHECKER_STATUSES = frozenset({"fail", "failed", "timeout", "skipped"})
_URL_IN_TEXT = re.compile(r"(?P<url>[A-Za-z][A-Za-z0-9+.-]*://[^\s\"'<>]+)")
_CREDENTIAL_KEYS = frozenset(
    {
        "apikey",
        "authorization",
        "accesstoken",
        "refreshtoken",
        "password",
        "passwd",
        "secret",
        "secretkey",
        "accesskey",
        "credential",
        "credentials",
        "xapikey",
    }
)
_REDACTED = "[REDACTED]"


class CosmosEvaluatorError(RuntimeError):
    """Base error for external Cosmos evaluation."""


class CosmosEvaluatorInitializationError(CosmosEvaluatorError):
    """The evaluator endpoint or local client configuration is invalid."""


class CosmosEvaluatorServiceError(CosmosEvaluatorError):
    """The external service or its response contract failed."""


class CosmosEvaluatorAmbiguousPostError(CosmosEvaluatorServiceError):
    """A submission may have reached Arbitrator but cannot be tracked safely."""


@dataclass(frozen=True)
class CosmosCheckerSpec:
    """One checker selected through evaluator configuration."""

    name: str
    gate: bool = True
    metadata_key: str | None = None


@dataclass(frozen=True)
class CosmosPreflightResult:
    """Validated discovery information returned by :meth:`preflight`."""

    version: str
    commit_sha: str
    healthy_checkers: tuple[str, ...]
    dependency_graph_checkers: tuple[str, ...]


@dataclass
class CosmosEvaluationOutcome:
    """Result of one submit-and-poll evaluation.

    ``operational_success`` distinguishes an external service/contract failure
    from a completed quality verdict.  ``quality_passed`` is ``None`` only for
    an operational failure; observe-only checks produce a vacuous ``True``
    quality verdict while their opaque results remain in metadata.
    """

    operational_success: bool
    quality_passed: bool | None
    provider_metadata: dict[str, Any]
    compatibility_aliases: dict[str, Any]
    error: str | None = None

    @property
    def is_quality_failure(self) -> bool:
        """Return whether checker execution succeeded but a gate rejected it."""

        return self.operational_success and self.quality_passed is False

    @property
    def is_service_failure(self) -> bool:
        """Return whether transport, execution, or result validation failed."""

        return not self.operational_success

    @property
    def failed_checkers(self) -> tuple[str, ...]:
        """Return checker names whose normalized verdict is false."""

        checks = self.provider_metadata.get("checks")
        if not isinstance(checks, Mapping):
            return ()
        failed = []
        for name, value in checks.items():
            if not isinstance(name, str) or not isinstance(value, Mapping):
                continue
            verdicts = (value, *value.values())
            if any(
                isinstance(verdict, Mapping) and verdict.get("passed") is False
                for verdict in verdicts
            ):
                failed.append(name)
        return tuple(failed)


def _as_mapping(value: Any, label: str) -> dict[str, Any]:
    """Return a deep-copied plain mapping from a schema model or mapping."""

    if hasattr(value, "model_dump"):
        value = value.model_dump(exclude_none=True)
    if not isinstance(value, Mapping):
        raise CosmosEvaluatorInitializationError(f"{label} must be a mapping")
    return deepcopy(dict(value))


def _field(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def resolve_stream_key(
    candidate_uri: str, configured_stream_key: str | None = None
) -> str:
    """Resolve the one stable stream key used throughout an evaluation request."""

    try:
        return _resolve_stream_key(candidate_uri, configured_stream_key)
    except ValueError as exc:
        raise CosmosEvaluatorServiceError(str(exc)) from exc


def _credential_key(key: Any) -> bool:
    if not isinstance(key, str):
        return False
    normalized = re.sub(r"[^a-z0-9]", "", key.lower())
    return normalized in _CREDENTIAL_KEYS or normalized.endswith("apikey")


def _redact_url(url: str) -> str:
    """Remove URL userinfo, query, and fragment without exposing parse errors."""

    try:
        parsed = urlsplit(url)
        if not parsed.scheme or not parsed.netloc:
            return url

        hostname = parsed.hostname
        if not hostname:
            return urlunsplit((parsed.scheme, "", parsed.path, "", ""))
        host = f"[{hostname}]" if ":" in hostname else hostname
        try:
            port = parsed.port
        except ValueError:
            port = None
        if port is not None:
            host = f"{host}:{port}"
        return urlunsplit((parsed.scheme, host, parsed.path, "", ""))
    except (TypeError, ValueError):
        return url


def _redact_string(value: str, secrets: Sequence[str]) -> str:
    redacted = value
    for secret in secrets:
        if secret:
            redacted = redacted.replace(secret, _REDACTED)

    parsed = urlsplit(redacted)
    if parsed.scheme and parsed.netloc and redacted == parsed.geturl():
        return _redact_url(redacted)

    return _URL_IN_TEXT.sub(lambda match: _redact_url(match.group("url")), redacted)


def redact_sensitive_data(value: Any, *, secrets: Sequence[str] = ()) -> Any:
    """Deep-copy persistable data while removing credentials and signed URLs.

    Outbound requests intentionally retain their media URL queries.  This
    helper is applied only to diagnostics and metadata returned to callers.
    """

    if isinstance(value, Mapping):
        redacted: dict[Any, Any] = {}
        for key, item in value.items():
            if _credential_key(key):
                redacted[key] = _REDACTED
            else:
                redacted[key] = redact_sensitive_data(item, secrets=secrets)
        return redacted
    if isinstance(value, list):
        return [redact_sensitive_data(item, secrets=secrets) for item in value]
    if isinstance(value, tuple):
        return tuple(redact_sensitive_data(item, secrets=secrets) for item in value)
    if isinstance(value, set):
        return {redact_sensitive_data(item, secrets=secrets) for item in value}
    if isinstance(value, str):
        return _redact_string(value, secrets)
    return deepcopy(value)


class CosmosEvaluatorClient:
    """Synchronous, checker-agnostic client for one Arbitrator endpoint."""

    def __init__(
        self,
        endpoints: Sequence[Any],
        evaluator_config: Any,
        *,
        session: Any | None = None,
        logger: logging.Logger | None = None,
        monotonic: Callable[[], float] | None = None,
        sleep: Callable[[float], None] | None = None,
    ) -> None:
        config = _as_mapping(evaluator_config, "cosmos_evaluator configuration")
        if not config.get("enabled", True):
            raise CosmosEvaluatorInitializationError(
                "cannot initialize a disabled cosmos_evaluator"
            )

        endpoint_id = config.get("endpoint_id")
        if not isinstance(endpoint_id, str) or not endpoint_id.strip():
            raise CosmosEvaluatorInitializationError(
                "cosmos_evaluator.endpoint_id must be non-empty"
            )
        endpoint_id = endpoint_id.strip()
        matches = [
            endpoint for endpoint in endpoints if _field(endpoint, "id") == endpoint_id
        ]
        if not matches:
            raise CosmosEvaluatorInitializationError(
                f"cosmos_evaluator endpoint_id {endpoint_id!r} was not found"
            )
        if len(matches) > 1:
            raise CosmosEvaluatorInitializationError(
                f"cosmos_evaluator endpoint_id {endpoint_id!r} is not unique"
            )
        endpoint = matches[0]
        if _field(endpoint, "role") != "evaluator":
            raise CosmosEvaluatorInitializationError(
                f"cosmos_evaluator endpoint_id {endpoint_id!r} must have role 'evaluator'"
            )
        if _field(endpoint, "adapter") != ARBITRATOR_ADAPTER:
            raise CosmosEvaluatorInitializationError(
                f"cosmos_evaluator endpoint_id {endpoint_id!r} must use adapter "
                f"{ARBITRATOR_ADAPTER!r}"
            )

        self.base_url = self._normalize_base_url(_field(endpoint, "url"))
        timeout_value = _field(endpoint, "timeout", None)
        self.timeout = (
            DEFAULT_TIMEOUT_SECONDS if timeout_value is None else float(timeout_value)
        )
        if self.timeout <= 0:
            raise CosmosEvaluatorInitializationError(
                "cosmos_evaluator endpoint timeout must be positive"
            )

        key_env = _field(endpoint, "api_key_env")
        if key_env is not None:
            if not isinstance(key_env, str) or not key_env.strip():
                raise CosmosEvaluatorInitializationError(
                    "cosmos_evaluator api_key_env must be a non-empty environment variable name"
                )
            key_env = key_env.strip()
            api_key = os.getenv(key_env)
            if not api_key:
                raise CosmosEvaluatorInitializationError(
                    f"API key environment variable {key_env!r} is unset"
                )
        else:
            api_key = None

        self._headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self._secrets = (api_key,) if api_key else ()
        self._owns_session = session is None
        self._session = requests.Session() if session is None else session
        self._logger = logger or logging.getLogger(__name__)
        self._monotonic = monotonic or time.monotonic
        self._sleep = sleep or time.sleep

        poll_interval = float(config.get("poll_interval_seconds", 5.0))
        if poll_interval <= 0:
            raise CosmosEvaluatorInitializationError(
                "cosmos_evaluator poll_interval_seconds must be positive"
            )
        self.poll_interval_seconds = poll_interval
        self.configured_stream_key = config.get("stream_key")
        if self.configured_stream_key is not None:
            self.configured_stream_key = str(self.configured_stream_key).strip()
            if not self.configured_stream_key:
                raise CosmosEvaluatorInitializationError(
                    "cosmos_evaluator stream_key must be non-empty"
                )

        output_prefix = config.get("output_storage_prefix")
        if not isinstance(output_prefix, str) or not output_prefix.strip():
            raise CosmosEvaluatorInitializationError(
                "enabled cosmos_evaluator requires output_storage_prefix"
            )
        self.output_storage_prefix = output_prefix.strip()
        self.additional_inputs = deepcopy(config.get("inputs") or {})
        if not isinstance(self.additional_inputs, MutableMapping):
            raise CosmosEvaluatorInitializationError(
                "cosmos_evaluator.inputs must be a mapping"
            )
        if "augmented_video_urls" in self.additional_inputs:
            raise CosmosEvaluatorInitializationError(
                "cosmos_evaluator.inputs reserves augmented_video_urls for runtime"
            )
        self.shared_config = deepcopy(config.get("config") or {})
        if not isinstance(self.shared_config, MutableMapping):
            raise CosmosEvaluatorInitializationError(
                "cosmos_evaluator.config must be a mapping"
            )
        self.merge_captioning_selections = bool(
            config.get("merge_captioning_selections", False)
        )
        self.checks = self._parse_checks(config.get("checks"))
        self._preflight_result: CosmosPreflightResult | None = None

    @classmethod
    def from_config(
        cls,
        endpoints: Sequence[Any],
        evaluator_config: Any,
        **kwargs: Any,
    ) -> CosmosEvaluatorClient:
        """Construct a client from endpoint-registry and evaluator config values."""

        return cls(endpoints, evaluator_config, **kwargs)

    @staticmethod
    def _normalize_base_url(value: Any) -> str:
        if not isinstance(value, str) or not value.strip():
            raise CosmosEvaluatorInitializationError(
                "cosmos_evaluator endpoint URL must be non-empty"
            )
        base_url = value.strip().rstrip("/")
        parsed = urlsplit(base_url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise CosmosEvaluatorInitializationError(
                "cosmos_evaluator endpoint URL must be an absolute HTTP(S) URL"
            )
        if parsed.username is not None or parsed.password is not None:
            raise CosmosEvaluatorInitializationError(
                "cosmos_evaluator endpoint URL must not contain userinfo"
            )
        if parsed.query or parsed.fragment:
            raise CosmosEvaluatorInitializationError(
                "cosmos_evaluator endpoint URL must not contain a query or fragment"
            )
        return base_url

    @staticmethod
    def _parse_checks(value: Any) -> tuple[CosmosCheckerSpec, ...]:
        if (
            not isinstance(value, Sequence)
            or isinstance(value, (str, bytes))
            or not value
        ):
            raise CosmosEvaluatorInitializationError(
                "enabled cosmos_evaluator requires at least one check"
            )
        checks: list[CosmosCheckerSpec] = []
        names: set[str] = set()
        aliases: set[str] = set()
        for item in value:
            item_map = _as_mapping(item, "cosmos_evaluator check")
            name = item_map.get("name")
            if not isinstance(name, str) or not name.strip():
                raise CosmosEvaluatorInitializationError(
                    "cosmos_evaluator check name must be non-empty"
                )
            name = name.strip()
            if name in names:
                raise CosmosEvaluatorInitializationError(
                    f"duplicate cosmos_evaluator checker name {name!r}"
                )
            names.add(name)

            metadata_key = item_map.get("metadata_key")
            if metadata_key is not None:
                if not isinstance(metadata_key, str) or not metadata_key.strip():
                    raise CosmosEvaluatorInitializationError(
                        "cosmos_evaluator metadata_key must be non-empty"
                    )
                metadata_key = metadata_key.strip()
                if metadata_key in aliases:
                    raise CosmosEvaluatorInitializationError(
                        f"duplicate cosmos_evaluator metadata_key {metadata_key!r}"
                    )
                aliases.add(metadata_key)
            checks.append(
                CosmosCheckerSpec(
                    name=name,
                    gate=bool(item_map.get("gate", True)),
                    metadata_key=metadata_key,
                )
            )
        return tuple(checks)

    def preflight(self) -> CosmosPreflightResult:
        """Validate service health and discovery before candidate generation."""

        deadline = self._monotonic() + self.timeout
        health = self._get_success_data("/health", deadline)
        if health.get("status") != "healthy":
            raise CosmosEvaluatorServiceError(
                "Cosmos Evaluator /health did not report status 'healthy'"
            )
        version = health.get("version")
        commit_sha = health.get("git_sha")
        if not isinstance(version, str) or not isinstance(commit_sha, str):
            raise CosmosEvaluatorServiceError(
                "Cosmos Evaluator /health omitted version or git_sha"
            )

        checker_data = self._get_success_data("/checkers", deadline)
        healthy = checker_data.get("checkers")
        if not isinstance(healthy, list) or not all(
            isinstance(name, str) for name in healthy
        ):
            raise CosmosEvaluatorServiceError(
                "Cosmos Evaluator /checkers returned an invalid checker list"
            )

        graph_data = self._get_success_data("/dependency-graph", deadline)
        dependency_graph = graph_data.get("checkers")
        if not isinstance(dependency_graph, Mapping) or not all(
            isinstance(name, str) for name in dependency_graph
        ):
            raise CosmosEvaluatorServiceError(
                "Cosmos Evaluator /dependency-graph returned an invalid checker map"
            )

        selected = {check.name for check in self.checks}
        missing_healthy = sorted(selected.difference(healthy))
        if missing_healthy:
            raise CosmosEvaluatorServiceError(
                "configured Cosmos checker(s) are not healthy: "
                + ", ".join(missing_healthy)
            )
        missing_graph = sorted(selected.difference(dependency_graph))
        if missing_graph:
            raise CosmosEvaluatorServiceError(
                "configured Cosmos checker(s) are absent from the dependency graph: "
                + ", ".join(missing_graph)
            )

        result = CosmosPreflightResult(
            version=version,
            commit_sha=commit_sha,
            healthy_checkers=tuple(healthy),
            dependency_graph_checkers=tuple(dependency_graph),
        )
        self._preflight_result = result
        return result

    def build_request(
        self,
        candidate_video_uri: str,
        *,
        selected_variables: Mapping[str, Any] | None = None,
        variable_options: Mapping[str, Any] | None = None,
    ) -> tuple[str, dict[str, Any]]:
        """Build a request and return the single resolved stream key with it."""

        stream_key = resolve_stream_key(candidate_video_uri, self.configured_stream_key)
        return stream_key, self._build_request_for_stream(
            candidate_video_uri,
            stream_key,
            selected_variables=selected_variables,
            variable_options=variable_options,
        )

    def _build_request_for_stream(
        self,
        candidate_video_uri: str,
        stream_key: str,
        *,
        selected_variables: Mapping[str, Any] | None,
        variable_options: Mapping[str, Any] | None,
    ) -> dict[str, Any]:
        inputs = deepcopy(dict(self.additional_inputs))
        for field in ("original_video_urls", "world_model_video_urls"):
            value = inputs.get(field)
            if value is not None:
                if not isinstance(value, Mapping):
                    raise CosmosEvaluatorServiceError(
                        f"cosmos_evaluator.inputs.{field} must be a stream-key mapping"
                    )
                if stream_key not in value:
                    raise CosmosEvaluatorServiceError(
                        f"cosmos_evaluator.inputs.{field} lacks resolved stream key "
                        f"{stream_key!r}"
                    )

        inputs["augmented_video_urls"] = {stream_key: candidate_video_uri}
        shared_config = deepcopy(dict(self.shared_config))
        if self.merge_captioning_selections:
            self._merge_runtime_mapping(
                shared_config,
                "selected_variables",
                selected_variables,
            )
            self._merge_runtime_mapping(
                shared_config,
                "variable_options",
                variable_options,
            )
        inputs["config"] = shared_config
        return {
            "inputs": inputs,
            "checks": [check.name for check in self.checks],
            "output_storage_prefix": self.output_storage_prefix,
        }

    @staticmethod
    def _merge_runtime_mapping(
        config: MutableMapping[str, Any],
        key: str,
        runtime_value: Mapping[str, Any] | None,
    ) -> None:
        if runtime_value is None:
            return
        if not isinstance(runtime_value, Mapping):
            raise CosmosEvaluatorServiceError(f"runtime {key} must be a mapping")
        configured_value = config.get(key)
        if configured_value is None:
            merged: dict[str, Any] = {}
        elif isinstance(configured_value, Mapping):
            merged = deepcopy(dict(configured_value))
        else:
            raise CosmosEvaluatorServiceError(
                f"cosmos_evaluator.config.{key} must be a mapping when merging selections"
            )
        merged.update(deepcopy(dict(runtime_value)))
        config[key] = merged

    def evaluate(
        self,
        candidate_video_uri: str,
        *,
        selected_variables: Mapping[str, Any] | None = None,
        variable_options: Mapping[str, Any] | None = None,
    ) -> CosmosEvaluationOutcome:
        """Submit one candidate exactly once and return its typed outcome."""

        request_id: int | None = None
        stream_key: str | None = None
        execution: Mapping[str, Any] | None = None
        try:
            # Endpoint timeout covers Arbitrator submission and polling; local
            # request construction is intentionally outside that I/O budget.
            stream_key, request_payload = self.build_request(
                candidate_video_uri,
                selected_variables=selected_variables,
                variable_options=variable_options,
            )
            deadline = self._monotonic() + self.timeout
            response = self._post_once("/process/async", request_payload, deadline)
            try:
                data = self._success_data(response, "/process/async")
                request_id = data.get("request_id")
                if (
                    not isinstance(request_id, int)
                    or isinstance(request_id, bool)
                    or request_id < 1
                ):
                    raise CosmosEvaluatorServiceError(
                        "Cosmos Evaluator submission omitted a valid data.request_id"
                    )
            except CosmosEvaluatorError as exc:
                # HTTP 202 means Arbitrator may have accepted the work. Without
                # a valid request_id the client cannot poll or prove that a
                # second POST would be safe.
                raise CosmosEvaluatorAmbiguousPostError(
                    "Cosmos Evaluator accepted the submission but returned no "
                    "usable request_id; request was not resubmitted"
                ) from exc

            while True:
                snapshot = self._get_success_data(f"/executions/{request_id}", deadline)
                execution_value = snapshot.get("execution")
                if not isinstance(execution_value, Mapping):
                    raise CosmosEvaluatorServiceError(
                        "Cosmos Evaluator poll response omitted data.execution"
                    )
                execution = execution_value
                observed_request_id = execution.get("request_id")
                if (
                    observed_request_id is not None
                    and observed_request_id != request_id
                ):
                    raise CosmosEvaluatorServiceError(
                        "Cosmos Evaluator poll response returned a different request_id"
                    )
                is_terminal = execution.get("is_terminal")
                if not isinstance(is_terminal, bool):
                    raise CosmosEvaluatorServiceError(
                        "Cosmos Evaluator execution.is_terminal is not a boolean"
                    )
                if is_terminal:
                    return self._terminal_outcome(
                        request_id, stream_key, snapshot, execution
                    )
                self._sleep_with_deadline(self.poll_interval_seconds, deadline)
        except (CosmosEvaluatorError, requests.RequestException, ValueError) as exc:
            safe_error = str(redact_sensitive_data(str(exc), secrets=self._secrets))
            provider = self._provider_base(request_id, stream_key, execution)
            provider["error_code"] = (
                "ambiguous_post"
                if isinstance(exc, CosmosEvaluatorAmbiguousPostError)
                else "service_failure"
            )
            provider["error"] = safe_error
            provider = redact_sensitive_data(provider, secrets=self._secrets)
            self._logger.error("Cosmos Evaluator service failure: %s", safe_error)
            return CosmosEvaluationOutcome(
                operational_success=False,
                quality_passed=None,
                provider_metadata=provider,
                compatibility_aliases={},
                error=safe_error,
            )

    def evaluate_candidate(
        self,
        candidate_video_uri: str,
        **kwargs: Any,
    ) -> CosmosEvaluationOutcome:
        """Descriptive alias for :meth:`evaluate` used by pipeline callers."""

        return self.evaluate(candidate_video_uri, **kwargs)

    def _terminal_outcome(
        self,
        request_id: int,
        stream_key: str,
        snapshot: Mapping[str, Any],
        execution: Mapping[str, Any],
    ) -> CosmosEvaluationOutcome:
        status = execution.get("status")
        if status != "success":
            raise CosmosEvaluatorServiceError(
                f"Cosmos Evaluator execution ended with status {status!r}"
            )

        checkers_group = snapshot.get("checkers")
        if isinstance(checkers_group, Mapping):
            items = checkers_group.get("items")
            if isinstance(items, list):
                selected_names = {check.name for check in self.checks}
                for item in items:
                    if not isinstance(item, Mapping):
                        continue
                    checker_name = item.get("checker_name")
                    checker_status = str(item.get("status") or "").lower()
                    if (
                        checker_name in selected_names
                        and checker_status in _FAILED_CHECKER_STATUSES
                    ):
                        raise CosmosEvaluatorServiceError(
                            f"Cosmos checker {checker_name!r} ended with status "
                            f"{checker_status!r}"
                        )

        response_json = execution.get("response_json")
        if isinstance(response_json, str):
            try:
                response_json = json.loads(response_json)
            except json.JSONDecodeError as exc:
                raise CosmosEvaluatorServiceError(
                    "Cosmos Evaluator execution.response_json is not valid JSON"
                ) from exc
        if not isinstance(response_json, Mapping):
            raise CosmosEvaluatorServiceError(
                "Cosmos Evaluator execution.response_json is not a mapping"
            )
        checker_results = response_json.get("checker_results")
        if not isinstance(checker_results, Mapping):
            raise CosmosEvaluatorServiceError(
                "Cosmos Evaluator response_json omitted checker_results"
            )

        normalized_checks: dict[str, Any] = {}
        aliases: dict[str, Any] = {}
        gate_verdicts: list[bool] = []
        fully_errored_checks: list[tuple[str, int]] = []
        for check in self.checks:
            if check.name not in checker_results:
                raise CosmosEvaluatorServiceError(
                    f"Cosmos Evaluator result omitted checker {check.name!r}"
                )
            checker_value = checker_results[check.name]
            if isinstance(checker_value, Mapping) and stream_key in checker_value:
                result = checker_value[stream_key]
            else:
                result = checker_value
            if result is None:
                raise CosmosEvaluatorServiceError(
                    f"Cosmos checker {check.name!r} returned no result"
                )
            if isinstance(result, Mapping):
                skipped_error = result.get("error")
                if isinstance(skipped_error, str) and skipped_error.lower().startswith(
                    "skipped:"
                ):
                    raise CosmosEvaluatorServiceError(
                        f"Cosmos checker {check.name!r} was skipped"
                    )
                summary = result.get("summary")
                if isinstance(summary, Mapping):
                    total = summary.get("total_checks")
                    errored = summary.get("errored_checks")
                    if (
                        isinstance(total, int)
                        and not isinstance(total, bool)
                        and total > 0
                        and isinstance(errored, int)
                        and not isinstance(errored, bool)
                        and errored == total
                    ):
                        fully_errored_checks.append((check.name, total))

            passed: bool | None = None
            if isinstance(result, Mapping) and isinstance(result.get("passed"), bool):
                passed = result["passed"]
            if check.gate:
                if passed is None:
                    raise CosmosEvaluatorServiceError(
                        f"gating Cosmos checker {check.name!r} must return a "
                        "top-level boolean passed field"
                    )
                gate_verdicts.append(passed)

            result_view: dict[str, Any] = {"result": deepcopy(result)}
            if passed is not None:
                result_view["passed"] = passed
            per_stream_view = {stream_key: result_view}
            normalized_checks[check.name] = per_stream_view
            if check.metadata_key:
                aliases[check.metadata_key] = deepcopy(result_view)

        provider = self._provider_base(request_id, stream_key, execution)
        provider["checks"] = normalized_checks
        operational_error = None
        if fully_errored_checks:
            operational_error = "; ".join(
                f"Cosmos checker {name!r} produced no verdicts: all {total} "
                "checks errored"
                for name, total in fully_errored_checks
            )
            provider["error"] = operational_error
        provider = redact_sensitive_data(provider, secrets=self._secrets)
        aliases = redact_sensitive_data(aliases, secrets=self._secrets)
        operational_success = not fully_errored_checks
        quality_passed = all(gate_verdicts) if operational_success else None
        return CosmosEvaluationOutcome(
            operational_success=operational_success,
            quality_passed=quality_passed,
            provider_metadata=provider,
            compatibility_aliases=aliases,
            error=operational_error,
        )

    @staticmethod
    def _provider_base(
        request_id: int | None,
        stream_key: str | None,
        execution: Mapping[str, Any] | None,
    ) -> dict[str, Any]:
        execution = execution or {}
        return {
            "request_id": request_id,
            "status": execution.get("status", "service_failure"),
            "version": execution.get("version"),
            "commit_sha": execution.get("commit_sha"),
            "stream_key": stream_key,
            "checks": {},
        }

    def _get_success_data(self, path: str, deadline: float) -> dict[str, Any]:
        response = self._get_with_retries(path, deadline)
        return self._success_data(response, path)

    def _get_with_retries(self, path: str, deadline: float) -> Any:
        url = f"{self.base_url}{path}"
        last_error: Exception | None = None
        for attempt in range(MAX_GET_ATTEMPTS):
            remaining = self._remaining(deadline)
            try:
                response = self._session.get(
                    url,
                    headers=dict(self._headers),
                    timeout=remaining,
                )
            except requests.RequestException as exc:
                last_error = exc
                if attempt + 1 >= MAX_GET_ATTEMPTS:
                    break
                self._sleep_with_deadline(_GET_BACKOFF_SECONDS * (2**attempt), deadline)
                continue

            status_code = getattr(response, "status_code", None)
            if status_code in _TRANSIENT_GET_STATUSES:
                last_error = CosmosEvaluatorServiceError(
                    f"transient HTTP {status_code} from Cosmos Evaluator GET {path}"
                )
                if attempt + 1 >= MAX_GET_ATTEMPTS:
                    break
                self._sleep_with_deadline(_GET_BACKOFF_SECONDS * (2**attempt), deadline)
                continue
            if status_code != 200:
                raise CosmosEvaluatorServiceError(
                    f"HTTP {status_code} from Cosmos Evaluator GET {path}"
                )
            return response

        if isinstance(last_error, CosmosEvaluatorError):
            raise last_error
        raise CosmosEvaluatorServiceError(
            f"Cosmos Evaluator GET {path} failed after {MAX_GET_ATTEMPTS} attempts"
        ) from last_error

    def _post_once(self, path: str, payload: Mapping[str, Any], deadline: float) -> Any:
        remaining = self._remaining(deadline)
        # Discard pooled keep-alive connections before the non-idempotent
        # submission. Production clients replace their owned session so the
        # POST cannot reuse a connection opened by preflight or earlier polls.
        self._session.close()
        if self._owns_session:
            self._session = requests.Session()
        try:
            response = self._session.post(
                f"{self.base_url}{path}",
                headers=dict(self._headers),
                json=deepcopy(dict(payload)),
                timeout=remaining,
            )
        except requests.RequestException as exc:
            raise CosmosEvaluatorAmbiguousPostError(
                "Cosmos Evaluator submission failed; request was not resubmitted"
            ) from exc
        if getattr(response, "status_code", None) != 202:
            raise CosmosEvaluatorServiceError(
                "Cosmos Evaluator submission expected HTTP 202 but received "
                f"{getattr(response, 'status_code', None)}; request was not resubmitted"
            )
        return response

    @staticmethod
    def _success_data(response: Any, path: str) -> dict[str, Any]:
        try:
            payload = response.json()
        except (ValueError, AttributeError) as exc:
            raise CosmosEvaluatorServiceError(
                f"Cosmos Evaluator {path} returned invalid JSON"
            ) from exc
        if not isinstance(payload, Mapping):
            raise CosmosEvaluatorServiceError(
                f"Cosmos Evaluator {path} returned a non-object envelope"
            )
        if payload.get("success") is not True:
            raise CosmosEvaluatorServiceError(
                f"Cosmos Evaluator {path} returned an unsuccessful envelope"
            )
        if not isinstance(payload.get("timestamp"), str) or not payload["timestamp"]:
            raise CosmosEvaluatorServiceError(
                f"Cosmos Evaluator {path} omitted the envelope timestamp"
            )
        data = payload.get("data")
        if not isinstance(data, Mapping):
            raise CosmosEvaluatorServiceError(
                f"Cosmos Evaluator {path} returned invalid envelope data"
            )
        return dict(data)

    def _remaining(self, deadline: float) -> float:
        remaining = deadline - self._monotonic()
        if remaining <= 0:
            raise CosmosEvaluatorServiceError(
                "Cosmos Evaluator operation exceeded its total timeout"
            )
        return remaining

    def _sleep_with_deadline(self, seconds: float, deadline: float) -> None:
        remaining = self._remaining(deadline)
        sleep_for = min(seconds, remaining)
        if sleep_for > 0:
            self._sleep(sleep_for)
        self._remaining(deadline)


__all__ = [
    "ARBITRATOR_ADAPTER",
    "CosmosCheckerSpec",
    "CosmosEvaluationOutcome",
    "CosmosEvaluatorClient",
    "CosmosEvaluatorError",
    "CosmosEvaluatorInitializationError",
    "CosmosEvaluatorServiceError",
    "CosmosPreflightResult",
    "redact_sensitive_data",
    "resolve_stream_key",
]
