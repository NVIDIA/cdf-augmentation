# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Root pipeline configuration model with cross-section validation."""

import os
from typing import Any, List, Optional

from pydantic import BaseModel, Field, model_validator

from aug_utils.endpoint_registry import (
    effective_adapter,
    resolve_endpoint,
    select_endpoint,
)
from aug_utils.stream_keys import resolve_stream_key

from .adapters import KNOWN_ADAPTERS
from .augmentation import AugmentationConfig
from .captioning import CaptioningConfig
from .data import DataSample
from .data_processing import DataProcessingConfig
from .endpoints import Endpoint
from .evaluators import EvaluatorEntry
from .pipeline import PipelineSettings


COSMOS_EVALUATOR_ADAPTER = "cosmos.evaluator.arbitrator"

# Root fields written by cli.py independently of evaluator aliases. Built-in
# evaluator blocks are handled separately because only enabled evaluators write
# them; the Cosmos provider block is reserved unconditionally below.
_ROOT_METADATA_FIELDS = frozenset(
    {
        "prompt",
        "selections",
        "caption_path",
        "output_media_path",
        "input_media_path",
        "control_media",
        "prompt_builder",
        "scene_description",
        "scene_description_source",
        "captioning",
        "endpoints",
        "natural_caption",
        "alignment",
        "transcode",
    }
)


def _has_role(endpoints: List[Endpoint], role: str) -> bool:
    """Return True when at least one endpoint declares ``role``."""
    return any(getattr(ep, "role", None) == role for ep in endpoints)


def _is_remote_path(path: Any) -> bool:
    """Call the pipeline's storage-path classifier without an import cycle."""
    from aug_utils.common import is_remote_path

    return isinstance(path, str) and is_remote_path(path)


class PipelineConfig(BaseModel):
    """Root configuration model for the unified augmentation pipeline.

    Usage::

        import yaml
        from aug_utils.schema import PipelineConfig

        with open("config.yaml") as f:
            raw = yaml.safe_load(f)
        config = PipelineConfig(**raw)
    """

    data: List[DataSample] = Field(min_length=1)
    endpoints: List[Endpoint] = Field(default_factory=list)
    pipeline: PipelineSettings = Field(default_factory=PipelineSettings)
    captioning: Optional[CaptioningConfig] = Field(default=None)
    augmentation: Optional[AugmentationConfig] = Field(default=None)
    data_processing: Optional[DataProcessingConfig] = Field(default=None)
    evaluators: Optional[List[EvaluatorEntry]] = Field(default=None)

    @model_validator(mode="after")
    def validate_endpoint_requirements(self):
        """Ensure required endpoints exist for the configured modules."""
        # Augmentation -> endpoint mapping. Every model now needs an endpoint
        # resolvable from the registry; the resolved endpoint's adapter must be
        # one we know how to construct.
        resolved_ep = None
        if self.augmentation is not None:
            name = self.augmentation.model.name
            try:
                resolved_ep = resolve_endpoint(self.endpoints, name)
            except ValueError as exc:
                raise ValueError(
                    f"augmentation.model.name={name!r} requires a matching "
                    f"endpoint in the endpoints list: {exc}"
                ) from exc

            adapter = effective_adapter(resolved_ep)
            if adapter not in KNOWN_ADAPTERS:
                raise ValueError(
                    f"endpoint {resolved_ep.id!r} resolves to unknown adapter "
                    f"{adapter!r}; known adapters: {sorted(KNOWN_ADAPTERS)}"
                )

        # Validate data.inputs.rgb is present when the model requires it.
        # rgb is optional only for video_predict + text2world.
        requires_rgb = True
        if resolved_ep is not None:
            role = getattr(resolved_ep, "role", None)
            inference_type = self.augmentation.parameters.inference_type
            inference_type = (
                inference_type.value
                if hasattr(inference_type, "value")
                else inference_type
            )
            if role == "video_predict" and inference_type == "text2world":
                requires_rgb = False

        if requires_rgb:
            for i, sample in enumerate(self.data):
                inputs = sample.inputs
                rgb = inputs.rgb if inputs is not None else None
                if not rgb:
                    raise ValueError(
                        f"data[{i}].inputs.rgb is required for "
                        f"the configured model (only video_predict "
                        f"with inference_type=text2world allows null inputs)"
                    )

        # Captioning -> endpoint mapping (env vars may satisfy the requirement).
        if self.captioning is not None:
            if self.captioning.vlm is not None:
                if not (
                    _has_role(self.endpoints, "vlm") or os.getenv("VLM_ENDPOINT_URL")
                ):
                    raise ValueError(
                        "captioning.vlm requires an endpoint with role 'vlm' "
                        "(or the VLM_ENDPOINT_URL env var)"
                    )
            if self.captioning.llm is not None:
                llm = self.captioning.llm
                # text and file_path captioners don't need an LLM endpoint
                needs_llm_endpoint = llm.text is None and llm.file_path is None
                if needs_llm_endpoint and not (
                    _has_role(self.endpoints, "llm")
                    or os.getenv("LLM_ENDPOINT_URL")
                    or os.getenv("LLM_CAPTION_ENDPOINT_URL")
                ):
                    raise ValueError(
                        "captioning.llm requires an endpoint with role 'llm' "
                        "(or the LLM_ENDPOINT_URL / LLM_CAPTION_ENDPOINT_URL "
                        "env var), unless 'text' or 'file_path' is set"
                    )

        # VLM-template prompting uses exactly one selected ID from each flat
        # attribute catalog for every sample.
        template = self.captioning.template if self.captioning is not None else None
        for index, sample in enumerate(self.data):
            inputs = sample.inputs
            selections = inputs.prompt_attributes if inputs is not None else None
            if template is None:
                if selections is not None:
                    raise ValueError(
                        f"data[{index}].inputs.prompt_attributes requires "
                        "captioning.template"
                    )
                continue

            if selections is None:
                raise ValueError(
                    f"data[{index}].inputs.prompt_attributes is required when "
                    "captioning.template is configured"
                )
            for attribute_name in ("event_type", "motion_level", "aftermath"):
                selected_id = getattr(selections, attribute_name)
                catalog = getattr(template.attributes, attribute_name)
                if selected_id not in catalog:
                    raise ValueError(
                        f"data[{index}].inputs.prompt_attributes.{attribute_name}="
                        f"{selected_id!r} is not defined in "
                        f"captioning.template.attributes.{attribute_name}"
                    )

        # --- BYOM endpoint-selection guards -------------------------------
        # Endpoint ids must be unique (only the non-None ones are checked).
        ids = [ep.id for ep in self.endpoints if ep.id is not None]
        dupes = sorted({i for i in ids if ids.count(i) > 1})
        if dupes:
            raise ValueError(
                f"duplicate endpoint ids in endpoints: {dupes}; ids must be unique"
            )

        def _count_role(role: str) -> int:
            return sum(1 for ep in self.endpoints if ep.role == role)

        # Every consumer's endpoint_id must resolve and have a matching role.
        for consumer, role, endpoint_id in self._endpoint_selectors():
            if endpoint_id is None:
                continue
            try:
                select_endpoint(self.endpoints, role, endpoint_id, required=False)
            except ValueError as exc:
                raise ValueError(f"{consumer}: {exc}") from exc

        # Ambiguous role without a selector: with 2+ endpoints of a role the
        # consumer MUST pick one by endpoint_id. An env URL override does NOT
        # disambiguate (it can't say which of the several is meant), and at
        # runtime select_endpoint() raises on the ambiguous match regardless of
        # the override — so reject it here with a clear message instead of a
        # startup crash.
        cap = self.captioning
        if cap is not None:
            if (
                cap.vlm is not None
                and cap.vlm.endpoint_id is None
                and _count_role("vlm") > 1
            ):
                raise ValueError(
                    "captioning.vlm: multiple endpoints with role 'vlm' exist; "
                    "set captioning.vlm.endpoint_id to choose one"
                )
            if cap.llm is not None:
                llm = cap.llm
                needs_llm_endpoint = llm.text is None and llm.file_path is None
                if (
                    needs_llm_endpoint
                    and llm.endpoint_id is None
                    and _count_role("llm") > 1
                ):
                    raise ValueError(
                        "captioning.llm: multiple endpoints with role 'llm' exist; "
                        "set captioning.llm.endpoint_id to choose one"
                    )

        # Evaluators, like captioning, need a resolvable transport when no
        # endpoint_id pins one: first require the role to exist (endpoint or env
        # override), then reject an ambiguous choice among several. Without the
        # presence check the config validates but `_init_evaluators` builds a
        # generator/verifier against an empty endpoint and fails at runtime.
        def _require_role(role: str, env_var: str, label: str) -> None:
            # Presence: an env URL override can supply the (single) endpoint.
            if not (_has_role(self.endpoints, role) or os.getenv(env_var)):
                raise ValueError(
                    f"{label} requires an endpoint with role '{role}' "
                    f"(or the {env_var} env var)"
                )
            # Ambiguity: 2+ endpoints of a role must be disambiguated by
            # endpoint_id — the env override can't pick which one, and runtime
            # select_endpoint() would raise on the ambiguous match.
            if _count_role(role) > 1:
                raise ValueError(
                    f"{label}: multiple endpoints with role '{role}' exist; "
                    "set endpoint_id to choose one"
                )

        for idx, entry in enumerate(self.evaluators or []):
            av = entry.attribute_verification
            if av is not None and av.enabled:
                # An enabled attribute_verification always builds an LLM question
                # generator AND a VLM verifier; missing either makes it a silent
                # no-op (attribute_verifier is None) that strict eval reports as a
                # pass. Require both transports.
                qg = av.question_generation
                if qg is None or qg.endpoint_id is None:
                    _require_role(
                        "llm",
                        "LLM_ENDPOINT_URL",
                        f"evaluators[{idx}].attribute_verification.question_generation",
                    )
                inline_vlm = av.vlm_verification
                if inline_vlm is None or inline_vlm.endpoint_id is None:
                    _require_role(
                        "vlm",
                        "VLM_ENDPOINT_URL",
                        f"evaluators[{idx}].attribute_verification.vlm_verification",
                    )
            vv = entry.vlm_verification
            if vv is not None and vv.endpoint_id is None:
                _require_role(
                    "vlm",
                    "VLM_ENDPOINT_URL",
                    f"evaluators[{idx}].vlm_verification",
                )

        enabled_cosmos = [
            (idx, entry.cosmos_evaluator)
            for idx, entry in enumerate(self.evaluators or [])
            if entry.cosmos_evaluator is not None and entry.cosmos_evaluator.enabled
        ]
        if len(enabled_cosmos) > 1:
            raise ValueError(
                "at most one enabled cosmos_evaluator entry is allowed, "
                f"got {len(enabled_cosmos)}"
            )

        if enabled_cosmos:
            idx, cosmos = enabled_cosmos[0]
            try:
                evaluator_endpoint = select_endpoint(
                    self.endpoints,
                    "evaluator",
                    cosmos.endpoint_id,
                    required=True,
                )
            except ValueError as exc:
                raise ValueError(
                    f"evaluators[{idx}].cosmos_evaluator.endpoint_id: {exc}"
                ) from exc
            if evaluator_endpoint.adapter != COSMOS_EVALUATOR_ADAPTER:
                raise ValueError(
                    f"evaluators[{idx}].cosmos_evaluator endpoint "
                    f"{cosmos.endpoint_id!r} must explicitly use adapter "
                    f"{COSMOS_EVALUATOR_ADAPTER!r}"
                )

            for sample_idx, sample in enumerate(self.data):
                inputs = sample.inputs
                if inputs is not None:
                    if inputs.rgb is not None and not _is_remote_path(inputs.rgb):
                        raise ValueError(
                            f"data[{sample_idx}].inputs.rgb must be externally "
                            "readable when cosmos_evaluator is enabled"
                        )
                    controls = inputs.controls
                    if controls is not None:
                        for control_name, control_path in controls.model_dump(
                            exclude_none=True
                        ).items():
                            if not _is_remote_path(control_path):
                                raise ValueError(
                                    f"data[{sample_idx}].inputs.controls."
                                    f"{control_name} must be externally readable "
                                    "when cosmos_evaluator is enabled"
                                )
                if not _is_remote_path(sample.output.video):
                    raise ValueError(
                        f"data[{sample_idx}].output.video must be externally "
                        "readable when cosmos_evaluator is enabled"
                    )

            if cosmos.stream_key is not None:
                required_stream_keys = {cosmos.stream_key}
            else:
                required_stream_keys = {
                    resolve_stream_key(sample.output.video) for sample in self.data
                }

            for field_name in ("original_video_urls", "world_model_video_urls"):
                value = cosmos.inputs.get(field_name)
                if value is None:
                    continue
                if not isinstance(value, dict):
                    raise ValueError(
                        f"evaluators[{idx}].cosmos_evaluator.inputs.{field_name} "
                        "must be a mapping of stream key to remote URI"
                    )
                for stream_key, media_path in value.items():
                    if not isinstance(stream_key, str) or not stream_key.strip():
                        raise ValueError(
                            f"evaluators[{idx}].cosmos_evaluator.inputs."
                            f"{field_name} contains a blank stream key"
                        )
                    if not _is_remote_path(media_path):
                        raise ValueError(
                            f"evaluators[{idx}].cosmos_evaluator.inputs."
                            f"{field_name}[{stream_key!r}] must be externally readable"
                        )
                missing_stream_keys = sorted(required_stream_keys.difference(value))
                if missing_stream_keys:
                    raise ValueError(
                        f"evaluators[{idx}].cosmos_evaluator.inputs.{field_name} "
                        f"lacks resolved stream key(s): {missing_stream_keys}"
                    )

            rds_hq_url = cosmos.inputs.get("rds_hq_url")
            if rds_hq_url is not None and not _is_remote_path(rds_hq_url):
                raise ValueError(
                    f"evaluators[{idx}].cosmos_evaluator.inputs.rds_hq_url "
                    "must be externally readable"
                )

            enabled_builtin_keys = set()
            for entry in self.evaluators or []:
                hc = entry.hallucination_check
                if hc is not None and hc.enabled:
                    enabled_builtin_keys.add("hallucination_check")
                av = entry.attribute_verification
                if av is not None and av.enabled:
                    enabled_builtin_keys.add("attribute_verification")

            aliases = set()
            for check in cosmos.checks:
                alias = check.metadata_key
                if alias is None:
                    continue
                if alias == "cosmos_evaluator":
                    raise ValueError(
                        "Cosmos metadata_key 'cosmos_evaluator' is reserved for "
                        "the provider block"
                    )
                if alias in _ROOT_METADATA_FIELDS:
                    raise ValueError(
                        f"Cosmos metadata_key {alias!r} collides with an existing "
                        "root metadata field"
                    )
                if alias in enabled_builtin_keys:
                    raise ValueError(
                        f"Cosmos metadata_key {alias!r} collides with an enabled "
                        "built-in evaluator"
                    )
                if alias in aliases:
                    raise ValueError(f"duplicate Cosmos metadata_key alias {alias!r}")
                aliases.add(alias)

        return self

    def _endpoint_selectors(self):
        """Yield ``(consumer_label, required_role, endpoint_id)`` for every
        consumer that can pick a specific endpoint by id."""
        cap = self.captioning
        if cap is not None:
            if cap.vlm is not None:
                yield ("captioning.vlm.endpoint_id", "vlm", cap.vlm.endpoint_id)
            if cap.llm is not None:
                yield ("captioning.llm.endpoint_id", "llm", cap.llm.endpoint_id)
        for idx, entry in enumerate(self.evaluators or []):
            av = entry.attribute_verification
            if av is not None:
                qg = av.question_generation
                if qg is not None:
                    yield (
                        f"evaluators[{idx}].attribute_verification."
                        "question_generation.endpoint_id",
                        "llm",
                        qg.endpoint_id,
                    )
                inline_vlm = av.vlm_verification
                if inline_vlm is not None:
                    yield (
                        f"evaluators[{idx}].attribute_verification."
                        "vlm_verification.endpoint_id",
                        "vlm",
                        inline_vlm.endpoint_id,
                    )
            vv = entry.vlm_verification
            if vv is not None:
                yield (
                    f"evaluators[{idx}].vlm_verification.endpoint_id",
                    "vlm",
                    vv.endpoint_id,
                )
            cosmos = entry.cosmos_evaluator
            if cosmos is not None and cosmos.enabled:
                yield (
                    f"evaluators[{idx}].cosmos_evaluator.endpoint_id",
                    "evaluator",
                    cosmos.endpoint_id,
                )
