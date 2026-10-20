# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Evaluator configuration models.

Evaluators are specified as a list, where each entry contains exactly one of:
- ``hallucination_check``
- ``attribute_verification`` (may contain an inline ``vlm_verification``)
- ``cosmos_evaluator`` (external Cosmos Evaluator Arbitrator)

The legacy standalone ``vlm_verification`` field remains parseable only so the
validator can return a targeted migration error. It has no independent
question or acceptance contract and therefore must never validate as an
executable evaluator.
"""

from typing import Any, Dict, List, Literal, Optional
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .base import BaseInferenceParameters


# ---------------------------------------------------------------------------
# Hallucination Check
# ---------------------------------------------------------------------------


class HallucinationParams(BaseModel):
    """Low-level parameters for motion-artifact hallucination detection."""

    grad_thresh: float = Field(default=10.0)
    blur_ksize: int = Field(default=7)
    morph_k: int = Field(default=3)
    dist_tol_px: float = Field(default=7.0)
    max_frames: Optional[int] = Field(default=None)


class HallucinationCheckEvaluator(BaseModel):
    """Evaluator that detects hallucinated motion artifacts."""

    model_config = ConfigDict(extra="forbid")
    enabled: bool = Field(default=True)
    threshold: float = Field(default=0.682, ge=0.0, le=1.0)
    params: HallucinationParams = Field(default_factory=HallucinationParams)


# ---------------------------------------------------------------------------
# VLM Verification
# ---------------------------------------------------------------------------


class VLMVerificationEvaluator(BaseModel):
    """Evaluator that uses a VLM to answer verification questions."""

    executor_type: Optional[Literal["local", "remote"]] = Field(default=None)
    url: Optional[str] = Field(default=None)
    system_prompt: str = Field(default="")
    frames: int = Field(
        default=1,
        ge=1,
        description="Frames to sample from a video, evenly spaced (1 = first frame "
        "only). Use >1 so a mid-video event is visible to the VLM.",
    )
    parameters: BaseInferenceParameters = Field(default_factory=BaseInferenceParameters)
    endpoint_id: Optional[str] = Field(
        default=None,
        description="Pick a specific endpoints[].id; defaults to the single endpoint of the matching role",
    )

    @model_validator(mode="after")
    def validate_remote_executor(self):
        """Ensure url is set when executor_type is remote."""
        if self.executor_type == "remote" and not self.url:
            raise ValueError("url must be specified when executor_type is 'remote'")
        return self


# ---------------------------------------------------------------------------
# Attribute Verification
# ---------------------------------------------------------------------------


class QuestionGenerationConfig(BaseModel):
    """LLM-based MCQ question generation for attribute verification."""

    system_prompt: str = Field(default="")
    generate_options: bool = Field(
        default=False,
        description="Let the LLM invent the distractor options instead of drawing "
        "from verification_options. The correct answer still stays pinned to the "
        "variable's (ground-truth) value.",
    )
    parameters: BaseInferenceParameters = Field(default_factory=BaseInferenceParameters)
    endpoint_id: Optional[str] = Field(
        default=None,
        description="Pick a specific endpoints[].id; defaults to the single endpoint of the matching role",
    )


class ExtraQuestion(BaseModel):
    """A fixed verification question (not LLM-generated)."""

    variable: Optional[str] = Field(
        default=None, description="Variable name this question verifies"
    )
    question: str = Field(description="The question text")
    options: Dict[str, str] = Field(
        description="Answer options keyed by letter (e.g., {'A': '...', 'B': '...'})"
    )
    correct_answer: str = Field(description="Letter of the correct answer")
    request_reasoning: bool = Field(default=False)

    @model_validator(mode="after")
    def validate_correct_answer(self):
        """Ensure correct_answer is a valid key in options."""
        if self.correct_answer not in self.options:
            raise ValueError(
                f"correct_answer '{self.correct_answer}' must be one of "
                f"the option keys: {list(self.options.keys())}"
            )
        return self


class NaturalCaptionConfig(BaseModel):
    """Configuration for generating a natural-language caption after verification passes."""

    system_prompt: Optional[str] = Field(default=None)
    user_prompt_template: Optional[str] = Field(default=None)


class AttributeVerificationEvaluator(BaseModel):
    """Evaluator that checks generated media against target attributes via MCQ."""

    model_config = ConfigDict(extra="forbid")
    executor_type: Optional[Literal["local", "remote"]] = Field(default=None)
    url: Optional[str] = Field(default=None)
    enabled: bool = Field(default=True)
    generate_natural_caption_on_pass: bool = Field(default=False)
    natural_caption: Optional[NaturalCaptionConfig] = Field(default=None)
    exclude_variables: List[str] = Field(default_factory=list)
    extra_questions: Optional[List[ExtraQuestion]] = Field(default=None)
    question_generation: QuestionGenerationConfig = Field(
        default_factory=QuestionGenerationConfig
    )
    vlm_verification: Optional[VLMVerificationEvaluator] = Field(default=None)

    @model_validator(mode="after")
    def validate_remote_executor(self):
        """Ensure url is set when executor_type is remote."""
        if self.executor_type == "remote" and not self.url:
            raise ValueError("url must be specified when executor_type is 'remote'")
        return self


# ---------------------------------------------------------------------------
# External Cosmos Evaluator
# ---------------------------------------------------------------------------


class CosmosEvaluatorCheck(BaseModel):
    """One checker selected from an external Cosmos Evaluator deployment."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(description="Free-form checker name advertised by Arbitrator")
    gate: bool = Field(description="Whether this check affects candidate acceptance")
    metadata_key: Optional[str] = Field(
        default=None,
        description="Optional root-level compatibility alias for this check's result",
    )

    @field_validator("name")
    @classmethod
    def validate_name(cls, value: str) -> str:
        """Checker names are opaque but must be non-empty."""
        value = value.strip()
        if not value:
            raise ValueError("Cosmos checker name must not be blank")
        return value

    @field_validator("metadata_key")
    @classmethod
    def validate_metadata_key(cls, value: Optional[str]) -> Optional[str]:
        """Compatibility aliases, when supplied, must be non-empty."""
        if value is None:
            return None
        value = value.strip()
        if not value:
            raise ValueError("Cosmos checker metadata_key must not be blank")
        return value


class CosmosEvaluator(BaseModel):
    """External Cosmos Evaluator Arbitrator configuration."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = Field(default=True)
    endpoint_id: str = Field(
        description="Select an endpoints[] entry with role 'evaluator'"
    )
    poll_interval_seconds: float = Field(default=5.0, gt=0.0)
    stream_key: Optional[str] = Field(
        default=None,
        description="Stable per-stream key; derived from output.video when omitted",
    )
    output_storage_prefix: Optional[str] = Field(
        default=None,
        description="Relative object-storage key prefix for checker artifacts",
    )
    checks: List[CosmosEvaluatorCheck] = Field(min_length=1)
    inputs: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional Arbitrator inputs; augmented_video_urls is runtime-owned",
    )
    config: Dict[str, Any] = Field(
        default_factory=dict,
        description="Shared opaque config forwarded to every selected checker",
    )
    merge_captioning_selections: bool = Field(default=False)

    @field_validator("endpoint_id")
    @classmethod
    def validate_endpoint_id(cls, value: str) -> str:
        """The evaluator endpoint selector must be explicit and non-empty."""
        value = value.strip()
        if not value:
            raise ValueError("Cosmos evaluator endpoint_id must not be blank")
        return value

    @field_validator("stream_key")
    @classmethod
    def validate_stream_key(cls, value: Optional[str]) -> Optional[str]:
        """Normalize an explicit stream key without changing its contents."""
        if value is None:
            return None
        value = value.strip()
        if not value:
            raise ValueError("Cosmos evaluator stream_key must not be blank")
        return value

    @field_validator("output_storage_prefix")
    @classmethod
    def strip_output_storage_prefix(cls, value: Optional[str]) -> Optional[str]:
        """Strip only surrounding whitespace; preserve an optional trailing slash."""
        return value.strip() if value is not None else None

    @model_validator(mode="after")
    def validate_cosmos_contract(self):
        """Validate checker uniqueness and Arbitrator-owned request fields."""
        names = [check.name for check in self.checks]
        duplicates = sorted({name for name in names if names.count(name) > 1})
        if duplicates:
            raise ValueError(f"duplicate Cosmos checker names: {duplicates}")

        if "augmented_video_urls" in self.inputs:
            raise ValueError(
                "cosmos_evaluator.inputs.augmented_video_urls is reserved for "
                "the generated candidate"
            )

        prefix = self.output_storage_prefix
        if self.enabled and not prefix:
            raise ValueError(
                "output_storage_prefix is required when cosmos_evaluator is enabled"
            )
        if not prefix:
            return self

        if prefix.startswith("/"):
            raise ValueError("output_storage_prefix must not start with '/'")
        if "\\" in prefix:
            raise ValueError("output_storage_prefix must not contain backslashes")
        if any(ord(char) < 32 or ord(char) == 127 for char in prefix):
            raise ValueError(
                "output_storage_prefix must not contain ASCII control characters"
            )

        try:
            parsed = urlsplit(prefix)
        except ValueError as exc:
            raise ValueError(
                "output_storage_prefix must be a relative key prefix"
            ) from exc
        if parsed.scheme or parsed.netloc or parsed.query or parsed.fragment:
            raise ValueError(
                "output_storage_prefix must be a relative key prefix without a "
                "scheme, authority, query, or fragment"
            )

        segments = parsed.path.split("/")
        if segments and segments[-1] == "":
            segments = segments[:-1]
        if any(segment in ("", ".", "..") for segment in segments):
            raise ValueError(
                "output_storage_prefix must not contain empty, '.', or '..' path segments"
            )
        return self


# ---------------------------------------------------------------------------
# Evaluator Entry (list item)
# ---------------------------------------------------------------------------


class EvaluatorEntry(BaseModel):
    """A single entry in the ``evaluators`` list.

    Exactly one evaluator field must be set.
    """

    hallucination_check: Optional[HallucinationCheckEvaluator] = Field(default=None)
    attribute_verification: Optional[AttributeVerificationEvaluator] = Field(
        default=None
    )
    vlm_verification: Optional[VLMVerificationEvaluator] = Field(default=None)
    cosmos_evaluator: Optional[CosmosEvaluator] = Field(default=None)

    @model_validator(mode="after")
    def exactly_one_evaluator(self):
        """Ensure exactly one evaluator type is specified per entry."""
        set_fields = [
            f
            for f in (
                "hallucination_check",
                "attribute_verification",
                "vlm_verification",
                "cosmos_evaluator",
            )
            if getattr(self, f) is not None
        ]
        if len(set_fields) != 1:
            raise ValueError(
                f"Each evaluator entry must specify exactly one evaluator type, "
                f"got {len(set_fields)}: {set_fields}"
            )
        if self.vlm_verification is not None:
            raise ValueError(
                "standalone vlm_verification is not supported; move it under "
                "attribute_verification.vlm_verification so the pipeline has "
                "questions and expected answers to evaluate"
            )
        return self
