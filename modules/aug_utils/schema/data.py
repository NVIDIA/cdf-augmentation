# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Data sample models for pipeline input/output specification."""

from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class ControlInputs(BaseModel):
    """Optional control modality paths for Cosmos video transfer."""

    edge: Optional[str] = Field(default=None, description="Edge control video path")
    depth: Optional[str] = Field(default=None, description="Depth control video path")
    seg: Optional[str] = Field(
        default=None, description="Segmentation control video path"
    )
    vis: Optional[str] = Field(
        default=None, description="Visibility control video path"
    )
    wsm: Optional[str] = Field(
        default=None, description="World scenario model (WSM) control video path"
    )


class PromptAttributes(BaseModel):
    """Selected deterministic prompt attributes for one sample."""

    model_config = ConfigDict(extra="forbid")

    event_type: str = Field(description="Selected event type ID")
    motion_level: str = Field(description="Selected motion level ID")
    aftermath: str = Field(description="Selected aftermath ID")

    @field_validator("event_type", "motion_level", "aftermath")
    @classmethod
    def _strip_non_empty_selection(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("prompt attribute selections must not be empty")
        return value


class DataInputs(BaseModel):
    """Input specification for a single data sample."""

    rgb: Optional[str] = Field(
        default=None,
        description="Path to the input RGB video or image (not required for text2world)",
    )
    controls: Optional[ControlInputs] = Field(
        default=None,
        description="Control modality inputs for Cosmos video transfer",
    )
    prompt_attributes: Optional[PromptAttributes] = Field(
        default=None,
        description=(
            "Per-sample event_type, motion_level, and aftermath IDs selected "
            "from captioning.template.attributes"
        ),
    )


class DataOutput(BaseModel):
    """Output specification for a single data sample."""

    video: str = Field(description="Path for the output video or image")
    caption: str = Field(description="Path for the generated caption/prompt text file")
    metadata: str = Field(description="Path for the output metadata JSON file")
    evaluation: Optional[str] = Field(
        default=None,
        description="Path for the evaluation results JSON file",
    )

    @model_validator(mode="before")
    @classmethod
    def _accept_media_as_video(cls, values):
        """Accept ``output.media`` as an alias for ``output.video``."""
        if isinstance(values, dict) and "media" in values and "video" not in values:
            values["video"] = values.pop("media")
        return values


class DataSample(BaseModel):
    """A single input/output sample to process."""

    inputs: Optional[DataInputs] = Field(
        default=None,
        description="Input specification (null for text2world with no input media)",
    )
    output: DataOutput
