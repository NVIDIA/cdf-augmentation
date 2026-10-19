# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unified pipeline configuration schema.

This package defines the Pydantic models that serve as the single source of
truth for the YAML configuration structure used by the augmentation pipeline.
Import ``PipelineConfig`` and call ``PipelineConfig(**yaml_dict)`` to validate
and parse a raw config dictionary.
"""

from .adapters import KNOWN_ADAPTERS
from .base import BaseInferenceParameters
from .data import ControlInputs, DataInputs, DataOutput, DataSample, PromptAttributes
from .endpoints import Endpoint
from .pipeline import EvaluationSettings, LoggingConfig, PipelineSettings
from .captioning import (
    CaptioningConfig,
    LLMCaptioningConfig,
    LLMParameters,
    TemplateCaptioningConfig,
    TemplateAttributeCatalog,
    VLMCaptioningConfig,
    VLMParameters,
)
from .augmentation import (
    AugmentationConfig,
    AugmentationParameters,
    ModalitiesConfig,
    ModelConfig,
    ModelNameEnum,
    PredictInferenceType,
)
from .data_processing import AlignmentParameters, DataProcessingConfig
from .preprocessing import PreprocessingConfig, ResizeConfig
from .evaluators import (
    AttributeVerificationEvaluator,
    CosmosEvaluator,
    CosmosEvaluatorCheck,
    EvaluatorEntry,
    ExtraQuestion,
    HallucinationCheckEvaluator,
    HallucinationParams,
    NaturalCaptionConfig,
    QuestionGenerationConfig,
    VLMVerificationEvaluator,
)
from .config import PipelineConfig

__all__ = [
    "PipelineConfig",
    "KNOWN_ADAPTERS",
    "BaseInferenceParameters",
    "ControlInputs",
    "DataInputs",
    "DataOutput",
    "PromptAttributes",
    "DataSample",
    "Endpoint",
    "EvaluationSettings",
    "LoggingConfig",
    "PipelineSettings",
    "CaptioningConfig",
    "LLMCaptioningConfig",
    "LLMParameters",
    "TemplateAttributeCatalog",
    "TemplateCaptioningConfig",
    "VLMCaptioningConfig",
    "VLMParameters",
    "AugmentationConfig",
    "AugmentationParameters",
    "ModalitiesConfig",
    "ModelConfig",
    "ModelNameEnum",
    "PredictInferenceType",
    "AlignmentParameters",
    "DataProcessingConfig",
    "PreprocessingConfig",
    "ResizeConfig",
    "AttributeVerificationEvaluator",
    "CosmosEvaluator",
    "CosmosEvaluatorCheck",
    "EvaluatorEntry",
    "ExtraQuestion",
    "HallucinationCheckEvaluator",
    "HallucinationParams",
    "NaturalCaptionConfig",
    "QuestionGenerationConfig",
    "VLMVerificationEvaluator",
]
