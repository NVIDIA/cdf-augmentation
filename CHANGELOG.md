---
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: "Release Notes"
sidebar-title: "Release Notes"
description: "Changelog and feature history for Cosmos Data Factory (CDF) Augmentation releases."
description-agent: "Includes the CDF Augmentation release notes. Use when users ask about recent changes, the release cadence, or where to track versioned assets."
keywords: ["cdf augmentation release notes", "cdf augmentation changelog"]
content:
  type: "reference"
---

Use this page to track Cosmos Data Factory (CDF) Augmentation release highlights.
The latest release appears first.

## v1.2.0

CDF Augmentation v1.2.0 tags the endpoint-only image as `cdf-augmentation:1.2.0`. Video data augmentation cookbooks use one Qwen3.8-27B-FP8 service for both VLM and LLM captioning and evaluator Q&A.

## v1.1.0

CDF Augmentation v1.1.0 is the bring-your-own-model (BYOM) release. The fixed per-role endpoint block is replaced by an endpoint registry: a flat list where every entry declares its own role, API contract, and credentials. Generation is split into pluggable transport adapters behind one generic executor. Any OpenAI-compatible or NIM server (vLLM, vLLM-OMNI, a Visual GenAI NIM on `/v1/infer`, or a hosted service such as Gemini 3 Pro Image or Veo 3.1) can now be driven from config alone, with no code change and no model weights in the image. Refer to *Supported Endpoints* in the [README](README.md#supported-endpoints) for the full contract-to-server mapping.

**Breaking.** `endpoints` changed from a mapping to a list and `augmentation.model.executor_type` was removed, so v1.0.x configs must be migrated. The refreshed examples in `configs/` show the new shape. Several example configs were also renamed. Refer to the [config cookbook](configs/cookbook/README.md) for current use-case names.

### Bring Your Own Model

- Replaced the fixed `endpoints` mapping (`vlm`, `llm`, `cosmos_transfer`, `cosmos_predict`, `image_edit`) with an **endpoint registry**: a list whose entries each carry `role`, `url`, `model`, and optional `adapter`, `id`, `api_key_env`, and `timeout`.
- `augmentation.model.name` is now a free-form string resolved against the registry by endpoint `id`, then `role`, then the legacy model-name-to-role map, so adding a model no longer means adding an enum value and a generator class.
- Several endpoints can share a role. Captioning and evaluator sections select one with `endpoint_id`. Two or more same-role endpoints must each declare an `id`, because an environment-variable override cannot disambiguate them.
- Roles exercised by the shipped configs: `vlm`, `llm`, `image_edit`, `image2video`, `video_transfer`, `video_predict`.

### Media Handling and Evaluation

- Added `data_processing.transcode`, normalizing generated video to VP9 so output codec no longer varies by model. A source already in VP9 is copied byte-identically unless `force` is set, and still images are skipped rather than wrapped into a one-frame MP4.
- Added the libvpx VP9 encoder to the FFmpeg build, plus both a software and a hardware VP9 decoder, so VP9 output decodes without a GPU.
- Added an optional CPU input-resize preprocessing step (`data_processing.preprocessing.resize`) with target megapixels or explicit width/height, grid snapping, and selectable interpolation. Use this when a model needs a larger input than the source crop.
- Added deterministic VLM-template captioning (`captioning.template`, which requires `captioning.vlm`): the VLM describes the scene and a validated template composes the final prompt across three attribute axes, with no second LLM call.
- Verification now samples frames with PyAV instead of `cv2.VideoCapture`, and `vlm_verification.frames` controls how many frames the VLM receives.
- The hallucination check tolerates an input/output frame-count mismatch instead of failing, and no longer drops a surplus input frame.

## v1.0.1

CDF Augmentation v1.0.1 adds arm64 build support, a configurable request timeout and automatic retries for image-edit generation, Image Attribute Augmentation workflow updates, and fail-fast behavior on generation failures and undecodable input instead of falsely reporting success.

- Added `arm64`/`aarch64` build support so the images build and run on ARM hosts in addition to `x86_64`.
- Added a configurable per-request timeout (`pipeline.request_timeout`, default 120s) that bounds each generation endpoint call and serves as the base for exponential retry backoff.
- Added automatic retries with exponential backoff (`request_timeout`, then 2×, 4×, …) up to `pipeline.retry` on generation/endpoint failure; the OpenAI SDK's own retries are disabled so `pipeline.retry` is the single retry path.
- Added a dedicated Image Attribute Augmentation workflow and expanded augmentation guidance, making image attribute augmentation easier to configure and run at scale.
- Failed samples now log the failed input and exit non-zero instead of falsely reporting success. The run no longer calls `tracker.complete()` when a sample exhausts its retries.
- Fail fast on undecodable input videos in the local `Cosmos` executor instead of failing later.

## v1.0.0

Initial release of CDF Augmentation, the media augmentation pipeline (captioning, generation, evaluation) with support for `cosmos-transfer2.5`, `cosmos-predict`, and `image-edit` models.
