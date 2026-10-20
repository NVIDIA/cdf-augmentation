# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build & Run Commands

```bash
# Run the pipeline
uv run modules/cli.py --config configs/<config_file>.yaml

# Run with OmegaConf CLI overrides
uv run modules/cli.py --config configs/<config_file>.yaml \
  data.0.inputs.rgb=/path/to/video.mp4

# Run tests
uv run pytest tests/
uv run pytest tests/schema/test_schema.py -v          # schema validation tests only
uv run pytest tests/schema/test_schema.py::TestValidConfigs::test_minimal_cosmos_transfer -v  # single test

# Lint
uv run ruff check modules/
uv run ruff format --check modules/

# Docker inference — no GPU needed for remote inference. The image ENTRYPOINT is
# already `uv run --no-sync /workspace/modules/cli.py`, so pass only CLI args.
docker run --rm --network host \
  -v "$(pwd)/configs:/workspace/configs" \
  -v "$(pwd)/data:/workspace/data" \
  nvcr.io/nvidia/cdf-augmentation:1.2.0 \
  --config configs/<config_file>.yaml
```

Add `--env-file examples/.env` only when an endpoint needs an API key.

Add `--gpus` only for the `data_processing.alignment` post-processor (cupy) and
whenever a configured stage must decode **H.264** locally,
since the image ships only the hardware `h264_cuvid` decoder. VP9 decodes in software.
`--network host` is needed only to reach endpoints on the host's `localhost`.
For VDA-specific service and network setup, see
[`configs/cookbook/video-data-augmentation/README.md`](configs/cookbook/video-data-augmentation/README.md).

## Architecture

The pipeline processes media through: **Captioning -> Generation -> Evaluation**, orchestrated by `modules/cli.py`.

### Unified Pydantic Schema (`modules/aug_utils/schema/`)

All YAML configs are validated against a single `PipelineConfig` root model. The schema has 7 top-level sections:

| Section | Purpose |
|---------|---------|
| `data` | List of input/output sample paths (rgb, controls, video, caption, metadata) |
| `endpoints` | **List** of endpoint registry entries — see below |
| `pipeline` | Retry count, request timeout, logging level, evaluation settings (strict, retain_failures) |
| `captioning` | `vlm`, `llm`, and/or `template` sub-sections — captioner type is **inferred from which sub-sections are present** |
| `augmentation` | Model name (free-form string), generation parameters, control modalities |
| `data_processing` | `preprocessing.resize` (before generation), `alignment` and `transcode` (after) |
| `evaluators` | Ordered quality checks configured for the pipeline |

Cross-section validation in `config.py` ensures endpoint requirements match the configured model and captioning strategy.

### Endpoint Registry (BYOM)

`endpoints` is a **flat list**, not a per-role mapping. Each entry declares its own role and API contract, so any OpenAI-compatible or NIM server can be driven from config alone:

```yaml
endpoints:
  - id: vlm_qwen              # optional; REQUIRED when 2+ entries share a role
    role: vlm                 # vlm | llm | image_edit | image2video | video_transfer | video_predict
    url: "http://localhost:8000/v1"
    model: "Qwen/Qwen3.6-27B-FP8"
    adapter: openai.chat.completions   # optional; defaults per role
    api_key_env: VLM_API_KEY  # env var NAME — never a literal key
    timeout: 600              # optional per-endpoint override
```

Consumers (captioning, evaluators) select an endpoint with `endpoint_id`; generation resolves `augmentation.model.name` against the registry by `id`, then `role`, then the legacy model-name→role map (`aug_utils/endpoint_registry.py`).

### Adapters and Executor

Transport is a pluggable **adapter**; execution is one generic `BaseExecutor`. There are no per-model generator classes and no `executor_type` — all inference is remote.

| Adapter | API route | Default for roles |
|---------|-----------|-------------------|
| `openai.chat.completions` | `POST /v1/chat/completions` | `vlm`, `llm` |
| `openai.images.edits` | `POST /v1/images/edits` (multipart) | — |
| `nim` | `POST /v1/infer` | `image_edit`, `video_transfer`, `video_predict` |
| `openai.video.sync` | `POST /v1/videos/sync` (multipart → MP4 bytes) | `image2video` |
| `openai.video.async` | create → poll → download | — |
| `passthrough` | none (echoes input; test seam) | — |

`KNOWN_ADAPTERS` in `aug_utils/schema/adapters.py` mirrors the `ADAPTERS` dict in `generation/factory.py` — **update both** when adding one. Only explicitly-set generation params are sent on the wire (`exclude_unset`), so one model's defaults never leak into another's payload.

### Factory Dispatch Pattern

Both `captioning/factory.py` and `generation/factory.py` accept the full config and dispatch on field presence:

- **Captioning** (`create_captioner`): `llm.text` → TextCaptioner, `llm.file_path` → FileCaptioner, `template` (requires `vlm`) → template captioner, vlm+llm → VLMLLMCaptioner, vlm only → VLMCaptioner, llm only → LLMCaptioner. Combining `vlm` with `llm.text`/`llm.file_path`, or `template` with `llm`, is an error.
- **Generation** (`create_generator`): resolves the endpoint, picks its adapter, wraps it in an executor.

### Config Validation Flow

`modules/aug_utils/common.py:validate_config_structure()` wraps Pydantic validation. Returns the validated `PipelineConfig` model or `None` on failure. Downstream code still uses raw dicts in some places (accessed via `config_dict`).

### Evaluation Pipeline

Evaluators run in order after generation, with retry on failure (seed incremented each attempt):
1. **Hallucination check** — optical-flow-based motion artifact detection between input/output
2. **Attribute verification** — LLM generates MCQ questions from variables, VLM answers them against the output
3. **VLM verification** — standalone VLM-based quality check

### Module Import Structure

`pyproject.toml` sets `pythonpath = [".", "modules"]`. Modules import each other relative to `modules/` (e.g., `from captioning.factory import create_captioner`, not `from modules.captioning...`).

### Storage Abstraction

All file I/O uses `multistorageclient` (aliased as `msc`) which transparently handles local paths, S3, GCS, Azure, and HTTP URLs. Use `msc.open()` and `msc.is_file()` instead of built-in `open()` for data paths.

### Environment Variables

**API keys never come from config.** `Endpoint` has no `api_key` field — only `api_key_env`, naming the env var to read. `resolve_api_key()` (`generation/adapters/base.py`) tries `api_key_env` first, then the role default: `VLM_API_KEY`, `LLM_API_KEY`, and `BUILD_NVIDIA_API_KEY` for all generation roles. A stray `api_key:` in a config is silently ignored, not rejected.

**URL overrides apply to captioning/evaluator consumers only.** `VLM_ENDPOINT_URL`, `LLM_ENDPOINT_URL`, and `LLM_CAPTION_ENDPOINT_URL` take precedence over the config's `url` for `vlm`/`llm` roles. Generation endpoints resolve purely from the registry — `COSMOS_ENDPOINT_URL`, `COSMOS_PREDICT_ENDPOINT_URL`, `IMAGE_EDIT_ENDPOINT_URL`, and `IMAGE_EDIT_API_KEY` still appear in `validate_environment()`'s log list but are **legacy and no longer applied**.
