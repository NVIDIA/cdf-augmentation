<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Cosmos Data Factory (CDF) Augmentation Pipeline

This pipeline augments generic camera data through generative AI models. It supports multimodal inputs such as image, video, and text, and produces augmented outputs through backends such as Cosmos Transfer NVIDIA Inference Microservice (NIM), Cosmos Predict NIM, image-edit, and other models (for example, Veo 3.1, Gemma, and Gemini). Optional steps include captioning, template-based prompt generation, data preprocessing, and data verification.

Every model is reached over a **remote HTTP endpoint**. Each backend is one entry in the config `endpoints` list, so you can add or swap a model with a config change rather than a code change. No model weights ship in the image. Any server that speaks an OpenAI-compatible API (vLLM, vLLM-OMNI) or the NIM `/v1/infer` contract can act as a backend, as can a hosted service such as Gemini 3 Pro Image or Veo 3.1. That keeps the container lightweight and makes it straightforward to plug in your own models for augmentation.

## Pipeline Flow

The pipeline takes two inputs: source media (image, video, or text) and a single YAML config. It returns verified augmented data. The config drives every stage, so a run is fully defined by that one file.

An optional captioning stage sends the source media to a vision-language model (VLM) or large language model (LLM) to produce the prompts that steer generation. Prompts can also come from a template filled with variables, from fixed text, or from a prompt file. The augmentation workflow then passes the media and its prompt to a deployed model server (a NIM, a vLLM-OMNI server, or any other OpenAI-compatible inference server), which returns the augmented data.

Optional evaluators gate that output. Built-in hallucination checking flags motion artifacts that are absent from the source. Built-in attribute verification confirms requested attributes. The checker-agnostic Cosmos Evaluator integration can delegate one or more gates to an Arbitrator deployment. Quality failures can send the candidate back to generation for another attempt with an incremented seed. External service failures stop without spending a generation retry.

<p align="center">
  <img src="docs/Augmentation_v1.1.0.png" alt="CDF Augmentation pipeline framework" width="900" />
</p>

## Supported Endpoints

Transport is a pluggable **adapter**: the API contract an endpoint speaks. Any server that speaks one of these contracts works. Match the route your server exposes, then set that `adapter:` on the endpoint.

| API route | `adapter:` | Roles | Example config |
| --- | --- | --- | --- |
| `/v1/infer` | `nim` | **`video_transfer`**, **`video_predict`**, **`image_edit`**, `image2video` | `config_image_edit_attribute_nim.yaml` |
| `/v1/chat/completions` | `openai.chat.completions` | **`vlm`**, **`llm`**, `image_edit`, `video_predict` | `config_image_edit_attribute_chat_api.yaml` |
| `/v1/images/edits` | `openai.images.edits` | `image_edit` | `config_image_edit_attribute_images_api.yaml` |
| `/v1/videos/sync` | `openai.video.sync` | **`image2video`**, `video_transfer` | `config_image2video_cosmos3.yaml` |
| `/videos` → poll → `/content` | `openai.video.async` | `image2video` | `config_image2video_veo31.yaml` |
| `/health`, `/checkers`, `/dependency-graph`, `/process/async`, `/executions/{id}` | `cosmos.evaluator.arbitrator` | `evaluator` | `config_video_transfer_CT3_omni.yaml` |

**Bold** roles take that adapter by default. Set `adapter:` only to override. Tested backends: self-hosted NVIDIA NIM (image-edit), local vLLM / vLLM-OMNI (Qwen, Cosmos 3 Super), and hosted NVIDIA APIs (Gemma, Gemini, Veo 3.1). Cosmos Predict 2.5 is supported over `nim` as well, but ships no example config yet. Each row lists one representative config. Refer to the [config cookbook](configs/cookbook/README.md) for the full set, grouped by use case under `configs/cookbook/<use-case>/`.

For `cosmos.evaluator.arbitrator`, the `evaluator` endpoint addresses the
Arbitrator service. A cookbook can reuse its `vlm` and `llm` URL/model values
for checker settings, but Arbitrator must independently reach and authenticate
to those services.

## Requirements

- Docker. Inference itself needs **no GPU** and no NVIDIA Container Runtime: every model runs on your remote endpoints. Add the [NVIDIA Container Runtime](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) and `--gpus` only for these local processing requirements:
  - **Decoding H.264 video locally:** the image ships only the hardware `h264_cuvid` decoder (software AVC decode is off for licensing), so any configured stage that decodes H.264 in the augmentation container must use NVDEC. MPEG-4 Part 2, MJPEG, and VP9 retain software decoders.
  - **`data_processing.alignment`:** its mutual-information search is cupy-only with no CPU fallback, so this post-processor requires a GPU. Used mainly by the Defect Image Generation use case.

  Video *encoding* never needs a GPU: `transcode` writes VP9 with the software `libvpx-vp9` encoder (`h264_nvenc` is deliberately not built into the image).
- Reachable endpoint URLs for every role used by the selected config. Endpoints can be local (for example, vLLM or a local NIM) or remote (for example, hosted APIs).
- API keys only for endpoints that require authentication, passed through environment variables named by each endpoint's `api_key_env` (never hardcoded in YAML). Local endpoints need none.
- Optional: [multi-storage-client](https://nvidia.github.io/multi-storage-client/) configuration if your data lives in S3, GCS, Azure, or other remote storage.

## Installation

### Clone the Repository

```bash
git clone https://github.com/NVIDIA/cdf-augmentation.git
cd cdf-augmentation
```

The pipeline entrypoint is `modules/cli.py` (run it inside the container; refer to [Usage](#usage)).

### Set Up the API Keys

Skip this section for a fully local, unauthenticated deployment. Local vLLM servers and local NIMs need no key.

Each endpoint names its own environment variable through `api_key_env`. The key itself is never written in YAML, so it cannot leak through a CLI override or an error log:

```yaml
endpoints:
  - id: llm_gemma
    role: llm
    url: "https://integrate.api.nvidia.com/v1"
    model: "google/gemma-4-31b-it"
    api_key_env: NVIDIA_API_KEY    # the env var NAME, never the key itself
```

A key resolves in this order: the variable named by `api_key_env` → the role's default variable → no key. So `api_key_env` only needs setting when an endpoint uses something other than its role default:

| Role | Default variable |
| --- | --- |
| `vlm` | `VLM_API_KEY` |
| `llm` | `LLM_API_KEY` |
| `image_edit`, `image2video`, `video_transfer`, `video_predict` | `BUILD_NVIDIA_API_KEY` |

For generation, captioning, and built-in verification, an unset `api_key_env`
variable falls back to the role default rather than silently sending no key. An
external Cosmos evaluator reads only its explicitly named variable and fails
initialization if that variable is unset.

**Passing keys into the container.** Either put them in an env file:

```bash
cat > ./examples/.env <<'EOF'
# Only the variables your configs actually name. Local endpoints need none.
VLM_API_KEY=<your-vlm-api-key>
LLM_API_KEY=<your-llm-api-key>
BUILD_NVIDIA_API_KEY=<your-nvidia-api-key>  # role default for the generation roles
NVIDIA_API_KEY=<your-nvidia-api-key>        # hosted NVIDIA (integrate.api.nvidia.com, inference-api)
VEO_API_KEY=<your-veo-api-key>              # Veo 3.1 image-to-video
GEMINI_API_KEY=<your-gemini-api-key>        # hosted Gemini image edit

# Optional
LOG_LEVEL=INFO
EOF

docker run --rm --network host --env-file examples/.env ...
```

Alternatively, forward an already-exported variable. Under `sudo`, `-E` is required or the variable will not reach Docker:

```bash
export VEO_API_KEY=<your-veo-api-key>
sudo -E docker run --rm --network host -e VEO_API_KEY ...
```

Two common failures: `--env-file` makes Docker fail if the file does not exist, so omit the flag entirely for keyless runs. The `NGC_API_KEY` used to pull and start a NIM server is for that server's own container. The augmentation container never reads it.

> Endpoint URLs and model names belong in each config's `endpoints:` list, not in the env file. The legacy `VLM_ENDPOINT_URL` / `LLM_ENDPOINT_URL` / `*_ENDPOINT_MODEL` variables still override the **captioning and verification** consumers, but generation endpoints always come from `endpoints:`.

### Optional Storage Configuration

The pipeline uses [multi-storage-client](https://nvidia.github.io/multi-storage-client/) for unified local and cloud I/O. Configure it only when your configs reference remote paths (`s3://`, `gs://`, `az://`, or `https://`).

Provide configuration in one of these ways:

- Create `examples/secrets.json` and mount it to `/var/secrets/secrets.json`
- Set `MULTISTORAGECLIENT_CONFIGURATION` as an environment variable

Example `examples/secrets.json`:

```bash
cat > ./examples/secrets.json << 'EOF'
{
  "BUILD_NVIDIA_API_KEY": "<your-api-key>",
  "MULTISTORAGECLIENT_CONFIGURATION": {
    "profiles": {
      "<your-profile-name>": {
        "storage_provider": {
          "type": "s3",
          "options": {
            "base_path": "<your-base-path>",
            "region_name": "<your-region>",
            "endpoint_url": "<your-endpoint-url>",
            "infer_content_type": true
          }
        },
        "credentials_provider": {
          "type": "S3Credentials",
          "options": {
            "access_key": "<your-access-key>",
            "secret_key": "<your-secret-key>"
          }
        }
      }
    },
    "path_mapping": {
      "<your-https-bucket-url>/": "msc://<your-profile-name>/",
      "s3://<your-bucket>/": "msc://<your-profile-name>/"
    }
  }
}
EOF
```

If you do not provide storage configuration, the pipeline uses local filesystem paths.

## Usage

The commands below assume your user can access the Docker daemon. Prefix them with `sudo` if not, and use `sudo -E` when forwarding exported environment variables.

### Get the Docker Image

The pipeline ships a single endpoint-only image (remote-API inference, no local Cosmos/torch), published on NGC for `linux/amd64` and `linux/arm64`:

```bash
docker pull nvcr.io/nvidia/cdf-augmentation:1.2.0
```

To build the image from source instead, run the following from the repository root, then use `cdf-augmentation:1.2.0` in place of the NGC image name in the commands below:

```bash
DOCKER_BUILDKIT=0 docker build -t cdf-augmentation:1.2.0 -f docker/Dockerfile .
```

The image source-builds FFmpeg, Python, OpenCV, and PyAV against a restricted codec allow-list, so the first build can take several minutes. Release builds are validated with the legacy builder (`DOCKER_BUILDKIT=0`).

`examples/docker-compose.yml` wraps the build and run steps below if you prefer Compose.

### Deploy the Model Endpoints

The container calls models over HTTP, so bring up the endpoints your config names before running it. A first run of `config_video_transfer_CT3_omni.yaml` needs a VLM and Cosmos3-Super:

```bash
# VLM on :8000
docker run -d --name vlm --runtime nvidia --gpus '"device=0"' --ipc=host \
  -p 8000:8000 \
  -e NVIDIA_DRIVER_CAPABILITIES=compute,utility,video \
  nvcr.io/nvidia/vllm:26.05.post1-py3 \
  vllm serve Qwen/Qwen3.8-27B-FP8 --host 0.0.0.0 --port 8000 \
    --trust-remote-code --max-model-len 32768 --dtype auto \
    --max-num-seqs 16 \
    --default-chat-template-kwargs '{"enable_thinking": false}'

# Cosmos3-Super on :8001
docker run -d --name cosmos3-transfer --runtime nvidia --gpus '"device=2,3"' \
  --ipc=host -p 127.0.0.1:8001:8001 \
  vllm/vllm-omni@sha256:6d2630c7d637b699557573f2c3fee8df5d4d0cd718977aa22549ed6a6ef30587 \
  vllm serve nvidia/Cosmos3-Super \
    --omni --model-class-name Cosmos3OmniDiffusersPipeline \
    --no-guardrails --allowed-local-media-path / \
    --cfg-parallel-size 1 --use-hsdp --hsdp-shard-size 2 \
    --port 8001 --init-timeout 1800
```

Each returns `200` after it is ready. Check vLLM on `/health`:

```bash
curl -s -o /dev/null -w '%{http_code}\n' http://localhost:8000/health
curl -s -o /dev/null -w '%{http_code}\n' http://localhost:8001/health
```

Model servers take minutes to load. A run started too early fails with connection refused. Other backends follow the same shape. Refer to each cookbook config's `endpoints:` block for the ports and models it expects.

### Run Inference

The image's entrypoint is the CLI (`uv run --no-sync /workspace/modules/cli.py`), so pass CLI arguments directly. Mount your `configs/` and `data/`, then select a config:

```bash
docker run --rm --network host \
  --gpus '"device=1"' \
  -v "$(pwd)/configs:/workspace/configs" \
  -v "$(pwd)/data:/workspace/data" \
  nvcr.io/nvidia/cdf-augmentation:1.2.0 \
  --config configs/cookbook/video-data-augmentation/config_video_transfer_CT3_omni.yaml \
  pipeline.retry=2
```

- `--gpus` is pinned to a GPU **not** running a model server. The evaluators decode the input video through NVDEC, and sharing a saturated GPU can crash it. H.264 decoding is hardware-only in this image (`h264_cuvid`); no software AVC decoder is included, so H.264 has no CPU fallback. MPEG-4 Part 2, MJPEG, and VP9 retain software decoders. Drop the flag entirely if your config runs no evaluators, no `data_processing.alignment`, and decodes no H.264. Refer to [Requirements](#requirements).
- `--network host` lets the container reach endpoints on the host's `localhost`. When all your endpoints are remote URLs, prefer the default bridge network. Refer to *Security Notes* in [pipeline-operations.md](skills/cdf-augmentation/references/pipeline-operations.md#security-notes).
- Add `--env-file examples/.env` when an endpoint needs an API key. Refer to [Set Up the API Keys](#set-up-the-api-keys). It is omitted here because a local, unauthenticated deployment needs no keys, and `--env-file` fails if the file does not exist.
- If you use cloud storage, also mount your secrets file: `-v "$(pwd)/examples/secrets.json:/var/secrets/secrets.json:ro"`.

For Video Data Augmentation (VDA) service deployment, networking, storage, and test commands, refer to the [Video Data Augmentation cookbook README](configs/cookbook/video-data-augmentation/README.md).

> **Note:** The container runs as user `nvidia` (UID 10000), which must be able to read the mounted `configs/`/`data/` and write outputs under `data/`. Create the output directory up front and give it to that user, or the run fails on its first write:
>
> ```bash
> mkdir -p data/video/output && sudo chown 10000:10000 data/video/output
> ```
>
> Alternatively, run the container as your own user so ownership matches the host: `docker run --rm --user "$(id -u):$(id -g)" ...`. Avoid world-writable mounts. If neither option works, grant access to just UID 10000 with an ACL (`setfacl -R -m u:10000:rwX data/`) rather than `chmod -R o+rwX`.

## Configuration and Reference

Pipeline behavior is controlled by a single YAML config file with seven sections: `data`, `endpoints`, `pipeline`, `captioning`, `augmentation`, `data_processing`, and `evaluators`. The full schema, per-section examples, troubleshooting, and security guidance live in the **augmentation skill** reference docs (loaded on demand by agents, and browsable here):

- [Configuration schema and per-section examples](skills/cdf-augmentation/references/configuration-schema.md)
- [Config decision tree: model and captioning selection](skills/cdf-augmentation/references/config-decision-tree.md)
- [Captioning strategies](skills/cdf-augmentation/references/captioning-strategy-guide.md)
- [Evaluator setup](skills/cdf-augmentation/references/evaluator-setup-guide.md)
- [Troubleshooting](skills/cdf-augmentation/references/troubleshooting.md)
- Security notes: refer to *Security Notes* in [pipeline-operations.md](skills/cdf-augmentation/references/pipeline-operations.md#security-notes)

Refer to *Reference Files* in [`skills/cdf-augmentation/SKILL.md`](skills/cdf-augmentation/SKILL.md#reference-files) for the full index.

## Contributing

Contributions are welcome. All commits must be signed off under the [Developer Certificate of Origin](https://developercertificate.org/). Refer to [CONTRIBUTING.md](CONTRIBUTING.md) for sign-off instructions and the full DCO text.

## Notice

**NOTICE AND DISCLAIMER:** This software automatically retrieves, accesses or interacts with external materials. Those retrieved materials are not distributed with this software and are governed solely by separate terms, conditions and licenses. You are solely responsible for finding, reviewing and complying with all applicable terms, conditions, and licenses, and for verifying the security, integrity and suitability of any retrieved materials for your specific use case. This software is provided "AS IS", without warranty of any kind. The author makes no representations or warranties regarding any retrieved materials, and assumes no liability for any losses, damages, liabilities or legal consequences from your use or inability to use this software or any retrieved materials. Use this software and the retrieved materials at your own risk.
