# Cosmos Data Factory (CDF) Video Data Augmentation

This cookbook contains one end-to-end use case and one focused world scenario model (WSM) software quality assurance (SQA) case.

The following table lists the pipelines in this cookbook:

| Pipeline | Config |
|---|---|
| Cosmos3-Super with Hallucination and Attribute Verification | [config_video_transfer_CT3_omni.yaml](config_video_transfer_CT3_omni.yaml) |
| Cosmos3-Super with an Uploaded WSM Control | [config_video_transfer_CT3_wsm_omni.yaml](config_video_transfer_CT3_wsm_omni.yaml) |

Complete the common setup for the end-to-end case. The WSM case only needs
[Get the Augmentation Image](#get-the-augmentation-image) and
[Configure Storage](#configure-storage), plus the Cosmos3-Super service from the
end-to-end case. It does not call the shared captioning or
evaluator services.

Run commands from the augmentation repository root and use one shell for an
entire case. Docker pulls missing images when the services start.

## Requirements

- Linux with Docker 28.1.1 or later, Docker Compose v2, and the NVIDIA container runtime.
- Docker access without `sudo`.
- `curl`, `jq`, `openssl`, and the AWS CLI.
- Docker access to the required NGC images.
- `S8K_ACCESS_KEY` and `S8K_SECRET_KEY` from an approved secret store.
- A SwiftStack bucket readable by the augmentation and evaluator containers.
- GPU `0` for VLM/LLM, with at least 80 GB of memory.
- GPUs `2,3` for the Cosmos3-Super service.
- The container names used below must be unused.

The examples do not require a Cosmos Evaluator client API key. The database
keys below are internal credentials between the evaluator services.
Both pipeline cases share `~/.cosmos_evaluator/deployment/msc.json`.

## Common Setup

### Get the Augmentation Image

Pull the published image from NGC:

```bash
docker pull nvcr.io/nvidia/cdf-augmentation:1.2.0
```

To test changes in the current checkout instead, build `cdf-augmentation:dev`
and use it in place of the NGC image name in the commands below:

```bash
DOCKER_BUILDKIT=0 docker build \
  --build-arg TARGETARCH=amd64 \
  -t cdf-augmentation:dev \
  -f docker/Dockerfile .
```

### Configure Storage

Provide the storage credentials as environment variables:

```bash
export COSMOS_DEPLOY_DIR="${HOME}/.cosmos_evaluator/deployment"
mkdir -p "${COSMOS_DEPLOY_DIR}"
chmod 700 "${COSMOS_DEPLOY_DIR}"

export VDA_BUCKET="<BUCKET>"
export S8K_ENDPOINT_URL="<S8K_ENDPOINT_URL>"
export S8K_ACCESS_KEY="<S8K_ACCESS_KEY>"
export S8K_SECRET_KEY="<S8K_SECRET_KEY>"

test "${VDA_BUCKET}" != '<BUCKET>'
test "${S8K_ENDPOINT_URL}" != '<S8K_ENDPOINT_URL>'
```

Create `${COSMOS_DEPLOY_DIR}/msc.json`. Replace `<BUCKET>` with the same bucket
name as `VDA_BUCKET`, replace `<S8K_ENDPOINT_URL>` with the same URL as
`S8K_ENDPOINT_URL`, and keep the environment references literal:

```json
{
  "profiles": {
    "<BUCKET>": {
      "storage_provider": {
        "type": "s8k",
        "options": {
          "base_path": "<BUCKET>",
          "region_name": "us-east-1",
          "endpoint_url": "<S8K_ENDPOINT_URL>"
        }
      },
      "credentials_provider": {
        "type": "S3Credentials",
        "options": {
          "access_key": "${S8K_ACCESS_KEY}",
          "secret_key": "${S8K_SECRET_KEY}"
        }
      }
    }
  },
  "path_mapping": {
    "s3://<BUCKET>/": "msc://<BUCKET>/"
  }
}
```

### Upload the Shared Test Input

Upload the repository's test video once and keep its S3 URI in the shell. The
attribute-verification case uses this object as `data.0.inputs.rgb`:

```bash
export VDA_INPUT_URI="s3://${VDA_BUCKET}/inputs/video-data-augmentation/sample_input_stable93.mp4"

test -f data/sample_input_stable93.mp4

AWS_ACCESS_KEY_ID="${S8K_ACCESS_KEY}" \
AWS_SECRET_ACCESS_KEY="${S8K_SECRET_KEY}" \
AWS_DEFAULT_REGION="us-east-1" \
aws --endpoint-url "${S8K_ENDPOINT_URL}" \
  s3 cp data/sample_input_stable93.mp4 "${VDA_INPUT_URI}"
```

### Start the Shared Services

Create the `cdf` network, generate temporary database keys, and start the
VLM/LLM and Cosmos Evaluator services from
[video-data-augmentation-services.yml](../../../examples/video-data-augmentation-services.yml):

```bash
docker network inspect cdf >/dev/null 2>&1 || \
  docker network create cdf

export POSTGRES_PASSWORD="$(openssl rand -hex 32)"
export DATABASE_READ_KEY="$(openssl rand -hex 32)"
export DATABASE_ADMIN_KEY="$(openssl rand -hex 32)"

docker compose \
  -f examples/video-data-augmentation-services.yml \
  up -d --wait --wait-timeout 1800
```

Check the shared endpoints. Each health command prints **200** when the
service is ready:

```bash
curl -s -o /dev/null -w '%{http_code}\n' http://localhost:8000/health
curl -s -o /dev/null -w '%{http_code}\n' http://localhost:8080/health
curl -fsS http://localhost:8080/checkers | jq
```

The checker list must include `metropolis.hallucination` and
`metropolis.attribute_verification`.

## Cosmos3-Super with Hallucination and Attribute Verification

This case runs Cosmos3-Super transfer with VLM captioning, LLM prompt
generation, Hallucination, and Attribute Verification. Every container joins
the `cdf` network and uses the endpoint names from the config.

### Start Cosmos3-Super

```bash
docker run -d \
  --name cosmos3-transfer \
  --network cdf \
  --runtime nvidia \
  --gpus '"device=2,3"' \
  --ipc=host \
  -p 127.0.0.1:8001:8001 \
  vllm/vllm-omni@sha256:6d2630c7d637b699557573f2c3fee8df5d4d0cd718977aa22549ed6a6ef30587 \
  vllm serve nvidia/Cosmos3-Super \
    --omni \
    --model-class-name Cosmos3OmniDiffusersPipeline \
    --no-guardrails \
    --allowed-local-media-path / \
    --cfg-parallel-size 1 \
    --use-hsdp \
    --hsdp-shard-size 2 \
    --port 8001 \
    --init-timeout 1800

# Continue after this prints 200.
curl -s -o /dev/null -w '%{http_code}\n' http://localhost:8001/health
```

### Run Augmentation

The input uses the shared S3 object from common setup. Endpoint URLs come
directly from the config:

```bash
docker run --rm \
  --network cdf \
  --user "$(id -u):$(id -g)" \
  -e HOME=/tmp \
  -e S8K_ACCESS_KEY \
  -e S8K_SECRET_KEY \
  -e MSC_CONFIG=/run/secrets/cosmos-msc.json \
  -v "${COSMOS_DEPLOY_DIR}/msc.json:/run/secrets/cosmos-msc.json:ro" \
  -v "$(pwd)/configs:/workspace/configs:ro" \
  -v "$(pwd)/data:/workspace/data:ro" \
  nvcr.io/nvidia/cdf-augmentation:1.2.0 \
  --config configs/cookbook/video-data-augmentation/config_video_transfer_CT3_omni.yaml \
  "data.0.inputs.rgb=${VDA_INPUT_URI}" \
  "data.0.output.video=s3://${VDA_BUCKET}/\${data.0.output.prefix}/ct3_omni_snow_night.mp4" \
  "data.0.output.caption=s3://${VDA_BUCKET}/\${data.0.output.prefix}/ct3_omni_snow_night.txt" \
  "data.0.output.metadata=s3://${VDA_BUCKET}/\${data.0.output.prefix}/ct3_omni_snow_night_metadata.json" \
  "data.0.output.evaluation=s3://${VDA_BUCKET}/\${data.0.output.prefix}/ct3_omni_snow_night_eval.json"
```

Results are available at `s3://${VDA_BUCKET}/output/`.

## Cosmos3-Super with an Uploaded WSM Control

This focused case verifies the HTTP WSM path: the augmentation client uploads
the RGB video as `input_reference`, uploads the WSM video as
`control_reference`, and sends `control_type=wsm`. It uses a fixed prompt so no
captioning or evaluator endpoint is involved.

Start the Cosmos3-Super service as described in [Start Cosmos3-Super](#start-cosmos3-super). Place a matching RGB
and WSM video pair under `data/`, or override both paths in the command below.
The reference parameters generate 121 frames at 10 FPS; when using a different
pair, update `num_frames`, `extra_params.max_frames`, and
`extra_params.num_video_frames_per_chunk` together.

```bash
test -f data/sample_input.mp4
test -f data/sample_wsm.mp4

export CT3_WSM_OUTPUT_PREFIX="output_wsm"

docker run --rm \
  --network cdf \
  --user "$(id -u):$(id -g)" \
  -e HOME=/tmp \
  -e S8K_ACCESS_KEY \
  -e S8K_SECRET_KEY \
  -e MSC_CONFIG=/run/secrets/cosmos-msc.json \
  -v "${COSMOS_DEPLOY_DIR}/msc.json:/run/secrets/cosmos-msc.json:ro" \
  -v "$(pwd)/configs:/workspace/configs:ro" \
  -v "$(pwd)/data:/workspace/data:ro" \
  nvcr.io/nvidia/cdf-augmentation:1.2.0 \
  --config configs/cookbook/video-data-augmentation/config_video_transfer_CT3_wsm_omni.yaml \
  "data.0.output.prefix=${CT3_WSM_OUTPUT_PREFIX}" \
  "data.0.output.video=s3://${VDA_BUCKET}/\${data.0.output.prefix}/ct3_omni_wsm.mp4" \
  "data.0.output.caption=s3://${VDA_BUCKET}/\${data.0.output.prefix}/ct3_omni_wsm.txt" \
  "data.0.output.metadata=s3://${VDA_BUCKET}/\${data.0.output.prefix}/ct3_omni_wsm_metadata.json"
```

Results are available at
`s3://${VDA_BUCKET}/${CT3_WSM_OUTPUT_PREFIX}/`.

## Inspect and Stop Services

Show the shared-service logs:

```bash
docker compose \
  -f examples/video-data-augmentation-services.yml \
  logs --tail 200
```

Show the transfer-service log:

```bash
docker logs --tail 200 cosmos3-transfer
```

Stop the shared services and remove the transfer container if it exists:

```bash
docker compose \
  -f examples/video-data-augmentation-services.yml \
  down

if docker container inspect cosmos3-transfer >/dev/null 2>&1; then
  docker rm --force cosmos3-transfer
fi
```

Images and the external `cdf` network are preserved.
