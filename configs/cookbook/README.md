# Config Cookbook

Example configs grouped by use case. Each folder holds everything that use case
needs: runtime configs plus any distribution config that generates them in batch.

To get started, use
[video-data-augmentation/config_video_transfer_CT3_omni.yaml](video-data-augmentation/config_video_transfer_CT3_omni.yaml):
VLM captioning, LLM prompt generation, Cosmos3-Super, then hallucination and
attribute checks.
It is also the Docker quickstart example.

The following table lists cookbook folders by use case:

| Folder | Configs | What It Covers |
|--------|---------|----------------|
| [event-video-generation](event-video-generation/) | 8 | All image-to-video work. The Smart Spaces pipeline (seed image T2I → event video I2V with Cosmos 3 Super, runtime + distribution for each stage), plus standalone Cosmos 3 and Veo 3.1 I2V examples including the deterministic VLM-template captioning variant |
| [image-attribute-augmentation](image-attribute-augmentation/) | 6 | Person clothing/accessory edits across the three image-edit API contracts, plus hosted Gemini and hosted-Gemma LLM variants, and a 1,000-sample distribution |
| [defect-image-generation](defect-image-generation/) | 3 | PCB defect rendering with MI alignment back into the input frame, across three API contracts |
| [video-data-augmentation](video-data-augmentation/) | 3 | Cosmos Transfer 3 over vLLM-OMNI (including a focused WSM-upload SQA reference), plus the generic batch-generation example (weather / time-of-day / road-condition variables) |

## Run a Config

```bash
uv run modules/cli.py --config configs/cookbook/<folder>/<config>.yaml
```

Distribution configs go through the batch generator instead:

```bash
uv run modules/config_distribution_generation/generate_augmentation_configs.py \
  --workflow configs/cookbook/<folder>/<distribution_config>.yaml
```

Every config points at example endpoints and `/path/to/...` or `s3://<...>`
placeholders. Before you run, repoint `endpoints[].url`, `data`, and the distribution
`data_dir` / `config_output` / `output_root` fields at your own deployment. A
distribution config's `example_augmentation_config` resolves relative to that config's
own directory.

Adapter contracts, captioning strategies, and evaluator wiring are documented in
[config-decision-tree.md](../../skills/cdf-augmentation/references/config-decision-tree.md).
