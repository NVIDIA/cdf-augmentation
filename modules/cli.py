# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
from copy import deepcopy
import os
import sys
import logging
import yaml
import json
import time

import multistorageclient as msc
from omegaconf import OmegaConf

from captioning.factory import create_captioner
from generation.factory import create_generator
from generation.utils import generate_with_retries
from verification.llm_question_generator import LLMQuestionGenerator
from verification.vlm_verifier import VLMVerifier
from verification.core import AttributeVerifier
from verification.hallucination_checker import HallucinationChecker
from verification.cosmos_evaluator import (
    CosmosEvaluatorClient,
    CosmosEvaluatorError,
    CosmosEvaluationOutcome,
)
from aug_utils.common import (
    redact_overrides,
    validate_config_structure,
    validate_sample_data_availability,
)
from aug_utils.endpoint_registry import effective_adapter, select_endpoint
from aug_utils.nvcf import load_secrets, NVCFProgressTracker


def validate_environment(logger: logging.Logger):
    """Log which optional env vars are set."""
    optional_env_vars = [
        "LOG_LEVEL",
        "VLM_API_KEY",
        "LLM_API_KEY",
        "VLM_ENDPOINT_URL",
        "VLM_ENDPOINT_MODEL",
        "LLM_ENDPOINT_URL",
        "LLM_ENDPOINT_MODEL",
        "COSMOS_ENDPOINT_URL",
        "COSMOS_ENDPOINT_MODEL",
        "COSMOS_PREDICT_ENDPOINT_URL",
        "IMAGE_EDIT_ENDPOINT_URL",
        "IMAGE_EDIT_API_KEY",
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "AWS_DEFAULT_REGION",
        "AWS_ENDPOINT_URL",
        "AWS_S3_BUCKET",
        "AWS_S3_ADDRESSING_STYLE",
    ]
    for var in optional_env_vars:
        if not os.getenv(var):
            logger.debug(f"Environment variable {var} is not set")
    logger.info(
        "Environment: keys from terminal, endpoint/model from config when unset."
    )


# ---------------------------------------------------------------------------
# Evaluator initialisation helpers
# ---------------------------------------------------------------------------


def _init_evaluators(config, logger):
    """Initialise evaluators from the ``evaluators`` list in the config.

    Returns (hallucination_checker, attribute_verifier, av_config_obj,
    vlm_verifier). The bare ``vlm_verifier`` comes back too so a
    ``vlm_verification``-only config still reports the endpoint it used.
    """
    hallucination_checker = None
    llm_question_generator = None
    vlm_verifier = None
    av_config_obj = None  # keep reference for retry/extra_questions later

    evaluators = config.get("evaluators") or []
    if hasattr(evaluators, "__iter__") and not isinstance(evaluators, (str, dict)):
        pass  # already a list
    else:
        evaluators = []

    # endpoints is now a registry LIST (Endpoint objects or dicts).
    endpoints = config.get("endpoints") or []

    def _resolve_consumer_endpoint(role: str, endpoint_id: str = None):
        """Resolve url/model/api_key_env for an evaluator consumer.

        Returns a 3-tuple of strings/None. None-safe: when no endpoint resolves
        (e.g. URL supplied only via env), url/model are "" and api_key_env is
        None so the adapter falls back to the role-default key env var.
        """
        ep = select_endpoint(endpoints, role, endpoint_id, required=False)
        if ep is None:
            return "", "", None

        def _get(name):
            return ep.get(name) if isinstance(ep, dict) else getattr(ep, name, None)

        return (
            _get("url") or "",
            _get("model") or "",
            _get("api_key_env"),
        )

    for entry in evaluators:
        # Support both Pydantic model and raw dict
        if hasattr(entry, "model_dump"):
            entry = entry.model_dump(exclude_none=False)

        if entry.get("hallucination_check") is not None:
            hc = entry["hallucination_check"]
            if hc.get("enabled", True):
                logger.info("Initializing hallucination checker...")
                hallucination_checker = HallucinationChecker(
                    params=hc.get("params", {}),
                    logger=logger,
                )
                logger.info("Hallucination checker initialized successfully")

        elif entry.get("attribute_verification") is not None:
            av = entry["attribute_verification"]
            av_config_obj = av
            if av.get("enabled", True):
                logger.info(
                    "Initializing attribute verification (question generator)..."
                )
                qg = av.get("question_generation", {})
                qg_endpoint_id = qg.get("endpoint_id")
                qg_url, qg_model, qg_key_env = _resolve_consumer_endpoint(
                    "llm", qg_endpoint_id
                )
                llm_question_generator = LLMQuestionGenerator.from_config(
                    config_params=qg,
                    system_prompt=qg.get("system_prompt", ""),
                    endpoint=(os.getenv("LLM_ENDPOINT_URL") or qg_url),
                    model=(os.getenv("LLM_ENDPOINT_MODEL") or qg_model),
                    api_key_env=qg_key_env,
                    logger=logger,
                )
                # Initialize VLM verifier from inline vlm_verification config
                # if present, otherwise fall back to endpoints.vlm with defaults.
                if vlm_verifier is None:
                    inline_vlm = av.get("vlm_verification")
                    vlm_endpoint_id = (inline_vlm or {}).get("endpoint_id")
                    (
                        vlm_url,
                        vlm_model_ep,
                        vlm_key_env,
                    ) = _resolve_consumer_endpoint("vlm", vlm_endpoint_id)
                    vlm_endpoint = os.getenv("VLM_ENDPOINT_URL") or vlm_url
                    vlm_model = os.getenv("VLM_ENDPOINT_MODEL") or vlm_model_ep
                    if inline_vlm is not None and vlm_endpoint:
                        logger.info(
                            "Initializing VLM verifier from attribute_verification.vlm_verification..."
                        )
                        vlm_verifier = VLMVerifier.from_config(
                            config_params=inline_vlm,
                            endpoint=vlm_endpoint,
                            model=vlm_model,
                            api_key_env=vlm_key_env,
                            logger=logger,
                        )
                    elif vlm_endpoint:
                        logger.info(
                            "Initializing VLM verifier from endpoints.vlm with defaults..."
                        )
                        vlm_verifier = VLMVerifier.from_config(
                            config_params={"system_prompt": "", "parameters": {}},
                            endpoint=vlm_endpoint,
                            model=vlm_model,
                            api_key_env=vlm_key_env,
                            logger=logger,
                        )

        elif entry.get("vlm_verification") is not None:
            vv = entry["vlm_verification"]
            vv_endpoint_id = vv.get("endpoint_id")
            vv_url, vv_model, vv_key_env = _resolve_consumer_endpoint(
                "vlm", vv_endpoint_id
            )
            logger.info("Initializing VLM verifier...")
            vlm_verifier = VLMVerifier.from_config(
                config_params=vv,
                endpoint=(os.getenv("VLM_ENDPOINT_URL") or vv_url),
                model=(os.getenv("VLM_ENDPOINT_MODEL") or vv_model),
                api_key_env=vv_key_env,
                logger=logger,
            )

    attribute_verifier = None
    if llm_question_generator is not None and vlm_verifier is not None:
        attribute_verifier = AttributeVerifier(
            question_generator=llm_question_generator,
            vlm_verifier=vlm_verifier,
            logger=logger,
        )
        logger.info("Attribute verification initialized successfully")

    return hallucination_checker, attribute_verifier, av_config_obj, vlm_verifier


def _init_cosmos_evaluator(pipeline_config, logger):
    """Initialize the single enabled external Cosmos evaluator, if configured.

    This stays separate from :func:`_init_evaluators` so the built-in
    initializer retains its established four-value interface. Disabled entries
    are ignored without constructing a client or performing network I/O.
    """
    for entry in pipeline_config.evaluators or []:
        evaluator_config = entry.cosmos_evaluator
        if evaluator_config is None or not evaluator_config.enabled:
            continue
        logger.info("Initializing external Cosmos evaluator...")
        client = CosmosEvaluatorClient.from_config(
            pipeline_config.endpoints,
            evaluator_config,
            logger=logger,
        )
        logger.info("External Cosmos evaluator initialized successfully")
        return client
    return None


def _write_prompt_file(prompt_path: str, prompt: str, logger: logging.Logger) -> bool:
    """Persist the current prompt so retries keep metadata in sync."""
    try:
        with msc.open(prompt_path, "w") as f:
            f.write(prompt)
        logger.info(f"Prompt written to {prompt_path}")
        return True
    except Exception as e:
        logger.error(f"Failed to write prompt file {prompt_path}: {e}")
        return False


def _generate_prompt(
    sample: dict,
    captioner,
    logger: logging.Logger,
    rgb_path: str | None = None,
    write_prompt: bool = True,
):
    """Generate a prompt for a sample and optionally persist it."""
    if captioner is None:
        logger.error("No captioner configured to generate a prompt")
        return None

    logger.info("Running captioning...")
    try:
        if hasattr(captioner, "get_caption_for_sample"):
            prompt = captioner.get_caption_for_sample(sample, rgb_path)
        else:
            prompt = captioner.get_caption(rgb_path)
        logger.debug(f"Caption: {prompt}")
        logger.info("Captioning completed successfully")
    except Exception as e:
        logger.error(f"Failed to generate caption for {rgb_path}: {e}")
        return None

    if prompt is None:
        logger.error("No prompt was produced from captioning")
        return None

    if write_prompt and not _write_prompt_file(
        sample["output"]["caption"], prompt, logger
    ):
        return None

    return prompt


def _first_values(values: dict) -> dict:
    """Normalize configured variable lists for compact metadata."""
    return {
        key: (value[0] if isinstance(value, list) and value else value)
        for key, value in (values or {}).items()
    }


def _captioning_selections(config: dict, sample: dict | None = None) -> dict:
    """Return configured template IDs or existing LLM selections."""
    captioning_cfg = config.get("captioning") or {}
    if captioning_cfg.get("template") is not None:
        inputs = (sample or {}).get("inputs") or {}
        selections = inputs.get("prompt_attributes") or {}
        if not isinstance(selections, dict):
            return {}
        return {
            key: value.strip() if isinstance(value, str) else value
            for key, value in selections.items()
        }

    llm_cfg = captioning_cfg.get("llm") or {}
    values = llm_cfg.get("verification_values") or llm_cfg.get("variables") or {}
    return _first_values(values)


def _candidate_evaluation_inputs(
    config: dict, sample: dict | None = None
) -> tuple[dict, dict]:
    """Resolve candidate-specific values and option pools for any evaluator."""
    captioning_cfg = config.get("captioning") or {}
    template_cfg = captioning_cfg.get("template")
    if template_cfg is not None:
        selected_variables = _captioning_selections(config, sample)
        catalogs = template_cfg.get("attributes") or {}
        variable_options = {}
        for variable_name, selected_value in selected_variables.items():
            catalog = catalogs.get(variable_name)
            variable_options[variable_name] = (
                list(catalog) if isinstance(catalog, dict) else [selected_value]
            )
        return selected_variables, variable_options

    llm_cfg = captioning_cfg.get("llm") or {}
    variables = llm_cfg.get("variables") or {}
    verification_values = llm_cfg.get("verification_values")
    selected_variables = _first_values(
        verification_values if verification_values is not None else variables
    )
    verification_options = llm_cfg.get("verification_options")
    variable_options = (
        verification_options if verification_options is not None else variables
    )
    return selected_variables, variable_options


def _attribute_verification_inputs(
    config: dict, sample: dict | None = None
) -> tuple[dict, dict]:
    """Backward-compatible wrapper for the built-in attribute verifier."""
    return _candidate_evaluation_inputs(config, sample)


def _cosmos_outcome_views(
    outcome: CosmosEvaluationOutcome,
    *,
    attempt: int,
    seed: int,
) -> tuple[dict, dict]:
    """Return persistable provider and compatibility views for one attempt."""
    provider = deepcopy(outcome.provider_metadata)
    provider.update(
        {
            "operational_success": outcome.operational_success,
            "quality_passed": outcome.quality_passed,
            "attempt": attempt,
            "seed_used": seed,
        }
    )
    if outcome.error and "error" not in provider:
        provider["error"] = outcome.error
    return provider, deepcopy(outcome.compatibility_aliases)


def _common_message_prefix(message_lists: list) -> list:
    """Longest run of leading chat turns identical across every call.

    A stage that calls repeatedly resends the same system turn each time; only
    the user turn carries the per-call question. Hoisting the shared prefix
    keeps the system prompt in the record exactly once.
    """
    if not message_lists or not all(isinstance(m, list) for m in message_lists):
        return []
    prefix = []
    for turns in zip(*message_lists):
        if any(turn != turns[0] for turn in turns[1:]):
            break
        prefix.append(turns[0])
    return prefix


def _fold_repeated_request_fields(requests: list) -> tuple:
    """Split ``requests`` into ``(shared, per_call)``.

    A multi-call stage resends its whole configuration on every call — model,
    sampling params, the response schema, the system prompt — so recording each
    call verbatim repeats kilobytes that never change. Everything identical
    across all calls moves to ``shared``; each entry keeps only what differed.
    Returns ``(None, requests)`` unchanged when there is nothing to fold.
    """
    if len(requests) < 2 or not all(isinstance(item, dict) for item in requests):
        return None, requests

    first = requests[0]
    shared = {
        key: value
        for key, value in first.items()
        if key != "messages"
        and all(key in other and other[key] == value for other in requests[1:])
    }
    prefix = _common_message_prefix([item.get("messages") for item in requests])
    if prefix:
        shared["messages"] = prefix

    per_call = []
    for item in requests:
        rest = {key: value for key, value in item.items() if key not in shared}
        if prefix:
            rest["messages"] = (item.get("messages") or [])[len(prefix) :]
        per_call.append(rest)
    return (shared or None), per_call


def _endpoint_record(adapter, stage: str | None = None) -> dict | None:
    """Describe the endpoint an adapter is bound to, or ``None`` if there is none.

    Reads the adapter rather than the config so the record shows the endpoint
    that actually served the call, including a URL an env override redirected
    (``VLM_ENDPOINT_URL`` and friends apply to captioning/evaluator consumers).

    ``requests`` carries the wire body of every successful call this stage made
    for the sample, media elided — for a chat stage those are the system and
    user prompts that produced the caption or the judgements. Whatever every
    call sent identically (model, sampling params, response schema, the system
    turn) is hoisted to ``request_shared`` so it appears once; each ``requests``
    entry keeps only what differed. ``calls`` is the true call count, which
    exceeds ``len(requests)`` when a stage ran past ``MAX_RECORDED_REQUESTS``.
    """
    if adapter is None:
        return None

    endpoint = getattr(adapter, "endpoint", None)
    try:
        contract = effective_adapter(endpoint) if endpoint is not None else None
    except ValueError:
        contract = None

    record = {"stage": stage} if stage else {}
    record.update(
        {
            "id": getattr(endpoint, "id", None),
            "role": getattr(adapter, "role", None) or getattr(endpoint, "role", None),
            # request_url is the concrete route every adapter POSTs to.
            "url": getattr(adapter, "request_url", None)
            or getattr(adapter, "url", None),
            "adapter": contract,
            "model": getattr(adapter, "model", None),
            "timeout": getattr(adapter, "timeout", None),
        }
    )
    requests = list(getattr(adapter, "recorded_requests", None) or [])
    if requests:
        shared, per_call = _fold_repeated_request_fields(requests)
        # The wire body repeats the endpoint's own model on every call; keep it
        # only where it differs. Applied to both shapes so a one-call stage and
        # a many-call stage of the same config report the same fields — and
        # keyed on presence, since a single-model NIM sends no model at all and
        # `None == None` would otherwise pop a key that was never there.
        if shared and shared.get("model") == record["model"]:
            shared.pop("model", None)
        per_call = [
            {
                k: v
                for k, v in item.items()
                if not (k == "model" and v == record["model"])
            }
            if isinstance(item, dict)
            else item
            for item in per_call
        ]
        if shared:
            record["request_shared"] = shared
        record["requests"] = per_call
        # Always report the true call count, so a capped list can't be mistaken
        # for the complete set.
        record["calls"] = getattr(adapter, "recorded_call_count", len(requests))
    return record


def _generator_adapter(generator):
    """Return the adapter a generator calls through (direct or via executor)."""
    return getattr(generator, "adapter", None) or getattr(
        getattr(generator, "executor", None), "adapter", None
    )


def _pipeline_stages(captioner, generator, attribute_verifier, vlm_verifier=None):
    """``(stage, adapter)`` for every endpoint this run is wired to, in order.

    Shared by the metadata record and the per-sample reset so the two can never
    walk different sets of adapters.
    """
    stages = []

    # Captioning: a composite captioner holds one sub-captioner per role; a
    # single-model one holds the adapter directly. Text/file captioners call
    # nothing and contribute no endpoint.
    if captioner is not None:
        for attribute, stage in (
            ("vlm_captioner", "captioning.vlm"),
            ("llm_captioner", "captioning.llm"),
        ):
            part = getattr(captioner, attribute, None)
            if part is not None:
                stages.append((stage, getattr(part, "adapter", None)))
        # Gate on whether an adapter was actually found, not on whether `stages`
        # is non-empty: a sub-captioner that exposes no adapter appends a None
        # that gets filtered out later, so testing the list alone would skip
        # this fallback and leave the run with no captioning record at all.
        if not any(adapter is not None for _stage, adapter in stages):
            adapter = getattr(captioner, "adapter", None)
            if adapter is not None:
                role = getattr(adapter, "role", None) or "model"
                stages.append((f"captioning.{role}", adapter))

    stages.append(("generation", _generator_adapter(generator)))

    if attribute_verifier is not None:
        stages.append(
            (
                "attribute_verification.question_generation",
                getattr(
                    getattr(attribute_verifier, "question_generator", None),
                    "adapter",
                    None,
                ),
            )
        )
        stages.append(
            (
                "attribute_verification.vlm_verification",
                getattr(
                    getattr(attribute_verifier, "vlm_verifier", None), "adapter", None
                ),
            )
        )
    elif vlm_verifier is not None:
        # Standalone vlm_verification evaluator (no attribute verification).
        stages.append(("vlm_verification", getattr(vlm_verifier, "adapter", None)))

    return stages


def _reset_request_records(captioner, generator, attribute_verifier, vlm_verifier=None):
    """Clear each adapter's recorded requests before a sample.

    Adapters are built once and reused for every sample, so without this a
    sample's metadata would carry the previous sample's calls too.
    """
    for _stage, adapter in _pipeline_stages(
        captioner, generator, attribute_verifier, vlm_verifier
    ):
        reset = getattr(adapter, "reset_requests", None)
        if callable(reset):
            reset()


def _pipeline_endpoints(
    captioner,
    generator,
    attribute_verifier,
    vlm_verifier=None,
    generation_seconds: float | None = None,
) -> list:
    """Every inference endpoint this run is *wired to*, in pipeline order.

    Captioning and evaluation call their own endpoints, so recording only the
    generation endpoint would leave most of the run unattributed — the caption
    that conditioned the output, and the judgements that passed it, each came
    from a model worth naming.

    This reflects wiring, not call history: a stage that was configured but
    skipped for this sample (a reused caption file, verification never reached)
    still appears, with no ``requests``.
    """
    records = []
    for stage, adapter in _pipeline_stages(
        captioner, generator, attribute_verifier, vlm_verifier
    ):
        record = _endpoint_record(adapter, stage)
        if record is None:
            continue
        if stage == "generation" and generation_seconds is not None:
            record["duration_seconds"] = generation_seconds
        records.append(record)
    return records


# Captioner class -> the strategy name the config selected it with. Keeps the
# metadata readable without making the reader map class names back to config.
_CAPTIONER_STRATEGIES = {
    "TextCaptioner": "text",
    "FileCaptioner": "file",
    "TemplateCaptioner": "template",
    "VLMLLMCaptioner": "vlm+llm",
    "VLMCaptioner": "vlm",
    "LLMCaptioner": "llm",
}


def _captioning_provenance(captioner) -> dict:
    """How this run's prompt was built.

    A text/template captioner calls no endpoint, so it contributes nothing to
    the endpoint records — yet it is what produced the prompt. Without the
    template, metadata shows the rendered result and the substituted values but
    not the string that combined them, so two runs off different templates are
    indistinguishable after the fact.
    """
    if captioner is None:
        return {}

    name = type(captioner).__name__
    record = {"strategy": _CAPTIONER_STRATEGIES.get(name, name)}
    template = getattr(captioner, "template", None)
    if isinstance(template, str) and template:
        record["template"] = template
    return record


def _prompt_builder_provenance(captioner) -> dict:
    """Expose compact metadata only for a completed VLM-template build."""
    if getattr(captioner, "last_prompt_builder", None) != "vlm_template":
        return {}

    return {
        "prompt_builder": "vlm_template",
        "scene_description": getattr(captioner, "last_scene_description", None),
        "scene_description_source": getattr(
            captioner, "last_scene_description_source", None
        ),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="validate the resolved config and exit without endpoint calls or inference",
    )
    args, unknown_args = parser.parse_known_args()

    # Initialize logger
    logger = logging.getLogger(__name__)
    log_level = os.getenv("LOG_LEVEL", "INFO")
    logging.basicConfig(
        level=log_level, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)

    print()
    logger.info("=" * 80)
    logger.info("Cosmos Data Factory (CDF) Augmentation")
    logger.info(f"Start Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 80)

    load_secrets(logger)

    # ---- Load & validate config ----
    try:
        with msc.open(args.config, "r") as f:
            base_config = yaml.safe_load(f)
            conf = OmegaConf.create(base_config)
            OmegaConf.set_struct(conf, False)
            if unknown_args:
                # Redact secret-like overrides (e.g. *.api_key=...) so a key
                # passed on the CLI never lands in the logs.
                logger.info(f"Applying CLI overrides: {redact_overrides(unknown_args)}")
                conf.merge_with_dotlist(unknown_args)
            config_dict = OmegaConf.to_container(conf, resolve=True)
    except Exception as e:
        logger.error(
            f"Failed to read config file {args.config} or apply overrides: {e}"
        )
        # Exit non-zero: nothing ran, so a bare `return` would report success to
        # CI/batch callers for a config that was never even loaded.
        sys.exit(1)

    validate_environment(logger)

    pipeline_config = validate_config_structure(config_dict, logger)
    if pipeline_config is None:
        logger.error(
            "Configuration validation failed. Please fix the configuration file."
        )
        sys.exit(1)

    if args.validate_only:
        logger.info("Configuration validation passed; no endpoint calls were made")
        return

    # External evaluation is preflighted before any captioning or generation so
    # a service/discovery problem cannot waste a generated candidate.
    try:
        cosmos_evaluator = _init_cosmos_evaluator(pipeline_config, logger)
        if cosmos_evaluator is not None:
            logger.info("Preflighting external Cosmos evaluator...")
            cosmos_evaluator.preflight()
            logger.info("External Cosmos evaluator preflight passed")
    except CosmosEvaluatorError as exc:
        logger.error("External Cosmos evaluator preflight failed: %s", exc)
        sys.exit(1)
    except Exception as exc:
        logger.error(
            "External Cosmos evaluator preflight failed (%s)",
            type(exc).__name__,
        )
        sys.exit(1)

    # For convenience keep a plain dict view for sections that still need it
    config = config_dict
    config_dir = os.path.dirname(args.config)

    # ---- Initialize captioner ----
    captioner = None
    if config.get("captioning"):
        logger.info("Initializing captioner...")
        try:
            captioner = create_captioner(config, logger, config_dir=config_dir)
            logger.info("Captioner initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize captioner: {e}")
            sys.exit(1)

    # ---- Initialize evaluators ----
    (
        hallucination_checker,
        attribute_verifier,
        av_config_obj,
        standalone_vlm_verifier,
    ) = _init_evaluators(config, logger)

    # ---- Initialize generator ----
    generator = None
    if config.get("augmentation"):
        try:
            generator = create_generator(config, logger)
            logger.info("Generator initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize generator: {e}")
            sys.exit(1)

    overwrite_caption = os.getenv("OVERWRITE_CAPTION", "true")

    # ---- NVCF progress tracker ----
    tracker = NVCFProgressTracker(logger=logger)
    data_samples = config["data"]
    total_samples = len(data_samples)
    samples_processed = 0
    models_fetched = False

    # ---- Process samples ----
    for sample in data_samples:
        logger.info("=" * 80)
        # Adapters are reused across samples; start each one's provenance clean.
        _reset_request_records(
            captioner, generator, attribute_verifier, standalone_vlm_verifier
        )
        sample_inputs = sample.get("inputs") or {}
        rgb_path = sample_inputs.get("rgb") if isinstance(sample_inputs, dict) else None
        logger.info(f"Processing sample: {rgb_path or 'N/A (text2world)'}")
        try:
            logger.debug(f"Processing sample: {sample}")

            if not validate_sample_data_availability(sample, config, logger):
                logger.error("Sample validation failed, skipping sample")
                continue

            output_media_path = sample["output"]["video"]

            # ---- Preprocessing: input resize (optional) ----
            # Resize feeds GENERATION only (gen_input_path). The ORIGINAL input
            # (rgb_path) stays the reference for alignment, captioning, and the
            # hallucination check — so alignment registers the output back to the
            # TRUE input frame (small reference -> fast, output in the original
            # frame). Useful for routes that edit at the input resolution (e.g.
            # openai.images.edits). CPU-only (cv2), so it never imports cupy.
            gen_input_path = rgb_path
            _dp_model = pipeline_config.data_processing
            _prep_model = _dp_model.preprocessing if _dp_model else None
            _resize_cfg = _prep_model.resize if _prep_model else None
            if _resize_cfg is not None and _resize_cfg.enabled and rgb_path:
                try:
                    from data_processing import resize_input

                    resized_input_path = os.path.join(
                        os.path.dirname(output_media_path), "_resized_input.png"
                    )
                    logger.info("Running preprocessing: input resize...")
                    resize_input(
                        rgb_path, resized_input_path, _resize_cfg.model_dump(), logger
                    )
                    gen_input_path = resized_input_path
                except Exception as e:
                    logger.error(f"Input resize failed for {rgb_path}: {e}")
                    continue

            prompt = None
            control_inputs = {}
            natural_caption_from_vlm = None
            gen_elapsed = None
            cosmos_evaluator_results = None
            cosmos_compatibility_aliases = {}
            cosmos_operational_success = True
            cosmos_quality_passed = True

            # ---- Captioning ----
            caption_exists = msc.is_file(sample["output"]["caption"])
            caption_content = None
            if caption_exists:
                try:
                    with msc.open(sample["output"]["caption"], "r") as f:
                        caption_content = f.read()
                    logger.debug(f"Caption file exists: {sample['output']['caption']}")
                except Exception as e:
                    logger.error(
                        f"Failed to read caption file {sample['output']['caption']}: {e}"
                    )
                    caption_exists = False

            if not caption_exists or overwrite_caption == "true":
                prompt = _generate_prompt(sample, captioner, logger, rgb_path=rgb_path)
                if prompt is None:
                    continue
            else:
                prompt = caption_content
                reset_provenance = getattr(captioner, "reset_provenance", None)
                if callable(reset_provenance):
                    reset_provenance()
                logger.debug("Using existing prompt from file")

            # ---- Generation + Verification retry loop ----
            if generator is not None:
                hallucination_enabled = hallucination_checker is not None
                verification_enabled = attribute_verifier is not None

                max_retries = pipeline_config.pipeline.retry
                request_timeout = pipeline_config.pipeline.request_timeout
                if max_retries > 0:
                    logger.info(
                        f"Retry enabled (pipeline.retry): up to {max_retries} "
                        f"retries on evaluator failure (seed re-roll) or "
                        f"generation/endpoint failure (exponential backoff from "
                        f"{request_timeout:.0f}s)."
                    )

                # Seed handling
                aug_params = config.get("augmentation", {}).get("parameters", {})
                regenerate_caption_on_retry = (
                    pipeline_config.pipeline.regenerate_caption_on_retry
                )
                original_seed = aug_params.get("seed")
                if original_seed in (None, "None", "none"):
                    if hasattr(generator, "seed") and generator.seed is not None:
                        original_seed = generator.seed
                    elif (
                        hasattr(generator, "executor")
                        and hasattr(generator.executor, "seed")
                        and generator.executor.seed is not None
                    ):
                        original_seed = generator.executor.seed
                    else:
                        original_seed = int(time.time())

                current_seed = original_seed
                hallucination_results = None
                verification_results = None
                passed_hallucination_check = True
                generation_succeeded = False
                max_attempts = max_retries + 1

                for attempt in range(max_attempts):
                    if attempt > 0:
                        current_seed = original_seed + attempt
                        logger.info("=" * 80)
                        logger.info(
                            f"RETRY ATTEMPT {attempt}/{max_retries}: seed {current_seed}"
                        )
                        logger.info("=" * 80)
                        if models_fetched and not os.environ.get("HF_HUB_OFFLINE"):
                            os.environ["HF_HUB_OFFLINE"] = "1"
                            logger.info("Set HF_HUB_OFFLINE=1 for retry (cache warm)")
                        if hasattr(generator, "seed"):
                            generator.seed = current_seed
                        elif hasattr(generator, "executor") and hasattr(
                            generator.executor, "seed"
                        ):
                            generator.executor.seed = current_seed

                        if regenerate_caption_on_retry and captioner is not None:
                            logger.info("Regenerating prompt for retry...")
                            prompt = _generate_prompt(
                                sample, captioner, logger, rgb_path=rgb_path
                            )
                            if prompt is None:
                                logger.error("Retry captioning failed; aborting sample")
                                break

                    print()
                    logger.info(f"Running generation (seed: {current_seed})...")

                    if prompt is None:
                        logger.error("Prompt is required for generation")
                        break

                    # Load control inputs from augmentation.modalities
                    try:
                        control_inputs = {}
                        modalities_cfg = (
                            config.get("augmentation", {}).get("modalities") or {}
                        )
                        if (
                            sample_inputs
                            and "controls" in sample_inputs
                            and sample_inputs["controls"]
                        ):
                            for mod_key in ("edge", "depth", "seg", "vis"):
                                weight = modalities_cfg.get(mod_key)
                                if (
                                    weight is not None
                                    and mod_key in sample_inputs["controls"]
                                ):
                                    control_inputs[mod_key] = sample_inputs["controls"][
                                        mod_key
                                    ]
                            if not control_inputs:
                                control_inputs = {
                                    k: v
                                    for k, v in sample_inputs["controls"].items()
                                    if v is not None
                                    and str(v) not in ("None", "none", "null")
                                }
                            logger.debug(
                                f"Control inputs: {list(control_inputs.keys())}"
                            )
                    except (TypeError, KeyError, ValueError, AttributeError) as e:
                        logger.error(
                            f"Failed to prepare control inputs -> {output_media_path}: {e}"
                        )
                        break

                    # Generate, re-calling the endpoint on failure up to the
                    # pipeline.retry budget. Each call is bounded by request_timeout,
                    # which is also the exponential-backoff base between retries.
                    success, output_path, attempt_elapsed = generate_with_retries(
                        generator,
                        prompt,
                        gen_input_path,
                        control_inputs,
                        output_media_path,
                        max_retries,
                        request_timeout,
                        logger,
                    )
                    if not success:
                        logger.error(
                            f"Generation failed after {max_retries + 1} attempt(s) "
                            f"-> {output_media_path}"
                        )
                        # Leave gen_elapsed on the attempt that produced the
                        # retained output: generate_with_retries reports its
                        # final attempt's elapsed whether or not it succeeded,
                        # and an evaluator retry that fails here does not
                        # replace the output an earlier attempt wrote.
                        break
                    gen_elapsed = attempt_elapsed
                    models_fetched = True
                    generation_succeeded = True
                    sample["output"]["video"] = output_path
                    # Cosmos results are scoped to the retained candidate. A
                    # successful retry replaces that candidate, so its external
                    # evaluation state must start clean.
                    cosmos_evaluator_results = None
                    cosmos_compatibility_aliases = {}
                    cosmos_operational_success = True
                    cosmos_quality_passed = True
                    logger.info(
                        f"Generation done in {gen_elapsed / 60:.2f} min, "
                        f"saved to: {output_path}"
                    )

                    # ---- Hallucination check ----
                    passed_hallucination_check = True
                    if hallucination_enabled and rgb_path is not None:
                        print()
                        logger.info("Running hallucination check...")
                        try:
                            # Get threshold from evaluator config
                            hc_threshold = 0.682
                            for entry in config.get("evaluators") or []:
                                if hasattr(entry, "model_dump"):
                                    entry = entry.model_dump(exclude_none=False)
                                hc = entry.get("hallucination_check")
                                if hc:
                                    hc_threshold = hc.get("threshold", 0.682)

                            check_passed, details = (
                                hallucination_checker.check_hallucination(
                                    original_video_path=rgb_path,
                                    augmented_video_path=sample["output"]["video"],
                                )
                            )
                            score = details.get("score", 0.0)
                            passed_hallucination_check = (
                                check_passed and score >= hc_threshold
                            )
                            hallucination_results = {
                                "passed": passed_hallucination_check,
                                "score": score,
                                "threshold": hc_threshold,
                                "details": details,
                                "attempt": attempt + 1,
                                "seed_used": current_seed,
                            }
                            if not passed_hallucination_check:
                                logger.warning(
                                    f"Hallucination check failed (score={score:.4f})"
                                )
                                if attempt < max_retries:
                                    continue
                                else:
                                    logger.warning(
                                        "Exhausted retries for hallucination check"
                                    )
                            else:
                                logger.info(
                                    f"Hallucination check passed (score={score:.4f})"
                                )
                        except Exception as e:
                            logger.error(f"Hallucination check error: {e}")
                            hallucination_results = {
                                "passed": False,
                                "error": str(e),
                                "attempt": attempt + 1,
                                "seed_used": current_seed,
                            }
                            passed_hallucination_check = False

                    # ---- Attribute verification ----
                    if passed_hallucination_check and verification_enabled:
                        print()
                        logger.info("Running attribute verification...")
                        try:
                            selected_variables, variable_options = (
                                _attribute_verification_inputs(config, sample)
                            )
                            exclude_variables = set(
                                (av_config_obj or {}).get("exclude_variables") or []
                            )
                            if exclude_variables:
                                selected_variables = {
                                    k: v
                                    for k, v in selected_variables.items()
                                    if k not in exclude_variables
                                }
                                variable_options = {
                                    k: v
                                    for k, v in variable_options.items()
                                    if k not in exclude_variables
                                }
                            extra_questions = (av_config_obj or {}).get(
                                "extra_questions"
                            ) or []

                            passed_attr, ver_details = (
                                attribute_verifier.verify_video_attributes(
                                    video_path=sample["output"]["video"],
                                    selected_variables=selected_variables,
                                    variable_options=variable_options,
                                    extra_questions=extra_questions,
                                )
                            )
                            verification_results = {
                                "passed": passed_attr,
                                "details": ver_details,
                                "attempt": attempt + 1,
                                "seed_used": current_seed,
                            }
                            if exclude_variables:
                                verification_results["excluded_variables"] = sorted(
                                    exclude_variables
                                )

                            if passed_attr:
                                # Natural caption generation on pass
                                gen_cap = (av_config_obj or {}).get(
                                    "generate_natural_caption_on_pass", False
                                )
                                cap_attrs = selected_variables
                                if gen_cap and attribute_verifier and cap_attrs:
                                    attrs_text = "; ".join(
                                        f"{k.replace('_', ' ')}: {v}"
                                        for k, v in cap_attrs.items()
                                    )
                                    nat_cfg = (av_config_obj or {}).get(
                                        "natural_caption"
                                    ) or {}
                                    user_tpl = nat_cfg.get("user_prompt_template") or (
                                        "This multi-view image shows a person with the following clothing: {attributes_text}. "
                                        "Write one natural sentence describing the person's appearance."
                                    )
                                    user_prompt = user_tpl.format(
                                        attributes_text=attrs_text
                                    )
                                    natural_caption_from_vlm = attribute_verifier.vlm_verifier.generate_description(
                                        sample["output"]["video"],
                                        user_prompt,
                                        system_prompt_override=nat_cfg.get(
                                            "system_prompt"
                                        ),
                                    )
                                    if natural_caption_from_vlm:
                                        logger.info(
                                            "Generated natural caption from VLM."
                                        )
                                logger.info("All attribute checks passed.")
                            else:
                                logger.warning("Some attribute checks failed.")
                                if attempt < max_retries:
                                    logger.info(
                                        f"Retrying (attempt {attempt + 2}/{max_attempts})"
                                    )
                                    continue
                                else:
                                    logger.warning(
                                        "Exhausted retries for attribute verification"
                                    )
                                    break
                        except Exception as e:
                            logger.error(f"Attribute verification error: {e}")
                            verification_results = {
                                "passed": False,
                                "error": str(e),
                                "attempt": attempt + 1,
                                "seed_used": current_seed,
                            }
                            break
                    elif not passed_hallucination_check:
                        if attempt < max_retries:
                            continue
                        break

                    # ---- External Cosmos evaluation ----
                    # Reaching this point means every enabled built-in gate has
                    # passed for this candidate. The client owns Arbitrator
                    # inputs and transport; the CLI supplies only the generated
                    # URI plus an optional checker-neutral captioning overlay.
                    if cosmos_evaluator is not None:
                        print()
                        logger.info("Running external Cosmos evaluation...")
                        selected_variables = None
                        variable_options = None
                        if cosmos_evaluator.merge_captioning_selections:
                            selected_variables, variable_options = (
                                _candidate_evaluation_inputs(config, sample)
                            )
                        try:
                            cosmos_outcome = cosmos_evaluator.evaluate_candidate(
                                sample["output"]["video"],
                                selected_variables=selected_variables,
                                variable_options=variable_options,
                            )
                            (
                                cosmos_evaluator_results,
                                cosmos_compatibility_aliases,
                            ) = _cosmos_outcome_views(
                                cosmos_outcome,
                                attempt=attempt + 1,
                                seed=current_seed,
                            )
                        except Exception as exc:
                            # The client normally converts expected failures to
                            # typed outcomes. Treat an unexpected exception as a
                            # service failure without persisting its potentially
                            # sensitive message.
                            cosmos_operational_success = False
                            cosmos_quality_passed = False
                            cosmos_evaluator_results = {
                                "status": "service_failure",
                                "operational_success": False,
                                "quality_passed": None,
                                "attempt": attempt + 1,
                                "seed_used": current_seed,
                                "error": f"unexpected {type(exc).__name__}",
                            }
                            cosmos_compatibility_aliases = {}
                            logger.error(
                                "External Cosmos evaluator failed unexpectedly (%s)",
                                type(exc).__name__,
                            )
                            break

                        cosmos_operational_success = cosmos_outcome.operational_success
                        failed_cosmos_checkers = ", ".join(
                            cosmos_outcome.failed_checkers
                        )
                        failed_checker_suffix = (
                            f"; failed checker(s): {failed_cosmos_checkers}"
                            if failed_cosmos_checkers
                            else ""
                        )
                        if not cosmos_operational_success:
                            cosmos_quality_passed = False
                            logger.error(
                                "External Cosmos evaluator service/contract failure; "
                                "stopping candidate retries%s",
                                failed_checker_suffix,
                            )
                            break

                        cosmos_quality_passed = cosmos_outcome.quality_passed is True
                        if not cosmos_quality_passed:
                            logger.warning(
                                "External Cosmos quality gate rejected the candidate%s",
                                failed_checker_suffix,
                            )
                            if attempt < max_retries:
                                logger.info(
                                    f"Retrying (attempt {attempt + 2}/{max_attempts})"
                                )
                                continue
                            logger.warning(
                                "Exhausted retries for external Cosmos evaluation"
                            )
                            break
                        logger.info("External Cosmos evaluation passed")

                    # All enabled built-in and external gates passed, or no
                    # external evaluator is configured.
                    break

                # Restore seed
                if original_seed is not None:
                    if hasattr(generator, "seed"):
                        generator.seed = original_seed
                    elif hasattr(generator, "executor") and hasattr(
                        generator.executor, "seed"
                    ):
                        generator.executor.seed = original_seed
            else:
                logger.warning("No generator configured, skipping generation")
                hallucination_results = None
                verification_results = None
                generation_succeeded = False

            # A generator ran but produced no output after exhausting retries —
            # log it and skip output/metadata so it isn't counted as completed.
            if generator is not None and not generation_succeeded:
                logger.error(f"Sample failed: {rgb_path or 'unknown'}")
                continue

            # ---- Did the configured evaluators ultimately pass? ----
            # Generation can succeed while an evaluator (hallucination or
            # attribute verification) fails after exhausting its retries — e.g.
            # the LLM question generator can't be reached. Under
            # pipeline.evaluation.strict (default True) that is a FAILED sample:
            # it must NOT be counted as processed, so the run exits non-zero and
            # is never reported "complete". retain_failures (default True) keeps
            # the output files for inspection; when False they are discarded.
            eval_settings = pipeline_config.pipeline.evaluation
            evaluation_passed = True
            if generator is not None:
                if hallucination_enabled:
                    evaluation_passed = passed_hallucination_check
                if verification_enabled and evaluation_passed:
                    evaluation_passed = bool(
                        verification_results and verification_results.get("passed")
                    )
                if cosmos_evaluator is not None:
                    evaluation_passed = (
                        evaluation_passed
                        and cosmos_operational_success
                        and cosmos_quality_passed
                    )
            sample_failed = eval_settings.strict and not evaluation_passed
            if sample_failed and not eval_settings.retain_failures:
                logger.error(
                    "Evaluation failed (retain_failures=False) — discarding "
                    f"output for: {rgb_path or 'unknown'}"
                )
                for _discard in (
                    sample["output"].get("video"),
                    sample["output"].get("caption"),
                    sample["output"].get("metadata"),
                    sample["output"].get("evaluation"),
                ):
                    try:
                        if _discard and msc.is_file(_discard):
                            msc.delete(_discard)
                    except Exception as _del_err:
                        logger.debug(f"Could not delete {_discard}: {_del_err}")
                continue

            # ---- Data processing (alignment, ...) ----
            alignment_params = None
            dp_model = pipeline_config.data_processing
            align_model = dp_model.alignment if dp_model else None
            if align_model is not None:
                if not generation_succeeded:
                    logger.warning(
                        "Skipping alignment: generation did not succeed this run"
                    )
                elif not rgb_path:
                    logger.warning("Skipping alignment: no reference (rgb) input")
                else:
                    logger.info("Running data processing: alignment...")
                    # Imported lazily: alignment is GPU-only (cupy), which the
                    # endpoint-only slim image does not ship. Endpoint runs that
                    # don't configure alignment never import it.
                    from data_processing import run_alignment

                    alignment_params = run_alignment(
                        ref_path=rgb_path,
                        align_path=sample["output"]["video"],
                        dest_path=sample["output"]["video"],
                        config=align_model.model_dump(),
                        logger=logger,
                    )

            # ---- Data processing: transcode ----
            # The generated bitstream is whatever the model endpoint returned,
            # so without this the output codec varies by model. Image outputs
            # are skipped inside transcode_video.
            transcode_results = None
            transcode_model = dp_model.transcode if dp_model else None
            if transcode_model is not None:
                if not generation_succeeded:
                    # Only reachable with no generator configured (a generator
                    # that failed already `continue`d above), so nothing was
                    # written to normalize.
                    logger.warning(
                        "Skipping transcode: no generator configured, nothing generated"
                    )
                elif sample_failed:
                    # Retained only for inspection (retain_failures), not shipped
                    # as a dataset artifact -- don't spend a re-encode on it.
                    logger.warning(
                        "Skipping transcode: sample failed evaluation "
                        "(output retained for inspection)"
                    )
                else:
                    from data_processing import transcode_video

                    transcode_results = transcode_video(
                        src_path=sample["output"]["video"],
                        dest_path=sample["output"]["video"],
                        config=transcode_model.model_dump(),
                        logger=logger,
                    )
                    if transcode_results.get("error"):
                        # Not fatal (the generated video is still valid), but the
                        # single-codec guarantee does not hold for this sample.
                        logger.warning(
                            "Output was NOT normalized to "
                            f"{transcode_results.get('target_codec')}: "
                            f"{transcode_results['error']}"
                        )

            # ---- Write metadata ----
            try:
                metadata_selections = _captioning_selections(config, sample)

                metadata = {
                    "prompt": prompt,
                    "selections": metadata_selections,
                    "caption_path": sample["output"]["caption"],
                    "output_media_path": sample["output"]["video"],
                    "input_media_path": rgb_path,
                    "control_media": control_inputs,
                }
                metadata.update(_prompt_builder_provenance(captioner))
                captioning_record = _captioning_provenance(captioner)
                if captioning_record:
                    metadata["captioning"] = captioning_record
                # Per sample, not hoisted: each entry carries that sample's own
                # requests (prompts, seed) and the generation timing.
                endpoint_records = _pipeline_endpoints(
                    captioner,
                    generator,
                    attribute_verifier,
                    standalone_vlm_verifier,
                    gen_elapsed,
                )
                if endpoint_records:
                    metadata["endpoints"] = endpoint_records
                if hallucination_results is not None:
                    metadata["hallucination_check"] = hallucination_results
                if verification_results is not None:
                    metadata["attribute_verification"] = verification_results
                if cosmos_evaluator_results is not None:
                    metadata["cosmos_evaluator"] = cosmos_evaluator_results
                    metadata.update(cosmos_compatibility_aliases)
                if natural_caption_from_vlm:
                    metadata["natural_caption"] = natural_caption_from_vlm
                if alignment_params:
                    metadata["alignment"] = alignment_params
                if transcode_results is not None:
                    metadata["transcode"] = transcode_results

                with msc.open(sample["output"]["metadata"], "w") as f:
                    json.dump(metadata, f, indent=2)
                logger.info(f"Metadata written to {sample['output']['metadata']}")

                # Write evaluation file if path specified
                eval_path = sample["output"].get("evaluation")
                if eval_path and (
                    hallucination_results is not None
                    or verification_results is not None
                    or cosmos_evaluator_results is not None
                ):
                    try:
                        eval_output = {}
                        if hallucination_results is not None:
                            eval_output["hallucination_check"] = hallucination_results
                        if verification_results is not None:
                            eval_output["attribute_verification"] = verification_results
                        if cosmos_evaluator_results is not None:
                            eval_output["cosmos_evaluator"] = cosmos_evaluator_results
                            eval_output.update(cosmos_compatibility_aliases)
                        with msc.open(eval_path, "w") as f:
                            json.dump(eval_output, f, indent=2)
                        logger.info(f"Evaluation results written to {eval_path}")
                    except Exception as e:
                        logger.error(f"Failed to write evaluation results: {e}")

            except Exception as e:
                logger.error(f"Failed to write metadata: {e}")
                continue

            logger.info("=" * 80)
            # A strict evaluator failure (output retained above) is NOT a
            # completed sample: skip the success count so the final tally exits
            # non-zero and the task is not marked complete.
            if sample_failed:
                logger.error(
                    "Sample evaluation FAILED (output retained for inspection): "
                    f"{rgb_path or 'unknown'}"
                )
                continue
            samples_processed += 1
            tracker.update(
                percent=(samples_processed / total_samples) * 100,
                metadata={
                    "lastProcessedSample": (sample.get("inputs") or {}).get(
                        "rgb", "unknown"
                    )
                },
            )

        except Exception as e:
            logger.error(f"Unexpected error processing sample: {e}")
            continue

    # samples_processed counts only fully successful samples, so any shortfall
    # means a sample failed (e.g. generation exhausted its retries); exit
    # non-zero without marking the task complete.
    if samples_processed < total_samples:
        logger.error(
            f"{total_samples - samples_processed}/{total_samples} sample(s) "
            f"failed; pipeline aborted without completion."
        )
        sys.exit(1)

    tracker.complete(metadata={"totalSamplesProcessed": samples_processed})


if __name__ == "__main__":
    main()
