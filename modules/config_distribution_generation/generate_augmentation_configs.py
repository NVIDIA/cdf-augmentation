#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Augmentation Config Generator

This script generates multiple augmentation configuration files from a workflow
config that specifies:
- Input data directory with media files (images and/or videos)
- Number of augmentations per media file
- Variables to augment with their probability distributions
- Example augmentation config to use as template

Usage:
    python generate_augmentation_configs.py --workflow workflow_input.yaml
    python generate_augmentation_configs.py --workflow workflow_input.yaml --dry-run
"""

import argparse
import ast
import json
import logging
import math
import random
import sys
import os
import posixpath
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from urllib.parse import urlparse, urlunparse

import multistorageclient as msc
import yaml

# Add parent directory to path to import from modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from modules.aug_utils.common import is_remote_path
from modules.aug_utils.nvcf import load_secrets
from modules.aug_utils.constants import MEDIA_EXTENSIONS, is_image, is_video
from modules.aug_utils.stream_keys import resolve_stream_key


_MSC_REVERSE_PATH_MAPPINGS: Optional[List[tuple[str, str]]] = None


def setup_logging(level: str = "INFO") -> logging.Logger:
    """Setup logging configuration."""
    logging.basicConfig(
        level=level, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    return logging.getLogger(__name__)


def _join_path(base: str, *parts: str) -> str:
    """Join local or remote path segments."""
    if is_remote_path(base):
        parsed = urlparse(base)
        joined_path = parsed.path
        for part in parts:
            joined_path = posixpath.join(joined_path, str(part))
        return urlunparse(parsed._replace(path=joined_path))
    return os.path.join(base, *map(str, parts))


def _parent_path(path: str) -> str:
    """Return parent directory for local or remote path."""
    if is_remote_path(path):
        parsed = urlparse(path)
        parent = posixpath.dirname(parsed.path.rstrip("/"))
        return urlunparse(parsed._replace(path=parent))
    return str(Path(path).expanduser().resolve().parent)


def _normalize_path(path: str, base_dir: Optional[str] = None) -> str:
    """Normalize a path while preserving remote URIs."""
    if path is None:
        return path

    path = str(path)
    if is_remote_path(path):
        return path

    if os.path.isabs(path):
        return str(Path(path).expanduser().resolve())

    if base_dir:
        if is_remote_path(base_dir):
            return _join_path(base_dir, path)
        return str((Path(base_dir).expanduser().resolve() / path).resolve())

    return str(Path(path).expanduser().resolve())


def _path_name(path: str) -> str:
    """Return filename portion for local or remote path."""
    if is_remote_path(path):
        return posixpath.basename(urlparse(path).path.rstrip("/"))
    return Path(path).name


def _path_stem(path: str) -> str:
    """Return stem portion for local or remote path."""
    name = _path_name(path)
    if "." not in name:
        return name
    return name.rsplit(".", 1)[0]


def _path_suffix(path: str) -> str:
    """Return suffix portion for local or remote path."""
    name = _path_name(path)
    if "." not in name:
        return ""
    return f".{name.rsplit('.', 1)[1]}"


def _extract_source_id(path: str) -> str:
    """
    Extract source ID from media filename stem.

    Expected naming pattern is "<id>_<dataset...>", e.g. "094247_PA100K.jpg".
    Falls back to full stem when "_" is not present.
    """
    stem = _path_stem(path)
    return stem.split("_", 1)[0] if "_" in stem else stem


def _path_parent(path: str) -> str:
    """Return containing directory for local or remote path."""
    if is_remote_path(path):
        parsed = urlparse(path)
        return urlunparse(parsed._replace(path=posixpath.dirname(parsed.path)))
    return str(Path(path).parent)


def _relative_for_log(path: str, root: str) -> str:
    """Best-effort relative path for logging."""
    if is_remote_path(path) or is_remote_path(root):
        parsed_path = urlparse(path)
        parsed_root = urlparse(root)
        if (
            parsed_path.scheme == parsed_root.scheme
            and parsed_path.netloc == parsed_root.netloc
            and parsed_path.path.startswith(parsed_root.path.rstrip("/") + "/")
        ):
            return parsed_path.path[len(parsed_root.path.rstrip("/") + "/") :]
        return path

    try:
        return str(Path(path).relative_to(Path(root)))
    except ValueError:
        return path


def _entry_to_path(entry: Any, root: str) -> Optional[str]:
    """Convert an MSC list entry into a usable path string."""
    if isinstance(entry, str):
        if is_remote_path(entry) or os.path.isabs(entry):
            return entry
        return _join_path(root, entry)

    candidate = None
    if isinstance(entry, dict):
        for key in ("path", "uri", "key", "name"):
            if isinstance(entry.get(key), str):
                candidate = entry[key]
                break
    else:
        for attr in ("path", "uri", "key", "name"):
            value = getattr(entry, attr, None)
            if isinstance(value, str):
                candidate = value
                break

    if candidate is None:
        return None

    if is_remote_path(candidate) or os.path.isabs(candidate):
        return candidate
    return _join_path(root, candidate)


def _load_msc_reverse_path_mappings(
    logger: Optional[logging.Logger] = None,
) -> List[tuple[str, str]]:
    """Load reverse MSC path mappings as (msc_prefix, external_prefix)."""
    global _MSC_REVERSE_PATH_MAPPINGS
    if _MSC_REVERSE_PATH_MAPPINGS is not None:
        return _MSC_REVERSE_PATH_MAPPINGS

    config: Optional[Dict[str, Any]] = None
    config_path = os.environ.get("MSC_CONFIG")
    if config_path and os.path.exists(config_path):
        try:
            with open(config_path, "r") as f:
                config = json.load(f)
        except Exception as e:
            if logger:
                logger.debug(f"Failed to read MSC_CONFIG file {config_path}: {e}")

    if config is None and "MULTISTORAGECLIENT_CONFIGURATION" in os.environ:
        try:
            config = json.loads(os.environ["MULTISTORAGECLIENT_CONFIGURATION"])
        except Exception as e:
            if logger:
                logger.debug(f"Failed to parse MULTISTORAGECLIENT_CONFIGURATION: {e}")

    reverse_mappings: List[tuple[str, str]] = []
    path_mapping = (config or {}).get("path_mapping", {})
    for external_prefix, msc_prefix in path_mapping.items():
        if isinstance(external_prefix, str) and isinstance(msc_prefix, str):
            reverse_mappings.append((msc_prefix, external_prefix))

    def _mapping_priority(item: tuple[str, str]) -> tuple[int, int]:
        _, external_prefix = item
        if external_prefix.startswith("s3://"):
            priority = 0
        elif external_prefix.startswith(("gs://", "az://")):
            priority = 1
        elif external_prefix.startswith(("http://", "https://")):
            priority = 2
        else:
            priority = 3
        return (priority, -len(external_prefix))

    reverse_mappings.sort(key=_mapping_priority)
    _MSC_REVERSE_PATH_MAPPINGS = reverse_mappings
    return reverse_mappings


def _to_external_storage_path(
    path: str, logger: Optional[logging.Logger] = None
) -> str:
    """Convert msc:// paths to a preferred external path, favoring s3:// mappings."""
    if not isinstance(path, str) or not path.startswith("msc://"):
        return path

    for msc_prefix, external_prefix in _load_msc_reverse_path_mappings(logger):
        if path == msc_prefix.rstrip("/"):
            suffix = ""
        elif path.startswith(msc_prefix):
            suffix = path[len(msc_prefix) :]
        else:
            continue

        if suffix:
            return _join_path(external_prefix, suffix)
        return external_prefix

    return path


def _list_media_via_msc(data_dir: str, logger: logging.Logger) -> List[str]:
    """List media files using whichever MSC listing API is available."""
    media_files: List[str] = []

    if hasattr(msc, "list_recursive"):
        for entry in msc.list_recursive(data_dir):
            path = _entry_to_path(entry, data_dir)
            if path and any(
                str(path).lower().endswith(ext) for ext in MEDIA_EXTENSIONS
            ):
                media_files.append(path)
        return media_files

    if hasattr(msc, "glob"):
        logger.debug(
            "multistorageclient has no top-level list_recursive; falling back to glob"
        )
        for ext in MEDIA_EXTENSIONS:
            pattern = _join_path(data_dir, "**", f"*{ext}")
            for entry in msc.glob(pattern):
                path = _entry_to_path(entry, data_dir)
                if path:
                    media_files.append(path)
        return media_files

    raise AttributeError(
        "multistorageclient has neither 'list_recursive' nor 'glob' available"
    )


def load_workflow_config(config_path: str, logger: logging.Logger) -> Dict[str, Any]:
    """Load and validate workflow configuration."""
    logger.info(f"Loading workflow config from {config_path}")

    if not msc.is_file(config_path):
        raise FileNotFoundError(f"Workflow config not found: {config_path}")

    with msc.open(config_path, "r") as f:
        try:
            config = yaml.safe_load(f)
        except yaml.YAMLError as e:
            logger.error(f"Failed to parse workflow config: {e}")
            raise

    workflow_type = config.get("workflow_type")

    # Validate required fields
    if workflow_type == "seed_image_gen_t2i":
        required_fields = [
            "env_type",
            "n_augmentations",
            "config_output",
            "example_augmentation_config",
            "output_root",
        ]
    else:
        required_fields = [
            "data_dir",
            "n_augmentations",
            "config_output",
            "example_augmentation_config",
            "variables",
        ]
    for field in required_fields:
        if field not in config:
            raise ValueError(f"Missing required field in workflow config: {field}")

    workflow_dir = _parent_path(config_path)
    if "data_dir" in config:
        config["data_dir"] = _normalize_path(config["data_dir"], base_dir=workflow_dir)
    config["config_output"] = _normalize_path(
        config["config_output"], base_dir=workflow_dir
    )
    config["example_augmentation_config"] = _normalize_path(
        config["example_augmentation_config"], base_dir=workflow_dir
    )
    if config.get("output_root") is not None:
        config["output_root"] = _normalize_path(
            config["output_root"], base_dir=workflow_dir
        )

    logger.info("Workflow config loaded successfully")
    if "data_dir" in config:
        logger.info(f"  Data directory: {config['data_dir']}")
    if workflow_type == "seed_image_gen_t2i":
        logger.info(f"  Environment type: {config['env_type']}")
    logger.info(f"  Augmentations per video: {config['n_augmentations']}")
    logger.info(f"  Output directory: {config['config_output']}")
    if "variables" in config:
        logger.info(f"  Variables: {list(config['variables'].keys())}")
    if config.get("conditional_variables"):
        logger.info(
            f"  Conditional variables: {list(config['conditional_variables'].keys())}"
        )

    return config


def load_example_config(config_path: str, logger: logging.Logger) -> Dict[str, Any]:
    """Load example augmentation configuration."""
    logger.info(f"Loading example augmentation config from {config_path}")

    if not msc.is_file(config_path):
        raise FileNotFoundError(f"Example config not found: {config_path}")

    with msc.open(config_path, "r") as f:
        config = yaml.safe_load(f)

    logger.info("Example config loaded successfully")
    return config


def discover_media(data_dir: str, logger: logging.Logger) -> List[str]:
    """Discover all media files (images and videos) in the data directory."""
    logger.info(f"Discovering media files in {data_dir}")

    media_files: List[str] = []

    try:
        media_files = _list_media_via_msc(data_dir, logger)
    except Exception as e:
        if is_remote_path(data_dir):
            logger.error(f"Failed to recursively list remote data directory: {e}")
            raise
        logger.warning(
            "MSC recursive listing failed for local path %s, falling back to pathlib: %s",
            data_dir,
            e,
        )
        local_root = Path(data_dir)
        if not local_root.exists():
            raise FileNotFoundError(f"Data directory not found: {data_dir}")
        for ext in MEDIA_EXTENSIONS:
            media_files.extend(str(path) for path in local_root.rglob(f"*{ext}"))

    media_files = sorted(set(media_files))

    logger.info(f"Found {len(media_files)} media files")

    # Count by type
    images = [f for f in media_files if is_image(str(f))]
    videos = [f for f in media_files if is_video(str(f))]
    logger.info(f"  - {len(images)} image(s)")
    logger.info(f"  - {len(videos)} video(s)")

    for media_file in media_files:
        logger.debug(f"  - {_relative_for_log(media_file, data_dir)}")

    if len(media_files) == 0:
        logger.warning("No media files found in data directory")

    return media_files


def parse_distribution(dist_config: Any, logger: logging.Logger) -> Dict[str, float]:
    """
    Parse distribution configuration and normalize probabilities.

    Expects a mapping of category -> probability, either as a native dict or as a
    string containing only a Python/YAML-style dict literal, e.g.:
        {"clear": 0.5, "cloudy": 0.25, "rainy": 0.25}

    String values are parsed with ``ast.literal_eval`` (literals only). Calls,
    imports, and attribute access are rejected so workflow YAML cannot execute
    arbitrary code through this path.
    """
    if isinstance(dist_config, dict):
        distribution: Any = dist_config
    elif isinstance(dist_config, str):
        try:
            distribution = ast.literal_eval(dist_config)
        # literal_eval raises ValueError/SyntaxError on non-literals, and can raise
        # TypeError/MemoryError/RecursionError on pathological input.
        except (ValueError, SyntaxError, TypeError, MemoryError, RecursionError) as e:
            logger.error(f"Failed to parse distribution string: {dist_config}")
            raise ValueError(f"Invalid distribution format: {e}") from e
    else:
        raise ValueError(f"Invalid distribution type: {type(dist_config)}")

    if not isinstance(distribution, dict):
        raise ValueError(
            "Invalid distribution format: expected a mapping of "
            f"str -> number, got {type(distribution).__name__}"
        )
    if not distribution:
        raise ValueError("Invalid distribution format: distribution must be non-empty")

    for key, value in distribution.items():
        if not isinstance(key, str):
            raise ValueError(
                "Invalid distribution format: keys must be strings, "
                f"got {type(key).__name__}"
            )
        # bool is a subclass of int; reject it explicitly for probabilities.
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(
                "Invalid distribution format: probabilities must be numbers, "
                f"got {type(value).__name__} for key {key!r}"
            )
        # NaN/inf survive the comparisons below and normalize to NaN, so reject them
        # here rather than failing later inside random.choices().
        if not math.isfinite(value):
            raise ValueError(
                "Invalid distribution format: probabilities must be finite, "
                f"got {value} for key {key!r}"
            )
        if value < 0:
            raise ValueError(
                "Invalid distribution format: probabilities must be non-negative, "
                f"got {value} for key {key!r}"
            )

    # Normalize probabilities to sum to 1.0
    total = sum(distribution.values())
    if not math.isfinite(total) or total <= 0:
        raise ValueError(
            "Invalid distribution format: probabilities must sum to a positive, "
            f"finite value, got {total}"
        )
    if abs(total - 1.0) > 0.01:
        logger.warning(
            f"Distribution probabilities sum to {total:.3f}, normalizing to 1.0"
        )
        distribution = {k: v / total for k, v in distribution.items()}

    return distribution


def _is_lookup_distribution(value: Any) -> bool:
    """
    Return True if value is a lookup (nested) distribution: key -> {key -> probability}.
    Such variables are not sampled directly; they are used to resolve other variables:
    if variable X is sampled to value V and some lookup L has key V, we sample from L[V].
    """
    if not isinstance(value, dict) or len(value) == 0:
        return False
    first = next(iter(value.values()))
    if not isinstance(first, dict):
        return False
    return all(isinstance(v, (int, float)) for v in first.values())


def _sample_from_lookup(
    key: str, lookup: Dict[str, Dict[str, float]], logger: logging.Logger
) -> Optional[str]:
    """
    If key is in lookup, sample from lookup[key] distribution and return the result.
    Otherwise return None (caller keeps original key).
    """
    if key not in lookup:
        return None
    sub = lookup[key]
    if not isinstance(sub, dict) or not sub:
        return None
    try:
        parsed = parse_distribution(sub, logger)
        values = list(parsed.keys())
        probs = list(parsed.values())
        return random.choices(values, weights=probs, k=1)[0]
    except Exception:
        return None


def _resolve_lookup_value(
    value: str,
    lookups: Dict[str, Dict[str, Dict[str, float]]],
    logger: logging.Logger,
) -> str:
    """Resolve a sampled direct value through the first matching lookup."""
    for lookup_name, lookup in lookups.items():
        resolved = _sample_from_lookup(value, lookup, logger)
        if resolved is not None:
            logger.debug(f"  Resolved {value!r} via {lookup_name!r} -> {resolved!r}")
            return resolved
    return value


def _possible_lookup_outputs(
    value: str,
    lookups: Dict[str, Dict[str, Dict[str, float]]],
    logger: logging.Logger,
) -> List[str]:
    """Return all possible resolved outputs for a direct value."""
    for lookup in lookups.values():
        if value in lookup:
            return list(parse_distribution(lookup[value], logger).keys())
    return [value]


def _validate_and_order_conditional_variables(
    direct_config: Dict[str, Any],
    lookups: Dict[str, Dict[str, Dict[str, float]]],
    conditional_config: Optional[Dict[str, Any]],
    logger: logging.Logger,
) -> List[str]:
    """Validate conditional variable definitions and return sampling order."""
    if conditional_config is None:
        return []
    if not isinstance(conditional_config, dict):
        raise ValueError(
            "conditional_variables must be a mapping of variable names to definitions"
        )

    conditional_names = set(conditional_config.keys())
    duplicate_names = conditional_names.intersection(
        set(direct_config.keys()).union(lookups.keys())
    )
    if duplicate_names:
        dupes = ", ".join(sorted(duplicate_names))
        raise ValueError(
            f"conditional_variables cannot reuse names already defined in variables: {dupes}"
        )

    ordered_conditionals: List[str] = []
    available_variables = set(direct_config.keys())
    remaining = dict(conditional_config)
    possible_values: Dict[str, set[str]] = {}

    for var_name, distribution in direct_config.items():
        outcomes: set[str] = set()
        for candidate in parse_distribution(distribution, logger).keys():
            outcomes.update(_possible_lookup_outputs(candidate, lookups, logger))
        possible_values[var_name] = outcomes

    while remaining:
        progressed = False
        for var_name, config in list(remaining.items()):
            if not isinstance(config, dict):
                raise ValueError(
                    f"conditional_variables.{var_name} must be a mapping with "
                    "'depends_on' and 'distributions'"
                )

            unknown_keys = set(config.keys()) - {"depends_on", "distributions"}
            if unknown_keys:
                extras = ", ".join(sorted(unknown_keys))
                raise ValueError(
                    f"conditional_variables.{var_name} has unsupported keys: {extras}"
                )

            depends_on = config.get("depends_on")
            distributions = config.get("distributions")
            if not isinstance(depends_on, str) or not depends_on:
                raise ValueError(
                    f"conditional_variables.{var_name}.depends_on must be a non-empty string"
                )
            if not isinstance(distributions, dict) or not distributions:
                raise ValueError(
                    f"conditional_variables.{var_name}.distributions must be a non-empty mapping"
                )
            if depends_on == var_name:
                raise ValueError(
                    f"conditional variable '{var_name}' cannot depend on itself"
                )

            if depends_on not in available_variables:
                if depends_on in conditional_names:
                    continue
                raise ValueError(
                    f"conditional variable '{var_name}' depends_on '{depends_on}', "
                    "which is not defined in variables or conditional_variables"
                )

            parent_possible_values = possible_values[depends_on]
            missing_parent_values = parent_possible_values - set(distributions.keys())
            extra_parent_values = set(distributions.keys()) - parent_possible_values
            if missing_parent_values:
                missing = ", ".join(sorted(missing_parent_values))
                raise ValueError(
                    f"conditional variable '{var_name}' is missing distributions for "
                    f"{depends_on} values: {missing}"
                )
            if extra_parent_values:
                extras = ", ".join(sorted(extra_parent_values))
                raise ValueError(
                    f"conditional variable '{var_name}' defines distributions for unsupported "
                    f"{depends_on} values: {extras}"
                )

            outcomes: set[str] = set()
            for parent_value, distribution_config in distributions.items():
                parsed = parse_distribution(distribution_config, logger)
                if not parsed:
                    raise ValueError(
                        f"conditional_variables.{var_name}.distributions.{parent_value} "
                        "must contain at least one outcome"
                    )
                outcomes.update(parsed.keys())

            ordered_conditionals.append(var_name)
            available_variables.add(var_name)
            possible_values[var_name] = outcomes
            del remaining[var_name]
            progressed = True

        if not progressed:
            unresolved = ", ".join(sorted(remaining.keys()))
            raise ValueError(
                "conditional_variables contain a dependency cycle or unresolved parent "
                f"references: {unresolved}"
            )

    return ordered_conditionals


def sample_variable_values(
    variables_config: Dict[str, Any],
    conditional_config: Optional[Dict[str, Any]],
    n_samples: int,
    seed: int,
    logger: logging.Logger,
) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
    """
    Sample variable values according to specified distributions.

    Variables can be:
    - Direct (flat): {"option_a": 0.5, "option_b": 0.5}. Sampled and written to config.
    - Lookup (nested): {"key1": {"variant_x": 0.3, "variant_y": 0.7}, ...}. Not written.
      When a direct variable is sampled to value V, if any lookup has key V, we sample
      from that lookup's sub-distribution and use the result (e.g. top_outer_color
      sampled to "blue" then resolved via colors["blue"] to "navy blue").
    - Conditional: Declared separately in workflow YAML under `conditional_variables`.
      These are sampled after direct variables and can depend on already sampled values
      (e.g. road_condition depends on weather).

    Args:
        variables_config: Dict mapping variable names to direct or lookup distributions
        conditional_config: Optional mapping of variable names to conditional definitions
        n_samples: Number of samples to generate
        seed: Random seed for reproducibility
        logger: Logger instance

    Returns:
        Tuple of:
        - sampled values written into generated configs
        - sampled base values used for verification metadata
    """
    logger.info(f"Sampling {n_samples} variable configurations")

    random.seed(seed)

    # Partition into direct (flat) and lookup (nested) variables
    direct_config = {}
    lookups = {}
    for var_name, dist_config in variables_config.items():
        if _is_lookup_distribution(dist_config):
            lookups[var_name] = dist_config
            logger.info(
                f"  Lookup variable '{var_name}' (keys: {list(dist_config.keys())})"
            )
        else:
            direct_config[var_name] = dist_config

    ordered_conditionals = _validate_and_order_conditional_variables(
        direct_config, lookups, conditional_config, logger
    )
    if ordered_conditionals:
        logger.info(f"  Conditional variables: {ordered_conditionals}")

    distributions = {}
    for var_name, dist_config in direct_config.items():
        distributions[var_name] = parse_distribution(dist_config, logger)
        logger.debug(f"  {var_name}: {distributions[var_name]}")

    samples = []
    samples_base = []  # base values before lookup resolution (for MCQ verification)
    for i in range(n_samples):
        sample = {}
        sample_base = {}
        for var_name, distribution in distributions.items():
            values = list(distribution.keys())
            probabilities = list(distribution.values())
            base_value = random.choices(values, weights=probabilities, k=1)[0]
            value = _resolve_lookup_value(base_value, lookups, logger)
            sample[var_name] = value
            sample_base[var_name] = base_value

        for var_name in ordered_conditionals:
            conditional_def = conditional_config[var_name]
            parent_name = conditional_def["depends_on"]
            parent_value = sample[parent_name]
            distribution = conditional_def["distributions"].get(parent_value)
            if distribution is None:
                raise ValueError(
                    f"conditional variable '{var_name}' has no distribution for "
                    f"{parent_name}={parent_value!r}"
                )

            parsed_distribution = parse_distribution(distribution, logger)
            values = list(parsed_distribution.keys())
            probabilities = list(parsed_distribution.values())
            value = random.choices(values, weights=probabilities, k=1)[0]
            sample[var_name] = value
            sample_base[var_name] = value
        samples.append(sample)
        samples_base.append(sample_base)

    logger.info("Sampled distribution statistics:")
    for var_name in list(distributions.keys()) + ordered_conditionals:
        counts = {}
        for sample in samples:
            val = sample[var_name]
            counts[val] = counts.get(val, 0) + 1
        logger.info(f"  {var_name}:")
        for val, count in sorted(counts.items()):
            actual_prob = count / n_samples
            if var_name in distributions:
                expected_prob = distributions[var_name].get(val, 0)
                logger.info(
                    f"    {val}: actual={actual_prob:.3f} expected={expected_prob:.3f}"
                )
            else:
                logger.info(f"    {val}: actual={actual_prob:.3f}")

    return samples, samples_base


def generate_output_paths(
    media_path: str,
    aug_index: int,
    data_dir: str,
    base_output_name: str = "output",
    output_root: Optional[str] = None,
    output_extension: Optional[str] = None,
) -> Dict[str, str]:
    """
    Generate output paths for augmented media (image or video).

    If output_root is set, paths are under output_root/{media_stem}/aug_{index}/.
    Otherwise paths are under media_dir/{media_stem}/aug_{index}/ (same parent as input).

    Creates structure:
        {root}/{media_stem}/aug_{index}/output.<ext>
        {root}/{media_stem}/aug_{index}/output_prompt.txt
        {root}/{media_stem}/aug_{index}/output_metadata.json

    By default, the output file has the same extension as the input media.
    Set output_extension for workflows that change media type, such as
    image-to-video generation.
    """
    media_stem = _path_stem(media_path)
    media_dir = _path_parent(media_path)
    media_ext = output_extension or _path_suffix(media_path)

    # Create augmentation subdirectory under output_root or next to media
    base = output_root if output_root is not None else media_dir
    aug_subdir = _join_path(base, media_stem, f"aug_{aug_index}")

    return {
        "video": _join_path(aug_subdir, f"{base_output_name}{media_ext}"),
        "caption": _join_path(aug_subdir, f"{base_output_name}_prompt.txt"),
        "metadata": _join_path(aug_subdir, f"{base_output_name}_metadata.json"),
    }


def _is_image_to_video_config(config: Dict[str, Any], media_path: str) -> bool:
    """Return whether this config generates video from an image input."""
    if not is_image(str(media_path)):
        return False

    model_config = config.get("augmentation", {}).get("model", {})
    model_name = str(model_config.get("name", "")).lower()
    return "image2video" in model_name or "image-to-video" in model_name


def _slugify_name(value: str) -> str:
    """Create a stable path-friendly name from a user-facing label."""
    slug = "".join(c.lower() if c.isalnum() else "_" for c in str(value)).strip("_")
    while "__" in slug:
        slug = slug.replace("__", "_")
    return slug or "item"


_EXTERNAL_CHECKER_URI_SCHEMES = frozenset({"s3", "gs", "az", "http", "https"})
_INPUT_RGB_REFERENCE = "${data.0.inputs.rgb}"
_SYNCHRONIZED_COSMOS_CHECKS = frozenset(
    {"metropolis.hallucination", "metropolis.attribute_verification"}
)


def _is_external_checker_uri(value: Any) -> bool:
    """Return whether a URI is directly readable by external checker services."""
    if not isinstance(value, str):
        return False
    parsed = urlparse(value)
    return parsed.scheme.lower() in _EXTERNAL_CHECKER_URI_SCHEMES and bool(
        parsed.netloc
    )


def _relative_output_prefix(output_video: str) -> str:
    """Derive the checker artifact prefix from a generated remote output path."""
    parsed = urlparse(output_video)
    prefix = posixpath.dirname(parsed.path).strip("/")
    if not prefix:
        raise ValueError(
            "external Cosmos evaluation requires generated output.video to have "
            "a non-empty object-storage directory"
        )
    return prefix


def _synchronize_cosmos_evaluator_batch_config(
    config: Dict[str, Any],
    data_entry: Dict[str, Any],
    sampled_variables: Dict[str, str],
) -> None:
    """Synchronize generated media paths and selections for external Cosmos checks."""
    evaluators = config.get("evaluators") or []
    if not isinstance(evaluators, list):
        return

    for entry in evaluators:
        if not isinstance(entry, dict):
            continue
        cosmos = entry.get("cosmos_evaluator")
        if not isinstance(cosmos, dict) or not cosmos.get("enabled", True):
            continue

        checks = cosmos.get("checks") or []
        check_names = {
            check.get("name")
            for check in checks
            if isinstance(check, dict) and isinstance(check.get("name"), str)
        }
        relevant_checks = check_names & _SYNCHRONIZED_COSMOS_CHECKS
        if not relevant_checks:
            continue

        output = data_entry.get("output")
        if not isinstance(output, dict):
            raise ValueError(
                "external Cosmos evaluation requires data[0].output to be a mapping"
            )
        output_video = output.get("video")
        if not _is_external_checker_uri(output_video):
            raise ValueError(
                "external Cosmos evaluation requires generated output.video to use "
                "s3://, gs://, az://, http://, or https:// storage"
            )

        if "evaluation" in output:
            output["evaluation"] = _join_path(
                _path_parent(output_video), "evaluation.json"
            )
            if not _is_external_checker_uri(output["evaluation"]):
                raise ValueError(
                    "external Cosmos evaluation requires data[0].output.evaluation "
                    "to use externally readable remote storage"
                )

        output_prefix = _relative_output_prefix(output_video)
        output["prefix"] = output_prefix
        cosmos["output_storage_prefix"] = (
            f"{output_prefix.rstrip('/')}/cosmos-evaluator/"
        )

        if "metropolis.hallucination" in relevant_checks:
            inputs = data_entry.get("inputs")
            input_uri = inputs.get("rgb") if isinstance(inputs, dict) else None
            if not _is_external_checker_uri(input_uri):
                raise ValueError(
                    "metropolis.hallucination requires data[0].inputs.rgb to use "
                    "s3://, gs://, az://, http://, or https:// storage"
                )

            stream_key = cosmos.get("stream_key")
            if stream_key is None:
                stream_key = resolve_stream_key(output_video)
            elif not isinstance(stream_key, str) or not stream_key.strip():
                raise ValueError(
                    "metropolis.hallucination requires a nonempty "
                    "cosmos_evaluator.stream_key when configured"
                )
            cosmos_inputs = cosmos.get("inputs")
            original_urls = (
                cosmos_inputs.get("original_video_urls")
                if isinstance(cosmos_inputs, dict)
                else None
            )
            if not isinstance(original_urls, dict):
                raise ValueError(
                    "metropolis.hallucination requires an existing "
                    "cosmos_evaluator.inputs.original_video_urls mapping"
                )
            if stream_key in original_urls:
                original_url = original_urls[stream_key]
            elif cosmos.get("stream_key") is None and len(original_urls) == 1:
                original_url = next(iter(original_urls.values()))
                original_urls.clear()
                original_urls[stream_key] = original_url
            else:
                raise ValueError(
                    "metropolis.hallucination requires an existing "
                    "cosmos_evaluator.inputs.original_video_urls entry for "
                    f"stream_key {stream_key!r}"
                )
            if original_url != _INPUT_RGB_REFERENCE and not _is_external_checker_uri(
                original_url
            ):
                raise ValueError(
                    "metropolis.hallucination requires the template's existing "
                    "cosmos_evaluator.inputs.original_video_urls entry for "
                    f"stream_key {stream_key!r} to reference data[0].inputs.rgb "
                    "or use s3://, gs://, az://, http://, or https:// storage"
                )
            original_urls[stream_key] = input_uri

        if "metropolis.attribute_verification" in relevant_checks:
            checker_config = cosmos.get("config")
            selected_variables = (
                checker_config.get("selected_variables")
                if isinstance(checker_config, dict)
                else None
            )
            variable_options = (
                checker_config.get("variable_options")
                if isinstance(checker_config, dict)
                else None
            )
            if not isinstance(selected_variables, dict):
                raise ValueError(
                    "metropolis.attribute_verification requires a "
                    "config.selected_variables mapping"
                )
            if variable_options is not None and not isinstance(variable_options, dict):
                raise ValueError(
                    "metropolis.attribute_verification config.variable_options "
                    "must be a mapping when configured"
                )

            for name in list(selected_variables):
                if name not in sampled_variables:
                    continue
                selected_value = sampled_variables[name]
                if variable_options is not None:
                    options = variable_options.get(name)
                    if not isinstance(options, list):
                        raise ValueError(
                            "metropolis.attribute_verification requires "
                            f"config.variable_options.{name} to be a list"
                        )
                    occurrences = options.count(selected_value)
                    if occurrences != 1:
                        raise ValueError(
                            f"sampled {name} value {selected_value!r} must appear exactly "
                            "once in cosmos_evaluator.config.variable_options."
                            f"{name}; found {occurrences}"
                        )
                selected_variables[name] = selected_value


def _set_dotted_path(config: Dict[str, Any], dotted_path: str, value: Any) -> None:
    """Set a nested dictionary/list value using a simple dot path."""
    current: Any = config
    parts = str(dotted_path).split(".")
    for part in parts[:-1]:
        if isinstance(current, list):
            current = current[int(part)]
        else:
            current = current.setdefault(part, {})
    last = parts[-1]
    if isinstance(current, list):
        current[int(last)] = value
    else:
        current[last] = value


def generate_seed(media_path: str, aug_index: int, base_seed: int = 42) -> int:
    """Generate deterministic seed based on media name and augmentation index."""
    # Use hash of media stem + index for deterministic but unique seeds
    media_stem = _path_stem(media_path)
    seed_string = f"{media_stem}_{aug_index}_{base_seed}"
    return hash(seed_string) % (2**31)


def create_augmentation_config(
    example_config: Dict[str, Any],
    media_path: str,
    sampled_variables: Dict[str, str],
    aug_index: int,
    data_dir: str,
    logger: logging.Logger,
    output_root: Optional[str] = None,
    sampled_base_variables: Optional[Dict[str, str]] = None,
    source_id: Optional[str] = None,
    source_id_image_count: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Create augmentation config for a specific media file and augmentation.

    Args:
        example_config: Template configuration to clone
        media_path: Path to input media (image or video)
        sampled_variables: Sampled variable values for this augmentation (resolved, e.g. "navy blue")
        aug_index: Index of this augmentation (0 to n-1)
        data_dir: Base data directory
        logger: Logger instance
        sampled_base_variables: Optional base values before lookup (e.g. "blue"); used for MCQ verification

    Returns:
        Complete augmentation configuration dict
    """
    import copy

    # Deep copy to avoid modifying original
    config = copy.deepcopy(example_config)

    # Generate unique seed for this augmentation
    seed = generate_seed(media_path, aug_index)

    # Update data section
    if "data" not in config:
        raise ValueError("Example config must have 'data' section")
    data_config = config["data"]
    # Accept both list format (data: [- inputs: ...]) and single-dict format (data: { inputs: ... })
    if isinstance(data_config, dict):
        data_config = [data_config]
    if len(data_config) == 0:
        raise ValueError(
            "Example config must have 'data' section with at least one entry"
        )

    # Use first data entry as template
    data_entry = data_config[0]

    # Update input paths
    data_entry["inputs"]["rgb"] = _to_external_storage_path(str(media_path), logger)

    # Update output paths (under output_root if set, else next to media)
    is_image_to_video = _is_image_to_video_config(config, media_path)
    output_paths = generate_output_paths(
        media_path,
        aug_index,
        data_dir,
        output_root=output_root,
        output_extension=".mp4" if is_image_to_video else None,
    )
    output_paths = {
        key: _to_external_storage_path(value, logger)
        for key, value in output_paths.items()
    }
    data_entry["output"].update(output_paths)

    # When shoe_type is "barefoot", force shoe_color to "none" in generated YAML (variables and verification_values)
    vars_to_write = dict(sampled_variables)
    if vars_to_write.get("shoe_type") == "barefoot" and "shoe_color" in vars_to_write:
        vars_to_write["shoe_color"] = "none"
    base_to_write = None
    if sampled_base_variables is not None:
        base_to_write = dict(sampled_base_variables)
        if (
            base_to_write.get("shoe_type") == "barefoot"
            and "shoe_color" in base_to_write
        ):
            base_to_write["shoe_color"] = "none"

    _synchronize_cosmos_evaluator_batch_config(
        config,
        data_entry,
        vars_to_write,
    )

    # Update captioning variables for direct LLM prompt generation
    if "captioning" in config:
        captioning_config = config["captioning"]
        llm_config = captioning_config.get("llm")
        if isinstance(llm_config, dict):
            llm_config["variables"] = {
                var_name: [sampled_value]
                for var_name, sampled_value in vars_to_write.items()
            }
            logger.debug(
                "  Set captioning.llm.variables for runtime LLM prompt generation"
            )
            if base_to_write is not None:
                llm_config["verification_values"] = {
                    var_name: [base_value]
                    for var_name, base_value in base_to_write.items()
                }
                logger.debug("  Set captioning.llm.verification_values (base) for MCQ")

        captioning_type = captioning_config.get("type")
        if captioning_type == "llm" or "variables" in captioning_config:
            captioning_config["variables"] = {
                var_name: [sampled_value]
                for var_name, sampled_value in vars_to_write.items()
            }
            logger.debug("  Set captioning.variables for runtime LLM prompt generation")
        if base_to_write is not None and (
            captioning_type == "llm"
            or "verification_values" in captioning_config
            or "verification_options" in captioning_config
        ):
            captioning_config["verification_values"] = {
                var_name: [base_value] for var_name, base_value in base_to_write.items()
            }
            logger.debug("  Set captioning.verification_values (base) for MCQ")

    # Update template_generation variables (resolved values for prompt)
    if "template_generation" in config and "variables" in config["template_generation"]:
        for var_name, sampled_value in vars_to_write.items():
            if var_name in config["template_generation"]["variables"]:
                # Replace the variable's value list with single sampled value
                config["template_generation"]["variables"][var_name] = [sampled_value]
                logger.debug(f"  Set {var_name} = {sampled_value}")
            else:
                logger.warning(
                    f"Variable '{var_name}' not found in example config, skipping"
                )

        # Write base values for MCQ verification (main colors/options, not fine-grained variants)
        if base_to_write is not None:
            config["template_generation"]["verification_values"] = {
                var_name: [base_value]
                for var_name, base_value in base_to_write.items()
                if var_name in config["template_generation"]["variables"]
            }
            logger.debug("  Set verification_values (base) for MCQ")

    # Update seeds for reproducibility
    if "prompt_generation" in config:
        config["prompt_generation"]["seed"] = seed

    # Update seeds for cosmos (legacy)
    if "cosmos" in config and "parameters" in config["cosmos"]:
        config["cosmos"]["parameters"]["seed"] = seed
        # Update inference name to be unique
        media_stem = _path_stem(media_path)
        config["cosmos"]["parameters"]["inference_name"] = (
            f"{media_stem}_aug{aug_index}"
        )

    # Update seeds for new generation format
    if "generation" in config:
        generation_type = config["generation"].get("type")

        # Update cosmos_transfer seeds
        if (
            generation_type == "cosmos_transfer"
            and "cosmos_transfer" in config["generation"]
        ):
            if "parameters" in config["generation"]["cosmos_transfer"]:
                config["generation"]["cosmos_transfer"]["parameters"]["seed"] = seed
                media_stem = _path_stem(media_path)
                config["generation"]["cosmos_transfer"]["parameters"][
                    "inference_name"
                ] = f"{media_stem}_aug{aug_index}"

        # Update qwen_image_edit seeds
        elif (
            generation_type == "qwen_image_edit"
            and "qwen_image_edit" in config["generation"]
        ):
            if "parameters" in config["generation"]["qwen_image_edit"]:
                config["generation"]["qwen_image_edit"]["parameters"]["seed"] = seed

    # Runtime context for downstream auditing and prompt policy.
    if source_id is not None or source_id_image_count is not None:
        ctx = config.setdefault("runtime_context", {})
        if source_id is not None:
            ctx["source_id"] = source_id
        if source_id_image_count is not None:
            count = int(source_id_image_count)
            ctx["source_id_image_count"] = count
            ctx["is_multi_view_id"] = count > 1

    return config


def save_config(
    config: Dict[str, Any],
    output_dir: str,
    media_path: str,
    aug_index: int,
    logger: logging.Logger,
) -> str:
    """Save augmentation config to YAML file."""
    media_stem = _path_stem(media_path)
    config_filename = f"config_{media_stem}_aug{aug_index}.yaml"
    config_path = _join_path(output_dir, config_filename)

    # Create local output directory if it doesn't exist
    if not is_remote_path(output_dir):
        Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Save config with nice formatting
    with msc.open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False, width=120)

    logger.debug(f"Saved config to {config_path}")
    return config_path


def create_seed_image_t2i_config(
    example_config: Dict[str, Any],
    workflow_config: Dict[str, Any],
    aug_index: int,
    logger: logging.Logger,
) -> Dict[str, Any]:
    """Create a T2I seed-image runtime config from an environment workflow."""
    import copy

    config = copy.deepcopy(example_config)

    env_type = str(workflow_config["env_type"])
    env_description = str(workflow_config.get("env_description", env_type))
    env_slug = _slugify_name(env_type)
    output_root = workflow_config["output_root"]
    output_basename = str(workflow_config.get("output_basename", "seed_image"))
    seed_base = int(workflow_config.get("seed_base", 42))

    data_config = config.get("data")
    if isinstance(data_config, dict):
        data_config = [data_config]
    if not data_config:
        raise ValueError(
            "Example config must have 'data' section with at least one entry"
        )
    data_entry = data_config[0]

    output_template = data_entry.get("output", {}).get("video", "")
    output_ext = _path_suffix(output_template) or ".png"
    output_name = output_basename
    if not _path_suffix(output_name):
        output_name = f"{output_name}{output_ext}"

    output_dir = _join_path(output_root, env_slug, f"aug_{aug_index}")
    data_entry.setdefault("inputs", {})["rgb"] = None
    data_entry.setdefault("output", {}).update(
        {
            "video": _join_path(output_dir, output_name),
            "caption": _join_path(output_dir, f"{_path_stem(output_name)}_prompt.txt"),
            "metadata": _join_path(
                output_dir, f"{_path_stem(output_name)}_metadata.json"
            ),
        }
    )

    variables = {
        "env_type": [env_type],
        "env_description": [env_description],
    }
    variation_instructions = workflow_config.get("variation_instructions") or []
    if variation_instructions:
        variables["variation_instruction"] = [
            variation_instructions[aug_index % len(variation_instructions)]
        ]

    llm_config = config.setdefault("captioning", {}).setdefault("llm", {})
    llm_config["variables"] = variables

    _set_dotted_path(config, "augmentation.parameters.seed", seed_base + aug_index)
    for dotted_path, value in (workflow_config.get("runtime_overrides") or {}).items():
        _set_dotted_path(config, dotted_path, value)

    ctx = config.setdefault("runtime_context", {})
    ctx["env_type"] = env_type
    ctx["aug_index"] = aug_index

    logger.debug("  T2I seed config %d variables: %s", aug_index, variables)
    return config


def save_named_config(
    config: Dict[str, Any],
    output_dir: str,
    config_filename: str,
    logger: logging.Logger,
) -> str:
    """Save augmentation config to a specific YAML filename."""
    config_path = _join_path(output_dir, config_filename)

    if not is_remote_path(output_dir):
        Path(output_dir).mkdir(parents=True, exist_ok=True)

    with msc.open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False, width=120)

    logger.debug(f"Saved config to {config_path}")
    return config_path


def generate_configs(
    workflow_config: Dict[str, Any],
    logger: logging.Logger,
    dry_run: bool = False,
) -> List[str]:
    """
    Main function to generate all augmentation configs.

    Args:
        workflow_config: Loaded workflow configuration
        logger: Logger instance
        dry_run: If True, don't actually write files

    Returns:
        List of generated config file paths
    """
    # Parse workflow config
    if workflow_config.get("config_output") is None:
        raise ValueError(
            "workflow config has no 'config_output' (output directory). "
            "Add config_output: <path> at the top level of your YAML."
        )
    workflow_type = workflow_config.get("workflow_type")
    n_augmentations = int(workflow_config["n_augmentations"])
    config_output = workflow_config["config_output"]
    output_root = workflow_config.get("output_root")
    if output_root is not None:
        logger.info(f"Output root for augmented media: {output_root}")
    example_config_path = workflow_config["example_augmentation_config"]

    # Load example config
    example_config = load_example_config(example_config_path, logger)

    if workflow_type == "seed_image_gen_t2i":
        env_slug = _slugify_name(workflow_config["env_type"])
        total_configs = n_augmentations
        logger.info(f"Will generate {total_configs} seed-image config(s)")

        if dry_run:
            logger.info("DRY RUN MODE - No files will be written")

        generated_configs = []
        for aug_idx in range(n_augmentations):
            config = create_seed_image_t2i_config(
                example_config, workflow_config, aug_idx, logger
            )
            if not dry_run:
                config_path = save_named_config(
                    config,
                    config_output,
                    f"config_{env_slug}_aug{aug_idx}.yaml",
                    logger,
                )
                generated_configs.append(config_path)
            else:
                generated_configs.append(f"DRY_RUN_config_{env_slug}_aug{aug_idx}.yaml")
        return generated_configs

    data_dir = workflow_config["data_dir"]
    variables_config = workflow_config["variables"]
    conditional_config = workflow_config.get("conditional_variables") or {}

    # Discover media files (images and videos)
    media_files = discover_media(data_dir, logger)

    if len(media_files) == 0:
        logger.error("No media files found. Exiting.")
        return []

    # Calculate total number of configs to generate
    total_configs = len(media_files) * n_augmentations
    logger.info(
        f"Will generate {total_configs} configs ({len(media_files)} media files × {n_augmentations} augmentations)"
    )

    id_image_counts: Dict[str, int] = {}
    for media_path in media_files:
        source_id = _extract_source_id(media_path)
        id_image_counts[source_id] = id_image_counts.get(source_id, 0) + 1
    logger.info(
        "Discovered %d unique source IDs (multi-view IDs: %d)",
        len(id_image_counts),
        sum(1 for c in id_image_counts.values() if c > 1),
    )

    if dry_run:
        logger.info("DRY RUN MODE - No files will be written")

    # Sample all variable values upfront for better distribution
    all_samples, all_samples_base = sample_variable_values(
        variables_config,
        conditional_config,
        total_configs,
        seed=42,  # Use fixed seed for reproducibility
        logger=logger,
    )

    # Generate configs
    generated_configs = []
    sample_idx = 0

    for media_idx, media_path in enumerate(media_files):
        source_id = _extract_source_id(media_path)
        source_id_image_count = id_image_counts.get(source_id, 1)
        logger.info(
            f"Processing media {media_idx + 1}/{len(media_files)}: {_path_name(media_path)}"
        )
        logger.debug(
            "  Source ID %s has %d image(s); multi-view=%s",
            source_id,
            source_id_image_count,
            source_id_image_count > 1,
        )

        for aug_idx in range(n_augmentations):
            sampled_variables = all_samples[sample_idx]
            sampled_base_variables = all_samples_base[sample_idx]
            sample_idx += 1

            logger.debug(f"  Augmentation {aug_idx}: {sampled_variables}")

            # Create config
            config = create_augmentation_config(
                example_config,
                media_path,
                sampled_variables,
                aug_idx,
                data_dir,
                logger,
                output_root=output_root,
                sampled_base_variables=sampled_base_variables,
                source_id=source_id,
                source_id_image_count=source_id_image_count,
            )

            # Save config
            if not dry_run:
                config_path = save_config(
                    config, config_output, media_path, aug_idx, logger
                )
                generated_configs.append(config_path)
            else:
                # Just report what would be created
                media_stem = _path_stem(media_path)
                config_filename = f"config_{media_stem}_aug{aug_idx}.yaml"
                logger.info(f"  Would create: {config_filename}")

    logger.info(f"Successfully generated {len(generated_configs)} augmentation configs")
    logger.info(f"Output directory: {config_output}")

    return generated_configs


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Generate augmentation configuration files from workflow config",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate configs from workflow
  python generate_augmentation_configs.py --workflow workflow_input.yaml

  # Dry run to see what would be generated
  python generate_augmentation_configs.py --workflow workflow_input.yaml --dry-run

  # Debug mode with verbose logging
  python generate_augmentation_configs.py --workflow workflow_input.yaml --log-level DEBUG
        """,
    )

    parser.add_argument(
        "--workflow",
        type=str,
        required=True,
        help="Path to workflow configuration YAML file",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Dry run mode - show what would be generated without writing files",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)",
    )

    args = parser.parse_args()

    # Setup logging
    logger = setup_logging(args.log_level)

    # Load secrets / MSC config before opening any local or remote files
    load_secrets(logger)

    try:
        # Load workflow config
        workflow_config_path = _normalize_path(args.workflow)
        workflow_config = load_workflow_config(workflow_config_path, logger)

        # Generate configs
        generated_configs = generate_configs(
            workflow_config, logger, dry_run=args.dry_run
        )

        if not args.dry_run:
            logger.info("=" * 80)
            logger.info("Generation complete!")
            logger.info(f"Generated {len(generated_configs)} configuration files")
            logger.info(f"Location: {workflow_config['config_output']}")
            logger.info("=" * 80)

        return 0

    except Exception as e:
        logger.error(f"Failed to generate configs: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
