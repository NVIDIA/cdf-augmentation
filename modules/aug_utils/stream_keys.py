# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Deterministic stream-key resolution shared by schema and runtime."""

import hashlib
import re
import unicodedata
from pathlib import PurePosixPath
from urllib.parse import unquote, urlsplit, urlunsplit


_STREAM_KEY_UNSAFE = re.compile(r"[^A-Za-z0-9._-]+")


def resolve_stream_key(
    candidate_uri: str, configured_stream_key: str | None = None
) -> str:
    """Resolve the one stable stream key used throughout an evaluation request."""

    if configured_stream_key is not None:
        key = configured_stream_key.strip()
        if not key:
            raise ValueError("configured stream_key is empty")
        return key

    if not isinstance(candidate_uri, str) or not candidate_uri.strip():
        raise ValueError("generated candidate URI is empty")

    parsed = urlsplit(candidate_uri)
    candidate_without_query = urlunsplit(
        (parsed.scheme, parsed.netloc, parsed.path, "", "")
    )
    components = [component for component in parsed.path.split("/") if component]
    filename = unquote(components[-1]) if components else ""
    stem = PurePosixPath(filename).stem if filename else ""
    stem = unicodedata.normalize("NFKC", stem)
    normalized = _STREAM_KEY_UNSAFE.sub("_", stem).strip("._-")
    if normalized:
        return normalized

    digest = hashlib.sha256(candidate_without_query.encode("utf-8")).hexdigest()[:12]
    return f"stream-{digest}"
