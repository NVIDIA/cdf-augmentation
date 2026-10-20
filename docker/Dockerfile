# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Endpoint-only image: remote API inference, no local Cosmos/torch. Keeps the
# CUDA base image and the LGPL/OSRB-cleared FFmpeg build from the full image
# (both cleared through the legal process); only the local Cosmos/torch
# inference stack is dropped.
FROM nvidia/cuda:13.0.2-devel-ubuntu24.04

# Use bash with pipefail so RUN steps fail on upstream errors inside pipes.
SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# Run as non-root user for security (10000 avoids conflict with base image's UID 1000)
ARG APP_UID=10000
ARG APP_GID=10000
RUN groupadd --gid ${APP_GID} nvidia \
    && useradd --uid ${APP_UID} --gid ${APP_GID} --create-home --shell /bin/bash nvidia

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    CUDA_MODULE_LOADING=LAZY \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility,video \
    UV_LINK_MODE=copy

# -------- OS deps (FFmpeg+OpenCV build toolchain + runtime libs) --------
# Source-built LGPL FFmpeg for a clean scan posture: FFmpeg in /usr/local,
# opencv from sdist with WITH_FFMPEG=OFF, PyAV source-built against
# /usr/local FFmpeg.
RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        cmake \
        curl \
        git \
        jq \
        ninja-build \
        pkg-config \
        yasm \
        nasm \
        libgnutls28-dev \
        libgl1 \
        libglib2.0-0 \
        libgomp1 \
        libjpeg-dev \
        libpng-dev \
        libtiff-dev \
        libvpx-dev \
        libwebp-dev \
        zlib1g-dev \
    && apt-get upgrade -y --no-install-recommends \
    && rm -rf /var/lib/apt/lists/*

# -------- FFmpeg source build (LGPL, OSRB-cleared codec allow-list) --------
# Apache 2.0 posture: no GPL/nonfree. Upstream source tarballs preserved
# under /usr/share/oss-sources/source-builds/ for OSS compliance.
ARG FFMPEG_VERSION=9.0.1
ARG FFMPEG_SHA256=cf38e0e28c7e5605942c4a77755349b0145804a397af37eb1fb4c77cb237f635
ARG NV_CODEC_HEADERS_VERSION=n12.1.14.0
ARG NV_CODEC_HEADERS_COMMIT=1889e62e2d35ff7aa9baca2bceb14f053785e6f1
RUN set -euxo pipefail; \
    mkdir -p /tmp/ffmpeg-build /usr/share/oss-sources/source-builds && cd /tmp/ffmpeg-build; \
    for attempt in 1 2 3 4 5; do \
        git clone https://github.com/FFmpeg/nv-codec-headers.git && \
            git -C nv-codec-headers checkout "${NV_CODEC_HEADERS_COMMIT}" && break; \
        rm -rf nv-codec-headers; \
        [ "$attempt" = 5 ] && { echo "nv-codec-headers clone failed after 5 attempts"; exit 1; }; \
        sleep $((attempt * 3)); \
    done; \
    tar -C /tmp/ffmpeg-build --exclude='.git' \
        -czf "/usr/share/oss-sources/source-builds/nv-codec-headers-${NV_CODEC_HEADERS_VERSION}.tar.gz" nv-codec-headers; \
    make -C nv-codec-headers -j"$(nproc)" && make -C nv-codec-headers install; \
    `# --retry alone will NOT retry a mid-transfer reset (curl exit 56/35), which` \
    `# is how ffmpeg.org intermittently fails this ~11MB fetch; --retry-all-errors` \
    `# is the flag that covers it.` \
    curl -fsSL --retry 5 --retry-all-errors --retry-delay 3 \
         --connect-timeout 20 --max-time 600 \
         "https://ffmpeg.org/releases/ffmpeg-${FFMPEG_VERSION}.tar.xz" -o ffmpeg.tar.xz; \
    echo "${FFMPEG_SHA256}  ffmpeg.tar.xz" | sha256sum -c -; \
    cp ffmpeg.tar.xz "/usr/share/oss-sources/source-builds/ffmpeg-${FFMPEG_VERSION}.tar.xz"; \
    tar -xf ffmpeg.tar.xz; \
    cd "ffmpeg-${FFMPEG_VERSION}"; \
    ./configure \
      `# Build setup` \
      --prefix=/usr/local \
      --disable-debug --disable-doc --disable-static \
      --enable-shared --enable-pic --enable-gnutls \
      `# LGPL guard` \
      --disable-nonfree --disable-gpl --disable-vulkan --disable-cuda-nvcc \
      `# OSRB allow-list (2026-05-23): H.264 is HW-only; SW h264 stays off` \
      `# because MPEG-LA AVC covers decode as well as encode.` \
      `# VP9 (2026-07-28) is royalty-free -- no patent pool -- so both the native` \
      `# SW decoder and the libvpx encoder fit the LGPL/Apache-2.0 no-nonfree` \
      `# posture (libvpx is BSD-3-Clause; --enable-libvpx needs no --enable-gpl).` \
      `# VIDEO OUTPUT IS VP9-ONLY (UAL parity): the h264_nvenc and mpeg4 encoders` \
      `# were dropped so augmentation cannot emit H.264/MPEG-4 Part 2. mjpeg stays` \
      `# for jpg image output; rawvideo for uncompressed interchange. H.264 and` \
      `# MPEG-4 remain readable (decoders kept) -- input support is unchanged.` \
      --disable-encoders --disable-decoders --disable-everything \
      --enable-encoder=mjpeg \
      --enable-libvpx --enable-encoder=libvpx_vp9 --enable-encoder=rawvideo \
      --enable-decoder=h264_cuvid --enable-decoder=mpeg4 --enable-decoder=mjpeg \
      --enable-decoder=vp9 --enable-decoder=vp9_cuvid \
      --enable-nvdec \
      `# Cosmos read/write IO scaffolding` \
      --enable-protocol=file \
      --enable-demuxer=mov \
      --enable-muxer=mp4 --enable-muxer=image2 \
      `# vp9_superframe is what the mp4 muxer auto-inserts for VP9; request it` \
      `# explicitly rather than relying on implicit selection under` \
      `# --disable-everything.` \
      --enable-parser=h264 --enable-parser=vp9 \
      --enable-bsf=h264_mp4toannexb --enable-bsf=vp9_superframe \
      --enable-filter=fps --enable-filter=scale --enable-filter=format \
      `# imageio_ffmpeg encoder probe: ffmpeg -f lavfi -i nullsrc -vcodec X -f null -` \
      --enable-indev=lavfi --enable-filter=nullsrc \
      --enable-decoder=wrapped_avframe --enable-muxer=null --enable-protocol=pipe \
      `# imageio_ffmpeg write path: -f rawvideo -i - (stdin via fd: in FFmpeg 8)` \
      --enable-demuxer=rawvideo --enable-decoder=rawvideo --enable-protocol=fd \
      --extra-cflags="-I/usr/local/include" \
      --extra-ldflags="-L/usr/local/lib"; \
    make -j"$(nproc)"; \
    make install; \
    ldconfig; \
    rm -rf /tmp/ffmpeg-build; \
    ffmpeg -hide_banner -version | head -n 3; \
    ! ffmpeg -buildconf | grep -E "enable-gpl|enable-nonfree"; \
    # Assert every --enable-X flag actually produced a working component. \
    # Format: <section>:<name>  (section is the ffmpeg listing flag, minus the trailing 's'). \
    for c in encoders:mjpeg encoders:libvpx-vp9 encoders:rawvideo \
             decoders:h264_cuvid decoders:mpeg4 decoders:mjpeg decoders:vp9 \
             decoders:vp9_cuvid \
             decoders:wrapped_avframe decoders:rawvideo \
             demuxers:mov demuxers:rawvideo \
             muxers:mp4 muxers:image2 muxers:null \
             protocols:file protocols:pipe protocols:fd \
             devices:lavfi bsfs:h264_mp4toannexb bsfs:vp9_superframe \
             filters:fps filters:scale filters:format filters:nullsrc; do \
        sec="${c%:*}"; name="${c#*:}"; \
        ffmpeg -hide_banner "-${sec}" 2>/dev/null | grep -qE "\\b${name}\\b" \
            || { echo "MISSING: ${name} in ${sec}"; exit 1; }; \
    done; \
    # Video output is VP9-only: assert the H.264/MPEG-4 ENCODERS are gone, so a \
    # future flag change can't silently reintroduce a non-VP9 output path. \
    # (Their DECODERS must stay -- input support is unchanged -- so this checks \
    # the encoder listing only.) \
    for name in h264_nvenc mpeg4 h263; do \
        ! ffmpeg -hide_banner -encoders 2>/dev/null | grep -qE "\\b${name}\\b" \
            || { echo "UNEXPECTED: ${name} encoder present (output must be VP9-only)"; exit 1; }; \
    done; \
    # End-to-end smoke test: lavfi -> wrapped_avframe -> rawvideo -> null muxer via fd. \
    ffmpeg -hide_banner -f lavfi -i nullsrc=s=256x256:d=8 -vcodec rawvideo -f null - \
        || { echo "ERROR: encoder-probe smoke test failed"; exit 1; }; \
    # VP9 output round-trip: encode VP9 into MP4, then decode it back. Proves the \
    # libvpx_vp9 encoder + mp4 muxer + native vp9 decoder all work together. The \
    # read-back runs without a GPU, so it also asserts VP9 decode stays software \
    # (adding vp9_cuvid must not make VP9 decode GPU-only). \
    ffmpeg -hide_banner -y -f lavfi -i nullsrc=s=64x64:d=1 -vcodec libvpx-vp9 \
        -frames:v 2 /tmp/vp9_smoke.mp4 \
        || { echo "ERROR: VP9 encode smoke test failed"; exit 1; }; \
    # Re-encode with rawvideo rather than the null muxer's default: the \
    # wrapped_avframe ENCODER is not in the allow-list, so `-f null -` alone \
    # would fail on encoder selection before ever decoding. \
    ffmpeg -hide_banner -v error -i /tmp/vp9_smoke.mp4 -vcodec rawvideo -f null - \
        || { echo "ERROR: VP9 decode round-trip smoke test failed"; exit 1; }; \
    rm -f /tmp/vp9_smoke.mp4

RUN apt-get purge -y --auto-remove yasm nasm && rm -rf /var/lib/apt/lists/*

# Install uv system-wide. Tag + multi-arch manifest digest pinned so the copied
# binary can't drift from the archived source tarball, and ARM64 stays intact.
COPY --from=ghcr.io/astral-sh/uv:0.12.24@sha256:3af4716e991d6956a41e573eab705d0ee08500cd829ed30293eb8472f372c65a /uv /uvx /usr/local/bin/
ENV PATH="/usr/local/bin:$PATH"
# Put uv and Python in a location we can chown for the non-root user. Use a
# disposable cache outside /root from the first build-time invocation so
# bundled libraries from prebuilt wheels cannot survive in the final image as
# scan-only artifacts.
ENV UV_HOME=/opt/uv \
    UV_CACHE_DIR=/tmp/uv-cache
# uv installs managed CPython here instead of the default ~/.local/share/uv
# (under /root, mode 700) which the non-root runtime user cannot traverse —
# otherwise `uv run` fails to canonicalize the venv interpreter symlink. /opt/uv
# is chowned to the nvidia user further down, so the interpreter resolves.
ENV UV_PYTHON_INSTALL_DIR=/opt/uv/python
ARG UV_VERSION=0.12.24
RUN mkdir -p /opt/uv /usr/share/oss-sources && \
    curl -fsSL "https://github.com/astral-sh/uv/archive/refs/tags/${UV_VERSION}.tar.gz" \
        -o "/usr/share/oss-sources/uv-${UV_VERSION}.tar.gz"

WORKDIR /workspace

# Copy source. This image is endpoint-only (no local Cosmos/torch inference):
# all generation goes through the adapter/executor layer over remote endpoints.
COPY modules /workspace/modules
COPY configs /workspace/configs
# Build from the committed lockfile (reviewed for OSRB/compliance) — do NOT
# re-resolve at build time. `uv sync --locked` below fails if it drifts.
COPY pyproject.toml uv.lock ./

# Python 3.13 to match the repo's .python-version (3.13.16); pinned to a full
# patch for reproducibility.
ARG PYTHON_VERSION=3.13.16
ARG PYTHON_SHA256=f4b1bfb3c79b5bb11b8d228a12504163b4c0dab4d679828d8f5f26b6cb6ab35d
# Install the requested CPython and preserve the matching upstream source
# tarball for OSS compliance (PYTHON_VERSION is a full pinned patch version).
RUN set -euxo pipefail; \
    uv python install --no-cache "${PYTHON_VERSION}"; \
    python_bin="$(uv python find --managed-python "${PYTHON_VERSION}")"; \
    "${python_bin}" -c 'import pyexpat, sys; expected = tuple(map(int, sys.argv[1].split("."))); assert sys.version_info[:3] == expected, sys.version; assert pyexpat.version_info >= (2, 8, 0), pyexpat.EXPAT_VERSION' "${PYTHON_VERSION}"; \
    mkdir -p /usr/share/oss-sources; \
    curl -fsSL "https://www.python.org/ftp/python/${PYTHON_VERSION}/Python-${PYTHON_VERSION}.tar.xz" \
        -o "/usr/share/oss-sources/Python-${PYTHON_VERSION}.tar.xz"; \
    echo "${PYTHON_SHA256}  /usr/share/oss-sources/Python-${PYTHON_VERSION}.tar.xz" | sha256sum -c -
# The Linux wheels for PyAV and OpenCV bundle private FFmpeg builds. Skip those
# two packages here; the controlled source builds below install them without
# introducing duplicate FFmpeg copies into uv's cache or the virtualenv.
RUN uv sync --locked --python ${PYTHON_VERSION} --link-mode=copy --no-dev \
        --no-install-package av \
        --no-install-package opencv-python-headless \
        --no-cache

# -------- OpenCV: rebuild from sdist with WITH_FFMPEG=OFF (Apache 2.0 release) --------
ARG OPENCV_PYTHON_VERSION=4.12.0.88
RUN set -euxo pipefail; \
    rm -rf /workspace/.venv/lib/python3.13/site-packages/cv2 \
           /workspace/.venv/lib/python3.13/site-packages/opencv_python.libs \
           /workspace/.venv/lib/python3.13/site-packages/opencv_python-*.dist-info \
           /workspace/.venv/lib/python3.13/site-packages/opencv_python_headless.libs \
           /workspace/.venv/lib/python3.13/site-packages/opencv_python_headless-*.dist-info \
           /workspace/.venv/lib/python3.13/site-packages/opencv_contrib_python*.dist-info \
           /workspace/.venv/lib/python3.13/site-packages/opencv_contrib_python*.libs; \
    uv pip install --no-cache --link-mode=copy \
        "setuptools>=80.9.0" "wheel>=0.46.2" \
        "scikit-build>=0.18,<1" "cmake>=3.27,<4" "numpy>=2.1.0,<3" "pip>=25.3"
RUN set -euxo pipefail; \
    ENABLE_CONTRIB=0 \
    ENABLE_HEADLESS=1 \
    CMAKE_ARGS="-DWITH_FFMPEG=OFF -DWITH_GSTREAMER=OFF -DWITH_V4L=OFF -DWITH_LIBV4L=OFF -DWITH_QT=OFF -DWITH_GTK=OFF -DOPENCV_ENABLE_NONFREE=OFF -DBUILD_TESTS=OFF -DBUILD_PERF_TESTS=OFF -DBUILD_EXAMPLES=OFF" \
    /workspace/.venv/bin/python -m pip install --no-cache-dir --no-build-isolation \
        --no-binary=opencv-python \
        "opencv-python==${OPENCV_PYTHON_VERSION}"
RUN /workspace/.venv/bin/python -c "import re, sys, cv2; info = cv2.getBuildInformation(); sys.exit('OpenCV unexpectedly built WITH FFmpeg') if re.search(r'FFMPEG:\\s+YES', info) else print('cv2', cv2.__version__, 'WITH_FFMPEG=OFF OK')" && \
    ! find /workspace/.venv/lib/python3.13/site-packages -name 'cv2*.so' -exec ldd {} \; \
        | grep -E 'libav(codec|device|filter|format|util)|libswresample|libswscale'

# -------- PyAV: source build linked to /usr/local LGPL FFmpeg --------
ARG PYAV_VERSION=18.1.0
RUN set -euxo pipefail; \
    rm -rf /workspace/.venv/lib/python3.13/site-packages/av \
           /workspace/.venv/lib/python3.13/site-packages/av-*.dist-info \
           /workspace/.venv/lib/python3.13/site-packages/av.libs; \
    uv pip install --no-cache --link-mode=copy \
        "setuptools>=80.9.0" "cython>=3.1.0,<4" "pip>=25.3"; \
    PKG_CONFIG_PATH="/usr/local/lib/pkgconfig:${PKG_CONFIG_PATH:-}" \
    /workspace/.venv/bin/python -m pip install --no-cache-dir --no-build-isolation \
        --no-binary av \
        "av==${PYAV_VERSION}"; \
    /workspace/.venv/bin/python -c "import av, av._core; print('pyav', av.__version__, 'core', av._core.__file__)"

# Preserve sdists for the python packages we source-built, alongside FFmpeg/
# nv-codec-headers sources under /usr/share/oss-sources/source-builds/. The
# PyPI JSON entry carries both the sdist URL and its published SHA-256 digest;
# we verify the downloaded file against the latter.
RUN set -e; \
    mkdir -p /usr/share/oss-sources/source-builds; \
    for spec in "opencv-python:${OPENCV_PYTHON_VERSION}" "av:${PYAV_VERSION}"; do \
        name="${spec%%:*}"; ver="${spec##*:}"; \
        meta=$(curl -fsSL "https://pypi.org/pypi/${name}/${ver}/json" \
            | jq -r '.urls[] | select(.packagetype=="sdist") | [.url, .digests.sha256] | @tsv' | head -n1); \
        url="${meta%%$'\t'*}"; sha="${meta##*$'\t'}"; \
        [ -n "$url" ] && [ -n "$sha" ] || { echo "No sdist metadata for ${name}==${ver}"; exit 1; }; \
        out="/usr/share/oss-sources/source-builds/$(basename "$url")"; \
        curl -fsSL "$url" -o "$out"; \
        echo "${sha}  ${out}" | sha256sum -c -; \
    done

# Defensive scan: fail if any opencv_python*.libs/libav* slipped back in.
RUN set -euxo pipefail; \
    leftovers="$(find /workspace/.venv/lib/python3.13/site-packages \( \
        -path '*opencv_python*.libs/libavcodec*' -o \
        -path '*opencv_python*.libs/libavdevice*' -o \
        -path '*opencv_python*.libs/libavfilter*' -o \
        -path '*opencv_python*.libs/libavformat*' -o \
        -path '*opencv_python*.libs/libavutil*' -o \
        -path '*opencv_python*.libs/libswresample*' -o \
        -path '*opencv_python*.libs/libswscale*' \
        -o -path '*av.libs/libav*' \
        \) -print 2>/dev/null || true)"; \
    if [ -n "$leftovers" ]; then echo "$leftovers"; exit 1; fi

# -------- cupy: GPU MI alignment for data_processing (Defect Image Generation use case) --------
# modules/data_processing/alignment.py requires cupy (GPU-only). The CUDA 13
# devel base provides the runtime libraries, so no `[ctk]` extra is needed. The
# host driver (libcuda.so) is provided at runtime via `--gpus`. Version pinned
# for OSRB scanning and reproducibility.
RUN uv pip install --no-cache --link-mode=copy "cupy-cuda13x==14.0.1"

# Build caches are not runtime dependencies. Keep them out of the final rootfs
# and assert that the historical root-owned uv cache cannot regress unnoticed.
RUN rm -rf /root/.cache/uv /tmp/uv-cache && \
    test ! -e /root/.cache/uv && \
    test ! -e /tmp/uv-cache

# Remove build-only packages to shrink attack surface. Nsight Compute ships with
# the CUDA devel base but is a GPU profiler this endpoint-only image never runs;
# its bundled Go plugin (plugins/efa_metrics/nic_sampler, built with go1.23.4)
# pulls every Go stdlib/toolchain CVE into the container scan. Its packages are
# matched by pattern so the purge keeps working on arm64 and across base bumps.
# The apt-get upgrade re-applies Ubuntu security updates: the upgrade in the
# first apt layer is served from CI's registry cache, so it can lag the archive
# (e.g. libssl3t64/openssl). This layer rebuilds whenever source changes.
RUN set -euxo pipefail; \
    nsight_pkgs="$(dpkg-query -W -f='${db:Status-Abbrev} ${Package}\n' \
        'cuda-nsight-compute-*' 'nsight-compute-*' 2>/dev/null \
        | awk '$1 == "ii" {print $2}' || true)"; \
    apt-get update; \
    apt-get purge -y --auto-remove \
        build-essential \
        cmake \
        ninja-build \
        pkg-config \
        git \
        libgnutls28-dev \
        ${nsight_pkgs}; \
    DEBIAN_FRONTEND=noninteractive apt-get upgrade -y --no-install-recommends; \
    rm -rf /opt/nvidia/nsight-compute /var/lib/apt/lists/*

# Defensive scan: fail if any Go binary is left. Scanners report every Go
# stdlib CVE for the toolchain a binary was built with, and nothing in this
# image needs Go.
RUN go_bins="$(grep -rlas -F 'Go buildinf:' /usr /opt /workspace || true)"; \
    if [ -n "$go_bins" ]; then echo "Unexpected Go binaries:"; echo "$go_bins"; exit 1; fi

# Ensure non-root user can use the app and venv interpreter (UV_HOME)
RUN chown -R nvidia:nvidia /workspace /opt/uv

USER nvidia

# UV_CACHE_DIR above is writable for arbitrary runtime UIDs. Disable auto-sync
# since the venv is baked and the source mount may be read-only.
ENV UV_NO_SYNC=1

# Third-party / container license attribution (OSS compliance), same as the
# full image. NGC-DL-CONTAINER-LICENSE ships in the CUDA base at /.
COPY --chown=nvidia:nvidia LICENSE-3rd-party.txt VERSION /workspace/
RUN cp /NGC-DL-CONTAINER-LICENSE /workspace/NGC-DL-CONTAINER-LICENSE

# Default entrypoint runs Python; override CMD to choose script
ENTRYPOINT ["uv", "run", "--no-sync", "/workspace/modules/cli.py"]
