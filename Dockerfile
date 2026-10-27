# One image for the API and the worker (the command picks the role).
#   CPU (default):  docker build -t ras .
#   GPU (CUDA 12.1): docker build -t ras-gpu --build-arg TORCH_INDEX=https://download.pytorch.org/whl/cu121 .
# CUDA wheels bundle the CUDA runtime; the host only needs an NVIDIA driver (ECS GPU-optimised AMI has it).

ARG PYTHON=3.12

FROM python:${PYTHON}-slim AS build
ARG TORCH_INDEX=https://download.pytorch.org/whl/cpu
ARG TORCH_VERSION=2.2.2
ARG TORCHAUDIO_VERSION=2.2.2
RUN apt-get update && apt-get install -y --no-install-recommends build-essential && rm -rf /var/lib/apt/lists/*
RUN python -m venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
WORKDIR /src
# the large, rarely changing layer first so code edits rebuild in seconds
RUN pip install "torch==${TORCH_VERSION}" "torchaudio==${TORCHAUDIO_VERSION}" --index-url "${TORCH_INDEX}"
COPY pyproject.toml README.md ./
RUN mkdir ras && touch ras/__init__.py && pip install ".[transkun,service,aws,models]" && pip uninstall -y ras \
    && rm -rf ras build *.egg-info
COPY ras ./ras
RUN pip install --no-deps . && python -c "import ras, ras.pipeline; print('ras', ras.__version__)"
# optional open-source models baked into the image (checksum-verified): "hft dynamics" by default (27 MB),
# add "kong" (164 MB) when a rubric uses pedal_source kong/agree; "" for none
ARG DOWNLOAD_MODELS="hft dynamics"
RUN mkdir -p /opt/models && if [ -n "${DOWNLOAD_MODELS}" ]; then \
      RAS_MODELS_DIR=/opt/models ras models download ${DOWNLOAD_MODELS}; fi

FROM python:${PYTHON}-slim AS runtime
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg libsndfile1 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --uid 10001 --create-home --shell /usr/sbin/nologin ras \
    && mkdir -p /data && chown ras /data
COPY --from=build /opt/venv /opt/venv
COPY --from=build /opt/models /opt/models
ENV PATH=/opt/venv/bin:$PATH PYTHONUNBUFFERED=1 RAS_DATA_DIR=/data MPLCONFIGDIR=/tmp/mpl RAS_MODELS_DIR=/opt/models
WORKDIR /app
USER ras
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s \
    CMD python -c "import urllib.request,sys; urllib.request.urlopen('http://127.0.0.1:8000/v1/health', timeout=4)" || exit 1
CMD ["ras", "serve", "--host", "0.0.0.0", "--port", "8000", "--no-with-worker"]

# test image: the runtime plus pytest and the test suite (used by docker compose run tests)
FROM runtime AS test
USER root
RUN pip install pytest
COPY tests ./tests
COPY data/rubrics ./data/rubrics
COPY pyproject.toml ./
USER ras
CMD ["pytest", "-m", "cloud", "-p", "no:cacheprovider"]
