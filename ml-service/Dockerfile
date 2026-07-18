# ml-service GPU image.
#
# CUDA base MUST match the CUDA line targeted by the pinned torch (cu126) and
# paddlepaddle-gpu (cu126) wheels in requirements.txt. We use CUDA 12.6 runtime
# with cuDNN.
FROM nvidia/cuda:12.6.3-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

# System deps: python3.10 (ubuntu22.04 default), libs PaddleOCR/OpenCV/Pillow need.
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip python3-dev \
        libgl1 libglib2.0-0 libgomp1 \
        ca-certificates curl \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3 /usr/local/bin/python

WORKDIR /app

# Install torch (cu126) and paddlepaddle-gpu (cu126) from their custom indexes
# FIRST so the rest of requirements resolves against the already-installed GPU
# wheels, then install the remaining pinned deps from PyPI.
COPY requirements.txt ./
RUN python -m pip install --upgrade pip \
    && python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu126 \
    && python -m pip install paddlepaddle-gpu==3.1.0 -i https://www.paddlepaddle.org.cn/packages/stable/cu126/ \
    && python -m pip install -r requirements.txt

# Application code.
COPY app ./app
COPY scripts ./scripts

EXPOSE 8000

# GPU verification (run manually / via compose):
#   docker compose run --rm ml-service python -c "import torch; assert torch.cuda.is_available()"

HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
    CMD curl -fsS http://localhost:8000/healthz || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
