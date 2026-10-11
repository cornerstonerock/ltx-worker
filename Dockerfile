# RunPod Serverless base (PyTorch 2.8 / CUDA 12.8 / Python 3.11).
# NOTE: torch >=2.8 is required — ComfyUI's pinned comfy-kitchen==0.2.37 uses
# builtin generics (list[int]) in torch.library custom ops, which torch 2.4's
# infer_schema rejects at import time and kills the worker on startup.
FROM runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1

# System deps: git (ComfyUI), ffmpeg (ComfyUI video nodes), curl.
RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# ComfyUI (API mode; the worker drives it over localhost).
RUN git clone https://github.com/comfyanonymous/ComfyUI.git /comfyui
WORKDIR /comfyui
RUN pip install --no-cache-dir -r requirements.txt

# Worker Python deps (torch etc. already in the base image).
COPY requirements.txt /tmp/worker-requirements.txt
RUN pip install --no-cache-dir -r /tmp/worker-requirements.txt

# Worker code + entrypoint.
COPY rp_handler.py entrypoint.sh /app/
RUN chmod +x /app/entrypoint.sh \
    && mkdir -p /comfyui/models/diffusion_models \
                 /comfyui/models/loras \
                 /comfyui/models/text_encoders \
                 /comfyui/models/vae \
                 /comfyui/models/latent_upscale_models \
                 /comfyui/input /comfyui/output

# Models live on the RunPod network volume (see entrypoint.sh), mounted at
# /runpod-volume. The volume's ltx-2.5/ tree is symlinked into ComfyUI.
VOLUME /runpod-volume

EXPOSE 8188

ENTRYPOINT ["/app/entrypoint.sh"]
