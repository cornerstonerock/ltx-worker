#!/bin/bash
# Container entrypoint: link network-volume models, boot ComfyUI, start worker.
set -e

# If a RunPod network volume is attached at /runpod-volume with a models/
# tree, symlink it into ComfyUI's model dirs (volume wins over anything local).
# LTX-2.5 files live under /runpod-volume/models/ltx-2.5/ with ComfyUI-style
# subdirs (diffusion_models/, text_encoders/, vae/, loras/).
if [ -d "/runpod-volume/models/ltx-2.5" ]; then
  for d in diffusion_models loras text_encoders vae latent_upscale_models; do
    if [ -d "/runpod-volume/models/ltx-2.5/$d" ]; then
      rm -rf "/comfyui/models/$d"
      ln -sfn "/runpod-volume/models/ltx-2.5/$d" "/comfyui/models/$d"
      echo "[entrypoint] linked /runpod-volume/models/ltx-2.5/$d"
    fi
  done
fi

# Sanity check: the worker needs its model files before accepting jobs.
missing=0
for f in \
  "diffusion_models/ltx-2.5-22b-distilled-transformer-comfy-int8-convrot.safetensors" \
  "text_encoders/gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors" \
  "vae/ltx-2.5-video-vae-conv-bf16.safetensors" \
  "vae/ltx-2.5-audio-vae-bf16.safetensors" ; do
  if [ ! -f "/comfyui/models/$f" ]; then
    echo "[entrypoint] MISSING MODEL: /comfyui/models/$f" >&2
    missing=1
  fi
done
if [ "$missing" = "1" ]; then
  echo "[entrypoint] One or more models are missing. Check the network volume." >&2
  exit 1
fi

# Boot ComfyUI in API mode (no UI needed on serverless).
# --lowvram: the int8 transformer + 15GB Gemma encoder need careful VRAM
# management on 24GB cards. Never use --use-sage-attention with comfy_quant
# int8 models (crashes).
cd /comfyui
python main.py --listen 127.0.0.1 --port 8188 --disable-auto-launch --lowvram \
  > /tmp/comfyui.log 2>&1 &
COMFY_PID=$!

# Wait for the API to answer (up to ~5 min on slow cold starts).
echo "[entrypoint] waiting for ComfyUI..."
for i in $(seq 1 60); do
  if curl -sf http://127.0.0.1:8188/system_stats > /dev/null 2>&1; then
    echo "[entrypoint] ComfyUI is up"
    break
  fi
  if ! kill -0 $COMFY_PID 2>/dev/null; then
    echo "[entrypoint] ComfyUI died during startup. Log tail:" >&2
    tail -50 /tmp/comfyui.log >&2
    exit 1
  fi
  sleep 5
done

# Hand off to the RunPod serverless loop (foreground).
exec python /app/rp_handler.py
