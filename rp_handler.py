"""
LTX-2.5 RunPod Serverless worker — text-to-video + image-to-video with
synchronized audio.

Implements the PulseClone worker contract:
  input:  { prompt, negative_prompt?, image_url?, num_frames?,
            width?, height?, fps?, seed? }
  output: { video: "data:video/mp4;base64,...", metadata: {...} }

Architecture: ComfyUI (API mode) running the official Lightricks LTX-2.5
single-stage distilled workflow — joint audio/video diffusion:
  UNETLoader (int8 distilled transformer) + CLIPLoader (type:"ltxv", Gemma 4)
  + VAELoader x2 (video conv VAE + audio VAE)
  -> CLIPTextEncode x2 -> LTXVConditioning (frame_rate stamped)
  -> EmptyLTXVLatentVideo + LTXVEmptyLatentAudio -> LTXVConcatAVLatent
     (i2v: LTXVPreprocess + LTXVImgToVideoInplace injects the start frame)
  -> RandomNoise + CFGGuider (cfg=1) + KSamplerSelect (euler_ancestral)
     + ManualSigmas (8 distilled sigmas) + SamplerCustomAdvanced
  -> LTXVSeparateAVLatent -> VAEDecodeTiled + LTXVAudioVAEDecode
  -> CreateVideo (muxes frames + waveform) -> SaveVideo

Video and audio are generated JOINTLY in one diffusion pass from a combined
AV latent, so the audio is natively synchronized (no separate TTS/mux step).
"""

import base64
import io
import json
import os
import random
import time
import uuid

import requests
from PIL import Image

import runpod

COMFY_URL = os.environ.get("COMFYUI_URL", "http://127.0.0.1:8188")
INPUT_DIR = os.environ.get("COMFYUI_INPUT_DIR", "/comfyui/input")
OUTPUT_DIR = os.environ.get("COMFYUI_OUTPUT_DIR", "/comfyui/output")

# ----------------------------------------------------------------------------
# Models — Lightricks/LTX-2.5 (verified file names on the network volume)
# ----------------------------------------------------------------------------
UNET_FILE = "ltx-2.5-22b-distilled-transformer-comfy-int8-convrot.safetensors"
CLIP_FILE = "gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors"
# Conv video VAE = lower memory & faster than the diffusion VAE (24GB-safe).
VAE_VIDEO_FILE = "ltx-2.5-video-vae-conv-bf16.safetensors"
VAE_AUDIO_FILE = "ltx-2.5-audio-vae-bf16.safetensors"

# Distilled sigmas from the official LTX-2.5 single-stage workflow (8 steps).
DISTILLED_SIGMAS = "1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0"
CFG = 1.0
SAMPLER = "euler_ancestral"

DEFAULT_NEGATIVE = (
    "low quality, blurry, watermark, distorted, deformed, bad anatomy, "
    "extra fingers, bad hands, bad face, disfigured"
)

# Poll the local ComfyUI queue for this long before giving up.
GENERATION_TIMEOUT_S = int(os.environ.get("GENERATION_TIMEOUT_S", "1800"))


class WorkerError(Exception):
    pass


# ----------------------------------------------------------------------------
# Input validation / normalization
# ----------------------------------------------------------------------------
def snap_frames(n):
    # LTX-2.5 requires 8n+1 frames.
    n = max(9, min(121, int(n)))
    return 8 * round((n - 1) / 8) + 1


def snap_dim(n):
    # EmptyLTXVLatentVideo steps dims by 32.
    n = max(256, min(1024, int(n)))
    return (n // 32) * 32


def validate(event_input):
    if not isinstance(event_input, dict):
        raise WorkerError("input must be a JSON object")
    prompt = event_input.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        raise WorkerError("prompt is required and must be a non-empty string")
    if len(prompt) > 2000:
        raise WorkerError("prompt must be <= 2000 characters")

    image_url = event_input.get("image_url")
    if image_url is not None and not isinstance(image_url, str):
        raise WorkerError("image_url must be a string")

    return {
        "prompt": prompt.strip(),
        "negative_prompt": str(event_input.get("negative_prompt") or DEFAULT_NEGATIVE)[:1000],
        "image_url": image_url.strip() if isinstance(image_url, str) and image_url.strip() else None,
        "num_frames": snap_frames(event_input.get("num_frames", 97)),
        "width": snap_dim(event_input.get("width", 768)),
        "height": snap_dim(event_input.get("height", 512)),
        "fps": max(8, min(30, int(event_input.get("fps", 24)))),
        "seed": event_input.get("seed"),
    }


# ----------------------------------------------------------------------------
# ComfyUI workflow construction (API format)
# ----------------------------------------------------------------------------
def build_workflow(p):
    seed = p["seed"]
    if not isinstance(seed, int):
        seed = random.randint(0, 2**31 - 1)

    wf = {
        # --- Loaders ---
        "unet": {
            "class_type": "UNETLoader",
            "inputs": {"unet_name": UNET_FILE, "weight_dtype": "default"},
        },
        # NEVER use LTXVGemmaCLIPModelLoader with the int8-convrot encoder
        # (transformers.from_pretrained chokes on comfy_quant format).
        "clip": {
            "class_type": "CLIPLoader",
            "inputs": {"clip_name": CLIP_FILE, "type": "ltxv", "device": "default"},
        },
        "vae_video": {
            "class_type": "VAELoader",
            "inputs": {"vae_name": VAE_VIDEO_FILE},
        },
        "vae_audio": {
            "class_type": "VAELoader",
            "inputs": {"vae_name": VAE_AUDIO_FILE},
        },
        # --- Conditioning ---
        "pos": {
            "class_type": "CLIPTextEncode",
            "inputs": {"clip": ["clip", 0], "text": p["prompt"]},
        },
        "neg": {
            "class_type": "CLIPTextEncode",
            "inputs": {"clip": ["clip", 0], "text": p["negative_prompt"]},
        },
        "cond": {
            "class_type": "LTXVConditioning",
            "inputs": {
                "positive": ["pos", 0],
                "negative": ["neg", 0],
                "frame_rate": float(p["fps"]),
            },
        },
        # --- Latents ---
        "latent_video": {
            "class_type": "EmptyLTXVLatentVideo",
            "inputs": {
                "width": p["width"],
                "height": p["height"],
                "length": p["num_frames"],
                "batch_size": 1,
            },
        },
        "latent_audio": {
            "class_type": "LTXVEmptyLatentAudio",
            "inputs": {
                "frames_number": p["num_frames"],
                "frame_rate": float(p["fps"]),
                "batch_size": 1,
                "audio_vae": ["vae_audio", 0],
            },
        },
    }

    if p["image_url"]:
        # Image-to-video: preprocess the start frame and inject it into the
        # video latent in place (strength ~0.7 keeps identity, allows motion).
        wf["loadimg"] = {
            "class_type": "LoadImage",
            "inputs": {"image": p["_image_filename"]},
        }
        wf["preprocess"] = {
            "class_type": "LTXVPreprocess",
            "inputs": {"image": ["loadimg", 0], "img_compression": 18},
        }
        wf["i2v"] = {
            "class_type": "LTXVImgToVideoInplace",
            "inputs": {
                "vae": ["vae_video", 0],
                "image": ["preprocess", 0],
                "latent": ["latent_video", 0],
                "strength": 0.7,
                "bypass": False,
            },
        }
        video_latent_ref = ["i2v", 0]
        mode = "i2v"
    else:
        video_latent_ref = ["latent_video", 0]
        mode = "t2v"

    wf["latent"] = {
        "class_type": "LTXVConcatAVLatent",
        "inputs": {
            "video_latent": video_latent_ref,
            "audio_latent": ["latent_audio", 0],
        },
    }

    # --- Joint AV sampler (8-step distilled) ---
    wf["noise"] = {
        "class_type": "RandomNoise",
        "inputs": {"noise_seed": seed},
    }
    wf["guider"] = {
        "class_type": "CFGGuider",
        "inputs": {
            "model": ["unet", 0],
            "positive": ["cond", 0],
            "negative": ["cond", 1],
            "cfg": CFG,
        },
    }
    wf["sampler"] = {
        "class_type": "KSamplerSelect",
        "inputs": {"sampler_name": SAMPLER},
    }
    wf["sigmas"] = {
        "class_type": "ManualSigmas",
        "inputs": {"sigmas": DISTILLED_SIGMAS},
    }
    wf["sample"] = {
        "class_type": "SamplerCustomAdvanced",
        "inputs": {
            "noise": ["noise", 0],
            "guider": ["guider", 0],
            "sampler": ["sampler", 0],
            "sigmas": ["sigmas", 0],
            "latent_image": ["latent", 0],
        },
    }
    wf["split"] = {
        "class_type": "LTXVSeparateAVLatent",
        "inputs": {"av_latent": ["sample", 0]},
    }

    # --- Decode + mux ---
    wf["decode_video"] = {
        "class_type": "VAEDecodeTiled",
        "inputs": {
            "samples": ["split", 0],
            "vae": ["vae_video", 0],
            "tile_size": 512,
            "overlap": 64,
            "temporal_size": 64,
            "temporal_overlap": 8,
        },
    }
    wf["decode_audio"] = {
        "class_type": "LTXVAudioVAEDecode",
        "inputs": {"samples": ["split", 1], "audio_vae": ["vae_audio", 0]},
    }
    wf["mkvideo"] = {
        "class_type": "CreateVideo",
        "inputs": {
            "images": ["decode_video", 0],
            "audio": ["decode_audio", 0],
            "fps": float(p["fps"]),
        },
    }
    wf["save"] = {
        "class_type": "SaveVideo",
        "inputs": {
            "video": ["mkvideo", 0],
            "filename_prefix": "ltx25",
            "format": "auto",
            "codec": "auto",
        },
    }
    return wf, mode, seed


# ----------------------------------------------------------------------------
# ComfyUI API helpers
# ----------------------------------------------------------------------------
def comfy_post(path, payload=None, timeout=60):
    r = requests.post(f"{COMFY_URL}{path}", json=payload, timeout=timeout)
    r.raise_for_status()
    return r.json()


def comfy_get(path, timeout=30):
    r = requests.get(f"{COMFY_URL}{path}", timeout=timeout)
    r.raise_for_status()
    return r.json()


def queue_and_wait(workflow):
    queued = comfy_post("/prompt", {"prompt": workflow})
    prompt_id = queued.get("prompt_id")
    if not prompt_id:
        raise WorkerError(f"ComfyUI did not return a prompt_id: {queued}")

    deadline = time.time() + GENERATION_TIMEOUT_S
    while time.time() < deadline:
        time.sleep(5)
        try:
            history = comfy_get(f"/history/{prompt_id}")
        except requests.RequestException:
            continue  # transient — keep polling
        entry = history.get(prompt_id)
        if not entry:
            continue
        status = entry.get("status", {})
        # Check for execution errors FIRST: a failed run can still carry an
        # (empty) "outputs" key, which would otherwise mask the real message.
        if status.get("status_str") == "error":
            raise WorkerError(f"ComfyUI execution error: {status.get('messages')}")
        if status.get("completed") or "outputs" in entry:
            outputs = entry.get("outputs", {})
            for node_id, node_out in outputs.items():
                for key in ("gifs", "videos", "images"):
                    for f in node_out.get(key, []):
                        name = f.get("filename", "")
                        if name.lower().endswith((".mp4", ".webm", ".mov")):
                            return name, f.get("subfolder", "")
            raise WorkerError(f"ComfyUI finished but produced no video file: {outputs}")
    raise WorkerError("ComfyUI generation timed out")


# ----------------------------------------------------------------------------
# Image handling (i2v start frame)
# ----------------------------------------------------------------------------
def fetch_start_frame(image_url, width, height):
    # Accept both remote URLs and data: URLs (PulseClone often passes the
    # character reference as base64). Decode to raw bytes first.
    if image_url.startswith("data:"):
        try:
            header, b64 = image_url.split(",", 1)
        except ValueError:
            raise WorkerError("image_url data URL is malformed")
        if ";base64" not in header:
            raise WorkerError("only base64 data URLs are supported for image_url")
        try:
            raw = base64.b64decode(b64)
        except Exception as e:
            raise WorkerError(f"image_url data URL is not valid base64: {e}")
    else:
        try:
            r = requests.get(image_url, timeout=60, headers={"User-Agent": "pulseclone-ltx-worker/1.0"})
            r.raise_for_status()
            raw = r.content
        except requests.RequestException as e:
            raise WorkerError(f"Could not download image_url: {e}")
    try:
        img = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception as e:
        raise WorkerError(f"image_url is not a readable image: {e}")
    # Center-crop to the target aspect ratio first (never stretch the face),
    # then resize to the exact render dimensions.
    target_ratio = width / height
    img_ratio = img.width / img.height
    if img_ratio > target_ratio:  # too wide: crop the sides
        new_w = int(img.height * target_ratio)
        left = (img.width - new_w) // 2
        img = img.crop((left, 0, left + new_w, img.height))
    elif img_ratio < target_ratio:  # too tall: crop top/bottom
        new_h = int(img.width / target_ratio)
        top = (img.height - new_h) // 2
        img = img.crop((0, top, img.width, top + new_h))
    img = img.resize((width, height), Image.LANCZOS)
    filename = f"i2v_{uuid.uuid4().hex}.png"
    os.makedirs(INPUT_DIR, exist_ok=True)
    img.save(os.path.join(INPUT_DIR, filename))
    return filename


# ----------------------------------------------------------------------------
# RunPod handler
# ----------------------------------------------------------------------------
def handler(job):
    event_input = job.get("input") if isinstance(job, dict) else None
    p = validate(event_input or {})

    if p["image_url"]:
        p["_image_filename"] = fetch_start_frame(p["image_url"], p["width"], p["height"])

    workflow, mode, seed = build_workflow(p)
    filename, _subfolder = queue_and_wait(workflow)

    video_path = os.path.join(OUTPUT_DIR, filename)
    if not os.path.exists(video_path):
        # ComfyUI may nest output under a dated subfolder; search for it.
        found = None
        for root, _dirs, files in os.walk(OUTPUT_DIR):
            if filename in files:
                found = os.path.join(root, filename)
                break
        if not found:
            raise WorkerError(f"ComfyUI reported {filename} but the file is missing")
        video_path = found

    with open(video_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("ascii")

    return {
        "video": f"data:video/mp4;base64,{b64}",
        "metadata": {
            "mode": mode,
            "width": p["width"],
            "height": p["height"],
            "num_frames": p["num_frames"],
            "fps": p["fps"],
            "seed": seed,
            "model": "ltx-2.5-22b-distilled-int8",
            "audio": "native-synchronized",
            "filename": filename,
        },
    }


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
