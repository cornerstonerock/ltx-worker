# LTX-2.5 RunPod Serverless Worker

Text-to-video + image-to-video with **natively synchronized audio**, powered by
Lightricks LTX-2.5 (22B joint audio/video diffusion model, int8 distilled build)
running in ComfyUI.

## How it works

Single-stage distilled workflow (8 steps, CFG 1.0, euler_ancestral):
1. `UNETLoader` (int8 transformer) + `CLIPLoader` (type `ltxv`, Gemma 4) + 2× `VAELoader` (video conv VAE + audio VAE)
2. `CLIPTextEncode` ×2 → `LTXVConditioning` (frame rate stamped)
3. `EmptyLTXVLatentVideo` + `LTXVEmptyLatentAudio` → `LTXVConcatAVLatent` (joint AV latent)
   - I2V: `LTXVPreprocess` + `LTXVImgToVideoInplace` injects the start frame
4. `SamplerCustomAdvanced` denoises the **joint** latent (video+audio together)
5. `LTXVSeparateAVLatent` → `VAEDecodeTiled` + `LTXVAudioVAEDecode`
6. `CreateVideo` muxes frames + waveform → `SaveVideo`

Audio is generated in the same diffusion pass as video — no separate TTS step.

## API contract

Input:
```json
{
  "prompt": "string (required)",
  "negative_prompt": "string (optional)",
  "image_url": "string URL or data: URL (optional — enables i2v)",
  "num_frames": 97 (8n+1, clamped 9–121),
  "width": 768, "height": 512 (snapped to /32, clamped 256–1024),
  "fps": 24 (8–30),
  "seed": 123 (optional)
}
```

Output:
```json
{
  "video": "data:video/mp4;base64,...",
  "metadata": { "mode": "t2v|i2v", "width": 768, "height": 512,
                "num_frames": 97, "fps": 24, "seed": 123,
                "model": "ltx-2.5-22b-distilled-int8",
                "audio": "native-synchronized", "filename": "ltx25_....mp4" }
}
```

## Models (network volume)

Expected at `/runpod-volume/models/ltx-2.5/`:
- `diffusion_models/ltx-2.5-22b-distilled-transformer-comfy-int8-convrot.safetensors`
- `text_encoders/gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors`
- `vae/ltx-2.5-video-vae-conv-bf16.safetensors` (conv = lower VRAM)
- `vae/ltx-2.5-audio-vae-bf16.safetensors`

## VRAM notes (24GB cards)

- Target: single-stage, 768×512, ≤97 frames, `--lowvram`, conv video VAE.
- Never use `--use-sage-attention` with comfy_quant int8 (crashes).
- Never use `LTXVGemmaCLIPModelLoader` with the int8 encoder — use `CLIPLoader` type `ltxv`.
- If OOM: fewer frames (65), smaller tiles, or 512×320.

## License note

LTX-2.5 weights are under Lightricks' Community License (free commercial use
under $10M annual revenue, with restrictions — including a competition clause).
Review before production use. AI-generated content disclosure required.
