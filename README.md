# MekaHime Pipeline V1 (STT -> LLM -> TTS -> RVC)

MIT License. Developed by MekaHime Inc. This repository contains retired legacy code that the company has approved for open-source release. Use is at your own risk; MekaHime Inc. assumes no responsibility or liability for user actions and provides no support or maintenance obligations for this project.

For current developments, updates, and community discussion, please visit `www.MekaHime.com` and join the Discord server linked on the site.

This folder contains an end-to-end voice pipeline that uses:

1) Whisper STT + WebRTC VAD
2) Ollama LLM
3) Coqui TTS
4) RVC voice conversion

## Environment used for this run

- Tsted only on Linux (Ubuntu 24.04)
- GPU: NVIDIA RTX 40-series with CUDA 12.7
- Not guaranteed to work on other setups, but you can try.

## Configure paths (required)

Edit the user-configurable paths at the top of `MKHM_Pipeline_V1.py`:

- `WHISPER_REALTIME_DIR`
- `COQUI_TTS_DIR`
- `RVC_STS_DIR`
- `INITIAL_PROMPT_PATH`

The script expects these RVC assets relative to `RVC_STS_DIR`:

- `assets/weights/*.pth`
- `logs/*.index` (optional but recommended for quality)
- `assets/hubert/hubert_base.pt` (required)
- `assets/rmvpe/rmvpe.pt` or `assets/rmvpe/rmvpe_inputs.pth` (optional; enables RMVPE)

## To-install list

System tools:

- `ollama` (LLM server)
- `ffmpeg` (Whisper audio decoding)
- `portaudio` (for `sounddevice`)
- `pulseaudio` tools (`pactl`, `paplay` or `aplay`)

Python packages (minimum):

- `numpy`
- `torch`
- `openai-whisper`
- `webrtcvad`
- `sounddevice`
- `soundfile`
- `scipy`
- `requests`
- `TTS` (Coqui TTS)

Local repos/paths:

- Whisper real-time repo (path set in `WHISPER_REALTIME_DIR`)
- Coqui TTS repo (path set in `COQUI_TTS_DIR`)
- RVC repo (path set in `RVC_STS_DIR`)

## Requirements and notes

- Python 3.9+ recommended.
- A working microphone and audio output device.
- Optional CUDA GPU for faster Whisper/TTS/RVC.
- Ollama model used by default: `huihui_ai/qwen2.5-1m-abliterated:14b`.
- The persona prompt file is loaded from `INITIAL_PROMPT_PATH` and can be edited to change the assistant personality.


## Ideal folder setup

Example layout (update the paths in `MKHM_Pipeline_V1.py` to match):

```
/path/to/
  whisper_real_time/
  coqui_tts/
  rvc_sts/
    assets/
      hubert/
        hubert_base.pt
      rmvpe/
        rmvpe.pt
        rmvpe_inputs.pth
      weights/
        your_model.pth
    logs/
      your_model.index
  initial_prompt.txt
```

## Pip requirements

Install the Python dependencies with pip:

```
pip install numpy torch openai-whisper webrtcvad sounddevice soundfile scipy requests TTS
```

## Run

```bash
python3 MKHM_Pipeline_V1.py
```
