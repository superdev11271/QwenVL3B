# Qwen VL API

FastAPI service for OCR and text correction powered by [Qwen2.5-VL](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct), running inference through [llama.cpp](https://github.com/ggml-org/llama.cpp) `llama-server`.

The API accepts images or raw text, applies language-specific system prompts, and optionally detects emotion for TTS-style annotation.

## Architecture

```
Client → FastAPI (:8000) → llama-server OpenAI API (:8080) → Qwen2.5-VL GGUF
```

- **FastAPI** (`main.py`) — HTTP endpoints, image handling, prompt routing
- **llama-server** — GPU-accelerated multimodal inference via GGUF + mmproj files

## Requirements

- NVIDIA GPU with [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) (for Docker)
- Qwen2.5-VL GGUF model files mounted at `/models`, for example:
  - `Qwen2.5-VL-3B-Instruct-GGUF/Qwen2.5-VL-3B-Instruct-Q8_0.gguf`
  - `Qwen2.5-VL-3B-Instruct-GGUF/mmproj-model-f16.gguf`

## Quick Start (Docker)

```bash
docker build -t qwen-api .

docker run --gpus all \
  -v /path/to/models:/models \
  -p 8000:8000 \
  qwen-api
```

Check health:

```bash
curl http://localhost:8000/health
```

## Local Development

Run `llama-server` and the API separately.

**1. Start llama-server**

```bash
./llama-server \
  --model /models/Qwen2.5-VL-3B-Instruct-GGUF/Qwen2.5-VL-3B-Instruct-Q8_0.gguf \
  --mmproj /models/Qwen2.5-VL-3B-Instruct-GGUF/mmproj-model-f16.gguf \
  --host 0.0.0.0 \
  --port 8080 \
  --ctx-size 16384 \
  --n-gpu-layers -1 \
  --no-jinja
```

**2. Install dependencies and start FastAPI**

```bash
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8000
```

## API

### `GET /health`

Returns service status.

```json
{"status": "ok"}
```

### `POST /ocr`

Extract and correct text from an image.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `file` | file | required | Image file |
| `lang` | string | `en` | `en` or `zh` |
| `skip_emotion` | bool | `true` | Skip emotion detection pass |

```bash
curl -X POST http://localhost:8000/ocr \
  -F "file=@test.png" \
  -F "lang=en" \
  -F "skip_emotion=true"
```

```json
{
  "result": "Corrected text from the image.",
  "emotion": null
}
```

### `POST /fix`

Correct grammar and spelling in plain text.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `text` | string | required | Input text |
| `lang` | string | `en` | `en` or `zh` |
| `skip_emotion` | bool | `true` | Skip emotion detection pass |

```bash
curl -X POST http://localhost:8000/fix \
  -F "text=This are a example sentance." \
  -F "lang=en"
```

```json
{
  "result": "This is an example sentence.",
  "emotion": null
}
```

Set `skip_emotion=false` to run a second pass that annotates the result with expressive TTS tags.

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `LLAMA_API_BASE` | `http://localhost:8080/v1` | llama-server OpenAI-compatible API URL |
| `LLAMA_API_KEY` | `not-needed` | API key passed to the OpenAI client |
| `MODEL_NAME` | `local-model` | Model name sent to llama-server |
| `MODEL_FILE` | `/models/.../Qwen2.5-VL-3B-Instruct-Q8_0.gguf` | Main GGUF model path (Docker) |
| `MMPROJ_FILE` | `/models/.../mmproj-model-f16.gguf` | Vision projector GGUF path (Docker) |
| `LLAMA_HOST` | `0.0.0.0` | llama-server bind address (Docker) |
| `LLAMA_PORT` | `8080` | llama-server port (Docker) |
| `CTX_SIZE` | `16384` | Context window size (Docker) |
| `N_THREADS` | `4` | CPU threads for llama-server (Docker) |
| `N_GPU_LAYERS` | `-1` | GPU layers to offload; `-1` = all (Docker) |

## Project Layout

```
.
├── main.py              # FastAPI application
├── entrypoint.sh        # Starts llama-server, then uvicorn (Docker)
├── Dockerfile
├── requirements.txt
├── system_prompts/      # OCR, text fix, and emotion prompts (en/zh)
└── test.py              # Legacy local transformers test script
```

## Ports

| Port | Service |
|------|---------|
| `8000` | FastAPI (public API) |
| `8080` | llama-server (internal, OpenAI-compatible) |
