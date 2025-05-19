default:
  just --list

run-triton-preprocessing:
  docker container run -it --rm --gpus all -v $(pwd):/wsc2025 -p 8000-8002:8000-8002 -e PYTHONDONTWRITEBYTECODE=1 nvcr.io/nvidia/tritonserver:25.06-trtllm-python-py3 tritonserver --model-repository /wsc2025/model_repository --model-control-mode explicit --load-model=preprocessing --log-verbose 2

probe-triton-preprocessing:
  curl -X POST localhost:8000/v2/models/preprocessing/generate --json '{"prompt": "What is 2^7-2^2?"}'

adhoc-decode x:
  uv run ipython3 -c "from transformers import GemmaTokenizer; GemmaTokenizer.from_pretrained('./checkpoints/google/gemma-3-1b-it/').decode({{ x }})"
