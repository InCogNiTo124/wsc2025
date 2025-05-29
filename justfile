default:
  just --list

run-triton-gemma:
  docker container run -it --rm --gpus all -v $(pwd):/wsc2025 -p 8000-8002:8000-8002 -e PYTHONDONTWRITEBYTECODE=1 nvcr.io/nvidia/tritonserver:25.06-trtllm-python-py3 tritonserver --model-repository /wsc2025/model_repository --model-control-mode explicit --load-model=gemma_torch --log-verbose 2

run-triton-postprocessing:
  docker container run -it --rm --gpus all -v $(pwd):/wsc2025 -p 8000-8002:8000-8002 -e PYTHONDONTWRITEBYTECODE=1 nvcr.io/nvidia/tritonserver:25.06-trtllm-python-py3 tritonserver --model-repository /wsc2025/model_repository --model-control-mode explicit --load-model=postprocessing --log-verbose 2

run-triton-preprocessing:
  docker container run -it --rm --gpus all -v $(pwd):/wsc2025 -p 8000-8002:8000-8002 -e PYTHONDONTWRITEBYTECODE=1 nvcr.io/nvidia/tritonserver:25.06-trtllm-python-py3 tritonserver --model-repository /wsc2025/model_repository --model-control-mode explicit --load-model=preprocessing --log-verbose 2

probe-triton-gemma:
  curl -X POST localhost:8000/v2/models/gemma_torch/generate --json '{"input_token_ids": [2,105,2364,107,3689,563,236743,236778,236884,236832,236772,236778,236884,236778,236881,106,107,105,4368,107]}'

probe-triton-postprocessing:
  curl -X POST localhost:8000/v2/models/postprocessing/generate --json '{"token_ids": [2,105,2364,107,3689,563,236743,236778,236884,236832,236772,236778,236884,236778,236881,106,107,105,4368,107]}'

probe-triton-preprocessing:
  curl -X POST localhost:8000/v2/models/preprocessing/generate --json '{"prompt": "What is 2^7-2^2?"}'

adhoc-decode x:
  uv run ipython3 -c "from transformers import GemmaTokenizer; GemmaTokenizer.from_pretrained('./checkpoints/google/gemma-3-1b-it/').decode({{ x }})"
