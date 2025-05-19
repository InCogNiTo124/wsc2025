local_ip := `ip addr show eth0 | grep -oP 'inet \K[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}(?=/)'`
test-probe-models: docker-compose
  echo "Visit http://{{ local_ip }}:8000/grafana to see graphs"
  # this probes models so that you can see something in the graphs
  # also, since this deployment is now behind trafik, I changed the api
  sleep 1s
  curl -X POST localhost:8000/triton/ensemble_torch --json '{"prompt": "What is 2^7-2^4?"}'
  curl -X POST localhost:8000/triton/ensemble_torch --json '{"prompt": "Integrate x dx from 1 to 3"}'
  curl -X POST localhost:8000/triton/ensemble_torch --json '{"prompt": "Differentiate x^2-3x+1"}'

  sleep 10s

  curl -X POST localhost:8000/triton/ensemble_trt --json '{"prompt": "What is 2^7-2^4?", "request_output_len": 2048, "end_id": 106}'
  curl -X POST localhost:8000/triton/ensemble_trt --json '{"prompt": "Integrate x dx from 1 to 3", "request_output_len": 2048, "end_id": 106}'
  curl -X POST localhost:8000/triton/ensemble_trt --json '{"prompt": "Differentiate x^2-3x+1", "request_output_len": 2048, "end_id": 106}'


docker-compose: write-local-ip
  docker compose up -d 

write-local-ip:
  # horribly ugly hack in order to serve grafana behind traefik.
  # this is needed because every user's instance gets a different IP address
  # so I can't hardcode a value in the configuration, I need 
  sed -i 's/\${local_ip}/{{local_ip}}/g' docker-compose.yaml

run-triton-all:
  docker container run -it --rm --gpus all -v $(pwd):/wsc2025 -p 8000-8002:8000-8002 -e PYTHONDONTWRITEBYTECODE=1 nvcr.io/nvidia/tritonserver:25.06-trtllm-python-py3 tritonserver --model-repository /wsc2025/model_repository --log-verbose 2

run-triton-ensemble-trt:
  docker container run -it --rm --gpus all -v $(pwd):/wsc2025 -p 8000-8002:8000-8002 -e PYTHONDONTWRITEBYTECODE=1 nvcr.io/nvidia/tritonserver:25.06-trtllm-python-py3 tritonserver --model-repository /wsc2025/model_repository --model-control-mode explicit --load-model=ensemble_trt --load-model=tensorrt_llm --load-model=preprocessing --load-model=postprocessing --log-verbose 2

run-triton-tensorrtllm:
  docker container run -it --rm --gpus all -v $(pwd):/wsc2025 -p 8000-8002:8000-8002 -e PYTHONDONTWRITEBYTECODE=1 nvcr.io/nvidia/tritonserver:25.06-trtllm-python-py3 tritonserver --model-repository /wsc2025/model_repository --model-control-mode explicit --load-model=tensorrt_llm --log-verbose 2

run-triton-ensemble-torch:
  docker container run -it --rm --gpus all -v $(pwd):/wsc2025 -p 8000-8002:8000-8002 -e PYTHONDONTWRITEBYTECODE=1 nvcr.io/nvidia/tritonserver:25.06-trtllm-python-py3 tritonserver --model-repository /wsc2025/model_repository --model-control-mode explicit --load-model=preprocessing --load-model=gemma_torch --load-model=postprocessing --load-model=ensemble_torch --log-verbose 2

run-triton-gemma:
  docker container run -it --rm --gpus all -v $(pwd):/wsc2025 -p 8000-8002:8000-8002 -e PYTHONDONTWRITEBYTECODE=1 nvcr.io/nvidia/tritonserver:25.06-trtllm-python-py3 tritonserver --model-repository /wsc2025/model_repository --model-control-mode explicit --load-model=gemma_torch --log-verbose 2

run-triton-postprocessing:
  docker container run -it --rm --gpus all -v $(pwd):/wsc2025 -p 8000-8002:8000-8002 -e PYTHONDONTWRITEBYTECODE=1 nvcr.io/nvidia/tritonserver:25.06-trtllm-python-py3 tritonserver --model-repository /wsc2025/model_repository --model-control-mode explicit --load-model=postprocessing --log-verbose 2

run-triton-preprocessing:
  docker container run -it --rm --gpus all -v $(pwd):/wsc2025 -p 8000-8002:8000-8002 -e PYTHONDONTWRITEBYTECODE=1 nvcr.io/nvidia/tritonserver:25.06-trtllm-python-py3 tritonserver --model-repository /wsc2025/model_repository --model-control-mode explicit --load-model=preprocessing --log-verbose 2

probe-triton-ensemble-trt:
  curl -X POST localhost:8000/v2/models/ensemble_trt/generate --json '{"prompt": "What is 2^7-2^2?", "request_output_len": 2048, "end_id": 106}'

probe-triton-tensorrtllm:
  curl -X POST localhost:8000/v2/models/tensorrt_llm/generate --json '{"input_ids": [2,105,2364,107,3689,563,236743,236778,236884,236832,236772,236778,236884,236778,236881,106,107,105,4368,107], "request_output_len": 256}'

probe-triton-ensemble-torch:
  curl -X POST localhost:8000/v2/models/ensemble_torch/generate --json '{"prompt": "What is 2^7-2^2?"}'

probe-triton-gemma:
  curl -X POST localhost:8000/v2/models/gemma_torch/generate --json '{"input_token_ids": [2,105,2364,107,3689,563,236743,236778,236884,236832,236772,236778,236884,236778,236881,106,107,105,4368,107]}'

probe-triton-postprocessing:
  curl -X POST localhost:8000/v2/models/postprocessing/generate --json '{"token_ids": [2,105,2364,107,3689,563,236743,236778,236884,236832,236772,236778,236884,236778,236881,106,107,105,4368,107]}'

probe-triton-preprocessing:
  curl -X POST localhost:8000/v2/models/preprocessing/generate --json '{"prompt": "What is 2^7-2^2?"}'

convert-checkpoint:
  uv run convert_checkpoint.py --ckpt-type hf --model-dir checkpoints/google/gemma-3-1b-it --dtype bfloat16 --world-size 1 --output-model-dir checkpoints/gemma3_converted

trtllm-build:
  uv run trtllm-build --checkpoint_dir checkpoints/gemma3_converted --gemm_plugin auto --max_batch_size 1 --max_input_len 1024 --max_seq_len 2048 --output_dir checkpoints/trt_out

adhoc-decode x:
  uv run ipython3 -c "from transformers import GemmaTokenizer; GemmaTokenizer.from_pretrained('./checkpoints/google/gemma-3-1b-it/').decode({{ x }})"
