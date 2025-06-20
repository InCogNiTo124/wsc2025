import json

import torch

torch.inference_mode()
DEVICE = torch.device("cuda:0")

import triton_python_backend_utils as pb_utils

from transformers import Gemma3ForCausalLM

import numpy as np


class TritonPythonModel:
    def initialize(self, args):
        # Parse model configs
        model_config = json.loads(args["model_config"])
        model_dir = model_config["parameters"]["model_dir"]["string_value"]

        self.model = (
            Gemma3ForCausalLM.from_pretrained(model_dir, device_map=DEVICE)
            .bfloat16()
            .eval()
        )
        self.model.generation_config.max_new_tokens = 256

        return

    def execute(self, requests):
        responses = []
        for request in requests:
            input_token_ids_tensor = pb_utils.get_input_tensor_by_name(
                request, "input_token_ids"
            ).as_numpy()

            input_token_ids_torch = torch.tensor(
                [input_token_ids_tensor], device=DEVICE
            )

            output_token_ids = self.model.generate(input_token_ids_torch).cpu().numpy()

            output_tokens_tensor = pb_utils.Tensor(
                "output_token_ids", output_token_ids.astype(np.int32).reshape(1, 1, -1)
            )

            response = pb_utils.InferenceResponse([output_tokens_tensor])

            responses.append(response)
        return responses

    def finalize(self):
        print("Cleaning up gemma_torch...")
