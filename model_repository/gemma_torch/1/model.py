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

        self.model = ...
        return

    def execute(self, requests):
        responses = []
        for request in requests:
            input_token_ids_tensor = pb_utils.get_input_tensor_by_name(
                request, "INPUT_NAME"
            ).as_numpy()

            ### YOUR CODE HERE ###

            input_token_ids_torch = ...

            output_token_ids = ...

            output_tensor = pb_utils.Tensor("OUTPUT_NAME", ...)

            response = pb_utils.InferenceResponse([output_tensor])

            responses.append(response)
        return responses

    def finalize(self):
        print("Cleaning up gemma_torch...")
