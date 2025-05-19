import json

import triton_python_backend_utils as pb_utils

from transformers import GemmaTokenizer

import numpy as np


class TritonPythonModel:
    def initialize(self, args):
        # Parse model configs
        model_config = json.loads(args["model_config"])
        tokenizer_dir = model_config["parameters"]["tokenizer_dir"]["string_value"]

        # 1. load tokenizer
        self.tokenizer = ...
        return

    def execute(self, requests):
        responses = []
        for request in requests:
            input_tensor = pb_utils.get_input_tensor_by_name(
                request, "INPUT_NAME"
            ).as_numpy()
            prompt = input_tensor[0].decode("utf-8")

            # 2. tokenize the prompt
            tokenized = ...
            token_ids = ...

            token_ids_tensor = pb_utils.Tensor("OUTPUT_NAME", token_ids[0])
            response = pb_utils.InferenceResponse(output_tensors=[token_ids_tensor])
            responses.append(response)

        return responses

    def finalize(self):
        print("Cleaning up preprocessing...")
