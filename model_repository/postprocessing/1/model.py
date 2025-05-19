import json

import numpy as np

import triton_python_backend_utils as pb_utils

from transformers import GemmaTokenizer


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
            token_ids_tensor = pb_utils.get_input_tensor_by_name(
                request, "INPUT_NAME"
            ).as_numpy()

            ### YOUR CODE HERE

            decoded_text = ...

            output_tensor = pb_utils.Tensor(
                "OUTPUT_NAME", np.array(decoded_text, dtype=object)
            )
            response = pb_utils.InferenceResponse([output_tensor])
            responses.append(response)
        return responses

    def finalize(self):
        print("Cleaning up postprocessing...")
