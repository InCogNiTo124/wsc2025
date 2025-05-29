import json

import numpy as np

import triton_python_backend_utils as pb_utils

from transformers import GemmaTokenizer


class TritonPythonModel:
    def initialize(self, args):
        # Parse model configs
        model_config = json.loads(args["model_config"])
        tokenizer_dir = model_config["parameters"]["tokenizer_dir"]["string_value"]

        self.tokenizer = GemmaTokenizer.from_pretrained(tokenizer_dir)
        return

    def execute(self, requests):
        responses = []
        for request in requests:
            token_ids_tensor = pb_utils.get_input_tensor_by_name(
                request, "token_ids"
            ).as_numpy()  # shape(-1)

            decoded_text = self.tokenizer.decode(token_ids_tensor)

            output_tensor = pb_utils.Tensor(
                "decoded_text", np.array(decoded_text, dtype=object)
            )
            response = pb_utils.InferenceResponse([output_tensor])
            responses.append(response)
        return responses

    def finalize(self):
        print("Cleaning up postprocessing...")
