import json

import triton_python_backend_utils as pb_utils

from transformers import GemmaTokenizer

import numpy as np


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
            input_tensor = pb_utils.get_input_tensor_by_name(
                request, "prompt"
            ).as_numpy()
            prompt = input_tensor[0].decode("utf-8")

            tokenized = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="np")
            token_ids = tokenized.input_ids.astype(np.int32)

            token_ids_tensor = pb_utils.Tensor("token_ids", token_ids[0])
            response = pb_utils.InferenceResponse(output_tensors=[token_ids_tensor])
            responses.append(response)

        return responses

    def finalize(self):
        print("Cleaning up preprocessing...")
