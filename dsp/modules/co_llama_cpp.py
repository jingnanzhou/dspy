
from dsp.modules.lm import LM

from llama_cpp import Llama
import os
from typing import Any, Literal

class LlamaCpp(LM):
    """
    Args:
        model (str, optional): Name of local llm
        model_path 
        timeout_s (float): Timeout period (in seconds) for the post request to llm.
        **kwargs: Additional arguments to pass to the API.
    """

    def __init__(
        self,
        model: str,
        model_path: str ,
        timeout_s: float = 120,
        temperature: float = 0.0,
        max_tokens: int = 150,
        top_p: int = 1,
        top_k: int = 20,
        frequency_penalty: float = 0,
        presence_penalty: float = 0,
        n: int = 1,
        num_ctx: int = 4096,
        **kwargs,
    ):
        super().__init__(model)

        
        self.model_path = model_path

        self.model_name = model
        path = os.path.join(self.model_path, self.model_name)
        self.model =  Llama(model_path=path,
                #n_gpu_layers=-1, # Uncomment to use GPU acceleration
                # seed=1337, # Uncomment to set a specific seed
                n_ctx=num_ctx, # Uncomment to increase the context window
        )

        self.timeout_s = timeout_s
        self.kwargs = {
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": top_p,
            "top_k": top_k,
            "frequency_penalty": frequency_penalty,
            "presence_penalty": presence_penalty,
            "n": n,
            "num_ctx": num_ctx,
            **kwargs,
        }

        # Ollama uses num_predict instead of max_tokens
        self.kwargs["num_predict"] = self.kwargs["max_tokens"]

        self.history: list[dict[str, Any]] = []
        self.version = kwargs["version"] if "version" in kwargs else ""

        # Ollama occasionally does not send `prompt_eval_count` in response body.
        # https://github.com/stanfordnlp/dspy/issues/293
        self._prev_prompt_eval_count = 0

    def basic_request(self, prompt: str, **kwargs):
        raw_kwargs = kwargs

        kwargs = {**self.kwargs, **kwargs}


        output =self.model.create_chat_completion(
            messages=[{
                        "role": "user",
                        "content": prompt,
                }],
            max_tokens = 1000,
            stop = [],

            temperature = 0.8,
            top_p = 0.95,
            model = None,
        )


        response = {
            "prompt": prompt,
            "choices": output["choices"],
        }



        history = {
            "prompt": prompt,
            "response": response,
            "kwargs": kwargs,
            "raw_kwargs": raw_kwargs,
        }
        self.history.append(history)
        
        return response


    def __call__(
        self,
        prompt: str,
        only_completed: bool = True,
        return_sorted: bool = False,
        **kwargs,
    ) -> list[dict[str, Any]]:

        output =self.model.create_chat_completion(
            messages=[{
                        "role": "user",
                        "content": prompt,
                }],
        max_tokens = 1000,
        stop = [],

        temperature = 0.8,
        top_p = 0.95,
        model = None

        )
        #choices = output["choices"][0]
        #def _get_choice_text(self, choice: dict[str, Any]) -> str:
        #    return choice["message"]["content"]
        #completions = [_get_choice_text(c) for c in choices]

        completions=[output["choices"][0]["message"]["content"]]
        return completions
