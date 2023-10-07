from typing import Any, Dict, List, Union

import pydantic
import torch
from transformers import AutoTokenizer


class GeneratorConfig(pydantic.BaseModel):
    """Config for generation."""

    repetition_penalty: float = 1.2
    beam_search: bool = True
    num_beams: int = 5
    early_stopping: bool = True
    max_length: int = 64
    no_repeat_ngram_size: int = 2
    top_k: int = 2000
    top_p: float = 0.95
    beam_search_params: Dict[str, Any] = {
        "max_length": max_length,
        "num_beams": num_beams,
        "no_repeat_ngram_size": no_repeat_ngram_size,
        "early_stopping": early_stopping,
        "repetition_penalty": repetition_penalty,
    }


class Tokenizer:
    def __init__(self, tokenizer, max_seq_len: int = None) -> None:
        """Wrapper for tokenizer.

        :param tokenizer: the tokenizer to be wrapped
        :param max_seq_len: the maximum sequence length
        """
        super().__init__()
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

    def __getattr__(self, attribute: str):
        if hasattr(self.tokenizer, attribute):
            return getattr(self.tokenizer, attribute)
        else:
            raise AttributeError(f"{attribute} not found")

    def __call__(
        self, sentences: List[str], device: torch.device = None
    ) -> Union[Dict, AutoTokenizer]:
        tokenized = self.tokenizer(
            sentences,
            truncation=True,
            padding=True,
            max_length=self.max_seq_len,
            return_tensors="pt",
        )
        if device is not None:
            return {key: tensor.to(device) for key, tensor in tokenized.items()}
        return tokenized

    def decode(self, x: Dict[str, torch.Tensor]):
        """Decode the tokenized sentences.

        :param x: the tokenized sentences
        :return: the decoded sentences
        """
        return [
            self.tokenizer.decode(sentence[:sentence_len])
            for sentence, sentence_len in zip(x["input_ids"], x["attention_mask"].sum(dim=-1))
        ]

    def batch_decode(self, encoded_outputs: torch.Tensor) -> List[str]:
        """Decode the tokenized sentences in batch.

        :param encoded_outputs the encoded outputs
        :return: the decoded sentences
        """
        return self.tokenizer.batch_decode(encoded_outputs.cpu(), skip_special_tokens=True)

    def __len__(self):
        return len(self.tokenizer)
