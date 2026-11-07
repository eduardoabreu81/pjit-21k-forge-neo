# P-JiT 21K support for Forge Neo
# The P-JiT text encoder: the outputs of Qwen3.5-4B's full-attention layers 7, 15 and 27 for the caption in Qwen's chat
# template with thinking off, side by side, 7680 features per token (upstream's pjit_21k/text_encoder.py).
#
# Captions are cut at 256 tokens, template included, as upstream's tokenizer call does, and zero-padded to 256: the DiT
# reads the caption's length from the padding. Every caption has the same shape, so Forge batches them as they are, and
# the positive and the negative prompt run in one pass, as upstream's do.

import torch
from backend.args import dynamic_args
from backend.text_processing import emphasis
from backend.text_processing._comfy import INF, SDClipModel, SDTokenizer

# <|endoftext|>: never part of the template, so Forge's end-of-text masking never cuts the caption
PAD = 248044
TEXT_LAYERS = [7, 15, 27]
TEMPLATE = "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
MAX_LENGTH = 256


def pad_caption(states: torch.Tensor) -> torch.Tensor:
    """(T, D) -> (MAX_LENGTH, D), with zeros after the caption."""
    return torch.cat([states, states.new_zeros(MAX_LENGTH - states.shape[0], states.shape[1])], dim=0)


class PJiTTextEngine:
    def __init__(self, text_encoder, tokenizer):
        self.text_encoder = SDClipModel(text_encoder, layer=list(TEXT_LAYERS), layer_idx=None,
                                        special_tokens={"pad": PAD}, layer_norm_hidden_state=False)
        self.tokenizer = SDTokenizer(tokenizer, pad_with_end=False, has_start_token=False, has_end_token=False,
                                     pad_to_max_length=False, max_length=INF, min_length=1, pad_token=PAD)

    @property
    def emphasis(self) -> "emphasis.Emphasis":
        return emphasis.EmphasisNone()

    def token_ids(self, text: str) -> list[int]:
        # Qwen's template trims the message; "", " " or only LoRA tags are upstream's empty negative prompt
        return self.tokenizer.tokenizer.encode(TEMPLATE.format(text.strip()), add_special_tokens=False)[:MAX_LENGTH]

    def tokenize(self, texts: str | list[str]) -> int | list[int]:
        if isinstance(texts, str):
            return len(self.token_ids(texts))
        return [len(self.token_ids(t)) for t in texts]

    def __call__(self, texts: list[str]) -> list[torch.Tensor]:
        if any(emphasis.uses_emphasis(text) for text in texts):
            dynamic_args.last_extra_generation_params["Emphasis"] = "None"

        zs = []
        cache = {}
        for line in texts:
            if line not in cache:
                cache[line] = self.encode(line)
            zs.append(cache[line])
        return zs

    def encode(self, line: str) -> torch.Tensor:
        states = self.text_encoder.encode_token_weights([[(token, 1.0) for token in self.token_ids(line)]])[0]
        # (1, T, 7680) -> (256, 7680): no batch dimension, as Forge's native engines return their conditioning
        return pad_caption(states[0])
