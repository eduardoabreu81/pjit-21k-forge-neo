"""Minimal stand-ins for the Forge Neo pieces the extension uses, for CPU tests without Forge.

Only the names the extension touches exist; anything else raises AttributeError, so a test notices when the
extension starts depending on more of Forge. Same shape as the FLUX 3 extension's forge_stubs.
"""

import sys
import types
from contextlib import contextmanager

import torch
import torch.nn.functional as F


def module(name, **attributes):
    result = types.ModuleType(name)
    for key, value in attributes.items():
        setattr(result, key, value)
    return result


class ForgeRMSNorm(torch.nn.RMSNorm):
    """Forge's operations.RMSNorm without its weight management: add=True scales by 1 + weight."""

    def __init__(self, *args, add=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.add = add

    def forward(self, x):
        weight = self.weight + 1.0 if self.add else self.weight
        return F.rms_norm(x, self.normalized_shape, weight, self.eps)


@contextmanager
def forge_operations():
    """torch.nn.RMSNorm as Forge's using_forge_operations installs it while a model is built."""
    original = torch.nn.RMSNorm
    torch.nn.RMSNorm = ForgeRMSNorm
    try:
        yield
    finally:
        torch.nn.RMSNorm = original


class FakeHFTokenizer:
    """One token per character, ids above the special range; PAD 248044 never appears in text."""

    def encode(self, text, add_special_tokens=True):
        return [1000 + ord(c) for c in text]


class FakeSDTokenizer:
    def __init__(self, tokenizer_object=None, **kwargs):
        self.tokenizer = tokenizer_object


class FakeSDClipModel:
    """encode_token_weights as Forge's does with layer "last": states (1, T, D), token t's features all t."""

    dim = 3

    def __init__(self, transformer=None, **kwargs):
        self.transformer = transformer
        self.kwargs = kwargs
        self.calls = []

    def encode_token_weights(self, token_weight_pairs):
        tokens = [pair[0] for pair in token_weight_pairs[0]]
        self.calls.append(tokens)
        states = torch.arange(len(tokens), dtype=torch.float32)[None, :, None].expand(1, len(tokens), self.dim)
        return states.clone(), None


def text_processing():
    """backend.args and backend.text_processing, enough to import forge_pjit.native.text_engine."""
    comfy = module("backend.text_processing._comfy", INF=float("inf"), SDTokenizer=FakeSDTokenizer,
                   SDClipModel=FakeSDClipModel)
    emphasis = module("backend.text_processing.emphasis", uses_emphasis=lambda text: "(" in text,
                      EmphasisNone=type("EmphasisNone", (), {}))
    package = module("backend.text_processing", emphasis=emphasis, _comfy=comfy)
    args = module("backend.args", dynamic_args=types.SimpleNamespace(last_extra_generation_params={}))
    return {"backend": module("backend", args=args, text_processing=package), "backend.args": args,
            "backend.text_processing": package, "backend.text_processing.emphasis": emphasis,
            "backend.text_processing._comfy": comfy}


@contextmanager
def installed(stubs):
    """The stubs in sys.modules for the duration, and the extension's modules re-imported against them."""
    saved = {name: sys.modules.get(name) for name in stubs}
    ours = {name: mod for name, mod in sys.modules.items() if name.startswith("forge_pjit")}
    sys.modules.update(stubs)
    for name in ours:
        del sys.modules[name]
    try:
        yield
    finally:
        for name, mod in saved.items():
            if mod is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = mod
        for name in [n for n in sys.modules if n.startswith("forge_pjit")]:
            del sys.modules[name]
        sys.modules.update(ours)
