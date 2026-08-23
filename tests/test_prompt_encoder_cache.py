from __future__ import annotations

from types import SimpleNamespace

import torch
import torch.nn as nn

from wm3d_wam.training import trainer as trainer_module
from wm3d_wam.training.trainer import PromptEncoderCache


class RecordingEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(()))
        self.moves: list[torch.device] = []

    def to(self, *args, **kwargs):
        device = kwargs.get("device", args[0] if args else None)
        self.moves.append(torch.device(device))
        return self


def test_prompt_cache_offloads_frozen_encoder_after_a_cache_miss(monkeypatch) -> None:
    encoder = RecordingEncoder()
    components = SimpleNamespace(text_encoder=encoder, tokenizer=object())
    encode_calls: list[tuple[str, ...]] = []
    empty_cache_calls: list[bool] = []

    def fake_encode(components, prompts, *, device, dtype):
        del components, device
        encode_calls.append(tuple(prompts))
        context = torch.ones((len(prompts), 2, 3), dtype=dtype)
        mask = torch.ones((len(prompts), 2), dtype=torch.bool)
        return context, mask

    monkeypatch.setattr(
        trainer_module,
        "encode_local_wan_prompts",
        fake_encode,
    )
    monkeypatch.setattr(
        torch.cuda,
        "empty_cache",
        lambda: empty_cache_calls.append(True),
    )
    cache = PromptEncoderCache(
        components=components,
        device=torch.device("cuda", 0),
        dtype=torch.bfloat16,
        max_entries=2,
    )

    first = cache.encode("pick up the cup")
    second = cache.encode("pick up the cup")

    assert encoder.moves == [torch.device("cuda", 0), torch.device("cpu")]
    assert encode_calls == [("pick up the cup",)]
    assert empty_cache_calls == [True]
    assert first[0] is second[0]
    assert first[1] is second[1]


def test_prompt_cache_batches_all_microbatch_misses_into_one_transfer(monkeypatch) -> None:
    encoder = RecordingEncoder()
    components = SimpleNamespace(text_encoder=encoder, tokenizer=object())
    encode_calls: list[tuple[str, ...]] = []
    empty_cache_calls: list[bool] = []

    def fake_encode(components, prompts, *, device, dtype):
        del components, device
        encode_calls.append(tuple(prompts))
        context = torch.ones((len(prompts), 2, 3), dtype=dtype)
        mask = torch.ones((len(prompts), 2), dtype=torch.bool)
        return context, mask

    monkeypatch.setattr(trainer_module, "encode_local_wan_prompts", fake_encode)
    monkeypatch.setattr(
        torch.cuda,
        "empty_cache",
        lambda: empty_cache_calls.append(True),
    )
    cache = PromptEncoderCache(
        components=components,
        device=torch.device("cuda", 0),
        dtype=torch.bfloat16,
        max_entries=4,
    )

    context, mask = cache.encode_batch(["task a", "task b", "task a"])
    cached_context, cached_mask = cache.encode_batch(["task b", "task a"])

    assert encoder.moves == [torch.device("cuda", 0), torch.device("cpu")]
    assert encode_calls == [("task a", "task b")]
    assert empty_cache_calls == [True]
    assert context.shape == (3, 2, 3)
    assert mask.shape == (3, 2)
    assert cached_context.shape == (2, 2, 3)
    assert cached_mask.shape == (2, 2)


def test_prompt_cache_can_keep_frozen_encoder_resident(monkeypatch) -> None:
    encoder = RecordingEncoder()
    components = SimpleNamespace(text_encoder=encoder, tokenizer=object())
    encode_calls: list[tuple[str, ...]] = []
    empty_cache_calls: list[bool] = []

    def fake_encode(components, prompts, *, device, dtype):
        del components, device
        encode_calls.append(tuple(prompts))
        context = torch.ones((len(prompts), 2, 3), dtype=dtype)
        mask = torch.ones((len(prompts), 2), dtype=torch.bool)
        return context, mask

    monkeypatch.setattr(trainer_module, "encode_local_wan_prompts", fake_encode)
    monkeypatch.setattr(
        torch.cuda,
        "empty_cache",
        lambda: empty_cache_calls.append(True),
    )
    cache = PromptEncoderCache(
        components=components,
        device=torch.device("cuda", 0),
        dtype=torch.bfloat16,
        max_entries=4,
        offload_after_encode=False,
    )

    cache.encode_batch(["task a", "task b"])
    cache.encode("task c")

    assert encoder.moves == [torch.device("cuda", 0)]
    assert encode_calls == [("task a", "task b"), ("task c",)]
    assert empty_cache_calls == []
