"""Routing memory contracts with real persistence and local encoder downloads."""

import base64
from datetime import datetime
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from transformers import AutoModel, AutoTokenizer

from openclaw_router.config import MemoryConfig, OpenClawConfig
from openclaw_router.memory import ContrieverEmbedder, MemoryBank


@pytest.fixture
def local_encoder(monkeypatch):
    downloads = []

    class Tokenizer:
        include_mask = True

        def __init__(self):
            self.calls = []

        def __call__(self, texts, **options):
            self.calls.append((list(texts), options))
            vocabulary = {"east": 1, "north": 2, "west": 3, "café": 4}
            rows = [[vocabulary[word] for word in text.split()][:options["max_length"]] for text in texts]
            width = max(len(row) for row in rows)
            inputs = {"input_ids": torch.tensor([row + [0] * (width - len(row)) for row in rows])}
            if self.include_mask:
                inputs["attention_mask"] = torch.tensor([[1] * len(row) + [0] * (width - len(row)) for row in rows])
            return inputs

    class Encoder(torch.nn.Module):
        def __init__(self, dimensions):
            super().__init__()
            # Padding has a large vector, so ignoring the attention mask corrupts ranking.
            vectors = torch.tensor([[99., 99., 99.], [2., 0., 0.], [0., 3., 0.], [-2., 0., 0.], [0., 0., 5.]])
            self.register_buffer("vectors", vectors[:, :dimensions])

        def forward(self, input_ids, **_kwargs):
            return SimpleNamespace(last_hidden_state=self.vectors[input_ids])

    tokenizer = Tokenizer()

    def download_tokenizer(name):
        downloads.append(("tokenizer", name))
        return tokenizer

    def download_model(name):
        downloads.append(("model", name))
        return Encoder(2 if name == "tiny-2d" else 3)

    monkeypatch.setattr(AutoTokenizer, "from_pretrained", download_tokenizer)
    monkeypatch.setattr(AutoModel, "from_pretrained", download_model)
    return SimpleNamespace(downloads=downloads, tokenizer=tokenizer)


def config_for(tmp_path, **options):
    return MemoryConfig(enabled=True, path=str(tmp_path / "history" / "memory.jsonl"), retriever_model="tiny-3d", **options)


def test_encoder_lazily_loads_once_masks_padding_and_normalizes(local_encoder):
    embedder = ContrieverEmbedder("tiny-3d", device="cpu", max_length=5)
    assert local_encoder.downloads == []
    vectors = embedder.embed(["east", "east north"])
    np.testing.assert_allclose(vectors[0], [1., 0., 0.])
    np.testing.assert_allclose(vectors[1], np.array([2., 3., 0.]) / np.sqrt(13), rtol=1e-6)
    assert vectors.dtype == np.float32
    np.testing.assert_allclose(np.linalg.norm(vectors, axis=1), [1., 1.])
    np.testing.assert_allclose(embedder.embed(["north"]), [[0., 1., 0.]])
    assert local_encoder.downloads == [("tokenizer", "tiny-3d"), ("model", "tiny-3d")]
    assert local_encoder.tokenizer.calls[0] == (
        ["east", "east north"],
        {"padding": True, "truncation": True, "max_length": 5, "return_tensors": "pt"},
    )


def test_encoder_without_attention_mask_uses_first_token(local_encoder):
    local_encoder.tokenizer.include_mask = False
    embedder = ContrieverEmbedder("tiny-3d", device="cpu")
    np.testing.assert_allclose(embedder.embed(["east north"]), [[1., 0., 0.]])


def test_memory_round_trip_preserves_unicode_metadata_and_cosine_order(tmp_path, local_encoder):
    cfg = config_for(tmp_path)
    bank = MemoryBank(cfg)
    bank.add("east", "small", strategy="llm", user="alice")
    bank.add("east north", "medium", strategy="rules", user="bob")
    bank.add("west", "large", strategy="llm", user="bob")
    bank.add("café", "unicode", strategy="llm")
    raw = bank.path.read_text(encoding="utf-8")
    assert "café" in raw
    records = [json.loads(line) for line in raw.splitlines()]
    assert len(records) == 4
    assert all(datetime.fromisoformat(record["ts"]).utcoffset() is not None for record in records)
    assert all(len(base64.b64decode(record["emb_b64"])) == 3 * 4 for record in records)

    reloaded = MemoryBank(cfg)
    results = reloaded.retrieve("east", top_k=20)
    assert [result["model"] for result in results] == ["small", "medium", "unicode", "large"]
    assert [result["score"] for result in results] == pytest.approx([1., 2 / np.sqrt(13), 0., -1.])
    assert results[0] == {
        "query": "east", "model": "small", "score": pytest.approx(1.),
        "ts": records[0]["ts"], "strategy": "llm", "user": "alice",
    }
    assert reloaded.retrieve("café", top_k=1)[0]["query"] == "café"


def test_stored_query_is_trimmed_and_bounded_before_encoding(tmp_path, local_encoder):
    bank = MemoryBank(config_for(tmp_path, max_query_chars=4))
    bank.add("  east north  ", " model ", strategy=" rules ", user=" alice ")
    result = bank.retrieve("east")[0]
    assert (result["query"], result["model"], result["strategy"], result["user"]) == ("east", "model", "rules", "alice")
    assert local_encoder.tokenizer.calls[0][0] == ["east"]
    assert json.loads(bank.path.read_text(encoding="utf-8"))["query"] == "east"


@pytest.mark.parametrize("query, model", [("", "model"), (" \n ", "model"), (None, "model"), ("east", " "), ("east", None)])
def test_empty_inputs_do_not_create_files_or_download_models(tmp_path, local_encoder, query, model):
    bank = MemoryBank(config_for(tmp_path))
    bank.add(query, model)
    assert not bank.path.exists()
    assert bank.retrieve("east") == []
    assert bank.retrieve(" ") == []
    assert local_encoder.downloads == []


@pytest.mark.parametrize("per_user, user, strategy, expected", [
    (True, "alice", None, ["small"]),
    (True, "bob", "llm", ["large"]),
    (True, "missing", None, []),
    (False, "alice", None, ["small", "medium", "large"]),
    (True, None, "rules", ["medium"]),
    (False, None, "missing", []),
])
def test_user_and_strategy_filters_apply_before_ranking(tmp_path, local_encoder, per_user, user, strategy, expected):
    bank = MemoryBank(config_for(tmp_path, per_user=per_user))
    bank.add("east", "small", strategy="llm", user="alice")
    bank.add("east north", "medium", strategy="rules", user="bob")
    bank.add("west", "large", strategy="llm", user="bob")
    assert [item["model"] for item in bank.retrieve("east", top_k=10, user=user, strategy_filter=strategy)] == expected


def test_top_k_limits_and_disabled_retrieval_skip_encoding(tmp_path, local_encoder):
    bank = MemoryBank(config_for(tmp_path, top_k=1))
    bank.add("east", "small")
    bank.add("north", "large")
    assert [item["model"] for item in bank.retrieve("east")] == ["small"]
    before = len(local_encoder.tokenizer.calls)
    assert bank.retrieve("east", top_k=0) == []
    assert bank.retrieve("east", top_k=-1) == []
    assert bank.retrieve("") == []
    assert len(local_encoder.tokenizer.calls) == before


def test_reload_skips_corrupt_records_and_incompatible_embedding_dimensions(tmp_path, local_encoder):
    cfg = config_for(tmp_path)
    bank = MemoryBank(cfg)
    bank.add("east", "small")
    bank.add("north", "large")
    valid_lines = bank.path.read_text(encoding="utf-8").splitlines()
    invalid_records = [
        "", "not JSON", json.dumps({"query": "east"}),
        json.dumps({"query": "west", "model": "invalid", "emb_b64": "not-base64!"}),
        json.dumps({"query": "west", "model": "old-encoder", "emb_b64": base64.b64encode(np.ones(2, dtype=np.float32).tobytes()).decode("ascii")}),
    ]
    bank.path.write_text("\n".join([valid_lines[0], *invalid_records, valid_lines[1]]) + "\n", encoding="utf-8")
    assert [item["model"] for item in MemoryBank(cfg).retrieve("east")] == ["small", "large"]


def test_retriever_dimension_change_does_not_append_incompatible_records(tmp_path, local_encoder):
    bank = MemoryBank(config_for(tmp_path))
    bank.add("east", "small")
    previous = bank.path.read_bytes()
    cfg = config_for(tmp_path)
    cfg.retriever_model = "tiny-2d"
    upgraded = MemoryBank(cfg)
    assert upgraded.retrieve("east") == []
    upgraded.add("north", "new-model")
    assert upgraded.path.read_bytes() == previous


def test_relative_and_environment_paths_resolve_against_config_directory(tmp_path, monkeypatch, local_encoder):
    monkeypatch.setenv("LLMROUTER_TEST_MEMORY_SUBDIR", "history")
    cfg = MemoryConfig(path="$LLMROUTER_TEST_MEMORY_SUBDIR/store.jsonl", retriever_model="tiny-3d")
    bank = MemoryBank(cfg, config_dir=str(tmp_path))
    bank.add("east", "small")
    assert bank.path == tmp_path / "history" / "store.jsonl"
    assert bank.path.is_file()


def test_default_path_does_not_create_home_storage_until_first_add(tmp_path, monkeypatch, local_encoder):
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    bank = MemoryBank(MemoryConfig())
    assert bank.path == tmp_path / ".llmrouter" / "openclaw_memory.jsonl"
    assert not bank.path.exists()
    assert local_encoder.downloads == []


def test_yaml_memory_config_controls_real_bank_storage_and_retrieval(tmp_path, local_encoder):
    path = tmp_path / "config.yaml"
    path.write_text(
        "memory:\n  enabled: 'true'\n  path: history/memory.jsonl\n  top_k: 1\n"
        "  retriever_model: tiny-3d\n  device: cpu\n  max_query_chars: 4\n  per_user: 'true'\n",
        encoding="utf-8",
    )
    config = OpenClawConfig.from_yaml(str(path))
    bank = MemoryBank(config.memory, config.config_dir)
    bank.add("east north", "small", user="alice")
    bank.add("west", "large", user="bob")
    assert bank.path == tmp_path / "history" / "memory.jsonl"
    assert [item["query"] for item in bank.retrieve("east", user="alice")] == ["east"]
    assert bank.retrieve("east", user="missing") == []


@pytest.mark.parametrize("invalid_record", ["[]", "17"])
@pytest.mark.xfail(strict=True, raises=AttributeError, reason="Known bug: valid non-object JSONL records crash memory reload instead of being skipped")
def test_non_object_jsonl_records_do_not_discard_valid_history(tmp_path, local_encoder, invalid_record):
    cfg = config_for(tmp_path)
    bank = MemoryBank(cfg)
    bank.add("east", "small")
    bank.add("north", "large")
    lines = bank.path.read_text(encoding="utf-8").splitlines()
    bank.path.write_text("\n".join([lines[0], invalid_record, lines[1]]) + "\n", encoding="utf-8")
    assert [item["model"] for item in MemoryBank(cfg).retrieve("east")] == ["small", "large"]
