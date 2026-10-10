"""Cover actual local file IO, router persistence and embedding pooling."""

import json
import pickle
from types import SimpleNamespace

import pandas as pd
import pytest
import torch
import torch.nn as nn
import yaml

from llmrouter.models.meta_router import MetaRouter
from llmrouter.utils import data_loader, embeddings, model_loader
from llmrouter.utils.dataframe_utils import clean_df
from llmrouter.utils.router_helpers import format_api_request_with_task
from llmrouter.utils.tensor_utils import to_tensor


@pytest.fixture(autouse=True)
def isolated_torch_rng():
    with torch.random.fork_rng(devices=[]):
        yield


class LocalRouter(MetaRouter):
    def route_single(self, query):
        return self.model(query)

    def route_batch(self, batch):
        return self.model(batch)


@pytest.mark.parametrize("suffix", [".pt", ".pkl", ".PT"])
def test_model_state_round_trip_creates_parent_and_preserves_tensor_values(tmp_path, suffix):
    path = tmp_path / "nested" / f"model{suffix}"
    state = {"weight": torch.tensor([[1.0, -2.0], [3.0, 4.0]])}

    assert model_loader.save_model(state, str(path)) is True
    actual = model_loader.load_model(str(path))

    torch.testing.assert_close(actual["weight"], state["weight"])
    assert actual["weight"].device.type == "cpu"


@pytest.mark.parametrize("parent_exists", [False, True])
def test_model_loader_reports_missing_checkpoint(tmp_path, parent_exists):
    path = tmp_path / "nested" / "missing.pt"
    if parent_exists:
        path.parent.mkdir()
    with pytest.raises(FileNotFoundError):
        model_loader.load_model(str(path))


def test_model_io_rejects_unsupported_extensions_and_corrupt_files(tmp_path):
    unsupported = tmp_path / "model.bin"
    assert model_loader.save_model({"weight": 1}, str(unsupported)) is False
    unsupported.write_text("not a model", encoding="utf-8")
    with pytest.raises(ValueError, match="Unsupported file extension"):
        model_loader.load_model(str(unsupported))
    corrupt = tmp_path / "model.pkl"
    corrupt.write_bytes(b"not a pickle")
    with pytest.raises(pickle.UnpicklingError):
        model_loader.load_model(str(corrupt))


def test_meta_router_forward_and_saved_state_restore_real_predictions(tmp_path):
    router = LocalRouter(nn.Linear(2, 1), resources={"name": "local"})
    inputs = torch.tensor([[1.0, -1.0], [2.0, 3.0]])
    with torch.no_grad():
        router.model.weight.copy_(torch.tensor([[2.0, 3.0]]))
        router.model.bias.fill_(1.0)
    expected = torch.tensor([[0.0], [14.0]])
    torch.testing.assert_close(router(inputs), expected)
    assert router.compute_metrics(expected, inputs) == {}
    checkpoint = tmp_path / "router.pt"
    router.save_router(str(checkpoint))
    with torch.no_grad():
        router.model.weight.zero_()
        router.model.bias.zero_()
    router.load_router(str(checkpoint))
    torch.testing.assert_close(router.route_single(inputs), expected)


def test_meta_router_loads_real_yaml_and_metric_weights(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump({"metric": {"weights": {"accuracy": 0.8, "cost": 0.2}}}))
    router = LocalRouter(nn.Identity(), str(path))

    assert router.metric_weights == [0.8, 0.2]
    assert router.llm_data is None
    assert router.query_data_train is None
    assert router.resources is None
    with pytest.raises(FileNotFoundError, match="YAML file not found"):
        LocalRouter(nn.Identity(), str(tmp_path / "missing.yaml"))


def test_jsonl_conversion_and_csv_reload_preserve_rows_and_unicode(tmp_path):
    path = tmp_path / "data.jsonl"
    rows = [{"query": "你好", "score": 0.9}, {"query": "question", "score": 0.2}]
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")

    assert data_loader.load_jsonl(str(path)) == rows
    actual = data_loader.jsonl_to_csv(str(path))
    pd.testing.assert_frame_equal(actual, pd.DataFrame(rows))
    pd.testing.assert_frame_equal(data_loader.load_csv(str(path.with_suffix('.csv'))), actual)
    explicit = tmp_path / "explicit.csv"
    pd.testing.assert_frame_equal(data_loader.jsonl_to_csv(str(path), str(explicit)), actual)
    assert explicit.is_file()


@pytest.mark.parametrize("loader", [data_loader.load_jsonl, data_loader.jsonl_to_csv, data_loader.load_csv, data_loader.load_pt])
def test_optional_data_loaders_return_none_for_missing_files(tmp_path, loader):
    assert loader(str(tmp_path / "missing")) is None


def test_invalid_jsonl_and_tensor_files_do_not_produce_partial_data(tmp_path):
    path = tmp_path / "invalid.jsonl"
    path.write_text('{"query": "first"}\nnot-json\n', encoding="utf-8")
    assert data_loader.load_jsonl(str(path)) is None
    assert data_loader.jsonl_to_csv(str(path)) is None
    path = tmp_path / "invalid.pt"
    path.write_bytes(b"not a tensor")
    assert data_loader.load_pt(str(path)) is None
    tensor = torch.tensor([[1.0, 2.0]])
    torch.save(tensor, path)
    torch.testing.assert_close(data_loader.load_pt(str(path)), tensor)


@pytest.mark.parametrize("text", [
    "[[1, 2], [3, 4]]",
    "tensor([[1, 2], [3, 4]], device='cuda:0', dtype=torch.float64)",
])
def test_tensor_text_conversion_strips_device_metadata_without_executing_code(text):
    tensor = to_tensor(text)
    torch.testing.assert_close(tensor, torch.tensor([[1.0, 2.0], [3.0, 4.0]]))
    assert tensor.device.type == "cpu"
    with pytest.raises((ValueError, SyntaxError)):
        to_tensor("__import__('os').getcwd()")


def test_dataframe_cleanup_preserves_input_and_normalizes_schema():
    original = pd.DataFrame([{
        "query": "question", "query_embedding": [1.0], "formatted_query": "prompt",
        "prompt_tokens": 3, "completion_tokens": 2, "gt": "answer", "model_name": "model-a",
    }])
    snapshot = original.copy(deep=True)
    actual = clean_df(original)

    pd.testing.assert_frame_equal(original, snapshot)
    assert actual.to_dict("records") == [{
        "query": "question", "input_tokens": 3, "output_tokens": 2,
        "ground_truth": "answer", "model_name": "model-a", "user_id": None, "fig_id": None,
    }]


def test_task_request_splits_real_system_and_user_prompts_and_falls_back():
    plain = format_api_request_with_task("Question", None, "https://example.test/v1", "alias", "provider-model")
    assert plain == {
        "query": "Question", "system_prompt": None, "api_endpoint": "https://example.test/v1",
        "model_name": "alias", "api_name": "provider-model",
    }
    formatted = format_api_request_with_task("2 + 2?", "gsm8k", "https://example.test/v1", "alias", "provider-model")
    assert isinstance(formatted["query"], str) and "2 + 2?" in formatted["query"]
    assert isinstance(formatted["system_prompt"], str) and formatted["system_prompt"]
    fallback = format_api_request_with_task("Question", "unknown-task", "https://example.test/v1", "alias", "provider-model")
    assert fallback == plain


class TokenBatch(dict):
    def to(self, device):
        return TokenBatch({key: value.to(device) for key, value in self.items()})


def test_embedding_mean_pool_excludes_padding_and_preserves_single_batch_shapes(monkeypatch):
    calls = []

    def tokenize(texts, **kwargs):
        calls.append((texts, kwargs))
        return TokenBatch(
            input_ids=torch.tensor([[1, 2, 0]] * len(texts)),
            attention_mask=torch.tensor([[1, 1, 0]] * len(texts)),
        )

    def encode(**inputs):
        assert not torch.is_grad_enabled()
        return SimpleNamespace(last_hidden_state=torch.tensor([
            [[2.0, 4.0], [4.0, 8.0], [1000.0, 1000.0]],
        ] * inputs["input_ids"].shape[0]))

    monkeypatch.setattr(embeddings, "_tokenizer", tokenize)
    monkeypatch.setattr(embeddings, "_model", encode)
    monkeypatch.setattr(embeddings, "_device", torch.device("cpu"))
    single = embeddings.get_longformer_embedding("first")
    batch = embeddings.get_longformer_embedding(["first", "second"])

    torch.testing.assert_close(single, torch.tensor([3.0, 6.0]))
    torch.testing.assert_close(batch, torch.tensor([[3.0, 6.0], [3.0, 6.0]]))
    assert single.device.type == batch.device.type == "cpu"
    assert calls[0][0] == ["first"]
    assert calls[1][1]["truncation"] is True
    assert calls[1][1]["max_length"] == 4096


def test_embedding_device_override_and_parallel_failure_contract(monkeypatch):
    monkeypatch.setattr(embeddings, "_device", None)
    monkeypatch.setenv("LLMROUTER_EMBEDDING_DEVICE", "cpu")
    assert embeddings._get_device() == torch.device("cpu")
    monkeypatch.setattr(embeddings, "get_longformer_embedding", lambda text: torch.tensor([1.0, 2.0]))
    identifier, tensor, success = embeddings.parallel_embedding_task((7, "question"))
    assert identifier == 7 and success is True
    torch.testing.assert_close(tensor, torch.tensor([1.0, 2.0]))

    def fail(text):
        raise RuntimeError("encoder unavailable")

    monkeypatch.setattr(embeddings, "get_longformer_embedding", fail)
    assert embeddings.parallel_embedding_task((8, "question")) == (8, None, False)
