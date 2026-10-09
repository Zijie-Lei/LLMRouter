"""Keep routing rows aligned with real persisted embeddings and JSONL output."""

import json
from copy import deepcopy

import numpy as np
import pandas as pd
import pytest
import torch

from llmrouter.utils import data_processing, embeddings


class ReverseCompletionPool:
    """Control completion order while running the real embedding task function."""

    def __init__(self, workers):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def imap_unordered(self, function, tasks):
        return iter(reversed([function(task) for task in tasks]))


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_final_processing_deduplicates_embeddings_without_reordering_model_rows(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    original = pd.DataFrame([
        {"task_name": "task", "query": "second", "gt": "b", "metric": "em", "query_embedding": np.array([2.0, 4.0]), "model_name": "model-a", "prompt_tokens": 3, "completion_tokens": 2},
        {"task_name": "task", "query": "first", "gt": "a", "metric": "em", "query_embedding": "[1.0, 3.0]", "model_name": "model-b", "prompt_tokens": 4, "completion_tokens": 1},
        {"task_name": "task", "query": "second", "gt": "b", "metric": "em", "query_embedding": np.array([2.0, 4.0]), "model_name": "model-c", "prompt_tokens": 5, "completion_tokens": 3},
        {"task_name": "task", "query": "third", "gt": "c", "metric": "em", "query_embedding": np.array([5.0, 7.0]), "model_name": "model-d", "prompt_tokens": 1, "completion_tokens": 4},
        {"task_name": "task", "query": "first", "gt": "a", "metric": "em", "query_embedding": "[1.0, 3.0]", "model_name": "model-e", "prompt_tokens": 2, "completion_tokens": 5},
    ])
    snapshot = pd.DataFrame(deepcopy(original.to_dict("records")))

    train, test, vectors = data_processing.process_final_data(original)

    pd.testing.assert_frame_equal(original, snapshot)
    assert [len(train), len(test)] == [4, 1]
    assert list(pd.concat([train, test])["model_name"]) == list(original["model_name"])
    combined = pd.concat([train, test])
    for query, expected in [("first", [1.0, 3.0]), ("second", [2.0, 4.0]), ("third", [5.0, 7.0])]:
        ids = set(combined.loc[combined["query"] == query, "embedding_id"])
        assert len(ids) == 1
        torch.testing.assert_close(vectors[ids.pop()], torch.tensor(expected))
    assert len(vectors) == 3
    assert "query_embedding" not in train.columns and "gt" not in train.columns
    assert list(combined["input_tokens"]) == [3, 4, 5, 1, 2]
    assert list(combined["output_tokens"]) == [2, 1, 3, 4, 5]
    actual = torch.load(tmp_path / "query_embeddings.pt", weights_only=True)
    for key in vectors:
        torch.testing.assert_close(actual[key], vectors[key])
    assert [row["model_name"] for row in read_jsonl(tmp_path / "default_routing_train_data.jsonl")] == ["model-a", "model-b", "model-c", "model-d"]
    assert [row["model_name"] for row in read_jsonl(tmp_path / "default_routing_test_data.jsonl")] == ["model-e"]


@pytest.mark.parametrize("representation", ["tensor", "numpy", "list", "text"])
def test_unified_processing_uses_one_id_across_splits_and_preserves_failed_embedding_rows(tmp_path, monkeypatch, representation):
    observed = []

    def encode(text):
        observed.append(text)
        if text == "cannot embed":
            raise RuntimeError("local encoder failure")
        values = [1.0, 3.0] if text == "shared" else [2.0, 4.0]
        if representation == "tensor":
            return torch.tensor(values, dtype=torch.float64)
        if representation == "numpy":
            return np.array(values, dtype=np.float64)
        if representation == "text":
            return str(values)
        return values

    monkeypatch.setattr(embeddings, "get_longformer_embedding", encode)
    # A real thread pool often happens to finish these tiny tasks in input
    # order. Force the inverse schedule so removing sorting breaks alignment.
    monkeypatch.setattr(data_processing, "ThreadPool", ReverseCompletionPool)
    queries = [
        {"task_name": "task", "query": "shared", "ground_truth": ["a", "alias"], "metric": "em"},
        {"task_name": "task", "query": "cannot embed", "ground_truth": "b", "metric": "em"},
        {"task_name": "task", "query": "new", "ground_truth": "c", "metric": "em"},
    ]
    train = pd.DataFrame([
        {**queries[0], "model_name": "a", "choices": {"A": "one", "B": "two"}, "prompt_tokens": 3, "completion_tokens": 2},
        {**queries[1], "model_name": "b", "choices": None, "prompt_tokens": 4, "completion_tokens": 1},
    ])
    test = pd.DataFrame([
        {**queries[2], "model_name": "c", "choices": '{"A": "three"}', "prompt_tokens": 5, "completion_tokens": 6},
        {**queries[0], "model_name": "d", "choices": "{'A': 'one'}", "prompt_tokens": 7, "completion_tokens": 8},
    ]).rename(columns={"ground_truth": "gt"})
    train_snapshot = pd.DataFrame(deepcopy(train.to_dict("records")))
    test_snapshot = pd.DataFrame(deepcopy(test.to_dict("records")))
    output = tmp_path / "nested"
    vectors, train_result, test_result = data_processing.process_unified_embeddings_and_routing(
        train, test, queries[:2], [queries[0], queries[2]],
        str(output / "vectors.pt"), str(output / "train.jsonl"), str(output / "test.jsonl"),
    )

    pd.testing.assert_frame_equal(train, train_snapshot)
    pd.testing.assert_frame_equal(test, test_snapshot)
    assert sorted(observed) == ["cannot embed", "new", "shared"]
    assert set(vectors) == {0, 1}
    torch.testing.assert_close(vectors[0], torch.tensor([1.0, 3.0]))
    torch.testing.assert_close(vectors[1], torch.tensor([2.0, 4.0]))
    assert train_result.loc[0, "embedding_id"] == test_result.loc[1, "embedding_id"] == 0
    assert pd.isna(train_result.loc[1, "embedding_id"])
    assert test_result.loc[0, "embedding_id"] == 1
    train_rows, test_rows = read_jsonl(output / "train.jsonl"), read_jsonl(output / "test.jsonl")
    assert [row["model_name"] for row in train_rows + test_rows] == ["a", "b", "c", "d"]
    assert train_rows[0]["embedding_id"] == 0
    assert [row["embedding_id"] for row in test_rows] == [1, 0]
    assert train_rows[0]["ground_truth"] == ["a", "alias"]
    assert train_rows[0]["choices"] == "{'A': 'one', 'B': 'two'}"
    assert test_rows[0]["choices"] == "{'A': 'three'}"
    assert test_rows[1]["choices"] == "{'A': 'one'}"
    assert all(row["user_id"] is None and row["fig_id"] is None for row in train_rows + test_rows)
    assert [row["input_tokens"] for row in train_rows + test_rows] == [3, 4, 5, 7]
    saved = torch.load(output / "vectors.pt", weights_only=True)
    for key in vectors:
        torch.testing.assert_close(saved[key], vectors[key])


def test_unified_query_identity_includes_task_answers_and_metric(tmp_path, monkeypatch):
    monkeypatch.setattr(embeddings, "get_longformer_embedding", lambda text: [1.0, 2.0])
    queries = [
        {"task_name": "first", "query": "same text", "ground_truth": "a", "metric": "em"},
        {"task_name": "second", "query": "same text", "ground_truth": "a", "metric": "em"},
        {"task_name": "first", "query": "same text", "ground_truth": "b", "metric": "em"},
        {"task_name": "first", "query": "same text", "ground_truth": "a", "metric": "f1"},
    ]
    frame = pd.DataFrame([{**query, "model_name": "m"} for query in queries])
    vectors, train, test = data_processing.process_unified_embeddings_and_routing(
        frame.iloc[:2], frame.iloc[2:], queries[:2], queries[2:],
        str(tmp_path / "vectors.pt"), str(tmp_path / "train.jsonl"), str(tmp_path / "test.jsonl"),
    )
    assert set(vectors) == {0, 1, 2, 3}
    assert list(train["embedding_id"]) == [0, 1]
    assert list(test["embedding_id"]) == [2, 3]


def test_unified_processing_keeps_unmatched_routing_rows_without_fabricating_embeddings(tmp_path, monkeypatch):
    monkeypatch.setattr(embeddings, "get_longformer_embedding", lambda text: [1.0])
    query = {"task_name": "task", "query": "known", "ground_truth": "answer", "metric": "em"}
    frame = pd.DataFrame([{**query, "query": "missing"}])
    vectors, train, test = data_processing.process_unified_embeddings_and_routing(
        frame, frame.iloc[:0], [query], [],
        str(tmp_path / "vectors.pt"), str(tmp_path / "train.jsonl"), str(tmp_path / "test.jsonl"),
    )
    assert set(vectors) == {0}
    assert pd.isna(train.loc[0, "embedding_id"])
    assert pd.isna(read_jsonl(tmp_path / "train.jsonl")[0]["embedding_id"])
    assert test.empty and read_jsonl(tmp_path / "test.jsonl") == []


@pytest.mark.parametrize("failure_mode", ["encoder_failure", "unmatched_query"])
@pytest.mark.xfail(strict=True, raises=ValueError, reason="Existing defect: missing embedding IDs serialize as NaN, which is not valid JSON, instead of null")
def test_missing_embedding_ids_are_written_as_valid_json_null(tmp_path, monkeypatch, failure_mode):
    def encode(text):
        if failure_mode == "encoder_failure":
            raise RuntimeError("encoder unavailable")
        return [1.0]

    monkeypatch.setattr(embeddings, "get_longformer_embedding", encode)
    query = {"task_name": "task", "query": "known", "ground_truth": "answer", "metric": "em"}
    frame = pd.DataFrame([{**query, "query": "unknown" if failure_mode == "unmatched_query" else "known"}])
    data_processing.process_unified_embeddings_and_routing(
        frame, frame.iloc[:0], [query], [],
        str(tmp_path / "vectors.pt"), str(tmp_path / "train.jsonl"), str(tmp_path / "test.jsonl"),
    )

    def reject_non_json_constant(value):
        raise ValueError(f"Invalid JSON constant: {value}")

    record = json.loads((tmp_path / "train.jsonl").read_text(), parse_constant=reject_non_json_constant)
    assert record["embedding_id"] is None


@pytest.mark.parametrize("force_reverse_completion", [False, True])
def test_parallel_embedding_results_preserve_input_order_and_failure_identity(monkeypatch, capsys, force_reverse_completion):
    def encode(text):
        if text == "bad":
            raise ValueError("invalid query")
        return torch.tensor([float(len(text))])

    monkeypatch.setattr(embeddings, "get_longformer_embedding", encode)
    if force_reverse_completion:
        monkeypatch.setattr(data_processing, "ThreadPool", ReverseCompletionPool)
    result = data_processing.generate_embeddings_for_data([{"query": "longer"}, {"query": "bad"}, {"query": "x"}])
    assert [r[0] for r in result] == [0, 1, 2]
    torch.testing.assert_close(result[0][1], torch.tensor([6.0]))
    assert result[1] == (1, None, False)
    torch.testing.assert_close(result[2][1], torch.tensor([1.0]))
    assert [r[2] for r in result] == [True, False, True]
    assert "Success: 2, Fail: 1" in capsys.readouterr().out
    assert data_processing.generate_embeddings_for_data([]) == []
