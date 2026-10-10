"""Tiny-data contracts for existing Hybrid, Threshold, and TaskAware policies."""

from copy import deepcopy
import importlib
import io
import json
import random
from types import SimpleNamespace
import urllib.error

import numpy as np
import pytest
import torch
import yaml

from custom_routers.taskawarerouter.router import TaskAwareRouter
from custom_routers.thresholdrouter.router import ThresholdRouter
from custom_routers.thresholdrouter.trainer import ThresholdRouterTrainer
from llmrouter.models.hybrid_llm.router import HybridLLMRouter
from llmrouter.models.hybrid_llm.trainer import HybridLLMTrainer
from llmrouter.utils import load_model


@pytest.fixture(autouse=True)
def isolated_random_state():
    python_state, numpy_state = random.getstate(), np.random.get_state()
    try:
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(13)
            yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)


def write_jsonl(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def write_yaml(path, config):
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return str(path)


@pytest.fixture
def hybrid_data(tmp_path):
    vectors = torch.tensor([[-2.0], [-1.0], [1.0], [2.0]])
    embeddings = tmp_path / "embeddings.pt"
    torch.save(vectors, embeddings)
    models = tmp_path / "models.json"
    models.write_text(json.dumps({
        "large": {"size": "12B", "model": "api/large", "service": "Large", "api_endpoint": "https://large.example.test/v1"},
        "ignored": {"size": "500M"},
        "small": {"size": "3B", "model": "api/small", "service": "Small"},
        "middle": {"size": "7b"},
    }), encoding="utf-8")
    rows = [
        {"query": f"q-{i}", "embedding_id": i, "model_name": name, "performance": performance}
        for i in (3, 1, 0, 2)
        for name, performance in [("small", float(i >= 2)), ("large", float(i < 2))]
    ]
    routing = tmp_path / "routing.jsonl"
    write_jsonl(routing, rows)
    queries = tmp_path / "queries.jsonl"
    write_jsonl(queries, [{"query": "low", "id": "first"}, {"query": "high", "id": "second"}])
    config = {
        "data_path": {"llm_data": str(models), "routing_data_train": str(routing), "query_embedding_data": str(embeddings), "query_data_test": str(queries)},
        "model_path": {"ini_model_path": str(tmp_path / "initial.pkl"), "save_model_path": str(tmp_path / "model.pkl"), "load_model_path": str(tmp_path / "model.pkl")},
        "hparam": {"hidden_layer_sizes": [], "solver": "lbfgs", "alpha": 0.0, "random_state": 13, "max_iter": 300, "tol": 1e-8},
        "api_endpoint": "https://default.example.test/v1",
    }
    return SimpleNamespace(root=tmp_path, config=config, rows=rows, vectors=vectors, models=models, routing=routing)


@pytest.fixture
def hybrid(hybrid_data, monkeypatch):
    data = hybrid_data
    path = write_yaml(data.root / "config.yaml", data.config)
    router = HybridLLMRouter(path)
    HybridLLMTrainer(router, device="cpu").train()
    module = importlib.import_module("llmrouter.models.hybrid_llm.router")
    lookup = {"low": torch.tensor([-4.0]), "high": torch.tensor([4.0]), "middle": torch.tensor([0.0])}
    monkeypatch.setattr(module, "get_longformer_embedding", lambda text: lookup[text].clone())
    return SimpleNamespace(router=router, module=module, path=path, data=data)


@pytest.mark.parametrize("mode,expected", [
    ("deterministic", [0.0, 0.0, 1.0, 1.0]),
    ("probabilistic", [1 / (1 + np.exp(2)), 1 / (1 + np.exp(2)), 1 / (1 + np.exp(-2)), 1 / (1 + np.exp(-2))]),
    ("transformed", [0.0, 0.0, 1.0, 1.0]),
])
def test_hybrid_initialization_aligns_embeddings_and_quality_labels(hybrid_data, mode, expected):
    config = deepcopy(hybrid_data.config)
    config.update(router_mode=mode, router_tau=0.5)
    router = HybridLLMRouter(write_yaml(hybrid_data.root / "config.yaml", config))

    assert (router.small_model_name, router.large_model_name) == ("small", "large")
    np.testing.assert_array_equal(router.query_embedding_list, hybrid_data.vectors.numpy())
    assert router.router_label_list == pytest.approx(expected)


def test_hybrid_training_pickle_reload_and_clamped_decisions(hybrid):
    saved = load_model(hybrid.data.config["model_path"]["save_model_path"])
    assert saved.predict(hybrid.data.vectors.numpy()).tolist() == pytest.approx([-0.1, 0.2, 0.8, 1.1], abs=1e-5)
    fresh = HybridLLMRouter(hybrid.path)
    for text, expected_model, expected_score in [("low", "large", 0.0), ("high", "small", 1.0)]:
        source = {"query": text, "id": text, "metadata": {"source": "caller"}}
        result = fresh.route_single(source)
        assert result == {**source, "model_name": expected_model, "router_score": expected_score}
        assert result is not source
        assert "model_name" not in source


def test_hybrid_threshold_includes_exact_tie_and_rejects_next_higher_threshold(hybrid):
    score = float(load_model(hybrid.data.config["model_path"]["save_model_path"]).predict([[0.0]])[0])
    hybrid.router.router_threshold = score
    assert hybrid.router.route_single({"query": "middle"})["model_name"] == "small"
    hybrid.router.router_threshold = np.nextafter(score, 1.0)
    assert hybrid.router.route_single({"query": "middle"})["model_name"] == "large"


def test_hybrid_batch_preserves_identity_provider_options_and_tokens(hybrid, monkeypatch):
    calls = []

    def provider(request, **options):
        calls.append((deepcopy(request), options))
        return dict(request, response="answer", prompt_tokens=7, completion_tokens=2)

    monkeypatch.setattr(hybrid.module, "call_api", provider)
    rows = [{"query": "high", "id": "first", "ground_truth": "answer", "metric": "em"}, {"query": "low", "id": "second"}]
    before = deepcopy(rows)
    results = hybrid.router.route_batch(rows)

    assert rows == before
    assert [result["id"] for result in results] == ["first", "second"]
    assert [result["query"] for result in results] == ["high", "low"]
    assert [result["model_name"] for result in results] == ["small", "large"]
    assert results[0]["task_performance"] == 1.0
    for result in results:
        assert (result["prompt_tokens"], result["completion_tokens"]) == (7, 2)
        assert (result["input_token"], result["output_token"]) == (7, 2)
        assert result["success"] is True
    assert calls == [
        ({"query": "high", "model_name": "small", "api_name": "api/small", "service": "Small", "api_endpoint": "https://default.example.test/v1"}, {"max_tokens": 1024, "temperature": 0.7}),
        ({"query": "low", "model_name": "large", "api_name": "api/large", "service": "Large", "api_endpoint": "https://large.example.test/v1"}, {"max_tokens": 1024, "temperature": 0.7}),
    ]


def test_hybrid_provider_failures_continue_later_queries(hybrid, monkeypatch):
    outcomes = iter([{"error": "unavailable"}, RuntimeError("disconnected"), {"response": "recovered", "prompt_tokens": 3, "completion_tokens": 1}])

    def provider(request, **options):
        outcome = next(outcomes)
        if isinstance(outcome, Exception):
            raise outcome
        return dict(request, **outcome)

    monkeypatch.setattr(hybrid.module, "call_api", provider)
    results = hybrid.router.route_batch(["low", "high", "low"])
    assert [result["query"] for result in results] == ["low", "high", "low"]
    assert [result["success"] for result in results] == [False, False, True]
    assert [result["input_token"] for result in results] == [0, 0, 3]
    assert [result["output_token"] for result in results] == [0, 0, 1]


def test_hybrid_default_test_data_single_query_and_empty_batches(hybrid, monkeypatch):
    monkeypatch.setattr(hybrid.module, "call_api", lambda request, **options: dict(request, response="answer"))
    before = deepcopy(hybrid.router.query_data_test)
    assert [row["id"] for row in hybrid.router.route_batch()] == ["first", "second"]
    assert hybrid.router.query_data_test == before
    assert hybrid.router.route_batch("high")[0]["model_name"] == "small"
    assert hybrid.router.route_batch([]) == []
    hybrid.router.query_data_test = None
    assert hybrid.router.route_batch() == []


def test_hybrid_missing_endpoint_raises_before_provider(hybrid, monkeypatch):
    calls = []
    monkeypatch.setattr(hybrid.module, "call_api", lambda *args, **options: calls.append(args))
    hybrid.router.cfg.pop("api_endpoint")
    with pytest.raises(ValueError, match="API endpoint not found for model 'small'"):
        hybrid.router.route_batch("high")
    assert calls == []


@pytest.mark.parametrize("kind", ["missing-models", "one-model", "invalid-mode", "no-pair"])
def test_hybrid_rejects_unusable_initialization_data(hybrid_data, kind):
    config = deepcopy(hybrid_data.config)
    if kind == "missing-models":
        config["data_path"].pop("llm_data")
        expected, message = ValueError, "llm_data missing"
    elif kind == "one-model":
        hybrid_data.models.write_text(json.dumps({"small": {"size": "3B"}}), encoding="utf-8")
        expected, message = ValueError, "Need at least 2 models"
    elif kind == "invalid-mode":
        config["router_mode"] = "invalid"
        expected, message = AssertionError, "Router mode"
    else:
        write_jsonl(hybrid_data.routing, [{"query": "q", "embedding_id": 0, "model_name": "middle", "performance": 1}])
        expected, message = ValueError, "No routing data found"
    with pytest.raises(expected, match=message):
        HybridLLMRouter(write_yaml(hybrid_data.root / "invalid.yaml", config))


@pytest.fixture
def threshold(tmp_path):
    models = tmp_path / "models.json"
    models.write_text(json.dumps({"small": {"size": "3B"}, "large": {"size": "12B"}}), encoding="utf-8")
    config = {"data_path": {"llm_data": str(models)}, "hparam": {"embedding_dim": 2, "hidden_dim": 4, "threshold": 0.5}}
    path = tmp_path / "threshold.yaml"
    router = ThresholdRouter(write_yaml(path, config))
    with torch.no_grad():
        for parameter in router.model.parameters():
            parameter.zero_()
        router.model.network[0].weight[0, 0] = 1.0
        router.model.network[3].weight[0, 0] = 1.0
        router.model.network[5].weight[0, 0] = 1.0
        router.model.network[5].bias[0] = -1.0
    return SimpleNamespace(router=router, config=config, path=path, models=models)


@pytest.mark.parametrize("tensor_input", [False, True])
@pytest.mark.parametrize("feature,model,score", [(0.0, "small", 1 / (1 + np.exp(1))), (1.0, "large", 0.5), (2.0, "large", 1 / (1 + np.exp(-1)))])
def test_threshold_real_network_boundary_and_model_aliases(threshold, tensor_input, feature, model, score):
    embedding = torch.tensor([feature, 0.0]) if tensor_input else [feature, 0.0]
    source = {"query": "question", "embedding": embedding}
    threshold.router.model.train()
    result = threshold.router.route_single(source)
    assert result["model_name"] == result["predicted_llm"] == result["predicted_llm_name"] == model
    assert result["difficulty_score"] == pytest.approx(score)
    assert result["threshold"] == 0.5
    assert result["query"] == "question"
    assert result["method"] == "threshold"
    assert threshold.router.model.training is False
    assert "model_name" not in source


def test_threshold_checkpoint_reload_and_batch_order(threshold, tmp_path):
    checkpoint = tmp_path / "threshold.pt"
    threshold.router.save_router(str(checkpoint))
    threshold.config["model_path"] = {"load_model_path": str(checkpoint)}
    fresh = ThresholdRouter(write_yaml(threshold.path, threshold.config))
    batch = [{"query": "first", "embedding": [0.0, 0.0]}, {"query": "second", "embedding": [1.0, 0.0]}]
    before = deepcopy(batch)
    results = fresh.route_batch(batch)
    assert batch == before
    assert [result["query"] for result in results] == ["first", "second"]
    assert [result["model_name"] for result in results] == ["small", "large"]
    for key, value in threshold.router.state_dict().items():
        torch.testing.assert_close(fresh.state_dict()[key], value)
    assert fresh.route_batch([]) == []


@pytest.mark.parametrize("loaded_embeddings", [False, True])
def test_threshold_requires_explicit_embedding(threshold, loaded_embeddings):
    if loaded_embeddings:
        threshold.router.query_embeddings = {"question": [1.0, 0.0]}
    with pytest.raises(ValueError, match="embedding"):
        threshold.router.route_single({"query": "question"})


@pytest.mark.parametrize("models,error", [({}, "No LLM data"), ({"one": {}}, "At least 2 LLMs")])
def test_threshold_rejects_missing_candidates(threshold, models, error):
    threshold.models.write_text(json.dumps(models), encoding="utf-8")
    with pytest.raises(ValueError, match=error):
        ThresholdRouter(str(threshold.path))


@pytest.mark.xfail(strict=True, raises=TypeError, reason="Threshold trainer iterates DataFrame columns as record dictionaries")
def test_threshold_trainer_consumes_real_loaded_training_records(threshold, tmp_path):
    rows = [{"embedding": [0.0, 0.0], "best_llm": "small"}, {"embedding": [2.0, 0.0], "best_llm": "large"}]
    path = tmp_path / "training.jsonl"
    write_jsonl(path, rows)
    threshold.config["data_path"]["routing_data_train"] = str(path)
    router = ThresholdRouter(write_yaml(threshold.path, threshold.config))
    features, labels = ThresholdRouterTrainer(router, threshold.config).prepare_training_data()
    torch.testing.assert_close(features, torch.tensor([[0.0, 0.0], [2.0, 0.0]]))
    torch.testing.assert_close(labels, torch.tensor([[0.0], [1.0]]))


@pytest.mark.xfail(strict=True, raises=AttributeError, reason="Threshold train() reads self.config, which its constructor never stores")
def test_threshold_one_epoch_training_saves_checkpoint(threshold, tmp_path):
    checkpoint = tmp_path / "trained_threshold.pt"
    threshold.config["hparam"].update(train_epoch=1, batch_size=2, learning_rate=0.05)
    threshold.config["model_path"] = {"save_model_path": str(checkpoint)}
    # Use the trainer's intended record sequence to isolate its missing-config
    # defect from the separately tested DataLoader/DataFrame incompatibility.
    threshold.router.routing_data_train = [
        {"embedding": [0.0, 0.0], "best_llm": "small"},
        {"embedding": [2.0, 0.0], "best_llm": "large"},
    ]
    trainer = ThresholdRouterTrainer(threshold.router, threshold.config)
    trainer.train()
    assert checkpoint.is_file()
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    for key, value in threshold.router.state_dict().items():
        torch.testing.assert_close(saved[key], value)


@pytest.fixture
def taskaware(tmp_path, monkeypatch):
    names = ["fallback", "judge", "mistral-7b-instruct-v0.3", "llama3-70b-instruct", "llama-3.1-8b-instruct", "mixtral-8x22b-instruct-v0.1", "qwen2.5-7b-instruct", "llama-3.3-nemotron-super-49b-v1"]
    models = tmp_path / "models.json"
    models.write_text(json.dumps({name: {"api_endpoint": "https://judge.example.test/v1"} for name in names}), encoding="utf-8")
    config = {"data_path": {"llm_data": str(models)}, "router": {"judge_model": "judge", "judge_api_model": "judge-api", "judge_timeout": 2}}
    router = TaskAwareRouter(write_yaml(tmp_path / "taskaware.yaml", config))
    module = importlib.import_module("custom_routers.taskawarerouter.router")
    state = SimpleNamespace(router=router, module=module, calls=[], judgment={"task": "coding", "complexity": "simple"}, failure=None)
    monkeypatch.setenv("API_KEYS", "fake-judge-key")

    def urlopen(request, timeout):
        state.calls.append((request, timeout))
        if state.failure is not None:
            raise state.failure
        content = json.dumps(state.judgment) if not isinstance(state.judgment, str) else state.judgment
        return io.BytesIO(json.dumps({"choices": [{"message": {"content": content}}]}).encode())

    monkeypatch.setattr(module.urllib.request, "urlopen", urlopen)
    return state


@pytest.mark.parametrize("task,simple,complex_model", [
    ("coding", "mistral-7b-instruct-v0.3", "llama3-70b-instruct"),
    ("design", "llama-3.1-8b-instruct", "mixtral-8x22b-instruct-v0.1"),
    ("planning", "qwen2.5-7b-instruct", "llama-3.3-nemotron-super-49b-v1"),
    ("research", "qwen2.5-7b-instruct", "mixtral-8x22b-instruct-v0.1"),
    ("language", "llama-3.1-8b-instruct", "mixtral-8x22b-instruct-v0.1"),
])
@pytest.mark.parametrize("complexity", ["simple", "complex"])
def test_taskaware_real_judge_request_and_task_decisions(taskaware, task, simple, complex_model, complexity):
    taskaware.judgment = {"task": task, "complexity": complexity}
    result = taskaware.router.route_single({"query": "Classify this query"})
    expected = simple if complexity == "simple" else complex_model
    assert result == {"query": "Classify this query", "task": task, "complexity": complexity, "model_name": expected, "predicted_llm": expected}
    request, timeout = taskaware.calls[0]
    assert request.get_method() == "POST"
    assert request.full_url == "https://judge.example.test/v1/chat/completions"
    assert request.get_header("Authorization") == "Bearer fake-judge-key"
    assert request.get_header("Content-type") == "application/json"
    payload = json.loads(request.data)
    assert payload["model"] == "judge-api"
    assert "Query: Classify this query" in payload["messages"][0]["content"]
    assert (payload["max_tokens"], payload["temperature"], timeout) == (50, 0, 2)


@pytest.mark.parametrize("failure", ["http", "url", "json"])
@pytest.mark.parametrize("query,complexity,model", [("write a function", "simple", "mistral-7b-instruct-v0.3"), ("build an ENTERPRISE pipeline", "complex", "llama3-70b-instruct")])
def test_taskaware_supported_judge_failures_use_keyword_fallback(taskaware, failure, query, complexity, model):
    if failure == "http":
        taskaware.failure = urllib.error.HTTPError("https://judge.example.test", 503, "unavailable", None, None)
    elif failure == "url":
        taskaware.failure = urllib.error.URLError("unreachable")
    else:
        taskaware.judgment = "invalid JSON"
    result = taskaware.router.route_single({"query": query})
    assert (result["task"], result["complexity"], result["model_name"]) == ("coding", complexity, model)


def test_taskaware_missing_or_unknown_task_choices_use_available_fallback(taskaware):
    taskaware.judgment = {"task": "unknown", "complexity": "complex"}
    assert taskaware.router.route_single({"query": "first"})["model_name"] == "fallback"
    taskaware.judgment = {"task": "coding", "complexity": "simple"}
    taskaware.router.llm_data.pop("mistral-7b-instruct-v0.3")
    assert taskaware.router.route_single({"query": "second"})["model_name"] == "fallback"


def test_taskaware_batch_order_and_inputs_are_preserved(taskaware):
    rows = [{"query": "first"}, {"query": "second"}]
    before = deepcopy(rows)
    results = taskaware.router.route_batch(rows)
    assert rows == before
    assert [row["query"] for row in results] == ["first", "second"]
    assert len(taskaware.calls) == 2
    assert taskaware.router.route_batch([]) == []


@pytest.mark.xfail(strict=True, raises=AttributeError, reason="TaskAware fails instead of using safe defaults for a valid JSON judgment of the wrong type")
def test_taskaware_wrong_judgment_shape_uses_fallback(taskaware):
    taskaware.judgment = []
    result = taskaware.router.route_single({"query": "write a function"})
    assert result["task"] == "coding"


@pytest.mark.xfail(strict=True, raises=TimeoutError, reason="TaskAware does not catch direct urllib timeout errors for its documented safe fallback")
def test_taskaware_direct_timeout_uses_fallback(taskaware):
    taskaware.failure = TimeoutError("judge timed out")
    result = taskaware.router.route_single({"query": "write a function"})
    assert result["model_name"] == "mistral-7b-instruct-v0.3"
