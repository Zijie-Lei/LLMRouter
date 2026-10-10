"""Offline KNN/SVM contracts using real data loading, sklearn fitting, and pickle IO."""

import copy
import importlib
import json
import pickle
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import yaml


@pytest.fixture(params=["knn", "svm"])
def classical_router(request, tmp_path, monkeypatch):
    name = request.param
    router_module = importlib.import_module(f"llmrouter.models.{name}router.router")
    trainer_module = importlib.import_module(f"llmrouter.models.{name}router.trainer")
    router_class = getattr(router_module, f"{name.upper()}Router")
    trainer_class = getattr(trainer_module, f"{name.upper()}RouterTrainer")
    embeddings = torch.tensor([[-2., -1.], [-2., 1.], [2., -1.], [2., 1.]])
    embedding_path = tmp_path / "embeddings.pt"
    torch.save(embeddings, embedding_path)

    # File order differs from query order, and each query has competing models.
    # This catches training on every row, the first row, or mismatched embedding IDs.
    training = []
    for index in (3, 1, 2, 0):
        winner = "small" if index < 2 else "large"
        for model in ("large", "small"):
            training.append({
                "query": f"training-{index}", "embedding_id": index,
                "model_name": model, "performance": .9 if model == winner else .1,
            })
    training_path = tmp_path / "routing.jsonl"
    training_path.write_text("".join(json.dumps(row) + "\n" for row in training), encoding="utf-8")
    test_queries = [{"query": "new-warm", "id": "first"}, {"query": "new-cold", "id": "second"}]
    test_path = tmp_path / "queries.jsonl"
    test_path.write_text("".join(json.dumps(row) + "\n" for row in test_queries), encoding="utf-8")
    candidates = {
        "small": {"model": "provider/small-v2", "service": "SmallVendor"},
        "large": {
            "model": "provider/large-v3", "service": "LargeVendor",
            "api_endpoint": "https://large.example.invalid/v1",
        },
    }
    candidate_path = tmp_path / "llms.json"
    candidate_path.write_text(json.dumps(candidates), encoding="utf-8")
    model_path = tmp_path / "saved" / f"{name}.pkl"
    config = {
        "data_path": {
            "routing_data_train": str(training_path), "query_embedding_data": str(embedding_path),
            "query_data_test": str(test_path), "llm_data": str(candidate_path),
        },
        "model_path": {
            "ini_model_path": str(tmp_path / "initial.pkl"),
            "save_model_path": str(model_path), "load_model_path": str(model_path),
        },
        "hparam": {"n_neighbors": 1, "weights": "distance", "n_jobs": 1} if name == "knn"
        else {"kernel": "linear", "C": 1., "random_state": 7},
        "api_endpoint": "https://default.example.invalid/v1",
        "metric": {"weights": {"performance": 1., "cost": 0.}},
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    local_embeddings = {
        **{f"training-{i}": embedding for i, embedding in enumerate(embeddings)},
        "new-cold": torch.tensor([-3., 0.]), "new-warm": torch.tensor([3., 0.]),
        "wrong-dimensions": torch.tensor([1., 2., 3.]),
    }
    monkeypatch.setattr(router_module, "get_longformer_embedding", lambda text: local_embeddings[text].clone())
    router = router_class(str(config_path))
    return SimpleNamespace(
        name=name, module=router_module, router=router, trainer_class=trainer_class,
        router_class=router_class, config_path=config_path, model_path=model_path,
        embeddings=embeddings,
    )


@pytest.fixture
def trained_router(classical_router):
    classical_router.trainer_class(classical_router.router, device="cpu").train()
    return classical_router


def capture_provider(monkeypatch, subject, responses=None):
    calls = []
    outcomes = iter(responses) if responses is not None else None

    def provider(request, **options):
        calls.append((copy.deepcopy(request), options))
        outcome = next(outcomes) if outcomes is not None else {
            "response": "answer", "prompt_tokens": 7, "completion_tokens": 3,
        }
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    monkeypatch.setattr(subject.module, "call_api", provider)
    return calls


def test_training_selects_the_best_candidate_and_round_trips(trained_router):
    subject = trained_router
    router = subject.router
    assert router.model_name_list == ["small", "small", "large", "large"]
    np.testing.assert_array_equal(router.query_embedding_list, subject.embeddings.numpy())
    assert subject.model_path.is_file()
    with subject.model_path.open("rb") as handle:
        saved = pickle.load(handle)
    assert saved.predict(subject.embeddings.numpy()).tolist() == ["small", "small", "large", "large"]

    # A fresh router must use the fitted on-disk estimator, not the trainer's instance.
    reloaded = subject.router_class(str(subject.config_path))
    for query, expected in (("new-cold", "small"), ("new-warm", "large")):
        original = {"query": query, "request_id": query, "metadata": {"source": "caller"}}
        output = reloaded.route_single(original)
        assert output == {**original, "model_name": expected}
        assert output is not original
        assert "model_name" not in original


def test_trainer_uses_existing_initial_estimator_hyperparameters(classical_router):
    subject = classical_router
    router = subject.router
    estimator = getattr(router, f"{subject.name}_model")
    changed_parameters = {"n_neighbors": 3} if subject.name == "knn" else {"C": 4.}
    estimator.set_params(**changed_parameters)
    initial_path = subject.config_path.parent / "initial.pkl"
    with initial_path.open("wb") as handle:
        pickle.dump(estimator, handle)
    # The configured checkpoint overrides a different estimator in the new router.
    fresh = subject.router_class(str(subject.config_path))
    subject.trainer_class(fresh, device="cpu").train()
    with subject.model_path.open("rb") as handle:
        saved = pickle.load(handle)
    for key, value in changed_parameters.items():
        assert saved.get_params()[key] == value
    assert saved.predict(subject.embeddings.numpy()).tolist() == ["small", "small", "large", "large"]


def test_batch_preserves_order_metadata_and_provider_identity(trained_router, monkeypatch):
    subject = trained_router
    calls = capture_provider(monkeypatch, subject, [
        {"response": "A", "prompt_tokens": 11, "completion_tokens": 2},
        {"response": "B", "prompt_tokens": 13, "completion_tokens": 4},
    ])
    batch = [
        {"query": "new-warm", "id": "first", "answer": "A", "metric": "em"},
        {"query": "new-cold", "id": "second", "gt": "B", "metric": "em"},
    ]
    original = copy.deepcopy(batch)
    results = subject.router.route_batch(batch)
    assert batch == original
    assert [result["id"] for result in results] == ["first", "second"]
    assert [result["model_name"] for result in results] == ["large", "small"]
    assert [result["response"] for result in results] == ["A", "B"]
    assert [result["task_performance"] for result in results] == [1., 1.]
    for result, source, tokens in zip(results, batch, [(11, 2), (13, 4)], strict=True):
        assert all(result[key] == value for key, value in source.items())
        assert (result["prompt_tokens"], result["completion_tokens"]) == tokens
        assert (result["input_token"], result["output_token"]) == tokens
        assert result["success"] is True
    assert calls == [
        ({"query": "new-warm", "model_name": "large", "api_name": "provider/large-v3",
          "api_endpoint": "https://large.example.invalid/v1", "service": "LargeVendor"},
         {"max_tokens": 1024, "temperature": .7}),
        ({"query": "new-cold", "model_name": "small", "api_name": "provider/small-v2",
          "api_endpoint": "https://default.example.invalid/v1", "service": "SmallVendor"},
         {"max_tokens": 1024, "temperature": .7}),
    ]


def test_provider_failures_do_not_abort_later_queries(trained_router, monkeypatch):
    subject = trained_router
    calls = capture_provider(monkeypatch, subject, [
        {"error": "provider unavailable"}, RuntimeError("provider disconnected"),
        {"response": "recovered", "prompt_tokens": 5, "completion_tokens": 1},
    ])
    results = subject.router.route_batch(["new-cold", "new-warm", "new-cold"])
    assert len(calls) == 3
    assert [result["success"] for result in results] == [False, False, True]
    assert [result["response"] for result in results] == ["", "", "recovered"]
    assert [result["input_token"] for result in results] == [0, 0, 5]
    assert [result["output_token"] for result in results] == [0, 0, 1]
    assert [result["query"] for result in results] == ["new-cold", "new-warm", "new-cold"]


def test_loaded_test_queries_single_text_and_empty_batches(trained_router, monkeypatch, capsys):
    subject = trained_router
    calls = capture_provider(monkeypatch, subject)
    original = copy.deepcopy(subject.router.query_data_test)
    results = subject.router.route_batch()
    assert [result["model_name"] for result in results] == ["large", "small"]
    assert subject.router.query_data_test == original
    assert subject.router.route_batch("new-cold")[0]["model_name"] == "small"
    assert subject.router.route_batch([]) == []
    subject.router.query_data_test = None
    assert subject.router.route_batch() == []
    assert "No batch provided and no test data" in capsys.readouterr().out
    assert len(calls) == 3


@pytest.mark.parametrize("task, choices", [("unknown-task", None), ("commonsense_qa", {})])
def test_invalid_task_formatting_falls_back_to_original_query(trained_router, monkeypatch, capsys, task, choices):
    subject = trained_router
    calls = capture_provider(monkeypatch, subject)
    row = {"query": "new-cold", "task_name": task, "choices": choices}
    result = subject.router.route_batch([row], task_name="gsm8k")[0]
    assert calls[0][0]["query"] == "new-cold"
    assert result["query"] == "new-cold"
    assert "formatted_query" not in result
    assert result["success"] is True
    assert f"Failed to format query with task '{task}'" in capsys.readouterr().out


def test_endpoint_and_api_model_fall_back_when_candidate_metadata_is_absent(trained_router, monkeypatch):
    subject = trained_router
    subject.router.llm_data = {}
    calls = capture_provider(monkeypatch, subject)
    result = subject.router.route_batch("new-cold")[0]
    assert result["model_name"] == "small"
    assert calls[0][0] == {
        "query": "new-cold", "model_name": "small", "api_name": "small",
        "api_endpoint": "https://default.example.invalid/v1",
    }


def test_missing_endpoint_is_rejected_before_calling_provider(trained_router, monkeypatch):
    subject = trained_router
    subject.router.cfg.pop("api_endpoint")
    calls = capture_provider(monkeypatch, subject)
    with pytest.raises(ValueError, match="API endpoint not found for model 'small'"):
        subject.router.route_batch("new-cold")
    assert calls == []


def test_required_query_and_embedding_dimensions_are_validated(trained_router):
    with pytest.raises(KeyError, match="query"):
        trained_router.router.route_single({"id": "missing-query"})
    with pytest.raises(ValueError, match="features"):
        trained_router.router.route_single({"query": "wrong-dimensions"})


def test_missing_checkpoint_is_rejected(classical_router):
    with pytest.raises(FileNotFoundError, match="does not exist"):
        classical_router.router.route_single({"query": "new-cold"})


def test_missing_yaml_is_rejected(classical_router):
    with pytest.raises(FileNotFoundError, match="YAML file not found"):
        classical_router.router_class(str(classical_router.config_path.parent / "missing.yaml"))


@pytest.mark.xfail(strict=True, raises=AssertionError, reason="Known bug: classical route_batch forwards a prompt dict as user content instead of splitting system/user strings")
def test_formatted_task_reaches_provider_as_system_and_user_messages(trained_router, monkeypatch):
    from llmrouter.utils import api_calling
    from llmrouter.utils.prompting import generate_task_query

    messages = []

    def completion(**kwargs):
        messages.extend(kwargs["messages"])
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="A"))],
            usage=SimpleNamespace(prompt_tokens=4, completion_tokens=2, total_tokens=6),
        )

    monkeypatch.setenv("API_KEYS", "offline-test-key")
    monkeypatch.setattr(api_calling, "completion", completion)
    monkeypatch.setattr(api_calling, "_api_key_counters", {})
    query = {"query": "new-cold", "choices": ["first", "second", "third", "fourth"]}
    trained_router.router.route_batch([query], task_name="mmlu")
    formatted = generate_task_query("mmlu", query)
    assert messages == [
        {"role": "system", "content": formatted["system"]},
        {"role": "user", "content": formatted["user"]},
    ]
