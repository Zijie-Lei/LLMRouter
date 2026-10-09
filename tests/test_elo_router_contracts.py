"""Real Elo battle construction, MLE fitting, lazy reload, and batch execution."""

from copy import deepcopy
import importlib
import json
import math
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch
import yaml

from llmrouter.models.elorouter.router import EloRouter
from llmrouter.models.elorouter.trainer import EloRouterTrainer
from llmrouter.utils import load_model, save_model


@pytest.fixture(autouse=True)
def isolated_torch_rng():
    with torch.random.fork_rng(devices=[]):
        yield


@pytest.fixture
def elo_data(tmp_path):
    training = [
        {"query": f"q-{i}", "model_name": model, "performance": int(model == winner)}
        for i in range(12)
        for winner in ["strong" if i < 9 else "weak"]
        for model in ("weak", "strong")
    ]
    routing = tmp_path / "routing.jsonl"
    routing.write_text("".join(json.dumps(row) + "\n" for row in training), encoding="utf-8")
    models = tmp_path / "models.json"
    models.write_text(json.dumps({
        "weak": {"model": "api/weak"},
        "strong": {"model": "api/strong", "service": "StrongProvider", "api_endpoint": "https://strong.example.test/v1"},
    }), encoding="utf-8")
    queries = tmp_path / "queries.jsonl"
    queries.write_text(' {"query": "first", "id": "first"}\n{"query": "second", "id": "second"}\n', encoding="utf-8")
    model = tmp_path / "model.pkl"
    config = {
        "data_path": {"routing_data_train": str(routing), "llm_data": str(models), "query_data_test": str(queries)},
        "model_path": {"save_model_path": str(model), "load_model_path": str(model)},
        "api_endpoint": "https://default.example.test/v1",
    }
    path = tmp_path / "elo.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return SimpleNamespace(path=path, config=config, model=model, routing=routing)


@pytest.fixture
def elo(elo_data):
    router = EloRouter(str(elo_data.path))
    trainer = EloRouterTrainer(router, device="cpu")
    trainer.train()
    return SimpleNamespace(router=router, trainer=trainer, data=elo_data, module=importlib.import_module("llmrouter.models.elorouter.router"))


def test_elo_battles_are_symmetric_and_ratings_match_observed_odds(elo):
    battles = elo.trainer._build_battles()
    assert len(battles) == 24
    for index in range(0, len(battles), 2):
        forward, reverse = battles.iloc[index], battles.iloc[index + 1]
        assert forward["model_a"] == reverse["model_b"]
        assert forward["model_b"] == reverse["model_a"]
        assert (forward["winner"], reverse["winner"]) == ("model_a", "model_b")
    ratings = load_model(str(elo.data.model))
    assert isinstance(ratings, pd.Series)
    assert ratings.index.tolist() == ["strong", "weak"]
    assert ratings["strong"] - ratings["weak"] == pytest.approx(400 * math.log10(3), abs=0.2)
    assert ratings.mean() == pytest.approx(1000)
    assert np.isfinite(ratings).all()


@pytest.mark.parametrize("checkpoint_type", ["series", "dict"])
def test_elo_lazy_checkpoint_load_is_cached_and_preserves_query_identity(elo, checkpoint_type):
    if checkpoint_type == "dict":
        save_model({"weak": 900.0, "strong": 1100.0}, str(elo.data.model))
    fresh = EloRouter(str(elo.data.path))
    assert fresh.model_loaded is False
    assert fresh.elo_scores is None
    assert fresh.model_names == ["strong", "weak"]
    source = {"query": "first", "id": "request-1", "metadata": {"owner": "caller"}}

    result = fresh.route_single(source)

    assert result == {**source, "model_name": "strong"}
    assert result is not source
    assert "model_name" not in source
    assert fresh.model_loaded is True
    elo.data.model.unlink()
    assert fresh.route_single({"query": "second"}) == {"query": "second", "model_name": "strong"}


def test_elo_batch_preserves_original_rows_and_forwards_provider_options(elo, monkeypatch):
    calls = []

    def provider(request, **options):
        calls.append((deepcopy(request), options))
        return dict(request, response=request["query"], prompt_tokens=5, completion_tokens=2)

    monkeypatch.setattr(elo.module, "call_api", provider)
    batch = [{"query": "first", "id": "1", "answer": "first", "metric": "em"}, {"query": "second", "id": "2"}]
    before = deepcopy(batch)
    results = elo.router.route_batch(batch)

    assert batch == before
    assert [result["id"] for result in results] == ["1", "2"]
    assert [result["query"] for result in results] == ["first", "second"]
    assert [result["response"] for result in results] == ["first", "second"]
    assert results[0]["task_performance"] == 1.0
    assert all(result["model_name"] == "strong" and result["success"] is True for result in results)
    assert all((result["input_token"], result["output_token"]) == (5, 2) for result in results)
    assert all((result["prompt_tokens"], result["completion_tokens"]) == (5, 2) for result in results)
    assert calls == [
        ({"query": query, "model_name": "strong", "api_name": "api/strong", "api_endpoint": "https://strong.example.test/v1", "service": "StrongProvider"}, {"max_tokens": 1024, "temperature": 0.7})
        for query in ("first", "second")
    ]


def test_elo_provider_failures_do_not_abort_the_batch(elo, monkeypatch):
    outcomes = iter([{"error": "unavailable"}, RuntimeError("disconnected"), {"response": "answer", "prompt_tokens": 3, "completion_tokens": 1}])

    def provider(request, **options):
        outcome = next(outcomes)
        if isinstance(outcome, Exception):
            raise outcome
        return dict(request, **outcome)

    monkeypatch.setattr(elo.module, "call_api", provider)
    results = elo.router.route_batch(["first", "second", "third"])
    assert [result["query"] for result in results] == ["first", "second", "third"]
    assert [result["success"] for result in results] == [False, False, True]
    assert [result["input_token"] for result in results] == [0, 0, 3]
    assert [result["output_token"] for result in results] == [0, 0, 1]


def test_elo_loaded_test_data_single_input_and_empty_batches(elo, monkeypatch):
    monkeypatch.setattr(elo.module, "call_api", lambda request, **options: dict(request, response="answer"))
    before = deepcopy(elo.router.query_data_test)
    assert [result["id"] for result in elo.router.route_batch()] == ["first", "second"]
    assert elo.router.query_data_test == before
    assert elo.router.route_batch("single")[0]["query"] == "single"
    assert elo.router.route_batch([]) == []
    elo.router.query_data_test = None
    assert elo.router.route_batch() == []


def test_elo_missing_checkpoint_is_lazy_error(elo_data):
    router = EloRouter(str(elo_data.path))
    assert router.model_loaded is False
    with pytest.raises(FileNotFoundError, match="File does not exist"):
        router.route_single({"query": "first"})


def test_elo_rejects_empty_rating_checkpoint(elo):
    save_model({}, str(elo.data.model))
    with pytest.raises(ValueError, match="empty"):
        elo.router.route_single({"query": "first"})


def test_elo_trainer_rejects_training_with_no_competing_candidates(elo_data):
    elo_data.routing.write_text('{"query": "first", "model_name": "only", "performance": 1}\n', encoding="utf-8")
    router = EloRouter(str(elo_data.path))
    with pytest.raises(ValueError, match="Empty battle table"):
        EloRouterTrainer(router, device="cpu").train()


def test_elo_missing_candidate_endpoint_uses_configured_default(elo, monkeypatch):
    elo.router.llm_data["strong"].pop("api_endpoint")
    calls = []

    def provider(request, **options):
        calls.append(request)
        return dict(request, response="answer")

    monkeypatch.setattr(elo.module, "call_api", provider)
    assert elo.router.route_batch("first")[0]["success"] is True
    assert calls[0]["api_endpoint"] == "https://default.example.test/v1"
    elo.router.cfg.pop("api_endpoint")
    with pytest.raises(ValueError, match="API endpoint not found for model 'strong'"):
        elo.router.route_batch("second")
    assert len(calls) == 1
