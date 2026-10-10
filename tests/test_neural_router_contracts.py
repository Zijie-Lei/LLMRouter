"""Exercise existing small CPU MLP/MF models, trainers and checkpoint routing."""

from copy import deepcopy
import importlib
import json

import pytest
from sklearn.linear_model import LogisticRegression
import torch
import torch.nn as nn
import yaml

from llmrouter.models.mfrouter.router import BilinearMF, MFRouter
from llmrouter.models.mfrouter.trainer import MFRouterTrainer, PairwiseDataset
from llmrouter.models.mlprouter.router import MLPClassifierNN, MLPRouter
from llmrouter.models.mlprouter.trainer import MLPTrainer
from llmrouter.utils import load_model, save_model


@pytest.fixture(autouse=True)
def isolated_torch_rng():
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(17)
        yield


@pytest.fixture
def local_data(tmp_path):
    embeddings = torch.tensor([[2.0, 0.0], [1.0, 0.0], [0.0, 2.0], [0.0, 1.0]])
    queries = ["query-0", "query-1", "query-2", "query-3"]
    labels = ["model-a", "model-a", "model-b", "model-b"]
    rows = [
        {"query": query, "embedding_id": index, "model_name": model, "performance": 0.9 if model == label else 0.1}
        for index, (query, label) in enumerate(zip(queries, labels))
        for model in ("model-a", "model-b")
    ]
    routing = tmp_path / "routing.jsonl"
    routing.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    embedding_path = tmp_path / "embeddings.pt"
    torch.save(embeddings, embedding_path)
    models = tmp_path / "models.json"
    models.write_text(json.dumps({
        model: {"model": f"provider-{model}", "api_endpoint": "https://example.test/v1", "service": "test"}
        for model in ("model-a", "model-b")
    }), encoding="utf-8")
    return tmp_path, embeddings, queries, labels, {
        "data_path": {
            "routing_data_train": str(routing), "query_embedding_data": str(embedding_path), "llm_data": str(models),
        },
        "model_path": {
            "ini_model_path": str(tmp_path / "initial.pkl"),
            "save_model_path": str(tmp_path / "checkpoint.pt"),
            "load_model_path": str(tmp_path / "checkpoint.pt"),
        },
        "hparam": {
            "hidden_layer_sizes": [], "activation": "identity", "lr": 0.15,
            "epochs": 4, "batch_size": 4, "alpha": 0.0,
            "latent_dim": 2, "text_dim": 2, "noise_alpha": 0.0,
        },
    }


def make_router(local_data, kind, monkeypatch):
    root, vectors, queries, labels, cfg = local_data
    path = root / "config.yaml"
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    cls = MLPRouter if kind == "mlprouter" else MFRouter
    router = cls(str(path))
    module = importlib.import_module(f"llmrouter.models.{kind}.router")
    lookup = dict(zip(queries, vectors))
    monkeypatch.setattr(module, "get_longformer_embedding", lambda text: lookup[text].clone())
    return router, module


@pytest.mark.parametrize("activation,expected", [
    ("relu", [-2.0, 4.0]),
    ("tanh", [-2.487216, 2.689650]),
    ("logistic", [-0.3429143, 1.4926528]),
    ("identity", [-4.0, 5.0]),
])
def test_mlp_activation_and_prediction_on_known_weights(activation, expected):
    model = MLPClassifierNN(2, [2], 2, activation)
    with torch.no_grad():
        model.layers[0].weight.copy_(torch.eye(2))
        model.layers[0].bias.zero_()
        model.layers[1].weight.copy_(torch.tensor([[2.0, -1.0], [-1.0, 2.0]]))
        model.layers[1].bias.zero_()
    inputs = torch.tensor([[-1.0, 2.0], [2.0, -1.0]])

    torch.testing.assert_close(model(inputs)[0], torch.tensor(expected), atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(model.predict(inputs), torch.tensor([1, 0]))
    assert model.device.type == "cpu"
    assert model.predict(inputs).requires_grad is False


def configure_mf(model):
    with torch.no_grad():
        model.P.weight.copy_(torch.eye(2))
        model.text_proj.weight.copy_(torch.eye(2))
        model.classifier.weight.copy_(torch.tensor([[1.0, 2.0]]))


def test_bilinear_scores_have_known_ranking_and_pairwise_antisymmetry():
    model = BilinearMF(dim=2, num_models=2, text_dim=2)
    configure_mf(model)
    query = model.project_text(torch.tensor([3.0, 1.0]))

    torch.testing.assert_close(model.score_all(query), torch.tensor([3.0, 2.0]))
    margin = model(torch.tensor([0]), torch.tensor([1]), query)
    reverse = model(torch.tensor([1]), torch.tensor([0]), query)
    torch.testing.assert_close(margin, torch.tensor([1.0]))
    torch.testing.assert_close(reverse, -margin)
    assert model.device.type == "cpu"
    margin.sum().backward()
    assert model.P.weight.grad is not None and torch.isfinite(model.P.weight.grad).all()
    assert model.text_proj.weight.grad is not None and torch.isfinite(model.text_proj.weight.grad).all()


@pytest.mark.parametrize("kind", ["mlprouter", "mfrouter"])
def test_real_cpu_training_saves_checkpoint_and_routes_known_queries(local_data, monkeypatch, kind):
    router, _ = make_router(local_data, kind, monkeypatch)
    _, vectors, queries, labels, cfg = local_data
    if kind == "mlprouter":
        trainer = MLPTrainer(router)
        with torch.no_grad():
            for parameter in trainer.model.parameters():
                parameter.zero_()
        criterion = nn.CrossEntropyLoss()
        target = router.label_indices
        loss_before = criterion(trainer.model(vectors), target).item()
        trainer.train()
        loss_after = criterion(trainer.model(vectors), target).item()
        assert loss_after < loss_before
        assert trainer.evaluate() == 1.0
    else:
        trainer = MFRouterTrainer(router)
        configure_mf(trainer.model)
        state_before = deepcopy(trainer.model.state_dict())
        trainer.train()
        assert any(not torch.equal(state_before[key], value) for key, value in trainer.model.state_dict().items())
        assert len(router.pairs) == len(queries)
        for sample in router.pairs:
            expected = labels[sample["embedding_id"]]
            assert router.idx_to_model[sample["winner"]] == expected
            assert sample["winner"] != sample["loser"]

    loaded_state = load_model(cfg["model_path"]["save_model_path"])
    for key, expected in trainer.model.state_dict().items():
        torch.testing.assert_close(loaded_state[key], expected)
    for query, expected in zip(queries, labels):
        request = {"query": query, "request_id": f"request-{query}"}
        snapshot = deepcopy(request)
        result = router.route_single(request)
        assert result["model_name"] == expected
        assert result["request_id"] == request["request_id"]
        assert request == snapshot


def test_mlp_still_loads_legacy_sklearn_checkpoint(local_data, monkeypatch):
    router, _ = make_router(local_data, "mlprouter", monkeypatch)
    _, vectors, queries, labels, cfg = local_data
    checkpoint = str(local_data[0] / "legacy.pkl")
    cfg["model_path"]["load_model_path"] = checkpoint
    router.cfg["model_path"]["load_model_path"] = checkpoint
    model = LogisticRegression().fit(vectors.numpy(), labels)
    assert save_model(model, checkpoint)

    assert [router.route_single({"query": query})["model_name"] for query in queries] == labels


@pytest.mark.parametrize("kind", ["mlprouter", "mfrouter"])
@pytest.mark.parametrize("provider_failure", [False, True])
def test_neural_batch_preserves_order_identity_tokens_and_provider_failure(local_data, monkeypatch, kind, provider_failure):
    router, module = make_router(local_data, kind, monkeypatch)
    _, _, queries, labels, cfg = local_data
    if kind == "mlprouter":
        model = router.mlp_model
        with torch.no_grad():
            model.layers[0].weight.copy_(torch.eye(2))
            model.layers[0].bias.zero_()
    else:
        model = BilinearMF(2, 2, 2)
        configure_mf(model)
    assert save_model(model.state_dict(), cfg["model_path"]["save_model_path"])
    calls = []

    def provider(request, **kwargs):
        calls.append(deepcopy(request))
        if provider_failure and request["query"] == queries[1]:
            return dict(request, error="provider unavailable", response="", prompt_tokens=0, completion_tokens=0)
        return dict(request, response="answer", prompt_tokens=3, completion_tokens=2)

    monkeypatch.setattr(module, "call_api", provider)
    rows = [{"query": query, "request_id": index} for index, query in enumerate(queries)]
    snapshot = deepcopy(rows)
    results = router.route_batch(rows)

    assert rows == snapshot
    assert [result["model_name"] for result in results] == labels
    assert [result["request_id"] for result in results] == list(range(4))
    assert [result["query"] for result in results] == queries
    assert [result["success"] for result in results] == ([True, False, True, True] if provider_failure else [True] * 4)
    for result in results:
        assert result["input_token"] == result["prompt_tokens"] == (3 if result["success"] else 0)
        assert result["output_token"] == result["completion_tokens"] == (2 if result["success"] else 0)
    assert [request["api_name"] for request in calls] == [f"provider-{label}" for label in labels]
    assert all(request["service"] == "test" for request in calls)


@pytest.mark.parametrize("kind", ["mlprouter", "mfrouter"])
def test_neural_routing_rejects_missing_checkpoint(local_data, monkeypatch, kind):
    router, _ = make_router(local_data, kind, monkeypatch)
    with pytest.raises(FileNotFoundError):
        router.route_single({"query": "query-0"})


@pytest.mark.parametrize("as_numpy", [False, True])
def test_pairwise_dataset_uses_precomputed_embeddings_with_correct_ids(as_numpy):
    vectors = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    data = vectors.numpy() if as_numpy else vectors
    dataset = PairwiseDataset([{"winner": 1, "loser": 0, "embedding_id": 1}], data)
    sample = dataset[0]

    assert len(dataset) == 1
    assert sample["winner"].dtype == sample["loser"].dtype == torch.long
    assert sample["winner"].item() == 1 and sample["loser"].item() == 0
    torch.testing.assert_close(sample["embedding"], torch.tensor([3.0, 4.0]))


def test_pairwise_dataset_embedding_fallback_only_uses_encoder_boundary(monkeypatch):
    module = importlib.import_module("llmrouter.models.mfrouter.trainer")
    captured = []

    def encoder(text):
        captured.append(text)
        return torch.tensor([5.0, 6.0])

    monkeypatch.setattr(module, "get_longformer_embedding", encoder)
    sample = PairwiseDataset([{"winner": 0, "loser": 1, "query": "question"}])[0]
    assert captured == ["question"]
    torch.testing.assert_close(sample["embedding"], torch.tensor([5.0, 6.0]))
