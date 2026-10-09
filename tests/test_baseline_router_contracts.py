"""Offline size-ranking and random baseline contracts through real constructors."""

import copy
import importlib
import json
import random
from types import SimpleNamespace

import pytest
import yaml


@pytest.fixture(autouse=True)
def preserve_global_random_state():
    state = random.getstate()
    try:
        yield
    finally:
        random.setstate(state)


def write_config(tmp_path, candidates, *, seed=19, include_test_queries=True):
    candidates_path = tmp_path / "llms.json"
    candidates_path.write_text(json.dumps(candidates), encoding="utf-8")
    config = {
        "data_path": {"llm_data": str(candidates_path)}, "hparam": {"seed": seed},
        "api_endpoint": "https://default.example.invalid/v1",
    }
    if include_test_queries:
        queries_path = tmp_path / "queries.jsonl"
        queries_path.write_text(
            json.dumps({"query": "first loaded query", "id": 1}) + "\n" +
            json.dumps({"query": "second loaded query", "id": 2}) + "\n", encoding="utf-8",
        )
        config["data_path"]["query_data_test"] = str(queries_path)
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return path


@pytest.fixture(params=["smallest", "largest"])
def size_router(request, tmp_path):
    module = importlib.import_module(f"llmrouter.models.{request.param}_llm.router")
    router_class = getattr(module, f"{request.param.title()}LLM")
    candidates = {
        "large": {
            "size": "70b", "model": "vendor/large", "service": "LargeVendor",
            "api_endpoint": "https://large.example.invalid/v1",
        },
        "small": {"size": "1.5B", "model": "vendor/small", "service": "SmallVendor"},
        # Only sizes ending in B are eligible, as the router documentation states.
        "million": {"size": "1M"}, "trillion": {"size": "1T"},
        "numeric": {"size": 100}, "unsized": {"size": None},
    }
    path = write_config(tmp_path, candidates)
    return SimpleNamespace(
        module=module, router=router_class(str(path)), router_class=router_class,
        config_path=path, expected="small" if request.param == "smallest" else "large",
    )


def capture_provider(monkeypatch, subject, outcomes=None):
    calls = []
    responses = iter(outcomes) if outcomes is not None else None

    def provider(request, **options):
        calls.append((copy.deepcopy(request), options))
        response = next(responses) if responses is not None else {
            "response": "A", "prompt_tokens": 9, "completion_tokens": 2,
        }
        if isinstance(response, Exception):
            raise response
        return response

    monkeypatch.setattr(subject.module, "call_api", provider)
    return calls


@pytest.mark.parametrize("size, expected", [
    ("7B", 7.), (" 13b ", 13.), ("512M", .512), ("2000K", .002),
    ("1.5T", 1500.), ("2.5", 2.5), ("not-a-size", 0.), (None, 0.),
])
def test_size_parser_units_and_invalid_metadata(size_router, size, expected):
    assert size_router.module.parse_size(size) == pytest.approx(expected)


def test_size_rankings_preserve_single_query_metadata(size_router):
    query = {"query": "content does not change model ranking", "id": 17, "context": {"tenant": "example"}}
    original = copy.deepcopy(query)
    result = size_router.router.route_single(query)
    assert result == {**query, "model_name": size_router.expected}
    assert query == original
    assert result is not query


def test_size_batch_preserves_order_and_execution_metadata(size_router, monkeypatch):
    subject = size_router
    calls = capture_provider(monkeypatch, subject)
    batch = [{"query": "first", "id": 3, "ground_truth": "A", "metric": "em"}, {"query": "second", "id": 1}]
    original = copy.deepcopy(batch)
    results = subject.router.route_batch(batch)
    assert batch == original
    assert [result["id"] for result in results] == [3, 1]
    assert [result["query"] for result in results] == ["first", "second"]
    assert [result["model_name"] for result in results] == [subject.expected] * 2
    assert results[0]["task_performance"] == 1.
    assert "task_performance" not in results[1]
    for result in results:
        assert result["success"] is True
        assert result["response"] == "A"
        assert result["prompt_tokens"] == result["input_token"] == 9
        assert result["completion_tokens"] == result["output_token"] == 2
    expected_endpoint = "https://large.example.invalid/v1" if subject.expected == "large" else "https://default.example.invalid/v1"
    for (request, options), query in zip(calls, batch, strict=True):
        assert request == {
            "query": query["query"], "model_name": subject.expected,
            "api_name": f"vendor/{subject.expected}", "api_endpoint": expected_endpoint,
            "service": f"{subject.expected.title()}Vendor",
        }
        assert options == {"max_tokens": 1024, "temperature": .7}


def test_size_provider_errors_are_per_query_and_later_queries_continue(size_router, monkeypatch):
    calls = capture_provider(monkeypatch, size_router, [
        {"error": "rate limited"}, RuntimeError("upstream disconnected"),
        {"response": "recovered", "prompt_tokens": 8, "completion_tokens": 1},
    ])
    results = size_router.router.route_batch(["first", "second", "third"])
    assert len(calls) == 3
    assert [result["success"] for result in results] == [False, False, True]
    assert [result["response"] for result in results] == ["", "", "recovered"]
    assert [result["prompt_tokens"] for result in results] == [0, 0, 8]
    assert [result["output_token"] for result in results] == [0, 0, 1]


def test_size_default_test_queries_single_input_and_empty_batch(size_router, monkeypatch, capsys):
    subject = size_router
    calls = capture_provider(monkeypatch, subject)
    original = copy.deepcopy(subject.router.query_data_test)
    assert [result["id"] for result in subject.router.route_batch()] == [1, 2]
    assert subject.router.query_data_test == original
    assert subject.router.route_batch({"query": "single", "id": 9})[0]["id"] == 9
    assert subject.router.route_batch([]) == []
    subject.router.query_data_test = None
    assert subject.router.route_batch() == []
    assert "No batch provided and no test data" in capsys.readouterr().out
    assert len(calls) == 3


def test_size_unknown_task_falls_back_without_mutating_input(size_router, monkeypatch, capsys):
    calls = capture_provider(monkeypatch, size_router)
    query = {"query": "unchanged", "task_name": "unknown-task", "id": 4}
    result = size_router.router.route_batch([query], task_name="gsm8k")[0]
    assert calls[0][0]["query"] == "unchanged"
    assert "formatted_query" not in result
    assert query == {"query": "unchanged", "task_name": "unknown-task", "id": 4}
    assert "Failed to format query with task 'unknown-task'" in capsys.readouterr().out


def test_size_missing_endpoint_is_rejected_before_provider_call(size_router, monkeypatch):
    subject = size_router
    subject.router.llm_data[subject.expected].pop("api_endpoint", None)
    subject.router.cfg.pop("api_endpoint")
    calls = capture_provider(monkeypatch, subject)
    with pytest.raises(ValueError, match="API endpoint not found"):
        subject.router.route_batch("a query")
    assert calls == []


@pytest.mark.parametrize("candidates, message", [
    ({}, "LLM data not loaded"),
    ({"million": {"size": "512M"}, "numeric": {"size": 7}}, "No models with size ending"),
])
def test_size_missing_or_ineligible_model_pools_are_rejected(size_router, tmp_path, candidates, message):
    config_path = write_config(tmp_path, candidates, include_test_queries=False)
    router = size_router.router_class(str(config_path))
    with pytest.raises(ValueError, match=message):
        router.route_single({"query": "test"})
    with pytest.raises(ValueError, match=message):
        router.route_batch([{"query": "test"}])


@pytest.mark.xfail(strict=True, raises=AssertionError, reason="Known bug: size route_batch forwards a prompt dict as user content instead of splitting system/user strings")
def test_size_task_formatting_reaches_provider_as_system_and_user_strings(size_router, monkeypatch):
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
    query = {"query": "a multiple choice question", "choices": ["first", "second", "third", "fourth"]}
    size_router.router.route_batch([query], task_name="mmlu")
    formatted = generate_task_query("mmlu", query)
    assert messages == [
        {"role": "system", "content": formatted["system"]},
        {"role": "user", "content": formatted["user"]},
    ]


@pytest.mark.parametrize("method", ["route_single", "route_batch"])
@pytest.mark.xfail(strict=True, raises=KeyError, reason="Known bug: size filtering dereferences a missing 'size' key instead of ignoring the ineligible candidate")
def test_size_candidates_without_size_are_ignored(size_router, monkeypatch, method):
    size_router.router.llm_data["missing-size"] = {}
    capture_provider(monkeypatch, size_router)
    query = {"query": "ranking with incomplete metadata"}
    if method == "route_single":
        result = size_router.router.route_single(query)
    else:
        result = size_router.router.route_batch([query])[0]
    assert result["model_name"] == size_router.expected


def test_random_constructor_seed_replays_mixed_single_and_batch_calls(tmp_path):
    from custom_routers.randomrouter.router import RandomRouter

    candidates = {"one": {}, "two": {}, "three": {}}
    config_path = write_config(tmp_path, candidates)
    query = {"query": "single", "id": 7}
    batch = [{"query": f"batch-{index}"} for index in range(12)]
    original_batch = copy.deepcopy(batch)
    router = RandomRouter(str(config_path))
    first = router(query)
    outputs = router(batch)
    reloaded = RandomRouter(str(config_path))
    assert reloaded(query) == first
    assert reloaded(batch) == outputs
    assert batch == original_batch
    assert query == {"query": "single", "id": 7}
    assert [output["query"] for output in outputs] == [row["query"] for row in batch]
    for result in [first, *outputs]:
        assert result["model_name"] in candidates
        assert result["predicted_llm"] == result["predicted_llm_name"] == result["model_name"]
        assert result["method"] == "random"
        assert result["confidence"] == 1.
    assert router.route_batch([]) == []


def test_random_single_candidate_and_missing_query(tmp_path):
    from custom_routers.randomrouter.router import RandomRouter

    router = RandomRouter(str(write_config(tmp_path, {"only": {}}, seed=None)))
    assert router.llm_names == ["only"]
    assert router.route_single({}) == {
        "query": "", "model_name": "only", "predicted_llm": "only",
        "predicted_llm_name": "only", "method": "random", "confidence": 1.,
    }
    assert [output["model_name"] for output in router.route_batch([{"query": "a"}, {"query": "b"}])] == ["only", "only"]


def test_random_empty_pool_is_rejected(tmp_path):
    from custom_routers.randomrouter.router import RandomRouter

    with pytest.raises(ValueError, match="No LLM data found"):
        RandomRouter(str(write_config(tmp_path, {})))


@pytest.mark.parametrize("package, name", [
    ("llmrouter.models.smallest_llm.router", "SmallestLLM"),
    ("llmrouter.models.largest_llm.router", "LargestLLM"),
    ("custom_routers.randomrouter.router", "RandomRouter"),
])
def test_baseline_missing_yaml_is_rejected(tmp_path, package, name):
    router_class = getattr(importlib.import_module(package), name)
    with pytest.raises(FileNotFoundError, match="YAML file not found"):
        router_class(str(tmp_path / "missing.yaml"))
