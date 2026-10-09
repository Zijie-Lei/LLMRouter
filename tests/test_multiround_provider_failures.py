"""Exercise real multi-round pipelines with local encoders and upstream fakes."""

import importlib
import json
import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch
import torch.nn as nn


ROUTER_NAMES = ["llmmultiroundrouter", "knnmultiroundrouter"]
FAILURE_STAGES = [
    ("llmmultiroundrouter", "decomposition"),
    ("llmmultiroundrouter", "execution"),
    ("llmmultiroundrouter", "aggregation"),
    ("knnmultiroundrouter", "execution"),
    ("knnmultiroundrouter", "aggregation"),
]


def build_pipeline(router_name, monkeypatch, failure_stage=None):
    module = importlib.import_module(f"llmrouter.models.{router_name}.router")
    cls = module.LLMMultiRoundRouter if router_name == "llmmultiroundrouter" else module.KNNMultiRoundRouter
    router = cls.__new__(cls)
    nn.Module.__init__(router)
    router.cfg = {}
    router.use_local_llm = False
    router.base_model = "base"
    router.api_endpoint = "https://example.test/v1"
    router.llm_data = {
        "base": {"model": "base", "api_endpoint": router.api_endpoint},
        "worker": {"model": "worker-api", "api_endpoint": router.api_endpoint},
    }
    router.DECOMP_ROUTE_PROMPT = "decomposition:{query}"
    router.DECOMP_PROMPT = "decomposition:{query}"
    router.AGENT_PROMPT = "execution:{query}"
    router.DECOMP_COT_PROMPT = "aggregation:{query}\n{info}"
    if router_name == "knnmultiroundrouter":
        router.knn_model_loaded = True
        router.knn_model = SimpleNamespace(predict=lambda embeddings: ["worker"])
        monkeypatch.setattr(module, "get_longformer_embedding", lambda query: torch.zeros(3))

    state = SimpleNamespace(mode="mapping", calls=[], decomposition=None)

    def call_api(request, **kwargs):
        stage, _, query = request["query"].splitlines()[0].partition(":")
        state.calls.append((stage, query))
        if query == "provider-error" and stage == failure_stage:
            if state.mode == "raised":
                raise RuntimeError("provider unavailable")
            if state.mode == "missing":
                return dict(request, prompt_tokens=0, completion_tokens=0)
            if state.mode == "none":
                return dict(request, response=None, prompt_tokens=0, completion_tokens=0)
            return dict(
                request, response="upstream failed", error="provider unavailable",
                prompt_tokens=0, completion_tokens=0,
            )
        if stage == "decomposition":
            return dict(request, response=state.decomposition or f"{query}: worker")
        if stage == "execution":
            return dict(request, response="worker answer", prompt_tokens=2, completion_tokens=1)
        return dict(request, response="final answer")

    monkeypatch.setattr(module, "call_api", call_api)
    return SimpleNamespace(router=router, state=state, name=router_name, stage=failure_stage)


@pytest.fixture(params=FAILURE_STAGES)
def pipeline(request, monkeypatch):
    router_name, stage = request.param
    return build_pipeline(router_name, monkeypatch, failure_stage=stage)


@pytest.mark.parametrize("mode", ["mapping", "raised", "missing", "none"])
def test_pipeline_preserves_failure_status_and_stage(pipeline, mode):
    pipeline.state.mode = mode

    result = pipeline.router.route_single({"query": "provider-error", "request_id": "request-1"})

    assert result["success"] is False
    assert result["query"] == "provider-error"
    assert result["request_id"] == "request-1"
    assert result["failure_stage"] == pipeline.stage
    assert result["error"] == ("API did not return a response" if mode in {"missing", "none"} else "provider unavailable")
    assert pipeline.state.calls[-1] == (pipeline.stage, "provider-error")
    if pipeline.stage == "execution":
        assert result["model_name"] in {"worker", "worker-api"}
        assert result["api_model_name"] == "worker-api"
    else:
        assert result["model_name"] == result["api_model_name"] == "base"


@pytest.mark.parametrize("mode", ["raised", "missing", "none"])
def test_chat_failure_is_nonempty_error_text(pipeline, mode):
    pipeline.state.mode = mode

    result = pipeline.router.route_single("provider-error")

    assert isinstance(result, str)
    expected_error = "API did not return a response" if mode in {"missing", "none"} else "provider unavailable"
    assert expected_error in result


@pytest.mark.parametrize("truth_key", ["ground_truth", "gt", "answer"])
@pytest.mark.parametrize("truth_value", ["expected answer", 0])
def test_failed_evaluation_counts_as_zero_accuracy(pipeline, truth_key, truth_value):
    result = pipeline.router.route_single({
        "query": "provider-error", truth_key: truth_value, "task_performance": 1.0,
    })

    assert result["success"] is False
    assert result[truth_key] == truth_value
    assert result["task_performance"] == 0.0


def test_knn_local_decomposition_failure_never_calls_provider(monkeypatch):
    pipeline = build_pipeline("knnmultiroundrouter", monkeypatch)
    router = pipeline.router
    module = importlib.import_module("llmrouter.models.knnmultiroundrouter.router")
    router.use_local_llm = True
    router.local_tokenizer = SimpleNamespace(apply_chat_template=lambda *args, **kwargs: "local prompt")

    def generate(*args, **kwargs):
        raise RuntimeError("local decomposition unavailable")

    router.local_llm = SimpleNamespace(generate=generate)
    monkeypatch.setattr(module, "SamplingParams", lambda **kwargs: SimpleNamespace(**kwargs))

    result = router.route_single({"query": "first", "ground_truth": 0})

    assert result["success"] is False
    assert result["failure_stage"] == "decomposition"
    assert result["error"] == "local decomposition unavailable"
    assert result["model_name"] == "base"
    assert result["task_performance"] == 0.0
    assert pipeline.state.calls == []
    assert "local decomposition unavailable" in router.route_single("first")
    assert pipeline.state.calls == []


@pytest.mark.parametrize("successful_prefix", [False, True])
def test_subquery_failure_stops_later_work_and_keeps_prior_tokens(monkeypatch, successful_prefix):
    pipeline = build_pipeline("llmmultiroundrouter", monkeypatch, failure_stage="execution")
    subqueries = (["first"] if successful_prefix else []) + ["provider-error", "never-run"]
    pipeline.state.decomposition = "\n".join(f"{query}: worker" for query in subqueries)

    result = pipeline.router.route_single({"query": "whole question", "ground_truth": "expected answer"})

    expected_calls = [("decomposition", "whole question")]
    expected_calls.extend(("execution", query) for query in subqueries[:-1])
    assert pipeline.state.calls == expected_calls
    assert result["query"] == "whole question"
    assert result["failure_stage"] == "execution"
    assert result["prompt_tokens"] == result["input_token"] == (2 if successful_prefix else 0)
    assert result["completion_tokens"] == result["output_token"] == (1 if successful_prefix else 0)
    assert result["task_performance"] == 0.0


@pytest.mark.parametrize("failure_source", ["encoder", "predict"])
def test_knn_routing_failure_has_its_own_stage(monkeypatch, failure_source):
    pipeline = build_pipeline("knnmultiroundrouter", monkeypatch)

    def fail(*args, **kwargs):
        raise RuntimeError(f"{failure_source} unavailable")

    if failure_source == "encoder":
        module = importlib.import_module("llmrouter.models.knnmultiroundrouter.router")
        monkeypatch.setattr(module, "get_longformer_embedding", fail)
    else:
        pipeline.router.knn_model.predict = fail

    result = pipeline.router.route_single({"query": "first"})

    assert result["success"] is False
    assert result["failure_stage"] == "routing"
    assert result["error"] == f"{failure_source} unavailable"
    assert pipeline.state.calls == []


@pytest.mark.parametrize("router_name", ROUTER_NAMES)
def test_execution_configuration_failure_keeps_model_identity(monkeypatch, router_name):
    pipeline = build_pipeline(router_name, monkeypatch)
    pipeline.router.llm_data["worker"].pop("api_endpoint")
    pipeline.router.api_endpoint = None

    result = pipeline.router.route_single({"query": "first"})

    assert result["success"] is False
    assert result["failure_stage"] == "execution"
    assert "API endpoint not found" in result["error"]
    assert result["model_name"] in {"worker", "worker-api"}
    assert result["api_model_name"] == "worker-api"
    assert "task_performance" not in result


@pytest.mark.parametrize("router_name", ROUTER_NAMES)
def test_successful_chat_and_evaluation_contracts_are_preserved(router_name, monkeypatch):
    pipeline = build_pipeline(router_name, monkeypatch)

    assert pipeline.router.route_single("first") == "final answer"
    result = pipeline.router.route_single({"query": "first", "request_id": "request-1"})

    assert result["success"] is True
    assert result["response"] == "final answer"
    assert result["request_id"] == "request-1"
    assert result["prompt_tokens"] == result["input_token"] == 2
    assert result["completion_tokens"] == result["output_token"] == 1
    assert "error" not in result


@pytest.mark.parametrize("entrypoint", ["unified", "standalone"])
@pytest.mark.parametrize("batch", [False, True])
def test_multiround_cli_provider_failure_exits_after_all_results(
    pipeline, monkeypatch, tmp_path, capsys, entrypoint, batch,
):
    from llmrouter.cli import router_inference, router_main

    monkeypatch.setattr(router_inference, "load_router", lambda *args: pipeline.router)
    config = tmp_path / "config.yaml"
    config.write_text("{}\n", encoding="utf-8")
    args = ["--router", pipeline.name, "--config", str(config)]
    if batch:
        queries = ["first", "provider-error", "last"]
        source = tmp_path / "queries.json"
        source.write_text(json.dumps(queries), encoding="utf-8")
        args.extend(["--input", str(source)])
    else:
        queries = ["provider-error"]
        args.extend(["--query", "provider-error"])

    if entrypoint == "unified":
        module = ModuleType("llmrouter.cli.router_inference")
        for name in (
            "load_router", "route_query", "infer_query",
            "load_queries_from_file", "save_results_to_file",
        ):
            setattr(module, name, getattr(router_inference, name))
        monkeypatch.setitem(sys.modules, module.__name__, module)
        run = lambda: router_main.main(["infer", *args])
    else:
        monkeypatch.setattr(sys, "argv", ["router_inference", *args])
        run = router_inference.main

    with pytest.raises(SystemExit) as error:
        run()

    assert error.value.code == 1
    contents = json.loads(capsys.readouterr().out)
    results = contents if batch else [contents]
    assert [result["query"] for result in results] == queries
    assert [result["success"] for result in results] == ([True, False, True] if batch else [False])
    failure = results[1] if batch else results[0]
    assert failure["error"] == "provider unavailable"
    assert failure["routing_result"]["failure_stage"] == pipeline.stage
    assert failure["response"] == "upstream failed"
