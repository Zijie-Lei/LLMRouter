"""Offline exit-status contracts for both inference CLI entry points."""

import json
import sys
from types import ModuleType, SimpleNamespace

import pytest


class StubRouter:
    cfg = {}
    llm_data = {
        "chosen": {
            "model": "upstream-model",
            "api_endpoint": "https://example.test/v1",
        }
    }

    def route_single(self, query):
        if query["query"] == "route-error":
            raise ValueError("routing unavailable")
        if query["query"] == "missing-model":
            return {}
        return {"model_name": "chosen", "query": query["query"]}


@pytest.fixture(params=["unified", "standalone"])
def cli(request, monkeypatch, tmp_path):
    from llmrouter.cli import router_inference, router_main

    config = tmp_path / "config.yaml"
    config.write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(router_inference, "load_router", lambda *args: StubRouter())

    def provider_result(query, **kwargs):
        if query["query"] == "provider-error":
            return {"response": "API Error: rate limited", "error": "rate limited"}
        return {"response": "answer"}

    monkeypatch.setattr(router_inference, "call_api", provider_result)

    if request.param == "unified":
        # Expose only the inference operations the command needs. Router loading
        # stays local while query and output handling use the real functions.
        module = ModuleType("llmrouter.cli.router_inference")
        for name in (
            "load_router", "route_query", "infer_query",
            "load_queries_from_file", "save_results_to_file",
        ):
            setattr(module, name, getattr(router_inference, name))
        monkeypatch.setitem(sys.modules, module.__name__, module)

        def run(args):
            return router_main.main(["infer", *args])
    else:
        def run(args):
            monkeypatch.setattr(sys, "argv", ["router_inference", *args])
            return router_inference.main()

    return SimpleNamespace(
        run=lambda args: run(["--router", "stub", "--config", str(config), *args]),
    )


@pytest.mark.parametrize("route_only", [False, True])
def test_single_failed_query_exits_nonzero_after_emitting_json(cli, capsys, route_only):
    args = ["--query", "route-error"]
    if route_only:
        args.append("--route-only")

    with pytest.raises(SystemExit) as error:
        cli.run(args)

    assert error.value.code == 1
    result = json.loads(capsys.readouterr().out)
    assert result["success"] is False
    assert result["query"] == "route-error"
    assert result["error"] == "routing unavailable"


@pytest.mark.parametrize("route_only", [False, True])
def test_missing_model_keeps_query_in_error_json(cli, capsys, route_only):
    args = ["--query", "missing-model"]
    if route_only:
        args.append("--route-only")

    with pytest.raises(SystemExit) as error:
        cli.run(args)

    assert error.value.code == 1
    result = json.loads(capsys.readouterr().out)
    assert result["success"] is False
    assert result["query"] == "missing-model"
    assert result["error"] == "Router did not return a model name"


@pytest.mark.parametrize("route_only", [False, True])
def test_mixed_batch_keeps_all_results_and_exits_nonzero(cli, tmp_path, capsys, route_only):
    queries = ["first", "route-error", "last"]
    source = tmp_path / "queries.json"
    source.write_text(json.dumps(queries), encoding="utf-8")
    args = ["--input", str(source)]
    if route_only:
        args.append("--route-only")

    with pytest.raises(SystemExit) as error:
        cli.run(args)

    assert error.value.code == 1
    results = json.loads(capsys.readouterr().out)
    assert [result["query"] for result in results] == queries
    assert [result["success"] for result in results] == [True, False, True]


@pytest.mark.parametrize("route_only", [False, True])
@pytest.mark.parametrize("batch", [False, True])
def test_successful_queries_complete_normally(cli, tmp_path, capsys, route_only, batch):
    if batch:
        source = tmp_path / "queries.txt"
        source.write_text("first\nlast\n", encoding="utf-8")
        args = ["--input", str(source)]
    else:
        args = ["--query", "first"]
    if route_only:
        args.append("--route-only")

    assert cli.run(args) is None

    result = json.loads(capsys.readouterr().out)
    results = result if batch else [result]
    assert [item["query"] for item in results] == (["first", "last"] if batch else ["first"])
    assert all(item["success"] is True for item in results)
    assert all(item["model_name"] == "chosen" for item in results)
    if not route_only:
        assert all(item["response"] == "answer" for item in results)


@pytest.mark.parametrize("output_format", ["json", "jsonl"])
def test_provider_failure_is_written_before_failure_exit(cli, tmp_path, capsys, output_format):
    output = tmp_path / f"results.{output_format}"

    with pytest.raises(SystemExit) as error:
        cli.run([
            "--query", "provider-error", "--output", str(output),
            "--output-format", output_format,
        ])

    assert error.value.code == 1
    assert capsys.readouterr().out == ""
    contents = output.read_text(encoding="utf-8")
    results = json.loads(contents) if output_format == "json" else [json.loads(line) for line in contents.splitlines()]
    assert len(results) == 1
    result = results[0]
    assert result["success"] is False
    assert result["query"] == "provider-error"
    assert result["error"] == "rate limited"
    assert result["response"] == "API Error: rate limited"
    assert result["model_name"] == "chosen"
    assert result["api_model_name"] == "upstream-model"
    assert result["routing_result"]["model_name"] == "chosen"


@pytest.mark.parametrize("provider_result", [{}, {"response": None}])
def test_missing_provider_response_is_reported(monkeypatch, provider_result):
    from llmrouter.cli import router_inference

    monkeypatch.setattr(router_inference, "call_api", lambda *args, **kwargs: provider_result)

    result = router_inference.infer_query("first", StubRouter(), "stub")

    assert result["success"] is False
    assert result["error"] == "API did not return a response"
    assert result["model_name"] == "chosen"
    assert result["api_model_name"] == "upstream-model"


def test_empty_successful_provider_response_is_preserved(monkeypatch):
    from llmrouter.cli import router_inference

    monkeypatch.setattr(router_inference, "call_api", lambda *args, **kwargs: {"response": ""})

    result = router_inference.infer_query("first", StubRouter(), "stub")

    assert result["success"] is True
    assert result["response"] == ""
    assert "error" not in result


def test_empty_provider_error_still_reports_failure(monkeypatch):
    from llmrouter.cli import router_inference

    monkeypatch.setattr(
        router_inference, "call_api",
        lambda *args, **kwargs: {"response": "API Error: ", "error": ""},
    )

    result = router_inference.infer_query("first", StubRouter(), "stub")

    assert result["success"] is False
    assert result["error"] == ""
    assert result["response"] == "API Error: "


@pytest.mark.parametrize("router_name", ["llmmultiroundrouter", "knnmultiroundrouter"])
@pytest.mark.parametrize("response", ["legacy answer", "API Error: a quoted example"])
def test_multiround_legacy_text_is_not_classified_by_its_prefix(router_name, response):
    from llmrouter.cli import router_inference

    received = []

    def route(query):
        received.append(query)
        return response

    result = router_inference.infer_query("first", SimpleNamespace(route_single=route), router_name)

    assert received == [{"query": "first"}]
    assert result["success"] is True
    assert result["response"] == response


@pytest.mark.parametrize("router_name", ["llmmultiroundrouter", "knnmultiroundrouter"])
@pytest.mark.parametrize("response", [None, [], {"success": False, "response": "partial"}])
def test_multiround_invalid_or_failed_results_are_not_successful(router_name, response):
    from llmrouter.cli import router_inference

    router = SimpleNamespace(route_single=lambda query: response)

    result = router_inference.infer_query("first", router, router_name)

    assert result["success"] is False
    assert result["query"] == "first"
    assert result["error"]
