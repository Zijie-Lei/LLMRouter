"""Serve configuration, routing, and upstream contracts without TCP sockets."""

import asyncio
from contextlib import ExitStack
import json
from types import SimpleNamespace

import httpx
from fastapi import HTTPException
from fastapi.testclient import TestClient
import pytest
from starlette.websockets import WebSocketDisconnect
import yaml

from llmrouter.serve import server
from llmrouter.serve.config import LLMConfig, ServeConfig


MESSAGES = [{"role": "user", "content": "Explain recursion"}]


@pytest.fixture
def application(monkeypatch):
    config = ServeConfig(
        router_name="offline-router", show_model_prefix=False,
        llms={
            "first": LLMConfig(
                name="first", provider="provider", model_id="backend-first",
                base_url="https://first.example.test/v1", max_tokens=64,
            ),
            "second": LLMConfig(
                name="second", provider="other", model_id="backend-second",
                base_url="https://second.example.test/v1", max_tokens=96,
            ),
        },
    )
    state = SimpleNamespace(config=config, requests=[], queries=[], route_result={"model_name": "second"}, route_error=None)

    class LocalRouter:
        def route_single(self, query):
            state.queries.append(query)
            if state.route_error is not None:
                raise state.route_error
            return state.route_result.copy()

    state.router = LocalRouter()
    monkeypatch.setattr(server.RouterAdapter, "_load_router", lambda adapter: setattr(adapter, "router", state.router))
    state.handle = lambda request: httpx.Response(
        200, json={"choices": [{"message": {"content": "upstream answer"}}]},
    )

    def handle(request):
        state.requests.append(request)
        return state.handle(request)

    transport = httpx.MockTransport(handle)
    real_async_client = httpx.AsyncClient
    monkeypatch.setattr(
        server.httpx, "AsyncClient",
        lambda *args, **kwargs: real_async_client(*args, transport=transport, **kwargs),
    )
    with ExitStack() as stack:
        def client(config_path=None):
            app = server.create_app(config_path=config_path) if config_path else server.create_app(config=state.config)
            return stack.enter_context(TestClient(app))

        state.client = client
        yield state


def test_yaml_configuration_reaches_health_and_model_endpoints(application, tmp_path):
    config_path = tmp_path / "serve.yaml"
    config_path.write_text(yaml.safe_dump({
        "serve": {"host": "127.0.0.1", "port": 9001, "show_model_prefix": False, "fail_on_routing_error": True},
        "router": {"name": "configured-router", "config_path": "router.yaml"},
        "api_keys": {"provider": ["fake-one", "fake-two"]},
        "llms": {
            "configured": {
                "provider": "provider", "model": "backend-configured",
                "base_url": "https://configured.example.test/v1", "api_key": "fake-model-key",
                "input_price": 0.2, "output_price": 0.4, "max_tokens": 32, "context_limit": 1000,
            },
            "defaults": {},
        },
    }), encoding="utf-8")

    config = ServeConfig.from_yaml(str(config_path))

    assert (config.host, config.port) == ("127.0.0.1", 9001)
    assert config.show_model_prefix is False
    assert config.fail_on_routing_error is True
    assert config.router_config_path == "router.yaml"
    assert config.api_keys == {"provider": ["fake-one", "fake-two"]}
    assert config.llms["configured"] == LLMConfig(
        name="configured", provider="provider", model_id="backend-configured",
        base_url="https://configured.example.test/v1", api_key="fake-model-key",
        input_price=0.2, output_price=0.4, max_tokens=32, context_limit=1000,
    )
    assert config.llms["defaults"].provider == "openai"
    assert config.llms["defaults"].model_id == "defaults"
    assert config.llms["defaults"].base_url == "https://api.openai.com/v1"
    client = application.client(config_path=str(config_path))
    assert client.get("/health").json() == {
        "status": "ok", "router": "configured-router", "llms": ["configured", "defaults"],
    }
    assert client.get("/v1/models").json() == {
        "data": [{"id": "configured", "object": "model"}, {"id": "defaults", "object": "model"}],
    }
    assert application.requests == []


def test_missing_configuration_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="Config file not found"):
        ServeConfig.from_yaml(str(tmp_path / "missing.yaml"))


def test_explicit_model_forwards_upstream_url_auth_and_generation_params(application):
    application.config.llms["first"].api_key = "fake-model-key"
    application.config.api_keys = {"provider": ["fake-provider-key"]}
    payload = {
        "model": "first", "messages": MESSAGES, "max_tokens": 91, "temperature": 0.25,
    }

    response = application.client().post("/v1/chat/completions", json=payload)

    assert response.status_code == 200
    assert response.json()["model"] == "first"
    assert response.json()["choices"][0]["message"]["content"] == "upstream answer"
    assert application.queries == []
    request = application.requests[0]
    assert request.method == "POST"
    assert str(request.url) == "https://first.example.test/v1/chat/completions"
    assert request.headers["Authorization"] == "Bearer fake-model-key"
    assert json.loads(request.content) == {
        "model": "backend-first", "messages": MESSAGES, "max_tokens": 64, "temperature": 0.25,
    }
    assert request.extensions["timeout"]["read"] == 120.0


def test_provider_keys_rotate_across_requests(application):
    application.config.api_keys = {"provider": ["fake-one", "fake-two"]}
    client = application.client()

    for _ in range(3):
        assert client.post("/v1/chat/completions", json={"model": "first", "messages": MESSAGES}).status_code == 200

    assert [request.headers["Authorization"] for request in application.requests] == [
        "Bearer fake-one", "Bearer fake-two", "Bearer fake-one",
    ]


@pytest.mark.parametrize("key", ["fake-environment-key", None])
def test_environment_key_resolution_and_absent_auth(application, monkeypatch, key):
    application.config.api_keys = {"provider": "${LLMROUTER_TEST_SERVE_KEY}"}
    if key is None:
        monkeypatch.delenv("LLMROUTER_TEST_SERVE_KEY", raising=False)
    else:
        monkeypatch.setenv("LLMROUTER_TEST_SERVE_KEY", key)

    response = application.client().post("/v1/chat/completions", json={"model": "first", "messages": MESSAGES})

    assert response.status_code == 200
    assert application.requests[0].headers.get("Authorization") == (f"Bearer {key}" if key else None)


@pytest.mark.parametrize("router_result,expected", [
    ({"model_name": "second"}, "second"),
    ({"predicted_llm": "second"}, "second"),
    ({"model_name": "SECOND"}, "second"),
    ({"model_name": "unlisted"}, "first"),
])
def test_auto_selection_uses_real_adapter_and_preserves_full_messages(application, router_result, expected):
    application.route_result = router_result
    messages = [
        {"role": "user", "content": "old query"},
        {"role": "assistant", "content": "old answer"},
        {"role": "user", "content": "x" * 520},
    ]

    response = application.client().post("/v1/chat/completions", json={"model": "auto", "messages": messages})

    assert response.status_code == 200
    assert response.json()["model"] == expected
    assert application.queries == [{"query": "x" * 500}]
    body = json.loads(application.requests[0].content)
    assert body["messages"] == messages
    assert body["model"] == f"backend-{expected}"


@pytest.mark.parametrize("strict", [False, True])
def test_routing_exception_obeys_failure_configuration(application, strict):
    application.config.fail_on_routing_error = strict
    application.route_error = RuntimeError("router unavailable")

    response = application.client().post("/v1/chat/completions", json={"messages": MESSAGES})

    if strict:
        assert response.status_code == 500
        assert response.json()["detail"] == "routing failed: RuntimeError: router unavailable"
        assert application.requests == []
    else:
        assert response.status_code == 200
        assert response.json()["model"] == "first"
        assert json.loads(application.requests[0].content)["model"] == "backend-first"


@pytest.mark.parametrize("payload", [
    {}, {"messages": [{"role": "user"}]}, {"messages": [{"role": "user", "content": 123}]},
])
def test_invalid_chat_request_is_rejected_before_routing_or_upstream(application, payload):
    response = application.client().post("/v1/chat/completions", json=payload)

    assert response.status_code == 422
    assert response.json()["detail"]
    assert application.queries == application.requests == []


@pytest.mark.parametrize("status", [401, 429, 503])
def test_upstream_http_failure_preserves_status_and_error_detail(application, status):
    application.handle = lambda request: httpx.Response(status, text="upstream rejected request")

    response = application.client().post("/v1/chat/completions", json={"model": "first", "messages": MESSAGES})

    assert response.status_code == status
    assert response.json() == {"detail": "upstream rejected request"}
    assert len(application.requests) == 1


def test_streaming_upstream_http_failure_becomes_an_sse_error(application):
    application.handle = lambda request: httpx.Response(429, text="upstream rate limited")

    response = application.client().post(
        "/v1/chat/completions", json={"model": "first", "messages": MESSAGES, "stream": True},
    )

    assert response.headers["content-type"].startswith("text/event-stream")
    assert json.loads(response.text.removeprefix("data: ").strip()) == {"error": "upstream rate limited"}
    assert json.loads(application.requests[0].content)["stream"] is True


@pytest.mark.parametrize("failure", ["http", "transport", "routing"])
def test_websocket_reports_failure_and_closes(application, failure):
    if failure == "http":
        application.handle = lambda request: httpx.Response(503, text="upstream unavailable")
        expected = "upstream unavailable"
    elif failure == "transport":
        def timeout(request):
            raise httpx.ReadTimeout("upstream timed out", request=request)

        application.handle = timeout
        expected = "upstream timed out"
    else:
        application.config.fail_on_routing_error = True
        application.route_error = RuntimeError("router unavailable")
        expected = "router unavailable"

    with application.client().websocket_connect("/v1/chat/ws") as websocket:
        websocket.send_json({"model": "auto" if failure == "routing" else "first", "messages": MESSAGES})
        assert websocket.receive_json() == {"error": expected}
        with pytest.raises(WebSocketDisconnect) as error:
            websocket.receive_text()
        assert error.value.code == 1000
    if failure == "routing":
        assert application.requests == []
    else:
        assert json.loads(application.requests[0].content)["stream"] is True


def test_unknown_backend_model_raises_before_opening_upstream(application):
    with pytest.raises(HTTPException) as error:
        asyncio.run(server.LLMBackend(application.config).call("missing", MESSAGES))

    assert error.value.status_code == 404
    assert error.value.detail == "LLM 'missing' not found"
    assert application.requests == []
