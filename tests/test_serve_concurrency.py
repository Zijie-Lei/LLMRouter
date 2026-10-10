"""Keep real HTTP/WebSocket routing responsive and diagnostics request-local."""

import asyncio
from concurrent.futures import ThreadPoolExecutor
import json
import threading
from types import SimpleNamespace

import httpx
from fastapi.testclient import TestClient
import pytest

from llmrouter.serve import server
from llmrouter.serve.config import LLMConfig, ServeConfig


class BoundedGuard:
    """Use the real adapter guard, with bounded failure and observable contention."""

    def __init__(self, lock):
        self.lock = lock
        self.contended = threading.Event()

    def __enter__(self):
        if not self.lock.acquire(blocking=False):
            self.contended.set()
            assert self.lock.acquire(timeout=5), "adapter guard acquisition timed out"
        return self

    def __exit__(self, *_args):
        self.lock.release()


@pytest.fixture
def serving(monkeypatch):
    state = SimpleNamespace(signals={"request": {"query": None}})

    def route(query):
        state.signals["request"]["query"] = query
        return {
            "model_name": query,
            "routing_reason": f"reason-{query}",
            "routing_signals": state.signals,
        }

    state.route = route

    class LocalRouter:
        def route_single(self, payload):
            return state.route(payload["query"])

    def load(adapter):
        adapter.router = LocalRouter()
        adapter._route_lock = BoundedGuard(adapter._route_lock)
        state.adapter = adapter

    monkeypatch.setattr(server.RouterAdapter, "_load_router", load)

    async def call(_backend, model, messages, *args, stream=False, **kwargs):
        text = messages[-1]["content"]
        if stream:
            async def chunks():
                yield "data: " + json.dumps({"model": model, "choices": [{"delta": {"content": text}}]}) + "\n\n"
                yield "data: [DONE]\n\n"
            return chunks()
        return {"model": model, "choices": [{"message": {"content": text}}]}

    monkeypatch.setattr(server.LLMBackend, "call", call)
    state.app = server.create_app(config=ServeConfig(
        router_name="local", fail_on_routing_error=True, show_model_prefix=False,
        llms={name: LLMConfig(name=name, provider="local", model_id=name, base_url="https://example.test/v1") for name in ("first", "second")},
    ))
    return state


def test_adapter_serializes_calls_to_a_mutable_router(serving):
    entered = threading.Event()
    second_entered = threading.Event()
    release = threading.Event()
    shared = SimpleNamespace(query=None)

    def mutable_route(query):
        shared.query = query
        if query == "first":
            entered.set()
            assert release.wait(5), "test router was not released"
        else:
            second_entered.set()
        return {"model_name": shared.query}

    serving.route = mutable_route
    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(serving.adapter.route, "first", ["first", "second"])
        second = None
        try:
            assert entered.wait(5)
            second = pool.submit(serving.adapter.route, "second", ["first", "second"])
            assert serving.adapter._route_lock.contended.wait(5)
            assert not second_entered.is_set()
        finally:
            release.set()
        assert first.result(timeout=15) == "first"
        assert second.result(timeout=15) == "second"


@pytest.mark.parametrize("stream", [False, True])
def test_http_health_responds_while_routing_is_blocked(serving, stream):
    async def exercise():
        loop = asyncio.get_running_loop()
        entered = asyncio.Event()
        release = threading.Event()
        original = serving.route

        def blocked(query):
            loop.call_soon_threadsafe(entered.set)
            if not release.wait(5):
                raise TimeoutError("test routing gate was not released")
            return original(query)

        serving.route = blocked
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=serving.app), base_url="http://app.test") as client:
            task = asyncio.create_task(client.post("/v1/chat/completions", json={
                "model": "auto", "messages": [{"role": "user", "content": "first"}], "stream": stream,
            }))
            try:
                await asyncio.wait_for(entered.wait(), 6)
                health = await asyncio.wait_for(client.get("/health"), 5)
                assert health.status_code == 200
                assert not task.done(), "routing must still be waiting when health returns"
            finally:
                release.set()
                result = (await asyncio.gather(task, return_exceptions=True))[0]
            assert isinstance(result, httpx.Response) and result.status_code == 200
            if stream:
                assert '"model": "first"' in result.text and "data: [DONE]" in result.text
            else:
                assert result.json() == {"model": "first", "choices": [{"message": {"content": "first"}}]}

    asyncio.run(exercise())


def test_websocket_health_responds_while_auto_routing_is_blocked(serving):
    entered = threading.Event()
    release = threading.Event()
    finished = threading.Event()
    original = serving.route

    def blocked(query):
        entered.set()
        try:
            if not release.wait(5):
                raise TimeoutError("test websocket routing gate was not released")
            return original(query)
        finally:
            finished.set()

    serving.route = blocked
    with TestClient(serving.app) as client:
        with client.websocket_connect("/v1/chat/ws") as websocket:
            websocket.send_json({"model": "auto", "messages": [{"role": "user", "content": "first"}]})
            try:
                assert entered.wait(5)
                assert client.get("/health").status_code == 200
                assert not finished.is_set(), "health must finish before the worker is released"
            finally:
                release.set()
            assert websocket.receive_json() == {"model": "first", "choices": [{"delta": {"content": "first"}}]}
            assert websocket.receive_text() == "data: [DONE]\n\n"


def test_parallel_http_responses_and_nested_diagnostics_stay_with_their_queries(serving, monkeypatch, capsys):
    async def exercise():
        completed = asyncio.Event()
        returned = 0
        original = server.run_in_threadpool

        async def synchronized_offload(function, *args, **kwargs):
            nonlocal returned
            result = await original(function, *args, **kwargs)
            returned += 1
            if returned == 2:
                completed.set()
            # Force both routing workers to finish before either HTTP handler
            # resumes. Reading shared diagnostics after await would mix requests.
            await asyncio.wait_for(completed.wait(), 5)
            return result

        monkeypatch.setattr(server, "run_in_threadpool", synchronized_offload)
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=serving.app), base_url="http://app.test") as client:
            responses = await asyncio.wait_for(asyncio.gather(*[
                client.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": query}]})
                for query in ("first", "second")
            ]), 15)
        assert [response.status_code for response in responses] == [200, 200]
        assert [response.json()["model"] for response in responses] == ["first", "second"]
        assert [response.json()["choices"][0]["message"]["content"] for response in responses] == ["first", "second"]

    asyncio.run(exercise())
    output = capsys.readouterr().out
    for query in ("first", "second"):
        assert f"Query: '{query}...' -> {query} reason=reason-{query} signals={{'request': {{'query': '{query}'}}}}" in output


@pytest.mark.parametrize("strict", [False, True])
def test_noncopyable_diagnostics_do_not_fail_successful_routing(serving, capsys, strict):
    serving.adapter.fail_on_error = strict
    original = serving.route

    def noncopyable(query):
        return {"model_name": query, "routing_signals": {"guard": threading.Lock()}}

    async def exercise():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=serving.app), base_url="http://app.test") as client:
            serving.route = noncopyable
            first = await asyncio.wait_for(client.post("/v1/chat/completions", json={
                "messages": [{"role": "user", "content": "first"}],
            }), 5)
            assert first.status_code == 200 and first.json()["model"] == "first"
            assert first.json()["choices"][0]["message"]["content"] == "first"
            serving.route = original
            second = await asyncio.wait_for(client.post("/v1/chat/completions", json={
                "messages": [{"role": "user", "content": "second"}],
            }), 5)
            assert second.status_code == 200 and second.json()["model"] == "second"

    asyncio.run(exercise())
    output = capsys.readouterr().out
    assert "-> first reason=routing diagnostics unavailable: TypeError" in output
    assert "-> second reason=reason-second signals={'request': {'query': 'second'}}" in output
