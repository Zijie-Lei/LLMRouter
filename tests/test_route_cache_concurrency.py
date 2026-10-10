"""Exercise sticky routing races through the real Responses API and cache lifecycle."""

import asyncio
from types import SimpleNamespace

import httpx
import pytest

from openclaw_router import server
from openclaw_router.config import LLMConfig, MediaConfig, OpenClawConfig, RouterConfig


def captured(function, name):
    """Inspect private bookkeeping without adding a production inspection API."""
    cells = dict(zip(function.__code__.co_freevars, function.__closure__ or ()))
    return cells[name].cell_contents


@pytest.fixture
def sticky(monkeypatch):
    state = SimpleNamespace(decisions=[])

    async def choose(query, user, candidates):
        return candidates[0]

    state.choose = choose

    async def select(_router, query, *, user=None, candidate_models=None):
        candidates = tuple(candidate_models)
        state.decisions.append((query, user, candidates))
        return await state.choose(query, user, candidates)

    async def call(_backend, model, messages, *args, **kwargs):
        return {
            "id": "chat-test", "model": model,
            "choices": [{"message": {"role": "assistant", "content": messages[-1]["content"]}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }

    monkeypatch.setattr(server.OpenClawRouter, "select_model", select)
    monkeypatch.setattr(server.LLMBackend, "call", call)
    state.app = server.create_app(config=OpenClawConfig(
        show_model_prefix=False, show_model_suffix=False,
        router=RouterConfig(strategy="random"), media=MediaConfig(enabled=False),
        llms={
            name: LLMConfig(name=name, provider="mock", model_id=name,
                            base_url="https://example.test/v1", context_limit=limit)
            for name, limit in (("first", 256), ("second", 4096))
        },
    ))
    endpoint = next(route.endpoint for route in state.app.routes if route.path == "/v1/responses")
    state.selector = captured(captured(endpoint, "_serve_chat_completions"), "_select_cached_model")
    state.locks = captured(state.selector, "route_cache_locks")
    state.cache = captured(state.selector, "codex_route_cache")
    return state


def request_payload(**changes):
    return {"model": "auto", "user": "alice", "prompt_cache_key": "turn", "input": "same", "max_output_tokens": 16, **changes}


def client_for(state):
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=state.app, raise_app_exceptions=False),
        base_url="http://app.test",
    )


async def wait_for_users(state, count):
    async def wait():
        # Yield scheduling, not a wall-clock delay: observe actual waiter registration.
        while not (len(state.locks) == 1 and next(iter(state.locks.values()))[1] == count):
            await asyncio.sleep(0)
    await asyncio.wait_for(wait(), 5)


async def finish(tasks):
    return await asyncio.wait_for(asyncio.gather(*tasks, return_exceptions=True), 15)


@pytest.mark.parametrize("other", [
    {"user": "bob"}, {"input": "other"},
    {"prompt_cache_key": "other-turn"}, {"model": "other-advertised"},
])
def test_same_turn_routes_once_without_blocking_distinct_keys_or_health(sticky, other):
    async def exercise():
        entered, release = asyncio.Event(), asyncio.Event()

        async def choose(query, user, candidates):
            if len(sticky.decisions) == 1:
                entered.set()
                await release.wait()
                return "first"
            return "second"

        sticky.choose = choose
        tasks = []
        async with client_for(sticky) as client:
            try:
                tasks.append(asyncio.create_task(client.post("/v1/responses", json=request_payload())))
                await asyncio.wait_for(entered.wait(), 5)
                tasks.append(asyncio.create_task(client.post("/v1/responses", json=request_payload())))
                await wait_for_users(sticky, 2)
                health = await asyncio.wait_for(client.get("/health"), 5)
                distinct = await asyncio.wait_for(client.post("/v1/responses", json=request_payload(**other)), 5)
                assert health.status_code == distinct.status_code == 200
                assert distinct.json()["model"] == "second"
                assert all(not task.done() for task in tasks)
            finally:
                release.set()
                responses = await finish(tasks)
            assert [response.status_code for response in responses] == [200, 200]
            assert [response.json()["model"] for response in responses] == ["first", "first"]
            again = await asyncio.wait_for(client.post("/v1/responses", json=request_payload()), 5)
            assert again.status_code == 200 and again.json()["model"] == "first"
        assert len(sticky.decisions) == 2
        assert not sticky.locks

    asyncio.run(exercise())


def test_waiting_request_rechecks_backend_context_eligibility(sticky):
    async def exercise():
        entered, release = asyncio.Event(), asyncio.Event()

        async def choose(query, user, candidates):
            if len(sticky.decisions) == 1:
                entered.set()
                await release.wait()
            return candidates[0]

        sticky.choose = choose
        history = [{"role": "assistant", "content": "history " * 200}, {"role": "user", "content": "same"}]
        tasks = []
        async with client_for(sticky) as client:
            try:
                tasks.append(asyncio.create_task(client.post("/v1/responses", json=request_payload())))
                await asyncio.wait_for(entered.wait(), 5)
                tasks.append(asyncio.create_task(client.post("/v1/responses", json=request_payload(input=history))))
                await wait_for_users(sticky, 2)
            finally:
                release.set()
                responses = await finish(tasks)
            assert [response.status_code for response in responses] == [200, 200]
            assert [response.json()["model"] for response in responses] == ["first", "second"]
            again = await asyncio.wait_for(client.post("/v1/responses", json=request_payload()), 5)
            assert again.status_code == 200 and again.json()["model"] == "second"
        assert [decision[2] for decision in sticky.decisions] == [("first", "second"), ("second",)]
        assert not sticky.locks

    asyncio.run(exercise())


@pytest.mark.parametrize("cancel_leader", [False, True])
def test_failed_or_cancelled_leader_releases_waiter_for_retry(sticky, cancel_leader):
    async def exercise():
        entered, release = asyncio.Event(), asyncio.Event()

        async def choose(query, user, candidates):
            if len(sticky.decisions) == 1:
                entered.set()
                await release.wait()
                raise RuntimeError("decision failed")
            return "second"

        sticky.choose = choose
        tasks = []
        async with client_for(sticky) as client:
            try:
                tasks.append(asyncio.create_task(client.post("/v1/responses", json=request_payload())))
                await asyncio.wait_for(entered.wait(), 5)
                tasks.append(asyncio.create_task(client.post("/v1/responses", json=request_payload())))
                await wait_for_users(sticky, 2)
                if cancel_leader:
                    tasks[0].cancel()
                    first = (await finish(tasks[:1]))[0]
                    assert isinstance(first, asyncio.CancelledError)
            finally:
                release.set()
                results = await finish(tasks)
            if cancel_leader:
                assert isinstance(results[0], asyncio.CancelledError)
            else:
                assert results[0].status_code == 500
            assert results[1].status_code == 200 and results[1].json()["model"] == "second"
            again = await asyncio.wait_for(client.post("/v1/responses", json=request_payload()), 5)
            assert again.status_code == 200 and again.json()["model"] == "second"
        assert len(sticky.decisions) == 2
        assert not sticky.locks

    asyncio.run(exercise())


def test_cancelled_waiter_keeps_the_original_lock_for_remaining_requests(sticky):
    async def exercise():
        entered, release = asyncio.Event(), asyncio.Event()

        async def choose(query, user, candidates):
            entered.set()
            await release.wait()
            return "first"

        sticky.choose = choose
        tasks = []
        async with client_for(sticky) as client:
            try:
                tasks.append(asyncio.create_task(client.post("/v1/responses", json=request_payload())))
                await asyncio.wait_for(entered.wait(), 5)
                tasks.append(asyncio.create_task(client.post("/v1/responses", json=request_payload())))
                await wait_for_users(sticky, 2)
                lock = next(iter(sticky.locks.values()))[0]
                tasks[1].cancel()
                assert isinstance((await finish(tasks[1:]))[0], asyncio.CancelledError)
                await wait_for_users(sticky, 1)
                tasks.append(asyncio.create_task(client.post("/v1/responses", json=request_payload())))
                await wait_for_users(sticky, 2)
                assert next(iter(sticky.locks.values()))[0] is lock
            finally:
                release.set()
                results = await finish(tasks)
            assert [results[index].json()["model"] for index in (0, 2)] == ["first", "first"]
            assert [results[index].status_code for index in (0, 2)] == [200, 200]
        assert len(sticky.decisions) == 1
        assert not sticky.locks

    asyncio.run(exercise())


def test_completed_turns_keep_cache_bounded_and_preserve_lru(sticky):
    async def exercise():
        def key(index):
            return ("alice", "auto", "turn", str(index))

        for index in range(1024):
            assert await sticky.selector(key(index), "same", "alice", ["first", "second"]) == "first"
            assert not sticky.locks
        assert len(sticky.cache) == 1024 and len(sticky.decisions) == 1024
        await sticky.selector(key(1), "same", "alice", ["first", "second"])
        await sticky.selector(key(1024), "same", "alice", ["first", "second"])
        assert len(sticky.cache) == 1024
        assert key(0) not in sticky.cache and key(1) in sticky.cache
        await sticky.selector(key(1), "same", "alice", ["first", "second"])
        assert len(sticky.decisions) == 1025
        await sticky.selector(key(0), "same", "alice", ["first", "second"])
        assert len(sticky.decisions) == 1026 and len(sticky.cache) == 1024
        assert not sticky.locks

    asyncio.run(exercise())
