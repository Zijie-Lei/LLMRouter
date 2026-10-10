"""Deterministic concurrency checks for OpenClaw routing and real memory I/O."""

import asyncio
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import threading
from types import SimpleNamespace

from fastapi import FastAPI
import httpx
import numpy as np
import pytest
import torch
from transformers import AutoModel, AutoTokenizer

from openclaw_router.config import LLMConfig, MemoryConfig, OpenClawConfig, RouterConfig
from openclaw_router.memory import ContrieverEmbedder, MemoryBank
from openclaw_router import routers


WAIT_SECONDS = 5


def wait_for(event):
    assert event.wait(WAIT_SECONDS), "Worker did not reach the expected boundary"


class ObservedLock:
    """Keep real mutual exclusion and signal when a second worker attempts it.

    Observing acquisition prevents a race test from passing because its second
    worker had not actually started the guarded operation before release.
    """

    def __init__(self, lock):
        self.lock = lock
        self.attempts = 0
        self.observation_lock = threading.Lock()
        self.contended = threading.Event()

    def __enter__(self):
        with self.observation_lock:
            self.attempts += 1
            if self.attempts == 2:
                self.contended.set()
        assert self.lock.acquire(timeout=WAIT_SECONDS), "Worker lock acquisition timed out"
        return self

    def __exit__(self, *_args):
        self.lock.release()


@pytest.fixture
def encoder_downloads(monkeypatch):
    """Fake downloads only; pooling, normalization, and Torch inference are real."""
    state = SimpleNamespace(
        downloads=[], models=[], tokenizer_calls=[],
        fail_stage=None, fail_once=False,
        device_entered=threading.Event(), device_release=threading.Event(),
        block_device=False,
        token_entered=threading.Event(), token_release=threading.Event(),
        block_token=False,
        forward_barrier=None, forward_gate=None, query_calls={},
        forward_threads=[], counter_lock=threading.Lock(),
    )

    def maybe_fail(stage):
        if state.fail_stage == stage and not state.fail_once:
            state.fail_once = True
            raise RuntimeError(f"{stage} initialization failed")

    class Tokenizer:
        def __call__(self, texts, **options):
            state.tokenizer_calls.append(list(texts))
            if state.block_token:
                state.block_token = False
                state.token_entered.set()
                wait_for(state.token_release)
            vocabulary = {"east": 1, "north": 2, "west": 3}
            rows = [[vocabulary[word] for word in text.split()][:options["max_length"]] for text in texts]
            width = max(map(len, rows))
            return {
                "input_ids": torch.tensor([row + [0] * (width - len(row)) for row in rows]),
                "attention_mask": torch.tensor([[1] * len(row) + [0] * (width - len(row)) for row in rows]),
            }

    class Encoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("vectors", torch.tensor([[99., 99., 99.], [2., 0., 0.], [0., 3., 0.], [-2., 0., 0.]]))
            self.device_ready = False

        def to(self, device):
            if state.block_device:
                state.device_entered.set()
                wait_for(state.device_release)
            maybe_fail("device")
            result = super().to(device)
            self.device_ready = True
            return result

        def eval(self):
            maybe_fail("eval")
            return super().eval()

        def forward(self, input_ids, **_kwargs):
            assert self.device_ready and not self.training, "Partially initialized encoder was used"
            first_token = int(input_ids[0, 0])
            with state.counter_lock:
                count = state.query_calls.get(first_token, 0) + 1
                state.query_calls[first_token] = count
                state.forward_threads.append(threading.get_ident())
            if state.forward_barrier is not None:
                state.forward_barrier.wait(WAIT_SECONDS)
            if state.forward_gate is not None:
                token, occurrence, entered, release = state.forward_gate
                if (first_token, count) == (token, occurrence):
                    entered.set()
                    wait_for(release)
            return SimpleNamespace(last_hidden_state=self.vectors[input_ids])

    def download_tokenizer(name):
        state.downloads.append(("tokenizer", name))
        maybe_fail("tokenizer")
        return Tokenizer()

    def download_model(name):
        state.downloads.append(("model", name))
        maybe_fail("model")
        model = Encoder()
        state.models.append(model)
        return model

    monkeypatch.setattr(AutoTokenizer, "from_pretrained", download_tokenizer)
    monkeypatch.setattr(AutoModel, "from_pretrained", download_model)
    return state


def memory_config(tmp_path, **options):
    return MemoryConfig(enabled=True, path=str(tmp_path / "history.jsonl"), retriever_model="tiny-local", **options)


def routing_config(strategy, memory=None):
    return OpenClawConfig(
        router=RouterConfig(strategy=strategy, provider="local", base_url="http://localhost/v1", auth_mode="none"),
        memory=memory or MemoryConfig(),
        llms={name: LLMConfig(name=name, provider="local", model_id=name, base_url="http://localhost/v1") for name in ("small", "large")},
    )


def test_concurrent_cold_embeddings_load_once_and_never_use_partial_model(encoder_downloads):
    state = encoder_downloads
    state.block_device = True
    embedder = ContrieverEmbedder("tiny-local")
    load_lock = ObservedLock(embedder._load_lock)
    embedder._load_lock = load_lock
    with ThreadPoolExecutor(max_workers=2) as workers:
        first = workers.submit(embedder.embed, ["east"])
        try:
            wait_for(state.device_entered)
            # Ready-state publication must follow device movement and eval.
            assert embedder._model is None and embedder._tokenizer is None
            second = workers.submit(embedder.embed, ["north"])
            wait_for(load_lock.contended)
            assert not first.done() and not second.done()
        finally:
            state.device_release.set()
        np.testing.assert_allclose(first.result(WAIT_SECONDS), [[1., 0., 0.]])
        np.testing.assert_allclose(second.result(WAIT_SECONDS), [[0., 1., 0.]])
    assert state.downloads == [("tokenizer", "tiny-local"), ("model", "tiny-local")]
    assert len(state.models) == 1 and not state.models[0].training


@pytest.mark.parametrize("stage", ["tokenizer", "model", "device", "eval"])
def test_encoder_initialization_failure_publishes_nothing_and_retries(encoder_downloads, stage):
    encoder_downloads.fail_stage = stage
    embedder = ContrieverEmbedder("tiny-local")
    embedder._load_lock = ObservedLock(embedder._load_lock)
    with pytest.raises(RuntimeError, match=f"{stage} initialization failed"):
        embedder.embed(["east"])
    assert (embedder._model, embedder._tokenizer, embedder._torch, embedder._device) == (None, None, None, None)
    np.testing.assert_allclose(embedder.embed(["north"]), [[0., 1., 0.]])
    downloads = list(encoder_downloads.downloads)
    np.testing.assert_allclose(embedder.embed(["east"]), [[1., 0., 0.]])
    assert encoder_downloads.downloads == downloads


def test_waiting_cold_embedding_retries_after_first_initializer_fails(encoder_downloads):
    state = encoder_downloads
    state.fail_stage = "eval"
    state.block_device = True
    embedder = ContrieverEmbedder("tiny-local")
    load_lock = ObservedLock(embedder._load_lock)
    embedder._load_lock = load_lock
    with ThreadPoolExecutor(max_workers=2) as workers:
        first = workers.submit(embedder.embed, ["east"])
        try:
            wait_for(state.device_entered)
            second = workers.submit(embedder.embed, ["north"])
            wait_for(load_lock.contended)
        finally:
            state.device_release.set()
        with pytest.raises(RuntimeError, match="eval initialization failed"):
            first.result(WAIT_SECONDS)
        np.testing.assert_allclose(second.result(WAIT_SECONDS), [[0., 1., 0.]])
    assert state.downloads == [("tokenizer", "tiny-local"), ("model", "tiny-local")] * 2
    assert state.models[0].training and not state.models[1].training


def test_tokenization_is_guarded_but_model_forward_can_run_concurrently(encoder_downloads):
    state = encoder_downloads
    embedder = ContrieverEmbedder("tiny-local")
    embedder.embed(["west"])
    tokenizer_lock = ObservedLock(embedder._tokenizer_lock)
    embedder._tokenizer_lock = tokenizer_lock
    state.block_token = True
    state.forward_barrier = threading.Barrier(2)
    with ThreadPoolExecutor(max_workers=2) as workers:
        first = workers.submit(embedder.embed, ["east"])
        try:
            wait_for(state.token_entered)
            second = workers.submit(embedder.embed, ["north"])
            wait_for(tokenizer_lock.contended)
            assert state.tokenizer_calls == [["west"], ["east"]]
        finally:
            state.token_release.set()
        np.testing.assert_allclose(first.result(WAIT_SECONDS), [[1., 0., 0.]])
        np.testing.assert_allclose(second.result(WAIT_SECONDS), [[0., 1., 0.]])
    assert state.tokenizer_calls == [["west"], ["east"], ["north"]]


def adapter_for(monkeypatch, backend):
    # Replace the model-loading boundary, while constructing and routing through
    # the real adapter (including its per-instance synchronization).
    monkeypatch.setattr(routers.LLMRouterAdapter, "_load_router", lambda self: setattr(self, "router", backend))
    return routers.LLMRouterAdapter("tiny-trained")


def test_adapter_serializes_mutable_backend_and_preserves_each_query_and_pool(monkeypatch):
    entered, release = threading.Event(), threading.Event()

    class Backend:
        query = None

        def route_single(self, request):
            self.query = request["query"]
            if self.query == "east":
                entered.set()
                wait_for(release)
            return {"model_name": {"east": "small", "north": "large"}[self.query]}

    adapter = adapter_for(monkeypatch, Backend())
    route_lock = ObservedLock(adapter._route_lock)
    adapter._route_lock = route_lock
    with ThreadPoolExecutor(max_workers=2) as workers:
        first = workers.submit(adapter.route, "east", ["small", "fallback-one"])
        try:
            wait_for(entered)
            second = workers.submit(adapter.route, "north", ["large", "fallback-two"])
            wait_for(route_lock.contended)
            assert adapter.router.query == "east"
        finally:
            release.set()
        assert first.result(WAIT_SECONDS) == "small"
        assert second.result(WAIT_SECONDS) == "large"


def test_adapter_backend_failure_releases_guard_and_keeps_fallback(monkeypatch):
    class Backend:
        def route_single(self, request):
            if request["query"] == "broken":
                raise RuntimeError("model failure")
            return {"predicted_llm": "large"}

    adapter = adapter_for(monkeypatch, Backend())
    adapter._route_lock = ObservedLock(adapter._route_lock)
    with ThreadPoolExecutor(max_workers=1) as workers:
        assert workers.submit(adapter.route, "broken", ["small"]).result(WAIT_SECONDS) == "small"
        assert workers.submit(adapter.route, "north", ["small", "large"]).result(WAIT_SECONDS) == "large"


def app_for(router):
    app = FastAPI()

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    @app.get("/route")
    async def route(query: str, user: str = "alice"):
        return {"model": await router.select_model(query, user=user)}

    return app


async def assert_health_during_blocked_route(router, entered, release, after_entry=None):
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app_for(router)), base_url="http://test") as client:
        pending = asyncio.create_task(client.get("/route", params={"query": "east", "user": "alice"}))
        try:
            await asyncio.wait_for(asyncio.to_thread(wait_for, entered), WAIT_SECONDS + 1)
            assert not pending.done(), "Blocking routing work ran on the event loop"
            response = await asyncio.wait_for(client.get("/health"), 1)
            assert response.json() == {"status": "ok"}
            assert not pending.done()
            if after_entry is not None:
                after_entry()
        finally:
            release.set()
            response = await asyncio.wait_for(pending, WAIT_SECONDS)
        assert response.status_code == 200 and response.json() == {"model": "small"}


def test_health_remains_responsive_during_blocking_ml_routing(monkeypatch):
    entered, release = threading.Event(), threading.Event()
    worker_threads = []

    class Backend:
        def route_single(self, request):
            worker_threads.append(threading.get_ident())
            entered.set()
            wait_for(release)
            assert request == {"query": "east"}
            return {"model_name": "small"}

    adapter = adapter_for(monkeypatch, Backend())
    router = routers.OpenClawRouter(routing_config("llmrouter"))
    router._llmrouter_adapter = adapter
    loop_thread = threading.get_ident()
    asyncio.run(assert_health_during_blocked_route(router, entered, release))
    assert len(worker_threads) == 1 and worker_threads[0] != loop_thread


@pytest.mark.parametrize("stage", ["retrieve", "record"])
def test_health_remains_responsive_during_real_memory_work(tmp_path, monkeypatch, encoder_downloads, stage):
    state = encoder_downloads
    router = routers.OpenClawRouter(routing_config("llm", memory_config(tmp_path, per_user=True)))
    bank = router._memory_bank
    bank.add("east", "small", strategy="llm", user="alice")
    bank.add("north", "large", strategy="llm", user="bob")
    state.query_calls.clear()
    state.forward_threads.clear()
    entered, release = threading.Event(), threading.Event()
    state.forward_gate = (1, 1 if stage == "retrieve" else 2, entered, release)
    provider_calls = []

    def provider(request):
        provider_calls.append(json.loads(request.content))
        return httpx.Response(200, json={"choices": [{"message": {"content": "small"}}]})

    real_client = httpx.AsyncClient
    monkeypatch.setattr(routers.httpx, "AsyncClient", lambda *args, **kwargs: real_client(
        *args, **({"transport": httpx.MockTransport(provider)} | kwargs),
    ))
    loop_thread = threading.get_ident()

    def check_stage():
        assert len(provider_calls) == (0 if stage == "retrieve" else 1)

    asyncio.run(assert_health_during_blocked_route(router, entered, release, check_stage))
    assert len(provider_calls) == 1
    prompt = provider_calls[0]["messages"][0]["content"]
    assert "'east' -> small" in prompt and "'north' -> large" not in prompt
    assert len(state.forward_threads) == 2 and all(thread != loop_thread for thread in state.forward_threads)
    assert len(bank.path.read_text(encoding="utf-8").splitlines()) == 3
    state.forward_gate = None
    assert {item["user"] for item in bank.retrieve("east", user="alice")} == {"alice"}


def test_parallel_llm_selection_keeps_memory_query_user_and_model_isolated(tmp_path, monkeypatch, encoder_downloads):
    router = routers.OpenClawRouter(routing_config("llm", memory_config(tmp_path, per_user=True)))
    bank = router._memory_bank
    bank.add("east", "small", strategy="llm", user="alice")
    bank.add("north", "large", strategy="llm", user="bob")
    prompts = {}
    both_providers = asyncio.Event()

    async def provider(request):
        prompt = json.loads(request.content)["messages"][0]["content"]
        query = prompt.rsplit("User query: ", 1)[1]
        prompts[query] = prompt
        if len(prompts) == 2:
            both_providers.set()
        await asyncio.wait_for(both_providers.wait(), WAIT_SECONDS)
        return httpx.Response(200, json={"choices": [{"message": {"content": {"east": "small", "north": "large"}[query]}}]})

    real_client = httpx.AsyncClient
    monkeypatch.setattr(routers.httpx, "AsyncClient", lambda *args, **kwargs: real_client(
        *args, **({"transport": httpx.MockTransport(provider)} | kwargs),
    ))

    async def select_both():
        return await asyncio.wait_for(asyncio.gather(
            router.select_model("east", user="alice"),
            router.select_model("north", user="bob"),
        ), WAIT_SECONDS + 1)

    assert asyncio.run(select_both()) == ["small", "large"]
    assert "'east' -> small" in prompts["east"] and "'north' -> large" not in prompts["east"]
    assert "'north' -> large" in prompts["north"] and "'east' -> small" not in prompts["north"]
    rows = [json.loads(line) for line in bank.path.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 4
    assert {(row["query"], row["model"], row["user"], row["strategy"]) for row in rows} == {
        ("east", "small", "alice", "llm"), ("north", "large", "bob", "llm"),
    }
    reloaded = MemoryBank(router.config.memory)
    for user, query, model in [("alice", "east", "small"), ("bob", "north", "large")]:
        results = reloaded.retrieve(query, user=user)
        assert len(results) == 2
        assert {(item["query"], item["model"], item["user"]): item["score"] for item in results} == {
            (query, model, user): pytest.approx(1.),
        }


def test_memory_append_and_index_publication_are_serialized(tmp_path, monkeypatch, encoder_downloads):
    bank = MemoryBank(memory_config(tmp_path))
    observed_lock = ObservedLock(bank._lock)
    bank._lock = observed_lock
    entered, release = threading.Event(), threading.Event()
    open_file = Path.open
    writes = []

    class Writer:
        def __init__(self, wrapped):
            self.wrapped = wrapped

        def __enter__(self):
            self.wrapped.__enter__()
            return self

        def __exit__(self, *args):
            return self.wrapped.__exit__(*args)

        def write(self, line):
            writes.append(json.loads(line))
            if len(writes) == 1:
                entered.set()
                wait_for(release)
            return self.wrapped.write(line)

    def gated_open(path, mode="r", *args, **kwargs):
        opened = open_file(path, mode, *args, **kwargs)
        return Writer(opened) if path == bank.path and mode == "a" else opened

    monkeypatch.setattr(Path, "open", gated_open)
    with ThreadPoolExecutor(max_workers=3) as workers:
        first = workers.submit(bank.add, "east", "small", user="alice")
        try:
            wait_for(entered)
            assert bank._metas == []
            assert bank._embeddings is None
            second = workers.submit(bank.add, "north", "large", user="bob")
            wait_for(observed_lock.contended)
            assert len(writes) == 1 and not second.done()
            snapshot = workers.submit(bank.retrieve, "east")
        finally:
            release.set()
        first.result(WAIT_SECONDS)
        second.result(WAIT_SECONDS)
        results = snapshot.result(WAIT_SECONDS)
    rows = [json.loads(line) for line in bank.path.read_text(encoding="utf-8").splitlines()]
    assert [(row["query"], row["model"], row["user"]) for row in rows] == [
        ("east", "small", "alice"), ("north", "large", "bob"),
    ]
    assert all((item["query"], item["model"]) in {(row["query"], row["model"]) for row in rows} for item in results)
    assert bank.retrieve("east") == MemoryBank(bank.cfg).retrieve("east")


def test_failed_memory_append_does_not_publish_phantom_history(tmp_path, monkeypatch, encoder_downloads):
    bank = MemoryBank(memory_config(tmp_path))
    bank._lock = ObservedLock(bank._lock)
    bank.add("east", "small", user="alice")
    previous_file = bank.path.read_bytes()
    open_file = Path.open

    def failing_open(path, mode="r", *args, **kwargs):
        if path == bank.path and mode == "a":
            raise OSError("history disk unavailable")
        return open_file(path, mode, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(Path, "open", failing_open)
        with pytest.raises(OSError, match="history disk unavailable"):
            bank.add("north", "not-persisted", user="bob")
        assert [item["model"] for item in bank.retrieve("north")] == ["small"]
        assert bank.path.read_bytes() == previous_file
    bank.add("north", "large", user="bob")
    assert bank.retrieve("north") == MemoryBank(bank.cfg).retrieve("north")
    assert [item["model"] for item in bank.retrieve("north")] == ["large", "small"]
