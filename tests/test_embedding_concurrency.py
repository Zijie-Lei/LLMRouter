"""Exercise shared embedding initialization with local tokenizers and Torch."""

from concurrent.futures import ThreadPoolExecutor
import threading
from types import SimpleNamespace

import pytest
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast

from llmrouter.utils import embeddings


TIMEOUT = 5


def wait_for(event, message):
    assert event.wait(TIMEOUT), message


class ObservedLock:
    """Expose a second acquisition attempt without requiring it to succeed."""

    def __init__(self):
        self.lock = threading.Lock()
        self.count_lock = threading.Lock()
        self.attempts = 0
        self.contended = threading.Event()

    def __enter__(self):
        with self.count_lock:
            self.attempts += 1
            if self.attempts == 2:
                self.contended.set()
        assert self.lock.acquire(timeout=TIMEOUT), "Embedding lock acquisition timed out"
        return self

    def __exit__(self, *_args):
        self.lock.release()


class LocalTokenizer:
    """A real fast tokenizer, with an observable shared-state call boundary."""

    def __init__(self):
        backend = Tokenizer(WordLevel(
            {"[PAD]": 0, "[UNK]": 1, "alpha": 2, "beta": 3, "gamma": 4},
            unk_token="[UNK]",
        ))
        backend.pre_tokenizer = Whitespace()
        self.tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=backend, pad_token="[PAD]", unk_token="[UNK]",
        )
        self.calls = []
        self.active = 0
        self.state_lock = threading.Lock()
        self.before_call = None

    def __call__(self, texts, **options):
        with self.state_lock:
            self.active += 1
            active = self.active
            self.calls.append((list(texts), options))
            attempt = len(self.calls)
        try:
            if active != 1:
                raise RuntimeError("Concurrent access to shared tokenizer state")
            if self.before_call is not None:
                self.before_call(attempt)
            return self.tokenizer(texts, **options)
        finally:
            with self.state_lock:
                self.active -= 1


class LocalModel(torch.nn.Module):
    def __init__(self, fail_stage=None, setup_started=None, release_setup=None):
        super().__init__()
        self.register_buffer("vectors", torch.tensor([
            [1000., 1000.], [0., 0.], [2., 4.], [6., 3.], [4., 8.],
        ]))
        self.fail_stage = fail_stage
        self.setup_started = setup_started
        self.release_setup = release_setup
        self.to_calls = []
        self.eval_calls = 0
        self.forward_lock = threading.Lock()
        self.forward_calls = 0
        self.both_forwards = threading.Event()
        self.release_forward = None

    def to(self, device):
        self.to_calls.append(device)
        if self.fail_stage == "to":
            raise RuntimeError("Initialization failed in to")
        return super().to(device)

    def eval(self):
        self.eval_calls += 1
        if self.setup_started is not None:
            self.setup_started.set()
            wait_for(self.release_setup, "Model setup was not released")
        if self.fail_stage == "eval":
            raise RuntimeError("Initialization failed in eval")
        return super().eval()

    def forward(self, input_ids, **_inputs):
        assert not self.training
        assert not torch.is_grad_enabled()
        with self.forward_lock:
            self.forward_calls += 1
            if self.forward_calls == 2:
                self.both_forwards.set()
        if self.release_forward is not None:
            wait_for(self.release_forward, "Concurrent forward passes were not released")
        return SimpleNamespace(last_hidden_state=self.vectors[input_ids])


@pytest.fixture(autouse=True)
def isolated_embedding_cache(monkeypatch):
    for name in ("_tokenizer", "_model", "_device"):
        monkeypatch.setattr(embeddings, name, None)
    monkeypatch.setattr(embeddings, "_init_lock", threading.Lock(), raising=False)
    monkeypatch.setattr(embeddings, "_tokenization_lock", threading.Lock(), raising=False)
    monkeypatch.setenv("LLMROUTER_EMBEDDING_DEVICE", "cpu")
    monkeypatch.setenv("TOKENIZERS_PARALLELISM", "false")


def install_downloads(monkeypatch, before_tokenizer=None, before_model=None, model_factory=None):
    calls = SimpleNamespace(tokenizer_names=[], model_names=[], tokenizers=[], models=[])

    def load_tokenizer(name):
        calls.tokenizer_names.append(name)
        if before_tokenizer is not None:
            before_tokenizer(len(calls.tokenizer_names))
        tokenizer = LocalTokenizer()
        calls.tokenizers.append(tokenizer)
        return tokenizer

    def load_model(name):
        calls.model_names.append(name)
        if before_model is not None:
            before_model(len(calls.model_names))
        model = model_factory(len(calls.model_names)) if model_factory else LocalModel()
        calls.models.append(model)
        return model

    monkeypatch.setattr(embeddings.AutoTokenizer, "from_pretrained", load_tokenizer)
    monkeypatch.setattr(embeddings.AutoModel, "from_pretrained", load_model)
    return calls


def assert_unpublished_cache():
    assert embeddings._tokenizer is None
    assert embeddings._model is None
    assert embeddings._device is None


def test_concurrent_cold_calls_load_once_and_publish_only_ready_model(monkeypatch):
    init_lock = ObservedLock()
    monkeypatch.setattr(embeddings, "_init_lock", init_lock)
    loading_started, release_loading = threading.Event(), threading.Event()
    setup_started, release_setup = threading.Event(), threading.Event()

    def block_loading(_attempt):
        loading_started.set()
        wait_for(release_loading, "Tokenizer loading was not released")

    calls = install_downloads(
        monkeypatch, before_tokenizer=block_loading,
        model_factory=lambda _attempt: LocalModel(setup_started=setup_started, release_setup=release_setup),
    )
    start = threading.Barrier(3)
    queries = ["alpha beta", ["gamma", "beta alpha"]]

    def encode(query):
        start.wait(timeout=TIMEOUT)
        return embeddings.get_longformer_embedding(query)

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(encode, query) for query in queries]
        try:
            start.wait(timeout=TIMEOUT)
            wait_for(loading_started, "No thread started initialization")
            wait_for(init_lock.contended, "Second caller did not wait for initialization")
            assert calls.tokenizer_names == [embeddings._MODEL_NAME]
            assert calls.model_names == []
            assert_unpublished_cache()
            release_loading.set()
            wait_for(setup_started, "Model did not enter eval setup")
            assert calls.model_names == [embeddings._MODEL_NAME]
            assert calls.models[0].training
            assert_unpublished_cache()
            release_setup.set()
            actual = [future.result(timeout=TIMEOUT) for future in futures]
        finally:
            release_loading.set()
            release_setup.set()
            start.abort()

    for query, result in zip(queries, actual):
        torch.testing.assert_close(result, embeddings.get_longformer_embedding(query))
        assert result.device.type == "cpu" and not result.requires_grad
    torch.testing.assert_close(actual[0], torch.tensor([4., 3.5]))
    torch.testing.assert_close(actual[1], torch.tensor([[4., 8.], [4., 3.5]]))
    assert calls.tokenizer_names == calls.model_names == [embeddings._MODEL_NAME]
    assert calls.models[0].to_calls == [torch.device("cpu")]
    assert calls.models[0].eval_calls == 1
    assert embeddings._tokenizer is calls.tokenizers[0]
    assert embeddings._model is calls.models[0]
    assert all(options == {
        "padding": True, "truncation": True, "max_length": 4096, "return_tensors": "pt",
    } for _, options in calls.tokenizers[0].calls)


@pytest.mark.parametrize("failure_stage", ["tokenizer", "model", "to", "eval"])
def test_failed_initialization_does_not_publish_cache_and_retries(monkeypatch, failure_stage):
    def fail_tokenizer(attempt):
        if failure_stage == "tokenizer" and attempt == 1:
            raise RuntimeError("Initialization failed in tokenizer")

    def fail_model(attempt):
        if failure_stage == "model" and attempt == 1:
            raise RuntimeError("Initialization failed in model")

    calls = install_downloads(
        monkeypatch, before_tokenizer=fail_tokenizer, before_model=fail_model,
        model_factory=lambda attempt: LocalModel(fail_stage=failure_stage if attempt == 1 else None),
    )
    with pytest.raises(RuntimeError, match=f"Initialization failed in {failure_stage}"):
        embeddings.get_longformer_embedding("alpha")
    assert_unpublished_cache()

    torch.testing.assert_close(embeddings.get_longformer_embedding("alpha"), torch.tensor([2., 4.]))
    assert len(calls.tokenizer_names) == 2
    assert len(calls.model_names) == (1 if failure_stage == "tokenizer" else 2)
    assert embeddings._tokenizer is calls.tokenizers[-1]
    assert embeddings._model is calls.models[-1]
    assert not embeddings._model.training
    assert embeddings._device == torch.device("cpu")


def test_waiting_caller_retries_after_initializer_failure(monkeypatch):
    init_lock = ObservedLock()
    monkeypatch.setattr(embeddings, "_init_lock", init_lock)
    failing_load_started, release_failure = threading.Event(), threading.Event()

    def fail_first_load(attempt):
        if attempt == 1:
            failing_load_started.set()
            wait_for(release_failure, "Failed model download was not released")
            raise RuntimeError("First download failed")

    calls = install_downloads(monkeypatch, before_model=fail_first_load)
    with ThreadPoolExecutor(max_workers=2) as executor:
        failed = executor.submit(embeddings.get_longformer_embedding, "alpha")
        try:
            wait_for(failing_load_started, "No thread reached the failing download")
            waiting = executor.submit(embeddings.get_longformer_embedding, "gamma")
            wait_for(init_lock.contended, "Second caller did not wait behind the failed download")
            assert_unpublished_cache()
            release_failure.set()
            with pytest.raises(RuntimeError, match="First download failed"):
                failed.result(timeout=TIMEOUT)
            torch.testing.assert_close(waiting.result(timeout=TIMEOUT), torch.tensor([4., 8.]))
        finally:
            release_failure.set()
    assert len(calls.tokenizer_names) == len(calls.model_names) == 2
    assert embeddings._tokenizer is calls.tokenizers[-1]
    assert embeddings._model is calls.models[-1]


def test_tokenization_is_serial_but_model_forwards_can_overlap(monkeypatch):
    calls = install_downloads(monkeypatch)
    embeddings.get_longformer_embedding("alpha")
    tokenizer, model = calls.tokenizers[0], calls.models[0]
    tokenizer.calls.clear()
    model.forward_calls = 0
    model.both_forwards.clear()
    token_lock = ObservedLock()
    monkeypatch.setattr(embeddings, "_tokenization_lock", token_lock)
    tokenization_started, release_tokenization = threading.Event(), threading.Event()
    release_forward = threading.Event()
    model.release_forward = release_forward

    def block_first_tokenization(attempt):
        if attempt == 1:
            tokenization_started.set()
            wait_for(release_tokenization, "First tokenizer call was not released")

    tokenizer.before_call = block_first_tokenization
    start = threading.Barrier(3)
    queries = ["alpha", ["beta gamma", "gamma"]]

    def encode(query):
        start.wait(timeout=TIMEOUT)
        return embeddings.get_longformer_embedding(query)

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(encode, query) for query in queries]
        try:
            start.wait(timeout=TIMEOUT)
            wait_for(tokenization_started, "First tokenizer call did not start")
            wait_for(token_lock.contended, "Second caller did not wait for tokenizer access")
            assert tokenizer.active == 1
            assert len(tokenizer.calls) == 1
            release_tokenization.set()
            wait_for(model.both_forwards, "Model forward passes were unnecessarily serialized")
            release_forward.set()
            actual = [future.result(timeout=TIMEOUT) for future in futures]
        finally:
            release_tokenization.set()
            release_forward.set()
            start.abort()

    tokenizer.before_call = None
    model.release_forward = None
    for query, result in zip(queries, actual):
        torch.testing.assert_close(result, embeddings.get_longformer_embedding(query))
    torch.testing.assert_close(actual[0], torch.tensor([2., 4.]))
    torch.testing.assert_close(actual[1], torch.tensor([[5., 5.5], [4., 8.]]))
    assert len(calls.tokenizer_names) == len(calls.model_names) == 1


def test_invalid_device_selection_leaves_empty_cache_and_can_retry(monkeypatch):
    calls = install_downloads(monkeypatch)
    monkeypatch.setenv("LLMROUTER_EMBEDDING_DEVICE", "invalid-device")
    with pytest.raises(RuntimeError):
        embeddings.get_longformer_embedding("alpha")
    assert_unpublished_cache()
    assert calls.tokenizer_names == calls.model_names == []
    monkeypatch.setenv("LLMROUTER_EMBEDDING_DEVICE", "cpu")
    torch.testing.assert_close(embeddings.get_longformer_embedding("alpha"), torch.tensor([2., 4.]))


def test_initialization_preserves_previously_cached_device_selection(monkeypatch):
    calls = install_downloads(monkeypatch)
    assert embeddings._get_device() == torch.device("cpu")
    monkeypatch.setenv("LLMROUTER_EMBEDDING_DEVICE", "invalid-device")
    torch.testing.assert_close(embeddings.get_longformer_embedding(["alpha"]), torch.tensor([2., 4.]))
    assert calls.models[0].to_calls == [torch.device("cpu")]
