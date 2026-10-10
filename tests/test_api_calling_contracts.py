"""Real API-call orchestration with the LiteLLM boundary kept offline."""

import copy
import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from llmrouter.utils import api_calling


def provider_response(content="provider answer", usage=True):
    tokens = SimpleNamespace(prompt_tokens=3, completion_tokens=2, total_tokens=5)
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
        usage=tokens if usage else None,
    )


@pytest.fixture
def upstream(monkeypatch):
    completion = Mock(return_value=provider_response())
    monkeypatch.setattr(api_calling, "completion", completion)
    monkeypatch.setattr(api_calling, "_api_key_counters", {})
    # Exercise the supported whitespace fallback without downloading a tokenizer.
    monkeypatch.setattr(api_calling, "GPT2TokenizerFast", None)
    clock = iter(10 + i / 4 for i in range(100))
    monkeypatch.setattr(api_calling, "time", SimpleNamespace(time=lambda: next(clock)))
    monkeypatch.delenv("API_KEYS", raising=False)
    return completion


@pytest.fixture
def request_data():
    return {
        "api_endpoint": "https://models.example.test/v1",
        "query": "Explain recursion",
        "model_name": "selected-model",
        "api_name": "vendor/backend-model",
        "request_id": "request-1",
        "metadata": {"group": "offline-test"},
    }


def test_success_forwards_endpoint_model_messages_and_generation_parameters(upstream, request_data):
    request_data["system_prompt"] = "Keep the answer short"
    before = copy.deepcopy(request_data)

    result = api_calling.call_api(
        request_data, api_keys_env="fake-key", max_tokens=91,
        temperature=0.25, top_p=0.8, timeout=17,
    )

    upstream.assert_called_once_with(
        model="openai/vendor/backend-model",
        messages=[
            {"role": "system", "content": "Keep the answer short"},
            {"role": "user", "content": "Explain recursion"},
        ],
        api_key="fake-key", api_base="https://models.example.test/v1",
        max_tokens=91, temperature=0.25, top_p=0.8, timeout=17,
    )
    assert request_data == before
    assert result is not request_data
    for key, value in before.items():
        assert result[key] == value
    assert result["response"] == "provider answer"
    assert result["prompt_tokens"] == 3
    assert result["completion_tokens"] == 2
    assert result["token_num"] == 5
    assert result["response_time"] == 0.25
    assert "error" not in result
    result["query"] = "changed output"
    assert request_data["query"] == "Explain recursion"


@pytest.mark.parametrize("encoded_keys", [
    "fake-one,fake-two", '["fake-one", "fake-two"]', '"fake-one"', "fake-one",
])
def test_supported_key_formats_reach_the_provider(upstream, request_data, encoded_keys):
    api_calling.call_api(request_data, api_keys_env=encoded_keys)

    assert upstream.call_args.kwargs["api_key"] == "fake-one"
    assert upstream.call_args.kwargs["messages"] == [{"role": "user", "content": "Explain recursion"}]


def test_explicit_key_override_takes_precedence_over_environment(upstream, request_data, monkeypatch):
    monkeypatch.setenv("API_KEYS", "fake-environment-key")

    api_calling.call_api(request_data, api_keys_env="fake-override-key")
    api_calling.call_api(request_data)

    assert [call.kwargs["api_key"] for call in upstream.call_args_list] == [
        "fake-override-key", "fake-environment-key",
    ]


def test_single_key_rotation_is_independent_per_endpoint_and_model(upstream, request_data):
    other_model = dict(request_data, api_name="vendor/other-model")
    other_endpoint = dict(request_data, api_endpoint="https://other.example.test/v1")
    for request in [request_data, request_data, other_model, other_endpoint, request_data]:
        api_calling.call_api(request, api_keys_env="fake-one,fake-two")

    assert [call.kwargs["api_key"] for call in upstream.call_args_list] == [
        "fake-one", "fake-two", "fake-one", "fake-one", "fake-one",
    ]


def test_service_specific_keys_are_case_insensitive_and_rotate_independently(upstream, request_data):
    keys = json.dumps({"ProviderA": ["fake-a1", "fake-a2"], "ProviderB": "fake-b1,fake-b2"})
    for service in ["providera", "PROVIDERA", "ProviderB", "ProviderA"]:
        api_calling.call_api(dict(request_data, service=service), api_keys_env=keys)

    assert [call.kwargs["api_key"] for call in upstream.call_args_list] == [
        "fake-a1", "fake-a2", "fake-b1", "fake-a1",
    ]


@pytest.mark.parametrize("endpoint", ["http://localhost:9000/v1", "http://127.0.0.1:9000/v1"])
def test_local_service_accepts_empty_key_without_network_access(upstream, request_data, endpoint):
    result = api_calling.call_api(
        dict(request_data, api_endpoint=endpoint, service="local"),
        api_keys_env='{"Local": ""}',
    )

    assert "error" not in result
    assert upstream.call_args.kwargs["api_key"] == "local"
    assert upstream.call_args.kwargs["api_base"] == endpoint


def test_remote_service_filters_empty_keys(upstream, request_data):
    api_calling.call_api(
        dict(request_data, service="Provider"),
        api_keys_env='{"Provider": ["", "fake-valid"]}',
    )

    assert upstream.call_args.kwargs["api_key"] == "fake-valid"


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="Known bug: substring-based localhost detection allows empty keys for remote URLs",
)
@pytest.mark.parametrize("endpoint", [
    "https://localhost.example.test/v1", "https://models.example.test/v1/localhost",
])
def test_remote_urls_containing_localhost_do_not_qualify_for_empty_local_keys(upstream, request_data, endpoint):
    result = api_calling.call_api(
        dict(request_data, api_endpoint=endpoint, service="local"),
        api_keys_env='{"Local": ""}',
    )

    upstream.assert_not_called()
    assert "No valid API keys available" in result["error"]


@pytest.mark.parametrize("service,encoded_keys,error_fragment", [
    (None, '{"Provider": "fake-key"}', "Service provider name is required"),
    ("unknown", '{"Provider": "fake-key"}', "not found in API_KEYS"),
    ("Provider", '{"Provider": []}', "No API keys available"),
    ("Provider", '{"Provider": ""}', "No valid API keys available"),
])
def test_service_key_errors_are_mapped_before_provider_call(
    upstream, request_data, service, encoded_keys, error_fragment,
):
    result = api_calling.call_api(dict(request_data, service=service), api_keys_env=encoded_keys)

    upstream.assert_not_called()
    assert error_fragment in result["error"]
    assert result["query"] == request_data["query"]
    assert result["model_name"] == request_data["model_name"]
    assert result["prompt_tokens"] == result["completion_tokens"] == result["token_num"] == 0


def test_batch_preserves_order_and_continues_after_provider_failure(upstream, request_data):
    requests = [dict(request_data, query=query, request_id=str(i)) for i, query in enumerate(["first", "failed", "last"])]
    before = copy.deepcopy(requests)
    upstream.side_effect = [provider_response("first answer"), RuntimeError("provider unavailable"), provider_response("last answer")]

    results = api_calling.call_api(requests, api_keys_env="fake-one,fake-two")
    upstream.side_effect = None
    api_calling.call_api(request_data, api_keys_env="fake-one,fake-two")

    assert requests == before
    assert isinstance(results, list)
    assert [result["query"] for result in results] == ["first", "failed", "last"]
    assert [result["request_id"] for result in results] == ["0", "1", "2"]
    assert results[0]["response"] == "first answer"
    assert results[1]["error"] == "provider unavailable"
    assert results[2]["response"] == "last answer"
    assert [call.kwargs["api_key"] for call in upstream.call_args_list] == [
        "fake-one", "fake-two", "fake-one", "fake-one",
    ]


def test_usage_free_response_uses_offline_token_estimation(upstream, request_data):
    upstream.return_value = provider_response("All good", usage=False)
    request_data["query"] = "How are you"

    result = api_calling.call_api(request_data, api_keys_env="fake-key")

    assert result["prompt_tokens"] == 3
    assert result["completion_tokens"] == 2
    assert result["token_num"] == 5
    assert "error" not in result


def test_zero_provider_usage_is_preserved(upstream, request_data):
    upstream.return_value.usage = SimpleNamespace(prompt_tokens=0, completion_tokens=0, total_tokens=0)

    result = api_calling.call_api(request_data, api_keys_env="fake-key")

    assert result["prompt_tokens"] == result["completion_tokens"] == result["token_num"] == 0
    assert "error" not in result


@pytest.mark.parametrize("failure", [RuntimeError("provider unavailable"), TimeoutError("provider timed out"), RuntimeError("x" * 250)])
def test_provider_failure_preserves_request_and_full_error(upstream, request_data, failure):
    upstream.side_effect = failure
    before = copy.deepcopy(request_data)

    result = api_calling.call_api(request_data, api_keys_env="fake-key")

    assert request_data == before
    for key, value in before.items():
        assert result[key] == value
    assert result["error"] == str(failure)
    assert result["response"] == f"API Error: {str(failure)[:200]}"
    assert result["prompt_tokens"] == result["completion_tokens"] == result["token_num"] == 0
    assert result["response_time"] == 0.25


def test_missing_litellm_dependency_raises_before_processing(upstream, request_data, monkeypatch):
    monkeypatch.setattr(api_calling, "completion", None)

    with pytest.raises(ImportError, match="litellm"):
        # API_KEYS is absent: parsing first would raise ValueError instead.
        api_calling.call_api(request_data)


def test_missing_api_keys_raises_before_provider_call(upstream, request_data):
    with pytest.raises(ValueError, match="API_KEYS environment variable is not set"):
        api_calling.call_api(request_data)

    upstream.assert_not_called()


def test_invalid_batch_is_validated_before_any_provider_call(upstream, request_data):
    invalid = dict(request_data)
    invalid.pop("api_name")

    with pytest.raises(ValueError, match="Missing required keys"):
        api_calling.call_api([request_data, invalid], api_keys_env="fake-key")

    upstream.assert_not_called()
