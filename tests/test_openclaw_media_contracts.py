"""Offline media parsing and request contracts through real HTTP serialization."""

import asyncio
import base64
import copy
from email import policy
from email.parser import BytesParser
import json

import httpx
import pytest

from openclaw_router import media
from openclaw_router.config import OpenClawConfig


@pytest.fixture
def media_config(monkeypatch, tmp_path):
    monkeypatch.delenv("LLMROUTER_TEST_MEDIA_KEY", raising=False)
    monkeypatch.setattr(media.tempfile, "tempdir", str(tmp_path))
    return media.MediaConfig(
        enabled=True, api_key="configured-test-key", api_key_env="LLMROUTER_TEST_MEDIA_KEY",
        base_url="https://media.example.invalid/v1", vision_model="tiny-vision",
        audio_model="tiny-whisper", image_prompt="Describe the visible objects.",
    )


@pytest.fixture
def provider_transport(monkeypatch):
    original_client = httpx.AsyncClient

    def install(response_body=None, *, status=200, error=None):
        requests = []

        def handler(request):
            requests.append(request)
            if error:
                raise error
            return httpx.Response(status, json=response_body)

        transport = httpx.MockTransport(handler)
        monkeypatch.setattr(media.httpx, "AsyncClient", lambda **kwargs: original_client(transport=transport, **kwargs))
        return requests

    return install


def run(coroutine):
    return asyncio.run(coroutine)


def vision_response(text):
    return {"choices": [{"message": {"content": text}}]}


@pytest.mark.parametrize("content, expected", [
    ("text remains unchanged", "text remains unchanged"),
    (["first", {"type": "text", "text": "second"}, {"type": "image", "data": "ignored"}], "first\nsecond"),
    (None, ""),
])
def test_disabled_media_extracts_text_without_provider_calls(media_config, provider_transport, content, expected):
    media_config.enabled = False
    requests = provider_transport(vision_response("unused"))
    original = copy.deepcopy(content)
    assert run(media.process_multimodal_content(content, media_config)) == (expected, None)
    assert content == original
    assert requests == []


@pytest.mark.parametrize("cfg_key, env_key, fallback, expected", [
    ("config-key", "environment-key", "fallback-key", "config-key"),
    (None, "environment-key", "fallback-key", "environment-key"),
    (None, None, "fallback-key", "fallback-key"),
    (None, None, None, None),
])
def test_image_credentials_use_config_environment_then_fallback(media_config, provider_transport, monkeypatch, cfg_key, env_key, fallback, expected):
    media_config.api_key = cfg_key
    if env_key:
        monkeypatch.setenv(media_config.api_key_env, env_key)
    requests = provider_transport(vision_response("a lighthouse"))
    result = run(media.describe_image("encoded-image", media_config, fallback_key=fallback))
    if expected is None:
        assert result == "[Image: API key not configured]"
        assert requests == []
    else:
        assert result == "[Image: a lighthouse]"
        assert requests[0].headers["authorization"] == f"Bearer {expected}"


@pytest.mark.parametrize("is_url, source, expected_url", [
    (False, "cG5nLWJ5dGVz", "data:image/png;base64,cG5nLWJ5dGVz"),
    (True, "https://images.example.invalid/picture.png", "https://images.example.invalid/picture.png"),
])
def test_image_provider_request_preserves_model_prompt_and_media_source(media_config, provider_transport, is_url, source, expected_url):
    requests = provider_transport(vision_response("a lighthouse"))
    assert run(media.describe_image(source, media_config, "image/png", is_url)) == "[Image: a lighthouse]"
    assert len(requests) == 1
    request = requests[0]
    assert str(request.url) == "https://media.example.invalid/v1/chat/completions"
    assert request.method == "POST"
    assert json.loads(request.content) == {
        "model": "tiny-vision", "max_tokens": 256, "temperature": .3,
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": "Describe the visible objects."},
            {"type": "image_url", "image_url": {"url": expected_url}},
        ]}],
    }


@pytest.mark.parametrize("status, body, expected", [
    (429, {}, "[Image: Vision API error 429]"),
    (200, {}, "[Image: No description available]"),
    (200, vision_response(""), "[Image: No description available]"),
])
def test_image_provider_errors_and_empty_descriptions(media_config, provider_transport, status, body, expected):
    provider_transport(body, status=status)
    assert run(media.describe_image("encoded", media_config)) == expected


def test_image_transport_exceptions_return_bounded_error_text(media_config, provider_transport):
    provider_transport(error=httpx.ReadTimeout("timeout " + "x" * 200))
    result = run(media.describe_image("encoded", media_config))
    assert result.startswith("[Image: Error - timeout ")
    assert len(result) <= len("[Image: Error - ]") + 100


@pytest.mark.parametrize("limit, expected", [(5, "[Image: abcde...]"), (0, "[Image: abcdefgh]")])
def test_image_description_limit_is_applied_to_provider_text(media_config, provider_transport, limit, expected):
    media_config.max_description_chars = limit
    provider_transport(vision_response("abcdefgh"))
    assert run(media.describe_image("encoded", media_config)) == expected


@pytest.mark.parametrize("format_name", ["text-placeholder", "list-placeholder", "data-url", "direct-image", "url-image", "image-only"])
def test_image_content_formats_use_real_file_loading_and_provider_requests(tmp_path, media_config, provider_transport, format_name):
    image_bytes = b"\x89PNG\r\n\x1a\nlocal-image"
    image_path = tmp_path / "image.png"
    image_path.write_bytes(image_bytes)
    encoded = base64.b64encode(image_bytes).decode("ascii")
    placeholder = f"What is visible? [media attached: {image_path} (image/png)]"
    if format_name == "text-placeholder":
        content = placeholder
    elif format_name == "list-placeholder":
        content = [{"type": "text", "text": placeholder}]
    else:
        source = f"data:image/png;base64,{encoded}"
        part = {"type": "image_url", "image_url": {"url": source}}
        if format_name == "direct-image":
            part = {"type": "image", "data": encoded, "mimeType": "image/png"}
        elif format_name == "url-image":
            part["image_url"]["url"] = "https://images.example.invalid/picture.png"
        content = [part] if format_name == "image-only" else [{"type": "text", "text": "What is visible?"}, part]
    original = copy.deepcopy(content)
    requests = provider_transport(vision_response("a lighthouse"))
    description = "[Image: a lighthouse]"
    expected_text = description if format_name == "image-only" else f"What is visible?\n\n{description}"
    assert run(media.process_multimodal_content(content, media_config)) == (expected_text, description)
    assert content == original
    sent_media = json.loads(requests[0].content)["messages"][0]["content"][1]["image_url"]["url"]
    assert sent_media == ("https://images.example.invalid/picture.png" if format_name == "url-image" else f"data:image/png;base64,{encoded}")


@pytest.mark.parametrize("as_list", [False, True])
def test_missing_local_image_falls_back_to_attachment_url(tmp_path, media_config, provider_transport, as_list):
    text = f"Describe it [media attached: {tmp_path / 'missing.png'} (image/png) | https://images.example.invalid/remote.png]"
    content = [{"type": "text", "text": text}] if as_list else text
    requests = provider_transport(vision_response("a tree"))
    assert run(media.process_multimodal_content(content, media_config)) == ("Describe it\n\n[Image: a tree]", "[Image: a tree]")
    sent_image = json.loads(requests[0].content)["messages"][0]["content"][1]
    assert sent_image["image_url"]["url"] == "https://images.example.invalid/remote.png"


@pytest.mark.parametrize("file_contents", [None, b"<!DOCTYPE html><html>not an image</html>", b"<html>not an image</html>"])
def test_unreadable_or_html_attachments_preserve_text_and_report_loading_failure(tmp_path, media_config, provider_transport, file_contents):
    path = tmp_path / "invalid.png"
    if file_contents is not None:
        path.write_bytes(file_contents)
    content = f"Original text [media attached: {path} (image/png)]"
    requests = provider_transport(vision_response("unused"))
    assert run(media.process_multimodal_content(content, media_config)) == (content, f"[Image: Could not load file {path}]")
    assert requests == []


@pytest.mark.parametrize("content, expected", [
    ("ordinary text", "ordinary text"),
    ([5, {"type": "image_url", "image_url": {"url": "data:image/png;base64"}}, {"type": "text", "text": "visible text"}], "visible text"),
    ([{"type": "image_url", "image_url": {"url": "data:application/pdf;base64,ignored"}}], ""),
    ([{"type": "unsupported", "data": "ignored"}, "text"], "text"),
])
def test_non_media_or_malformed_content_does_not_call_provider(media_config, provider_transport, content, expected):
    requests = provider_transport(vision_response("unused"))
    assert run(media.process_multimodal_content(content, media_config)) == (expected, None)
    assert requests == []


@pytest.mark.parametrize("mime, extension", [("audio/wav", ".wav"), ("audio/ogg", ".ogg"), ("audio/mp4", ".m4a"), (None, ".mp3")])
def test_audio_upload_contains_real_bytes_and_cleans_temporary_file(tmp_path, media_config, provider_transport, mime, extension):
    audio_bytes = b"RIFF\x00\x01offline-audio"
    requests = provider_transport({"text": "hello world"})
    result = run(media.transcribe_audio(base64.b64encode(audio_bytes).decode("ascii"), media_config, mime, language="en"))
    assert result == "[Audio transcript: hello world]"
    request = requests[0]
    assert str(request.url) == "https://media.example.invalid/v1/audio/transcriptions"
    assert request.headers["authorization"] == "Bearer configured-test-key"
    mime_message = BytesParser(policy=policy.default).parsebytes(
        b"Content-Type: " + request.headers["content-type"].encode("ascii") + b"\r\n\r\n" + request.content,
    )
    parts = {part.get_param("name", header="content-disposition"): part for part in mime_message.iter_parts()}
    assert parts["file"].get_filename() == f"audio{extension}"
    assert parts["file"].get_content_type() == (mime or "audio/mp3")
    assert parts["file"].get_payload(decode=True) == audio_bytes
    assert parts["model"].get_content() == "tiny-whisper"
    assert parts["language"].get_content() == "en"
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("status, body, expected", [
    (503, {}, "[Audio: Transcription API error 503]"),
    (200, {}, "[Audio: No speech detected]"),
    (200, {"text": "abcdefgh"}, "[Audio transcript: abcde...]"),
])
def test_audio_response_errors_empty_transcripts_and_length_limits(tmp_path, media_config, provider_transport, status, body, expected):
    media_config.max_description_chars = 5
    provider_transport(body, status=status)
    assert run(media.transcribe_audio("c291bmQ=", media_config)) == expected
    assert list(tmp_path.iterdir()) == []


def test_audio_transport_failure_still_cleans_temporary_file(tmp_path, media_config, provider_transport):
    provider_transport(error=httpx.ReadTimeout("transcription timeout"))
    assert run(media.transcribe_audio("c291bmQ=", media_config)) == "[Audio: Error - transcription timeout]"
    assert list(tmp_path.iterdir()) == []


def test_audio_content_is_transcribed_and_combined_with_original_text(media_config, provider_transport):
    requests = provider_transport({"text": "hello world"})
    content = [{"type": "text", "text": "Summarize this"}, {"type": "input_audio", "data": "c291bmQ=", "mime_type": "audio/wav"}]
    assert run(media.process_multimodal_content(content, media_config)) == (
        "Summarize this\n\n[Audio transcript: hello world]", "[Audio transcript: hello world]",
    )
    assert requests[0].url.path == "/v1/audio/transcriptions"


def test_audio_and_video_missing_keys_or_dependencies_skip_provider(media_config, provider_transport, monkeypatch):
    requests = provider_transport({"text": "unused"})
    media_config.api_key = None
    assert run(media.transcribe_audio("c291bmQ=", media_config)) == "[Audio: API key not configured]"
    monkeypatch.setattr(media, "HAS_CV2", True)
    assert run(media.describe_video("dmlkZW8=", media_config)) == "[Video: API key not configured]"
    monkeypatch.setattr(media, "HAS_CV2", False)
    assert run(media.describe_video("dmlkZW8=", media_config)) == "[Video: opencv-python not installed, cannot process video]"
    assert media._extract_video_frames("unopened-video.mp4") == []
    assert requests == []


def test_invalid_audio_and_video_base64_return_errors_without_creating_files(tmp_path, media_config, provider_transport, monkeypatch):
    requests = provider_transport({"text": "unused"})
    assert run(media.transcribe_audio("!invalid-base64!", media_config)).startswith("[Audio: Error - ")
    monkeypatch.setattr(media, "HAS_CV2", True)
    assert run(media.describe_video("!invalid-base64!", media_config)).startswith("[Video: Error - ")
    assert requests == []
    assert list(tmp_path.iterdir()) == []


def test_yaml_media_configuration_controls_real_request_and_description_limit(tmp_path, monkeypatch, provider_transport):
    monkeypatch.setenv("LLMROUTER_TEST_MEDIA_KEY", "yaml-test-key")
    path = tmp_path / "config.yaml"
    path.write_text(
        "media:\n  enabled: 'true'\n  api_key_env: LLMROUTER_TEST_MEDIA_KEY\n"
        "  base_url: https://media.example.invalid/v1\n  vision_model: yaml-vision\n"
        "  image_prompt: Count the objects.\n  max_description_chars: 3\n",
        encoding="utf-8",
    )
    config = OpenClawConfig.from_yaml(str(path))
    requests = provider_transport(vision_response("abcdef"))
    assert run(media.describe_image("encoded", config.media)) == "[Image: abc...]"
    body = json.loads(requests[0].content)
    assert body["model"] == "yaml-vision"
    assert body["messages"][0]["content"][0]["text"] == "Count the objects."
    assert requests[0].headers["authorization"] == "Bearer yaml-test-key"


@pytest.mark.xfail(strict=True, raises=AssertionError, reason="Known bug: standard nested OpenAI input_audio data/format are ignored during media detection")
def test_standard_openai_input_audio_is_transcribed(media_config, provider_transport):
    requests = provider_transport({"text": "hello world"})
    content = [
        {"type": "text", "text": "Summarize this"},
        {"type": "input_audio", "input_audio": {"data": "c291bmQ=", "format": "wav"}},
    ]
    assert run(media.process_multimodal_content(content, media_config)) == (
        "Summarize this\n\n[Audio transcript: hello world]", "[Audio transcript: hello world]",
    )
    assert len(requests) == 1
