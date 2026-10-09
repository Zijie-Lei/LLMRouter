"""Exercise template discovery, precedence, metadata and real YAML parsing."""

import pytest
import yaml

from llmrouter import prompts


@pytest.fixture
def prompt_dirs(tmp_path, monkeypatch):
    builtin = tmp_path / "builtin"
    custom = tmp_path / "custom"
    builtin.mkdir()
    monkeypatch.setattr(prompts, "_PROMPTS_DIR", builtin)
    monkeypatch.setattr(prompts, "_CUSTOM_TASKS_DIR", custom)
    return builtin, custom


def write_template(root, name, **data):
    path = root / f"{name}.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return path


@pytest.mark.parametrize("name", ["example", "task_prompts/example"])
def test_custom_template_overrides_builtin_and_keeps_metadata(prompt_dirs, name):
    builtin, custom = prompt_dirs
    write_template(builtin, "task_prompts/example", template="builtin {query}", origin="builtin")
    write_template(custom, "task_prompts/example", template="custom {query}", origin="custom")

    assert prompts.load_prompt_template(name).format(query="hello") == "custom hello"
    assert prompts.load_prompt_template_with_metadata(name) == {
        "template": "custom {query}", "origin": "custom",
    }


@pytest.mark.parametrize("custom_exists", [False, True])
@pytest.mark.parametrize("name", ["example", "task_prompts/example"])
def test_builtin_template_is_used_when_custom_template_is_absent(prompt_dirs, custom_exists, name):
    builtin, custom = prompt_dirs
    if custom_exists:
        custom.mkdir()
    write_template(builtin, "task_prompts/example", template="builtin {query}", origin="builtin")

    assert prompts.load_prompt_template(name) == "builtin {query}"
    assert prompts.load_prompt_template_with_metadata(name)["origin"] == "builtin"


@pytest.mark.parametrize("loader", [prompts.load_prompt_template, prompts.load_prompt_template_with_metadata])
def test_discovery_skips_hidden_and_cache_directories(prompt_dirs, loader):
    builtin, custom = prompt_dirs
    for root in (builtin, custom):
        write_template(root, ".hidden/example", template="hidden")
        write_template(root, "__pycache__/example", template="cached")

    with pytest.raises(FileNotFoundError, match="example.yaml"):
        loader("example")


@pytest.mark.parametrize("root_index", [0, 1])
def test_text_loader_rejects_yaml_without_template_key(prompt_dirs, root_index):
    root = prompt_dirs[root_index]
    write_template(root, "task_prompts/invalid", description="Missing template")

    with pytest.raises(ValueError, match="template.*key"):
        prompts.load_prompt_template("invalid")


def test_text_loader_does_not_hide_yaml_syntax_errors(prompt_dirs):
    builtin, _ = prompt_dirs
    (builtin / "invalid.yaml").write_text("template: [unterminated", encoding="utf-8")

    with pytest.raises(yaml.YAMLError):
        prompts.load_prompt_template("invalid")


@pytest.mark.parametrize("loader", [prompts.load_prompt_template, prompts.load_prompt_template_with_metadata])
def test_missing_template_error_identifies_search_locations(prompt_dirs, loader):
    builtin, custom = prompt_dirs
    custom.mkdir()

    with pytest.raises(FileNotFoundError) as error:
        loader("does-not-exist")

    assert "does-not-exist.yaml" in str(error.value)
    assert str(builtin) in str(error.value)
    assert str(custom) in str(error.value)
