"""Exercise the existing dataset converter with real files and fake LLM calls."""

import importlib
import json
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest


@pytest.fixture
def converter(monkeypatch):
    # The script prepends a path on import; don't change other tests' resolution.
    monkeypatch.setattr(sys, "path", sys.path.copy())
    module = importlib.import_module("llmrouter.utils.data_convert")
    monkeypatch.delenv("OPENAI_API_BASE", raising=False)
    monkeypatch.delenv("NVIDIA_API_BASE", raising=False)
    return module


def write_jsonl(path, rows):
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


@pytest.mark.parametrize("name,expected", [
    ("", ""),
    ("QWEN2.5-7B-INSTRUCT", "qwen/qwen2.5-7b-instruct"),
    ("hosted/llama-3.1-70b-instruct", "meta/llama-3.1-70b-instruct"),
    ("llama3-chatqa-1.5-8b", "meta/llama-3.1-8b-instruct"),
    ("qwen-new", "qwen/qwen-new"),
    ("llama-new", "meta/llama-new"),
    ("mistral-new", "mistralai/mistral-new"),
    ("gemma-new", "google/gemma-new"),
    ("provider/custom", "provider/custom"),
])
def test_model_alias_normalization(converter, name, expected):
    assert converter.normalize_model_name(name) == expected


@pytest.mark.parametrize("model,input_tokens,output_tokens,expected", [
    ("qwen/qwen2.5-7b-instruct", 1000, 500, 0.15),
    ("meta/llama-3.1-70b-instruct", 1000, 500, 1.05),
    ("google/gemma-2-9b-it", 1000, 0, 0.2),
    ("unknown", 0, 2000, 0.2),
    ("unknown", 0, 0, 0.0),
])
def test_cost_uses_input_and_output_token_counts(converter, model, input_tokens, output_tokens, expected):
    assert converter.calculate_cost(input_tokens, output_tokens, model) == pytest.approx(expected)


@pytest.mark.parametrize("raw,expected", [
    (None, None), ("null", None), ("None", None), ("", None),
    ("{'A': 'one', 'B': 'two'}", {"A": "one", "B": "two"}),
    (["one", "two"], ["one", "two"]),
    ("not valid", None), ("__import__('os').getcwd()", None),
])
def test_choice_parser_accepts_literals_and_rejects_executable_text(converter, raw, expected):
    assert converter.parse_choices(raw) == expected


@pytest.mark.parametrize("ground_truth,expected", [("四", ["四"]), (["four", "4"], ["four", "4"]), (4, ["4"])])
def test_single_record_preserves_answer_schema_prompt_and_identity(converter, ground_truth, expected):
    original = {"query": "二加二?", "ground_truth": ground_truth, "task_name": "gsm8k"}
    snapshot = json.loads(json.dumps(original))
    result = converter.convert_single_record(original, 7)

    assert original == snapshot
    assert result["id"] == "gsm8k_7"
    assert result["question"] == "二加二?"
    assert result["golden_answers"] == expected
    assert result["reward_model"] == {"ground_truth": {"target": expected}, "style": "rule"}
    assert result["data_source"] == "gsm8k"
    assert result["ability"] == "math-reasoning"
    assert result["extra_info"] == {"index": 7, "split": "test"}
    assert result["prompt"][0]["role"] == "user"
    assert "二加二?" in result["prompt"][0]["content"]


@pytest.mark.parametrize("task,expected", [
    ("mmlu", "fact-reasoning"), ("gsm8k", "math-reasoning"),
    ("arc_challenge", "commonsense-reasoning"), ("human_eval", "code-generation"),
    ("agentverse-logicgrid", "logical-reasoning"), ("unrecognized", "fact-reasoning"),
])
def test_dataset_ability_classification(converter, task, expected):
    assert converter.determine_ability(task, "a query") == expected


@pytest.mark.parametrize("environment", ["OPENAI_API_BASE", "NVIDIA_API_BASE", None])
def test_ability_provider_receives_bounded_prompt_and_configured_endpoint(converter, monkeypatch, environment):
    calls = []

    def complete(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="  CODE-GENERATION  "))])

    monkeypatch.setattr(converter, "completion", complete)
    if environment:
        monkeypatch.setenv(environment, "https://ability.example.test/v1")
    result = converter.call_llm_for_ability("Q" * 200 + "TRUNCATED", "mbpp", "test-key", api_base=None if environment else "https://ability.example.test/v1")

    assert result == "code-generation"
    assert len(calls) == 1
    assert calls[0]["api_base"] == "https://ability.example.test/v1"
    assert calls[0]["api_key"] == "test-key"
    assert calls[0]["model"] == "meta/llama-3.1-8b-instruct"
    assert calls[0]["max_tokens"] == 50
    assert calls[0]["temperature"] == 0.1
    assert "Q" * 200 in calls[0]["messages"][0]["content"]
    assert "TRUNCATED" not in calls[0]["messages"][0]["content"]


def test_missing_ability_endpoint_raises_before_provider_call(converter, monkeypatch):
    def forbidden(**kwargs):
        pytest.fail("Provider must not be called without an endpoint")

    monkeypatch.setattr(converter, "completion", forbidden)
    with pytest.raises(ValueError, match="API endpoint"):
        converter.call_llm_for_ability("question", "gsm8k", "test-key")


def test_provider_error_falls_back_to_dataset_ability(converter, monkeypatch, capsys):
    def fail(**kwargs):
        raise RuntimeError("provider unavailable")

    monkeypatch.setattr(converter, "completion", fail)
    assert converter.call_llm_for_ability("question", "gsm8k", "test-key", "https://example.test/v1") == "math-reasoning"
    assert "provider unavailable" in capsys.readouterr().out


def test_unrecognized_provider_ability_uses_default(converter, monkeypatch):
    monkeypatch.setattr(converter, "completion", lambda **kwargs: SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="not a category"))]))
    assert converter.call_llm_for_ability("question", "gsm8k", "key", "https://example.test/v1") == "fact-reasoning"


@pytest.mark.parametrize("mode", ["test", "train"])
def test_conversion_start_offset_skips_invalid_json(converter, tmp_path, capsys, mode):
    source, output = tmp_path / "in.jsonl", tmp_path / "out.json"
    source.write_text('\n'.join([
        json.dumps({"query": "outside before"}),
        json.dumps({"query": "selected 一", "ground_truth": "a", "task_name": "mmlu"}, ensure_ascii=False),
        "invalid json", json.dumps({"query": "selected two", "ground_truth": ["b"]}),
    ]) + '\n', encoding="utf-8")
    if mode == "test":
        converter.convert_data(str(source), str(output), start_line=2)
        records = read_jsonl(output)
        assert [r["question"] for r in records] == ["selected 一", "selected two"]
        assert [r["extra_info"]["index"] for r in records] == [1, 2]
    else:
        converter.convert_train_data(str(source), str(output), start_line=2)
        records = json.loads(output.read_text(encoding="utf-8"))
        assert [r["query"] for r in records] == ["selected 一", "selected two"]
    assert "Skipping invalid JSON line 3" in capsys.readouterr().out
    assert "一" in output.read_text(encoding="utf-8")


@pytest.mark.parametrize("mode", ["test", "train"])
@pytest.mark.xfail(strict=True, raises=AssertionError, reason="Existing defect: converters include the end line although CLI help specifies an exclusive end line")
def test_conversion_obeys_documented_exclusive_end_line(converter, tmp_path, mode):
    source, output = tmp_path / "in.jsonl", tmp_path / "out.json"
    write_jsonl(source, [{"query": "first"}, {"query": "excluded"}, {"query": "also excluded"}])
    if mode == "test":
        converter.convert_data(str(source), str(output), start_line=1, end_line=2)
        assert [row["question"] for row in read_jsonl(output)] == ["first"]
    else:
        converter.convert_train_data(str(source), str(output), start_line=1, end_line=2)
        assert [row["query"] for row in json.loads(output.read_text())] == ["first"]


def test_bad_test_record_does_not_abort_later_records(converter, tmp_path, capsys):
    source, output = tmp_path / "in.jsonl", tmp_path / "out.jsonl"
    write_jsonl(source, [{"query": "before"}, None, {"query": "after"}])
    converter.convert_data(str(source), str(output))
    assert [r["question"] for r in read_jsonl(output)] == ["before", "after"]
    assert "Error processing record 1" in capsys.readouterr().out
    batch = converter.process_batch([None, {"query": "after"}], 20)
    assert len(batch) == 1 and batch[0]["id"] == "default_21"


@pytest.mark.parametrize("api_key", ["test-key", " padded-key "])
def test_parallel_conversion_calls_provider_and_preserves_all_record_identities(converter, tmp_path, monkeypatch, api_key):
    source, output = tmp_path / "in.jsonl", tmp_path / "out.jsonl"
    rows = [{"query": f"query-{i}", "task_name": "mbpp"} for i in range(5)]
    write_jsonl(source, rows)
    monkeypatch.setattr(converter, "API_KEYS", api_key)
    monkeypatch.setenv("API_KEYS", "different-environment-key")
    monkeypatch.setenv("OPENAI_API_BASE", "https://example.test/v1")
    calls = []

    def complete(**kwargs):
        calls.append(kwargs)
        # Deliberately differs from MBPP's direct classification, so ignoring
        # LLM mode cannot satisfy this test.
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="logical-reasoning"))])

    monkeypatch.setattr(converter, "completion", complete)

    converter.convert_data(str(source), str(output), use_llm=True, max_workers=2)

    records = sorted(read_jsonl(output), key=lambda row: row["extra_info"]["index"])
    assert [r["question"] for r in records] == [r["query"] for r in rows]
    assert [r["id"] for r in records] == [f"mbpp_{i}" for i in range(5)]
    assert all(r["ability"] == "logical-reasoning" for r in records)
    assert len(calls) == len(rows)
    assert all(call["api_key"] == api_key and call["api_base"] == "https://example.test/v1" for call in calls)


@pytest.mark.parametrize("api_key", [None, "", " \t\n "], ids=["missing", "empty", "whitespace"])
def test_parallel_conversion_without_key_falls_back_without_losing_records(converter, tmp_path, monkeypatch, capsys, api_key):
    source, output = tmp_path / "in.jsonl", tmp_path / "out.jsonl"
    rows = [
        {"query": "二加二?", "ground_truth": 4, "task_name": "gsm8k"},
        {"query": "write a function", "ground_truth": ["solution"], "task_name": "mbpp"},
        {"query": "keep this row", "ground_truth": "answer"},
    ]
    write_jsonl(source, rows)
    monkeypatch.setattr(converter, "API_KEYS", api_key)
    # The converter's existing key source is the module setting, not the
    # environment; a valid endpoint must not trigger calls with a blank key.
    monkeypatch.setenv("API_KEYS", "different-environment-key")
    monkeypatch.setenv("OPENAI_API_BASE", "https://example.test/v1")
    complete = Mock()
    monkeypatch.setattr(converter, "completion", complete)

    converter.convert_data(str(source), str(output), use_llm=True, max_workers=2)

    records = read_jsonl(output)
    assert [r["question"] for r in records] == [r["query"] for r in rows]
    assert [r["id"] for r in records] == ["gsm8k_0", "mbpp_1", "default_2"]
    assert [r["extra_info"] for r in records] == [{"index": i, "split": "test"} for i in range(3)]
    assert [r["ability"] for r in records] == ["math-reasoning", "code-generation", "fact-reasoning"]
    assert [r["golden_answers"] for r in records] == [["4"], ["solution"], ["answer"]]
    assert [r["reward_model"]["ground_truth"]["target"] for r in records] == [["4"], ["solution"], ["answer"]]
    complete.assert_not_called()
    stdout = capsys.readouterr().out
    assert "Warning: API key not found" in stdout
    assert "Converting records directly (without using LLM)" in stdout


def test_no_key_fallback_recovers_after_bad_record_with_original_indices(converter, tmp_path, monkeypatch, capsys):
    source, output = tmp_path / "in.jsonl", tmp_path / "out.jsonl"
    write_jsonl(source, [{"query": "outside"}, {"query": "before"}, None, {"query": "after"}])
    monkeypatch.setattr(converter, "API_KEYS", None)
    monkeypatch.setenv("OPENAI_API_BASE", "https://example.test/v1")
    complete = Mock()
    monkeypatch.setattr(converter, "completion", complete)

    converter.convert_data(str(source), str(output), use_llm=True, max_workers=2, start_line=2)

    records = read_jsonl(output)
    assert [r["question"] for r in records] == ["before", "after"]
    assert [r["id"] for r in records] == ["default_1", "default_3"]
    assert [r["extra_info"]["index"] for r in records] == [1, 3]
    complete.assert_not_called()
    assert "Error processing record 2" in capsys.readouterr().out


def test_training_conversion_keeps_model_response_cost_and_performance_aligned(converter, tmp_path):
    source, output = tmp_path / "in.jsonl", tmp_path / "train.json"
    write_jsonl(source, [
        {"query": "question", "ground_truth": "answer", "task_name": "mmlu", "model_name": "qwen2.5-7b-instruct", "response": "first", "input_tokens": 1000, "output_tokens": 500, "performance": "0.3"},
        {"query": "question", "model_name": "empty-response", "response": "", "performance": 1.0},
        {"query": "question", "model_name": "llama-3.1-70b-instruct", "response": "best", "input_tokens": 2000, "output_tokens": 1000, "performance": 0.9},
        {"query": "second", "ground_truth": 42},
        {"query": "", "model_name": "ignored"},
    ])
    converter.convert_train_data(str(source), str(output))
    first, second = json.loads(output.read_text())

    assert first == {
        "q_id": 0, "query": "question", "gt": ["answer"], "source": "mmlu",
        "llm_name_list": ["qwen/qwen2.5-7b-instruct", "meta/llama-3.1-70b-instruct"],
        "response_list": ["first", "best"], "cost_list": pytest.approx([0.15, 2.1]),
        "perf_list": [0.3, 0.9], "best_perf_idx": 1, "best_llm_name": "meta/llama-3.1-70b-instruct",
    }
    assert second["q_id"] == 1 and second["gt"] == ["42"] and second["source"] == "default"
    assert second["llm_name_list"] == [] and second["best_llm_name"] == ""


def test_bad_training_group_does_not_abort_other_queries(converter, tmp_path, capsys):
    source, output = tmp_path / "in.jsonl", tmp_path / "train.json"
    write_jsonl(source, [
        {"query": "broken", "model_name": "model", "response": "answer", "performance": "not-a-number"},
        {"query": "good", "ground_truth": ["a", "b"]},
    ])
    converter.convert_train_data(str(source), str(output))
    records = json.loads(output.read_text())
    assert len(records) == 1 and records[0]["query"] == "good" and records[0]["gt"] == ["a", "b"]
    assert "Error processing query 0" in capsys.readouterr().out


def test_merge_preserves_rows_marks_splits_and_skips_bad_json(converter, tmp_path, capsys):
    test_path, train_path, output = tmp_path / "test.jsonl", tmp_path / "train.json", tmp_path / "merged.jsonl"
    write_jsonl(test_path, [{"question": "一", "extra_info": {"index": 7, "split": "wrong"}}, {"question": "two"}])
    with test_path.open("a") as stream:
        stream.write("invalid json\n")
    train_path.write_text(json.dumps([{"query": "train", "q_id": 3, "split": "wrong"}]))
    converter.merge_train_test(str(test_path), str(train_path), str(output))
    assert read_jsonl(output) == [
        {"question": "一", "extra_info": {"index": 7, "split": "test"}, "split": "test"},
        {"question": "two", "extra_info": {"split": "test"}, "split": "test"},
        {"query": "train", "q_id": 3, "split": "train"},
    ]
    assert "Skipping invalid test JSON line" in capsys.readouterr().out


@pytest.mark.parametrize("train_contents", [None, "bad json", '{"not": "an array"}'])
def test_merge_handles_missing_or_unreadable_optional_inputs(converter, tmp_path, capsys, train_contents):
    test_path, train_path, output = tmp_path / "missing-test", tmp_path / "train.json", tmp_path / "merged.jsonl"
    if train_contents is not None:
        train_path.write_text(train_contents)
    converter.merge_train_test(str(test_path), str(train_path), str(output))
    assert read_jsonl(output) == []
    assert "test file does not exist" in capsys.readouterr().out


@pytest.mark.parametrize("mode", ["test", "train", "merge"])
def test_converter_cli_drives_real_file_conversion(converter, tmp_path, monkeypatch, mode):
    source, output = tmp_path / "source.jsonl", tmp_path / "output.json"
    write_jsonl(source, [{"query": "question", "ground_truth": "answer"}])
    args = ["data-convert", "--mode", mode, "--output", str(output)]
    if mode == "merge":
        train = tmp_path / "train.json"
        train.write_text('[{"query": "training"}]')
        args += ["--test-file", str(source), "--train-file", str(train)]
    else:
        args += ["--input", str(source)]
    monkeypatch.setattr(sys, "argv", args)
    converter.main()
    assert output.is_file()
    if mode == "train":
        assert json.loads(output.read_text())[0]["query"] == "question"
    else:
        assert read_jsonl(output)[0]["query" if mode == "merge" else "question"] == "question"


@pytest.mark.parametrize("args", [
    ["--mode", "test"], ["--mode", "train"], ["--mode", "merge"],
    ["--mode", "test", "--input", "missing.jsonl"],
    ["--mode", "train", "--input", "missing.jsonl"],
])
def test_converter_cli_rejects_missing_inputs_without_writing_output(converter, tmp_path, monkeypatch, args):
    output = tmp_path / "output.json"
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["data-convert", "--output", str(output), *args])
    with pytest.raises(SystemExit) as exc:
        converter.main()
    assert exc.value.code == 1
    assert not output.exists()
