"""Protect existing scoring and batch-evaluation contracts with known answers."""

from copy import deepcopy
from types import ModuleType
import sys

import pytest
import torch

from llmrouter.evaluation import batch_evaluator
from llmrouter.utils import evaluation, prompting


@pytest.fixture(autouse=True)
def isolated_registries(monkeypatch):
    monkeypatch.setattr(batch_evaluator, "EVALUATION_METRICS", batch_evaluator.EVALUATION_METRICS.copy())
    monkeypatch.setattr(prompting, "PROMPT_REGISTRY", prompting.PROMPT_REGISTRY.copy())
    monkeypatch.setattr(prompting, "TASK_METRIC_REGISTRY", prompting.TASK_METRIC_REGISTRY.copy())


@pytest.mark.parametrize("prediction,truth,expected", [
    ("The RED fox!", "red fox", (1.0, 1.0, 1.0)),
    ("red fox jumps", "red wolf", (0.4, 1 / 3, 0.5)),
    ("yes", "no", (0.0, 0.0, 0.0)),
    ("", "red fox", (0.0, 0.0, 0.0)),
])
def test_f1_normalization_overlap_and_binary_answer_mismatch(prediction, truth, expected):
    assert evaluation.f1_score(prediction, truth) == pytest.approx(expected)


@pytest.mark.parametrize("metric,prediction,truth,expected", [
    ("em", "The RED fox!", "red fox", 1.0),
    ("em", "red fox", "red wolf", 0.0),
    ("em_mc", "I considered (A), but select (B)", "B", 1.0),
    ("em_mc", "I select (A)", "B", 0.0),
    ("cem", "The answer is red fox", "red fox", 1.0),
    ("cemf1", "red fox jumps", "red wolf", 0.4),
    ("f1", "red fox jumps", "red wolf", 0.4),
    ("gsm8k", "First 2, then the final total is $1,234", "work\n#### 1234", 1.0),
    ("gsm8k", "No numeric answer", "#### 1234", 0.0),
    ("gsm8k", "Final answer 1235", "#### 1234", 0.0),
])
def test_batch_metrics_preserve_metadata_and_score_known_answers(metric, prediction, truth, expected):
    rows = [{"prediction": prediction, "ground_truth": truth, "metric": metric, "request_id": 7}]
    snapshot = deepcopy(rows)

    result = batch_evaluator.evaluate_batch(rows)[0]

    assert rows == snapshot
    assert result["request_id"] == 7
    assert result["score"] == pytest.approx(expected)
    assert "evaluation_error" not in result


@pytest.mark.parametrize("prediction,truth,expected", [
    (r"Answer: \boxed{\frac{1}{2}}", "0.5", 1.0),
    (r"First \boxed{3}; final \boxed{4}", r"\boxed{4}", 1.0),
    (r"Final \boxed{5}", "4", 0.0),
    ("No boxed answer", "4", 0.0),
])
def test_math_task_scores_the_final_boxed_answer(prediction, truth, expected):
    assert evaluation.calculate_task_performance(prediction, truth, task_name="math") == expected


@pytest.mark.parametrize("prediction,truth,task,expected", [
    ("The red fox", "red fox", "natural_qa", 1.0),
    ("Selected (C)", "C", "mmlu", 1.0),
    ("The answer is 42", "#### 42", "gsm8k", 1.0),
    ("unrelated", "red fox", "unknown-task", 0.0),
])
def test_task_selects_the_existing_default_metric(prediction, truth, task, expected):
    assert evaluation.calculate_task_performance(prediction, truth, task_name=task) == expected


def test_task_performance_without_ground_truth_has_no_score():
    assert evaluation.calculate_task_performance("answer", None, task_name="natural_qa") is None


@pytest.mark.parametrize("row,fragment", [
    ({"ground_truth": "answer", "metric": "em"}, "prediction"),
    ({"prediction": "answer", "metric": "em"}, "ground_truth"),
    ({"prediction": "answer", "ground_truth": "answer"}, "Metric must"),
    ({"prediction": "answer", "ground_truth": "answer", "metric": "unregistered-metric"}, "Unknown metric"),
])
def test_batch_rejects_missing_fields_and_unknown_metrics(row, fragment):
    with pytest.raises(ValueError, match=fragment):
        batch_evaluator.evaluate_batch([row])


def test_custom_metric_receives_options_and_failure_does_not_discard_other_rows():
    calls = []

    @batch_evaluator.evaluation_metric("test-custom")
    def metric(prediction, truth, weight, **kwargs):
        calls.append((prediction, truth, weight, kwargs["request_id"]))
        if prediction == "broken":
            raise ValueError("invalid prediction")
        return weight if prediction == truth else 0.0

    rows = [
        {"prediction": "answer", "ground_truth": "answer", "weight": 0.75, "request_id": 1},
        {"prediction": "broken", "ground_truth": "answer", "weight": 0.5, "request_id": 2},
        {"prediction": "answer", "ground_truth": "answer", "weight": 1.0, "request_id": 3},
    ]
    results = batch_evaluator.evaluate_batch(rows, default_metric="test-custom")

    assert [row["score"] for row in results] == [0.75, 0.0, 1.0]
    assert results[1]["evaluation_error"] == "invalid prediction"
    assert len(calls) == 3
    assert "test-custom" in batch_evaluator.get_available_metrics()
    batch_evaluator.register_custom_metric("test-direct", metric)
    assert "test-direct" in batch_evaluator.get_available_metrics()


def test_prompt_and_metric_registries_share_custom_task_contract():
    @prompting.register_prompt("test-task", default_metric="test-score")
    def formatter(row):
        return {"system": "instruction", "user": row["query"]}

    batch_evaluator.register_custom_metric("test-score", lambda prediction, truth: prediction == truth)
    assert prompting.generate_task_query("test-task", {"query": "question"}) == {
        "system": "instruction", "user": "question",
    }
    assert evaluation.calculate_task_performance("answer", "answer", task_name="test-task") == 1.0
    prompting.register_task_metric("test-task", "em")
    assert evaluation.calculate_task_performance("answer", "different", task_name="test-task") == 0.0


def test_legacy_custom_prompt_formatter_returns_user_text():
    @prompting.register_prompt("test-legacy")
    def formatter(row):
        return f"Question: {row['query']}"

    assert prompting.generate_task_query("test-legacy", {"query": "question"}) == {
        "system": None, "user": "Question: question",
    }


def test_bert_metric_averages_external_scores_without_loading_models(monkeypatch):
    external = ModuleType("bert_score")
    calls = []

    def score(predictions, truths, **kwargs):
        calls.append((predictions, truths, kwargs))
        value = 0.5 if predictions == ["first"] else 1.0
        return torch.tensor([value]), torch.tensor([value]), torch.tensor([value])

    external.score = score
    monkeypatch.setitem(sys.modules, "bert_score", external)

    assert evaluation.get_bert_score(["first", "second"], ["truth-1", "truth-2"]) == pytest.approx(0.75)
    assert [call[1] for call in calls] == [["truth-1"], ["truth-2"]]
    assert all(call[2]["lang"] == "en" for call in calls)
