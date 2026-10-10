"""Check conversation extraction and known-answer pairwise preference scores."""

import json

import pytest

from llmrouter.utils import arena_conversation, conversation


@pytest.mark.parametrize("module", [conversation, arena_conversation])
@pytest.mark.parametrize("encoded", [False, True])
def test_prompt_and_response_extraction_ignore_other_roles_and_select_first_pair(module, encoded):
    messages = [
        {"role": "system", "content": "instructions"},
        {"role": "user", "content": "question 一"},
        {"role": "tool", "content": "tool output"},
        {"role": "assistant", "content": "answer 一"},
        {"role": "user", "content": "second question"},
        {"role": "assistant", "content": "second answer"},
    ]
    payload = json.dumps(messages) if encoded else messages
    assert module.extract_user_prompt(payload) == "question 一"
    assert module.extract_model_response(payload) == "answer 一"
    assert messages[0]["content"] == "instructions"
    if module is conversation:
        assert module.extract_user_prompt(payload, turn=2) == "second question"
        assert module.extract_model_response(payload, turn=2) == "second answer"
        assert module.extract_user_prompt(payload, turn=3) == ""
        assert module.extract_model_response(payload, turn=3) == ""


@pytest.mark.parametrize("module", [conversation, arena_conversation])
@pytest.mark.parametrize("payload", [None, "invalid json", [None], [{"role": "system", "content": "only system"}], [{"role": "user"}, {"role": "assistant"}]])
def test_unusable_conversations_return_empty_content(module, payload):
    assert module.extract_user_prompt(payload) == ""
    assert module.extract_model_response(payload) == ""


def pair(model_a, model_b, winner, *, query="question", answer_a=None, answer_b=None, turn=1):
    return {
        "conversation_a": [{"role": "user", "content": query}, {"role": "assistant", "content": answer_a or f"answer-{model_a}"}],
        "conversation_b": [{"role": "user", "content": query}, {"role": "assistant", "content": answer_b or f"answer-{model_b}"}],
        "model_a": model_a, "model_b": model_b, "winner": winner, "turn": turn,
    }


@pytest.mark.parametrize("module", [conversation, arena_conversation])
def test_pairwise_aggregation_produces_known_model_wins_comparison_counts_and_choices(module):
    data = [pair("a", "b", "model_a"), pair("c", "a", "model_b"), pair("b", "c", "tie")]
    snapshot = json.loads(json.dumps(data))
    groups = module.aggregate_preferences_by_query(data)
    scores = module.calculate_model_scores(groups)

    assert data == snapshot
    assert set(groups) == {"question"}
    assert groups["question"]["models"] == {"a": "answer-a", "b": "answer-b", "c": "answer-c"}
    assert [pref["winner"] for pref in groups["question"]["preferences"]] == ["a", "a", None]
    by_model = {result["model_name"]: result for result in scores}
    assert set(by_model) == {"a", "b", "c"}
    assert {name: result["wins"] for name, result in by_model.items()} == {"a": 2, "b": 0, "c": 0}
    assert {name: result["preference_score"] for name, result in by_model.items()} == {"a": 1.0, "b": 0.0, "c": 0.0}
    for name, result in by_model.items():
        assert result["query"] == "question"
        assert result["gt"] == f"answer-{name}"
        assert result["total_comparisons"] == 2
        assert result["metric"] == "preference_score"
        assert result["choices"][0] == result["gt"]
        assert set(result["choices"]) == {"answer-a", "answer-b", "answer-c"}
        suffix = f"_{name}_t1" if module is conversation else f"_{name}"
        assert result["question_id"].endswith(suffix)
        assert result["task_name"] == ("mt_bench" if module is conversation else "chatbot_arena")


@pytest.mark.parametrize("module", [conversation, arena_conversation])
def test_aggregation_falls_back_to_second_prompt_and_skips_incomplete_samples(module):
    valid = pair("a", "b", "model_a")
    valid["conversation_a"] = [{"role": "assistant", "content": "answer-a"}]
    data = [
        {"missing": "conversation fields"},
        {**pair("a", "b", "tie"), "conversation_a": [], "conversation_b": []},
        {**pair("a", "b", "tie"), "conversation_b": [{"role": "user", "content": "empty response"}]},
        {**pair("a", "b", "tie"), "model_a": None, "conversation_a": "invalid JSON"},
        valid,
    ]
    groups = module.aggregate_preferences_by_query(data)
    assert set(groups) == {"question"}
    assert len(groups["question"]["preferences"]) == 1
    assert groups["question"]["preferences"][0]["winner"] == "a"
    assert len(module.calculate_model_scores(groups)) == 2


@pytest.mark.parametrize("module", [conversation, arena_conversation])
def test_missing_judgment_retains_models_with_zero_scores_and_zero_comparisons(module):
    groups = module.aggregate_preferences_by_query([pair("a", "b", None)])
    assert groups["question"]["preferences"] == []
    results = module.calculate_model_scores(groups)
    assert len(results) == 2
    assert all(result["wins"] == result["preference_score"] == result["total_comparisons"] == 0 for result in results)
    assert module.aggregate_preferences_by_query([]) == {}
    assert module.calculate_model_scores({}) == []


def test_turn_filter_selects_the_matching_prompts_responses_and_judgment_metadata():
    second = pair("a", "b", "model_b", turn=2)
    second["conversation_a"] += [{"role": "user", "content": "follow-up"}, {"role": "assistant", "content": "follow-up-a"}]
    second["conversation_b"] += [{"role": "user", "content": "follow-up"}, {"role": "assistant", "content": "follow-up-b"}]
    second.update(judge="reviewer", question_id="source-id")
    groups = conversation.aggregate_preferences_by_query([pair("a", "b", "model_a"), second], turn_filter=2)

    assert set(groups) == {"follow-up"}
    assert groups["follow-up"]["turn"] == 2
    assert groups["follow-up"]["models"] == {"a": "follow-up-a", "b": "follow-up-b"}
    assert groups["follow-up"]["preferences"][0]["judge"] == "reviewer"
    assert groups["follow-up"]["preferences"][0]["question_id"] == "source-id"
    results = conversation.calculate_model_scores(groups)
    assert all(result["turn"] == 2 and result["question_id"].endswith("_t2") for result in results)
    assert {result["model_name"]: result["preference_score"] for result in results} == {"a": 0.0, "b": 1.0}
