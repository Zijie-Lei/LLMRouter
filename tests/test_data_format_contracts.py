"""Validate router datasets at their public format boundary."""

from copy import deepcopy

import pytest
from pydantic import ValidationError

from llmrouter.data.data import (
    DataFormatDetector,
    DataFormatType,
    GMTRouterInteraction,
    StandardDataFormat,
    StandardQueryData,
    get_format_requirements,
    print_format_help,
)


def interaction():
    return {
        "judge": "user-1", "model": "model-a", "question_id": "question-1",
        "turn": 1, "model_emb": [0.25, -0.75],
        "conversation": [{"query": "Question", "query_emb": [0.5, -0.5], "rating": 4.5}],
    }


@pytest.mark.parametrize("query", ["", "  \n\t"])
def test_standard_query_rejects_empty_text(query):
    with pytest.raises(ValidationError, match="Query cannot be empty"):
        StandardQueryData(query=query)


@pytest.mark.parametrize("data", [
    {"query": "Question", "task": "qa", "query_id": "question-1"},
    {"query_id": "question-1", "best_model": "model-a", "model_scores": {"model-a": 0.9}},
])
def test_detector_accepts_standard_query_and_routing_rows_without_mutation(data):
    original = deepcopy(data)
    valid, kind, error = DataFormatDetector().validate_and_detect(data)

    assert (valid, kind, error) == (True, DataFormatType.STANDARD, None)
    assert data == original


@pytest.mark.parametrize("data", [None, [], {}, {"query": " "}, {"best_model": "model-a"}])
def test_detector_reports_unknown_rows_without_raising(data):
    valid, kind, error = DataFormatDetector().validate_and_detect(data)

    assert valid is False
    assert kind is DataFormatType.UNKNOWN
    assert "Unknown data format" in error


@pytest.mark.parametrize("rating", [0.0, 5.0])
def test_gmt_accepts_rating_boundaries_and_takes_precedence_over_standard(rating):
    data = interaction()
    data["conversation"][0]["rating"] = rating
    data["query"] = "Also matches the standard format"

    valid, kind, error = DataFormatDetector().validate_and_detect(data)

    assert (valid, kind, error) == (True, DataFormatType.GMTROUTER, None)


@pytest.mark.parametrize("field,value", [
    ("turn", 0), ("conversation", []), ("model_emb", []),
])
def test_gmt_rejects_invalid_interaction_fields(field, value):
    data = interaction()
    data[field] = value

    with pytest.raises(ValidationError):
        GMTRouterInteraction(**data)
    assert DataFormatDetector().detect_format(data) is DataFormatType.UNKNOWN


@pytest.mark.parametrize("field,value", [
    ("query_emb", []), ("rating", -0.1), ("rating", 5.1),
])
def test_gmt_rejects_invalid_conversation_fields(field, value):
    data = interaction()
    data["conversation"][0][field] = value

    with pytest.raises(ValidationError):
        GMTRouterInteraction(**data)
    assert DataFormatDetector().detect_format(data) is DataFormatType.UNKNOWN


def test_format_requirements_examples_are_valid_and_match_required_fields():
    detector = DataFormatDetector()
    standard = get_format_requirements(DataFormatType.STANDARD)
    for name in ("example_query", "example_routing"):
        assert detector.detect_format(standard[name]) is DataFormatType.STANDARD
    gmt = get_format_requirements(DataFormatType.GMTROUTER)
    assert set(gmt["required_fields"]).issubset(gmt["example"])
    assert detector.detect_format(gmt["example"]) is DataFormatType.GMTROUTER
    assert StandardDataFormat().get_required_fields() == ["query"]
    assert detector.gmtrouter_validator.get_required_fields() == gmt["required_fields"]
    assert "error" in get_format_requirements(DataFormatType.UNKNOWN)


@pytest.mark.parametrize("kind", [None, DataFormatType.STANDARD, DataFormatType.GMTROUTER])
def test_format_help_describes_supported_formats(kind, capsys):
    print_format_help(kind)
    output = capsys.readouterr().out
    if kind is not DataFormatType.GMTROUTER:
        assert "Standard Router Format" in output
    if kind is not DataFormatType.STANDARD:
        assert "GMTRouter Format" in output
