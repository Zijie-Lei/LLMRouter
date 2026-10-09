"""Intentional failure for the fork's GitHub Checks UI demo PR."""


def test_ci_reports_a_regression():
    assert False, "Intentional CI demo failure: do not merge this PR."
