# Changelog

## Unreleased

### Fixed

- Guard shared embedding initialization and tokenizer configuration, preserving
  retry after failed setup and concurrent neural model forwards.
- Serialize mutable router calls and keep concurrent request diagnostics with
  their selected models while moving blocking work off server event loops.
- Reuse one route for concurrent requests in the same Responses API turn, with
  context eligibility checks and cancellation-safe waiter cleanup.
- Keep saved-history append order consistent with live-index publication and
  avoid publishing live records when append fails.
- Use direct dataset conversion when LLM mode has no usable API key, preserving
  records through the existing fallback.
- Write missing routing values as JSON `null` and validate JSON and UTF-8 in both
  routing splits before replacing existing embedding and JSONL outputs.
- Recover valid saved routing history when other records have malformed JSON,
  invalid fields, or corrupt embeddings.
- Include bundled YAML resources in wheel and source distributions.
- Report the installed `llmrouter-lib` version correctly.
- Return a failing CLI exit status for failed inference and propagate provider
  errors while preserving machine-readable results.
- Preserve multi-round provider failures through decomposition, execution, and
  aggregation, and keep the input query in routing error results.
- Show multi-round chat failures and retain zero evaluation scores for failures
  with supplied ground truth.
- Call the existing JSONL loader in the abductive time-series benchmark script.

### Added

- Offline PR tests, test and coverage artifacts, and a stable aggregate `CI` gate.
- Python correctness checks, workflow validation, strict engineering docs builds,
  PR dependency review, dependency updates, and Python CodeQL analysis.
- Fresh installed-wheel checks and retained distribution artifacts.
- Pinned release build tools and CI coverage for Python 3.10, 3.11, and 3.12.
- Commit-pinned CI checkouts and PR reruns that retain dependency review.
- Release validation and optional protected PyPI trusted publishing.
- Built engineering documentation retained as a CI artifact.
- Contribution, security, repository setup, and release guidance.

### Removed

- Orphaned `main` documentation deployment; the public site is deployed from `website`.
