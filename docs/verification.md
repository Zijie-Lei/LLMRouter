# Verification and independent review

This records local verification on October 9, 2026 for the
`codex/pr-ci-guardrails` workstream. Hosted GitHub checks and publication still
need to be exercised after the workflows land.

## Local results

| Check | Result |
| --- | --- |
| Offline core suite, with sockets disabled | 415 passed; 2 CUDA-only tests skipped; 10 strict expected failures |
| Offline FusionGate suite, separate process | 54 passed |
| Python correctness checks | Ruff passed |
| Workflow syntax | actionlint 1.7.12 passed; local ShellCheck integration disabled because ShellCheck was unavailable |
| Engineering documentation | Strict MkDocs build passed |
| Distribution build | Wheel built from sdist using pinned tools; strict Twine checks passed |
| Fresh installed wheel, outside the checkout | All 14 YAML resources matched source bytes; metadata, runtime, and both CLI versions matched 0.4.0 |
| Packaging regression check | The previous wheel failed the new check for missing YAML resources |
| Aggregate CI gate | Nine success, failure, cancellation, and skipped-job scenarios behaved as expected |
| Release guard | Seven scenarios covered valid tags, missing tags, version mismatch, off-main ancestry, and rejected version/tag formats |

The test runs used macOS and Python 3.12. The configured Linux matrix covers
Python 3.10, 3.11, and 3.12; these changes have not yet run in hosted CI.
CPU tests do not establish GPU compatibility. Earlier runs also confirmed that
the socket guard blocked a dependency's import-time network attempt. The final
core run reported an existing Starlette dependency deprecation warning.
Dependency review, CodeQL uploads, environment approval, and trusted publishing
require hosted verification.

## Incremental coverage expansion

The follow-up adds nine test files to the existing offline suite. Product code,
dependencies, and CI/workflow configuration are unchanged in this pass. Pytest
discovers these files automatically in CI.

Core line coverage increased from **28.4% to 38.9%**, from 4,058 to 5,562 covered
lines out of 14,282. This adds **211 passing cases and ten expected failures**.
The separate FusionGate coverage remains 85.9%.

| Existing module | Before | After |
| --- | --- | --- |
| KNN router | 13.9% | 100% |
| SVM router | 13.0% | 100% |
| Largest/smallest routers | 9.3% / 24.1% | 100% / 100% |
| MLP / matrix-factorization routers | 12.7% / 14.4% | 83.0% / 83.0% |
| Provider API helper | 9.4% | 91.9% |
| Model save/load helper | 14.0% | 100% |
| Prompt loading | 11.1% | 100% |
| Data format validation | 41.5% | 95.7% |
| Batch evaluation | 0% | 97.3% |
| HTTP/WebSocket serving | 57.1% | 76.5% |

These are executed-line measurements, including lines reached by expected
failures. They do not establish that every branch or behavior works correctly.
Tests now exercise real tiny-data classifier training and checkpoint reloads,
CPU MLP/MF training and routing, API request/key/token contracts, serving error
paths, prompt discovery, dataset validation, embedding pooling, and known-answer
evaluation scores. External providers, downloaded encoders, and optional metric
models are replaced only at those boundaries.

### Existing defects recorded by tests

Ten narrow `xfail(strict=True)` cases record three existing defects:

- KNN, SVM, largest, and smallest routers forward a formatted prompt dictionary
  as user-message content instead of splitting system and user strings (four
  cases).
- Largest/smallest routing raises `KeyError` when a candidate omits `size`, in
  both single and batch routing (four cases).
- URLs containing `localhost` in a remote hostname or URL path are incorrectly
  treated as local endpoints that permit empty API keys (two cases).

These defects remain unfixed in the coverage-only pass. They are explicitly
reported by pytest rather than counted as passing tests. Once a defect is fixed,
an unexpected pass fails CI until its expected-failure marker is removed.

Additional inspection found an unused `call_api` retry parameter and missing
explicit HTTP transport-error mapping. These are follow-up findings, not passing
regression guarantees. WebSocket transport failures already have handling and
are tested.

### Review of the new tests

Two Codex workers added the classical/baseline and API/serving tests in the same
workspace; the remaining tests and the full-suite verification ran in the parent
session. Claude independently reviewed the exact API helper and its new test
file, returning **APPROVE-WITH-NITS** with no blockers. Its one finding was a
dependency-check ordering assertion that could not detect reordering; the test
now omits API keys so parsing first would raise the wrong exception. The revised
test passes. A broader Claude pass timed out and is not counted as a completed
review of all nine files.

## Review panel

Claude, Codex, and Muse Code reviewed code in the same workspace independently.
The table records their original verdicts and how their findings were handled;
it does not imply that all three approved the revised tree.

| Reviewer | Original verdict | Outcome |
| --- | --- | --- |
| Claude | APPROVE-WITH-NITS for CI/release/packaging; BLOCK for runtime | Fixed immutable CI checkouts and removed manual CI dispatch. Fixed blank chat failures and missing failure scores, retained model identity, and strengthened pipeline tests. |
| Codex | BLOCK | Fixed multi-round provider failures being reported as successful; added tests of the real pipelines and both CLI entry points. |
| Muse Code | BLOCK | Fixed missing query identity, floating build tools, Python matrix mismatch, and hardened reusable-workflow cancellation. Adjudicated the check-name and fork-permission claims below. |

No defect was independently raised by two completed reviewers. Their unique
findings improved runtime failure handling and the integrity of reported CI
results. Claude's interactive launcher failed to start, and a broad
non-interactive pass was rate limited; the completed CI review used Claude
directly with the exact file contents supplied in one request.
The separate runtime review completed 54 turns of read-only inspection of the
CLI, router implementations, tests, and their callers.

The runtime fixes have 134 focused passing tests. These cover both CLI entry
points, provider errors with request fields echoed back, raised/missing/`None`
responses, local KNN decomposition failure, distinct KNN routing failures,
stopping before later sub-queries and aggregation, prior token totals, and
zero failure scores for all ground-truth aliases (including the value `0`).

### Contested findings

- **Required check name.** Muse proposed `CI / CI`; the required context remains
  the aggregate job name `CI`. Claude independently checked this and agreed.
  GitHub documents the importance of unique
  [job names across workflows](https://docs.github.com/en/repositories/configuring-branches-and-merges-in-your-repository/managing-protected-branches/about-protected-branches#require-status-checks-before-merging).
  The activation guide requires confirming the actual Checks API name before
  enabling the ruleset.
- **CodeQL on fork PRs.** Muse proposed excluding forks because their tokens are
  read-only. GitHub explicitly allows uploading code-scanning results when a
  `pull_request` event triggers the run; the
  [documented permission exception](https://github.com/github/docs/blob/main/content/code-security/reference/code-scanning/troubleshoot-analysis-errors/resource-not-accessible.md)
  supports keeping fork coverage.
- **Small duplicated exit policy.** Both CLI entry points keep their short
  exit-status check, with shared behavioral tests covering each entry point.
- **Distribution checker unit tests.** The actual build/install smoke check and
  its negative check against the old wheel exercise the packaging boundary.
  No mock-only test of the checker was added.

The corrected findings are locally verified. The next gates are the hosted runs
and [repository setup](releases.md), followed by the behavioral and operational
work in the [engineering roadmap](production-readiness.md).
