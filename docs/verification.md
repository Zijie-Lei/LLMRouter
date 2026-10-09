# Verification and independent review

This records local and hosted verification on October 9, 2026 for the
`codex/pr-ci-guardrails` workstream. Release environment approval and publication
remain untested.

## Local results

| Check | Result |
| --- | --- |
| Offline core suite, with sockets disabled | 615 passed; 2 CUDA-only tests skipped; 22 strict expected failures |
| Offline FusionGate suite, separate process | 54 passed |
| Python correctness checks | Ruff passed |
| Workflow syntax | actionlint 1.7.12 passed; local ShellCheck integration disabled because ShellCheck was unavailable |
| Engineering documentation | Strict MkDocs build passed |
| Distribution build | Wheel built from sdist using pinned tools; strict Twine checks passed |
| Fresh installed wheel, outside the checkout | All 14 YAML resources matched source bytes; metadata, runtime, and both CLI versions matched 0.4.0 |
| Packaging regression check | The previous wheel failed the new check for missing YAML resources |
| Aggregate CI gate | Nine success, failure, cancellation, and skipped-job scenarios behaved as expected |
| Release guard | Seven scenarios covered valid tags, missing tags, version mismatch, off-main ancestry, and rejected version/tag formats |

The local test runs used macOS and Python 3.12. The Linux matrix also passed on
Python 3.10, 3.11, and 3.12 in the fork's first hosted PR run.
CPU tests do not establish GPU compatibility. Earlier runs also confirmed that
the socket guard blocked a dependency's import-time network attempt. The final
core run reported existing Starlette and scikit-learn deprecation warnings.
Release environment approval and trusted publishing require separate hosted
verification.

## Hosted PR verification

[PR #1 in the fork](https://github.com/Zijie-Lei/LLMRouter/pull/1) exercised CI
before the workflows were merged into `main`.
The [CI run](https://github.com/Zijie-Lei/LLMRouter/actions/runs/37999927763) and
[CodeQL run](https://github.com/Zijie-Lei/LLMRouter/actions/runs/37999927746) passed.
Each Python version passed 415 core tests and 54 FusionGate tests, with two
CUDA skips and ten expected failures. Core line coverage was 38.94% on Python
3.10 and 38.93% on Python 3.11/3.12. JUnit, coverage, documentation, and
distribution artifacts were downloaded successfully.

The fork initially had dependency graph disabled. Dependency review passed
after enabling the graph and rerunning the failed jobs. Hosted actionlint also
passed with Ubuntu's ShellCheck integration, and CodeQL uploaded its results.
Dependency review checks introduced dependency changes; existing Dependabot
vulnerability alerts on inherited `main` remain a separate backlog.

[PR #2](https://github.com/Zijie-Lei/LLMRouter/pull/2) is a separate deliberate
failure demo and must not be merged. Its
[run](https://github.com/Zijie-Lei/LLMRouter/actions/runs/38000223735) reported
exactly one intentional test failure in each Python version, and the aggregate
`CI` check failed. Other checks passed and failed-run test artifacts were
available. The deliberate test is absent from `codex/pr-ci-guardrails`.

The fork's `main` remains unchanged. Requiring passing checks before merging
still needs an active branch rule or ruleset.

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

## Second incremental coverage pass

Seven additional test files increase core line coverage from **38.9% to 48.9%**,
from 5,562 to 6,980 covered lines out of the same 14,282. The core suite adds
**200 passing cases and 12 strict expected failures**, reaching 615 passing
tests, two CUDA skips, and 22 expected failures. The full local offline run
completed in 21.79 seconds on Python 3.12.
Product code, dependencies, and workflow configuration are unchanged.

| Existing module | Before this pass | After this pass |
| --- | --- | --- |
| Dataset conversion | 0% | 99.7% |
| Embedding/routing data processing | 10.3% | 99.3% |
| Multi-turn / arena conversation utilities | 5.4% / 5.2% | 100% / 95.8% |
| Hybrid router / trainer | 10.8% / 29.2% | 93.0% / 91.7% |
| Elo router / trainer | 11.7% / 24.0% | 92.2% / 100% |
| Threshold router / trainer | 19.1% / 11.9% | 95.6% / 82.1% |
| TaskAware router | 21.1% | 100% |
| OpenClaw memory / media | 14.8% / 11.2% | 98.9% / 75.3% |

The tests exercise real JSONL/tensor persistence and row alignment, answer and
model-cost schemas, known-answer conversation scoring, tiny-data sklearn
training and checkpoint reload, actual neural decision boundaries, memory
pooling/filtering/ranking, media parsing, HTTP request serialization, and
temporary-file cleanup. A controlled completion scheduler forces embedding
tasks out of order; real-thread cases also run. Encoder downloads and external
providers are replaced at their boundaries.

The twelve new expected failures record existing defects:

- Two converter modes include an end line documented as exclusive.
- LLM conversion without an API key drops records rather than using its
  announced fallback.
- Failed or unmatched embedding IDs serialize as the non-JSON constant `NaN`
  instead of `null` (two cases).
- Non-object JSONL entries crash memory reload (two cases).
- Standard nested OpenAI `input_audio` data is ignored.
- Threshold training cannot consume its loaded DataFrame, and after supplying
  a record sequence to isolate that defect, a real training epoch fails at
  checkpoint saving because the trainer has not retained its configuration.
- TaskAware routing misses fallback handling for a wrong-type JSON judgment
  and for a direct timeout (two cases).

These remain explicitly unfixed in the test-only pass. Executed-line coverage
includes their failing paths; it does not mean these behaviors pass. GPU and
downloaded-model paths, several router families, generation/evaluation scripts,
and other CLI paths still need coverage. No exclusions or coverage thresholds
were changed.

Two Codex workers authored the memory/media and policy/Elo tests in the shared
workspace. The parent reviewed those files and ran the complete suite; one
worker also reviewed the data/conversation tests. Claude Opus 5.5 independently
reviewed the initial data-processing and conversation tests, returning
**APPROVE-WITH-NITS** with no blockers. Review findings led to forced completion
ordering, assertions on successful persisted embedding IDs, independently
copied nested DataFrame values, exact model-ID suffixes, and a provider result
that differs from the direct fallback with explicit call-count checks. The
revised tests pass in the full suite. Claude did not review all seven files or
execute the tests, and two helper source files were outside its supplied scope.

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

The corrected findings are locally verified and the PR checks have also run
successfully in the fork. Remaining gates include [repository setup](releases.md)
and release verification, followed by the behavioral and operational
work in the [engineering roadmap](production-readiness.md).
