# Engineering roadmap

## Onboarding snapshot

The `codex/pr-ci-guardrails` branch started with offline CPU test jobs on Python
3.10 and 3.12, a package build, test reports, and a proposed `main` ruleset.
The selected suites cover API adapters, CLI behavior, plugins, and RACER; most
trained-router examples require external data or model artifacts.

Inspection of the existing 0.4.0 distributions found no bundled YAML resources,
and the installed version lookup reported 0.1.0. The existing package smoke check
only requested CLI help, so it missed both defects. The docs deployment had no
documentation sources to build. Live GitHub inspection on October 9, 2026 found
`main` unprotected and no repository rulesets. The public site is maintained
separately on `website`, with its own Pages workflow.

## This workstream

- Package every bundled prompt and default YAML, and check their bytes after a
  fresh wheel installation outside the checkout.
- Report the installed distribution version correctly.
- Make CLI automation fail when inference fails, including upstream API errors,
  while preserving its JSON results for debugging and batch processing.
- Preserve failures through multi-round pipelines, show errors in chat, and
  score failed evaluation rows as zero when ground truth is supplied.
- Keep a stable required `CI` check covering tests, package integrity, Python
  correctness checks, workflow syntax, strict docs builds, and PR dependency review.
- Test every advertised Python minor version: 3.10, 3.11, and 3.12.
- Pin release build tools and the backend, with Dependabot updates tested by CI.
- Retain validated wheel/sdist artifacts and test/coverage reports.
- Review high/critical vulnerabilities introduced by PR dependencies; schedule
  dependency updates and Python CodeQL analysis.
- Validate a release's version, tag ancestry, tests, and distributions before
  optional publication with a protected PyPI environment and trusted publishing.
- Build engineering documentation strictly and retain it as a CI artifact;
  remove the orphaned deployment workflow from `main`.

The code supplies these checks. Required checks, environment protections, and
trusted publishers still require [repository configuration](releases.md).
See [verification and independent review](verification.md) for test results,
review findings, and the limits of the local checks.

The subsequent coverage-only pass adds real KNN/SVM and CPU MLP/MF training and
checkpoint tests, baseline routing, API/serving contracts, prompt/data validation,
and scoring checks. Core line coverage is now 38.9%; whole-project regression
coverage is still incomplete. Existing defects are recorded as strict expected
failures without changing product behavior in that pass.

## Follow-up priorities

1. **Broaden real behavior coverage.** Extend the temporary-data tests to remaining
   router families and deepen serving transport/streaming failure coverage.
   Resolve the existing regressions recorded by strict expected-failure tests.
   Track coverage
   in critical modules before introducing justified coverage thresholds.
2. **Define supported installation modes.** `custom_routers` and `custom_tasks`
   currently require a source checkout. Make installed CLI capabilities explicit
   and reject unavailable router selection rather than silently substituting one.
3. **Establish a compatibility baseline.** Add maintained dependency constraints
   alongside latest-compatible CI. Define a Python support policy and separately
   validate GPU extras; CPU checks do not establish CUDA/vLLM compatibility.
4. **Harden serving operations.** Specify authentication, request limits,
   upstream timeouts, startup configuration checks, health/readiness semantics,
   structured logs, and graceful shutdown, with deployment-specific tests.
5. **Exercise the hosted pipeline.** Obtain green GitHub runs, inspect CodeQL
   findings, activate the proposed ruleset, validate a release without publishing,
   then verify an approved package deployment. Integrate engineering guidance
   into the public website through its maintained source branch.

This is a foundation for reliable contribution and delivery. Operational
readiness also depends on deployment configuration, load tests, and ownership of
the follow-up work.
