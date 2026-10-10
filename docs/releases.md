# Releases and repository setup

## Configure GitHub

Maintainers must configure these settings after the workflows land on `main`:

1. Wait for a successful `CI` check and apply
   [the proposed ruleset](https://github.com/ulab-uiuc/LLMRouter/blob/main/.github/main-ruleset.json).
   It requires `CI`, an up-to-date branch, one approval, and resolved review
   threads, and prevents branch deletion and force pushes. The local JSON file
   does not activate the rule. Confirm the aggregate job's check-run name is `CI`
   in the Checks API or required-check picker before activation; UI workflow/job
   labels can include a workflow prefix.
2. Enable the dependency graph and private vulnerability reporting. Confirm
   dependency review and the separately reported `CodeQL (Python)` analysis run
   successfully. Triage CodeQL findings before adding that check to merge rules.
3. Create a **`pypi`** environment requiring maintainer approval. Restrict it to
   `main`, disallow self-review, and disable protection bypass where available.
   Protect `v*` tags from modification/deletion using a tag ruleset.
4. In the PyPI `llmrouter-lib` project, add a trusted publisher for GitHub owner
   `ulab-uiuc`, repository `LLMRouter`, workflow filename `release.yml`, and
   environment `pypi`. This workflow uses OIDC; it needs no stored PyPI token.

Changing repository settings and creating tags are maintainer operations.

## Prepare a release

1. Update `project.version` in `pyproject.toml` and the unreleased changelog in a PR.
2. Review the change and merge it only after `CI` passes.
3. Create an immutable `vMAJOR.MINOR.PATCH` tag on the reviewed `main` commit.
   The tag version must equal the package version at that commit. This first
   release workflow accepts stable versions only.
4. Run **Release** from `main` using that tag and leave **publish** disabled.
   It validates tag ancestry and the package version, reruns the complete CI
   workflow at the tag's commit, and retains the validated distributions.
5. Inspect the results and artifacts. When ready to release, run **Release**
   with the same tag and **publish** enabled, then approve the `pypi` environment.
   This run revalidates and publishes its own artifacts with attestations.

The publishing job downloads the wheel and source archive produced by its
validation run. It does not rebuild after approval. Existing versions are not
silently skipped: a duplicate PyPI version causes publishing to fail.
The validation build uses the exact frontend, backend, and metadata-checking
versions in `requirements/build.txt` with build isolation disabled.

## Failed release or rollback

Validation failures prevent the publish job from running. Fix the issue in a
reviewed PR and prepare a new version/tag. Do not move an existing release tag.

After publication, PyPI versions cannot be overwritten. For a broken release,
notify maintainers, assess whether to yank it, and release a fixed version.
Consumers can pin the last known-good version while the fix is reviewed.

Engineering documentation is retained as a CI artifact. The public website and
its Pages deployment are maintained on `website`; site changes and rollbacks
belong in that branch's workflow.
