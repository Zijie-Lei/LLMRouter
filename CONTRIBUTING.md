# Contributing

## Automated checks

The `CI` workflow runs for every pull request targeting `main`, every push to
`main`, and merge-queue groups. It also supplies release
validation. It runs the offline unit suites on Python 3.10, 3.11, and 3.12 with CPU
PyTorch, checks Python correctness and workflow syntax, builds engineering docs
strictly, and reviews PR dependencies for introduced high/critical advisories.
It builds the wheel from the source distribution, then verifies an isolated
installation's version, entry point, and every bundled YAML resource outside
the checkout. Test reports, coverage, distributions, and built docs are artifacts.

The final check is named **`CI`**. It passes only when every applicable job
succeeds. Dependency review is required for PR events and intentionally skips
other events. Keep this name stable when configuring required checks. The
workflow has no path filters, so documentation-only PRs also get the required
check. CodeQL runs separately and should be triaged before becoming required.

Every job checks out the triggering commit (or the resolved release SHA).
Use GitHub's **Re-run jobs** on a PR run to retain its dependency-review gate.

GitHub must also have a branch rule requiring `CI` before it can block merges.
The intended rule for `main` requires an up-to-date branch, one approving
review, and resolved review threads, and prevents force pushes and deletion.
Repository administrators can configure that rule through the GitHub API.
The complete proposed rule is in [`.github/main-ruleset.json`](.github/main-ruleset.json).
Activate it after the new workflow has produced a successful `CI` check.

## Run the same tests locally

Use Python 3.10, 3.11, or 3.12 in a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip

# Linux: install the CPU wheel first, as CI does.
python -m pip install "torch==2.8.0" --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e ".[test,routefm]" "torch==2.8.0"
```

On macOS, install `torch==2.8.0` from the default PyPI index instead of the
Linux CPU index. Set the offline environment before running the tests:

```bash
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export LITELLM_LOCAL_MODEL_COST_MAP=True TOKENIZERS_PARALLELISM=false
export LLMROUTER_EMBEDDING_DEVICE=cpu CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONHASHSEED=0

python -m pytest --disable-socket --allow-unix-socket
python -m pytest custom_routers/fusion_gate/tests --disable-socket --allow-unix-socket
```

FusionGate runs in a separate process because its test bootstrap changes
Python import state. GPU-only RACER tests skip on CPU. Trained-router scripts
in `tests/train_test` and `tests/inference_test`, the live WebSocket script,
and the TaskAware live-service tests are outside the offline unit suite.

New unit tests should use temporary data, small tensors, and mocked upstream
services or encoders. Assert results and allow failures to raise; returning a
boolean from a pytest test does not establish a passing or failing result.
Tests should run without API keys, a running server, downloaded checkpoints,
or a GPU.

## Validate a distribution

```bash
python -m pip install -r requirements/build.txt
python -m build --no-isolation
python -m twine check --strict dist/*
```

Verify the installed package in a separate environment, away from the checkout:

```bash
project_dir="$PWD"
smoke_dir="$(mktemp -d /tmp/llmrouter-wheel-smoke.XXXXXX)"
python -m venv "$smoke_dir"
"$smoke_dir/bin/python" -m pip install --no-deps dist/*.whl
cd /tmp
"$smoke_dir/bin/python" -I "$project_dir/scripts/check_distribution.py" \
  --wheel "$project_dir"/dist/*.whl
cd "$project_dir"
```

Use a fresh smoke environment to prevent an editable installation from masking
packaging problems. `python -m build` builds the wheel from the sdist by default.
CI disables build isolation after installing the pinned backend and build tools,
so release validation uses those exact versions. Dependabot maintains the pins.

## Code, workflows, and docs

```bash
python -m pip install -r requirements/quality.txt
python -m ruff check .
actionlint
python -m mkdocs build --strict
```

Install actionlint 1.7.12 from its upstream release. CI checks the downloaded
binary's SHA-256. Ruff currently checks syntax errors, undefined names, invalid
comparisons, and invalid control flow across the repository; style cleanup is
separate work. Engineering docs are retained as the `engineering-docs` artifact.
The public website and its Pages workflow are maintained on `website`.

## Releases and security

See [the release and repository setup guide](docs/releases.md) for tag validation,
dry runs, protected PyPI publishing, and required GitHub settings. Publication
defaults to disabled and uses OIDC after the `pypi` environment is approved.
See [SECURITY.md](SECURITY.md) for private reporting and the current support policy.

## Submit a change

Open a pull request with a clear description and the test commands you ran.
Get a maintainer review, resolve feedback, and check that `CI` has passed
against the current `main` before merging.
