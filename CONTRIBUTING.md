# Contributing

## Automated checks

The `CI` workflow runs for every pull request targeting `main`, every push to
`main`, and manual dispatches. It runs the offline unit suites on Python 3.10
and 3.12 with CPU PyTorch, then builds and validates the package and checks the
installed CLI entry point. Test results and coverage reports are available as
workflow artifacts.

The final check is named **`CI`**. It passes only when both Python test jobs
and the package build succeed. Keep this name stable when configuring required
checks. The workflow has no path filters, so documentation-only PRs also get
the required check.

GitHub must also have a branch rule requiring `CI` before it can block merges.
The intended rule for `main` requires an up-to-date branch, one approving
review, and resolved review threads, and prevents force pushes and deletion.
Repository administrators can configure that rule through the GitHub API.
The complete proposed rule is in [`.github/main-ruleset.json`](.github/main-ruleset.json).
Activate it after the new workflow has produced a successful `CI` check.

## Run the same tests locally

Use Python 3.10 or 3.12 in a virtual environment:

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
python -m pip install "build>=1,<2" "twine>=6,<7"
python -m build
python -m twine check --strict dist/*
```

## Submit a change

Open a pull request with a clear description and the test commands you ran.
Get a maintainer review, resolve feedback, and check that `CI` has passed
against the current `main` before merging.
