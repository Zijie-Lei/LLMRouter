#!/usr/bin/env python3
"""Check an installed wheel without importing optional runtime dependencies.

Install the wheel with ``pip install --no-deps`` in a fresh virtual environment,
then run this script from outside the checkout, preferably with ``python -I``::

    python -I /checkout/scripts/check_distribution.py --wheel /checkout/dist/package.whl

The wheel supplies the expected release version. The checkout supplies the full
YAML resource inventory, so accidentally omitted assets cannot pass the check.
"""

import argparse
from email.parser import Parser
import importlib
from importlib import metadata, resources, util
import os
from pathlib import Path
import subprocess
import sys
import sysconfig
from types import ModuleType
from zipfile import ZipFile


DISTRIBUTION = "llmrouter-lib"
PACKAGES = ("llmrouter", "openclaw_router")


def check_distribution(wheel: Path, source_root: Path) -> None:
    if Path.cwd().resolve().is_relative_to(source_root):
        raise RuntimeError("Run this check from outside the source checkout.")

    expected_assets = {}
    for package in PACKAGES:
        package_assets = {
            path.relative_to(source_root).as_posix(): path.read_bytes()
            for path in (source_root / package).rglob("*")
            if path.is_file() and path.suffix in {".yaml", ".yml"}
        }
        if not package_assets:
            raise RuntimeError(f"No source YAML resources found for {package}.")
        expected_assets.update(package_assets)

    with ZipFile(wheel) as archive:
        metadata_files = [
            name for name in archive.namelist()
            if name.endswith(".dist-info/METADATA")
        ]
        if len(metadata_files) != 1:
            raise RuntimeError("Expected exactly one wheel metadata file.")
        wheel_metadata = Parser().parsestr(archive.read(metadata_files[0]).decode("utf-8"))
        if wheel_metadata["Name"] != DISTRIBUTION:
            raise RuntimeError(f"Expected a {DISTRIBUTION} wheel.")
        expected_version = wheel_metadata["Version"]
        if not expected_version:
            raise RuntimeError("Wheel metadata has no version.")
        for name, expected_content in expected_assets.items():
            if archive.read(name) != expected_content:
                raise RuntimeError(f"Wheel resource differs from source: {name}")

    installed = metadata.distribution(DISTRIBUTION)
    if installed.version != expected_version:
        raise RuntimeError(
            f"Installed version {installed.version} differs from wheel {expected_version}."
        )
    installed_files = {str(path) for path in installed.files or ()}
    missing_files = set(expected_assets) - installed_files
    if missing_files:
        raise RuntimeError(f"Resources missing from installed metadata: {sorted(missing_files)}")

    anchors = {}
    for package in PACKAGES:
        spec = util.find_spec(package)
        expected_origin = Path(installed.locate_file(f"{package}/__init__.py")).resolve()
        if spec is None or spec.origin is None or Path(spec.origin).resolve() != expected_origin:
            raise RuntimeError(f"{package} resolves outside the installed distribution.")
        if expected_origin == (source_root / package / "__init__.py").resolve():
            raise RuntimeError(f"{package} resolves to an editable source checkout.")
        # resources.files accepts a module anchor. Using its import spec lets us
        # inspect OpenClaw resources without executing its PyYAML-dependent imports.
        anchor = ModuleType(package)
        anchor.__spec__ = spec
        anchors[package] = resources.files(anchor)

    runtime = importlib.import_module("llmrouter")
    if runtime.__version__ != expected_version:
        raise RuntimeError(
            f"Runtime version {runtime.__version__} differs from wheel {expected_version}."
        )
    for name, expected_content in expected_assets.items():
        package, relative_path = name.split("/", 1)
        resource = anchors[package].joinpath(*relative_path.split("/"))
        if not resource.is_file() or resource.read_bytes() != expected_content:
            raise RuntimeError(f"Installed resource is missing or differs from source: {name}")
        if not expected_content.strip():
            raise RuntimeError(f"Bundled YAML resource is empty: {name}")

    cli = Path(sysconfig.get_path("scripts")) / ("llmrouter.exe" if os.name == "nt" else "llmrouter")
    cli_env = os.environ.copy()
    cli_env.pop("PYTHONPATH", None)
    for args, expected_output in (
        (["--help"], None),
        (["--version"], f"LLMRouter CLI v{expected_version}"),
        (["version"], f"LLMRouter version: {expected_version}"),
    ):
        result = subprocess.run(
            [str(cli), *args], check=True, capture_output=True, text=True,
            timeout=30, env=cli_env,
        )
        if expected_output is None:
            if "LLMRouter" not in result.stdout or "serve" not in result.stdout:
                raise RuntimeError("Installed CLI help is incomplete.")
        elif result.stdout.strip() != expected_output:
            raise RuntimeError(f"Unexpected installed CLI version output: {result.stdout!r}")

    print(
        f"Verified {DISTRIBUTION} {expected_version}: installed imports, CLI help/versions, "
        f"and all {len(expected_assets)} bundled YAML resources."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wheel", required=True, type=Path, help="Wheel that was installed")
    parser.add_argument(
        "--source-root", type=Path, default=Path(__file__).resolve().parents[1],
        help="Checkout supplying the expected YAML resource inventory",
    )
    args = parser.parse_args()
    try:
        check_distribution(args.wheel.resolve(), args.source_root.resolve())
    except (RuntimeError, OSError, KeyError, metadata.PackageNotFoundError,
            subprocess.SubprocessError) as error:
        print(f"Distribution check failed: {error}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
