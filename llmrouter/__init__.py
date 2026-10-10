from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("llmrouter-lib")
except PackageNotFoundError:  # pragma: no cover
    # A source checkout without installed metadata has no verified release version.
    __version__ = "0+unknown"
