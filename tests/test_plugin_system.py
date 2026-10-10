#!/usr/bin/env python3
"""Offline tests for discovery, loading, and CLI registration of real plugins."""

from __future__ import annotations

import importlib.util
import json
import multiprocessing as mp
import random
import shutil
import sys
from pathlib import Path
from unittest.mock import Mock

import pytest
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PLUGIN_NAMES = ("randomrouter", "thresholdrouter")


def _is_fixture_plugin_module(name: str) -> bool:
    return any(name == plugin or name.startswith(plugin + ".") for plugin in PLUGIN_NAMES)


@pytest.fixture(autouse=True)
def preserve_random_state():
    """RandomRouter seeds Python's global generator; restore it after each test."""
    state = random.getstate()
    try:
        yield
    finally:
        random.setstate(state)


@pytest.fixture
def plugin_registry(tmp_path, monkeypatch):
    """Discover real example plugins without scanning optional or user plugins."""
    from llmrouter import plugin_system

    plugin_dir = tmp_path / "plugins"
    plugin_dir.mkdir()
    for name in PLUGIN_NAMES:
        target = plugin_dir / name
        target.mkdir()
        for filename in ("__init__.py", "router.py", "trainer.py"):
            shutil.copyfile(PROJECT_ROOT / "custom_routers" / name / filename, target / filename)

    registry = plugin_system.PluginRegistry()
    monkeypatch.setattr(plugin_system, "_global_registry", registry)
    # Discovery prepends its plugin directory. Keep that path local to this test.
    monkeypatch.setattr(sys, "path", sys.path.copy())
    original_modules = {
        name: module for name, module in tuple(sys.modules.items())
        if _is_fixture_plugin_module(name)
    }
    for name in original_modules:
        sys.modules.pop(name)

    try:
        discovered = plugin_system.discover_and_register_plugins(
            plugin_dirs=[str(plugin_dir)], verbose=True
        )
        assert discovered is registry
        yield registry
    finally:
        for name in tuple(sys.modules):
            if _is_fixture_plugin_module(name):
                sys.modules.pop(name, None)
        sys.modules.update(original_modules)


@pytest.fixture
def router_config(tmp_path):
    """Load two local candidates through MetaRouter's real DataLoader."""
    candidates = tmp_path / "llm.json"
    candidates.write_text(json.dumps({
        "small": {"input_price": 0.1, "output_price": 0.1},
        "large": {"input_price": 1.0, "output_price": 1.0},
    }), encoding="utf-8")
    config = tmp_path / "config.yaml"
    config.write_text(yaml.safe_dump({
        "data_path": {"llm_data": str(candidates)},
        "hparam": {"seed": 42},
    }), encoding="utf-8")
    return str(config)


def test_plugin_discovery(plugin_registry):
    from llmrouter.models.base_trainer import BaseTrainer
    from llmrouter.models.meta_router import MetaRouter

    assert set(plugin_registry.get_router_names()) == set(PLUGIN_NAMES)
    expected_classes = {
        "randomrouter": ("RandomRouter", "RandomRouterTrainer"),
        "thresholdrouter": ("ThresholdRouter", "ThresholdRouterTrainer"),
    }
    for name, (router_name, trainer_name) in expected_classes.items():
        entry = plugin_registry.get_router(name)
        assert entry is not None, f"Plugin {name} was not discovered"
        router_class, trainer_class = entry
        assert router_class.__name__ == router_name
        assert issubclass(router_class, MetaRouter)
        assert trainer_class is not None
        assert trainer_class.__name__ == trainer_name
        assert issubclass(trainer_class, BaseTrainer)
        assert callable(router_class.route_single)
        assert callable(router_class.route_batch)
        assert plugin_registry.get_router(name.upper()) == entry
    assert plugin_registry.get_router("missing-plugin") is None


def _assert_random_result(result, query, candidates):
    assert result["query"] == query["query"]
    assert result["model_name"] in candidates
    assert result["predicted_llm"] == result["model_name"]
    assert result["predicted_llm_name"] == result["model_name"]
    assert result["method"] == "random"
    assert result["confidence"] == 1.0


def test_router_loading(plugin_registry, router_config):
    entry = plugin_registry.get_router("randomrouter")
    assert entry is not None
    router_class, _ = entry
    router = router_class(yaml_path=router_config)

    assert set(router.llm_names) == {"small", "large"}
    assert router.query_embedding_data is None
    assert router.routing_data_train is None
    assert router.routing_data_test is None
    query = {"query": "What is machine learning?"}
    result = router.route_single(query)
    _assert_random_result(result, query, router.llm_names)

    batch = [{"query": "Explain recursion"}, {"query": "What is an API?"}]
    results = router.route_batch(batch)
    assert len(results) == len(batch)
    for item, output in zip(batch, results, strict=True):
        _assert_random_result(output, item, router.llm_names)
    assert router.route_batch([]) == []

    # Re-loading the same seed must replay single and batch decisions.
    reloaded = router_class(yaml_path=router_config)
    assert reloaded.route_single(query) == result
    assert reloaded.route_batch(batch) == results


def test_cli_integration(plugin_registry, router_config, monkeypatch):
    from llmrouter import plugin_system

    discover = Mock(return_value=plugin_registry)
    monkeypatch.setattr(plugin_system, "discover_and_register_plugins", discover)
    # CLI import configures worker startup. This test only exercises registration.
    monkeypatch.setattr(mp, "set_start_method", lambda *args, **kwargs: None)
    monkeypatch.setenv("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

    # Execute a fresh module so a previously imported CLI registry cannot hide
    # failures, and do not replace the shared CLI module used by other tests.
    spec = importlib.util.spec_from_file_location(
        "llmrouter.cli._plugin_system_test",
        PROJECT_ROOT / "llmrouter" / "cli" / "router_inference.py",
    )
    assert spec is not None and spec.loader is not None
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)

    discover.assert_called_once_with(verbose=False)
    for name in PLUGIN_NAMES:
        assert cli.ROUTER_REGISTRY[name] is plugin_registry.get_router(name)[0]

    router = cli.load_router("RandomRouter", router_config)
    assert isinstance(router, plugin_registry.get_router("randomrouter")[0])
    query = {"query": "Route locally without calling a provider"}
    _assert_random_result(router.route_single(query), query, router.llm_names)
    with pytest.raises(ValueError, match="Unknown router"):
        cli.load_router("missing-plugin", router_config)


def main():
    return pytest.main([__file__, "-v"])


if __name__ == "__main__":
    raise SystemExit(main())
