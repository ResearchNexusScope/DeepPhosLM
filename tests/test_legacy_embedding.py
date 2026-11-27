"""Synthetic-only regression tests for L-003 (legacy embedding paths).

No real model artefact, data CSV, weight file or notebook is opened.  Each
legacy script is inspected as source text / AST, and its small pure path
helpers are lifted out of that AST and executed in an isolated namespace whose
``__file__`` points into ``tmp_path``.  This keeps the heavy training loops at
the top level of those scripts unexecuted while still testing the real helper
bodies against tiny synthetic CSVs.
"""

from __future__ import annotations

import argparse
import ast
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

READERS = ("evaluate.py", "train.py")
ALL_SCRIPTS = READERS + ("embedder.py",)
HELPER_NAMES = {
    "REPO_ROOT",
    "DEFAULT_EMBEDDING_PATH",
    "DEFAULT_EMBEDDING_DIR",
    "parse_embedding_args",
    "resolve_embedding_path",
    "resolve_embedding_output_path",
    "load_embedding",
    "save_embedding",
}


def _source(filename: str) -> str:
    return (REPO_ROOT / filename).read_text(encoding="utf-8")


def _tree(filename: str) -> ast.Module:
    return ast.parse(_source(filename), filename=filename)


def _calls(tree: ast.Module, name: str) -> list[ast.Call]:
    """Find calls to ``name(...)`` or ``obj.name(...)`` anywhere in *tree*."""
    found = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if (isinstance(func, ast.Name) and func.id == name) or (
            isinstance(func, ast.Attribute) and func.attr == name
        ):
            found.append(node)
    return found


def _string_arguments(node: ast.Call) -> list[str]:
    """String constants among a call's positional arguments."""
    return [
        arg.value
        for arg in node.args
        if isinstance(arg, ast.Constant) and isinstance(arg.value, str)
    ]


def _function(tree: ast.Module, name: str) -> ast.FunctionDef:
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"{name}() not found at module level")


def _helpers(filename: str, fake_file: Path) -> dict:
    """Execute only the pure path helpers from *filename* in a clean namespace."""
    nodes = []
    for node in _tree(filename).body:
        if isinstance(node, ast.FunctionDef) and node.name in HELPER_NAMES:
            nodes.append(node)
        elif isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id in HELPER_NAMES
            for target in node.targets
        ):
            nodes.append(node)
    assert nodes, f"no path helpers found in {filename}"
    module = ast.Module(body=nodes, type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = {
        "__file__": str(fake_file),
        "np": np,
        "pd": pd,
        "Path": Path,
        "argparse": argparse,
    }
    exec(compile(module, str(fake_file), "exec"), namespace)
    return namespace


# ---------------------------------------------------------------------------
# Source-text / AST guards: no undocumented, non-overridable hard-coded path.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("filename", ALL_SCRIPTS)
def test_no_absolute_path_literals(filename: str) -> None:
    source = _source(filename)
    assert not re.search(r"[A-Za-z]:[\\/]", source), filename
    for node in ast.walk(_tree(filename)):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            assert not re.match(r"\s*[A-Za-z]:[\\/]", node.value), node.value


@pytest.mark.parametrize("filename", READERS)
def test_readers_have_no_raw_hard_coded_genfromtxt(filename: str) -> None:
    tree = _tree(filename)
    for call in _calls(tree, "genfromtxt"):
        literals = _string_arguments(call)
        assert not any("MODELS_siam" in text for text in literals), literals
    assert not re.search(
        r"genfromtxt\(\s*r?[\"']MODELS_siam", _source(filename)
    ), filename


def test_embedder_has_no_raw_hard_coded_csv_write() -> None:
    tree = _tree("embedder.py")
    for call in _calls(tree, "to_csv"):
        literals = _string_arguments(call)
        assert not any("MODELS_siam" in text for text in literals), literals
    assert not re.search(
        r"to_csv\(\s*r?[\"']MODELS_siam", _source("embedder.py")
    )
    # The pixel/figure outputs are unrelated; only the embedding write changed.
    assert any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "save_embedding"
        for node in ast.walk(tree)
    )


@pytest.mark.parametrize("filename", ALL_SCRIPTS)
def test_embedding_cli_override_is_documented(filename: str) -> None:
    source = _source(filename)
    assert "REPO_ROOT = Path(__file__).resolve().parent" in source
    add_calls = _calls(_tree(filename), "add_argument")
    embedding_flags = [
        [text for text in _string_arguments(call) if text.startswith("-")]
        for call in add_calls
    ]
    assert any("--embedding" in flags for flags in embedding_flags), filename
    assert any("-EmbeddingPath" in flags for flags in embedding_flags), filename
    assert any(
        keyword.arg == "dest"
        and isinstance(keyword.value, ast.Constant)
        and keyword.value.value == "embedding"
        for call in add_calls
        for keyword in call.keywords
    ), filename


@pytest.mark.parametrize("filename", READERS)
def test_reader_path_parameter_and_missing_file_error(filename: str) -> None:
    source = _source(filename)
    assert "DEFAULT_EMBEDDING_PATH = REPO_ROOT /" in source
    tree = _tree(filename)
    for name in ("resolve_embedding_path", "load_embedding"):
        function = _function(tree, name)
        assert function.args.args[-1].arg == "path"
        assert isinstance(function.args.defaults[-1], ast.Constant)
        assert function.args.defaults[-1].value is None
    raises = [
        node
        for node in ast.walk(_function(tree, "load_embedding"))
        if isinstance(node, ast.Raise)
    ]
    assert raises, filename
    message = " ".join(
        node.value
        for raise_node in raises
        for node in ast.walk(raise_node.exc)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    )
    assert "embedder.py" in message, message
    assert "--embedding" in message, message


def test_embedder_output_parameter_and_parent_directory_creation() -> None:
    source = _source("embedder.py")
    assert "DEFAULT_EMBEDDING_DIR = REPO_ROOT /" in source
    tree = _tree("embedder.py")
    resolver = _function(tree, "resolve_embedding_output_path")
    assert resolver.args.args[0].arg == "run"
    assert resolver.args.args[-1].arg == "path"
    assert isinstance(resolver.args.defaults[-1], ast.Constant)
    assert resolver.args.defaults[-1].value is None
    saver = _function(tree, "save_embedding")
    assert saver.args.args[-1].arg == "path"
    mkdir_calls = [call for call in _calls(tree, "mkdir") if call in ast.walk(saver)]
    assert mkdir_calls, "save_embedding must create the parent directory"
    keywords = {
        keyword.arg: keyword.value.value
        for keyword in mkdir_calls[0].keywords
        if isinstance(keyword.value, ast.Constant)
    }
    assert keywords.get("parents") is True
    assert keywords.get("exist_ok") is True


# ---------------------------------------------------------------------------
# Behavioural tests, all driven by tiny synthetic CSVs under tmp_path.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("filename", READERS)
def test_reader_default_relative_absolute_and_cli_paths(tmp_path: Path, filename: str) -> None:
    helpers = _helpers(filename, tmp_path / filename)
    default_path = tmp_path / "MODELS_siam" / "LEGACY" / "emb_latest_embedding.csv"
    assert helpers["resolve_embedding_path"]() == default_path
    assert helpers["resolve_embedding_path"]("custom/emb.csv") == (
        tmp_path / "custom" / "emb.csv"
    )
    absolute = tmp_path / "absolute" / "emb.csv"
    assert helpers["resolve_embedding_path"](absolute) == absolute
    parsed = helpers["parse_embedding_args"](["--embedding", str(absolute)])
    assert parsed.embedding == str(absolute)
    alias = helpers["parse_embedding_args"](["-EmbeddingPath", str(absolute)])
    assert alias.embedding == str(absolute)


@pytest.mark.parametrize("filename", READERS)
def test_reader_loads_tiny_synthetic_csv(tmp_path: Path, filename: str) -> None:
    helpers = _helpers(filename, tmp_path / filename)
    synthetic = np.array([[0.25, -0.5, 1.0], [2.0, 3.0, -4.0]])
    csv_path = tmp_path / "synthetic_embedding.csv"
    np.savetxt(csv_path, synthetic, delimiter=",")
    np.testing.assert_allclose(helpers["load_embedding"](csv_path), synthetic)

    default_path = tmp_path / "MODELS_siam" / "LEGACY" / "emb_latest_embedding.csv"
    default_path.parent.mkdir(parents=True)
    np.savetxt(default_path, synthetic, delimiter=",")
    np.testing.assert_allclose(helpers["load_embedding"](), synthetic)


@pytest.mark.parametrize("filename", READERS)
def test_reader_missing_file_message_is_actionable(tmp_path: Path, filename: str) -> None:
    helpers = _helpers(filename, tmp_path / filename)
    with pytest.raises(FileNotFoundError) as exc_info:
        helpers["load_embedding"]()
    message = str(exc_info.value)
    assert "MODELS_siam" in message
    assert "embedder.py" in message
    assert "--embedding" in message


def test_embedder_write_creates_parents_and_honours_override(tmp_path: Path) -> None:
    helpers = _helpers("embedder.py", tmp_path / "embedder.py")
    synthetic = np.array([[1.0, 2.0], [3.0, 4.0]])
    default_path = (
        tmp_path / "MODELS_siam" / "LEGACY" / "emb_legacy_run_embedding.csv"
    )
    written = helpers["save_embedding"](synthetic, run="legacy_run")
    assert written == default_path
    assert default_path.is_file()
    np.testing.assert_allclose(np.genfromtxt(default_path, delimiter=","), synthetic)

    override = tmp_path / "nested" / "deeper" / "custom.csv"
    written = helpers["save_embedding"](synthetic, path=override)
    assert written == override and override.is_file()
    np.testing.assert_allclose(np.genfromtxt(override, delimiter=","), synthetic)
    assert helpers["resolve_embedding_output_path"](path="rel/emb.csv") == (
        tmp_path / "rel" / "emb.csv"
    )
    parsed = helpers["parse_embedding_args"](["-EmbeddingPath", str(override)])
    assert parsed.embedding == str(override)
