"""Static + tiny-synthetic regression tests for ``finalised_application.py``.

The legacy script performs file reads (and trains models) at import time, so
these tests deliberately NEVER import it.  Instead they parse its source with
``ast`` and assert the four audited defects stay fixed.  The only executable
code is a single pure helper (``get_encoded_motifs``) extracted from the AST
and run against tiny in-memory token lists -- no dataset, model or Drive file
is touched, and the whole module runs in well under a second.
"""

from __future__ import annotations

import argparse
import ast
import os
import re
from collections import Counter
from pathlib import Path

import pytest

APP_PATH = Path(__file__).resolve().parents[1] / "finalised_application.py"
SOURCE = APP_PATH.read_text(encoding="utf-8")
TREE = ast.parse(SOURCE, filename=str(APP_PATH))


# --------------------------------------------------------------------------- #
# AST helpers
# --------------------------------------------------------------------------- #
def _function_node(name):
    for node in TREE.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"function {name!r} not found at module level")


def _top_level_for(iter_name):
    for node in TREE.body:
        if (
            isinstance(node, ast.For)
            and isinstance(node.iter, ast.Name)
            and node.iter.id == iter_name
        ):
            return node
    raise AssertionError(f"top-level for-loop over {iter_name!r} not found")


def _target_names(target):
    if isinstance(target, ast.Name):
        return [target.id]
    if isinstance(target, (ast.Tuple, ast.List)):
        return [n for elt in target.elts for n in _target_names(elt)]
    return []


def _assign(target_name, root=None):
    root = TREE if root is None else root
    for node in ast.walk(root):
        if isinstance(node, ast.Assign) and any(
            target_name in _target_names(t) for t in node.targets
        ):
            return node
    raise AssertionError(f"assignment to {target_name!r} not found")


def _calls(root, name):
    found = []
    for node in ast.walk(root):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Name) and func.id == name:
            found.append(node)
        elif isinstance(func, ast.Attribute) and func.attr == name:
            found.append(node)
    return found


def _attr_calls(root, owner, method):
    found = []
    for node in ast.walk(root):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == method
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == owner
        ):
            found.append(node)
    return found


def _string_constants(root):
    return [
        node.value
        for node in ast.walk(root)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    ]


def _is_name(node, name):
    return isinstance(node, ast.Name) and node.id == name


def _unparse(node):
    return ast.unparse(node)


# --------------------------------------------------------------------------- #
# Defect 1 -- no held-out data in the vocabulary
# --------------------------------------------------------------------------- #
def test_vocabulary_comes_from_train_only():
    train_loop = _top_level_for("train_motifs")
    test_loop = _top_level_for("test_motifs")

    # Exactly one all_grams.extend, and it lives inside the TRAIN loop.
    extends = [
        call
        for call in _calls(TREE, "extend")
        if isinstance(call.func, ast.Attribute)
        and _is_name(call.func.value, "all_grams")
    ]
    assert len(extends) == 1, "all_grams must be accumulated in one place only"
    ext = extends[0]
    assert train_loop.lineno <= ext.lineno <= train_loop.end_lineno
    assert not (test_loop.lineno <= ext.lineno <= test_loop.end_lineno)

    # The held-out loop must not even mention all_grams.
    assert not any(_is_name(n, "all_grams") for n in ast.walk(test_loop))

    # vocab_size is still derived from the (train-only) grammar list.
    vocab_assign = _assign("vocab_size")
    assert any(_is_name(n, "all_grams") for n in ast.walk(vocab_assign.value))

    # The test encoding call reuses the vocabulary fitted by the train call.
    calls = _calls(TREE, "get_encoded_motifs")
    assert len(calls) == 2, "expected exactly one train and one test encoding call"
    train_assign = _assign("train_encoded_motifs")
    train_targets = [
        elt.id
        for elt in train_assign.targets[0].elts
        if isinstance(elt, ast.Name)
    ]
    assert "gram_to_int" in train_targets

    test_assign = _assign("test_encoded_motifs")
    test_call = test_assign.value
    assert isinstance(test_call, ast.Call)
    keywords = {kw.arg: kw.value for kw in test_call.keywords}
    assert "gram_to_int" in keywords, "test encoding must reuse the train vocabulary"
    assert _is_name(keywords["gram_to_int"], "gram_to_int")
    assert _is_name(test_call.args[0], "test_grammed_motifs")
    assert _is_name(test_call.args[1], "all_grams")
    # The train vocabulary is passed explicitly; no re-derivation from test.
    assert "gram_to_int" not in {
        kw.arg for kw in train_assign.value.keywords
    }


# --------------------------------------------------------------------------- #
# Defect 2 -- ROC-AUC is no longer labelled accuracy
# --------------------------------------------------------------------------- #
def test_microROC_value_is_named_roc_auc_and_computation_unchanged():
    roc_calls = _calls(TREE, "get_microROC")
    assert roc_calls, "get_microROC must still be called"
    for call in roc_calls:
        parent_assign = None
        for node in ast.walk(TREE):
            if isinstance(node, ast.Assign) and any(
                c is call for c in ast.walk(node.value)
            ):
                parent_assign = node
                break
        assert parent_assign is not None
        target_names = [
            t.id for t in parent_assign.targets if isinstance(t, ast.Name)
        ]
        assert target_names == ["roc_auc"], target_names
        assert _unparse(call) == "get_microROC(y_test, y_score)"

    assert "acc = get_microROC" not in SOURCE
    train_model = _function_node("train_model")
    scope = [
        t.targets[0].id
        for t in train_model.body
        if isinstance(t, ast.Assign) and isinstance(t.targets[0], ast.Name)
    ]
    assert "train_aucs" in scope and "val_aucs" in scope
    assert "train_accs" not in scope and "val_accs" not in scope
    assert not any(_is_name(n, "running_acc") for n in ast.walk(train_model))

    # The metric itself is untouched (micro-averaged ROC AUC).
    metric_fn = _function_node("get_microROC")
    metric_src = ast.get_source_segment(SOURCE, metric_fn)
    assert "roc_curve(y_test.ravel(), y_score.ravel())" in metric_src
    assert "return auc(fpr_micro, tpr_micro)" in metric_src


def test_training_print_reports_auc_not_accuracy():
    train_model = _function_node("train_model")
    assert not any("ACC:" in text for text in _string_constants(train_model))
    assert "ACC:" not in SOURCE

    auc_prints = [
        call
        for call in _calls(train_model, "print")
        if any("AUC:" in text for text in _string_constants(call))
    ]
    assert len(auc_prints) == 1
    # The format string is applied with `% (...)`; roc_auc sits in that tuple.
    assert any(_is_name(n, "roc_auc") for n in ast.walk(auc_prints[0]))


# --------------------------------------------------------------------------- #
# Defect 3 -- partial final batches
# --------------------------------------------------------------------------- #
def test_partial_batch_reshape_uses_runtime_batch_size():
    assert re.search(r"reshape\(\s*batch_size\s*,", SOURCE) is None
    train_model = _function_node("train_model")
    reshapes = [
        call
        for call in _calls(train_model, "reshape")
        if isinstance(call.func, ast.Attribute)
    ]
    assert len(reshapes) == 1
    reshape = reshapes[0]
    assert _unparse(reshape.args[0]) == "inputs.shape[0]"
    assert _unparse(reshape.args[1]) == "len(AMINOS)"
    assert _unparse(reshape.args[2]) == "len(train_motifs[0])"
    assert not any(_is_name(n, "batch_size") for n in ast.walk(reshape))


# --------------------------------------------------------------------------- #
# Defect 4 -- UMAP projection is visualisation-only
# --------------------------------------------------------------------------- #
def test_umap_fit_on_train_transform_test_for_visualisation():
    assert re.search(r"fit_transform\(\s*embedded\s*\)", SOURCE) is None
    assert not _attr_calls(TREE, "umapper", "fit_transform")

    fit_calls = _attr_calls(TREE, "umapper", "fit")
    transform_calls = _attr_calls(TREE, "umapper", "transform")
    assert len(fit_calls) == 1
    assert len(transform_calls) == 2
    assert _unparse(fit_calls[0].args[0]) == "train_embedded"
    assert {_unparse(c.args[0]) for c in transform_calls} == {
        "train_embedded",
        "test_embedded",
    }

    assert _unparse(_assign("n_train_embedded").value) == "len(train_encoded_motifs)"
    assert _unparse(_assign("train_embedded").value) == "embedded[:n_train_embedded]"
    assert _unparse(_assign("test_embedded").value) == "embedded[n_train_embedded:]"
    assert _unparse(_assign("pos_umap").value) == "np.vstack([train_umap, test_umap])"

    # Model inputs are unchanged: the embedding still comes from all motifs.
    embedded_call = _assign("embedded").value
    assert isinstance(embedded_call, ast.Call)
    assert _unparse(embedded_call.func) == "get_embedding"
    assert _is_name(embedded_call.args[1], "all_encoded_motifs")
    assert _unparse(_assign("all_encoded_motifs").value) == (
        "train_encoded_motifs + test_encoded_motifs"
    )


# --------------------------------------------------------------------------- #
# Unknown-token fallback -- executed on tiny synthetic grams (no module import)
# --------------------------------------------------------------------------- #
def test_unknown_token_fallback_reuses_train_vocabulary():
    unk_assign = _assign("UNKNOWN_TOKEN_ID")
    assert isinstance(unk_assign.value, ast.Constant)
    unknown_id = unk_assign.value.value
    assert unknown_id == 0, "id 0 is the reserved padding/unknown token"

    helper = _function_node("get_encoded_motifs")
    helper_src = ast.get_source_segment(SOURCE, helper)
    assert ".get(gram, UNKNOWN_TOKEN_ID)" in helper_src

    namespace = {"Counter": Counter, "UNKNOWN_TOKEN_ID": unknown_id}
    exec(compile(helper_src, "<get_encoded_motifs>", "exec"), namespace)
    encode = namespace["get_encoded_motifs"]

    train_grams = [list("ABA"), list("ACB")]  # train tokens: A, B, C
    train_tokens = [g for motif in train_grams for g in motif]
    encoded_train, mapping, _ = encode(train_grams, train_tokens)
    assert set(mapping.values()) == {1, 2, 3}
    assert [len(row) for row in encoded_train] == [3, 3]  # 1-mer shape preserved

    test_grams = [list("AZ"), list("ZQ")]  # Z and Q were never seen in training
    encoded_test, mapping_after, _ = encode(
        test_grams, train_tokens, gram_to_int=mapping
    )
    assert encoded_test == [[mapping["A"], unknown_id], [unknown_id, unknown_id]]
    assert [len(row) for row in encoded_test] == [2, 2]  # 1-mer shape preserved
    assert mapping_after == mapping, "test tokens must not enter the vocabulary"


# --------------------------------------------------------------------------- #
# Generated embedding artefact -- configurable path, loud failure
# --------------------------------------------------------------------------- #
def test_embedding_reads_use_resolver_with_repo_root_default_and_overrides():
    # No bare hard-coded artefact path may be handed to a file read.
    for call in _calls(TREE, "genfromtxt"):
        for arg in call.args:
            assert not (
                isinstance(arg, ast.Constant)
                and "emb_latest_embedding.csv" in str(arg.value)
            )
    assert "MODELS_siam/LEGACY/emb_latest_embedding.csv" not in SOURCE

    resolved_reads = [
        call
        for call in _calls(TREE, "genfromtxt")
        if any(
            isinstance(arg, ast.Call)
            and _unparse(arg.func) == "resolve_embedding_path"
            for arg in call.args
        )
    ]
    assert len(resolved_reads) == 2, "both embedding read sites must use the resolver"

    env_name = _assign("EMBEDDING_ENV_VAR").value.value
    assert env_name, "an explicit environment-variable name must be documented"
    assert _unparse(_assign("REPO_ROOT").value) == "Path(__file__).resolve().parent"

    default = _assign("DEFAULT_EMBEDDING_RELPATH").value
    assert isinstance(default, ast.BinOp) and isinstance(default.op, ast.Div)
    segments = {
        node.value
        for node in ast.walk(default)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    }
    assert segments == {"MODELS_siam", "LEGACY", "emb_latest_embedding.csv"}

    resolver_src = ast.get_source_segment(SOURCE, _function_node("resolve_embedding_path"))
    assert "os.environ.get(EMBEDDING_ENV_VAR)" in resolver_src
    assert "_embedding_cli_override()" in resolver_src
    assert "path.is_file()" in resolver_src  # missing-file branch is real
    cli_src = ast.get_source_segment(SOURCE, _function_node("_embedding_cli_override"))
    assert "--embedding" in cli_src


def test_embedding_resolver_override_and_missing_file_error(tmp_path, monkeypatch):
    """Execute just the two pure path helpers from the AST, with synthetic files."""
    env_name = _assign("EMBEDDING_ENV_VAR").value.value
    namespace = {
        "os": os,
        "argparse": argparse,
        "Path": Path,
        "REPO_ROOT": Path(__file__).resolve().parents[1],
        "EMBEDDING_ENV_VAR": env_name,
        "DEFAULT_EMBEDDING_RELPATH": (
            Path("MODELS_siam") / "LEGACY" / "emb_latest_embedding.csv"
        ),
    }
    for name in ("_embedding_cli_override", "resolve_embedding_path"):
        node = _function_node(name)
        exec(
            compile(ast.get_source_segment(SOURCE, node), "<embedding-helper>", "exec"),
            namespace,
        )

    synthetic = tmp_path / "synthetic_embedding.csv"
    synthetic.write_text("0,1\n1,0\n")  # tiny synthetic artefact, no real data

    assert namespace["_embedding_cli_override"](
        ["--embedding", str(synthetic), "--unrelated"]
    ) == str(synthetic)
    assert namespace["resolve_embedding_path"](str(synthetic)) == synthetic

    monkeypatch.setenv(env_name, str(synthetic))
    assert namespace["resolve_embedding_path"]() == synthetic

    with pytest.raises(FileNotFoundError) as exc:
        namespace["resolve_embedding_path"](str(tmp_path / "does-not-exist.csv"))
    message = str(exc.value)
    assert "not found" in message
    assert "embedder.py" in message
    assert "--embedding" in message
    assert env_name in message
