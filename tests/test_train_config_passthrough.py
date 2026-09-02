"""Regression test for a real bug: scripts/train.py's immunization_config
dict (now scripts/train.py::build_immunization_config) is a curated SUBSET of
the full YAML config, and DiffVaxImmunization only ever sees that subset via
self._config. Two keys were added to diffvax_immunization.py's config reads
(sd3_attack.masked_attack_probability, num_inference_steps) without being
added to this subset — both silently fell back to their .get(...) call's
hardcoded default for every real training run via scripts/train.py, with no
error, no warning. num_inference_steps happened to match its own default (4)
so it went unnoticed; masked_attack_probability did not (config said 0.5,
runtime used 0.0) — the masked-attack feature never actually fired.

This test statically extracts every top-level key self._config.get(...) or
self._config[...] reads in diffvax_immunization.py and asserts
build_immunization_config() forwards all of them, so a newly-added config
read can't reintroduce this exact gap unnoticed.
"""

import ast
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

_IMMUNIZATION_PATH = os.path.join(
    os.path.dirname(__file__), "..", "src", "diffvax", "immunization",
    "diffvax_immunization.py",
)


def _find_self_config_keys(path):
    """Statically find every literal string key in self._config.get("key", ...)
    or self._config["key"] anywhere in the given source file."""
    tree = ast.parse(open(path).read(), filename=path)
    keys = set()

    for node in ast.walk(tree):
        # self._config.get("key", default)
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "get"
            and isinstance(node.func.value, ast.Attribute)
            and node.func.value.attr == "_config"
            and node.args
            and isinstance(node.args[0], ast.Constant)
            and isinstance(node.args[0].value, str)
        ):
            keys.add(node.args[0].value)
        # self._config["key"]
        if (
            isinstance(node, ast.Subscript)
            and isinstance(node.value, ast.Attribute)
            and node.value.attr == "_config"
        ):
            sl = node.slice
            if isinstance(sl, ast.Constant) and isinstance(sl.value, str):
                keys.add(sl.value)

    return keys


def test_build_immunization_config_forwards_every_self_config_read():
    from train import build_immunization_config

    required_keys = _find_self_config_keys(_IMMUNIZATION_PATH)
    assert required_keys, "sanity check: the AST scan should have found some keys"

    # Full-shape config with every plausible top-level key present, so
    # build_immunization_config() can run without KeyErrors on required fields.
    full_config = {
        "iter_num": 1,
        "learning_rate": 1e-5,
        "immunization_model": "diffvax",
        "alpha": 4,
        "batch_size": 1,
        "train_all": True,
    }
    result = build_immunization_config(full_config)

    missing = required_keys - result.keys()
    assert not missing, (
        f"diffvax_immunization.py reads self._config keys {missing} that "
        f"build_immunization_config() does NOT forward — these will silently "
        f"fall back to their .get(...) call's hardcoded default for every "
        f"real training run via scripts/train.py, regardless of the YAML "
        f"config. Add {missing} to build_immunization_config()."
    )
