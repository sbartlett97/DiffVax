import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
import rewrite_edit_prompts as R  # noqa: E402

DATA = os.path.join(os.path.dirname(__file__), "..", "data")


def test_known_captions_become_instructions_and_instructions_are_untouched():
    cap = "A person on a rain-slicked urban alleyway"
    out = R.rewrite(cap)
    assert not out.startswith("A person") and out.startswith("Move the person")
    assert "unchanged" in out
    keep = "Turn this into a pencil sketch"
    assert R.rewrite(keep) == keep


@pytest.mark.skipif(not os.path.exists(os.path.join(DATA, "train", "metadata.jsonl")), reason="dataset not present")
def test_every_caption_in_the_real_dataset_is_mapped():
    caps = set()
    for split in ("train", "validation"):
        p = os.path.join(DATA, split, "metadata.jsonl")
        if os.path.exists(p):
            for l in open(p):
                if l.strip():
                    caps |= {x for x in json.loads(l).get("flux_prompts_orig", json.loads(l)["flux_prompts"]) if x.startswith("A person")}
    assert caps, "expected caption-style prompts"
    unmapped = [c for c in caps if R.rewrite(c) == c]
    assert not unmapped, unmapped
