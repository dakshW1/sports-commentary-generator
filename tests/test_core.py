"""Tests: one block per milestone in README.md. Model tests use `sshleifer/tiny-gpt2`
(same architecture and tokenizer as GPT-2, ~2 MB) so they run in seconds on CPU."""
import py_compile

import pytest
import torch

from app import core

TINY = "sshleifer/tiny-gpt2"


# Milestone 1: Create a small dataset of simulated sports events and commentary pairs.
def test_pairs_are_balanced_and_mention_the_event_facts():
    pairs = core.make_pairs(100, seed=0)
    assert {p["event"]["type"] for p in pairs} == set(core.EVENT_TYPES)
    for p in pairs:
        ev = p["event"]
        assert p["input"] == core.describe(ev) and "{" not in p["commentary"]
        assert ev["player"] in p["commentary"] or ev["team"] in p["commentary"]


def test_test_split_uses_unseen_teams_and_players():
    test = core.make_pairs(30, seed=2, teams=core.TEST_TEAMS, players=core.TEST_PLAYERS)
    assert not {p["event"]["team"] for p in test} & set(core.TEAMS)
    assert not {p["event"]["player"] for p in test} & set(core.PLAYERS)
    eq = [p["event"] for p in test if p["event"]["type"] == "equaliser"]
    assert all(e["score"].split("-")[0] == e["score"].split("-")[1] for e in eq)


def test_dataset_files_exist():
    assert len(core.read_jsonl(core.TRAIN_FILE)) >= 500 and len(core.read_jsonl(core.TEST_FILE)) >= 50


# Milestone 2: Implement LoRA adapter injection and training on a pre-trained LLM (e.g., GPT-2).
@pytest.mark.skip(reason="remove this line when you start milestone 2")
def test_milestone_2():
    raise NotImplementedError("write the test for milestone 2")


# Milestone 3: Build a Streamlit interface to input event descriptions and generate commentary.
@pytest.mark.skip(reason="remove this line when you start milestone 3")
def test_milestone_3():
    raise NotImplementedError("write the test for milestone 3")


# Milestone 4: Test the model's ability to produce relevant and engaging commentary.
@pytest.mark.skip(reason="remove this line when you start milestone 4")
def test_milestone_4():
    raise NotImplementedError("write the test for milestone 4")


# Milestone 5: Implement adapter merging for inference.
@pytest.mark.skip(reason="remove this line when you start milestone 5")
def test_milestone_5():
    raise NotImplementedError("write the test for milestone 5")

