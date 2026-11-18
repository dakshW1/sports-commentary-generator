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
@pytest.fixture
def tiny():
    torch.manual_seed(0)
    return core.load_base(TINY)


def test_lora_starts_as_identity_and_freezes_the_base(tiny):
    model, tok = tiny
    x = tok("Event: minute 3 | goal", return_tensors="pt")
    before = model(**x).logits
    replaced = core.inject_lora(model, ("c_attn",), r=4)
    assert len(replaced) == model.config.n_layer
    assert torch.allclose(model(**x).logits, before)  # B = 0 -> no change yet
    trainable, total = core.count_params(model)
    names = [n for n, p in model.named_parameters() if p.requires_grad]
    assert names and all(n.endswith((".A", ".B")) for n in names) and trainable < total


def test_prompt_is_masked_in_the_labels(tiny):
    _, tok = tiny
    pair = core.make_pairs(1, seed=0)[0]
    ids, labels = core.encode_pair(tok, pair)
    n_prompt = len(tok(core.prompt_for(pair["input"]))["input_ids"])
    assert labels[:n_prompt] == [-100] * n_prompt and labels[n_prompt:] == ids[n_prompt:]


PAIRS = core.make_pairs(40, seed=0)


@pytest.fixture(scope="module")
def trained():
    torch.manual_seed(0)
    model, tok = core.load_base(TINY)
    core.inject_lora(model, ("c_attn", "c_proj"), r=4, alpha=8)
    before = core.mean_loss(model, tok, PAIRS)
    history = core.train_lora(model, tok, PAIRS, epochs=3, lr=5e-2, batch_size=8, log=lambda *_: None)
    return model, tok, {"history": history, "before": before, "after": core.mean_loss(model, tok, PAIRS)}


def test_training_moves_only_the_adapter_and_lowers_loss(trained):
    model, _, stats = trained
    assert stats["history"] and stats["after"] < stats["before"]
    assert any(m.B.abs().sum() > 0 for m in model.modules() if isinstance(m, core.LoRALinear))


def test_adapter_save_and_load(trained, tmp_path):
    model, tok, _ = trained
    path = tmp_path / "a.pt"
    core.save_adapter(model, path, {"base": TINY, "r": 4, "alpha": 8, "targets": ["c_attn", "c_proj"]})
    loaded, _, meta = core.load_adapted(path=path)
    x = tok("Event: minute 10 | save", return_tensors="pt")
    assert meta["r"] == 4 and torch.allclose(model(**x).logits, loaded(**x).logits, atol=1e-5)


# Milestone 3: Build a Streamlit interface to input event descriptions and generate commentary.
def test_page_compiles_and_snippets_are_saved(tmp_path):
    py_compile.compile("app/main.py", doraise=True)
    ev = core.make_pairs(1, seed=3)[0]["event"]
    core.save_snippet(ev, "What a goal!", tmp_path / "s.jsonl")
    assert core.load_snippets(tmp_path / "s.jsonl")[0]["commentary"] == "What a goal!"
    assert core.load_snippets(tmp_path / "missing.jsonl") == []


def test_generate_returns_n_options(trained):
    model, tok, _ = trained
    lines = core.generate(model, tok, core.make_pairs(1, seed=4)[0]["event"], n=3, seed=0,
                          max_new_tokens=10)
    assert len(lines) == 3 and all(isinstance(line, str) for line in lines)


# Milestone 4: Test the model's ability to produce relevant and engaging commentary.
def test_relevance_checks_facts():
    for p in core.make_pairs(50, seed=0):  # every training template passes the event-word check
        assert core.relevance(p["event"], p["commentary"])["event"]
    ev = {"type": "red_card", "minute": 70, "team": "Cedar Park", "player": "Moreau", "score": "1-0"}
    assert all(core.relevance(ev, "RED CARD! Moreau is sent off, Cedar Park still lead 1-0").values())
    bad = core.relevance(ev, "Silva scores! 2-2")
    assert not bad["player"] and not bad["event"] and not bad["no_wrong_score"]


def test_distinct_n_measures_variety():
    assert core.distinct_n(["a b c", "a b c"]) == 0.5
    assert core.distinct_n(["a b c", "d e f"]) == 1.0


def test_evaluate_reports_rates(trained):
    model, tok, _ = trained
    res = core.evaluate(model, tok, core.make_pairs(4, seed=5))
    assert res["n"] == 4 and 0 <= res["all_facts"] <= 1 and 0 <= res["distinct_2"] <= 1


# Milestone 5: Implement adapter merging for inference.
def test_merging_gives_the_same_outputs_without_lora_layers(trained, tmp_path):
    model, tok, _ = trained
    path = tmp_path / "a.pt"
    core.save_adapter(model, path, {"base": TINY, "r": 4, "alpha": 8, "targets": ["c_attn", "c_proj"]})
    reference, _, _ = core.load_adapted(path=path)
    merged, _, _ = core.load_adapted(path=path)
    n = core.merge_lora(merged)
    assert n == 3 * merged.config.n_layer  # c_attn + attention c_proj + MLP c_proj
    assert not any(isinstance(m, core.LoRALinear) for m in merged.modules())
    assert core.max_logit_difference(merged, reference, tok, "Event: minute 5 | goal") < 1e-4
    core.save_merged(merged, tok, tmp_path / "merged")
    again, _ = core.load_merged(tmp_path / "merged")
    assert core.max_logit_difference(again, reference, tok, "Event: minute 5 | goal") < 1e-4


from app.cli import main  # noqa: E402


def test_cli_dataset_and_help(capsys, monkeypatch, tmp_path):
    monkeypatch.setattr(core, "TRAIN_FILE", tmp_path / "train.jsonl")
    monkeypatch.setattr(core, "TEST_FILE", tmp_path / "test.jsonl")
    main(["dataset", "--train", "20", "--test", "5"])
    assert "20 training pairs, 5 test pairs" in capsys.readouterr().out
