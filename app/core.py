"""Sports Commentary Generator: GPT-2 adapted with LoRA (written from scratch) to turn match events into
commentary.

  1. dataset    - simulated football events paired with commentary lines
  2. LoRA       - low-rank adapters injected into GPT-2's attention, trained while GPT-2 stays frozen
  3. UI         - Streamlit page in app/main.py
  4. evaluation - does the commentary mention the right player/team/minute/event, and is it varied?
  5. merging    - fold the adapters into GPT-2's weights for plain, faster inference
"""
from __future__ import annotations

import json
import math
import random
import time
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "dataset"
TRAIN_FILE = DATA_DIR / "train.jsonl"
TEST_FILE = DATA_DIR / "test.jsonl"
ADAPTER_FILE = ROOT / "models" / "lora_adapter.pt"
MERGED_DIR = ROOT / "models" / "merged"
SAVED_FILE = ROOT / "saved_commentary.jsonl"


# --- Milestone 1: Create a small dataset of simulated sports events and commentary pairs.

EVENT_TYPES = ["goal", "penalty_goal", "penalty_miss", "save", "yellow_card", "red_card",
               "substitution", "equaliser", "half_time", "full_time"]

TEAMS = ["Riverside Rovers", "Northgate United", "Harbour City", "Eastwood Athletic", "Kingsbridge FC",
         "Valley Wanderers", "Ironworks Town", "Lakeshore Rangers"]
PLAYERS = ["Okafor", "Silva", "Mehta", "Kowalski", "Haddad", "Larsen", "Nakamura", "Diallo", "Reyes",
           "Fischer", "O'Brien", "Petrov", "Mensah", "Costa", "Varga", "Iyer"]
# unseen in training: used for the test split, so evaluation checks generalisation, not memory
TEST_TEAMS = ["Moorfield Albion", "Cedar Park"]
TEST_PLAYERS = ["Banerjee", "Lindqvist", "Adeyemi", "Moreau"]

COMMENTARY = {
    "goal": ["GOAL! {player} scores for {team} in the {minute}th minute! It's {score} now!",
             "{player} finds the back of the net! {team} lead {score} after {minute} minutes.",
             "What a finish from {player}! {team} make it {score} on {minute} minutes.",
             "It's in! {player} strikes for {team}, and the score is {score}."],
    "penalty_goal": ["{player} steps up... and scores the penalty! {team} {score}, {minute} minutes gone.",
                     "Cool as you like from the spot, {player} converts for {team}. {score}.",
                     "Penalty scored! {player} sends the keeper the wrong way. {team} make it {score}."],
    "penalty_miss": ["{player} misses the penalty! {team} waste a golden chance on {minute} minutes.",
                     "Saved! {player}'s penalty is kept out and {team} stay on {score}.",
                     "Off the post! {player} can't believe it, the penalty stays out for {team}."],
    "save": ["Brilliant save! {player} denies {team} in the {minute}th minute.",
             "{player} at full stretch to keep it {score}! What a stop against {team}.",
             "Fingertips! {player} pushes it wide and {team} are frustrated."],
    "yellow_card": ["{player} goes into the book for {team} after a late challenge on {minute} minutes.",
                    "Yellow card for {player}. {team} need to be careful now.",
                    "The referee shows a yellow to {player} of {team}."],
    "red_card": ["RED CARD! {player} is sent off and {team} are down to ten men on {minute} minutes!",
                 "{player} sees red! {team} will have to defend {score} with ten.",
                 "Straight red for {player}! A huge blow for {team}."],
    "substitution": ["A change for {team}: {player} comes on in the {minute}th minute.",
                     "{player} is on for {team}, fresh legs with the score at {score}.",
                     "Substitution for {team} as {player} enters the game."],
    "equaliser": ["{player} equalises for {team}! It's {score}, and we're level on {minute} minutes!",
                  "All square! {player} levels it for {team}. {score}.",
                  "{team} are back in it! {player} with the equaliser, {score}."],
    "half_time": ["The referee blows for half time. {team} lead {score} at the break.",
                  "Half time here, and it's {score}. {team} will be the happier side.",
                  "That's the end of the first half with {team} on top at {score}."],
    "full_time": ["Full time! {team} win it {score}!",
                  "It's all over! {team} take the points with a {score} result.",
                  "The final whistle goes, {team} {score}. What a match!"],
}


def describe(event: dict) -> str:
    """The text the model reads: one structured line per event."""
    return (f"minute {event['minute']} | {event['type'].replace('_', ' ')} | team {event['team']} | "
            f"player {event['player']} | score {event['score']}")


def random_event(rng: random.Random, teams=TEAMS, players=PLAYERS, etype: str | None = None) -> dict:
    etype = etype or rng.choice(EVENT_TYPES)
    minute = {"half_time": 45, "full_time": 90}.get(etype, rng.randint(1, 90))
    a, b = rng.randint(0, 4), rng.randint(0, 3)
    if etype == "equaliser":
        b = a
    return {"type": etype, "minute": minute, "team": rng.choice(teams), "player": rng.choice(players),
            "score": f"{max(a, b)}-{min(a, b)}" if etype not in ("equaliser",) else f"{a}-{b}"}


def make_pairs(n: int, seed: int, teams=TEAMS, players=PLAYERS) -> list[dict]:
    """n (event, commentary) pairs, balanced over event types."""
    rng = random.Random(seed)
    pairs = []
    for i in range(n):
        ev = random_event(rng, teams, players, EVENT_TYPES[i % len(EVENT_TYPES)])
        line = rng.choice(COMMENTARY[ev["type"]]).format(**ev)
        pairs.append({"event": ev, "input": describe(ev), "commentary": line})
    rng.shuffle(pairs)
    return pairs


def write_jsonl(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(x) for x in Path(path).read_text(encoding="utf-8").splitlines() if x.strip()]


def build_dataset(n_train: int = 600, n_test: int = 60) -> tuple[list[dict], list[dict]]:
    train = make_pairs(n_train, seed=1)
    test = make_pairs(n_test, seed=2, teams=TEST_TEAMS, players=TEST_PLAYERS)
    write_jsonl(train, TRAIN_FILE)
    write_jsonl(test, TEST_FILE)
    return train, test


# --- Milestone 2: Implement LoRA adapter injection and training on a pre-trained LLM (e.g., GPT-2).

class LoRALinear(nn.Module):
    """Wraps a frozen GPT-2 Conv1D/Linear layer:  y = base(x) + (x @ A @ B) * alpha / r.

    A starts small and random, B starts at zero, so at step 0 the adapted model equals GPT-2 exactly.
    Only A and B (r * (d_in + d_out) numbers) are trained.
    """

    def __init__(self, base: nn.Module, r: int = 8, alpha: float = 16.0, dropout: float = 0.05):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad = False
        w = base.weight
        # GPT-2's Conv1D stores weight as (in, out); nn.Linear stores (out, in)
        self.conv1d = type(base).__name__ == "Conv1D"
        d_in, d_out = (w.shape[0], w.shape[1]) if self.conv1d else (w.shape[1], w.shape[0])
        self.r, self.scale = r, alpha / r
        self.A = nn.Parameter(torch.randn(d_in, r) / math.sqrt(d_in))
        self.B = nn.Parameter(torch.zeros(r, d_out))
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + (self.drop(x) @ self.A @ self.B) * self.scale

    def delta_weight(self) -> torch.Tensor:
        """The full-size weight change the adapter represents, in the base layer's layout."""
        d = (self.A @ self.B) * self.scale  # (in, out)
        return d if self.conv1d else d.T

    def merged(self) -> nn.Module:
        """The base layer with the adapter folded into its weight (no extra matmuls at inference)."""
        with torch.no_grad():
            self.base.weight += self.delta_weight()
        return self.base


def inject_lora(model: nn.Module, targets=("c_attn",), r: int = 8, alpha: float = 16.0) -> list[str]:
    """Freeze the whole model, then replace every sub-module whose name ends in a target with LoRALinear."""
    for p in model.parameters():
        p.requires_grad = False
    replaced = []
    for name, module in list(model.named_modules()):
        for child_name, child in list(module.named_children()):
            if child_name in targets:
                setattr(module, child_name, LoRALinear(child, r, alpha))
                replaced.append(f"{name}.{child_name}")
    return replaced


def count_params(model: nn.Module) -> tuple[int, int]:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return trainable, sum(p.numel() for p in model.parameters())


def load_base(name: str = "gpt2"):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(name)
    tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(name)
    return model, tok


def prompt_for(event_text: str) -> str:
    return f"Event: {event_text}\nCommentary:"


def encode_pair(tok, pair: dict, max_len: int = 96) -> tuple[list[int], list[int]]:
    """Token ids and labels; the event prompt is masked (-100) so only the commentary is learned."""
    p = tok(prompt_for(pair["input"]))["input_ids"]
    c = tok(" " + pair["commentary"])["input_ids"] + [tok.eos_token_id]
    ids = (p + c)[:max_len]
    return ids, ([-100] * len(p) + c)[:max_len]


def train_lora(model, tok, pairs: list[dict], epochs: int = 2, lr: float = 2e-3, batch_size: int = 8,
               seed: int = 0, log=print) -> list[dict]:
    """Plain PyTorch loop: AdamW on the adapter parameters only."""
    torch.manual_seed(seed)
    rng = random.Random(seed)
    params = [p for p in model.parameters() if p.requires_grad]
    if not params:
        raise ValueError("no trainable parameters: call inject_lora first")
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
    data = [encode_pair(tok, p) for p in pairs]
    steps = epochs * math.ceil(len(data) / batch_size)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: max(0.05, 1 - s / steps))
    model.train()
    history, step, start = [], 0, time.time()
    for epoch in range(epochs):
        rng.shuffle(data)
        for i in range(0, len(data), batch_size):
            batch = data[i:i + batch_size]
            width = max(len(x) for x, _ in batch)
            ids = torch.tensor([x + [tok.eos_token_id] * (width - len(x)) for x, _ in batch])
            labels = torch.tensor([y + [-100] * (width - len(y)) for _, y in batch])
            mask = torch.tensor([[1] * len(x) + [0] * (width - len(x)) for x, _ in batch])
            loss = model(input_ids=ids, attention_mask=mask, labels=labels).loss
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            step += 1
            if step % 10 == 0 or step == steps:
                history.append({"step": step, "epoch": epoch + 1, "loss": round(loss.item(), 4),
                                "seconds": round(time.time() - start, 1)})
                log(f"step {step}/{steps}  loss {loss.item():.3f}")
    model.eval()
    return history


@torch.no_grad()
def mean_loss(model, tok, pairs: list[dict]) -> float:
    """Average cross-entropy on the commentary tokens (lower = the model expects these lines more)."""
    model.eval()
    losses = []
    for pair in pairs:
        ids, labels = encode_pair(tok, pair)
        losses.append(model(input_ids=torch.tensor([ids]), labels=torch.tensor([labels])).loss.item())
    return sum(losses) / len(losses)


def lora_state(model: nn.Module) -> dict:
    """Only the adapter weights: a few hundred KB instead of GPT-2's 500 MB."""
    return {k: v for k, v in model.state_dict().items() if k.endswith(".A") or k.endswith(".B")}


def save_adapter(model, path: Path = ADAPTER_FILE, meta: dict | None = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state": lora_state(model), "meta": meta or {}}, path)


def load_adapted(base_name: str = "gpt2", path: Path = ADAPTER_FILE):
    """GPT-2 + saved adapter, ready to generate."""
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    meta = ckpt["meta"]
    model, tok = load_base(meta.get("base", base_name))
    inject_lora(model, tuple(meta.get("targets", ["c_attn"])), meta.get("r", 8), meta.get("alpha", 16.0))
    missing = model.load_state_dict(ckpt["state"], strict=False)
    assert not [k for k in missing.missing_keys if k.endswith((".A", ".B"))]
    model.eval()
    return model, tok, meta


@torch.no_grad()
def generate(model, tok, event: dict | str, n: int = 1, temperature: float = 0.8, top_p: float = 0.9,
             max_new_tokens: int = 40, seed: int | None = None) -> list[str]:
    """n commentary options for one event (sampling), or the single most likely line with n=1, temperature=0."""
    text = event if isinstance(event, str) else describe(event)
    ids = tok(prompt_for(text), return_tensors="pt")
    if seed is not None:
        torch.manual_seed(seed)
    out = model.generate(**ids, do_sample=temperature > 0, temperature=temperature or None,
                         top_p=top_p if temperature > 0 else None, max_new_tokens=max_new_tokens,
                         num_return_sequences=n, pad_token_id=tok.eos_token_id)
    lines = []
    for seq in out:
        new = tok.decode(seq[ids["input_ids"].shape[1]:], skip_special_tokens=True)
        lines.append(new.strip().split("\n")[0].strip())
    return lines


# --- Milestone 3: Build a Streamlit interface to input event descriptions and generate commentary.
# The page is app/main.py. These helpers keep it thin (and testable without a browser).

def save_snippet(event: dict, line: str, path: Path = SAVED_FILE) -> dict:
    record = {"event": event, "commentary": line, "saved_at": time.strftime("%Y-%m-%d %H:%M:%S")}
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")
    return record


def load_snippets(path: Path = SAVED_FILE) -> list[dict]:
    return read_jsonl(path) if Path(path).exists() else []


# --- Milestone 4: Test the model's ability to produce relevant and engaging commentary.
# Hint: Model evaluation
def milestone_4():  # TODO(milestone 4)
    raise NotImplementedError("TODO(milestone 4)")


# --- Milestone 5: Implement adapter merging for inference.
# Hint: Adapter merging
def milestone_5():  # TODO(milestone 5)
    raise NotImplementedError("TODO(milestone 5)")
