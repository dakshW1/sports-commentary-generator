# Sports Commentary Generator

[![Built with askThili](https://ask.thili.ai/api/v1/public/projects/2ayp53snpe/badge.svg)](https://ask.thili.ai/b/2ayp53snpe)
[![tests](../../actions/workflows/ci.yml/badge.svg)](../../actions/workflows/ci.yml)

Game developers need short commentary lines for match events: goals, cards, saves, substitutions. This project
adapts **GPT-2 with LoRA, written from scratch** (no PEFT library), so that it turns a structured event into a
commentary line. It comes with a Streamlit page for generating, comparing and saving lines, an evaluation against
plain GPT-2, and adapter merging for faster inference.

```text
Event: minute 48 | penalty goal | team Moorfield Albion | player Moreau | score 2-0
LoRA GPT-2:  Penalty scored! Moreau sends the keeper the wrong way! Moorfield Albion lead 2-0 on 48 minutes.
plain GPT-2: As the Premier League season comes to a close, we have two games against the top two teams ...
```

(Moorfield Albion and Moreau never appear in the training data: they are from the held-out test split.)

## Milestones

- [x] **1. Create a small dataset of simulated sports events and commentary pairs.** 600 training pairs and 60
      test pairs over 10 event types. The test split uses teams and players that never appear in training.
- [x] **2. Implement LoRA adapter injection and training on a pre-trained LLM (e.g., GPT-2).** `LoRALinear`,
      `inject_lora`, and a training loop on the adapter weights only (0.24% of GPT-2's parameters).
- [x] **3. Build a Streamlit interface to input event descriptions and generate commentary.** `app/main.py`
- [x] **4. Test the model's ability to produce relevant and engaging commentary.** Fact recall (player, team,
      event, score) and variety (distinct-2), compared with plain GPT-2.
- [x] **5. Implement adapter merging for inference.** The adapters are folded into GPT-2's weights; the outputs
      are checked to be the same and the speed is measured.

Stretch ideas done: several commentary options per event, saving snippets to a file, and per-line fact badges in
the UI.

## Run it

You need Python 3.12+ and git.

```bash
python3.12 -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt

python -m app.cli merge             # fold the included adapter (models/lora_adapter.pt) into GPT-2
streamlit run app/main.py           # the UI
python -m app.cli generate goal --team "Cedar Park" --player Lindqvist --minute 88 --score 3-2
python -m app.cli evaluate          # LoRA vs plain GPT-2 on the 60 held-out events
pytest

# retrain from scratch (about 4 minutes on a laptop CPU)
python -m app.cli dataset
python -m app.cli train             # r=8, alpha=16, adapters on every attention c_attn layer
```

The trained adapter is only 1.2 MB, so it is committed. The merged model is a full GPT-2 (480 MB), so `merge`
rebuilds it locally. The app also works without merging, using GPT-2 plus the adapter.

## How it works (`app/core.py`)

**1. Dataset.** `random_event` draws an event type, minute, team, player and score (half time is always minute
45; an equaliser always has a level score). `make_pairs` balances the 10 event types and fills 3-4 commentary
templates per type. `describe` turns an event into the line the model reads:
`minute 67 | goal | team Riverside Rovers | player Okafor | score 2-1`.

**2. LoRA from scratch.** `LoRALinear` wraps a frozen GPT-2 layer and adds a low-rank update:
`y = W x + (x A B) * alpha / r`, where `A` is `d_in x r` (small random values) and `B` is `r x d_out` (zeros). So
at step 0 the model is exactly GPT-2; a test checks the logits are identical. GPT-2's `Conv1D` stores its weight
as `(in, out)`, the transpose of `nn.Linear`, and `LoRALinear` handles both. `inject_lora` freezes the whole model
and swaps every `c_attn` (the fused query/key/value projection) for a `LoRALinear`. With r = 8 that is
**294,912 trainable parameters out of 124.7M (0.24%)**. `train_lora` masks the prompt in the labels (`-100`), so
only the commentary is learned, and runs AdamW on the A and B matrices only. `save_adapter` stores just A and B
(1.2 MB).

**3. UI.** `app/main.py` loads the merged model if it exists (otherwise GPT-2 + adapter), takes the event in a
form, generates 1-5 options with a temperature slider, shows which facts each line got right, and saves chosen
lines to `saved_commentary.jsonl`.

**4. Evaluation.** `relevance` checks each line for the player, the team, words that fit the event type (e.g. a
red card must say "red", "sent off" or "ten"), and no score other than the real one. `distinct_n` measures
variety. `evaluate` runs both on the held-out events.

**5. Merging.** `LoRALinear.merged()` adds `A B * alpha / r` into the base weight, and `merge_lora` replaces
every wrapper with its plain layer. The merged model has the same architecture and size as GPT-2, but the
adapter is baked in.

## Results

Training: 2 epochs, 150 steps, batch 8, learning rate 2e-3; loss 2.56 to 0.96, about 4 minutes on a CPU.

| 60 held-out events (unseen teams and players) | plain GPT-2 | **GPT-2 + LoRA** |
|---|---:|---:|
| mentions the team | 8% | **97%** |
| uses words that fit the event | 3% | **87%** |
| mentions the player | 7% | **65%** |
| no wrong score | 87% | **97%** |
| all four facts right | 0% | **53%** |
| distinct-2 (variety) | 0.75 | 0.59 |
| average words | 28 | 12 |

- Plain GPT-2 writes fluent but unrelated sports text (Premier League tables, the Portland Trail Blazers). The
  adapter teaches it the task format and to copy the facts from the event, with names it never saw in training.
- The weakest fact is the player (65%). In the templates the player is sometimes in the middle of a sentence,
  and the model sometimes invents a name instead (e.g. "Brown").
- Variety is lower than plain GPT-2 because the lines follow the template styles, which is the price of being on
  topic. More, and more varied, training lines would help.
- **Merging:** the maximum logit difference between the adapter model and the merged model is 2.4e-4 (float32
  rounding), and a forward pass goes from 39.6 ms to 33.3 ms (16% faster), because the extra `x A B` matmuls are
  gone.

**An experiment that failed:** some lines repeat themselves ("Substitution for Cedar Park. Substitution for Cedar
Park."), so I tried `no_repeat_ngram_size=3` during generation. All facts correct dropped from 53% to 7%. The rule
also counts n-grams in the *prompt*, so it stopped the model from copying "Moorfield Albion" and the score from
the event description, which is exactly what it needs to do. It was reverted.

## Limitations

- The commentary is template-based, so the model learns those styles; real commentary transcripts would make it
  far more varied.
- Football only, and one event at a time; real commentary also refers to what happened earlier in the match.

Built for the askThili drill *Build LoRA from Scratch*. Written with help from an AI coding assistant.
