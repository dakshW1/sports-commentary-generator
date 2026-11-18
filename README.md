# Sports Commentary Generator

[![Built with askThili](https://ask.thili.ai/api/v1/public/projects/2ayp53snpe/badge.svg)](https://ask.thili.ai/b/2ayp53snpe)

This project, "Sports Commentary Generator," provides game developers with a tool to create dynamic and context-aware commentary for simulated sports events. By fine-tuning a pre-trained LLM with LoRA, it generates snippets for events like goal alerts or score changes, proving the effectiveness of the `lfllm-lflora` technique.

## What you are building

**Goal.** You will have built a tool for game developers that uses a LoRA-adapted language model to generate dynamic and context-aware commentary snippets for simulated sports events.

**Done when.** You will have a trained LoRA adapter on a pre-trained LLM, capable of generating relevant and engaging sports commentary from event descriptions via a Streamlit interface. The adapter will be merged for efficient inference, and the model's output will be tested.

## Milestones

Tick these off as you go, and commit after each one.

- [ ] **1. Create a small dataset of simulated sports events and commentary pairs.**  
      Hint: Dataset creation
- [ ] **2. Implement LoRA adapter injection and training on a pre-trained LLM (e.g., GPT-2).**  
      Hint: LoRA adapter injection and training
- [ ] **3. Build a Streamlit interface to input event descriptions and generate commentary.**  
      Hint: Streamlit UI development
- [ ] **4. Test the model's ability to produce relevant and engaging commentary.**  
      Hint: Model evaluation
- [ ] **5. Implement adapter merging for inference.**  
      Hint: Adapter merging

## Run it

You need Python 3.12 and git.

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # then choose a model; see the comments inside
streamlit run app/main.py
pytest
```

Your work goes in `app/core.py` (one stub per milestone) and `tests/test_core.py`. The first tests pass already; they
check the plumbing.

## Settings and secrets

Model settings live in `.env`, which is ignored by git. **Never commit `.env` or paste a key into code.** If a key ever
reaches GitHub, treat it as leaked and make a new one.

## Stretch ideas

- Improve the Streamlit interface with more styling and user feedback.
- Allow the generation of multiple commentary options for a single event.
- Implement a simple mechanism to save generated commentary snippets.
