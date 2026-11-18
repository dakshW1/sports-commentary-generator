"""Training and evaluation commands: `python -m app.cli --help`. The UI is `streamlit run app/main.py`."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from app import core


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="sports-commentary-generator")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("dataset", help="build the simulated event/commentary dataset")
    p.add_argument("--train", type=int, default=600)
    p.add_argument("--test", type=int, default=60)

    p = sub.add_parser("train", help="inject LoRA into GPT-2 and train the adapter")
    p.add_argument("--base", default="gpt2")
    p.add_argument("--r", type=int, default=8)
    p.add_argument("--alpha", type=float, default=16.0)
    p.add_argument("--targets", nargs="+", default=["c_attn"])
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--adapter", type=Path, default=core.ADAPTER_FILE)

    p = sub.add_parser("generate", help="commentary for one event")
    p.add_argument("type", choices=core.EVENT_TYPES)
    p.add_argument("--team", default="Riverside Rovers")
    p.add_argument("--player", default="Okafor")
    p.add_argument("--minute", type=int, default=67)
    p.add_argument("--score", default="2-1")
    p.add_argument("-n", type=int, default=3)
    p.add_argument("--adapter", type=Path, default=core.ADAPTER_FILE)

    p = sub.add_parser("evaluate", help="relevance + variety on held-out events: LoRA vs plain GPT-2")
    p.add_argument("--adapter", type=Path, default=core.ADAPTER_FILE)
    p.add_argument("--limit", type=int, default=60)

    p = sub.add_parser("merge", help="fold the adapter into GPT-2, check it is identical, time it, save it")
    p.add_argument("--adapter", type=Path, default=core.ADAPTER_FILE)
    p.add_argument("--out", type=Path, default=core.MERGED_DIR)

    args = parser.parse_args(argv)

    if args.command == "dataset":
        train, test = core.build_dataset(args.train, args.test)
        print(f"{len(train)} training pairs, {len(test)} test pairs (unseen teams/players)")
        print(json.dumps(train[0], indent=1))
    elif args.command == "train":
        model, tok = core.load_base(args.base)
        replaced = core.inject_lora(model, tuple(args.targets), args.r, args.alpha)
        trainable, total = core.count_params(model)
        print(f"LoRA in {len(replaced)} layers | trainable {trainable:,} of {total:,} "
              f"({trainable / total:.2%})")
        history = core.train_lora(model, tok, core.read_jsonl(core.TRAIN_FILE), args.epochs, args.lr)
        core.save_adapter(model, args.adapter, {"base": args.base, "r": args.r, "alpha": args.alpha,
                                                "targets": args.targets, "history": history,
                                                "trainable": trainable, "total": total})
        print(f"saved adapter -> {args.adapter} ({args.adapter.stat().st_size / 1e6:.2f} MB)")
    elif args.command == "generate":
        model, tok, _ = core.load_adapted(path=args.adapter)
        event = {"type": args.type, "minute": args.minute, "team": args.team, "player": args.player,
                 "score": args.score}
        print(core.describe(event))
        for line in core.generate(model, tok, event, n=args.n):
            print(" -", line)
    elif args.command == "evaluate":
        test = core.read_jsonl(core.TEST_FILE)[:args.limit]
        model, tok, meta = core.load_adapted(path=args.adapter)
        lora = core.evaluate(model, tok, test)
        base, tok = core.load_base(meta.get("base", "gpt2"))
        plain = core.evaluate(base, tok, test)
        print(json.dumps({"lora": lora, "plain_gpt2": plain}, indent=1))
    elif args.command == "merge":
        adapted, tok, _ = core.load_adapted(path=args.adapter)
        text = core.prompt_for(core.describe(core.read_jsonl(core.TEST_FILE)[0]["event"]))
        before = core.time_forward(adapted, tok, text)
        reference, _, _ = core.load_adapted(path=args.adapter)
        n = core.merge_lora(adapted)
        diff = core.max_logit_difference(adapted, reference, tok, text)
        after = core.time_forward(adapted, tok, text)
        core.save_merged(adapted, tok, args.out)
        print(f"merged {n} layers | max logit difference vs adapter model: {diff:.2e}")
        print(f"forward pass: {before:.1f} ms with adapters -> {after:.1f} ms merged")
        print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
