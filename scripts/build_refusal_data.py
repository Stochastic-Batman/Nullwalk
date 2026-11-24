"""
The prompts are the published train/val/test splits of "Refusal in Language Models Is Mediated by a Single Direction" (github.com/andyrdt/refusal_direction, Apache-2.0), which is pinned to one commit, hence the COMMIT constant below.

Harmful prompts come from AdvBench, MaliciousInstruct, TDC2023, HarmBench (train/val) and JailbreakBench and others (test), while harmless prompts come from Alpaca. 

For every split, the harmless set is subsampled with a fixed seed to the size of the harmful set, so both classes are balanced.

Usage (from the repository root):
    uv run python scripts/build_refusal_data.py                  download and write data/refusal/
    uv run python scripts/build_refusal_data.py --source DIR     read the split files from a local clone

Output: data/refusal/{train,val,test}.json, each a JSON object with metadata and a list of examples {"id", "instruction", "label", "category"}. The last lines printed are a summary.
"""

import argparse
import json
import random
import urllib.request

from pathlib import Path

COMMIT = "9d852fae1a9121c78b29142de733cb1340770cc3"  # prompts are pinned to this commit only
BASE_URL = f"https://raw.githubusercontent.com/andyrdt/refusal_direction/{COMMIT}/dataset/splits"
SPLITS = ["train", "val", "test"]


def load_split(kind: str, split: str, source: Path | None) -> list[dict]:
    name = f"{kind}_{split}.json"

    if source is not None:
        return json.loads((source / name).read_text())
    
    with urllib.request.urlopen(f"{BASE_URL}/{name}", timeout=60) as r:
        return json.loads(r.read().decode("utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Build data/refusal/{train,val,test}.json")
    parser.add_argument("--source", type=Path, default=None, help="local dataset/splits directory instead of downloading")
    parser.add_argument("--out", type=Path, default=Path("data/refusal"))
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    seen: dict[str, str] = {}
    lines = []
    
    for split in SPLITS:
        harmful = load_split("harmful", split, args.source)
        harmless = load_split("harmless", split, args.source)
        harmless = rng.sample(harmless, k=min(len(harmful), len(harmless)))

        examples = []
        for label, rows in ((1, harmful), (0, harmless)):
            for i, row in enumerate(rows):
                text = row["instruction"].strip()
                examples.append({"id": f"{split}-{label}-{i:04d}", "instruction": text, "label": label, "category": row.get("category")})

        overlap = sum(1 for e in examples if e["instruction"] in seen and seen[e["instruction"]] != split)
        for e in examples:
            seen.setdefault(e["instruction"], split)

        meta = {"behavior": "refusal", "split": split, "source": f"andyrdt/refusal_direction@{COMMIT}", "seed": args.seed, "label_meaning": {"1": "harmful (D_1)", "0": "harmless (D_0)"}}
        path = args.out / f"{split}.json"
        path.write_text(json.dumps({"meta": meta, "examples": examples}, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
        lines.append(f"{split:<5} harmful={len(harmful):4d} harmless={len(harmless):4d} exact-duplicates-with-earlier-splits={overlap}  -> {path}")

    print("Summary:")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
