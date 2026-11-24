"""
Download or remove the Hugging Face models used by Nullwalk.

install(names: list[str], root: Path) -> list[str] was written by Claude Sonnet 5.5

Usage (from the repository root):
    uv run python scripts/download_models.py                     install every model in MODELS
    uv run python scripts/download_models.py --install a,b       install only the listed models
    uv run python scripts/download_models.py --uninstall         remove every model in MODELS
    uv run python scripts/download_models.py --uninstall a,b     remove only the listed models
    uv run python scripts/download_models.py --list              print the known names and exit

Models go to models/<name>/ (change with --dir). 
Each folder holds the weights (safetensors), the config and the model's own tokenizer (tokenizer.json, tokenizer_config.json, vocab.json, merges.txt), so installing or uninstalling a model always installs or uninstalls its tokenizer with it. 
After a download the script checks that these files are present.
The exact Hugging Face commit of each model is written to models/<name>/REVISION.

Gated models (wildguard) need a Hugging Face account that accepted the model's terms and a token on this machine: run `hf auth login` once before installing, and `hf auth logout` afterwards on shared accounts.
Without it the gated model is reported as FAILED and the other models still install.
"""

import argparse
import shutil
import sys
import time
from pathlib import Path


# short name -> Hugging Face repository. Add or remove entries here.
MODELS = {
    "qwen2.5-0.5b": "Qwen/Qwen2.5-0.5B-Instruct",
    "qwen2.5-1.5b": "Qwen/Qwen2.5-1.5B-Instruct",
    "qwen2.5-3b": "Qwen/Qwen2.5-3B-Instruct",
    "qwen2.5-7b": "Qwen/Qwen2.5-7B-Instruct",
    "qwen2.5-14b": "Qwen/Qwen2.5-14B-Instruct",
    # refusal judge (gated: one needs to accept the AI2 Responsible Use Guidelines on its Hugging Face page first)
    "wildguard": "allenai/wildguard",
}

ALLOW_PATTERNS = ["*.json", "*.safetensors", "merges.txt", "tokenizer.model", "LICENSE*"]
REQUIRED_FILES = ["config.json", "tokenizer_config.json"]
ALL = "__all__"


def check_complete(target: Path) -> None:
    missing = [f for f in REQUIRED_FILES if not (target / f).is_file()]
    
    if not ((target / "tokenizer.json").is_file() or (target / "tokenizer.model").is_file()):
        missing.append("tokenizer.json or tokenizer.model")
    
    if not any(target.glob("*.safetensors")):
        missing.append("*.safetensors")
    
    if missing:
        raise FileNotFoundError(f"missing after download: {', '.join(missing)}")


def parse_names(value: str) -> list[str]:
    if value == ALL:
        return list(MODELS)
    
    names = [n.strip() for n in value.split(",") if n.strip()]
    unknown = [n for n in names if n not in MODELS]
    
    if unknown or not names:
        sys.exit(f"unknown or empty model list: {value!r}; known: {', '.join(MODELS)}")
    
    return list(dict.fromkeys(names))


def size_gb(path: Path) -> float:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / 1e9


# The method below was written by Claude Sonnet 5.5
def install(names: list[str], root: Path) -> list[str]:
    from huggingface_hub import HfApi, snapshot_download

    api = HfApi()
    lines = []
    
    for name in names:
        repo = MODELS[name]
        target = root / name
        start = time.time()
        
        try:
            sha = api.model_info(repo).sha
            snapshot_download(repo_id=repo, revision=sha, local_dir=target, allow_patterns=ALLOW_PATTERNS)
            check_complete(target)
            (target / "REVISION").write_text(f"{repo}@{sha}\n")
            lines.append(f"installed {name:<14} {size_gb(target):7.2f} GB  {time.time() - start:6.0f} s  {repo}@{sha[:10]}")
        except Exception as e:  # keep going with the other models
            hint = ""
            if "Gated" in type(e).__name__ or "401" in str(e) or "403" in str(e):
                hint = "  (gated: accept the terms on the model page, then run `hf auth login`)"
            
            lines.append(f"FAILED {name:<14} {type(e).__name__}: {str(e).splitlines()[0] if str(e) else ''}{hint}")
    
    return lines


def uninstall(names: list[str], root: Path) -> list[str]:
    lines = []

    for name in names:
        target = root / name
        
        if target.exists():
            gb = size_gb(target)
            shutil.rmtree(target)
            lines.append(f"removed   {name:<14} {gb:7.2f} GB")
        else:
            lines.append(f"absent    {name:<14} (nothing to remove)")
    
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description="Download or remove Nullwalk models.")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--install", nargs="?", const=ALL, metavar="NAMES", help="comma-separated names; default: all")
    group.add_argument("--uninstall", nargs="?", const=ALL, metavar="NAMES", help="comma-separated names; no value: all")
    group.add_argument("--list", action="store_true", help="print the known model names and exit")
    parser.add_argument("--dir", default="models", help="where models are stored (default: models)")
    args = parser.parse_args()

    if args.list:
        for name, repo in MODELS.items():
            print(f"{name:<14} {repo}")
        return

    root = Path(args.dir)
    if args.uninstall is not None:
        lines = uninstall(parse_names(args.uninstall), root)
    else:
        root.mkdir(parents=True, exist_ok=True)
        lines = install(parse_names(args.install or ALL), root)

    print("Summary:")
    print("\n".join(lines))
    if root.exists():
        print(f"total in {root}/: {size_gb(root):.2f} GB")
    if any(line.startswith("FAILED") for line in lines):
        sys.exit(1)


if __name__ == "__main__":
    main()
