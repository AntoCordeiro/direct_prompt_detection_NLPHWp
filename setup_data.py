"""One-command data setup for a fresh clone.

    python setup_data.py                 # check Hugging Face access, download the datasets, build data/
    python setup_data.py --check         # only check login and access, download nothing
    python setup_data.py --with-models   # also pre-download every model used by run_all.py (~15 GB)
    python setup_data.py --force         # rebuild data/ even if it already exists

Three of the resources are gated on the Hugging Face Hub (HackAPrompt, Prompt-Guard-86M, Llama-Guard-3-1B), so you need
an account that has accepted their terms and a token from `hf auth login` (see README). The script tells you exactly
which request is still missing.
"""
import argparse
import sys

from huggingface_hub import HfApi, get_token, snapshot_download
from huggingface_hub.errors import GatedRepoError, HfHubHTTPError, RepositoryNotFoundError

import config as C

# (repo id, repo type, needed for)
DATASETS = [
    ("hackaprompt/hackaprompt-dataset", "dataset", "attacks (gated: accept the terms)"),
    ("deepset/prompt-injections", "dataset", "attacks + benign"),
    ("leolee99/NotInject", "dataset", "benign prompts with trigger words"),
    ("yahma/alpaca-cleaned", "dataset", "generic benign prompts"),
]
MODELS = [
    (C.PROMPT_GUARD, "model", "encoder detector (gated: accept the terms)"),
    (C.LLAMA_GUARD, "model", "Llama Guard detector (gated: accept the terms)"),
    (C.PROTECTAI, "model", "encoder detector"),
    ("microsoft/deberta-v3-base", "model", "fine-tuned DeBERTa"),
    (C.QWEN_1_5B, "model", "Qwen 1.5B"),
    (C.QWEN_3B, "model", "Qwen 3B"),
]
EXPECTED_SPLITS = ["train", "validation", "bench_hackaprompt", "bench_deepset_notinject", "notinject_heldout"]


def check_access(api, repos):
    """Return the list of repos the current token cannot read, printing the reason for each."""
    missing = []
    for repo, kind, note in repos:
        try:
            api.auth_check(repo, repo_type=kind)
            print(f"  ok       {repo}")
        except GatedRepoError:
            print(f"  GATED    {repo}: open https://huggingface.co/{'datasets/' if kind == 'dataset' else ''}{repo} "
                  f"while logged in and accept the terms / request access, then wait for approval")
            missing.append(repo)
        except RepositoryNotFoundError:
            print(f"  DENIED   {repo}: not found or no access (check the token and the access request)")
            missing.append(repo)
        except HfHubHTTPError as e:
            print(f"  ERROR    {repo}: {e}")
            missing.append(repo)
    return missing


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="only check login and access")
    ap.add_argument("--with-models", action="store_true", help="also download all model weights")
    ap.add_argument("--force", action="store_true", help="rebuild data/ even if the splits already exist")
    a = ap.parse_args()

    if not get_token():
        sys.exit("No Hugging Face token found. Create one at https://huggingface.co/settings/tokens (read access) and run "
                 "`hf auth login` (or set HF_TOKEN), then re-run. See the README section 'Hugging Face access'.")
    api = HfApi()
    print(f"logged in as: {api.whoami()['name']}")

    print("datasets:")
    missing = check_access(api, DATASETS)
    if a.with_models or a.check:
        print("models:")
        missing += check_access(api, MODELS)
    if missing:
        sys.exit(f"\nMissing access to {len(missing)} resource(s) listed above. Fix them and re-run this script.")
    if a.check:
        return print("\naccess OK")

    import data
    have = all((C.DATA_DIR / f"{n}.jsonl").exists() for n in EXPECTED_SPLITS)
    if have and not a.force:
        print("\ndata/ already built (use --force to rebuild)")
    else:
        print("\nbuilding the main splits (downloads the datasets on first use) ...")
        data.build_splits()
    print("building the extra (v2) splits ...")
    data.build_extra_splits()

    splits = data.load_splits()
    print("\ndata/ ready:")
    for name, df in splits.items():
        print(f"  {name:28s} {len(df):6d} rows")

    if a.with_models:
        print("\ndownloading models ...")
        for repo, _, _ in MODELS:
            print(f"  {repo}")
            snapshot_download(repo)
    print("\ndone. Next: python run_all.py --models everything --seeds 0 1 2")


if __name__ == "__main__":
    main()
