"""
cv.py — Quick 5-fold cross-validation to decide whether a change helps. Run this for EVERY
feature/parameter change instead of uploading to Kaggle (public leaderboard is too noisy).

Usage:
    python cv.py --note "added IS_WIDOWED"            # fast: LightGBM, lr=0.1 (~1 min)
    python cv.py --models xgb lgb --full --note "..."  # slower: final learning rates + blend tuning

Each run appends one line to experiments.csv. Only trust differences bigger than ~1 std/sqrt(5)
(about 0.007 Macro F1) — anything smaller is split noise.
"""
import argparse
import csv
import subprocess
import time
from datetime import datetime

from config import cfg
from data.loader import load_train_data
from evaluation.cv import cross_validate, tune_blend

EXPERIMENTS_LOG = cfg.BASE_DIR / "experiments.csv"


def git_commit() -> str:
    """Short commit hash, with '+dirty' if there are uncommitted changes."""
    try:
        sha = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=cfg.BASE_DIR, text=True).strip()
        dirty = subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"],
                                        cwd=cfg.BASE_DIR, text=True).strip()
        return sha + ("+dirty" if dirty else "")
    except Exception:
        return "unknown"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", nargs="+", default=["lgb"], choices=["xgb", "lgb"])
    parser.add_argument("--full", action="store_true", help="use config learning rates instead of fast lr=0.1")
    parser.add_argument("--note", default="", help="what changed in this experiment")
    args = parser.parse_args()

    overrides = {} if args.full else {"learning_rate": 0.1}
    model_params = {m: dict(overrides) for m in args.models}

    X, y, _ = load_train_data()
    print(f">>> {cfg.N_SPLITS}-fold CV | models={args.models} | {'full' if args.full else 'fast (lr=0.1)'}")
    t0 = time.time()
    res = cross_validate(X, y, model_params)

    row = {
        "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "commit": git_commit(),
        "mode": "full" if args.full else "fast",
        "note": args.note,
    }
    print("\n=== Results (Macro F1, mean ± std over folds) ===")
    for m in args.models:
        mean, std = res.summary(m)
        print(f"  {m}: {mean:.4f} ± {std:.4f}   best iterations: {res.best_iters[m]}")
        row[f"{m}_f1"] = f"{mean:.4f}"
        row[f"{m}_std"] = f"{std:.4f}"

    weights, prior_power, blend_f1 = tune_blend(res.oof, y)
    print(f"  tuned OOF blend: {blend_f1:.4f}  weights={weights}  prior_power={prior_power}")
    row.update(blend_f1=f"{blend_f1:.4f}", weights=weights, prior_power=prior_power,
               minutes=f"{(time.time() - t0) / 60:.1f}")

    # Fixed column order so the CSV stays readable when different models are used
    fields = ["date", "commit", "mode", "note", "lgb_f1", "lgb_std", "xgb_f1", "xgb_std",
              "blend_f1", "weights", "prior_power", "minutes"]
    is_new = not EXPERIMENTS_LOG.exists()
    with open(EXPERIMENTS_LOG, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, restval="")
        if is_new:
            writer.writeheader()
        writer.writerow(row)
    print(f"\n[Logged] {EXPERIMENTS_LOG.name}  ({row['minutes']} min)")


if __name__ == "__main__":
    main()
