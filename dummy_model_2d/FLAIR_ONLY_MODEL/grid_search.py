import os
import sys
import subprocess
import itertools
import pandas as pd
import json

def run_grid_search():
    batch_sizes    = [8] # testing: [4, 8, 16]
    optimizers     = ['adam'] # testing: ['adam', 'sgd']
    learning_rates = [1e-5] # testing: [1e-3, 1e-4, 1e-5]
    max_epochs     = 100

    results = []

    script_dir = os.path.dirname(os.path.abspath(__file__))
    train_script = os.path.join(script_dir, "train.py")
    runs_dir     = os.path.join(script_dir, "runs")

    combinations = list(itertools.product(batch_sizes, optimizers, learning_rates))
    total = len(combinations)

    print(f"Starting grid search over {total} combinations ...")
    print(f"  Dataset : dummy_model_2d/dataset/split/{{train|val|test}}")
    print(f"  Input   : 1-channel FLAIR (256x256x1)")
    print(f"  Epochs  : {max_epochs} (no early stopping)\n")

    for i, (bs, opt, lr) in enumerate(combinations):
        print(f"\n{'='*60}")
        print(f"[{i+1}/{total}]  Batch={bs}  Optimizer={opt}  LR={lr}")
        print(f"{'='*60}")

        # Snapshot existing runs so we can identify the new one
        runs_before = set(os.listdir(runs_dir)) if os.path.exists(runs_dir) else set()

        cmd = [
            sys.executable, train_script,
            "--epochs",     str(max_epochs),
            "--batch_size", str(bs),
            "--optimizer",  opt,
            "--lr",         str(lr),
        ]

        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError:
            print(f"[WARN] Training failed for bs={bs}, opt={opt}, lr={lr} — skipping")
            continue

        # Find the newly created run directory
        runs_after = set(os.listdir(runs_dir))
        new_runs   = runs_after - runs_before

        if not new_runs:
            print("[WARN] Could not find new run directory — skipping summary collection")
            continue

        latest_run   = sorted(new_runs)[0]          # timestamps are sortable
        summary_path = os.path.join(runs_dir, latest_run, "summary.json")

        if os.path.exists(summary_path):
            with open(summary_path, 'r') as f:
                summary = json.load(f)

            res = {
                "Run ID":        latest_run,
                "Batch Size":    bs,
                "Optimizer":     opt,
                "Learning Rate": lr,
                "Epochs Trained": summary.get("epochs_trained", 0),
                "Best Val Dice": summary.get("val_split", {}).get("best_dice", 0),
                "Best Val IoU":  summary.get("val_split", {}).get("best_iou",  0),
                "Test Dice":     summary.get("test_split", {}).get("dice_score", 0),
                "Test IoU":      summary.get("test_split", {}).get("iou_score",  0),
            }
            results.append(res)
            print(f"\n  Summary: Val Dice={res['Best Val Dice']:.4f}  "
                  f"Test Dice={res['Test Dice']:.4f}")
        else:
            print(f"[WARN] No summary.json found for run {latest_run}")

    # ── Print & save final results table ─────────────────────────────────────
    if results:
        df = pd.DataFrame(results)
        print("\n\n" + "="*80)
        print("GRID SEARCH COMPLETE — Results Summary")
        print("="*80)
        print(df.to_markdown(index=False))

        res_path = os.path.join(script_dir, "grid_search_results.csv")
        df.to_csv(res_path, index=False)
        print(f"\nFull results table saved to: {res_path}")
    else:
        print("No results collected.")

if __name__ == "__main__":
    run_grid_search()
