import os
import sys
import subprocess
import itertools
import pandas as pd
import json

def run_grid_search():
    batch_sizes = [8] # test: [4, 8, 16]
    optimizers = ['adam'] # test: ['adam', 'sgd']
    learning_rates = [1e-5]# test:  [1e-3, 1e-4, 1e-5]
    max_epochs = 100

    results = []
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    train_script = os.path.join(script_dir, "train.py")
    runs_dir = os.path.join(script_dir, "runs")
    
    # We will sort runs by creation time to find the newest run after each training
    # Or better, we can modify train.py to accept a specific run_id or just find the latest folder in runs/
    
    combinations = list(itertools.product(batch_sizes, optimizers, learning_rates))
    total = len(combinations)
    
    print(f"Starting grid search over {total} combinations...")
    
    for i, (bs, opt, lr) in enumerate(combinations):
        print(f"\n[{i+1}/{total}] Training with Batch Size: {bs}, Optimizer: {opt}, LR: {lr}")
        
        # Count existing runs to find the new one easily
        if os.path.exists(runs_dir):
            runs_before = set(os.listdir(runs_dir))
        else:
            runs_before = set()
            
        cmd = [
            sys.executable, train_script,
            "--epochs", str(max_epochs),
            "--batch_size", str(bs),
            "--optimizer", opt,
            "--lr", str(lr)
        ]
        
        # Run training
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Training failed for combination bs={bs}, opt={opt}, lr={lr}")
            continue
            
        # Find the new run directory
        runs_after = set(os.listdir(runs_dir))
        new_runs = runs_after - runs_before
        
        if new_runs:
            latest_run = list(new_runs)[0] # Assuming one new run
            summary_path = os.path.join(runs_dir, latest_run, "summary.json")
            
            if os.path.exists(summary_path):
                with open(summary_path, 'r') as f:
                    summary = json.load(f)
                
                res = {
                    "Run ID": latest_run,
                    "Batch Size": bs,
                    "Optimizer": opt,
                    "Learning Rate": lr,
                    "Epochs Trained": summary.get("epochs_trained", 0),
                    "Val Dice (ISBI)": summary.get("val_isbi2015", {}).get("best_dice", 0),
                    "Val IoU (ISBI)": summary.get("val_isbi2015", {}).get("best_iou", 0),
                    "Test Dice": summary.get("test_mslesseg", {}).get("dice_score", 0),
                    "Test IoU": summary.get("test_mslesseg", {}).get("iou_score", 0)
                }
                results.append(res)
            else:
                print(f"No summary found for {latest_run}")
                
    # Save results
    if results:
        df = pd.DataFrame(results)
        print("\n--- Grid Search Results ---")
        print(df.to_markdown(index=False))
        
        res_path = os.path.join(script_dir, "grid_search_results.csv")
        df.to_csv(res_path, index=False)
        print(f"\nResults saved to {res_path}")
    else:
        print("No results to save.")

if __name__ == "__main__":
    run_grid_search()
