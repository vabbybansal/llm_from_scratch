import os
import csv
import math
import wandb
from datetime import datetime
from tqdm import tqdm


def _ts():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


class TrainingLogger:
    def __init__(self, checkpoint_dir: str, wandb_project: str):
        self.log_path = os.path.join(checkpoint_dir, "metrics.csv")
        with open(self.log_path, "w", newline="") as f:
            csv.writer(f).writerow(["timestamp", "epoch", "step", "train_loss", "train_perplexity", "val_loss", "val_perplexity"])
        wandb.init(project=wandb_project)

    def log_train(self, epoch: int, step: int, loss: float):
        ppl = math.exp(loss)
        wandb.log({"train/loss": loss, "train/perplexity": ppl, "epoch": epoch}, step=step)
        with open(self.log_path, "a", newline="") as f:
            csv.writer(f).writerow([_ts(), epoch, step, f"{loss:.4f}", f"{ppl:.2f}", "", ""])
        return ppl

    def log_val(self, epoch: int, step: int, loss: float, end_of_epoch: bool, val_batches: int):
        ppl = math.exp(loss)
        label = "Eval (full)" if end_of_epoch else f"Eval (subset {val_batches})"
        tqdm.write(f"\n{label} | epoch {epoch}, step {step}, loss {loss:.4f}, perplexity {ppl:.2f}")
        wandb.log({"val/loss": loss, "val/perplexity": ppl, "epoch": epoch}, step=step)
        with open(self.log_path, "a", newline="") as f:
            csv.writer(f).writerow([_ts(), epoch, step, "", "", f"{loss:.4f}", f"{ppl:.2f}"])
        return ppl
