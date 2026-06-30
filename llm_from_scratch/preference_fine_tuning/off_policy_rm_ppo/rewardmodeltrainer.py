import os
import torch
import wandb
from datetime import date
from tqdm import tqdm
from llm_from_scratch.libs.utils import create_rm_classifier_from_lm_hf
from llm_from_scratch.libs.utils import get_device
from transformers import get_cosine_schedule_with_warmup

class RewardModelTrainer():
    def __init__(self, 
            sft_model_checkpoint_path,
            base_model_name_hf,
            sft_model_dtype,
            sft_model_tokenizer,
            dataloaders, 
            optimizer, 
            lr, 
            l2_reg,
            checkpoint_dir='checkpoints/rm/',
            wandb_project="llm-from-scratch-rm-trainer",
            device=None
        ):

        # Create the Reward Model classifier from the checkpointed SFT language model by chopping off the LM head and replacing it with a linear layer
        self.model = create_rm_classifier_from_lm_hf(
            base_model_name_hf, 
            sft_model_checkpoint_path, 
            sft_model_dtype, 
            sft_model_tokenizer)
        self.tokenizer = sft_model_tokenizer
        self.dataloaders = dataloaders
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr) if not optimizer else optimizer
        self.checkpoint_dir = checkpoint_dir
        self.wandb_project = wandb_project
        
        self.device = device if device else get_device()
        self.model.to(self.device)

        self.l2_reg = l2_reg

        os.makedirs(checkpoint_dir, exist_ok=True)   # ensure checkpoint dir exists before saving
        wandb.init(project=wandb_project, config={"lr": lr, "model": base_model_name_hf})

    def train(self, epochs, checkpoint_every_n_steps=20000, peek=False, peek_pairs=None):
        # cosine LR schedule with ~3% warmup over the whole run (total steps = batches/epoch × epochs)
        num_steps = len(self.dataloaders['train']) * epochs
        self.scheduler = get_cosine_schedule_with_warmup(self.optimizer, int(0.03 * num_steps), num_steps)
        # baseline at step 0: validate + peek the UNTRAINED RM (random score head) BEFORE any update,
        # so we capture the "before" (acc ≈ 0.5, margins ≈ 0, rewards ≈ 0) for the before/after story
        self.validation(epoch=0, step=0)
        if peek and peek_pairs:
            self.peek_score(0, 0, peek_pairs)
        for epoch in range(epochs):
            self.train_epoch(epoch, checkpoint_every_n_steps, peek, peek_pairs)
            val_loss = self.validation(epoch)
            self.checkpoint(epoch, val_loss)        # per-epoch named checkpoint (kept), so a finished epoch is never lost

    def train_epoch(self, epoch, checkpoint_every_n_steps, peek=False, peek_pairs=None):
        self.model.train()
        totals, n = {}, 0
        eval_every = max(1, len(self.dataloaders['train']) // 10)  # val + peek share one ~10% cadence (~10x/epoch)
        self.pbar = tqdm(self.dataloaders['train'], desc=f"Epoch {epoch} [train]")
        for i, batch in enumerate(self.pbar):
            metrics = self.train_step(batch)                       # dict: loss / bt / reg / acc
            for k, v in metrics.items():
                totals[k] = totals.get(k, 0.0) + v
            n += 1
            self.log_metrics(type='train', step=i, epoch=epoch, metrics=metrics)
            if i > 0 and i % eval_every == 0:                      # validate + peek together, every ~10%
                self.validation(epoch, step=i)
                if peek and peek_pairs:
                    self.peek_score(i, epoch, peek_pairs)
                self.model.train()                                 # validation()/peek_score leave eval — restore once
            if i > 0 and i % checkpoint_every_n_steps == 0:
                self.checkpoint(epoch, metrics['loss'], tag="latest")  # rolling mid-epoch save (overwrites) so a crash loses ≤N steps
        self.log_metrics(type='train_epoch', step=i, epoch=epoch,
                         metrics={k: v / n for k, v in totals.items()})

    def _score_one(self, messages):
        # tokenize one conversation -> a single scalar reward. b=1 with no padding, so the last token
        # is the literal last position (no attention-mask/pooling subtlety to worry about here).
        ids = self.tokenizer.apply_chat_template(messages, tokenize=True, return_dict=False, add_generation_prompt=False)
        ids = torch.tensor([ids], device=self.device)            # (1, L)
        mask = torch.ones_like(ids)
        return self.model(ids, attention_mask=mask).logits.squeeze().item()

    def peek_score(self, step, epoch, peek_pairs):
        # RM analogue of SFT's peek-generation: score fixed (prompt, chosen, rejected) pairs and print
        # r_chosen vs r_rejected. The qualitative signal is the MARGIN widening (chosen pulled above
        # rejected) over training — even while the BT loss barely moves, like SFT's loss-vs-peeks story.
        self.model.eval()
        sep = "─" * 60
        tqdm.write(f"\n{sep}\nPeek | epoch {epoch}, step {step}\n{sep}")
        with torch.no_grad():
            for i, pair in enumerate(peek_pairs, 1):
                chosen   = [{"role": "user", "content": pair["prompt"]}, {"role": "assistant", "content": pair["chosen"]}]
                rejected = [{"role": "user", "content": pair["prompt"]}, {"role": "assistant", "content": pair["rejected"]}]
                r_c, r_r = self._score_one(chosen), self._score_one(rejected)
                verdict = "✓" if r_c > r_r else "✗"   # did the RM rank the intended-better answer higher?
                tqdm.write(f"Peek {i}: [{pair['prompt']}]  margin={r_c - r_r:+.3f} {verdict}")
                tqdm.write(f"   chosen   r={r_c:+.3f} | {pair['chosen'][:70]}")
                tqdm.write(f"   rejected r={r_r:+.3f} | {pair['rejected'][:70]}")
        tqdm.write(f"{sep}\n")
        self.model.train()

    def validation(self, epoch, step=-1):
        self.model.eval()               # model mode: eval
        with torch.no_grad():           # Don't create the computation graph
            totals, n = {}, 0
            for batch in self.dataloaders['validation']:
                metrics = self.eval_step(batch)     # dict: loss / bt / reg / acc
                for k, v in metrics.items():
                    totals[k] = totals.get(k, 0.0) + v
                n += 1
            avg = {k: v / n for k, v in totals.items()}
            self.log_metrics(type='val', step=step, epoch=epoch, metrics=avg)
            return avg['loss']                      # returned so train() can checkpoint with the val loss

    def train_step(self, batch):
        self.optimizer.zero_grad()      # empty gradients
        loss, metrics = self.calculate_loss(batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)  # clip grads to tame spikes (esp. in bf16)
        self.optimizer.step()
        if self.scheduler:
            self.scheduler.step()       # advance the LR schedule once per optimizer step
        return metrics

    def eval_step(self, batch):
        _, metrics = self.calculate_loss(batch)
        return metrics

    def calculate_loss(self, batch):
        '''
        Reward-model loss = Bradley-Terry on the pair. Each side is scored to ONE scalar reward
        (last-token pooling inside the model); the loss is -logsigmoid(r_chosen - r_rejected), i.e.
        BCE-with-logits on the reward MARGIN with an implicit target of 1 (chosen should outscore
        rejected). It's shift-invariant: only the difference between the two rewards matters.
        '''
        # the collate returns a dict of four tensors (chosen/rejected x input_ids/attention_mask)
        chosen_ids    = batch["chosen_input_ids"].to(self.device)
        chosen_mask   = batch["chosen_attention_mask"].to(self.device)
        rejected_ids  = batch["rejected_input_ids"].to(self.device)
        rejected_mask = batch["rejected_attention_mask"].to(self.device)
        # HF SequenceClassification returns .logits of shape (B, 1); squeeze to (B,) = one scalar reward per sequence
        pos_logit = self.model(chosen_ids,   attention_mask=chosen_mask).logits.squeeze(-1)    # (B,)
        neg_logit = self.model(rejected_ids, attention_mask=rejected_mask).logits.squeeze(-1)  # (B,)
        margin = pos_logit - neg_logit  # (B,)

        bt = torch.nn.functional.binary_cross_entropy_with_logits(margin, torch.ones_like(margin))  # Bradley-Terry term
        # 2-class CE equivalent: F.cross_entropy(torch.stack([pos_logit, neg_logit], 1), zeros_long)

        # reward-L2 penalty. BT depends only on the margin (r_c − r_r), so it leaves the rewards'
        # absolute offset AND scale unconstrained — they drift and grow freely during training. Harmless
        # for the RM itself, but PPO consumes the reward as `r_RM − β·KL(policy‖ref)`: if the reward scale
        # balloons it swamps the KL penalty, so the policy ignores the reference and reward-hacks. This L2
        # bounds the reward scale near 0 (→ PPO-stable, transferable β) WITHOUT changing the ranking,
        # because it pulls r_c and r_r toward 0 symmetrically. (Offset is separately absorbed by PPO's
        # value baseline; this term only controls scale.)
        reg = (pos_logit.pow(2) + neg_logit.pow(2)).mean()
        loss = bt + self.l2_reg * reg

        acc = (pos_logit > neg_logit).float().mean()          # preference accuracy (metric, not loss)
        # detached floats for logging; 'reg' is the scaled contribution (λ·reg) so bt + reg == loss
        metrics = {
            'loss': loss.item(), 'bt': bt.item(), 'reg': (self.l2_reg * reg).item(), 'acc': acc.item(),
            'r_chosen': pos_logit.mean().item(),      # reward-scale diagnostics: with reg these converge
            'r_rejected': neg_logit.mean().item(),    # toward 0 (|r| shrinks); without reg they drift/grow
        }
        return loss, metrics

    def checkpoint(self, epoch, loss, tag=None):
        # tag="latest" overwrites a single rolling file; otherwise a kept per-epoch file.
        # date-stamp the filename so today's run is distinguishable from earlier days' — and "latest"
        # rolls *within* the day rather than clobbering yesterday's checkpoint.
        name = tag if tag else f"epoch{epoch}"
        today = date.today().isoformat()        # e.g. 2026-06-30
        torch.save({
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "loss": loss,
        }, f"{self.checkpoint_dir}/checkpoint_{today}_{name}.pt")

    def log_metrics(self, type, step, epoch, metrics):
        # metrics is a dict (e.g. {"loss": ..., "acc": ...}); log every key generically so adding
        # a metric later (like preference accuracy) needs no change here
        if type == 'train' and hasattr(self, 'pbar'):
            self.pbar.set_postfix(**{k: f"{v:.4f}" for k, v in metrics.items()})
        else:
            tqdm.write(f"[{type}] epoch={epoch} step={step} " +
                       " ".join(f"{k}={v:.4f}" for k, v in metrics.items()))
        wandb.log({**{f"{type}/{k}": v for k, v in metrics.items()}, "epoch": epoch, "step": step})

    def load_checkpoint(self, path):
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        return ckpt["epoch"], ckpt["loss"]