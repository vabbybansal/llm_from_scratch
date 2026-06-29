import os
import torch
import wandb
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

        os.makedirs(checkpoint_dir, exist_ok=True)   # ensure checkpoint dir exists before saving
        wandb.init(project=wandb_project, config={"lr": lr, "model": base_model_name_hf})

    def train(self, epochs, checkpoint_every_n_steps=20000,
              peek=False, peek_every_n_steps=200, peek_pairs=None):
        # cosine LR schedule with ~3% warmup over the whole run (total steps = batches/epoch × epochs)
        num_steps = len(self.dataloaders['train']) * epochs
        self.scheduler = get_cosine_schedule_with_warmup(self.optimizer, int(0.03 * num_steps), num_steps)
        for epoch in range(epochs):
            self.train_epoch(epoch, checkpoint_every_n_steps, peek, peek_every_n_steps, peek_pairs)
            val_loss = self.validation(epoch)
            self.checkpoint(epoch, val_loss)        # per-epoch named checkpoint (kept), so a finished epoch is never lost

    def calc_metrics(self, loss, acc):
        return {
            'loss': loss,
            'acc': acc,        # preference accuracy: random = 0.5, a decent RM ≈ 0.65–0.75
        }

    def train_epoch(self, epoch, checkpoint_every_n_steps, peek=False, peek_every_n_steps=200, peek_pairs=None):
        epoch_loss, epoch_acc = 0.0, 0.0
        self.model.train()
        self.pbar = tqdm(self.dataloaders['train'], desc=f"Epoch {epoch} [train]")
        for i, batch in enumerate(self.pbar):
            step_loss, step_acc = self.train_step(batch)
            epoch_loss += step_loss
            epoch_acc += step_acc
            self.log_metrics(type='train', step=i, epoch=epoch, metrics=self.calc_metrics(step_loss, step_acc))
            if peek and peek_pairs and i % peek_every_n_steps == 0:
                self.peek_score(i, epoch, peek_pairs)
            if i > 0 and i % checkpoint_every_n_steps == 0:
                self.checkpoint(epoch, step_loss, tag="latest")  # rolling mid-epoch save (overwrites) so a crash loses ≤N steps
        self.log_metrics(type='train_epoch', step=i, epoch=epoch,
                         metrics=self.calc_metrics(epoch_loss / (i+1), epoch_acc / (i+1)))

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

    def validation(self, epoch):

        self.model.eval()               # model mode: eval
        with torch.no_grad():           # Don't create the computation graph
            val_loss, val_acc = 0.0, 0.0
            for i, batch in enumerate(self.dataloaders['validation']):
                step_loss, step_acc = self.eval_step(batch)
                val_loss += step_loss
                val_acc += step_acc
            val_loss = val_loss / (i+1)
            val_acc = val_acc / (i+1)
            self.log_metrics(type='val', step=-1, epoch=epoch, metrics=self.calc_metrics(val_loss, val_acc))
            return val_loss                         # returned so train() can checkpoint with the val loss

    def train_step(self, batch):
        self.optimizer.zero_grad()      # empty gradients
        loss, acc = self.calculate_loss(batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)  # clip grads to tame spikes (esp. in bf16)
        self.optimizer.step()
        if self.scheduler:
            self.scheduler.step()       # advance the LR schedule once per optimizer step
        return loss.item(), acc

    def eval_step(self, batch):
        loss, acc = self.calculate_loss(batch)
        return loss.item(), acc

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
        loss = torch.nn.functional.binary_cross_entropy_with_logits(margin, torch.ones_like(margin))
        # loss = torch.nn.functional.cross_entropy(torch.stack([pos_logit, neg_logit], dim=1), torch.zeros(pos_logit.size(0), dtype=torch.long, device=pos_logit.device))
        acc = (pos_logit > neg_logit).float().mean().item()   # preference accuracy (metric, not loss): fraction of pairs ranked correctly
        return loss, acc

    def checkpoint(self, epoch, loss, tag=None):
        # tag="latest" overwrites a single rolling file; otherwise a kept per-epoch file
        name = tag if tag else f"epoch{epoch}"
        torch.save({
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "loss": loss,
        }, f"{self.checkpoint_dir}/checkpoint_{name}.pt")

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