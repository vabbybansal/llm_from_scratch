import os
import torch
import math
from tqdm import tqdm
import wandb
from transformers import get_cosine_schedule_with_warmup
from llm_from_scratch.libs.utils import get_device
from llm_from_scratch.supervised_fine_tuning.load_pretrained_hf_model import get_model, generate

class SupervisedFineTuner:
    def __init__(self, 
                pretrained_model_name,
                dataloaders:dict,
                optimizer=None,
                lr=2e-5,
                checkpoint_dir="checkpoints/sft",
                wandb_project="llm-from-scratch-pretrainer",
                device=None,
                # Must match the dataloaders' padding mode; also gates the per-step MPS cache refresh.
                # We default runs to STATIC padding (use_dynamic_padding=False in finetune.py) rather than
                # dynamic: dynamic padding makes every batch a different shape, which fragments the MPS
                # caching allocator over time until it OOMs (the allocator can't reuse mismatched-size
                # freed blocks). Static padding (every batch == max_length-1) keeps one constant shape, so
                # blocks are allocated once and recycled forever — stable memory, no fragmentation. The
                # cost (compute on pad tokens) is small because max_length is already short for our runs.
                use_dynamic_padding=True
            ):

        self.tokenizer, self.model = get_model(pretrained_model_name)
        self.dataloaders = dataloaders
        self.optimizer = optimizer if optimizer else torch.optim.AdamW(self.model.parameters(), lr=lr)
        self.checkpoint_dir = checkpoint_dir
        self.wandb_project = wandb_project
        self.use_dynamic_padding = use_dynamic_padding
        self.device = get_device() if not device else device
        self.model.to(self.device)
        self.scheduler = None                       # created in train() once total step count is known
        os.makedirs(checkpoint_dir, exist_ok=True)  # make sure checkpoint dir exists before saving
        wandb.init(project=wandb_project, config={"lr": lr, "model": pretrained_model_name})

    
    def train(self, epochs, peek, peek_every_n_steps, peek_prompts, checkpoint_every_n_steps=20000):
        # cosine LR schedule with ~3% warmup over the whole run (total steps = batches/epoch × epochs)
        num_steps = len(self.dataloaders['train']) * epochs
        self.scheduler = get_cosine_schedule_with_warmup(self.optimizer, int(0.03 * num_steps), num_steps)
        for epoch in range(epochs):
            self.train_epoch(epoch, peek, peek_every_n_steps, peek_prompts, checkpoint_every_n_steps)
            val_loss = self.validation(epoch)
            self.checkpoint(epoch, val_loss)        # per-epoch named checkpoint (kept), so a finished epoch is never lost

    def calc_metrics(self, loss):
        return {
            'loss': loss,
            'ppl': math.exp(loss)
        }

    def train_epoch(self, epoch, peek, peek_every_n_steps, peek_prompts, checkpoint_every_n_steps):
        epoch_loss = 0.0
        self.model.train()
        self.pbar = tqdm(self.dataloaders['train'], desc=f"Epoch {epoch} [train]")
        for i, batch in enumerate(self.pbar):
            step_loss = self.train_step(batch)
            epoch_loss += step_loss
            self.log_metrics(type='train', step=i, epoch=epoch, metrics=self.calc_metrics(step_loss))
            if peek and i % peek_every_n_steps == 0:
                self.peek_generate(i, epoch, peek_prompts)
            if i > 0 and i % checkpoint_every_n_steps == 0:
                self.checkpoint(epoch, step_loss, tag="latest")  # rolling mid-epoch save (overwrites) so a crash loses ≤N steps
            if self.use_dynamic_padding and self.device.type == "mps" and i % 100 == 0:
                torch.mps.empty_cache()  # only needed in dynamic mode: variable seq lengths fragment the MPS allocator
        epoch_loss = epoch_loss / (i+1)
        self.log_metrics(type='train_epoch', step=i, epoch=epoch, metrics=self.calc_metrics(epoch_loss))

    def validation(self, epoch):

        self.model.eval()               # model mode: eval
        with torch.no_grad():           # Don't create the computation graph
            val_loss = 0.0
            for i, batch in enumerate(self.dataloaders['validation']):
                val_loss += self.eval_step(batch)
            val_loss = val_loss / (i+1)
            metrics = self.calc_metrics(val_loss)
            self.log_metrics(type='val', step=-1, epoch=epoch, metrics=metrics)
            return val_loss                         # returned so train() can checkpoint with the val loss

    def train_step(self, batch):
        self.optimizer.zero_grad()      # empty gradients
        loss = self.calculate_loss(batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)  # clip grads to tame spikes (esp. in bf16)
        self.optimizer.step()
        if self.scheduler:
            self.scheduler.step()       # advance the LR schedule once per optimizer step
        return loss.item()

    def eval_step(self, batch):
        loss = self.calculate_loss(batch)
        return loss.item()

    def calculate_loss(self, batch):
        '''
        Loss is computed only on response tokens: the target y has -100 at prompt+padding positions,
        which cross_entropy ignores by default. The masking and input/target shift are done in the
        dataset (sft_dataset.py) — here we just forward and score.
        '''
        x,y = batch             # x=(batch, seq), y=(batch, seq) with -100 at masked positions
        x,y = x.to(self.device), y.to(self.device)    # move batch CPU -> GPU
        logits = self.model(x).logits  # HF returns a wrapper object; .logits is the (batch, seq, vocab) tensor
        # keep loss in bf16: the fp32 copy of a (batch, seq, 128k-vocab) logits tensor is ~1GB and OOMs on MPS
        loss = torch.nn.functional.cross_entropy(logits.flatten(0,1), y.flatten())
        return loss

    def checkpoint(self, epoch, loss, tag=None):
        # tag="latest" overwrites a single rolling file; otherwise a kept per-epoch file
        name = tag if tag else f"epoch{epoch}"
        torch.save({
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "loss": loss,
        }, f"{self.checkpoint_dir}/checkpoint_{name}.pt")

    def load_checkpoint(self, path):
        # load into the EXISTING model so the optimizer's param references stay valid (clean resume)
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        return ckpt["epoch"], ckpt["loss"]

    def peek_generate(self, step, epoch, peek_prompts):
        self.model.eval()
        sep = "─" * 60
        tqdm.write(f"\n{sep}\nPeek | epoch {epoch}, step {step}\n{sep}")
        for i, prompt in enumerate(peek_prompts, 1):
            output = generate(prompt, self.tokenizer, self.model)
            tqdm.write(f"Peek {i}: [{prompt}] -> {output}")
        tqdm.write(f"{sep}\n")
        self.model.train()

    def log_metrics(self, type, step, epoch, metrics):
        if type == 'train' and hasattr(self, 'pbar'):
            self.pbar.set_postfix(loss=f"{metrics['loss']:.4f}", ppl=f"{metrics['ppl']:.2f}")
        else:
            tqdm.write(f"[{type}] epoch={epoch} step={step} loss={metrics['loss']:.4f} ppl={metrics['ppl']:.2f}")
        wandb.log({f"{type}/loss": metrics['loss'], f"{type}/ppl": metrics['ppl'], "epoch": epoch, "step": step})