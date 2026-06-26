import torch
import math
from tqdm import tqdm
import wandb
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
                device=None
            ):

        self.tokenizer, self.model = get_model(pretrained_model_name)
        self.dataloaders = dataloaders
        self.optimizer = optimizer if optimizer else torch.optim.AdamW(self.model.parameters(), lr=lr)
        self.checkpoint_dir = checkpoint_dir
        self.wandb_project = wandb_project
        self.device = get_device() if not device else device
        self.model.to(self.device)
        wandb.init(project=wandb_project, config={"lr": lr, "model": pretrained_model_name})

    
    def train(self, epochs, peek, peek_every_n_steps, peek_prompts):
        for epoch in range(epochs):
            self.train_epoch(epoch, peek, peek_every_n_steps, peek_prompts)
            self.validation(epoch)

    def calc_metrics(self, loss):
        return {
            'loss': loss,
            'ppl': math.exp(loss)
        }

    def train_epoch(self, epoch, peek, peek_every_n_steps, peek_prompts):
        epoch_loss = 0.0
        self.model.train()
        self.pbar = tqdm(self.dataloaders['train'], desc=f"Epoch {epoch} [train]")
        for i, batch in enumerate(self.pbar):
            step_loss = self.train_step(batch)
            epoch_loss += step_loss
            self.log_metrics(type='train', step=i, epoch=epoch, metrics=self.calc_metrics(step_loss))
            if peek and i % peek_every_n_steps == 0:
                self.peek_generate(i, epoch, peek_prompts)
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

    def train_step(self, batch):
        self.optimizer.zero_grad()      # empty gradients
        loss = self.calculate_loss(batch)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def eval_step(self, batch):
        loss = self.calculate_loss(batch)
        return loss.item()

    def calculate_loss(self, batch):
        '''
        If the sequence length is smaller than the context length, we put padding token (-100)
        -100 token is automatically ignored by cross entropy loss
        The last target token for fine tuning is set as the EOS token to make the model learn to end answers
        Also, all the prompt tokens are also padded since we want the model to learn to answer the prompts and not repeat the given prompt
        '''
        x,y = batch             # (batch, context_size), (batch, context_size)
        x,y = x.to(self.device), y.to(self.device)    # send to device
        logits = self.model(x).logits  # (batch, context_size, vocab). Also, HF model returns a wrapper hence added .logits
        loss = torch.nn.functional.cross_entropy(logits.flatten(0,1), y.flatten())  # automatically finds the average loss over valid tokens
        return loss

    def checkpoint(self, epoch, loss):
        raise NotImplementedError

    def load_checkpoint(self, location):
        raise NotImplementedError

    def peek_generate(self, step, epoch, peek_prompts):
        self.model.eval()
        sep = "─" * 60
        tqdm.write(f"\n{sep}\nPeek | epoch {epoch}, step {step}\n{sep}")
        for i, prompt in enumerate(peek_prompts, 1):
            output = generate(prompt, self.tokenizer, self.model)
            tqdm.write(f"Peek {i}: {output}")
        tqdm.write(f"{sep}\n")
        self.model.train()

    def log_metrics(self, type, step, epoch, metrics):
        if type == 'train' and hasattr(self, 'pbar'):
            self.pbar.set_postfix(loss=f"{metrics['loss']:.4f}", ppl=f"{metrics['ppl']:.2f}")
        else:
            tqdm.write(f"[{type}] epoch={epoch} step={step} loss={metrics['loss']:.4f} ppl={metrics['ppl']:.2f}")
        wandb.log({f"{type}/loss": metrics['loss'], f"{type}/ppl": metrics['ppl'], "epoch": epoch, "step": step})