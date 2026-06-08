import os
import csv
import math
import torch
import wandb
from tqdm import tqdm

from llm_from_scratch.libs.tokenizer import Tokenizer
from llm_from_scratch.libs.utils import get_device, generate_text

class PreTrainLanguageModelDriver():
    def __init__(self, model, dataloaders: dict, optimizer=None, epochs=10, lr=1e-4,
                 checkpoint_dir="checkpoints", peek=True, peek_every_n_steps=500,
                 peek_prompts=("The earth is not flat because", "King minus female equals"),
                 tokenizer_model="gpt2", wandb_project="llm-from-scratch"):
        self.epochs = epochs
        self.device = get_device()
        self.model = model.to(self.device)
        self.dataloaders = dataloaders
        self.optimizer = optimizer or torch.optim.AdamW(self.model.parameters(), lr=lr)
        self.checkpoint_dir = checkpoint_dir
        self.peek = peek
        self.peek_every_n_steps = peek_every_n_steps
        self.peek_prompts = peek_prompts
        self.tokenizer = Tokenizer(model_name=tokenizer_model)
        wandb.init(project=wandb_project)
        self.log_path = os.path.join(checkpoint_dir, "metrics.csv")
        os.makedirs(checkpoint_dir, exist_ok=True)
        with open(self.log_path, "w", newline="") as f:
            csv.writer(f).writerow(["epoch", "step", "train_loss", "val_loss", "val_perplexity"])

    def calculate_loss_lm(self, batch) -> torch.tensor:
        '''
        input : a,b,c,(d)
        target: b,c,d
        LM Loss -> categorical cross entropy -> -∑plogq, where p is the target dist and q is the probability dist.
        Since p is one hot encoded, this boils down to calculating -∑logq just for the target tokens.
        Essentially, how different is the distribution of the target labels and the output prob dist.
        '''
        x,y = batch
        x,y = x.to(self.device), y.to(self.device)
        logits = self.model(x)
        # logits: (batch, seq, vocab) -> (batch*seq, vocab); y: (batch, seq) -> (batch*seq,)
        # cross_entropy looks up y[i] as the correct class index and computes -log(softmax(logits[i])[y[i]])
        loss = torch.nn.functional.cross_entropy(logits.flatten(0,1), y.flatten())
        return loss

    def peek_generate(self, step, epoch):
        if not self.peek:
            return
        self.model.eval()
        sep = "─" * 60
        print(f"\n{sep}")
        print(f"Peek | epoch {epoch}, step {step}")
        print(sep)
        for i, prompt in enumerate(self.peek_prompts, 1):
            input_ids = torch.tensor(self.tokenizer.encode(prompt)).unsqueeze(0).to(self.device)
            output_ids = generate_text(self.model, input_ids, max_new_tokens=50,
                                       context_size=self.model.pos_emb.num_embeddings)
            print(f"\nPeek {i}: {self.tokenizer.decode(output_ids[0].tolist())}")
        print(f"\n{sep}\n")
        self.model.train()

    def train(self):
        global_step = 0

        for epoch in range(self.epochs):
            self.model.train()
            pbar = tqdm(self.dataloaders['train'], desc=f"Epoch {epoch}")
            for i, batch in enumerate(pbar):
                self.optimizer.zero_grad()
                loss = self.calculate_loss_lm(batch)
                loss.backward()
                self.optimizer.step()
                pbar.set_postfix(loss=f"{loss.item():.4f}")
                wandb.log({"train/loss": loss.item(), "epoch": epoch}, step=global_step)
                with open(self.log_path, "a", newline="") as f:
                    csv.writer(f).writerow([epoch, global_step, f"{loss.item():.4f}", "", ""])
                global_step += 1

                if i % self.peek_every_n_steps == 0:
                    self.peek_generate(i, epoch)
            self.eval(epoch, global_step)

    def eval(self, epoch, global_step):
        self.model.eval()

        total_loss = torch.tensor(0.0).to(self.device)

        with torch.no_grad():
            for batch in self.dataloaders['validation']:
                total_loss += self.calculate_loss_lm(batch)
            loss = total_loss / len(self.dataloaders['validation'])

        perplexity = math.exp(loss.item())
        print(f"\nEval Loss | epoch {epoch}, loss {loss.item():.4f}, perplexity {perplexity:.2f}")
        wandb.log({"val/loss": loss.item(), "val/perplexity": perplexity, "epoch": epoch}, step=global_step)
        with open(self.log_path, "a", newline="") as f:
            csv.writer(f).writerow([epoch, global_step, "", f"{loss.item():.4f}", f"{perplexity:.2f}"])
        self.save_checkpoint(epoch, loss.item())

    def save_checkpoint(self, epoch, loss):
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        torch.save({
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "loss": loss,
        }, f"{self.checkpoint_dir}/checkpoint_epoch{epoch}.pt")

    def load_checkpoint(self, path):
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        return ckpt["epoch"], ckpt["loss"]
