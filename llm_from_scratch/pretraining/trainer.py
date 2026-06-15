import os
import math
import torch
from tqdm import tqdm
from llm_from_scratch.libs.tokenizer import Tokenizer
from llm_from_scratch.libs.utils import get_device, generate
from llm_from_scratch.pretraining.logger import TrainingLogger


class PreTrainLanguageModelDriver():
    def __init__(self, model, dataloaders: dict, optimizer=None, epochs=10, lr=1e-4,
                 checkpoint_dir="checkpoints", peek=True, peek_every_n_steps=500,
                 peek_prompts=("The earth is not flat because", "King minus male plus female equals"),
                 tokenizer_model="gpt2", wandb_project="llm-from-scratch",
                 val_every_n_steps=2000, val_batches=500):
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
        self.val_every_n_steps = val_every_n_steps
        self.val_batches = val_batches
        os.makedirs(checkpoint_dir, exist_ok=True)
        self.logger = TrainingLogger(checkpoint_dir, wandb_project)
        self.peek_log_path = os.path.join(checkpoint_dir, "peek.txt")
        open(self.peek_log_path, "w").close()  # clear on each run

    def calculate_loss_lm(self, batch) -> torch.tensor:
        '''
        input : a,b,c,(d)
        target: b,c,d
        LM Loss -> categorical cross entropy -> -∑plogq, where p is the target dist and q is the probability dist.
        Since p is one hot encoded, this boils down to calculating -∑logq just for the target tokens.
        Essentially, how different is the distribution of the target labels and the output prob dist.
        '''
        x, y = batch
        x, y = x.to(self.device), y.to(self.device)
        logits = self.model(x)
        # logits: (batch, seq, vocab) -> (batch*seq, vocab); y: (batch, seq) -> (batch*seq,)
        # cross_entropy looks up y[i] as the correct class index and computes -log(softmax(logits[i])[y[i]])
        loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), y.flatten())
        return loss

    def peek_generate(self, step, epoch):
        if not self.peek:
            return
        self.model.eval()
        sep = "─" * 60
        tqdm.write(f"\n{sep}")
        tqdm.write(f"Peek | epoch {epoch}, step {step}")
        tqdm.write(sep)
        lines = []
        for i, prompt in enumerate(self.peek_prompts, 1):
            input_ids = torch.tensor(self.tokenizer.encode(prompt)).unsqueeze(0).to(self.device)
            output_ids = generate(self.model, input_ids, max_new_tokens=50,
                                  context_size=self.model.pos_emb.num_embeddings, temperature=0.8, top_k=50)
            line = f"Peek {i}: {self.tokenizer.decode(output_ids[0].tolist())}"
            tqdm.write(f"\n{line}")
            lines.append(line)
        tqdm.write(f"\n{sep}\n")
        with open(self.peek_log_path, "a") as f:
            f.write(f"\n{sep}\nPeek | epoch {epoch}, step {step}\n{sep}\n")
            f.write("\n".join(lines) + "\n")
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
                train_ppl = self.logger.log_train(epoch, global_step, loss.item())
                pbar.set_postfix(loss=f"{loss.item():.4f}", ppl=f"{train_ppl:.1f}")
                global_step += 1

                if global_step % self.peek_every_n_steps == 0:
                    self.peek_generate(global_step, epoch)
                if global_step % self.val_every_n_steps == 0:
                    self.eval(epoch, global_step, end_of_epoch=False)
                    self.model.train()

            self.eval(epoch, global_step, end_of_epoch=True)

    def eval(self, epoch, global_step, end_of_epoch=True):
        self.model.eval()

        total_loss = torch.tensor(0.0).to(self.device)
        val_loader = self.dataloaders['validation']

        with torch.no_grad():
            for n, batch in enumerate(val_loader):
                if not end_of_epoch and n >= self.val_batches:
                    break
                total_loss += self.calculate_loss_lm(batch)
        batches_used = len(val_loader) if end_of_epoch else min(len(val_loader), self.val_batches)
        loss = total_loss / batches_used

        self.logger.log_val(epoch, global_step, loss.item(), end_of_epoch, self.val_batches)

        if end_of_epoch:
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
