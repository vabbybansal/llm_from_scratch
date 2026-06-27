from llm_from_scratch.pretraining.model.gpt import GPT
from llm_from_scratch.pretraining.data.data_loader import create_dataloaders
from llm_from_scratch.pretraining.trainer import PreTrainLanguageModelDriver
from llm_from_scratch.pretraining.model.constants import GPT2_SMALL, GPT2_MEDIUM, GPT2_LARGE
from llm_from_scratch.pretraining.load_weights import load_gpt2_weights

# model = GPT(
#     vocab_size=50257,
#     n_layers=6,
#     d_in=384,
#     context_length=256,
#     dropout=0.1,
#     n_heads=6,
# )

# To load GPT-2 pretrained weights instead of training from scratch:
# model = GPT(**GPT2_SMALL); model = load_gpt2_weights(model)
# model = GPT(**GPT2_MEDIUM)
# model = load_gpt2_weights(model, "gpt2-medium")

model = GPT(**GPT2_LARGE)
model = load_gpt2_weights(model, "gpt2-large")


dataloaders = create_dataloaders(batch_size=4, max_length=256, stride=128)

trainer = PreTrainLanguageModelDriver(
    model,
    dataloaders,
    epochs=1,
    lr=3e-4,
    peek=True,
    peek_every_n_steps=2000,
)

# epoch, loss = trainer.load_checkpoint("checkpoints/checkpoint_epoch0.pt")
# print(f"Resumed from epoch {epoch}, loss {loss:.4f}")

trainer.train()
