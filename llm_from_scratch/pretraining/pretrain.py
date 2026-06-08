from llm_from_scratch.libs.gpt import GPT
from llm_from_scratch.libs.data_loader import create_dataloaders
from llm_from_scratch.pretraining.trainer import PreTrainLanguageModelDriver

model = GPT(
    vocab_size=50257,
    n_layers=6,
    d_in=384,
    context_length=256,
    dropout=0.1,
    n_heads=6,
)

dataloaders = create_dataloaders(batch_size=6, max_length=256, stride=128)

trainer = PreTrainLanguageModelDriver(
    model,
    dataloaders,
    epochs=1,
    lr=3e-4,
    peek=True,
    peek_every_n_steps=2000,
)

trainer.train()
