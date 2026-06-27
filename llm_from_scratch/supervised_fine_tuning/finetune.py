from llm_from_scratch.supervised_fine_tuning.fine_tuner import SupervisedFineTuner
from llm_from_scratch.supervised_fine_tuning.load_pretrained_hf_model import get_tokenizer
from llm_from_scratch.supervised_fine_tuning.sft_dataset import create_sft_dataloaders

MODEL = "meta-llama/Llama-3.2-1B"

# Guard required because macOS spawns DataLoader workers by re-importing this file.
# Without it, every worker would re-run training and recursively spawn more workers.
if __name__ == "__main__":
    # tokenizer is needed up front to build the dataloaders (chat template + masking).
    # The trainer re-loads it (cheap, cached) alongside the model via get_model.
    tokenizer = get_tokenizer(MODEL)

    # static padding (use_dynamic_padding=False): every batch is the same (batch, max_length-1) shape,
    # so the MPS allocator never fragments — no OOM, no per-step cache refresh needed
    USE_DYNAMIC_PADDING = False

    # filter_small: keep only short (<=max_length) conversations from a bounded scan, capped to max_examples,
    # so the run fits in memory and finishes in hours instead of days
    dataloaders = create_sft_dataloaders(
        tokenizer, max_length=256, batch_size=8, num_workers=4,
        filter_small=True, max_examples=20000, filter_scan=50000,
        use_dynamic_padding=USE_DYNAMIC_PADDING,
    )

    trainer = SupervisedFineTuner(MODEL, dataloaders, lr=2e-5,
                                  use_dynamic_padding=USE_DYNAMIC_PADDING)

    trainer.train(
        epochs=1,
        checkpoint_every_n_steps=1000,   # ~2-3 rolling saves over this 2,475-step run so a crash doesn't lose it
        peek=True,
        peek_every_n_steps=200,
        peek_prompts=[
            "What is the capital of France?",
            "Write a haiku about the ocean.",
            "Explain the difference between machine learning and deep learning.",
        ],
    )
