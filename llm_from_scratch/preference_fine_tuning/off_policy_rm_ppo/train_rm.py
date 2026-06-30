import torch
from llm_from_scratch.supervised_fine_tuning.load_pretrained_hf_model import get_tokenizer
from llm_from_scratch.preference_fine_tuning.dataset.preference_dataset import create_preference_dataloaders
from llm_from_scratch.preference_fine_tuning.off_policy_rm_ppo.rewardmodeltrainer import RewardModelTrainer

MODEL = "meta-llama/Llama-3.2-1B"                          # base architecture (warmed from the SFT checkpoint below)
SFT_CHECKPOINT = "checkpoints/sft/checkpoint_epoch0.pt"   # the policy we trained in the SFT phase -> RM backbone

# Guard required because macOS spawns DataLoader workers by re-importing this file.
# Without it, every worker would re-run training and recursively spawn more workers.
if __name__ == "__main__":
    # tokenizer up front to build the pairwise dataloaders (chat template applied per chosen/rejected side).
    # borrows the Llama-3 chat template from the Instruct variant (the base has none).
    tokenizer = get_tokenizer(MODEL)

    # pad-token choice matters for the RM (issue #8): the HF reward model pools the last token via
    # pad_token_id on input_ids, so if pad_id == eos_id a response ending in EOS makes pooling land on
    # the wrong token. Use Llama 3.2's DEDICATED padding token so pad_id != eos_id -> unambiguous.
    # Setting it here propagates to BOTH the collate's pad value and the model's config.pad_token_id.
    tokenizer.pad_token = "<|finetune_right_pad_id|>"

    # static padding: constant (batch, max_length) shape on both sides -> no MPS allocator fragmentation
    USE_DYNAMIC_PADDING = False

    # The RM does TWO forward passes per example (chosen + rejected), so keep the batch modest on MPS.
    # filter_small keeps only pairs where BOTH sides fit in max_length, capped to max_examples.
    dataloaders = create_preference_dataloaders(
        tokenizer, max_length=256, batch_size=4, num_workers=4,
        dataset_name="allenai/llama-3.1-tulu-3-8b-preference-mixture",
        filter_small=True, max_examples=20000, filter_scan=50000,
        use_dynamic_padding=USE_DYNAMIC_PADDING,
    )

    trainer = RewardModelTrainer(
        sft_model_checkpoint_path=SFT_CHECKPOINT,
        base_model_name_hf=MODEL,
        sft_model_dtype=torch.bfloat16,
        sft_model_tokenizer=tokenizer,
        dataloaders=dataloaders,
        optimizer=None,            # built as AdamW(lr) inside the trainer
        lr=1e-5,                   # RMs typically train at a smaller LR than SFT (2e-5)
        l2_reg=1e-3,               # reward-L2: bounds reward scale so PPO's reward-vs-KL balance stays sane
        # l2_reg=0,               # reward-L2: bounds reward scale so PPO's reward-vs-KL balance stays sane
    )

    # RM "peek": fixed (prompt, chosen, rejected) triples. We watch r_chosen pull above r_rejected
    # (margin widening, verdict ✓) as the RM learns to rank — the qualitative read, like SFT's peeks.
    PEEK_PAIRS = [
        {
            "prompt": "What is the capital of France?",
            "chosen": "The capital of France is Paris.",
            "rejected": "I think it might be Lyon or somewhere in Germany, not totally sure.",
        },
        {
            "prompt": "Write a haiku about the ocean.",
            "chosen": "Endless blue expanse / waves whisper to the pale shore / gulls trace the salt wind",
            "rejected": "The ocean is big and has water and fish and it is very deep and blue ok.",
        },
        {
            "prompt": "Explain why the sky is blue.",
            "chosen": "Sunlight scatters off air molecules, and shorter blue wavelengths scatter most (Rayleigh scattering), so the sky looks blue.",
            "rejected": "The sky is blue because it reflects the ocean which is also blue.",
        },
    ]

    # to RESUME from a checkpoint instead of training fresh (load_checkpoint lives on the trainer):
    #   epoch, loss = trainer.load_checkpoint("checkpoints/rm/checkpoint_<YYYY-MM-DD>_latest.pt")
    # loads model + optimizer state; note it does NOT fast-forward the LR schedule or skip seen batches.

    # val + peek now share one cadence (~10% of the run), set inside train_epoch — no peek_every_n_steps knob
    trainer.train(epochs=1, checkpoint_every_n_steps=1000, peek=True, peek_pairs=PEEK_PAIRS)
