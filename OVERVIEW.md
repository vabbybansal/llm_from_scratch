# llm_from_scratch — Components & Sticking Points

The **index**: a high-level map of the codebase plus the key challenges encountered while building it, organized by phase.

---

## 1. What this repo is

An educational, build-from-primitives LLM project:

- **Pretraining** — a GPT-2-style decoder built from scratch in PyTorch + tiktoken, trained on WikiText.
- **Post-training (SFT)** — supervised fine-tuning of a *pretrained* **Llama-3.2-1B** (loaded via HuggingFace) on the **Tulu-3 SFT mixture**, reusing a similar trainer structure.

The point is to *understand every moving part*, not to build a production model.

---

## 2. Goals — the full roadmap

The ambition is to build **the entire modern LLM stack from primitives**, end to end. Each stage is a deliberate step along one throughline: **off-policy → on-policy**. Pretraining and SFT learn by *imitating* a fixed target (next token, expert response); every stage after them learns from completions scored *against the model itself*, which is what fixes SFT's exposure bias and hallucination (see §5 "Post-training / SFT"). We do all of it.

| # | Stage | Status | What it is / why it's here |
|---|---|---|---|
| 1 | **Pretraining** | ✅ done | Next-token prediction on raw text — the base language capability. Off-policy imitation of a static corpus. |
| 2 | **SFT** | ✅ done | Supervised fine-tuning on instruction/response pairs — installs the chat format + "answer the instruction" behavior. Still **off-policy** (targets written by humans / a frontier model), which is exactly why we don't stop here. |
| 3 | **DPO** | ⬜ planned | Direct Preference Optimization — learn from `(chosen, rejected)` pairs with a simple contrastive loss, **no reward model**, KL-anchored to a frozen reference. Offline/mixed-policy: the cheapest, most stable preference step (Tulu-3's first preference stage). |
| 4a | **RLHF — RM + PPO (off-policy RM)** | ⬜ planned | The classic InstructGPT recipe, and the one that teaches the *most machinery*: train a **reward model** on off-the-shelf preference comparisons (the **Tulu-3 preference mixture**, for pipeline coherence with our SFT), then optimize the policy **on-policy** with PPO (actor-critic) under a KL penalty to a reference. The full alignment loop — and a chance to *observe* reward hacking when the policy drifts off the RM's (off-policy) training distribution. |
| 4b | **RLHF — RM + PPO (on-policy / LLM-judged RM)** | ⬜ planned | One round of **iterative RLHF**: sample completions from *our own* SFT model, have a **local Instruct model (Ollama) judge** which is better → on-policy preference pairs → retrain the RM on-distribution → PPO again. Tests whether on-policy RM data curbs the reward hacking seen in 4a. |
| 5 | **RLHF — RM + GRPO** | ⬜ planned | Same reward-model setup, but swap PPO for **GRPO** — drop the value/critic network and estimate advantage from a *group* of sampled completions. Lighter than PPO; the DeepSeek-R1 lineage. |
| 6 | **RLVR** | ⬜ planned | RL with **Verifiable Rewards** — no learned RM; the reward is a programmatic check (math answer matches ground truth, instruction-following constraint satisfied). Fully on-policy; Tulu-3's final stage and the engine behind reasoning models. |
| 7 | **LoRA** | ⬜ planned | Cross-cutting efficiency unlock: freeze the base weights, train tiny low-rank adapters — *no* gradients/optimizer state for the 1.24B params. The thing that makes everything above (esp. RLHF's policy + frozen reference) actually fit a 48 GB Mac. |
| 8 | **Inference Engineering** | ⬜ planned | Make the trained weights *usable* — close the gap between our toy `generate()` and lab inference. Pointers: **KV cache** (kill the O(n²) recompute-every-step), **masking at decode** (mask *finished* sequences instead of burning compute on them; left-pad + adjusted `position_ids` for batched generation — see the causal-vs-padding mask note in §5), **continuous batching** (vLLM/TGI), **quantization** (INT8/INT4), **speculative decoding**, **beam search**, a composable **logits-processor/warper** chain (vectorized repetition penalty, no-repeat-ngram, bad-words, `min_new_tokens`), and **per-sequence EOS** (stop finished rows, keep going on the rest). The concrete to-do list lives in the `generate()` docstring in `libs/utils.py`. |

The progression is also the conceptual one: **imitation (1–2) → preference (3–5) → verifiable reward (6)**, with **LoRA (7)** as the efficiency substrate and **inference (8)** as the deployment layer.

---

## 3. Results — evidence it worked

### SFT: base model → instruction-follower (~1 hr, 20K-example Tulu-3 subset)

The clearest signal is **peek-generation** — sampling the same fixed prompts throughout training. The base Llama-3.2-1B (which has never seen the chat format) goes from gibberish to coherent, on-task answers (peeks condensed):

**Step 0 — base model, chat format unfamiliar:**
```
[What is the capital of France?]
  -> iteDatabaseBeginInitdropIfExists iteDatabaseBeginInit itageBeginInit ...
```

**Step ~2400 — end of one epoch:**
```
[What is the capital of France?]
  -> Paris is the capital of France.
[Write a haiku about the ocean.]
  -> The sea is blue. / The waves are crashing. / The water is salty and clear.
[Explain the difference between machine learning and deep learning.]
  -> Machine learning is a type of supervised learning ... Deep learning ... uses neural networks ...
```

Notably the **training loss barely moves** — the base model is already a strong language model, so SFT isn't teaching it language, it's teaching the **chat format + "answer the instruction" behavior**. That's invisible in the loss curve and obvious in the peeks. Final validation loss ≈ **1.23** (perplexity ≈ **3.4**).

### Pretraining: perplexity as the success signal

The from-scratch GPT has no instruction behavior to eyeball, so the metric is **validation perplexity = `exp(val cross-entropy)`** — the effective number of equally-likely tokens the model weighs per step (lower = more confident). It starts near chance and falls steadily (logged live to wandb + `checkpoints/metrics.csv`); the ~30M-param model lands around **perplexity ≈ 30**, roughly GPT-2-small's range (a 4× larger model) — a strong result at this scale. Peek samples (`checkpoints/peek.txt`) read as progressively more fluent English as the curve drops.

---

## 4. Component map

### Pretraining (custom GPT, `pretraining/`)

| Component | File | Role |
|---|---|---|
| `Tokenizer` | `pretraining/data/tokenizer.py` | tiktoken `gpt2` BPE wrapper |
| `LLMDataset` | `pretraining/data/llm_dataset.py` | sliding-window next-token pairs over EOS-joined docs |
| `create_dataloaders` | `pretraining/data/data_loader.py` | wraps dataset → train/val/test DataLoaders |
| `MultiHeadAttention` | `pretraining/model/multihead_attention.py` | causal MHA via upper-triangular mask buffer |
| `FeedForward` | `pretraining/model/feedforward.py` | 2-layer MLP, 4× expansion, GELU |
| `TransformerBlock` | `pretraining/model/transformer_block.py` | **pre-norm** + residuals |
| `GPT` | `pretraining/model/gpt.py` | token+pos embeddings → blocks → final LN → LM head (optional weight tying) |
| `PreTrainLanguageModelDriver` | `pretraining/trainer.py` | train/eval loop, loss, peek-generation, checkpoint |
| `TrainingLogger` | `pretraining/logger.py` | CSV + wandb sinks, perplexity |
| `load_gpt2_weights` | `pretraining/load_weights.py` | load OpenAI GPT-2 weights into the custom `GPT` |

### Post-training / SFT (`supervised_fine_tuning/`)

| Component | File | Role |
|---|---|---|
| `get_tokenizer` / `get_model` | `load_pretrained_hf_model.py` | load Llama-3.2-1B (bf16, MPS) + borrow chat template |
| `generate` (chat) | `load_pretrained_hf_model.py` | chat-formatted generation, stops on `<|eot_id|>` |
| `SFTDataset` | `sft_dataset.py` | chat-template tokenize → response-only `-100` mask → shift |
| `sft_collate` | `sft_dataset.py` | dynamic **or** static right-padding |
| `create_sft_dataloaders` | `sft_dataset.py` | load/filter/split Tulu-3, build loaders |
| `SupervisedFineTuner` | `fine_tuner.py` | train/eval loop, LR schedule, grad clip, checkpoint, peek |
| `finetune.py` | `finetune.py` | entry point wiring it all together |

### Preference alignment / RLHF (`preference_fine_tuning/`)

| Component | File | Role |
|---|---|---|
| `OffPolicyDataset` | `dataset/preference_dataset.py` | pairwise (chosen/rejected) chat-template tokenize — no `-100` mask, no shift |
| `preference_collate` | `dataset/preference_dataset.py` | right-pad chosen/rejected **+ build attention masks** (load-bearing for RM pooling) |
| `create_preference_dataloaders` | `dataset/preference_dataset.py` | load/filter/split the Tulu-3 preference mixture, build loaders |
| `RewardModelTrainer` | `off_policy_rm_ppo/rewardmodeltrainer.py` | Bradley-Terry loss, preference-accuracy metric, RM "peek", LR schedule, grad clip, checkpoint |
| `train_rm.py` | `off_policy_rm_ppo/train_rm.py` | entry point on Tulu-3; sets a distinct pad token for unambiguous pooling |

### Shared (`libs/`)

After the reorg, `libs/` holds only genuinely cross-stage utilities (the from-scratch GPT stack moved into `pretraining/`).

| Component | File | Role |
|---|---|---|
| `get_device` | `libs/utils.py` | pick the device (cuda > mps > cpu) |
| `generate` | `libs/utils.py` | autoregressive sampling (temp, top-k/p, eos, rep-penalty) — currently GPT-shaped, intended to become the cross-stage inference path (see TODOs) |
| `create_rm_classifier_from_lm_hf` | `libs/utils.py` | LM → reward model via HF `AutoModelForSequenceClassification(num_labels=1)`, backbone warmed from the SFT checkpoint |
| `create_rm_classifier_from_lm_scratch` | `libs/utils.py` | LM → reward model by hand: borrow the HF causal-LM backbone + a `Linear(d,1)` head + attention-mask last-token pooling |

---

## 5. Sticking points & things to keep in mind

> Concise pointers, expanded where the *why* isn't obvious.

### Pretraining

- **Decoding ≠ training.** Repetitive or incoherent output is usually a *decoding* artifact, not broken weights. Greedy decoding picks the single highest-probability token each step, and on a flat distribution that collapses into loops. Sampling with temperature + top-k/top-p truncates the unlikely tail and restores diversity. So before blaming the model, change the decoder.
- **Greedy returns the *mode* of the distribution.** Ideal when the distribution is peaked (factual Q&A, code, math — one right answer), pathological when it's flat (open-ended writing — the "mode" is just a marginally-ahead token that loops). Temperature 0 ⇒ greedy; higher temperature flattens the softmax toward uniform.
- **Temperature scales logits before softmax** — `softmax(logits / T)`. `T<1` sharpens (more deterministic), `T>1` flattens (more random), `T=1` = the model's true distribution. `T=0` is **special-cased to argmax** (greedy), since dividing by 0 gives ±inf.
- **Sampling draws from the distribution** via `torch.multinomial(probs, 1)` — picks a token index *proportional to* its probability (vs greedy `argmax`). This is what escapes the repetition attractor greedy falls into on flat distributions.
- **top-k / top-p (nucleus) truncate the tail before sampling.** top-k keeps the `k` highest-prob tokens; top-p keeps the smallest set whose cumulative prob ≥ `p`. Both set the implausible tail to `-inf` so you get diversity without garbage.
- **`repetition_penalty`** divides the logits of already-generated tokens (>1 discourages repeats) — a decode-time fix for loops.
- **Generation loop:** append one token per step (`torch.cat(..., dim=-1)`) until `max_new_tokens` or an `eos_token_id` is emitted; `logits[:, -1, :]` selects the last position's prediction `(b, vocab)` (integer indexing drops the seq dim — only the last step predicts the next token).
- **LR warmup → cosine decay.** Warmup ramps the LR up from ~0 over the first few hundred–thousand steps because Adam's gradient mean/variance estimates are unreliable early — a full LR then can destabilize training. After warmup, cosine decay eases the LR down a `cos` curve (fast progress early, careful settling late), which empirically reaches a lower final loss than constant or linear.
- **perplexity = exp(cross-entropy loss).** The "effective number of equally-likely tokens" the model is choosing among per step — lower is better. e.g. loss 1.2 ⇒ ppl ≈ 3.3 (deciding among ~3 tokens).
- **`cross_entropy` flattens `(b, seq, vocab) → (b·seq, vocab)`** and targets `→ (b·seq,)`, treating every token position as an independent classification over the vocab. It averages over all non-ignored (`!= -100`) positions automatically.
- **Validate more often than once per epoch.** A long epoch can silently diverge for hours; run a fast *subset* of val batches every N steps and full val at epoch end.
- **At small scale you're data-limited.** More tokens beat more epochs (extra epochs over the same data mostly memorize). And small/undertrained models have flat, "mushy" distributions where greedy has nothing solid to grab — another reason their raw output looks bad.
- **EOS separates documents; `stride` controls window overlap.** Docs are concatenated with an EOS token so the model learns "this text ended" instead of bleeding context across boundaries. The sliding window advances `stride` tokens per sample: `stride < max_length` ⇒ overlapping windows (more samples, some redundancy); `== max_length` ⇒ no overlap.
- **Split train/val by *document*, not random window.** Random windowing over concatenated text leaks near-identical overlapping windows into both splits, inflating val scores. Splitting at document boundaries keeps val genuinely unseen.
- **Pre-norm (LayerNorm *before* attention/FF, inside the residual).** Keeps a clean residual path and stabilizes gradients in deep stacks (post-norm is harder to train without careful warmup). Requires `d_in == d_out` per block so the residual add lines up.
- **Weight tying inflates the reported param count.** When the token embedding and output head share one tensor, tools like `torchinfo` still count both, over-reporting by ~`vocab × d_model`. The real model is smaller.
- **`model.train()` vs `model.eval()` toggles dropout.** Eval disables dropout so validation/inference is deterministic and uses the full network; forgetting to switch makes val noisy.

### Self-attention (the from-scratch causal MHA)

- **Q/K/V come from one combined projection.** `Linear(d_in, 3·d_out)` then `.split(d_out, dim=-1)` → one matmul instead of three, matching HF GPT-2's `c_attn` layout (eases weight loading).
- **Scale scores by `1/√d_k`.** `Q·K` is a sum of `d_k` products, so its magnitude grows with head dim; without scaling, softmax saturates and gradients vanish. `d_k = head_dim`.
- **Heads via `view` + `transpose`.** `view(b, num_tokens, n_heads, head_dim)` splits the channel dim, `transpose(1,2)` → `(b, n_heads, num_tokens, head_dim)` so heads attend in parallel; merge back with `transpose` then `contiguous().view` (transpose leaves memory non-contiguous).
- **Causal mask = `triu(ones, diagonal=1)` as a `register_buffer`**, `masked_fill_(-inf)` before softmax so each token sees only itself + earlier tokens. `register_buffer` moves with `.to(device)` but isn't a learned param; sliced `[:num_tokens, :num_tokens]` at runtime.
- **Two different masks, one slot: causal vs padding.** The causal mask above is *structural* — identical for every sequence, decoder-only, "no peeking at the future." It's a separate thing from a **padding mask** (HF's `attention_mask`, a `(b, seq)` tensor of 1/0) which is *per-sequence* and marks pad vs real for batching. HF folds **both** into a single additive `-inf` bias before softmax (a position is attended only if it's *both* in the past **and** not padding). Our from-scratch MHA carries only the causal mask — and that's enough for **right-padded** training: real tokens never attend to trailing pads (they sit in the future, already blocked), and pad outputs are discarded. The padding mask becomes **mandatory** only for **left-padding** (generation: pads now sit in the "past", so the causal mask won't block them) — and is needed, as plain index bookkeeping rather than attention, to locate the last real token when **pooling an RM**. (Encoders like BERT flip this: no causal mask, padding mask only.)
- **"Mask" is a general pre-softmax slot.** Beyond causal + padding it can carry **sliding-window / local** attention (Mistral), a **block-diagonal doc-packing** mask (blocks cross-document attention when several docs are packed into one sequence), **prefix-LM** (bidirectional over the prompt, causal over the completion), or a **distance bias** like ALiBi (added, not 0/−inf). And FlashAttention / PyTorch **SDPA** apply all of this *fused inside the kernel* via an `is_causal` flag + optional mask — never materializing the `(seq, seq)` matrix (the memory win noted in §"Memory & performance").
- **`num_tokens` comes from `X.shape`, not `context_length`.** The score matrix is `(num_tokens, num_tokens)`, sized by the actual input — which is why variable-length inputs work and `context_length` only bounds the pre-allocated mask.
- **`qkv_bias` must match the pretrained source** — GPT-2 uses Q/K/V biases, Llama omits all biases.

### Post-training / SFT

- **Loss on the response only — the core difference from pretraining.** Pretraining scores every token; SFT masks the prompt and padding to `-100` (which `cross_entropy` ignores), so loss is computed *only* on the assistant's response. The model learns *what to answer*, not to predict/echo the instruction it was handed.
- **SFT data is *off-policy* — and that's intended.** The Tulu-3 response targets were written by humans or a frontier model (GPT-4o / Claude 3.5 Sonnet), **not** sampled from the base model being trained. SFT is imitation learning / behavioral cloning, so the gap between teacher and student *is* the learning signal — you don't need (or want) in-distribution data here. The catch is that cloning a stronger teacher's answers can teach the weak model to **hallucinate confidently** (assert facts it doesn't internally know) and causes **exposure bias** (at inference it conditions on its own prefixes, never seen in training). That limitation is exactly why labs don't stop at SFT: they follow it with **on-policy** stages — DPO/RLVR (Tulu-3's path) or rejection-sampling SFT (RFT/STaR) — that train on completions sampled from the model itself.
- **Find the prompt/response boundary by tokenizing twice.** `full = apply_chat_template(messages)` vs `prompt = apply_chat_template(messages[:-1], add_generation_prompt=True)`. Everything up to `len(prompt)` is masked, the rest is kept — the length difference *is* the response.
- **The input→target shift must happen exactly once.** `logits[i]` predicts token `i+1`, so input and target are offset by one. Do it in the dataset ("Option B": `input=ids[:-1]`, `target=labels[1:]`, loss unshifted) *or* inside the loss (`logits[:-1]` vs `labels[1:]`) — never both, or you get a silent off-by-one.
- **Left-truncate over-length examples.** The response sits at the end, so chopping from the right would delete the very thing you train on. `ids[-max_length:]` keeps the response intact (drops oldest prompt tokens / BOS, which Llama tolerates).
- **GOTCHA: base models have no chat template.** `Llama-3.2-1B` (base) has `chat_template is None` → `apply_chat_template` raises. Borrow it from the `-Instruct` variant (`base.chat_template = instruct.chat_template`). Ideally also copy the Instruct generation config (stop tokens), or generation won't stop on `<|eot_id|>`.
- **GOTCHA: `apply_chat_template(tokenize=True)` returns a `BatchEncoding`, not a list.** `len()` gives `2` (the dict keys) and slicing silently operates on the wrong object. Pass `return_dict=False` for a flat list of ids.
- **EOS (`<|end_of_text|>`, 128001) ≠ EOT (`<|eot_id|>`, 128009).** EOS = the whole text is done (pretraining document separator); EOT = *this turn* is done (chat). SFT teaches the model to end its reply with EOT, so `generate()` must list EOT as a stop token or it sails past the response end and loops to `max_new_tokens`.
- **Right-pad for training, left-pad for generation.** Training reads loss at every position, so trailing pads are harmless: the causal mask hides future pads from real tokens, and pad targets are `-100` → no attention mask needed. Generation reads only `logits[:, -1]` and decodes the batch in lockstep, so every sequence's real last token must align at the right edge → left-pad, *with* an attention mask (and adjusted position ids) since pads now precede real tokens.
- **Pad input with `<PAD>`, target with `-100`.** The input needs an embeddable token id (its value is irrelevant — the causal mask makes pads invisible to real tokens); the target uses `-100` so those positions contribute zero loss. Two fill values for two different roles.

### Preference alignment / RLHF (RM + PPO)

- **The reward model is warmed from the SFT checkpoint, not the base pretrained model.** The RM is the **SFT model with the LM head swapped for a scalar head** (`AutoModelForSequenceClassification(num_labels=1)`), *not* a fresh head on the base model. Three reasons: (1) the RM must read chat-formatted `prompt + response`, and only the SFT model has learned that format; (2) SFT representations are already shaped around instruction-following — the exact axis preferences vary along — so the RM converges on far less preference data; (3) it's what the labs do (InstructGPT's RM, Tulu-3's `…-8B-RM`). In fact **all four PPO models trace back to the same SFT ancestor**: policy (actor), frozen reference, RM (+scalar head), and value/critic (usually initialized from the RM).
- **LM → reward model is a head swap, not surgery on the body.** Keep the entire backbone through the final norm (hidden states `(b, seq, d)`); just replace the LM head (`Linear(d, vocab)`) with a **scalar head** `Linear(d, 1, bias=False)`. Pool to one vector per sequence by taking the **last *non-pad* token's** hidden state — under causal attention only the final token has attended to the whole input (a bidirectional encoder uses a *front* `[CLS]`; a decoder uses the *last* token). Gotcha: with right-padding the literal last position is a pad, so index the last *real* token via the attention mask (`mask.sum(1)-1`), not `[:, -1]`. The head output is a **raw, unbounded scalar — no sigmoid/activation**; the sigmoid belongs in the Bradley-Terry loss on the *difference* `r_chosen − r_rejected`, not on the reward itself (squashing each reward saturates that difference and kills the gradient).
- **Loading SFT weights into the classifier needs `strict=False`.** Our SFT checkpoint is a *causal-LM* `state_dict`; the `SequenceClassification` wrapper's backbone keys match but the scalar `score` head is brand new — so load non-strict and verify that **only** the head appears in `missing_keys` (it's meant to be random and trained from scratch; if backbone keys are missing, the names don't line up and the backbone isn't actually warmed). Set `model.config.pad_token_id` so the reward is read from the last *real* token (decoder pooling), not a pad.
- **The RM loss is Bradley-Terry = BCE on the reward *margin* = 2-class cross-entropy.** `loss = -logsigmoid(r_chosen − r_rejected)`, equivalently `BCEWithLogits(margin, target=1)`; and because `σ(r_c − r_r) = softmax([r_c, r_r])[chosen]`, it's *also* exactly 2-class cross-entropy over the pair with "chosen" as the label. What it is **not**: independent BCE on each reward with labels `[1,0]` — that's *pointwise* (forces absolute reward values), not comparative. Always feed raw logits (the margin) to `BCEWithLogits`/`logsigmoid`; never `sigmoid`→`BCELoss` (the fused form avoids `log(0)` NaNs).
- **The RM metric is preference accuracy — and ~66% is good, not weak.** Perplexity is meaningless here; track `(r_chosen > r_rejected).mean()` (random = 0.5). Don't read the scale as 50→100: human inter-annotator agreement on pairwise preference is only ~65–75%, so the labels are ~70% self-consistent and *that's the ceiling*. Our 1B RM on the Tulu-3 mixture hit **val acc 0.66** — near the ceiling. Accuracy blends easy pairs (~80%+) and near-ties (~50%).
- **Bradley-Terry only pins the *margin* → the reward's absolute offset and scale are unconstrained.** A constant added to all rewards leaves the margin (and loss) unchanged, and the loss keeps dropping as margins widen, so absolute rewards both drift and grow (we watched them swing +1 → −3 → 0 across a run). Harmless for the RM itself — read the *margin* and ✓/✗, never absolute values — but it splits into two things at PPO time, and only one needs fixing here:
  - **Offset** is cancelled by the PPO **baseline** (`advantage = reward − V(prompt)`; see the baseline bullet below). A constant/per-prompt shift just gets absorbed into `V`, so it needs no fix.
  - **Scale relative to the KL penalty** is the real issue, and the baseline does *not* fix it. PPO's per-token reward is `r_total = r_RM − β·KL(policy‖reference)`; the KL term is a live penalty that pulls the policy back toward the SFT reference, and that tug-of-war only works if `r_RM` and `β·KL` are comparable in magnitude. If the RM is free to emit ±50 rewards, `β·KL` (≈ O(1)) is **swamped** → the KL guard effectively vanishes → the policy ignores the reference and **reward-hacks**. Advantage normalization can't rescue this: it rescales the whole `r_total` together, preserving the skewed `r_RM : β·KL` ratio.
  - **Fix:** a small **reward-L2 regularizer** `+ λ(r_c² + r_r²)` (λ~1e-3) bounds the reward scale near 0 *without changing ranking* (it pulls `r_c`, `r_r` toward 0 symmetrically, so BT still decides the order), which keeps `β` meaningful and transferable across runs — or equivalently, **normalize `r_RM` to ~unit scale just before PPO**.
- **An RM is only trustworthy on its training distribution — hence reward hacking.** The RM is a *proxy*; PPO actively searches for high reward, so when the policy drifts to regions the RM never saw it extrapolates badly and the policy games it (**reward over-optimization**). Off-the-shelf preference sets are off-distribution for *our* 1B policy, so the 4a RM is expected to be hackable — that's the point of observing it. Lab fixes: train the RM on the policy's *own* samples (InstructGPT), and refresh it as the policy moves (**iterative RLHF** — Llama-2's ~5 RM rounds); that's exactly the 4b plan (sample from our SFT model, judge with Ollama). The KL-to-reference penalty in PPO is the complementary guard (keeps the policy from wandering off-distribution in the first place).
- **RM pooling needs the attention mask, and pad==eos is a trap.** Unlike SFT (which leans on `-100` + causal masking), the RM has no per-token labels, so the **attention mask is the only record of where real tokens end** — the collate must build it. Two ways to find the last real token: the **HF** `SequenceClassification` scans `pad_token_id` over `input_ids`; the **scratch** RM uses `attention_mask.sum(1)-1`. The HF route breaks if `pad_token == eos` (a reply ending in EOS makes pooling land early), so set a **distinct pad token** (`<|finetune_right_pad_id|>`, id 128004); the scratch route is immune since the mask is authoritative.
- **Cross-prompt reward scale is the *baseline's* job — not the RM's, not the L2 reg's (looking ahead to PPO).** Vanilla policy gradient updates by `raw_reward × ∇logπ`, so a prompt whose rewards sit around 20 would dominate one around 5 *purely on magnitude* — high variance, big-number bias. PPO fixes this with a **baseline**: `advantage = reward − V(prompt)`, where the value/critic learns each prompt's expected reward, so only *beating your own prompt's expectation* drives the update (a 20 on a ~20 prompt and a 5 on a ~5 prompt both give advantage ≈ 0). That's why PPO needs the **value/critic** model, plus per-batch advantage normalization. Keep this distinct from the **global-scale** concern in the reward-L2 bullet above: the **baseline handles per-prompt offsets**; **L2 / reward-normalization keeps the RM's overall magnitude sane relative to the `−β·KL` term**. Two different problems.

### Architecture: GPT-2 → Llama 3.x (rationale for the pivot)

The post-training base was swapped from the scratch GPT-2 to a pretrained **Llama-3.2-1B** because **GPT-2's base output is too incoherent to *observe* the SFT effect** — it's hard to tell whether instruction-tuning helped. Llama-3.2-1B already writes fluently, so the before/after is obvious. Key arch differences (Llama vs GPT-2):

- **RoPE** (rotary) vs learned absolute position embeddings — relative position is baked into `Q·K` and generalizes beyond the training length; GPT-2's learned position table is a hard cap.
- **RMSNorm** vs LayerNorm — skips mean subtraction, ~same quality, faster.
- **SwiGLU** FFN vs GELU — gated, more capacity per parameter.
- **GQA** (grouped-query attention) — multiple Q heads share fewer K/V heads → smaller KV cache.
- **No bias** terms anywhere; **128K vocab** vs GPT-2's ~50K (which is what makes the logits tensor so memory-heavy — see below).

### Memory & performance (the hard-won ones on Mac/MPS)

- **Dtype is the first memory lever; quantization is the next.** FP32 = 4 bytes/param. **BF16 = 2 bytes with the *same dynamic range* as FP32** (only less mantissa precision) → the default for training+inference, halving both memory and bandwidth. FP16 is also 2 bytes but a narrower range (can destabilize training). To go further — inference or **QLoRA** — quantize weights to **INT8/INT4** (via `bitsandbytes`), shrinking them 2–8×; lossy, but it's how big models run on small hardware.
- **Full-FT memory = (bytes/param × params) fixed cost + activations.** The per-parameter fixed cost for AdamW, in pure bf16:
  - 2 bytes — weight (bf16)
  - 2 bytes — gradient (bf16)
  - 2 bytes — Adam `exp_avg` (1st moment / momentum)
  - 2 bytes — Adam `exp_avg_sq` (2nd moment / variance)
  - = **8 bytes/param** (the configuration used here). The "textbook" mixed-precision recipe instead keeps fp32 master weight + fp32 grad + fp32 Adam moments = **16 bytes/param** (more stable, double the memory). **The optimizer dominates** — Adam's two moment buffers alone are 2× the weights. For Llama-3.2-1B (1.24B params): `1.24B × 8 ≈ 10 GB` (or ~20 GB at 16 bytes), *before* activations.
- **Activations — easy to underestimate, especially with a 128K vocab.** These are the intermediate tensors kept for the backward pass; they scale with `batch × seq × …`. Rough terms for this run (batch 8, seq 256, 16 layers, hidden 2048, intermediate 8192):
  - **Logits:** `(8, 256, 128256) ≈ 262M` values `× 2 bytes` (bf16) ≈ **0.5 GB**; an fp32 `.float()` copy is `× 4 bytes` ≈ ~1 GB, doubling it — that single cast triggered an **OOM**.
  - **MLP intermediates** (`batch × seq × 8192 × 16 layers`) dominate the per-layer cost.
  - **Attention scores** would be `batch × heads × seq²`, but HF's memory-efficient SDPA never materializes them.
  Net: a few GB on top of the fixed cost.
- **Swap = death for throughput.** When unified memory overflows, macOS swaps tensors to SSD and every step that touches one waits on disk → throughput collapses (measured here: **~50 s/it vs ~1.3 s/it**). If `mactop` shows **Swap > 0** during training, you're over budget — cut `max_length`, `batch_size`, or the optimizer/dtype footprint until swap stays at 0.
- **Dynamic padding fragments the GPU caching allocator → OOM.** PyTorch caches freed memory *blocks by byte-size* and reuses them. Dynamic padding makes every batch a different sequence length → different block sizes → freed blocks can't be reused, so idle "other allocations" balloon (~45 GB idle in this run) until a fresh allocation fails — *external fragmentation from dynamic shapes*. Fixes: periodic `torch.mps.empty_cache()` (flush idle blocks), or avoid it entirely with **static padding** (every batch the same shape ⇒ blocks recycled forever). The default here is static. (CUDA's allocator splits/coalesces blocks and copes far better; production labs sidestep it with **sequence packing**.)
- **A full Tulu-3 epoch is intractable on a Mac.** `939K examples ÷ batch 4 ≈ 235K steps` — days to months. Filtering to short examples (≤ `max_length`) and capping the count (~20K) → ~2.5K steps → a ~1-hour run, at the cost of a bias toward short/simple tasks. The structural fix is **LoRA**: freeze the base weights so there are *no* gradients or optimizer state for the 1.24B params (the ~10 GB fixed cost largely vanishes; you train tiny adapters instead).
- **HF datasets don't consume RAM.** `datasets` stores Apache Arrow files on disk and memory-maps them: `__getitem__` pages in only the rows it reads, and those pages are clean + file-backed so the OS can evict them instantly. A 6 GB (or 60 GB) dataset costs **disk**, not your GPU/unified-memory budget — which is spent on model + optimizer + activations + in-flight batches.

### Data loading & infra (cross-cutting)

- **`num_workers>0` overlaps CPU data-prep with GPU compute.** With `num_workers=0`, `__getitem__` + collation run inline in the main process and block each step (GPU idles while the CPU tokenizes). With N workers, subprocesses prefetch whole batches into a queue so prep hides behind the previous step's GPU compute. The collate loop is cheap; the real hidden cost is tokenization.
- **macOS spawns DataLoader workers by *re-importing* your script.** Unlike Linux `fork` (which clones the running process in memory), macOS `spawn` starts a fresh interpreter that re-runs the file top-to-bottom to rebuild definitions. If training code sits at module level, every worker re-launches training → recursively spawns more workers. Wrap entry code in `if __name__ == "__main__"` so a worker's *import* doesn't trigger the *run*.
- **CPU does data prep + the `.to(device)` copy; GPU does the math.** Datasets/collation produce CPU tensors; the loop copies each batch to the GPU before the forward pass. On Apple unified memory that copy is cheap (CPU and GPU share physical RAM, no PCIe), so `pin_memory`/`non_blocking` (CUDA async-copy tricks) don't apply.
- **GPU memory hits a steady state — it doesn't grow per step — *unless you pin a graph*.** Refcounting + the caching allocator free each step's tensors when they go out of scope and recycle the slots for the next step. The classic leak: accumulating a graph-carrying tensor (`epoch_loss += loss` instead of `+= loss.item()`) keeps the whole autograd graph alive across steps → OOM creep. Use `.item()`/`.detach()` for anything you keep around.
- **Checkpoint per-epoch (kept) + rolling mid-epoch (overwrite).** A Tulu epoch is long, so per-epoch saves alone mean a mid-epoch crash loses everything. A rolling `latest` checkpoint every N steps bounds the loss to ≤ N steps while keeping disk flat (each full checkpoint = weights + optimizer ≈ several GB).
- **Monitoring a long run.** `wandb` for namespaced loss/ppl charts (`train/`, `val/`) + a `tqdm` postfix for live numbers; **peek-generation** every N steps is the *qualitative* eval — sampling from fixed prompts is how instruction-following is seen to "click" (the loss barely moves; the peeks reveal the transformation). On Mac, `mactop` reads GPU util + memory/swap far more legibly than `powermetrics`, and `caffeinate -i python …` stops the machine sleeping mid-training.

---

## 6. Where to look next

- **Architecture & commands:** [`CLAUDE.md`](CLAUDE.md).
- **Code:** the component tables in §3 map each concept to its file.
