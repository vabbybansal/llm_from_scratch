GPT2_SMALL = dict(vocab_size=50257, n_layers=12, d_in=768, context_length=1024,
                  dropout=0.1, n_heads=12, qkv_bias=True)

GPT2_MEDIUM = dict(vocab_size=50257, n_layers=24, d_in=1024, context_length=1024,
                   dropout=0.1, n_heads=16, qkv_bias=True)

GPT2_LARGE = dict(vocab_size=50257, n_layers=36, d_in=1280, context_length=1024,
                  dropout=0.1, n_heads=20, qkv_bias=True)
