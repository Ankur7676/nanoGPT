# example config for training GPT-2 (124M) with FlashAttention-2 kernels (the flash-attn package)
# same recipe as config/train_gpt2.py, only the flash-attn-specific flags differ.
# requires: pip install flash-attn (Linux + CUDA, Ampere-or-newer GPU)
# launch as:
# $ torchrun --standalone --nproc_per_node=8 train.py config/train_flash_attn.py

wandb_log = True
wandb_project = 'owt'
wandb_run_name = 'gpt2-124M-flash-attn'

# these make the total batch size be ~0.5M
# 12 batch size * 1024 block size * 5 gradaccum * 8 GPUs = 491,520
batch_size = 12
block_size = 1024
gradient_accumulation_steps = 5 * 8

# this makes total number of tokens be 300B
max_iters = 600000
lr_decay_iters = 600000

# eval stuff
eval_interval = 1000
eval_iters = 200
log_interval = 10

# weight decay
weight_decay = 1e-1

# flash-attn specific
use_flash_attn = True
window_size = -1 # -1 = full context; set e.g. 256 to use causal sliding-window attention instead
alibi = False # set True to add an ALiBi positional bias to attention scores
flash_attn_deterministic = False # set True for a bitwise-reproducible backward pass, at some speed cost