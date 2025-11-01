from applyllm.accelerators import (
    AcceleratorHelper, 
    DIR_MODE_MAP
)
# set up the torch mps environment and huggingface cache home, before importing datasets and transformers
AcceleratorHelper.init_torch_env(accelerator="mps", dir_setting=DIR_MODE_MAP.get("mac_local"))

import math
import inspect
from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F
import os

"""
https://github.com/karpathy/build-nanogpt/blob/master/train_gpt2.py
"""

class CausalSelfAttention(nn.Module):
    """Multi-headed attention layer"""

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        # setting a flag, quick and dirty way to scale the weights
        self.c_proj.NANOGPT_SCALE_INIT = 1
        # regularization
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        # not really a 'bias', more of a mask, but following the OpneAI/HF naming though
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                        .view(1, 1, config.block_size, config.block_size))
        
    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        # nh is the number of heads, hs is "head size", and C (number of channels) = nh * hs
        # e.g. in GPT-2 (124M): nh = 12, hs = 64, nh*hs = C = 768 channels in the Transformer
        qkv = self.c_attn(x)
        # query, key, value
        # query and key multiply to get how interesting they find each other
        q, k, v = qkv.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # [B, nh, T, hs]
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # [B, nh, T, hs]
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # [B, nh, T, hs]
        
        """ Flash Attention: kernel Fusion Operation"""
        # torch.compile is not detecting the Flash Attention, so we need to use the Flash Attention manually
        # utilize the online streaming softmax, to reduce the HBM memory usage
        # note: the flops doesn't matter, the entire memory access hierarchy matters for fast training operation
        # torch compile is great, and there are still a lot of optimization manually available to us.
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)

        """ replaced by the Flash Attention
        # attention (materializes the large (T, T) matrix for all the queries and keys)
        # query and key to interact to give us the attention
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        # autoregressive mask (to make sure token only attend to tokens before them, never to tokens in the future)
        att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        # softmax normalize the attention, and sums to 1 always 
        att = F.softmax(att, dim=-1)
        # matrix multiple with the values, is a weighted sum of values of the tokens we find interesting 
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        """

        # perform a concatenation operation 
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side
        # output project
        y = self.c_proj(y)
        return y

class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        # linear projection
        self.c_fc   = nn.Linear(config.n_embd, 4 * config.n_embd)
        # tensor flow as slow implementation of GELU, so there is tanh approximation version
        # we can just use nn.GELU(approximate='none')
        # but GPT-2 use the approximated GELU version
        self.gelu   = nn.GELU(approximate='tanh')
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1

    def forward(self, x):
        x = self.c_fc(x)
        # gelu non-linearity, GELU is RELU with a smooth transition
        x = self.gelu(x)
        x = self.c_proj(x)
        return x 

class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        # attention is a communication operation, where 1024 tokens lined up in a sequence
        # attention is a aggregation function, where the token exchange information
        # it is a pulling function, weighted sum, a reduce operation
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        # MLP happens every token individually. MLP is the map 
        self.mlp = MLP(config)
    
    # transformer forward pass is a repeated operation of map reduce.
    # reduce the information , and think the information individually they gathered.
    # every block refines the representation from the residual stream
    def forward(self, x):
        """Actual Forward pass of the block"""
        # input x first go through a layer normalization, then attention,
        # then residual connection, which is adding the input to the output of the attention,
        x = x + self.attn(self.ln_1(x))
        # then go through another layer normalization, then MLP (Feed Forward),
        # then residual stream again,
        x = x + self.mlp(self.ln_2(x))
        return x  

@dataclass
class GPTConfig:
    block_size: int = 1024 # max sequence length
    vocab_size: int = 50257 # number of tokens: 50,000 BPE merges + 256 bytes tokens + 1 <|endoftext|>
    n_layer: int = 12 # number of layers
    n_head: int = 12 # number of heads
    n_embd: int = 768 # embedding dimension
    # block_size: int = 256
    # vocab_size: int = 65
    # n_layer: int = 6
    # n_head: int = 6
    # n_embd: int = 384

class GPT(nn.Module):

    def __init__(self, config):
        """
        vocab_size is only used in the embedding layer wte, and the last lm_head layer as weight sharing schema
        increaing the vocab_size doesn't change the model, it introduce more token, and computation
        but for a square matrix, it allows fast computation overall on CUDA (we increase from 50257 to 50304)
        """
        super().__init__()
        self.config = config
        
        # ModuleDict is a dictionary that can contain sub-modules, to index using strings
        self.transformer = nn.ModuleDict(dict(
            # wte token embedding, in the original transformer paper is the Output Embedding
            wte = nn.Embedding(config.vocab_size, config.n_embd), # weights of token embeddings
            # wpe positional embedding, in the original transformer paper is the Positional Encoding
            wpe = nn.Embedding(config.block_size, config.n_embd), # weights of positional embeddings
            # nn.ModuleList is a list that can contain sub-modules, to index using integers
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            # final layer normalization, a new layer added in the GPT-2 paper
            ln_f = nn.LayerNorm(config.n_embd),
        ))
        # final classifier layer, final projection layer, the last linear layer before the Softmax in original transformer paper
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # weight sharing scheme, assign the wte weight witht he lm_head weight, not vise versa
        # it will get worse in training if we assign the lm_head weight with the wte weight
        self.transformer.wte.weight = self.lm_head.weight

        # init params according to the original gpt code
        # apply init weights to all sub-modules
        self.apply(self._init_weights)


    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, 'NANOGPT_SCALE_INIT'):
                # every single transformer block has two addition of attn and mlp 
                # in the forward method of the Block adding to the residual stream
                # that is why the 2 x self.config.n_layer coming from
                std *= (2 * self.config.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            # not using 0.01 std like in the original tf code of gpt2
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
 
  
    
    def forward(self, idx, targets=None):
        """
        this forward function is called
        model = GPT(GPTConfig())
        model(x)

        Args:
            targets: optional labels to compute the loss
        """
        # idx is of shape (B, T), token idx, B x T stacked idx 
        B, T = idx.size() 
        assert T <= self.config.block_size, f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}"
        # forward the token and posisition embeddings
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device) # shape (T)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (T, n_embd)
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (B, T, n_embd)
        # broad casting hidden inside + operator
        x = tok_emb + pos_emb
        # forward the blocks of the transformer
        for block in self.transformer.h:
            x = block(x)
        # forward the final layernorm and the classifier
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x) # (B, T, vocab_size)
        loss = None
        if targets is not None:
            # only compute the loss if targets are provided
            # flatten the logits to be of shape (B*T, vocab_size)
            # flatten the targets to be of shape (B*T) single tensor
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss 

    @classmethod
    def from_pretrained(cls, model_type):
        """Loads pretrained GPT-2 model weights from huggingface"""
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)
        
        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
        }[model_type]
        config_args['vocab_size'] = 50257 # always 50257 for GPT model checkpoints
        config_args['block_size'] = 1024  # always 1024 for GPT model checkpoints
        # create a from-scratch initialized minGPT model
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # discard this mask / buffer, not a param

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all o fthe parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # same, just the mask (buffer)
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over teh other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model
    
    def configure_optimizers(self, weight_decay, learning_rate, device):
        """
        layer decay is like regularization. We pull down the weights and force the network to use more of the weights
        avoid the weights to be too large, to distribute the network to work across more channels.
        """
        # start with all of the candidate parameters (that require gradients)
        param_dict = {pn: p for pn, p in self.named_parameters()} 
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad} 
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodeycay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {"params": decay_params, "weight_decay": weight_decay}, # layers in matrix multiplications, embeddings
            {"params": nodeycay_params, "weight_decay": 0.0}, # layer normal, scales and biases
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodeycay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodeycay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        # kernel fusion, instead of running multiple kernels, and iterate though the tensors
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and 'cuda' in device
        print(f"using fused AdamW: {use_fused}")
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=use_fused)
        return optimizer
# ----------------------------
import tiktoken

class DataLoaderLite:
    def __init__(self, B, T, data_path, model='gpt2'):
        self.B = B
        self.T = T
        self.data_path = data_path,
        self.model = model

        # at init load tokens from disk and store them in memory
        with open(data_path, 'r') as f:
            text = f.read()
        enc = tiktoken.get_encoding(self.model)
        tokens = enc.encode(text)
        self.tokens = torch.tensor(tokens)
        print(f"loaded {len(self.tokens)} tokens")
        print(f"1 epoch = {len(self.tokens) // (B * T)} batches")

        # state
        self.current_position = 0

    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position : self.current_position + B*T+1]
        x = (buf[:-1]).view(B, T) # inputs
        y = (buf[1:]).view(B, T) # targets
        # advance the position in the tensor
        self.current_position += B*T
        # if loading the next batch would be out of bounds, reset
        if self.current_position + (B*T + 1) > len(self.tokens):
            self.current_position = 0
        return x, y
    

# ----------------------------
# attempt to autodetect the device

import time 
device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = "mps"
print(f"using device: {device}")

# set the random seed for reproducibility
my_seed = 1337
torch.manual_seed(my_seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(my_seed)
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    torch.mps.manual_seed(my_seed)

# device = "cpu" # override 
# get the current file parent directory
parent_dir = os.path.dirname(__file__)
training_data_path = os.path.join(parent_dir, "data", "input.txt")


# train_loader = DataLoaderLite(B=4, T=32, data_path=training_data_path)
# Batch size B should be the number of power of 2 to run efficiently on the GPU
train_loader = DataLoaderLite(B=16, T=1024, data_path=training_data_path)

# set the float32 matmul precision to high, to use the high precision matmul kernels
# 'highest' uses FP32, 'high' uses TF32
# https://pytorch.org/docs/stable/generated/torch.set_float32_matmul_precision.html
# using TF32 speed up the nn.Linear operation, but the numbers are still Float32, 
# we are memory bound, still need to keep large tensors in memory
# BF16 exponent is 8 bits and mantissa is 7 bits, instead 10 bits in the TF32
# if you use FP16, you need to use gradient scaling, to avoid underflow
# BF16 is representing the same range as FP32, but with less precision

# move to backend specific settings
# torch.set_float32_matmul_precision('high')
if device == "cuda":
    # new API for CUDA devices
    torch.backends.cuda.matmul.fp32_precision = 'tf32'
    # 'ieee' for full precision   
    # If you also want to set for convolutions:
    # torch.backends.cudnn.conv.fp32_precision = 'tf32'
else:
    # mps doesn't support TF32, so this won't apply
    # You can safely skip or keep the old API for non-CUDA devices
    torch.set_float32_matmul_precision('high')


# torch.set_float32_matmul_precision('medium')

# get logits 
# use the default config to generate a random weights model
# model = GPT(GPTConfig())
# use better even number to represent the vocab size
# additional tokens doesn't break the model, but helps cuda to have fast computation.
# in the original GPT-2, the vocab size is 50257, for mps we also see some small speed up even introduce more compute.
model = GPT(GPTConfig(vocab_size=50304))


# move the entire model to the accelerator, moving all the tensors to the GPU
model.to(device)

# logits, loss = model(x, y)

# torch.compile
if device == "cuda":
    # torch.cuda.memory._record_memory_history(max_entries=10000)
    # model.to(device)
    # https://pytorch.org/tutorials/intermediate/torch_compile_tutorial.html
    # keep the data in the GPU chip memory instead of offloading
    # to the HBM (high bandwidth memory) aka global state of the GPU
    model = torch.compile(model)
elif device == "mps":
    # mps
    # model.to(device)
    # https://discuss.pytorch.org/t/jitting-fails-on-mps-device/192079
    model = torch.compile(model, backend="aot_eager")

# max_lr = 3e-4
max_lr = 6e-4 # 6.0 x 10^-4
min_lr = max_lr * 0.1
warmup_steps = 10
max_steps = 50
def get_lr(it):
    """
    learning rate schedule: cosine decay learning rate schedule with warmup 
    https://miro.medium.com/v2/resize:fit:720/format:webp/1*BJCssPOCn4u__NoAZs392w.png
    """
    # 1) linear warmup for warmup_iters steps
    if it < warmup_steps:
        return max_lr * (it+1) / warmup_steps
    # 2) if it > lr_decay_iters, return main learning rate
    if it > max_steps:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))   # coeff starts at 1 and decays to 0
    return min_lr + coeff * (max_lr - min_lr)

# optimize
# optimizer = torch.optim.AdamW(model.parameters(), lr=6e-4, betas=(0.9, 0.95), eps=1e-8)
optimizer = model.configure_optimizers(weight_decay=0.1, learning_rate=6e-4, device=device)

for step in range(max_steps):
    t0 = time.time()
    # keep the batch on the cpu, to not waste GPU memory
    x, y = train_loader.next_batch()
    # move tensors to the device
    x, y = x.to(device), y.to(device)

    # Enables autocasting for the forward pass (model + loss)
    # with torch.autocast(device_type="cuda"):
    if device == "cuda":
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            # start with a zero gradient
            optimizer.zero_grad()
            logits, loss = model(x, y)
    elif device == "mps":
        optimizer.zero_grad()
        logits, loss = model(x, y)


    # Exits the context manager before backward()
    # adds the loss to gradients
    loss.backward()
    # clip the global norm of the gradients to 1.0
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    # determine and set the learning rate for this iteration
    lr = get_lr(step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    # update the weights/parameters and decrease the loss
    optimizer.step()
    # cpu building up work queue for the GPU, so we need to wait for the GPU to finish
    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()
    t1 = time.time()
    dt = (t1 - t0)*1000 # time difference in milliseconds
    # thoughput in tokens per second during the training, an objective metric
    tokens_per_sec = (train_loader.B * train_loader.T) / (t1 - t0)
    # loss is a tensor with a single element, loss.item() will convert tensor to a single float on CPU
    # loss is a tensor on the GPU, so we need to move it to the CPU to print it
    # print(f"step {i}, loss: {loss.item()}, dt: {dt:.2f}ms, tokens/sec: {tokens_per_sec:.2f}")
    print(f"step {step:4d} | loss: {loss.item():.6f} | lr {lr:.4e} | norm: {norm:.4f} | dt: {dt:.2f}ms | tokens/sec: {tokens_per_sec:.2f}")
# we are overfitting a single batch, so that the transformer can memorize the sequence.
# we shall see the loss decrease to zero

# print the loss function
# print(loss)

# print(logits.shape)
# print(logits[0][0].shape)

# remove the model from the accelerator
del model
import gc 
gc.collect()
if device == "cuda":
    # torch.cuda.memory._dump_snapshot("../data/cuda_memory_snapshot.pkl")
    # torch.cuda.memory._record_memory_history(enabled=None)
    # Analyze the memory usage profile/snapshot.pkl file with PyTorch memory visualizer
    # https://pytorch.org/docs/stable/torch_cuda_memory.html
    torch.cuda.empty_cache()
elif device == "mps":
    torch.mps.empty_cache()

import sys; sys.exit(0)