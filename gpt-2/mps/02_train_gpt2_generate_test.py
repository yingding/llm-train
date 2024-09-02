import math
from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F

from applyllm.accelerators import (
    AcceleratorHelper, DIR_MODE_MAP
)
import os

# set up the torch mps environment and huggingface cache home, before importing datasets and transformers
AcceleratorHelper.init_mps_torch(dir_setting=DIR_MODE_MAP.get("mac_local"))

# ----------------------------
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
        # attention (materializes the large (T, T) matrix for all the queries and keys)
        # query and key to interact to give us the attention
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        # autoregressive mask (to make sure token only attend to tokens before them, never to tokens in the future)
        att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        # softmax normalize the attention, and sums to 1 always 
        att = F.softmax(att, dim=-1)
        # matrix multiple with the values, is a weighted sum of values of the tokens we find interesting 
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
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
    
    def forward(self, idx):
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
        return logits

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
    

# ----------------------------
num_return_sequences = 5
max_length = 30 
accelerator = 'mps' # 'cuda' or 'mps'

model = GPT.from_pretrained('gpt2')
# put the model in eval mode, you not going to train it
model.eval()
# move the entire model to the accelerator, moving all the tensors to the GPU
model.to(accelerator)

# prefix tokens
import tiktoken
enc = tiktoken.get_encoding('gpt2')
tokens = enc.encode("Hello, I'm a language model,")
tokens = torch.tensor(tokens, dtype=torch.long) # (8,)
tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1) # (5, 8)
x = tokens.to(accelerator)

# generate! right now x is (B, T) where B = 5, T = 8
# set the seed to 42
torch.manual_seed(42)
while x.size(1) < max_length:
    # forward the model to get the logits
    with torch.no_grad(): # doesn't need to cache intermediate activations
        logits = model(x) # (B, T, vocab_size)
        # take the logits at the last position
        logits = logits[:, -1, :] # (B, vocab_size)
        # get the probabilities
        probs = F.softmax(logits, dim=-1)
        # do top-k sampling of 50 (huggingface pipeline default)
        # topk_probs here becomes (5, 50), topk_indices is (5, 50)
        topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
        # select a token from the top-k probabilites
        ix = torch.multinomial(topk_probs, 1) # (B, 1)
        # gather the corresponding indices
        xcol = torch.gather(topk_indices, -1, ix) # (B, 1)
        # append to the sequence
        x = torch.cat((x, xcol), dim=1)

# print the generated text
for i in range (num_return_sequences):
    tokens = x[i, :max_length].tolist()
    decoded = enc.decode(tokens)
    print(">", decoded)

# remove the model from the accelerator
del model
import gc 
gc.collect()
torch.mps.empty_cache()


