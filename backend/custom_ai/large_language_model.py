import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np


class RotaryPositionalEmbedding(nn.Module):
    """rotary position embeddings for better long context"""

    def __init__(self, dim, max_seq_len=8192):
        super().__init__()

        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)

        self.max_seq_len = max_seq_len

    def forward(self, seq_len, device):
        t = torch.arange(seq_len, device=device).type_as(self.inv_freq)
        freqs = torch.einsum('i,j->ij', t, self.inv_freq)

        emb = torch.cat((freqs, freqs), dim=-1)

        return emb.cos(), emb.sin()


def apply_rotary_emb(x, cos, sin):
    """applies rotary embeddings"""

    x1, x2 = x[..., ::2], x[..., 1::2]

    rotated = torch.stack([-x2, x1], dim=-1).flatten(-2)

    x_rotated = (x * cos) + (rotated * sin)

    return x_rotated


class MultiQueryAttention(nn.Module):
    """multi-query attention for faster inference"""

    def __init__(self, d_model, num_heads, num_kv_heads=None):
        super().__init__()

        self.d_model = d_model
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads or num_heads
        self.head_dim = d_model // num_heads

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(d_model, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)

        self.rotary = RotaryPositionalEmbedding(self.head_dim)

    def forward(self, x, mask=None, use_cache=False, past_kv=None):
        batch_size, seq_len, _ = x.shape

        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_kv_heads, self.head_dim)

        cos, sin = self.rotary(seq_len, x.device)
        cos = cos[None, :, None, :]
        sin = sin[None, :, None, :]

        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)

        if past_kv is not None:
            past_k, past_v = past_kv
            k = torch.cat([past_k, k], dim=1)
            v = torch.cat([past_v, v], dim=1)

        if self.num_kv_heads < self.num_heads:
            repeat_factor = self.num_heads // self.num_kv_heads
            k = k.repeat_interleave(repeat_factor, dim=2)
            v = v.repeat_interleave(repeat_factor, dim=2)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        attn_weights = F.softmax(scores, dim=-1)

        output = torch.matmul(attn_weights, v)

        output = output.transpose(1, 2).contiguous()
        output = output.view(batch_size, seq_len, self.d_model)

        output = self.o_proj(output)

        if use_cache:
            return output, (k, v)

        return output


class SwiGLU(nn.Module):
    """swish gated linear unit - better than standard FFN"""

    def __init__(self, d_model, d_ff):
        super().__init__()

        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_ff, d_model, bias=False)
        self.w3 = nn.Linear(d_model, d_ff, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class RMSNorm(nn.Module):
    """root mean square layer normalization - more stable"""

    def __init__(self, dim, eps=1e-6):
        super().__init__()

        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * norm * self.weight


class TransformerBlock(nn.Module):
    """advanced transformer block with all optimizations"""

    def __init__(self, d_model, num_heads, d_ff, num_kv_heads=None, dropout=0.0):
        super().__init__()

        self.attention = MultiQueryAttention(d_model, num_heads, num_kv_heads)
        self.feed_forward = SwiGLU(d_model, d_ff)

        self.attention_norm = RMSNorm(d_model)
        self.ffn_norm = RMSNorm(d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None, use_cache=False, past_kv=None):
        residual = x

        x = self.attention_norm(x)

        if use_cache:
            attn_out, new_kv = self.attention(x, mask, use_cache=True, past_kv=past_kv)
        else:
            attn_out = self.attention(x, mask)
            new_kv = None

        x = residual + self.dropout(attn_out)

        residual = x
        x = self.ffn_norm(x)
        x = self.feed_forward(x)
        x = residual + self.dropout(x)

        if use_cache:
            return x, new_kv

        return x


class LargeLanguageModel(nn.Module):
    """custom large language model - grok level architecture"""

    def __init__(
        self,
        vocab_size=100000,
        d_model=4096,
        num_layers=32,
        num_heads=32,
        num_kv_heads=8,
        d_ff=14336,
        max_seq_len=8192,
        dropout=0.0,
        tie_embeddings=True
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_layers = num_layers
        self.max_seq_len = max_seq_len

        self.token_embedding = nn.Embedding(vocab_size, d_model)

        self.layers = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, num_kv_heads, dropout)
            for _ in range(num_layers)
        ])

        self.norm = RMSNorm(d_model)

        if tie_embeddings:
            self.lm_head = lambda x: F.linear(x, self.token_embedding.weight)
        else:
            self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids, mask=None, use_cache=False, past_kvs=None):
        x = self.token_embedding(input_ids)

        new_kvs = [] if use_cache else None

        for i, layer in enumerate(self.layers):
            past_kv = past_kvs[i] if past_kvs is not None else None

            if use_cache:
                x, new_kv = layer(x, mask, use_cache=True, past_kv=past_kv)
                new_kvs.append(new_kv)
            else:
                x = layer(x, mask)

        x = self.norm(x)

        logits = self.lm_head(x)

        if use_cache:
            return logits, new_kvs

        return logits

    def generate(
        self,
        input_ids,
        max_new_tokens=512,
        temperature=0.8,
        top_k=50,
        top_p=0.9,
        repetition_penalty=1.1,
        do_sample=True
    ):
        self.eval()

        past_kvs = None
        generated = input_ids

        with torch.no_grad():
            for _ in range(max_new_tokens):
                if generated.size(1) >= self.max_seq_len:
                    break

                if past_kvs is None:
                    logits, past_kvs = self.forward(generated, use_cache=True)
                else:
                    logits, past_kvs = self.forward(
                        generated[:, -1:],
                        use_cache=True,
                        past_kvs=past_kvs
                    )

                next_token_logits = logits[:, -1, :]

                for token_id in set(generated[0].tolist()):
                    next_token_logits[0, token_id] /= repetition_penalty

                next_token_logits = next_token_logits / temperature

                if do_sample:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)

                    if top_k > 0:
                        indices_to_remove = sorted_indices[..., top_k:]
                        next_token_logits.scatter_(1, indices_to_remove, float('-inf'))

                    if top_p < 1.0:
                        sorted_logits = F.softmax(sorted_logits, dim=-1)
                        cumulative_probs = torch.cumsum(sorted_logits, dim=-1)

                        sorted_indices_to_remove = cumulative_probs > top_p
                        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                        sorted_indices_to_remove[..., 0] = 0

                        indices_to_remove = sorted_indices[sorted_indices_to_remove]
                        next_token_logits.scatter_(1, indices_to_remove, float('-inf'))

                    probs = F.softmax(next_token_logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                else:
                    next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

                generated = torch.cat([generated, next_token], dim=1)

                if next_token.item() == 2:
                    break

        return generated


def create_grok_model():
    """creates grok-1 style 314B parameter model architecture"""

    return LargeLanguageModel(
        vocab_size=100000,
        d_model=6144,
        num_layers=64,
        num_heads=48,
        num_kv_heads=8,
        d_ff=32768,
        max_seq_len=8192
    )


def create_large_model():
    """creates 70B parameter model"""

    return LargeLanguageModel(
        vocab_size=100000,
        d_model=8192,
        num_layers=80,
        num_heads=64,
        num_kv_heads=8,
        d_ff=28672,
        max_seq_len=8192
    )


def create_medium_model():
    """creates 13B parameter model"""

    return LargeLanguageModel(
        vocab_size=50000,
        d_model=5120,
        num_layers=40,
        num_heads=40,
        num_kv_heads=8,
        d_ff=13824,
        max_seq_len=4096
    )


def create_small_model():
    """creates 7B parameter model"""

    return LargeLanguageModel(
        vocab_size=32000,
        d_model=4096,
        num_layers=32,
        num_heads=32,
        num_kv_heads=8,
        d_ff=11008,
        max_seq_len=4096
    )
