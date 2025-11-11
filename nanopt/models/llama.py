"""
Implementation of Llama 3.2 1B Model."
"""
from dataclasses import dataclass
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from torch.onnx.ops import rotary_embedding


@dataclass
class LlamaConfig:
    vocab_size: int = 128256
    hidden_size: int = 2048
    intermediate_size: int = 8192
    num_hidden_layers: int = 16
    num_attention_heads: int = 32
    num_key_value_heads: int = 8 # GQA
    max_position_embeddings: int = 131072 # 128k context
    rms_norm_eps: float = 1e-5
    rope_theta: float = 500_000.0
    attention_dropout: float = 0.0
    hidden_dropout: float = 0.0
    initializer_range: float = 0.02
    use_cache: bool = False
    pad_token_id: Optional[int] = None
    bos_token_id: int = 128_000
    eos_token_id: int = 128_001

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return self.weight * x

class RotaryEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE)."""
    def __init__(self,
                dim: int,
                max_position_embeddings: int = 2048,
                base: float = 10_000.0):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base

        # precompute frequencies
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float() / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, x: torch.Tensor, seq_len: int) -> tuple[torch.Tensor, torch.Tensor]:
        # Generate position indices
        t = torch.arange(seq_len, device=x.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos(), emb.sin()

def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary position embedding to q and k."""
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

class GroupedQueryAttention(nn.Module):
    """Grouped Query Attention (GQA)"""
    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads

        # Projections
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias = False)

        self.rotary_emb = RotaryEmbedding(
            self.head_dim,
            max_position_embeddings=config.max_position_embeddings,
            base=config.rope_theta,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None ) -> torch.Tensor:

        batch_size, seq_length, _ = hidden_states.shape

        # project to Q, K, V
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        # reshape for multi-head attention
        query_states = query_states.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(batch_size, seq_length, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(batch_size, seq_length, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        # apply rotary position embedding
        cos, sin = self.rotary_emb(value_states, seq_length)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        # repeat k/v heads for GQA
        key_states = key_states.repeat_interleave(self.num_key_value_groups, dim=1)
        value_states = value_states.repeat_interleave(self.num_key_value_groups, dim=1)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
        
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_output = torch.matmul(attn_weights, value_states)

        # Reshape and project
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(batch_size, seq_length, self.hidden_size)
        attn_output = self.o_proj(attn_output)

        return attn_output

class MLP(nn.Module):
    """SwiGLU MLP"""
    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))

class DecoderLayer(nn.Module):
    """Llama decoder layer with pre-norm"""
    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.self_attn = GroupedQueryAttention(config)
        self.mlp = MLP(config)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states, attention_mask)
        hidden_states = residual + hidden_states

        # MLP with pre-norm
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states

class LlamaModel(nn.Module):
    """Llama 3.2 1B Model"""
    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([DecoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        """Initialize weights"""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        hidden_states = self.embed_tokens(input_ids)

        # create mask if not provided
        if attention_mask is None:
            batch_size, seq_length = input_ids.shape
            attention_mask = torch.triu(
                torch.full((seq_length, seq_length), float("-inf"), device=input_ids.device),
                diagonal=1,
            )
        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask)

        hidden_states = self.norm(hidden_states)
        return hidden_states

class LlamaForCausalLM(nn.Module):
    """Slapping a language modeling head on top to turn the hidden states into logits for language modeling."""
    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.config = config
        self.model = LlamaModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Tie weights
        self.lm_head.weight = self.model.embed_tokens.weight

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> dict:
        hidden_states = self.model(input_ids, attention_mask=attention_mask)
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1: ].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100, # ignore padding tokens
            )
        return {"loss": loss, "logits": logits}

    def num_parameters(self, exclude_embeddings: bool = False) -> int:
        """Count number of parameters in the model."""
        n_params = sum(p.numel() for p in self.parameters())
        if exclude_embeddings:
            n_params -= self.model.embed_tokens.weight.numel()
        return n_params
    
    @torch.inference_mode()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 1.0,
        repetition_penalty: float = 1.0,
        do_sample: bool = True,
        **kwargs,
    ) -> torch.Tensor:
        """Generate text using greedy search or sampling."""
        batch_size, seq_length = input_ids.shape
        bos_token_id = input_ids[0, 0].item()

        # Prepare attention mask
        attention_mask = torch.ones((batch_size, seq_length), device=input_ids.device, dtype=torch.bool)
        for i in range(max_new_tokens):
            # Get logits
            outputs = self(input_ids, attention_mask=attention_mask)
            logits = outputs["logits"][:, -1, :]

            # Apply temperature and top-k/p filtering
            if temperature != 1.0:
                probs = torch.softmax(logits / temperature, dim=-1)
            else:
                probs = logits
            if top_k > 0:
                probs_topk, _ = torch.topk(probs, top_k)
                probs = torch.where(probs < probs_topk[..., -1], torch.zeros_like(probs), probs)
            if top_p < 1.0:
                probs_sorted, _ = torch.sort(probs, dim=-1, descending=True)
                cumsum = torch.cumsum(probs_sorted, dim=-1)
                mask = cumsum > top_p
                mask[..., 1:] = mask[..., :-1].clone()
                probs = torch.where(mask, torch.zeros_like(probs), probs)
            if repetition_penalty != 1.0:
                # create a mask of unique tokens seen in this position and all previous positions
                _, _, v = torch.unique_consecutive(input_ids, sorted=True, return_inverse=True, return_counts=True)
                # create a mask of tokens to apply penalty to
                penalty_mask = v >= 1
                # compute the penalty factor, 1.0 for penalty <= 1.0, > 1.0 if there are duplicates
                penalty = torch.where(penalty_mask, repetition_penalty, 1.0)
                probs = probs * penalty
            if do_sample:
                # Sample 2 new tokens
                probs = probs / repetition_penalty
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                # Greedy search
                next_token = torch.argmax(probs, dim=-1, keepdim=True)

            # Concatenate the selected token
            input_ids = torch.cat((input_ids, next_token), dim=-1)
            attention_mask = torch.cat((attention_mask, torch.ones((batch_size, 1), device=input_ids.device, dtype=torch.bool)), dim=-1)

        return input_ids

    def decode(self, input_ids: torch.Tensor, tokenizer) -> str:
        """Decode the generated tokens into a string."""
        return tokenizer.decode(input_ids[0].tolist(), skip_special_tokens=True)

if __name__ == "__main__":
    from transformers import AutoTokenizer
    config = LlamaConfig()
    model = LlamaForCausalLM(config)
    print(model.num_parameters())
    print(model.num_parameters(exclude_embeddings=True))
    generated = model.generate(torch.tensor([[config.bos_token_id]]))
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
    print(f"Generated: {model.decode(generated, tokenizer)}")
