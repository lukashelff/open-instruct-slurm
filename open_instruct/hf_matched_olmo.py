import flash_attn
import torch
import torch.nn as nn
import torch.nn.functional as F

from open_instruct import logger_utils

logger = logger_utils.setup_logger(__name__)


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin):
    q_type, k_type = q.dtype, k.dtype
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed.to(q_type), k_embed.to(k_type)


class HFMatchedRMSNorm(nn.Module):
    def __init__(self, size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return (self.weight * hidden_states).to(input_dtype)


class HFMatchedRotaryEmbedding(nn.Module):
    def __init__(self, head_dim, rope_theta):
        super().__init__()
        inv_freq = 1.0 / (rope_theta ** (torch.arange(0, head_dim, 2, dtype=torch.int64).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    @torch.no_grad()
    def forward(self, x, position_ids):
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)
        position_ids_expanded = position_ids[:, None, :].float()
        device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()
        return cos, sin


class HFMatchedAttention(nn.Module):
    def __init__(self, d_model, n_heads, n_kv_heads, head_dim, eps):
        super().__init__()
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        self.scaling = head_dim**-0.5
        self.w_q = nn.Linear(d_model, n_heads * head_dim, bias=False)
        self.w_k = nn.Linear(d_model, n_kv_heads * head_dim, bias=False)
        self.w_v = nn.Linear(d_model, n_kv_heads * head_dim, bias=False)
        self.w_out = nn.Linear(n_heads * head_dim, d_model, bias=False)
        self.q_norm = HFMatchedRMSNorm(n_heads * head_dim, eps)
        self.k_norm = HFMatchedRMSNorm(n_kv_heads * head_dim, eps)

    def forward(self, hidden_states, position_embeddings):
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_norm(self.w_q(hidden_states))
        key_states = self.k_norm(self.w_k(hidden_states))
        value_states = self.w_v(hidden_states)

        query_states = query_states.view(hidden_shape).transpose(1, 2)
        key_states = key_states.view(hidden_shape).transpose(1, 2)
        value_states = value_states.view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        attn_output = flash_attn.flash_attn_func(
            query_states, key_states, value_states, causal=True, softmax_scale=self.scaling
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        return self.w_out(attn_output)


class HFMatchedFeedForward(nn.Module):
    def __init__(self, d_model, intermediate_size):
        super().__init__()
        self.w1 = nn.Linear(d_model, intermediate_size, bias=False)
        self.w2 = nn.Linear(intermediate_size, d_model, bias=False)
        self.w3 = nn.Linear(d_model, intermediate_size, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class HFMatchedBlock(nn.Module):
    def __init__(self, d_model, n_heads, n_kv_heads, head_dim, intermediate_size, eps):
        super().__init__()
        self.attention = HFMatchedAttention(d_model, n_heads, n_kv_heads, head_dim, eps)
        self.attention_norm = HFMatchedRMSNorm(d_model, eps)
        self.feed_forward = HFMatchedFeedForward(d_model, intermediate_size)
        self.feed_forward_norm = HFMatchedRMSNorm(d_model, eps)

    def forward(self, hidden_states, position_embeddings):
        residual = hidden_states
        hidden_states = self.attention(hidden_states, position_embeddings)
        hidden_states = self.attention_norm(hidden_states)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.feed_forward(hidden_states)
        hidden_states = self.feed_forward_norm(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class HFMatchedLMHead(nn.Module):
    def __init__(self, d_model, vocab_size, eps):
        super().__init__()
        self.norm = HFMatchedRMSNorm(d_model, eps)
        self.w_out = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, hidden_states):
        return self.w_out(self.norm(hidden_states))


class HFMatchedOlmo2(nn.Module):
    def __init__(
        self, d_model, vocab_size, n_layers, n_heads, n_kv_heads, intermediate_size, rope_theta, eps, head_dim
    ):
        super().__init__()
        self.embeddings = nn.Embedding(vocab_size, d_model)
        self.blocks = nn.ModuleDict(
            {
                str(i): HFMatchedBlock(d_model, n_heads, n_kv_heads, head_dim, intermediate_size, eps)
                for i in range(n_layers)
            }
        )
        self.lm_head = HFMatchedLMHead(d_model, vocab_size, eps)
        self.rotary_emb = HFMatchedRotaryEmbedding(head_dim, rope_theta)

    def forward(self, input_ids):
        hidden_states = self.embeddings(input_ids)
        position_ids = torch.arange(input_ids.shape[1], device=input_ids.device).unsqueeze(0)
        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        for block in self.blocks.values():
            hidden_states = block(hidden_states, position_embeddings)
        return self.lm_head(hidden_states)

    @classmethod
    def from_hf_config(cls, hf_config):
        head_dim = getattr(hf_config, "head_dim", hf_config.hidden_size // hf_config.num_attention_heads)
        return cls(
            d_model=hf_config.hidden_size,
            vocab_size=hf_config.vocab_size,
            n_layers=hf_config.num_hidden_layers,
            n_heads=hf_config.num_attention_heads,
            n_kv_heads=hf_config.num_key_value_heads,
            intermediate_size=hf_config.intermediate_size,
            rope_theta=hf_config.rope_theta,
            eps=hf_config.rms_norm_eps,
            head_dim=head_dim,
        )

    @staticmethod
    def convert_hf_state_dict(hf_state_dict):
        converted = {}
        for key, value in hf_state_dict.items():
            new_key = key
            new_key = new_key.replace("model.embed_tokens.", "embeddings.")
            new_key = new_key.replace("model.layers.", "blocks.")
            new_key = new_key.replace(".self_attn.q_proj.", ".attention.w_q.")
            new_key = new_key.replace(".self_attn.k_proj.", ".attention.w_k.")
            new_key = new_key.replace(".self_attn.v_proj.", ".attention.w_v.")
            new_key = new_key.replace(".self_attn.o_proj.", ".attention.w_out.")
            new_key = new_key.replace(".self_attn.q_norm.", ".attention.q_norm.")
            new_key = new_key.replace(".self_attn.k_norm.", ".attention.k_norm.")
            new_key = new_key.replace(".post_attention_layernorm.", ".attention_norm.")
            new_key = new_key.replace(".mlp.gate_proj.", ".feed_forward.w1.")
            new_key = new_key.replace(".mlp.down_proj.", ".feed_forward.w2.")
            new_key = new_key.replace(".mlp.up_proj.", ".feed_forward.w3.")
            new_key = new_key.replace(".post_feedforward_layernorm.", ".feed_forward_norm.")
            new_key = new_key.replace("model.norm.", "lm_head.norm.")
            if new_key == "lm_head.weight":
                new_key = "lm_head.w_out.weight"
            converted[new_key] = value
        return converted
