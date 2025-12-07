import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding (same as original)."""
    def __init__(self, d_model, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return x


class GroupedQueryAttention(nn.Module):
    """
    Grouped-Query Attention (GQA) as used in Llama-2, Mistral, Gemma.
    Uses fewer key-value heads than query heads for parameter efficiency.

    Args:
        embed_dim: Embedding dimension
        num_heads: Number of query heads
        num_kv_heads: Number of key-value heads (must divide num_heads)
        dropout: Dropout probability
    """
    def __init__(self, embed_dim=160, num_heads=8, num_kv_heads=2, dropout=0.1):
        super().__init__()
        assert embed_dim % num_heads == 0, f"embed_dim={embed_dim} must be divisible by num_heads={num_heads}"
        assert num_heads % num_kv_heads == 0, f"num_heads={num_heads} must be divisible by num_kv_heads={num_kv_heads}"

        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = embed_dim // num_heads
        self.embed_dim = embed_dim

        # Q has full heads, K/V have fewer heads (shared across Q heads)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(embed_dim, embed_dim, bias=False)

        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.head_dim)

    def forward(self, x, mask=None, get_attention=False):
        """
        Args:
            x: Input tensor [batch, seq_len, embed_dim]
            mask: Attention mask [batch, seq_len] (1 = valid, 0 = masked)
            get_attention: Whether to return attention weights

        Returns:
            output: [batch, seq_len, embed_dim]
            attention_weights: [batch, num_heads, seq_len, seq_len] if get_attention=True
        """
        B, L, D = x.shape

        # Project and reshape
        Q = self.q_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)  # [B, num_heads, L, head_dim]
        K = self.k_proj(x).view(B, L, self.num_kv_heads, self.head_dim).transpose(1, 2)  # [B, num_kv_heads, L, head_dim]
        V = self.v_proj(x).view(B, L, self.num_kv_heads, self.head_dim).transpose(1, 2)  # [B, num_kv_heads, L, head_dim]

        # Repeat K/V to match Q heads
        repeat_factor = self.num_heads // self.num_kv_heads
        K = K.repeat_interleave(repeat_factor, dim=1)  # [B, num_heads, L, head_dim]
        V = V.repeat_interleave(repeat_factor, dim=1)  # [B, num_heads, L, head_dim]

        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale  # [B, num_heads, L, L]

        # Apply mask if provided
        if mask is not None:
            # mask shape: [B, L] -> expand to [B, 1, 1, L]
            mask = mask.unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(mask == 0, float('-inf'))

        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, V)  # [B, num_heads, L, head_dim]

        # Concatenate and project
        out = out.transpose(1, 2).contiguous().view(B, L, D)
        out = self.o_proj(out)

        if get_attention:
            return out, attn
        return out, None


class SwiGLU(nn.Module):
    """
    SwiGLU activation function as used in PaLM, Llama.
    More parameter-efficient than traditional ReLU-based FFN.

    SwiGLU(x) = (Swish(xW_gate) ⊙ xW_up) W_down
    where Swish(x) = x * sigmoid(x) = SiLU(x)
    """
    def __init__(self, embed_dim, dim_feedforward, dropout=0.1):
        super().__init__()
        self.w_gate = nn.Linear(embed_dim, dim_feedforward, bias=False)
        self.w_up = nn.Linear(embed_dim, dim_feedforward, bias=False)
        self.w_down = nn.Linear(dim_feedforward, embed_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        gate = F.silu(self.w_gate(x))  # Swish/SiLU activation
        up = self.w_up(x)
        return self.dropout(self.w_down(gate * up))


class UltraLightweightMemory(nn.Module):
    """
    Ultra-lightweight differentiable memory with low-rank projections.
    Simplified DNC-style memory for working memory functionality.

    Args:
        mem_slots: Number of memory slots
        mem_dim: Dimension of each memory slot
        rank: Rank for low-rank read/write projections
    """
    def __init__(self, mem_slots=16, mem_dim=160, rank=16):
        super().__init__()
        self.mem_slots = mem_slots
        self.mem_dim = mem_dim
        self.rank = rank

        # Global memory bank (buffer, not a trainable parameter)
        self.register_buffer('memory', torch.zeros(1, mem_slots, mem_dim))

        # Low-rank read/write projections (key innovation for parameter efficiency)
        self.read_key = nn.Linear(mem_dim, rank, bias=False)
        self.read_value = nn.Linear(rank, mem_dim, bias=False)
        self.write_key = nn.Linear(mem_dim, rank, bias=False)
        self.write_erase = nn.Linear(rank, mem_dim, bias=False)
        self.write_add = nn.Linear(rank, mem_dim, bias=False)

    def read(self, query):
        """
        Read from memory using attention mechanism.

        Args:
            query: [batch, seq_len, mem_dim]

        Returns:
            content: [batch, seq_len, mem_dim]
        """
        # Low-rank projection
        q_proj = self.read_key(query)  # [B, L, rank]

        # Expand memory to batch size
        bs = query.size(0)
        if self.memory.size(0) != bs:
            memory = self.memory.repeat(bs, 1, 1)
        else:
            memory = self.memory

        # Attention-based read (using only low-rank dimensions for efficiency)
        mem_proj = memory[..., :self.rank]  # [B, mem_slots, rank]
        scores = torch.bmm(q_proj, mem_proj.transpose(1, 2))  # [B, L, mem_slots]
        attn = F.softmax(scores / math.sqrt(self.rank), dim=-1)
        content = torch.bmm(attn, memory)  # [B, L, mem_dim]

        # Project back through read_value
        return self.read_value(content[..., :self.rank])

    def write(self, interface):
        """
        Write to memory using erase-then-add mechanism (DNC-style).

        Args:
            interface: [batch, seq_len, mem_dim]
        """
        # Low-rank projection
        i_proj = self.write_key(interface.mean(dim=1))  # [B, rank] - aggregate over sequence

        # Generate erase and add vectors
        erase = torch.sigmoid(self.write_erase(i_proj))  # [B, mem_dim]
        add = self.write_add(i_proj)  # [B, mem_dim]

        # Expand to memory dimensions
        erase = erase.unsqueeze(1)  # [B, 1, mem_dim]
        add = add.unsqueeze(1)  # [B, 1, mem_dim]

        # Update memory: erase then add
        bs = interface.size(0)
        if self.memory.size(0) != bs:
            self.memory = self.memory.repeat(bs, 1, 1)

        self.memory = self.memory * (1 - erase) + add

        # Normalize to keep stable
        self.memory = F.layer_norm(self.memory, (self.mem_dim,))


class SharedReasoningModule(nn.Module):
    """
    Shared reasoning buffer for explicit multi-step reasoning (Chain-of-Thought).
    All layers can read from and write to this shared scratchpad.

    Args:
        reasoning_tokens: Number of reasoning buffer tokens
        embed_dim: Embedding dimension
    """
    def __init__(self, reasoning_tokens=64, embed_dim=160):
        super().__init__()
        self.reasoning_tokens = reasoning_tokens
        self.embed_dim = embed_dim

        # Reasoning state buffer (shared across all layers)
        self.reasoning_embeddings = nn.Parameter(torch.randn(reasoning_tokens, embed_dim) * 0.02)

        # Low-rank projections for layer interaction
        self.project_to_reasoning = nn.Linear(embed_dim, 64, bias=False)
        self.project_from_reasoning = nn.Linear(64, embed_dim, bias=False)

    def refine(self, layer_output, layer_idx):
        """
        Refine reasoning state based on layer output.

        Args:
            layer_output: [batch, seq_len, embed_dim]
            layer_idx: Current layer index

        Returns:
            reasoning_signal: [batch, 1, embed_dim] - signal to add to layer output
        """
        # Extract reasoning query (aggregate over sequence)
        reasoning_query = self.project_to_reasoning(layer_output.mean(dim=1))  # [B, 64]

        # Attend to reasoning buffer
        scores = torch.matmul(reasoning_query, self.reasoning_embeddings[:, :64].T)  # [B, reasoning_tokens]
        attn = F.softmax(scores, dim=-1)

        # Read from buffer
        reasoning_content = torch.matmul(attn, self.reasoning_embeddings)  # [B, embed_dim]

        # Update reasoning buffer with exponential moving average
        with torch.no_grad():
            update = layer_output.mean(dim=1).mean(dim=0)  # [embed_dim] - average over batch and sequence
            self.reasoning_embeddings.data = 0.9 * self.reasoning_embeddings.data + 0.1 * update.unsqueeze(0)

        # Project back to layer space
        reasoning_signal = self.project_from_reasoning(reasoning_content[:, :64])
        return reasoning_signal.unsqueeze(1)  # [B, 1, embed_dim]


class MultiSpeedLayer(nn.Module):
    """
    Multi-speed learning layer with dynamic routing between fast and slow paths.
    Implements dual-process cognition (System 1/System 2).

    Args:
        embed_dim: Embedding dimension
        dim_feedforward: Feed-forward hidden dimension
        dropout: Dropout probability
    """
    def __init__(self, embed_dim=160, dim_feedforward=320, dropout=0.1):
        super().__init__()
        # Slow path: Full SwiGLU FFN transformation
        self.slow_ffn = SwiGLU(embed_dim, dim_feedforward, dropout)

        # Fast path: Direct connection (parameter-free, instant adaptation)
        # No additional parameters needed - uses identity

        # Dynamic routing gate (learned per-dimension)
        self.gate = nn.Parameter(torch.zeros(embed_dim))

    def forward(self, x):
        """
        Args:
            x: [batch, seq_len, embed_dim]

        Returns:
            output: [batch, seq_len, embed_dim]
        """
        # Slow path: Deep transformation
        slow_out = self.slow_ffn(x)

        # Fast path: Identity (can be extended with lightweight transformation)
        fast_out = x

        # Dynamic routing (learned alpha per dimension)
        alpha = torch.sigmoid(self.gate)  # [embed_dim]
        return alpha * slow_out + (1 - alpha) * fast_out


class EfficientTransformerBlock(nn.Module):
    """
    Efficient transformer block combining all cognitive mechanisms:
    - Grouped-Query Attention (parameter-efficient multi-head attention)
    - Multi-Speed FFN (fast/slow dual-process learning)
    - Working Memory Interface (low-rank memory read/write)
    - Reasoning Refinement (shared reasoning buffer)

    Args:
        d_model: Model dimension (embed_dim)
        num_heads: Number of query heads
        num_kv_heads: Number of key-value heads (GQA)
        dim_feedforward: FFN hidden dimension
        dropout: Dropout probability
        shared_memory: Shared UltraLightweightMemory instance
        shared_reasoning: Shared SharedReasoningModule instance
        layer_idx: Layer index (for reasoning refinement schedule)
    """
    def __init__(self, d_model=160, num_heads=8, num_kv_heads=2,
                 dim_feedforward=320, dropout=0.1,
                 shared_memory=None, shared_reasoning=None, layer_idx=0):
        super().__init__()
        self.layer_idx = layer_idx
        self.d_model = d_model

        # 1. Grouped-Query Attention
        self.norm1 = nn.LayerNorm(d_model)
        self.attention = GroupedQueryAttention(d_model, num_heads, num_kv_heads, dropout)
        self.dropout1 = nn.Dropout(dropout)

        # 2. Multi-Speed FFN
        self.norm2 = nn.LayerNorm(d_model)
        self.multi_speed = MultiSpeedLayer(d_model, dim_feedforward, dropout)
        self.dropout2 = nn.Dropout(dropout)

        # 3. Working Memory (shared instance)
        self.memory = shared_memory

        # 4. Reasoning Module (shared instance)
        self.reasoning = shared_reasoning

    def forward(self, x, src_mask=None, get_attention=False):
        """
        Args:
            x: [batch, seq_len, d_model]
            src_mask: Attention mask [batch, seq_len]
            get_attention: Whether to return attention weights

        Returns:
            output: [batch, seq_len, d_model]
            attention_weights: Attention weights if get_attention=True
        """
        # Use mask for compatibility
        mask = src_mask
        # Pre-LN architecture (more stable for small models)

        # Attention block
        residual = x
        x_norm = self.norm1(x)
        attn_out, attn_weights = self.attention(x_norm, mask, get_attention)
        x = residual + self.dropout1(attn_out)

        # FFN block (with multi-speed routing)
        residual = x
        x_norm = self.norm2(x)
        x = residual + self.dropout2(self.multi_speed(x_norm))

        # Memory interaction (read + write)
        if self.memory is not None:
            mem_content = self.memory.read(x)
            x = x + 0.1 * mem_content  # Residual memory contribution (scaled)
            self.memory.write(x)

        # Reasoning refinement (every 3rd layer: 2, 5, 8)
        if self.reasoning is not None and (self.layer_idx + 1) % 3 == 0:
            reasoning_signal = self.reasoning.refine(x, self.layer_idx)
            x = x + 0.1 * reasoning_signal  # Residual reasoning contribution (scaled)

        return x, attn_weights


class CrossAttention(nn.Module):
    """
    Cross-attention module for decoder using Grouped-Query Attention.
    Query comes from decoder, Key/Value from encoder.
    """
    def __init__(self, d_model=160, num_heads=8, num_kv_heads=2, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = d_model // num_heads
        self.d_model = d_model

        # Q from decoder, K/V from encoder
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(d_model, num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.head_dim)

    def forward(self, tgt, memory, memory_mask=None):
        """
        Args:
            tgt: Target (decoder query) [batch, tgt_len, d_model]
            memory: Encoder output [batch, src_len, d_model]
            memory_mask: Source mask [batch, src_len]

        Returns:
            output: [batch, tgt_len, d_model]
        """
        B, tgt_len, D = tgt.shape
        _, src_len, _ = memory.shape

        # Q from decoder, K/V from encoder
        Q = self.q_proj(tgt).view(B, tgt_len, self.num_heads, self.head_dim).transpose(1, 2)  # [B, num_heads, tgt_len, head_dim]
        K = self.k_proj(memory).view(B, src_len, self.num_kv_heads, self.head_dim).transpose(1, 2)  # [B, num_kv_heads, src_len, head_dim]
        V = self.v_proj(memory).view(B, src_len, self.num_kv_heads, self.head_dim).transpose(1, 2)  # [B, num_kv_heads, src_len, head_dim]

        # Repeat K/V to match Q heads
        repeat_factor = self.num_heads // self.num_kv_heads
        K = K.repeat_interleave(repeat_factor, dim=1)  # [B, num_heads, src_len, head_dim]
        V = V.repeat_interleave(repeat_factor, dim=1)  # [B, num_heads, src_len, head_dim]

        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale  # [B, num_heads, tgt_len, src_len]

        # Apply mask if provided
        if memory_mask is not None:
            # memory_mask shape: [B, src_len] -> expand to [B, 1, 1, src_len]
            mask = memory_mask.unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(mask == 0, float('-inf'))

        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, V)  # [B, num_heads, tgt_len, head_dim]

        # Concatenate and project
        out = out.transpose(1, 2).contiguous().view(B, tgt_len, D)
        out = self.o_proj(out)

        # Residual + norm
        tgt = tgt + self.dropout(out)
        tgt = self.norm(tgt)
        return tgt


# LoRA components (kept from original for fine-tuning support)
class LoRALinear(nn.Module):
    """
    Low-Rank Adaptation (LoRA) layer.
    Wraps a frozen linear layer and adds a trainable low-rank adapter.
    """
    def __init__(self, linear_layer, rank=8, alpha=16):
        super().__init__()
        self.linear = linear_layer
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        # Freeze the original linear layer
        for param in self.linear.parameters():
            param.requires_grad = False

        # LoRA weights
        self.lora_A = nn.Parameter(torch.zeros(self.linear.in_features, rank))
        self.lora_B = nn.Parameter(torch.zeros(rank, self.linear.out_features))

        self.reset_parameters()

    def reset_parameters(self):
        # Initialize A with Kaiming uniform, B with zeros
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x):
        # Original output (frozen)
        out = self.linear(x)

        # LoRA path (trainable)
        lora_out = (x @ self.lora_A @ self.lora_B) * self.scaling

        return out + lora_out

    def merge(self):
        """Merges LoRA weights into the original linear layer (for inference)."""
        if isinstance(self.linear, nn.Linear):
            update = (self.lora_A @ self.lora_B).t() * self.scaling
            self.linear.weight.data += update
            return self.linear
        return self


def apply_lora(model, rank=8, alpha=16, target_modules=['decoder_projection', 'slow_ffn']):
    """
    Replaces specified Linear layers with LoRALinear layers.
    """
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            if any(t in name for t in target_modules):
                parent_name = name.rsplit('.', 1)[0] if '.' in name else ''
                child_name = name.rsplit('.', 1)[1] if '.' in name else name

                if parent_name:
                    parent = model.get_submodule(parent_name)
                else:
                    parent = model

                print(f"Applying LoRA to {name}...")
                lora_layer = LoRALinear(module, rank, alpha)
                setattr(parent, child_name, lora_layer)
    return model
