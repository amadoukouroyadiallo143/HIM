import torch
import torch.nn as nn
import math
from src.components import (
    PositionalEncoding,
    EfficientTransformerBlock,
    CrossAttention,
    UltraLightweightMemory,
    SharedReasoningModule
)

class NLDirectResponse(nn.Module):
    """
    Optimized NL-DirectResponse model for 7M parameters with cognitive mechanisms.

    Architecture:
        - Vocabulary: 32,000 tokens
        - Embedding: 160 dimensions
        - Encoder: 10 EfficientTransformerBlocks
        - Attention: Grouped-Query Attention (8 Q heads, 2 KV heads)
        - FFN: SwiGLU with multi-speed routing
        - Memory: UltraLightweightMemory (16 slots, low-rank)
        - Reasoning: SharedReasoningModule (64 tokens)
        - Decoder: Cross-attention + learnable query

    Total parameters: ~7.06M
    """
    def __init__(self, vocab_size, embed_dim, num_heads, num_kv_heads, num_encoder_layers,
                 dim_feedforward, max_answer_len, dropout=0.1, mem_slots=16, mem_rank=16,
                 reasoning_tokens=64):
        super().__init__()
        self.embed_dim = embed_dim
        self.max_answer_len = max_answer_len

        # 1. Input Embedding Layer
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_encoder = PositionalEncoding(embed_dim)

        # 2. Shared Cognitive Components
        # Ultra-lightweight memory (shared across all layers)
        self.memory = UltraLightweightMemory(mem_slots=mem_slots, mem_dim=embed_dim, rank=mem_rank)

        # Shared reasoning module (for chain-of-thought)
        self.reasoning_module = SharedReasoningModule(reasoning_tokens=reasoning_tokens, embed_dim=embed_dim)

        # 3. Efficient Transformer Encoder (10 layers)
        self.transformer_encoder = nn.ModuleList([
            EfficientTransformerBlock(
                d_model=embed_dim,
                num_heads=num_heads,
                num_kv_heads=num_kv_heads,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                shared_memory=self.memory,
                shared_reasoning=self.reasoning_module,
                layer_idx=i
            )
            for i in range(num_encoder_layers)
        ])

        # 4. Direct Response Decoder
        # Learnable query for the decoder
        self.decoder_query = nn.Parameter(torch.randn(1, max_answer_len, embed_dim) * 0.02)
        self.decoder_cross_attention = CrossAttention(embed_dim, num_heads, num_kv_heads, dropout)
        self.decoder_projection = nn.Linear(embed_dim, vocab_size)

        # 5. Weight Tying (reduces parameters significantly)
        # Ties the weights of the embedding layer and the output projection layer
        self.tie_weights()

    def tie_weights(self):
        """
        Ties the weights of the input embedding and the output projection.
        This is a standard practice in many Transformer models (GPT, T5, etc.)
        to reduce parameters and improve regularization.
        """
        self.decoder_projection.weight = self.embedding.weight

    def forward(self, input_ids, src_padding_mask=None, get_attention=False):
        """
        Forward pass through the model.

        Args:
            input_ids: [batch_size, sequence_length]
            src_padding_mask: [batch_size, sequence_length] (1 = valid, 0 = masked)
            get_attention: Whether to return attention weights

        Returns:
            logits: [batch_size, max_answer_len, vocab_size]
            attention_weights: List of attention weights if get_attention=True
        """
        # 1. Input Embedding and Positional Encoding
        embedded_input = self.embedding(input_ids) * math.sqrt(self.embed_dim)
        embedded_input = self.pos_encoder(embedded_input)

        # 2. Process through Transformer Encoder
        encoder_output = embedded_input
        attention_weights = []

        for encoder_block in self.transformer_encoder:
            encoder_output, attn = encoder_block(encoder_output, src_mask=src_padding_mask, get_attention=get_attention)
            if get_attention and attn is not None:
                attention_weights.append(attn)

        # 3. Direct Response Decoder
        # Expand the decoder query for the batch
        batch_size = input_ids.size(0)
        decoder_input = self.decoder_query.repeat(batch_size, 1, 1)

        # Apply cross-attention
        decoder_output = self.decoder_cross_attention(decoder_input, encoder_output, memory_mask=src_padding_mask)

        # Project to vocab size
        logits = self.decoder_projection(decoder_output)  # [batch_size, max_answer_len, vocab_size]

        if get_attention:
            return logits, attention_weights
        return logits

    def reset_memory(self):
        """
        Resets the differentiable memory to zeros and detaches from graph.
        Should be called between batches to prevent cross-batch information leakage.
        """
        with torch.no_grad():
            self.memory.memory = torch.zeros(
                1, self.memory.mem_slots, self.memory.mem_dim
            ).to(self.memory.memory.device).detach()

    def forward_reconstruction(self, input_ids, src_padding_mask=None):
        """
        Forward pass for self-supervised reconstruction (MAML adaptation step).
        Returns logits for the input sequence (useful for meta-learning).

        Args:
            input_ids: [batch_size, sequence_length]
            src_padding_mask: [batch_size, sequence_length]

        Returns:
            logits: [batch_size, sequence_length, vocab_size]
        """
        # 1. Embedding
        embedded_input = self.embedding(input_ids) * math.sqrt(self.embed_dim)
        embedded_input = self.pos_encoder(embedded_input)

        # 2. Encoder
        encoder_output = embedded_input
        for encoder_block in self.transformer_encoder:
            encoder_output, _ = encoder_block(encoder_output, src_mask=src_padding_mask)

        # 3. Project encoder output directly to vocab (Reconstruction)
        # Re-use decoder projection layer as they share the embedding space
        logits = self.decoder_projection(encoder_output)
        return logits

    def get_num_params(self):
        """
        Count the number of trainable parameters.

        Returns:
            total_params: Total number of trainable parameters
            param_breakdown: Dictionary with parameter counts per component
        """
        total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        param_breakdown = {
            'embedding': sum(p.numel() for p in self.embedding.parameters()),
            'pos_encoding': 0,  # Buffer only, not trainable
            'memory': sum(p.numel() for p in self.memory.parameters()),
            'reasoning': sum(p.numel() for p in self.reasoning_module.parameters()),
            'encoder_layers': sum(p.numel() for block in self.transformer_encoder for p in block.parameters()),
            'decoder_query': self.decoder_query.numel(),
            'decoder_cross_attn': sum(p.numel() for p in self.decoder_cross_attention.parameters()),
            'decoder_projection': 0,  # Weight-tied with embedding
        }

        return total_params, param_breakdown

    def print_architecture_summary(self):
        """
        Print a detailed summary of the architecture and parameter counts.
        """
        total_params, breakdown = self.get_num_params()

        print("=" * 80)
        print("MODEL ARCHITECTURE SUMMARY")
        print("=" * 80)
        print(f"{'Component':<40} {'Parameters':>15} {'%':>10}")
        print("-" * 80)

        for name, count in breakdown.items():
            percentage = (count / total_params * 100) if total_params > 0 else 0
            print(f"{name:<40} {count:>15,} {percentage:>9.2f}%")

        print("-" * 80)
        print(f"{'TOTAL':<40} {total_params:>15,} {'100.00%':>10}")
        print("=" * 80)
        print(f"\nTarget: 7,000,000 parameters")
        print(f"Actual: {total_params:,} parameters")
        if total_params <= 7_000_000:
            diff = 7_000_000 - total_params
            print(f"✅ Within budget! ({diff:,} parameters remaining)")
        else:
            diff = total_params - 7_000_000
            print(f"⚠️  Over budget by {diff:,} parameters ({diff/7_000_000*100:.2f}%)")
        print("=" * 80)
