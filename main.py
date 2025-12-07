import torch
import torch.nn as nn
from transformers import AutoTokenizer
import os
import json
import argparse
import inspect

from src.model import NLDirectResponse
from src.dataset import get_dataloader
from src.train import train_model_advanced
from src.optimizer import Lion

def count_parameters(model):
    """
    Counts the number of trainable parameters in a PyTorch model.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def main(args):
    # 1. Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 2. Tokenizer
    custom_tokenizer_path = "model_artifacts/tokenizer"
    if not (os.path.exists(custom_tokenizer_path) and os.path.exists(os.path.join(custom_tokenizer_path, "tokenizer.json"))):
        print(f"Error: Custom tokenizer not found at '{custom_tokenizer_path}'.")
        print("Please train a custom tokenizer by running: python train_tokenizer.py")
        return

    print(f"Loading custom tokenizer from '{custom_tokenizer_path}'...")
    tokenizer = AutoTokenizer.from_pretrained(custom_tokenizer_path)
    
    # 3. Task-specific settings
    if args.task_type == 'wikitext':
        max_seq_len = 512 # Input sequence length
        max_answer_len = 64 # Reduced for efficiency (7M param model)
    else:
        raise ValueError(f"Unknown task_type: {args.task_type}")
    
    # Special tokens for WikiText are already handled by the tokenizer (<s>, </s>, etc.)
    # No need to add instruction-tuning tokens for this phase.

    # 4. Model Configuration
    # This dictionary holds all config, some for the model, some for the pipeline
    model_config = {
        "vocab_size": len(tokenizer), # Use full vocab size including special tokens
        "embed_dim": args.embed_dim,
        "num_heads": args.num_heads,
        "num_kv_heads": args.num_kv_heads,  # NEW: For Grouped-Query Attention
        "num_encoder_layers": args.num_encoder_layers,
        "dim_feedforward": args.dim_feedforward,
        "dropout": args.dropout,
        "max_answer_len": max_answer_len,
        "mem_slots": args.mem_slots,  # NEW: Memory slots
        "mem_rank": args.mem_rank,  # NEW: Low-rank memory projections
        "reasoning_tokens": args.reasoning_tokens,  # NEW: Reasoning buffer tokens
        "tokenizer_name": tokenizer.name_or_path,
        "max_seq_len": max_seq_len,
        "task_type": args.task_type,
    }

    # 5. Dataloaders
    train_dataloader = get_dataloader(
        dataset_type=args.task_type,
        tokenizer=tokenizer,
        max_seq_len=max_seq_len,
        max_answer_len=max_answer_len,
        batch_size=args.batch_size,
        split='train',
        cache_dir='cached_datasets'
    )
    val_dataloader = get_dataloader(
        dataset_type=args.task_type,
        tokenizer=tokenizer,
        max_seq_len=max_seq_len,
        max_answer_len=max_answer_len,
        batch_size=args.batch_size,
        split='validation',
        shuffle=False,
        cache_dir='cached_datasets'
    )

    # 6. Model, Optimizer, Criterion
    # Filter model_config to only include arguments expected by the model's __init__
    model_init_signature = inspect.signature(NLDirectResponse.__init__)
    model_args = {k: v for k, v in model_config.items() if k in model_init_signature.parameters}
    model = NLDirectResponse(**model_args)
    
    print(f"The model has {count_parameters(model):,} trainable parameters.")

    # Print detailed architecture summary
    model.print_architecture_summary()

    # Validate parameter count
    if count_parameters(model) > 7_000_000:
        print(f"\n⚠️  WARNING: Model exceeds 7M parameter budget!")
        print(f"   Current: {count_parameters(model):,} parameters")
        print(f"   Target:  7,000,000 parameters")
        print(f"   Excess:  {count_parameters(model) - 7_000_000:,} parameters")

    # 6. Train the model with ADVANCED training
    # Optimizer and criterion are created inside train_model_advanced for multi-speed learning
    print("\n" + "="*80)
    print("TRAINING CONFIGURATION")
    print("="*80)
    print(f"  Base Learning Rate: {args.learning_rate}")
    print(f"  Epochs: {args.num_epochs}")
    print(f"  Batch Size: {args.batch_size}")
    print(f"  Gradient Accumulation: {args.accumulation_steps}")
    print(f"  Mixed Precision (AMP): {args.fp16}")
    print(f"  Smart Training: {args.smart_training}")
    print(f"  Label Smoothing: {args.label_smoothing}")
    print(f"  Cognitive Losses: {args.use_cognitive_losses}")
    print("="*80 + "\n")

    train_model_advanced(
        model,
        train_dataloader,
        val_dataloader,
        tokenizer,
        device,
        num_epochs=args.num_epochs,
        model_config=model_config,
        base_lr=args.learning_rate,
        grad_clip_value=args.grad_clip_value,
        pruning_amount=args.pruning_amount,
        fine_tune_epochs=args.fine_tune_epochs,
        smart_training=args.smart_training,
        skip_threshold=args.skip_threshold,
        save_steps=args.save_steps,
        accumulation_steps=args.accumulation_steps,
        use_amp=args.fp16,
        use_cognitive_losses=args.use_cognitive_losses,
        label_smoothing=args.label_smoothing
    )

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train a Nested Learning Transformer model.")
    
    # Task arguments
    parser.add_argument('--task_type', type=str, default='wikitext', choices=['wikitext'], help='Task to perform.')
    
    # Model hyperparameters (optimized for 7M parameters)
    parser.add_argument('--embed_dim', type=int, default=160, help='Embedding dimension.')
    parser.add_argument('--num_heads', type=int, default=8, help='Number of query attention heads.')
    parser.add_argument('--num_kv_heads', type=int, default=2, help='Number of key-value heads (GQA).')
    parser.add_argument('--num_encoder_layers', type=int, default=10, help='Number of encoder layers.')
    parser.add_argument('--dim_feedforward', type=int, default=320, help='Dimension of feedforward network.')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate.')

    # Cognitive mechanism hyperparameters
    parser.add_argument('--mem_slots', type=int, default=16, help='Number of memory slots.')
    parser.add_argument('--mem_rank', type=int, default=16, help='Rank for low-rank memory projections.')
    parser.add_argument('--reasoning_tokens', type=int, default=64, help='Number of reasoning buffer tokens.')

    # Training hyperparameters (optimized for small models)
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Base learning rate (reduced for Lion optimizer).')
    parser.add_argument('--num_epochs', type=int, default=30, help='Number of training epochs (small models need more).')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size.')
    parser.add_argument('--grad_clip_value', type=float, default=1.0, help='Gradient clipping value.')
    parser.add_argument('--save_steps', type=int, default=1000, help='Save checkpoint every N steps.')
    parser.add_argument('--accumulation_steps', type=int, default=4, help='Gradient accumulation steps (effective batch=32).')
    parser.add_argument('--fp16', action='store_true', help='Enable Mixed Precision (AMP) training.')

    # Advanced training options
    parser.add_argument('--label_smoothing', type=float, default=0.1, help='Label smoothing factor (0-1).')
    parser.add_argument('--use_cognitive_losses', action='store_true', help='Enable cognitive auxiliary losses.')

    # Post-training optimization
    parser.add_argument('--pruning_amount', type=float, default=0.2, help='Amount of pruning to apply (0-1).')
    parser.add_argument('--fine_tune_epochs', type=int, default=2, help='Epochs to fine-tune after pruning.')

    # Smart Training Args
    parser.add_argument('--smart_training', action='store_true', help='Enable Active Data Selection (skip easy batches).')
    parser.add_argument('--skip_threshold', type=float, default=0.5, help='Loss threshold for skipping batches.')

    args = parser.parse_args()
    
    # Enable TF32 for faster training on Ampere GPUs
    torch.set_float32_matmul_precision('medium')
    
    if not torch.cuda.is_available():
        print("!"*80)
        print("WARNING: GPU not detected! Training will be extremely slow.")
        print("Please enable GPU in Google Colab: Runtime > Change runtime type > Hardware accelerator > GPU")
        print("!"*80)

    main(args)