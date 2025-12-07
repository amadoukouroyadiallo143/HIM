"""
Advanced Training Module for HIM 7M Parameters Model
Includes:
- Multi-speed learning rates (slow/fast/cognitive paths)
- Auxiliary cognitive losses (memory, reasoning, routing balance)
- Advanced cognitive metrics (memory utilization, reasoning evolution, routing stats)
- OneCycleLR scheduling
- Label smoothing
"""

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import OneCycleLR
from tqdm import tqdm
import os
import json
import re
import string
from collections import Counter
from nltk.util import ngrams
import matplotlib.pyplot as plt
import seaborn as sns
import torch.nn.utils.prune as prune
import torch.quantization
import evaluate
import numpy as np

from src.model import NLDirectResponse
from src.optimizer import Lion
from src.dataset import get_dataloader
from src.components import apply_lora
from transformers import AutoTokenizer

# --- Utility Functions ---

def normalize_text(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

def calculate_qa_metrics(predictions, original_answers_batch, tokenizer, ignore_index=-100):
    em_scores, f1_scores, bleu_scores = [], [], []
    bleu_metric = evaluate.load("sacrebleu")

    for i in range(predictions.shape[0]):
        pred_tokens = predictions[i].tolist()
        pred_str = tokenizer.decode([t for t in pred_tokens if t != ignore_index and t != tokenizer.pad_token_id], skip_special_tokens=True).strip()
        ground_truth_answers = original_answers_batch[i]

        # BLEU
        bleu_result = bleu_metric.compute(predictions=[pred_str], references=[ground_truth_answers])
        bleu_scores.append(bleu_result['score'])

        # EM & F1
        normalized_pred = normalize_text(pred_str)
        max_em, max_f1 = 0, 0
        for gt_answer in ground_truth_answers:
            normalized_gt = normalize_text(gt_answer)
            if normalized_pred == normalized_gt:
                max_em = 1
            prediction_tokens = normalized_pred.split()
            ground_truth_tokens = normalized_gt.split()
            common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
            num_common = sum(common.values())
            if num_common > 0:
                precision = num_common / len(prediction_tokens) if len(prediction_tokens) > 0 else 0
                recall = num_common / len(ground_truth_tokens) if len(ground_truth_tokens) > 0 else 0
                f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
                if f1 > max_f1: max_f1 = f1
        em_scores.append(max_em)
        f1_scores.append(max_f1)

    return sum(em_scores) / len(em_scores) if em_scores else 0, \
           sum(f1_scores) / len(f1_scores) if f1_scores else 0, \
           sum(bleu_scores) / len(bleu_scores) if bleu_scores else 0

def calculate_advanced_metrics(decoded_preds, decoded_labels):
    """Calculates diversity (distinct-n) and BERTScore."""
    all_tokens = [tok for pred in decoded_preds for tok in pred.split()]
    distinct_1 = len(set(ngrams(all_tokens, 1))) / len(all_tokens) if len(all_tokens) > 0 else 0
    distinct_2 = len(set(ngrams(all_tokens, 2))) / (len(all_tokens) -1) if len(all_tokens) > 1 else 0

    try:
        bertscore = evaluate.load("bertscore")
        bert_results = bertscore.compute(predictions=decoded_preds, references=decoded_labels, lang="en")
        avg_bert_f1 = sum(bert_results['f1']) / len(bert_results['f1'])
    except Exception as e:
        print(f"Warning: BERTScore calculation failed: {e}")
        avg_bert_f1 = 0.0

    return {"distinct_1": distinct_1, "distinct_2": distinct_2, "bertscore_f1": avg_bert_f1}


# --- COGNITIVE METRICS (NEW) ---

def evaluate_memory_usage(model):
    """
    Measure which memory slots are actively used.
    Returns utilization ratio (0-1).
    """
    if not hasattr(model, 'memory') or model.memory is None:
        return 0.0

    with torch.no_grad():
        mem = model.memory.memory.squeeze()  # [mem_slots, mem_dim]
        activation = torch.norm(mem, dim=-1)  # [mem_slots]
        utilization = (activation > activation.mean()).float().mean()

    return utilization.item()

def analyze_reasoning_paths(model):
    """
    Track reasoning buffer changes.
    Returns mean and std of reasoning state norms.
    """
    if not hasattr(model, 'reasoning_module') or model.reasoning_module is None:
        return {'mean': 0.0, 'std': 0.0}

    with torch.no_grad():
        reasoning_embeddings = model.reasoning_module.reasoning_embeddings
        norms = torch.norm(reasoning_embeddings, dim=-1)  # [reasoning_tokens]

    return {
        'mean': norms.mean().item(),
        'std': norms.std().item(),
        'max': norms.max().item(),
        'min': norms.min().item()
    }

def analyze_speed_routing(model):
    """
    Show which layers prefer fast vs slow paths.
    Returns list of routing preferences per layer.
    """
    routing_prefs = []
    for i, layer in enumerate(model.transformer_encoder):
        if hasattr(layer, 'multi_speed') and hasattr(layer.multi_speed, 'gate'):
            gate_value = torch.sigmoid(layer.multi_speed.gate).mean().item()
            routing_prefs.append({
                'layer': i,
                'slow_preference': gate_value,
                'fast_preference': 1 - gate_value
            })
    return routing_prefs


# --- AUXILIARY LOSSES (NEW) ---

def compute_cognitive_losses(model, base_loss, loss_weights=None):
    """
    Compute auxiliary losses for cognitive mechanisms.

    Args:
        model: NLDirectResponse model
        base_loss: Main language modeling loss
        loss_weights: Dict with keys 'memory', 'reasoning', 'routing' (defaults provided)

    Returns:
        total_loss: Combined loss
        loss_dict: Dict with individual loss components
    """
    if loss_weights is None:
        loss_weights = {
            'memory': 0.01,      # Memory sparsity weight
            'reasoning': 0.01,   # Reasoning diversity weight
            'routing': 0.01      # Routing balance weight
        }

    loss_dict = {'lm_loss': base_loss.item()}
    total_loss = base_loss

    # 1. Memory Sparsity Loss (encourage selective memory use)
    if hasattr(model, 'memory') and model.memory is not None:
        mem_sparsity = torch.mean(torch.abs(model.memory.memory))
        mem_loss = loss_weights['memory'] * mem_sparsity
        total_loss = total_loss + mem_loss
        loss_dict['mem_loss'] = mem_loss.item()
    else:
        loss_dict['mem_loss'] = 0.0

    # 2. Reasoning Diversity Loss (prevent mode collapse)
    if hasattr(model, 'reasoning_module') and model.reasoning_module is not None:
        reasoning_std = torch.std(model.reasoning_module.reasoning_embeddings)
        # Penalize if std < 0.5 (too similar embeddings)
        reasoning_loss = loss_weights['reasoning'] * torch.relu(0.5 - reasoning_std)
        total_loss = total_loss + reasoning_loss
        loss_dict['reasoning_loss'] = reasoning_loss.item()
    else:
        loss_dict['reasoning_loss'] = 0.0

    # 3. Routing Balance Loss (prevent always using one path)
    gate_means = []
    for layer in model.transformer_encoder:
        if hasattr(layer, 'multi_speed') and hasattr(layer.multi_speed, 'gate'):
            gate_means.append(torch.sigmoid(layer.multi_speed.gate).mean())

    if gate_means:
        # Penalize gates far from 0.5 (balanced routing)
        balance_loss = loss_weights['routing'] * sum([(g - 0.5) ** 2 for g in gate_means]) / len(gate_means)
        total_loss = total_loss + balance_loss
        loss_dict['routing_loss'] = balance_loss.item()
    else:
        loss_dict['routing_loss'] = 0.0

    return total_loss, loss_dict


def visualize_attention(attention_heads, tokens, epoch, layer_index, filename_prefix="attention_viz"):
    """Visualizes attention weights for each head in a layer and saves to a file."""
    num_heads = attention_heads.shape[0]
    seq_len = len(tokens)
    # Limit sequence length for visualization to avoid clutter
    display_len = min(seq_len, 50)
    attention_heads = attention_heads[:, :display_len, :display_len]
    tokens = tokens[:display_len]

    fig, axes = plt.subplots(1, num_heads, figsize=(num_heads * 4, 4))
    if num_heads == 1: axes = [axes]

    for i, ax in enumerate(axes):
        sns.heatmap(attention_heads[i], ax=ax, cmap='viridis', xticklabels=tokens, yticklabels=tokens, cbar=False)
        ax.set_title(f'Head {i+1}')
        ax.tick_params(axis='x', rotation=90)
        ax.tick_params(axis='y', rotation=0)

    plt.suptitle(f'Attention - Epoch {epoch + 1}, Layer {layer_index + 1}')
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    save_dir = "attention_visualizations"
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{filename_prefix}_epoch_{epoch+1}_layer_{layer_index+1}.png")
    plt.savefig(save_path)
    plt.close(fig)


# --- Training Components ---

class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0


def create_multi_speed_optimizer(model, base_lr=1e-3):
    """
    Create optimizer with different learning rates for slow/fast/cognitive paths.

    Args:
        model: NLDirectResponse model
        base_lr: Base learning rate

    Returns:
        optimizer: Lion optimizer with parameter groups
        param_groups_info: Dict describing parameter groups
    """
    # Categorize parameters
    slow_params = []
    fast_params = []
    cognitive_params = []
    standard_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        # Slow path: attention and FFN slow components
        if 'attention' in name or 'slow_ffn' in name:
            slow_params.append(param)
        # Fast path: gates and routing
        elif 'gate' in name or 'fast' in name:
            fast_params.append(param)
        # Cognitive: memory and reasoning
        elif 'memory' in name or 'reasoning' in name:
            cognitive_params.append(param)
        else:
            standard_params.append(param)

    # Create parameter groups with different learning rates
    param_groups = [
        {'params': slow_params, 'lr': base_lr * 0.3, 'name': 'slow'},
        {'params': fast_params, 'lr': base_lr * 1.5, 'name': 'fast'},
        {'params': cognitive_params, 'lr': base_lr * 0.5, 'name': 'cognitive'},
        {'params': standard_params, 'lr': base_lr, 'name': 'standard'}
    ]

    optimizer = Lion(param_groups, weight_decay=1e-2)

    param_groups_info = {
        'slow': len(slow_params),
        'fast': len(fast_params),
        'cognitive': len(cognitive_params),
        'standard': len(standard_params),
        'total': len(slow_params) + len(fast_params) + len(cognitive_params) + len(standard_params)
    }

    print(f"\nMulti-Speed Optimizer Created:")
    print(f"  Slow params (LR={base_lr*0.3:.2e}):      {param_groups_info['slow']}")
    print(f"  Fast params (LR={base_lr*1.5:.2e}):      {param_groups_info['fast']}")
    print(f"  Cognitive params (LR={base_lr*0.5:.2e}): {param_groups_info['cognitive']}")
    print(f"  Standard params (LR={base_lr:.2e}):      {param_groups_info['standard']}")
    print(f"  Total: {param_groups_info['total']}\n")

    return optimizer, param_groups_info


def train_epoch(model, dataloader, optimizer, criterion, device, epoch, writer,
                grad_clip_value, smart_training, skip_threshold, pad_token_id,
                save_steps=0, artifacts_dir="model_artifacts", accumulation_steps=1,
                use_amp=False, use_cognitive_losses=True):
    """
    Enhanced training epoch with cognitive losses and metrics.
    """
    model.train()
    total_loss = 0
    total_grad_norm = 0
    num_batches = 0
    skipped_batches = 0

    # Accumulators for cognitive losses
    cognitive_loss_accum = {'lm_loss': 0, 'mem_loss': 0, 'reasoning_loss': 0, 'routing_loss': 0}

    tqdm_bar = tqdm(dataloader, desc=f"Epoch {epoch+1} [Train]")

    # Initialize Scaler for AMP
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)

    for batch_idx, batch in enumerate(tqdm_bar):
        optimizer.zero_grad()
        model.reset_memory()  # Reset memory for each batch

        input_ids = batch['input_ids'].to(device)
        answer_ids = batch['answer_ids'].to(device)
        src_padding_mask = (input_ids == pad_token_id).to(device)

        # Forward pass
        with torch.amp.autocast(device.type, enabled=use_amp):
            logits = model(input_ids, src_padding_mask=src_padding_mask)
            base_loss = criterion(logits.view(-1, logits.size(-1)), answer_ids.view(-1))

            # Add cognitive losses
            if use_cognitive_losses:
                loss, loss_dict = compute_cognitive_losses(model, base_loss)
                for k, v in loss_dict.items():
                    cognitive_loss_accum[k] += v
            else:
                loss = base_loss
                cognitive_loss_accum['lm_loss'] += base_loss.item()

        # Smart Training Logic (skip easy batches)
        if smart_training and base_loss.item() < skip_threshold:
            skipped_batches += 1
            continue

        # Normalize loss for gradient accumulation
        loss = loss / accumulation_steps

        # Backward pass with AMP
        scaler.scale(loss).backward()

        if (batch_idx + 1) % accumulation_steps == 0:
            # Unscale before clipping
            scaler.unscale_(optimizer)
            grad_norm = nn.utils.clip_grad_norm_(model.parameters(), grad_clip_value).item()
            total_grad_norm += grad_norm

            # Step with scaler
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        # Restore loss for logging
        total_loss += loss.item() * accumulation_steps
        num_batches += 1

        tqdm_bar.set_postfix(
            loss=total_loss / num_batches,
            mem_loss=cognitive_loss_accum['mem_loss'] / num_batches if use_cognitive_losses else 0,
            skipped=skipped_batches,
            refresh=True
        )

        # Intra-Epoch Checkpointing
        if save_steps > 0 and num_batches % save_steps == 0:
            checkpoint_path = os.path.join(artifacts_dir, f"checkpoint_epoch_{epoch+1}_step_{num_batches}.pth")
            model.reset_memory()
            torch.save({
                'epoch': epoch,
                'step': num_batches,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss.item(),
            }, checkpoint_path)

    avg_loss = total_loss / num_batches if num_batches > 0 else 0
    avg_grad_norm = total_grad_norm / (num_batches - skipped_batches) if (num_batches - skipped_batches) > 0 else 0

    # Log to TensorBoard
    if writer:
        writer.add_scalar('Loss/train', avg_loss, epoch)
        writer.add_scalar('Gradient_Norm/train', avg_grad_norm, epoch)
        writer.add_scalar('Skipped_Batches/train', skipped_batches, epoch)

        if use_cognitive_losses:
            for loss_name, loss_val in cognitive_loss_accum.items():
                writer.add_scalar(f'CognitiveLoss/{loss_name}', loss_val / num_batches, epoch)

    return avg_loss


def validate_epoch(model, dataloader, criterion, device, epoch, writer, tokenizer,
                   model_config, pad_token_id, compute_cognitive_metrics=True):
    """
    Enhanced validation epoch with cognitive metrics.
    """
    model.eval()
    val_loss = 0
    num_batches = 0
    all_preds, all_labels = [], []

    tqdm_bar = tqdm(dataloader, desc=f"Epoch {epoch+1} [Val]")

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm_bar):
            input_ids = batch['input_ids'].to(device)
            answer_ids = batch['answer_ids'].to(device)
            src_padding_mask = (input_ids == pad_token_id).to(device)

            model.reset_memory()

            # Visualize attention for the first batch
            if batch_idx == 0:
                logits, attention_weights = model(input_ids, src_padding_mask=src_padding_mask, get_attention=True)
                first_sample_tokens = [t for t in tokenizer.convert_ids_to_tokens(input_ids[0]) if t != tokenizer.pad_token]
                for layer_idx, attn_layer in enumerate(attention_weights):
                    if attn_layer is not None:
                        visualize_attention(attn_layer[0].cpu().numpy(), first_sample_tokens, epoch, layer_idx)
            else:
                logits = model(input_ids, src_padding_mask=src_padding_mask)

            loss = criterion(logits.view(-1, logits.size(-1)), answer_ids.view(-1))
            val_loss += loss.item()
            num_batches += 1

            # Decode predictions for metrics
            preds = torch.argmax(logits, dim=-1)
            decoded_preds = [tokenizer.decode([t for t in p if t != pad_token_id], skip_special_tokens=True).strip() for p in preds.tolist()]
            all_preds.extend(decoded_preds)
            all_labels.extend([[s] for s in batch['original_response']])

            tqdm_bar.set_postfix(loss=val_loss / num_batches, refresh=True)

    avg_val_loss = val_loss / num_batches if num_batches > 0 else 0
    perplexity = torch.exp(torch.tensor(avg_val_loss))

    # Calculate Advanced Metrics
    adv_metrics = calculate_advanced_metrics(all_preds, all_labels)

    # Calculate Cognitive Metrics
    if compute_cognitive_metrics:
        mem_util = evaluate_memory_usage(model)
        reasoning_stats = analyze_reasoning_paths(model)
        routing_stats = analyze_speed_routing(model)

        print(f"\nEpoch {epoch+1} Validation Results:")
        print(f"  Val Loss: {avg_val_loss:.4f} | PPL: {perplexity:.2f}")
        print(f"  BERTScore: {adv_metrics['bertscore_f1']:.4f} | Distinct-2: {adv_metrics['distinct_2']:.4f}")
        print(f"  Memory Utilization: {mem_util:.2%}")
        print(f"  Reasoning Buffer (mean norm): {reasoning_stats['mean']:.4f}")

        if routing_stats:
            avg_slow = sum([s['slow_preference'] for s in routing_stats]) / len(routing_stats)
            print(f"  Routing: {avg_slow:.2%} slow, {(1-avg_slow):.2%} fast\n")
    else:
        print(f"Epoch {epoch+1} | Val Loss: {avg_val_loss:.4f} | PPL: {perplexity:.2f} | BERTScore: {adv_metrics['bertscore_f1']:.4f}")

    # TensorBoard Logging
    if writer:
        writer.add_scalar('Loss/val', avg_val_loss, epoch)
        writer.add_scalar('Perplexity/val', perplexity, epoch)
        writer.add_scalar('BERTScore/val', adv_metrics['bertscore_f1'], epoch)
        writer.add_scalar('Diversity/distinct-2', adv_metrics['distinct_2'], epoch)

        if compute_cognitive_metrics:
            writer.add_scalar('Cognitive/memory_utilization', mem_util, epoch)
            writer.add_scalar('Cognitive/reasoning_mean_norm', reasoning_stats['mean'], epoch)
            if routing_stats:
                for stat in routing_stats:
                    writer.add_scalar(f'Routing/layer_{stat["layer"]}_slow', stat['slow_preference'], epoch)

    return avg_val_loss


def train_model_advanced(model, train_dataloader, val_dataloader, tokenizer, device,
                        num_epochs=30, model_config=None, base_lr=1e-3, grad_clip_value=1.0,
                        pruning_amount=0.2, fine_tune_epochs=1, smart_training=False,
                        skip_threshold=0.5, use_tensorboard=True, save_steps=1000,
                        accumulation_steps=4, use_amp=True, use_cognitive_losses=True,
                        label_smoothing=0.1):
    """
    Advanced training function with multi-speed learning, cognitive losses, and OneCycleLR.

    Args:
        model: NLDirectResponse model
        train_dataloader: Training dataloader
        val_dataloader: Validation dataloader
        tokenizer: Tokenizer
        device: torch.device
        num_epochs: Number of training epochs
        model_config: Model configuration dict
        base_lr: Base learning rate (default 1e-3, higher for small models)
        grad_clip_value: Gradient clipping value
        pruning_amount: Amount of pruning (0-1)
        fine_tune_epochs: Epochs for fine-tuning after pruning
        smart_training: Enable active data selection
        skip_threshold: Loss threshold for skipping batches
        use_tensorboard: Enable TensorBoard logging
        save_steps: Save checkpoint every N steps
        accumulation_steps: Gradient accumulation steps
        use_amp: Use automatic mixed precision
        use_cognitive_losses: Enable cognitive auxiliary losses
        label_smoothing: Label smoothing factor (0-1)
    """

    # Initialize TensorBoard
    writer = None
    if use_tensorboard:
        try:
            from torch.utils.tensorboard import SummaryWriter
            writer = SummaryWriter()
        except ImportError:
            print("TensorBoard not found. Skipping logging.")

    model.to(device)

    # Create multi-speed optimizer
    optimizer, param_groups_info = create_multi_speed_optimizer(model, base_lr)

    # Loss function with label smoothing
    criterion = nn.CrossEntropyLoss(
        ignore_index=tokenizer.pad_token_id,
        label_smoothing=label_smoothing
    )
    print(f"Using CrossEntropyLoss with label smoothing: {label_smoothing}")

    # OneCycleLR Scheduler (better than ReduceLROnPlateau for small models)
    total_steps = num_epochs * len(train_dataloader)
    scheduler = OneCycleLR(
        optimizer,
        max_lr=[base_lr * 0.3, base_lr * 1.5, base_lr * 0.5, base_lr],  # Per parameter group
        total_steps=total_steps,
        pct_start=0.1,  # 10% warmup
        anneal_strategy='cos',
        div_factor=25,  # Initial LR = max_lr / 25
        final_div_factor=1000  # Final LR = max_lr / 1000
    )
    print(f"Using OneCycleLR: warmup 10%, cosine annealing, {total_steps} total steps\n")

    # Early Stopping
    early_stopping = EarlyStopping(patience=5, min_delta=0.001)

    # Artifacts Setup
    artifacts_dir = "model_artifacts"
    os.makedirs(artifacts_dir, exist_ok=True)
    os.makedirs("attention_visualizations", exist_ok=True)
    pad_token_id = tokenizer.pad_token_id

    # --- Main Training Loop ---
    print("="*80)
    print("STARTING ADVANCED TRAINING")
    print("="*80)

    for epoch in range(num_epochs):
        # Train
        train_loss = train_epoch(
            model, train_dataloader, optimizer, criterion, device, epoch, writer,
            grad_clip_value, smart_training, skip_threshold, pad_token_id,
            save_steps=save_steps, artifacts_dir=artifacts_dir,
            accumulation_steps=accumulation_steps, use_amp=use_amp,
            use_cognitive_losses=use_cognitive_losses
        )

        # Validate
        val_loss = validate_epoch(
            model, val_dataloader, criterion, device, epoch, writer,
            tokenizer, model_config, pad_token_id,
            compute_cognitive_metrics=True
        )

        # Scheduler Step (OneCycleLR steps after each batch, but we log LR per epoch)
        current_lr = optimizer.param_groups[0]['lr']
        if writer:
            writer.add_scalar('Learning_Rate', current_lr, epoch)

        # Save Checkpoint (Best & Latest)
        model.reset_memory()
        torch.save(model.state_dict(), os.path.join(artifacts_dir, "nl_direct_response_model.pth"))
        if early_stopping.best_loss == val_loss:
            torch.save(model.state_dict(), os.path.join(artifacts_dir, "nl_direct_response_model_best.pth"))
            print(f"✓ New best model saved with loss {val_loss:.4f}")

        # Save Config & Tokenizer
        tokenizer.save_pretrained(os.path.join(artifacts_dir, "tokenizer"))
        with open(os.path.join(artifacts_dir, "model_config.json"), 'w') as f:
            json.dump(model_config, f, indent=4)

        # Early Stopping Check
        early_stopping(val_loss)
        if early_stopping.early_stop:
            print("\n⚠ Early stopping triggered.")
            break

    if writer:
        writer.close()

    print("\n" + "="*80)
    print("TRAINING COMPLETE")
    print("="*80)

    # --- Post-Training Optimization ---
    print("\n--- Starting Post-Training Optimization ---")

    # 1. Pruning
    if pruning_amount > 0:
        print(f"Applying {pruning_amount*100:.0f}% unstructured pruning...")
        for module in model.modules():
            if isinstance(module, nn.Linear):
                prune.random_unstructured(module, name="weight", amount=pruning_amount)

        # 2. Fine-tuning after pruning
        if fine_tune_epochs > 0:
            print(f"Fine-tuning pruned model for {fine_tune_epochs} epochs...")
            for epoch in range(fine_tune_epochs):
                train_epoch(
                    model, train_dataloader, optimizer, criterion, device, num_epochs + epoch, writer,
                    grad_clip_value, smart_training, skip_threshold, pad_token_id,
                    save_steps=save_steps, artifacts_dir=artifacts_dir,
                    accumulation_steps=accumulation_steps, use_amp=use_amp,
                    use_cognitive_losses=use_cognitive_losses
                )

        # Make pruning permanent
        for module in model.modules():
            if isinstance(module, nn.Linear):
                try:
                    prune.remove(module, 'weight')
                except:
                    pass

        torch.save(model.state_dict(), os.path.join(artifacts_dir, "nl_direct_response_model_pruned.pth"))
        print("✓ Pruned model saved.")

    # 3. Quantization
    print("Applying Dynamic Quantization...")
    model.to("cpu").eval()
    quantized_model = torch.quantization.quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)
    torch.save(quantized_model.state_dict(), os.path.join(artifacts_dir, "nl_direct_response_model_quantized.pth"))
    print("✓ Quantized model saved.")

    print("\n" + "="*80)
    print("ALL OPTIMIZATION COMPLETE")
    print("="*80)
    print(f"\nModel artifacts saved to: {artifacts_dir}/")
    print(f"  - nl_direct_response_model_best.pth (best validation)")
    print(f"  - nl_direct_response_model_pruned.pth (pruned)")
    print(f"  - nl_direct_response_model_quantized.pth (quantized)")
