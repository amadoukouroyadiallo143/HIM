
"""
🧠 HIM - Hyper-Introspective Model (7M Parameters)
Converted from Jupyter Notebook.

Resolves SyntaxError: invalid syntax caused by '!' magic commands.
"""

import os
import sys
import json
import torch
# import matplotlib.pyplot as plt # Optional if not using interactive plotting

# Ensure current directory is in path for imports
sys.path.append(os.getcwd())

print("## 1️⃣ Configuration et Installation")

# Vérifier le GPU
print("Checking GPU...")
os.system('nvidia-smi')

# Installer les dépendances
print("Installing dependencies...")
# Using sys.executable to ensure we install in the current environment
os.system(f'{sys.executable} -m pip install -q torch transformers datasets tokenizers tqdm evaluate sacrebleu bert-score nltk matplotlib seaborn')

# Créer la structure de dossiers (equivalent to !mkdir -p)
print("Creating directory structure...")
os.makedirs('src', exist_ok=True)
os.makedirs('model_artifacts/tokenizer', exist_ok=True)

print("## 2️⃣ Entraînement du Tokenizer")

# Entraîner le tokenizer BPE (30K tokens)
print("Training tokenizer...")
os.system(f'{sys.executable} train_tokenizer.py --vocab_size 30000 --dataset_name wikitext --config_name wikitext-103-v1')

print("## 3️⃣ Entraînement du Modèle")

CONFIG = {
    'embed_dim': 160,
    'num_heads': 8,
    'num_kv_heads': 2,
    'num_encoder_layers': 10,
    'dim_feedforward': 320,
    'mem_slots': 16,
    'mem_rank': 16,
    'reasoning_tokens': 64,
    'num_epochs': 30,
    'batch_size': 16,
    'accumulation_steps': 4,
    'learning_rate': 1e-3,
    'grad_clip_value': 1.0,
    'dropout': 0.1,
    'label_smoothing': 0.1,
    'fp16': True,
    'use_cognitive_losses': True,
    'smart_training': False,
    'save_steps': 5000,
    'pruning_amount': 0.2,
    'fine_tune_epochs': 2
}

print("Starting training (main.py)...")
cmd = f"{sys.executable} main.py \
    --task_type wikitext \
    --embed_dim {CONFIG['embed_dim']} \
    --num_heads {CONFIG['num_heads']} \
    --num_kv_heads {CONFIG['num_kv_heads']} \
    --num_encoder_layers {CONFIG['num_encoder_layers']} \
    --dim_feedforward {CONFIG['dim_feedforward']} \
    --mem_slots {CONFIG['mem_slots']} \
    --mem_rank {CONFIG['mem_rank']} \
    --reasoning_tokens {CONFIG['reasoning_tokens']} \
    --num_epochs {CONFIG['num_epochs']} \
    --batch_size {CONFIG['batch_size']} \
    --accumulation_steps {CONFIG['accumulation_steps']} \
    --learning_rate {CONFIG['learning_rate']} \
    --grad_clip_value {CONFIG['grad_clip_value']} \
    --dropout {CONFIG['dropout']} \
    --label_smoothing {CONFIG['label_smoothing']} \
    --fp16 \
    --use_cognitive_losses \
    --pruning_amount {CONFIG['pruning_amount']} \
    --fine_tune_epochs {CONFIG['fine_tune_epochs']} \
    --save_steps {CONFIG['save_steps']}"

# Execute training
exit_code = os.system(cmd)
if exit_code != 0:
    print("Training failed or was interrupted.")
    sys.exit(exit_code)

print("Training script finished.")
