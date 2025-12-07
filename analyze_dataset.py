import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from transformers import AutoTokenizer
from src.dataset import get_dataloader
import os

def analyze_dataset(dataset_type, split):
    """
    Analyzes a dataset split and prints statistics about sequence lengths.
    Also generates visualization plots.
    """
    print(f"Analyzing dataset: {dataset_type}, split: {split}")

    # 1. Load Tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained("model_artifacts/tokenizer")
    except OSError:
        print("Tokenizer not found. Please run `train_tokenizer.py` first.")
        return

    # 2. Load DataLoader
    # Use a batch size of 1 to inspect each item individually without padding effects.
    dataloader = get_dataloader(
        dataset_type=dataset_type,
        tokenizer=tokenizer,
        max_seq_len=4096, # Use a large max_seq_len to avoid truncation during analysis
        max_answer_len=1024,
        batch_size=1,
        split=split,
        shuffle=False
    )

    if dataloader is None:
        print("Dataloader could not be created. Exiting.")
        return

    # 3. Iterate and collect lengths
    input_lengths = []
    target_lengths = []

    # Limit analysis to first N samples for speed on large datasets like WikiText
    MAX_SAMPLES = 5000 
    print(f"Scanning first {MAX_SAMPLES} samples...")

    for i, batch in enumerate(tqdm(dataloader, desc=f"Analyzing {split} set")):
        if i >= MAX_SAMPLES: break
        
        # The dataloader returns tensors in the batch
        input_ids = batch['input_ids']
        answer_ids = batch['answer_ids']
        
        # Count non-padding tokens
        input_len = (input_ids != tokenizer.pad_token_id).sum().item()
        target_len = (answer_ids != tokenizer.pad_token_id).sum().item()

        input_lengths.append(input_len)
        target_lengths.append(target_len)

    # 4. Calculate and Print Statistics
    total_samples = len(input_lengths)
    print(f"\n--- Analysis for '{dataset_type}/{split}' ---")
    print(f"Total Samples Scanned: {total_samples}")

    def print_stats(name, lengths):
        if not lengths:
            print(f"\nNo data for '{name}'.")
            return
        
        lengths_np = np.array(lengths)
        print(f"\nStatistics for '{name}' lengths (in tokens):")
        print(f"  - Mean:   {lengths_np.mean():.2f}")
        print(f"  - Median: {np.median(lengths_np):.2f}")
        print(f"  - Std Dev:{lengths_np.std():.2f}")
        print(f"  - Min:    {lengths_np.min()}")
        print(f"  - Max:    {lengths_np.max()}")
        print(f"  - 90th Percentile: {np.percentile(lengths_np, 90):.2f}")
        
        return lengths_np

    input_np = print_stats("Input (Context)", input_lengths)
    target_np = print_stats("Target (Continuation)", target_lengths)
    
    # 5. Visualizations
    print("\nGenerating visualizations...")
    os.makedirs("analysis_plots", exist_ok=True)
    
    # Set style
    sns.set_theme(style="whitegrid")
    
    # Plot 1: Distribution of Input Lengths
    plt.figure(figsize=(10, 6))
    sns.histplot(input_np, bins=30, kde=True, color="skyblue")
    plt.title(f"Distribution of Input Token Lengths ({dataset_type})")
    plt.xlabel("Number of Tokens")
    plt.ylabel("Count")
    plt.savefig(f"analysis_plots/{dataset_type}_{split}_input_dist.png")
    plt.close()
    
    # Plot 2: Distribution of Target Lengths
    plt.figure(figsize=(10, 6))
    sns.histplot(target_np, bins=30, kde=True, color="salmon")
    plt.title(f"Distribution of Target Token Lengths ({dataset_type})")
    plt.xlabel("Number of Tokens")
    plt.ylabel("Count")
    plt.savefig(f"analysis_plots/{dataset_type}_{split}_target_dist.png")
    plt.close()

    print(f"Plots saved to 'analysis_plots/' directory.")
    print("\nAnalysis complete.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Analyze dataset statistics.")
    parser.add_argument('dataset_type', type=str, default='wikitext', choices=['wikitext'], 
                        help='The type of dataset to analyze.')
    parser.add_argument('split', type=str, choices=['train', 'validation'],
                        help='The dataset split to analyze.')
    
    args = parser.parse_args()
    analyze_dataset(args.dataset_type, args.split)
