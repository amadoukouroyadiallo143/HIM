# Model Training and Hyperparameter Tuning Guide

This guide provides recommendations and strategies for effectively training the HIM model and tuning its hyperparameters to achieve optimal performance for your specific task (Question Answering or Summarization).

## 1. Start with a Good Baseline

Begin your experimentation with this balanced configuration. It serves as a solid starting point for comparison and should yield reasonable results.

```bash
python main.py --task_type squad --num_epochs 20 --learning_rate 0.0001 --batch_size 32 --norm_first
```

**Explanation of Baseline Parameters:**
*   `--task_type squad`: Focuses on the Question Answering task. Change to `summarization` for summarization.
*   `--num_epochs 20`: A reasonable number of epochs to observe initial learning trends.
*   `--learning_rate 0.0001`: A commonly effective learning rate for Adam-based optimizers in Transformer models.
*   `--batch_size 32`: A good compromise between training speed and stability. If you encounter out-of-memory errors, reduce this value (e.g., to 16).
*   `--norm_first`: Activates Pre-Layer Normalization, which often leads to more stable training for deeper models.

## 2. Interpreting Results with TensorBoard

TensorBoard is your primary tool for monitoring training progress and making informed tuning decisions. Launch it in a separate terminal while training:

```bash
tensorboard --logdir=runs
```
Then open `http://localhost:6006/` in your web browser. Pay close attention to these graphs:

*   **`Loss/val` (Validation Loss)**: This is crucial. Ensure it **decreases smoothly**. If it plateaus or starts increasing, the model might be overfitting or has stopped learning.
*   **`Perplexity/val` (Validation Perplexity)**: A lower perplexity indicates a better language model. It should decrease alongside validation loss.
*   **`F1/val` (for SQuAD) / `ROUGE-L/val` (for Summarization)**: These are your primary task-specific performance metrics. Aim for the **highest possible values**.
*   **`BERTScore_F1/val`**: Provides a semantic similarity score, often correlating better with human judgment than token-overlap metrics. Aim for higher values.
*   **`Diversity/distinct-2`**: Monitors the diversity of the generated text. A very low value might indicate repetitive or generic outputs.
*   **`Gradient_Norm/train`**: If this graph shows **spikes or extremely high values**, your training is unstable (exploding gradients). This is a strong indicator to **lower your learning rate**.
*   **`Learning_Rate`**: Observe how your learning rate scheduler adjusts the learning rate over epochs.

## 3. Hyperparameter Tuning Strategy

Once you have a stable baseline, follow a methodical approach to optimize performance.

### Step A: Tune Model Complexity (`num_encoder_layers`, `embed_dim`, `dim_feedforward`)

The goal here is to find the right "size" for your model. A larger model has more capacity but is slower and more prone to overfitting. A smaller model is faster but might underfit.

1.  **Experiment with a Larger Model:**
    ```bash
    python main.py --num_encoder_layers 6 --embed_dim 256 --dim_feedforward 1024
    ```
    *   **Observation**: Does the validation loss continue to decrease for longer? Do `F1/val` or `BERTScore_F1/val` improve? If so, the model might benefit from more capacity.

2.  **Experiment with a Smaller Model:**
    ```bash
    python main.py --num_encoder_layers 2 --embed_dim 128 --dim_feedforward 256
    ```
    *   **Observation**: Does the model converge faster but with lower peak performance? This might indicate underfitting.

**Decision**: Choose the model complexity that gives the best validation metrics without excessive training time or clear signs of overfitting (validation loss increasing while training loss decreases).

### Step B: Refine the Learning Rate (`learning_rate`)

Using the best model complexity from Step A, fine-tune the learning rate.

*   **If training is unstable (loss spikes, high gradient norm)**: **Lower the learning rate**. Try values like `0.00005` or `0.00002`.
    ```bash
    python main.py --learning_rate 0.00005
    ```
*   **If training is stable but very slow (loss decreases very gradually)**: You might be able to **increase the learning rate slightly**. Try `0.00015` or `0.0002`.
    ```bash
    python main.py --learning_rate 0.00015
    ```

### Step C: Experiment with Normalization (`--norm_first`)

For your best-performing configuration, try toggling the `--norm_first` flag.

*   **With Pre-LN (default in baseline):**
    ```bash
    python main.py --norm_first
    ```
*   **With Post-LN:**
    ```bash
    python main.py --no-norm_first # Note: use --no- prefix for store_true flags
    ```
    *   **Observation**: Does one configuration lead to faster convergence or better final metrics? Pre-LN is often more stable for deeper models.

### 4. Qualitative Analysis with Attention Visualizations

Regularly inspect the attention heatmaps generated in `attention_visualizations/`.

*   **Purpose**: These images show what parts of the input sequence the model is "focusing" on for each token.
*   **Interpretation**: Look for patterns. Does the model attend to relevant keywords? Does it ignore padding tokens? Are there any unexpected or diffuse attention patterns? This can provide qualitative insights into whether the model is learning meaningful relationships.

### 5. Post-Training Optimization

Remember that after training, the model automatically undergoes pruning and dynamic quantization. These steps are crucial for deploying efficient models.

*   `--pruning_amount`: Adjust the percentage of weights to prune. Higher values mean smaller models but can impact accuracy.
*   `--fine_tune_epochs`: The number of epochs to fine-tune the model after pruning. This helps recover any performance lost due to pruning.

By systematically following these steps and leveraging the provided tools, you can effectively train and optimize your HIM model.
