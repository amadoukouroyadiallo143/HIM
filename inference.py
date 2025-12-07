import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer
import argparse
import json
import time
import os
from typing import List, Dict, Optional, Tuple
import numpy as np

from src.model import NLDirectResponse

class AdvancedInference:
    """
    Advanced inference system for HIM with cognitive mechanism visualization.

    Features:
    - Advanced sampling: top-k, top-p (nucleus), temperature
    - Streaming generation
    - Cognitive mechanism visualization (memory, reasoning, routing)
    - Performance optimizations
    - Batch processing support
    """

    def __init__(self, model_path: str, device: str = "auto"):
        """
        Initialize inference engine.

        Args:
            model_path: Path to the saved model checkpoint
            device: Device to use ("cuda", "cpu", or "auto")
        """
        # Device setup
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        print(f"Loading model from: {model_path}")
        print(f"Using device: {self.device}")

        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=self.device)

        # Extract model config
        self.config = checkpoint.get('config', {})

        # Load tokenizer
        tokenizer_path = self.config.get('tokenizer_name', 'model_artifacts/tokenizer')
        if not os.path.exists(tokenizer_path):
            raise ValueError(f"Tokenizer not found at {tokenizer_path}. Please train tokenizer first.")

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        print(f"Loaded tokenizer from: {tokenizer_path}")

        # Initialize model with correct parameters
        model_args = {
            'vocab_size': self.config.get('vocab_size', len(self.tokenizer)),
            'embed_dim': self.config.get('embed_dim', 160),
            'num_heads': self.config.get('num_heads', 8),
            'num_kv_heads': self.config.get('num_kv_heads', 2),
            'num_encoder_layers': self.config.get('num_encoder_layers', 10),
            'dim_feedforward': self.config.get('dim_feedforward', 320),
            'max_answer_len': self.config.get('max_answer_len', 64),
            'dropout': 0.0,  # No dropout during inference
            'mem_slots': self.config.get('mem_slots', 16),
            'mem_rank': self.config.get('mem_rank', 16),
            'reasoning_tokens': self.config.get('reasoning_tokens', 64)
        }

        self.model = NLDirectResponse(**model_args)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()

        print(f"Model loaded successfully!")
        print(f"Parameters: {sum(p.numel() for p in self.model.parameters()):,}")

        # Cognitive tracking
        self.track_cognitive = True
        self.cognitive_history = {
            'memory_utilization': [],
            'reasoning_evolution': [],
            'routing_stats': []
        }

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        max_length: int = 64,
        temperature: float = 1.0,
        top_k: Optional[int] = 50,
        top_p: Optional[float] = 0.9,
        num_return_sequences: int = 1,
        stream: bool = False,
        visualize_cognitive: bool = False
    ) -> Dict:
        """
        Generate text with advanced sampling.

        Args:
            prompt: Input text prompt
            max_length: Maximum generation length
            temperature: Sampling temperature (higher = more random)
            top_k: Keep only top k tokens (None = disabled)
            top_p: Nucleus sampling threshold (None = disabled)
            num_return_sequences: Number of sequences to generate
            stream: Enable streaming generation
            visualize_cognitive: Show cognitive mechanism activations

        Returns:
            Dictionary with generated text and metadata
        """
        start_time = time.time()

        # Tokenize input
        inputs = self.tokenizer(
            prompt,
            return_tensors='pt',
            max_length=self.config.get('max_seq_len', 512),
            truncation=True,
            padding='max_length'
        )

        input_ids = inputs['input_ids'].to(self.device)
        attention_mask = inputs['attention_mask'].to(self.device)

        # Reset memory for clean generation
        self.model.reset_memory()

        # Reset cognitive tracking
        if visualize_cognitive:
            self.cognitive_history = {
                'memory_utilization': [],
                'reasoning_evolution': [],
                'routing_stats': []
            }

        # Generate
        if stream:
            return self._generate_streaming(
                input_ids, attention_mask, max_length,
                temperature, top_k, top_p, visualize_cognitive
            )
        else:
            return self._generate_batch(
                input_ids, attention_mask, max_length,
                temperature, top_k, top_p, num_return_sequences,
                visualize_cognitive, start_time
            )

    def _generate_batch(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        max_length: int,
        temperature: float,
        top_k: Optional[int],
        top_p: Optional[float],
        num_return_sequences: int,
        visualize_cognitive: bool,
        start_time: float
    ) -> Dict:
        """Batch generation (non-streaming)."""

        # Forward pass
        logits = self.model(input_ids, src_padding_mask=attention_mask)

        # Track cognitive mechanisms
        if visualize_cognitive:
            self._track_cognitive_state()

        # Process logits for generation
        generated_sequences = []

        for _ in range(num_return_sequences):
            # Sample from logits
            sampled_ids = self._sample_from_logits(
                logits[0],  # [max_answer_len, vocab_size]
                temperature=temperature,
                top_k=top_k,
                top_p=top_p
            )

            # Decode
            generated_text = self.tokenizer.decode(
                sampled_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True
            )

            generated_sequences.append(generated_text)

        # Compute generation stats
        end_time = time.time()
        generation_time = end_time - start_time
        tokens_per_second = max_length / generation_time if generation_time > 0 else 0

        result = {
            'generated_text': generated_sequences,
            'metadata': {
                'generation_time': generation_time,
                'tokens_per_second': tokens_per_second,
                'num_tokens': max_length,
                'temperature': temperature,
                'top_k': top_k,
                'top_p': top_p
            }
        }

        if visualize_cognitive:
            result['cognitive_analysis'] = self._format_cognitive_analysis()

        return result

    def _generate_streaming(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        max_length: int,
        temperature: float,
        top_k: Optional[int],
        top_p: Optional[float],
        visualize_cognitive: bool
    ) -> Dict:
        """Streaming generation (yields tokens as generated)."""

        print("\n[STREAMING MODE]")
        print("-" * 60)

        # Forward pass
        logits = self.model(input_ids, src_padding_mask=attention_mask)

        # Track cognitive
        if visualize_cognitive:
            self._track_cognitive_state()

        # Sample and stream tokens
        generated_tokens = []
        generated_text = ""

        for i in range(min(max_length, logits.size(1))):
            # Sample single token
            token_logits = logits[0, i, :]  # [vocab_size]
            token_id = self._sample_single_token(
                token_logits,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p
            )

            generated_tokens.append(token_id.item())

            # Decode and stream
            token_text = self.tokenizer.decode([token_id], skip_special_tokens=True)
            print(token_text, end='', flush=True)
            generated_text += token_text

            # Stop at EOS
            if token_id == self.tokenizer.eos_token_id:
                break

        print("\n" + "-" * 60)

        result = {
            'generated_text': [generated_text],
            'metadata': {
                'num_tokens': len(generated_tokens),
                'temperature': temperature,
                'top_k': top_k,
                'top_p': top_p
            }
        }

        if visualize_cognitive:
            result['cognitive_analysis'] = self._format_cognitive_analysis()

        return result

    def _sample_from_logits(
        self,
        logits: torch.Tensor,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None
    ) -> List[int]:
        """
        Sample sequence of tokens from logits with advanced sampling.

        Args:
            logits: [seq_len, vocab_size]
            temperature: Sampling temperature
            top_k: Top-k filtering
            top_p: Nucleus sampling

        Returns:
            List of token IDs
        """
        sampled_ids = []

        for i in range(logits.size(0)):
            token_id = self._sample_single_token(
                logits[i],
                temperature=temperature,
                top_k=top_k,
                top_p=top_p
            )
            sampled_ids.append(token_id.item())

            # Stop at EOS
            if token_id == self.tokenizer.eos_token_id:
                break

        return sampled_ids

    def _sample_single_token(
        self,
        logits: torch.Tensor,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None
    ) -> torch.Tensor:
        """
        Sample a single token from logits.

        Args:
            logits: [vocab_size]
            temperature: Sampling temperature
            top_k: Top-k filtering
            top_p: Nucleus sampling

        Returns:
            Sampled token ID
        """
        # Apply temperature
        if temperature != 1.0:
            logits = logits / temperature

        # Top-k filtering
        if top_k is not None:
            indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
            logits[indices_to_remove] = float('-inf')

        # Nucleus (top-p) filtering
        if top_p is not None:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

            # Remove tokens with cumulative probability above threshold
            sorted_indices_to_remove = cumulative_probs > top_p
            # Keep at least 1 token
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0

            indices_to_remove = sorted_indices[sorted_indices_to_remove]
            logits[indices_to_remove] = float('-inf')

        # Sample from distribution
        probs = F.softmax(logits, dim=-1)
        token_id = torch.multinomial(probs, num_samples=1)

        return token_id

    def _track_cognitive_state(self):
        """Track cognitive mechanism activations."""

        # Memory utilization
        memory_state = self.model.memory.memory.detach().cpu().numpy()
        memory_norms = np.linalg.norm(memory_state, axis=-1)
        active_slots = (memory_norms > 0.01).sum()
        utilization = active_slots / self.model.memory.mem_slots

        self.cognitive_history['memory_utilization'].append({
            'active_slots': int(active_slots),
            'total_slots': self.model.memory.mem_slots,
            'utilization_ratio': float(utilization)
        })

        # Reasoning evolution
        reasoning_state = self.model.reasoning_module.reasoning_embeddings.detach().cpu().numpy()
        reasoning_std = reasoning_state.std()
        reasoning_mean = reasoning_state.mean()

        self.cognitive_history['reasoning_evolution'].append({
            'mean': float(reasoning_mean),
            'std': float(reasoning_std),
            'diversity': float(reasoning_std / (abs(reasoning_mean) + 1e-8))
        })

        # Routing statistics (fast vs slow paths)
        routing_stats = []
        for i, layer in enumerate(self.model.transformer_encoder):
            gate_value = torch.sigmoid(layer.multi_speed.gate).mean().item()
            routing_stats.append({
                'layer': i,
                'slow_preference': float(gate_value),
                'fast_preference': float(1 - gate_value)
            })

        self.cognitive_history['routing_stats'] = routing_stats

    def _format_cognitive_analysis(self) -> Dict:
        """Format cognitive tracking data for display."""

        if not self.cognitive_history['memory_utilization']:
            return {}

        # Memory summary
        memory_data = self.cognitive_history['memory_utilization'][-1]

        # Reasoning summary
        reasoning_data = self.cognitive_history['reasoning_evolution'][-1]

        # Routing summary
        routing_data = self.cognitive_history['routing_stats']
        avg_slow_pref = np.mean([r['slow_preference'] for r in routing_data])

        return {
            'memory': {
                'active_slots': memory_data['active_slots'],
                'total_slots': memory_data['total_slots'],
                'utilization': f"{memory_data['utilization_ratio']*100:.1f}%"
            },
            'reasoning': {
                'mean_activation': f"{reasoning_data['mean']:.4f}",
                'std_activation': f"{reasoning_data['std']:.4f}",
                'diversity_score': f"{reasoning_data['diversity']:.4f}"
            },
            'routing': {
                'avg_slow_preference': f"{avg_slow_pref*100:.1f}%",
                'avg_fast_preference': f"{(1-avg_slow_pref)*100:.1f}%",
                'per_layer': routing_data
            }
        }

    def interactive_mode(self):
        """Interactive chat mode with cognitive visualization."""

        print("\n" + "="*80)
        print("HIM - Hyper-Introspective Model (7M Parameters)")
        print("Interactive Mode with Cognitive Visualization")
        print("="*80)
        print("\nCommands:")
        print("  /help        - Show this help")
        print("  /cognitive   - Toggle cognitive visualization")
        print("  /stream      - Toggle streaming mode")
        print("  /temp X      - Set temperature (0.1-2.0)")
        print("  /topk X      - Set top-k (1-100)")
        print("  /topp X      - Set top-p (0.0-1.0)")
        print("  /quit        - Exit")
        print("="*80 + "\n")

        # Default settings
        settings = {
            'temperature': 0.8,
            'top_k': 50,
            'top_p': 0.9,
            'stream': False,
            'visualize_cognitive': True
        }

        while True:
            try:
                prompt = input("\nYou: ").strip()

                if not prompt:
                    continue

                # Commands
                if prompt.startswith('/'):
                    cmd = prompt.lower().split()

                    if cmd[0] == '/quit':
                        print("\nExiting...")
                        break
                    elif cmd[0] == '/help':
                        print("\nSee commands above.")
                        continue
                    elif cmd[0] == '/cognitive':
                        settings['visualize_cognitive'] = not settings['visualize_cognitive']
                        print(f"\nCognitive visualization: {'ON' if settings['visualize_cognitive'] else 'OFF'}")
                        continue
                    elif cmd[0] == '/stream':
                        settings['stream'] = not settings['stream']
                        print(f"\nStreaming mode: {'ON' if settings['stream'] else 'OFF'}")
                        continue
                    elif cmd[0] == '/temp' and len(cmd) > 1:
                        settings['temperature'] = float(cmd[1])
                        print(f"\nTemperature set to: {settings['temperature']}")
                        continue
                    elif cmd[0] == '/topk' and len(cmd) > 1:
                        settings['top_k'] = int(cmd[1])
                        print(f"\nTop-k set to: {settings['top_k']}")
                        continue
                    elif cmd[0] == '/topp' and len(cmd) > 1:
                        settings['top_p'] = float(cmd[1])
                        print(f"\nTop-p set to: {settings['top_p']}")
                        continue
                    else:
                        print("\nUnknown command. Type /help for commands.")
                        continue

                # Generate response
                print("\nHIM: ", end='', flush=True)

                result = self.generate(
                    prompt=prompt,
                    temperature=settings['temperature'],
                    top_k=settings['top_k'],
                    top_p=settings['top_p'],
                    stream=settings['stream'],
                    visualize_cognitive=settings['visualize_cognitive']
                )

                if not settings['stream']:
                    print(result['generated_text'][0])

                # Display cognitive analysis
                if settings['visualize_cognitive'] and 'cognitive_analysis' in result:
                    self._display_cognitive_analysis(result['cognitive_analysis'])

            except KeyboardInterrupt:
                print("\n\nExiting...")
                break
            except Exception as e:
                print(f"\nError: {e}")

    def _display_cognitive_analysis(self, analysis: Dict):
        """Pretty print cognitive analysis."""

        print("\n" + "-"*60)
        print("COGNITIVE MECHANISMS ANALYSIS")
        print("-"*60)

        if 'memory' in analysis:
            mem = analysis['memory']
            print(f"Memory Utilization: {mem['utilization']} ({mem['active_slots']}/{mem['total_slots']} slots)")

        if 'reasoning' in analysis:
            reas = analysis['reasoning']
            print(f"Reasoning Diversity: {reas['diversity_score']} (std={reas['std_activation']})")

        if 'routing' in analysis:
            rout = analysis['routing']
            print(f"Routing Balance: Slow={rout['avg_slow_preference']}, Fast={rout['avg_fast_preference']}")

        print("-"*60)

    def batch_mode(self, input_file: str, output_file: str):
        """Process batch of inputs from JSON file."""

        print(f"\nProcessing batch from: {input_file}")

        with open(input_file, 'r', encoding='utf-8') as f:
            inputs = json.load(f)

        results = []

        for i, item in enumerate(inputs):
            prompt = item.get('prompt', '')
            settings = item.get('settings', {})

            print(f"\nProcessing {i+1}/{len(inputs)}: {prompt[:50]}...")

            result = self.generate(
                prompt=prompt,
                temperature=settings.get('temperature', 0.8),
                top_k=settings.get('top_k', 50),
                top_p=settings.get('top_p', 0.9),
                visualize_cognitive=settings.get('visualize_cognitive', False)
            )

            results.append({
                'prompt': prompt,
                'generated': result['generated_text'][0],
                'metadata': result['metadata'],
                'cognitive_analysis': result.get('cognitive_analysis', {})
            })

        # Save results
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        print(f"\nResults saved to: {output_file}")

def main():
    parser = argparse.ArgumentParser(description="HIM Inference Engine")

    parser.add_argument(
        '--model_path',
        type=str,
        default='model_artifacts/nl_direct_response_model_best.pth',
        help='Path to saved model checkpoint'
    )
    parser.add_argument(
        '--mode',
        type=str,
        default='interactive',
        choices=['interactive', 'batch'],
        help='Inference mode'
    )
    parser.add_argument(
        '--input_file',
        type=str,
        help='Input JSON file for batch mode'
    )
    parser.add_argument(
        '--output_file',
        type=str,
        help='Output JSON file for batch mode'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='auto',
        choices=['auto', 'cuda', 'cpu'],
        help='Device to use'
    )

    args = parser.parse_args()

    # Initialize inference engine
    engine = AdvancedInference(
        model_path=args.model_path,
        device=args.device
    )

    # Run mode
    if args.mode == 'interactive':
        engine.interactive_mode()
    elif args.mode == 'batch':
        if not args.input_file or not args.output_file:
            print("Error: --input_file and --output_file required for batch mode")
            return
        engine.batch_mode(args.input_file, args.output_file)

if __name__ == '__main__':
    main()
