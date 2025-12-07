import argparse
import logging
import os
import json
from typing import Iterator, List, Optional

from datasets import load_dataset
from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers, processors
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

class TokenizerTrainer:
    """
    A robust trainer for Byte-Level BPE Tokenizers (RoBERTa style).
    Features:
    - Byte-Level BPE (Handles unicode/emojis, no <unk>)
    - Streaming dataset loading (Memory efficient)
    - Full HuggingFace compatibility
    """
    def __init__(self, vocab_size: int = 30000, min_frequency: int = 2, dataset_name: str = "wikitext", config_name: str = "wikitext-103-v1"):
        self.vocab_size = vocab_size
        self.min_frequency = min_frequency
        self.dataset_name = dataset_name
        self.config_name = config_name
        # RoBERTa special tokens
        self.special_tokens = ["<s>", "<pad>", "</s>", "<unk>", "<mask>"] 

    def get_training_corpus(self) -> Iterator[str]:
        """Yields text batches from the dataset in a streaming fashion."""
        logger.info(f"Loading dataset: {self.dataset_name} ({self.config_name})")
        try:
            dataset = load_dataset(self.dataset_name, self.config_name, split="train", streaming=True)
            for i, example in enumerate(dataset):
                text = example['text'].strip()
                if text:
                    yield text
                if i % 50000 == 0 and i > 0:
                    logger.info(f"Processed {i} examples...")
        except Exception as e:
            logger.error(f"Error loading dataset: {e}")
            raise

    def train(self, output_dir: str):
        logger.info("Initializing Byte-Level BPE Tokenizer...")
        # GPT-2/RoBERTa style tokenizer
        # dropout=None for deterministic training
        tokenizer = Tokenizer(models.BPE(dropout=None, unk_token="<unk>"))
        
        # Pre-tokenizer: Split on whitespace and punctuation, but keep bytes
        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)
        
        # Decoder: Convert bytes back to string
        tokenizer.decoder = decoders.ByteLevel()
        
        trainer = trainers.BpeTrainer(
            vocab_size=self.vocab_size,
            min_frequency=self.min_frequency,
            show_progress=True,
            special_tokens=self.special_tokens,
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet()
        )

        logger.info(f"Starting training (Vocab Size: {self.vocab_size})...")
        tokenizer.train_from_iterator(self.get_training_corpus(), trainer=trainer)
        
        # Post-processing: <s> $A </s>
        # RoBERTa uses <s> for CLS and </s> for SEP
        cls_token_id = tokenizer.token_to_id("<s>")
        sep_token_id = tokenizer.token_to_id("</s>")
        
        tokenizer.post_processor = processors.TemplateProcessing(
            single="<s> $A </s>",
            pair="<s> $A </s> $B:1 </s>:1",
            special_tokens=[
                ("<s>", cls_token_id),
                ("</s>", sep_token_id),
            ],
        )
        
        self.save(tokenizer, output_dir)
        self.verify(tokenizer)

    def save(self, tokenizer: Tokenizer, output_dir: str):
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Saving tokenizer to {output_dir}")
        tokenizer.save(os.path.join(output_dir, "tokenizer.json"))
        
        # Save config for HuggingFace compatibility (RoBERTa)
        # This ensures AutoTokenizer.from_pretrained works correctly
        config = {
            "architectures": ["RobertaForMaskedLM"],
            "model_type": "roberta",
            "vocab_size": self.vocab_size,
            "bos_token_id": tokenizer.token_to_id("<s>"),
            "eos_token_id": tokenizer.token_to_id("</s>"),
            "pad_token_id": tokenizer.token_to_id("<pad>"),
            "mask_token_id": tokenizer.token_to_id("<mask>"),
            "unk_token_id": tokenizer.token_to_id("<unk>"),
            "max_position_embeddings": 514,
            "type_vocab_size": 1
        }
        
        with open(os.path.join(output_dir, "config.json"), "w") as f:
            json.dump(config, f, indent=4)
            
        # tokenizer_config.json
        tokenizer_config = {
            "do_lower_case": False, # BPE is case-sensitive usually
            "model_max_length": 512,
            "bos_token": "<s>",
            "eos_token": "</s>",
            "unk_token": "<unk>",
            "pad_token": "<pad>",
            "mask_token": "<mask>"
        }
        with open(os.path.join(output_dir, "tokenizer_config.json"), "w") as f:
            json.dump(tokenizer_config, f, indent=4)
            
        # special_tokens_map.json
        special_tokens_map = {
            "bos_token": "<s>",
            "eos_token": "</s>",
            "unk_token": "<unk>",
            "pad_token": "<pad>",
            "mask_token": "<mask>"
        }
        with open(os.path.join(output_dir, "special_tokens_map.json"), "w") as f:
            json.dump(special_tokens_map, f, indent=4)

        logger.info("✅ Tokenizer artifacts saved successfully.")

    def verify(self, tokenizer: Tokenizer):
        test_str = "Hello world! 🌍 This is a test of the robust tokenizer."
        encoded = tokenizer.encode(test_str)
        decoded = tokenizer.decode(encoded.ids)
        logger.info(f"--- Verification ---")
        logger.info(f"Input:    '{test_str}'")
        logger.info(f"Tokens:   {encoded.tokens}")
        logger.info(f"IDs:      {encoded.ids}")
        logger.info(f"Decoded:  '{decoded}'")
        logger.info(f"--------------------")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a robust Byte-Level BPE Tokenizer.")
    parser.add_argument("--vocab_size", type=int, default=30000, help="Vocabulary size (default: 30000 for efficient 7M param model).")
    parser.add_argument("--dataset_name", type=str, default="wikitext", help="Hugging Face dataset name.")
    parser.add_argument("--config_name", type=str, default="wikitext-103-v1", help="Dataset configuration name.")
    parser.add_argument("--output_dir", type=str, default="model_artifacts/tokenizer", help="Output directory.")
    
    args = parser.parse_args()
    
    trainer = TokenizerTrainer(
        vocab_size=args.vocab_size, 
        dataset_name=args.dataset_name, 
        config_name=args.config_name
    )
    trainer.train(args.output_dir)
