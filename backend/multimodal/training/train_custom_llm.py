"""
Executable Training Script for Custom Grok-Level LLM
Build your own large language model from scratch or fine-tune open-source models
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from datasets import load_dataset
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType
)
import bitsandbytes as bnb
from pathlib import Path
from tqdm import tqdm
import wandb


class CustomLLMConfig:
    """Configuration for custom LLM training"""

    # Base Model Options (All FREE):
    # - "meta-llama/Llama-3.1-8B" (FREE - Meta)
    # - "mistralai/Mistral-7B-v0.3" (FREE - Mistral AI)
    # - "microsoft/Phi-3-mini-4k-instruct" (FREE - Microsoft)
    # - "Qwen/Qwen2.5-7B" (FREE - Alibaba)

    BASE_MODEL = "mistralai/Mistral-7B-v0.3"  # FREE 7B model

    # Training Configuration
    OUTPUT_DIR = "./models/custom_grok_llm"
    DATASET = "openwebtext"  # FREE dataset
    MAX_LENGTH = 2048
    BATCH_SIZE = 4
    GRADIENT_ACCUMULATION_STEPS = 8
    LEARNING_RATE = 2e-5
    NUM_EPOCHS = 3
    WARMUP_STEPS = 100
    SAVE_STEPS = 500
    LOGGING_STEPS = 10

    # LoRA Configuration (Efficient Fine-tuning)
    USE_LORA = True
    LORA_R = 16
    LORA_ALPHA = 32
    LORA_DROPOUT = 0.05

    # 8-bit/4-bit Quantization (Save memory)
    USE_8BIT = True  # Can train on consumer GPUs

    # Advanced Features
    USE_FLASH_ATTENTION = True  # Faster training
    USE_GRADIENT_CHECKPOINTING = True  # Save memory


class TextDataset(Dataset):
    """Dataset for training LLM"""

    def __init__(self, dataset_name: str, tokenizer, max_length: int = 2048):
        self.tokenizer = tokenizer
        self.max_length = max_length

        print(f"Loading dataset: {dataset_name}")
        # Load free dataset
        if dataset_name == "openwebtext":
            self.dataset = load_dataset("openwebtext", split="train", streaming=False)
        elif dataset_name == "pile":
            self.dataset = load_dataset("EleutherAI/pile", split="train", streaming=True)
        else:
            # Custom dataset
            self.dataset = load_dataset(dataset_name, split="train")

        print(f"Dataset loaded: {len(self.dataset)} examples")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        text = item['text']

        # Tokenize
        encodings = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )

        return {
            'input_ids': encodings['input_ids'].squeeze(),
            'attention_mask': encodings['attention_mask'].squeeze(),
            'labels': encodings['input_ids'].squeeze()
        }


class CustomLLMTrainer:
    """Trainer for custom Grok-level LLM"""

    def __init__(self, config: CustomLLMConfig):
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Initialize Weights & Biases
        wandb.init(
            project="custom-grok-llm",
            config=vars(config)
        )

    def load_model(self):
        """Load base model and prepare for training"""
        print(f"Loading base model: {self.config.BASE_MODEL}")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.BASE_MODEL,
            trust_remote_code=True
        )

        # Set padding token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load model with 8-bit quantization
        if self.config.USE_8BIT:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.BASE_MODEL,
                load_in_8bit=True,
                device_map="auto",
                torch_dtype=torch.float16,
                trust_remote_code=True
            )
            self.model = prepare_model_for_kbit_training(self.model)
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.BASE_MODEL,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True
            )

        # Enable gradient checkpointing
        if self.config.USE_GRADIENT_CHECKPOINTING:
            self.model.gradient_checkpointing_enable()

        print(f"✓ Model loaded: {self.model.num_parameters():,} parameters")

        # Apply LoRA for efficient fine-tuning
        if self.config.USE_LORA:
            lora_config = LoraConfig(
                r=self.config.LORA_R,
                lora_alpha=self.config.LORA_ALPHA,
                lora_dropout=self.config.LORA_DROPOUT,
                bias="none",
                task_type=TaskType.CAUSAL_LM,
                target_modules=["q_proj", "v_proj", "k_proj", "o_proj"]
            )

            self.model = get_peft_model(self.model, lora_config)
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            print(f"✓ LoRA applied: {trainable_params:,} trainable parameters")

    def prepare_dataset(self):
        """Prepare training dataset"""
        print("Preparing dataset...")

        self.train_dataset = TextDataset(
            dataset_name=self.config.DATASET,
            tokenizer=self.tokenizer,
            max_length=self.config.MAX_LENGTH
        )

        print(f"✓ Dataset ready: {len(self.train_dataset)} examples")

    def train(self):
        """Train the model"""
        print("\n" + "=" * 60)
        print("STARTING CUSTOM GROK-LEVEL LLM TRAINING")
        print("=" * 60)

        # Training arguments
        training_args = TrainingArguments(
            output_dir=self.config.OUTPUT_DIR,
            num_train_epochs=self.config.NUM_EPOCHS,
            per_device_train_batch_size=self.config.BATCH_SIZE,
            gradient_accumulation_steps=self.config.GRADIENT_ACCUMULATION_STEPS,
            learning_rate=self.config.LEARNING_RATE,
            fp16=True if self.device == "cuda" else False,
            logging_steps=self.config.LOGGING_STEPS,
            save_steps=self.config.SAVE_STEPS,
            warmup_steps=self.config.WARMUP_STEPS,
            save_total_limit=3,
            report_to="wandb",
            logging_dir=f"{self.config.OUTPUT_DIR}/logs",
            optim="adamw_8bit" if self.config.USE_8BIT else "adamw_torch",
            gradient_checkpointing=self.config.USE_GRADIENT_CHECKPOINTING
        )

        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False  # Causal LM (not masked LM)
        )

        # Initialize trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            data_collator=data_collator
        )

        # Start training
        print("\nStarting training...")
        trainer.train()

        print("\n✅ Training complete!")

        # Save final model
        self.save_model("final")

    def save_model(self, checkpoint_name: str):
        """Save model checkpoint"""
        output_dir = Path(self.config.OUTPUT_DIR) / checkpoint_name

        if self.config.USE_LORA:
            # Save LoRA adapters
            self.model.save_pretrained(output_dir)
        else:
            # Save full model
            self.model.save_pretrained(output_dir)

        self.tokenizer.save_pretrained(output_dir)
        print(f"✓ Model saved to {output_dir}")

    def test_generation(self):
        """Test the trained model"""
        print("\n" + "=" * 60)
        print("TESTING TRAINED MODEL")
        print("=" * 60)

        self.model.eval()

        test_prompts = [
            "What is artificial intelligence?",
            "Explain quantum computing in simple terms:",
            "Write a short story about a robot:",
        ]

        for prompt in test_prompts:
            print(f"\nPrompt: {prompt}")

            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=200,
                    temperature=0.7,
                    top_p=0.9,
                    do_sample=True
                )

            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            print(f"Response: {response}")
            print("-" * 60)


def main():
    """Main execution"""

    # Configuration
    config = CustomLLMConfig()

    print("\n" + "=" * 80)
    print("CUSTOM GROK-LEVEL LLM TRAINING")
    print("=" * 80)
    print(f"Base Model: {config.BASE_MODEL} (FREE)")
    print(f"Dataset: {config.DATASET} (FREE)")
    print(f"Training Method: {'LoRA Fine-tuning' if config.USE_LORA else 'Full Fine-tuning'}")
    print(f"Quantization: {'8-bit' if config.USE_8BIT else 'FP16'}")
    print(f"Max Length: {config.MAX_LENGTH} tokens")
    print(f"Batch Size: {config.BATCH_SIZE} x {config.GRADIENT_ACCUMULATION_STEPS} accumulation")
    print(f"Epochs: {config.NUM_EPOCHS}")
    print("=" * 80 + "\n")

    # Create trainer
    trainer = CustomLLMTrainer(config)

    # Load model
    trainer.load_model()

    # Prepare dataset
    trainer.prepare_dataset()

    # Train
    trainer.train()

    # Test
    trainer.test_generation()

    print("\n✅ ALL DONE! Your custom Grok-level LLM is ready!")
    print(f"Model saved to: {config.OUTPUT_DIR}")


if __name__ == "__main__":
    main()
