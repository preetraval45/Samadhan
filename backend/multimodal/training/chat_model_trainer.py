"""
Custom Chat Model Training Pipeline
Train lightweight, fast custom models for conversational AI
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import BitsAndBytesConfig
import json
from pathlib import Path
from loguru import logger


class ChatModelTrainer:
    """
    Train custom lightweight chat models
    Uses LoRA for efficient fine-tuning
    """

    def __init__(
        self,
        base_model: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        output_dir: str = "./models/custom_chat"
    ):
        self.base_model = base_model
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Quantization config for memory efficiency
        self.bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )

        # LoRA config for efficient training
        self.lora_config = LoraConfig(
            r=16,  # Rank
            lora_alpha=32,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM"
        )

    def prepare_dataset(self, dataset_name: str = "Open-Orca/OpenOrca"):
        """
        Prepare training dataset
        Uses open-source conversational datasets
        """
        logger.info(f"Loading dataset: {dataset_name}")

        # Load dataset
        dataset = load_dataset(dataset_name, split="train", streaming=True)

        # Take subset for faster training
        dataset = dataset.take(10000)

        return dataset

    def create_custom_dataset(self, conversations: list):
        """
        Create custom dataset from your own conversations

        Format:
        [
            {"user": "Hello!", "assistant": "Hi! How can I help?"},
            {"user": "What is AI?", "assistant": "AI is..."}
        ]
        """
        formatted_data = []
        for conv in conversations:
            text = f"User: {conv['user']}\nAssistant: {conv['assistant']}"
            formatted_data.append({"text": text})

        return formatted_data

    def train(
        self,
        dataset=None,
        num_epochs: int = 3,
        batch_size: int = 4,
        learning_rate: float = 2e-4,
        max_steps: int = 1000
    ):
        """
        Train the model
        """
        logger.info(f"Starting training with base model: {self.base_model}")

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(self.base_model)
        tokenizer.pad_token = tokenizer.eos_token

        # Load model with quantization
        model = AutoModelForCausalLM.from_pretrained(
            self.base_model,
            quantization_config=self.bnb_config,
            device_map="auto",
            trust_remote_code=True
        )

        # Prepare for LoRA training
        model = prepare_model_for_kbit_training(model)
        model = get_peft_model(model, self.lora_config)

        logger.info(f"Trainable parameters: {model.print_trainable_parameters()}")

        # Prepare dataset
        if dataset is None:
            dataset = self.prepare_dataset()

        # Training arguments
        training_args = TrainingArguments(
            output_dir=str(self.output_dir),
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=4,
            learning_rate=learning_rate,
            fp16=True,
            save_steps=100,
            logging_steps=10,
            max_steps=max_steps,
            save_total_limit=2,
            warmup_steps=100,
            optim="paged_adamw_8bit"
        )

        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False
        )

        # Trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=dataset,
            data_collator=data_collator
        )

        # Train
        logger.info("Training started...")
        trainer.train()

        # Save model
        logger.info(f"Saving model to {self.output_dir}")
        model.save_pretrained(self.output_dir)
        tokenizer.save_pretrained(self.output_dir)

        logger.info("Training complete!")

        return model, tokenizer

    def export_for_inference(self, quantize: bool = True):
        """
        Export model for fast inference
        Applies quantization and optimization
        """
        from torch.quantization import quantize_dynamic

        logger.info("Exporting model for inference...")

        model = AutoModelForCausalLM.from_pretrained(
            str(self.output_dir),
            trust_remote_code=True
        )

        if quantize:
            # Dynamic quantization for CPU inference
            model = quantize_dynamic(
                model,
                {nn.Linear},
                dtype=torch.qint8
            )

        # Save optimized model
        optimized_dir = self.output_dir / "optimized"
        optimized_dir.mkdir(exist_ok=True)

        torch.save(model.state_dict(), optimized_dir / "model.pt")

        logger.info(f"Optimized model saved to {optimized_dir}")

        return model


class FastChatModel(nn.Module):
    """
    Ultra-lightweight custom chat model
    Built from scratch for maximum speed
    """

    def __init__(
        self,
        vocab_size: int = 32000,
        embedding_dim: int = 256,
        hidden_dim: int = 512,
        num_layers: int = 4,
        num_heads: int = 8
    ):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        # Lightweight transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            batch_first=True,
            norm_first=True
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        self.output = nn.Linear(embedding_dim, vocab_size)

    def forward(self, input_ids, attention_mask=None):
        # Embed tokens
        x = self.embedding(input_ids)

        # Transform
        x = self.transformer(x, src_key_padding_mask=attention_mask)

        # Output logits
        logits = self.output(x)

        return logits

    @staticmethod
    def train_from_scratch(
        dataset,
        vocab_size: int = 32000,
        num_epochs: int = 10,
        batch_size: int = 32,
        output_dir: str = "./models/fast_chat"
    ):
        """
        Train model from scratch on custom data
        """
        logger.info("Training FastChatModel from scratch...")

        # Create model
        model = FastChatModel(vocab_size=vocab_size)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)

        # Optimizer
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()

        # Training loop
        model.train()
        for epoch in range(num_epochs):
            total_loss = 0

            for batch in DataLoader(dataset, batch_size=batch_size):
                input_ids = batch["input_ids"].to(device)
                labels = batch["labels"].to(device)

                # Forward pass
                logits = model(input_ids)

                # Compute loss
                loss = criterion(
                    logits.view(-1, vocab_size),
                    labels.view(-1)
                )

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            avg_loss = total_loss / len(dataset)
            logger.info(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}")

        # Save model
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), output_path / "model.pt")

        logger.info(f"Model saved to {output_dir}")

        return model


# Training configurations for different use cases
TRAINING_CONFIGS = {
    "ultra_fast": {
        "base_model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "max_steps": 500,
        "batch_size": 8,
        "description": "Fastest training, good for testing"
    },
    "balanced": {
        "base_model": "microsoft/phi-2",
        "max_steps": 2000,
        "batch_size": 4,
        "description": "Balance between speed and quality"
    },
    "high_quality": {
        "base_model": "mistralai/Mistral-7B-Instruct-v0.2",
        "max_steps": 5000,
        "batch_size": 2,
        "description": "Best quality, slower training"
    }
}


if __name__ == "__main__":
    # Example: Train a custom chat model
    trainer = ChatModelTrainer()

    # Option 1: Train on public dataset
    # trainer.train(max_steps=100)

    # Option 2: Train on custom conversations
    custom_conversations = [
        {"user": "Hello!", "assistant": "Hi! How can I help you today?"},
        {"user": "What's the weather?", "assistant": "I don't have real-time weather data, but I can help you find weather information!"},
        # Add more conversations here
    ]

    dataset = trainer.create_custom_dataset(custom_conversations)
    # trainer.train(dataset=dataset, max_steps=100)

    # Export for fast inference
    # trainer.export_for_inference(quantize=True)

    logger.info("Training setup complete!")
