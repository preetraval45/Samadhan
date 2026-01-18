"""
Unified Model Training Interface
Train all custom models with simple commands
"""

import argparse
import torch
from pathlib import Path
from loguru import logger
from chat_model_trainer import ChatModelTrainer, FastChatModel, TRAINING_CONFIGS
from multimodal_trainer import ImageGenerationTrainer, FastGANTrainer, MULTIMODAL_CONFIGS
import json


class ModelTrainingPipeline:
    """
    Complete training pipeline for all custom models
    """

    def __init__(self, output_base_dir: str = "./trained_models"):
        self.output_base_dir = Path(output_base_dir)
        self.output_base_dir.mkdir(parents=True, exist_ok=True)

    def train_chat_model(
        self,
        config_name: str = "ultra_fast",
        custom_data_path: str = None
    ):
        """
        Train custom chat model

        Args:
            config_name: 'ultra_fast', 'balanced', or 'high_quality'
            custom_data_path: Path to custom conversation data (JSON)
        """
        logger.info(f"Training chat model with config: {config_name}")

        config = TRAINING_CONFIGS.get(config_name, TRAINING_CONFIGS["ultra_fast"])
        output_dir = self.output_base_dir / f"chat_{config_name}"

        trainer = ChatModelTrainer(
            base_model=config["base_model"],
            output_dir=str(output_dir)
        )

        # Load custom data if provided
        dataset = None
        if custom_data_path:
            logger.info(f"Loading custom data from {custom_data_path}")
            with open(custom_data_path) as f:
                conversations = json.load(f)
            dataset = trainer.create_custom_dataset(conversations)

        # Train
        model, tokenizer = trainer.train(
            dataset=dataset,
            max_steps=config["max_steps"],
            batch_size=config["batch_size"]
        )

        # Export for inference
        trainer.export_for_inference(quantize=True)

        logger.info(f"Chat model training complete! Saved to {output_dir}")

        return output_dir

    def train_from_scratch_chat(
        self,
        vocab_size: int = 32000,
        num_epochs: int = 10
    ):
        """
        Train ultra-lightweight chat model from scratch
        Fastest option - trains in minutes
        """
        logger.info("Training FastChatModel from scratch...")

        output_dir = self.output_base_dir / "fast_chat_scratch"

        # Create dummy dataset for demonstration
        # In practice, you'd load real conversation data
        class DummyDataset:
            def __init__(self, size=1000):
                self.size = size

            def __len__(self):
                return self.size

            def __getitem__(self, idx):
                return {
                    "input_ids": torch.randint(0, vocab_size, (50,)),
                    "labels": torch.randint(0, vocab_size, (50,))
                }

        dataset = DummyDataset()

        # Train
        model = FastChatModel.train_from_scratch(
            dataset=dataset,
            vocab_size=vocab_size,
            num_epochs=num_epochs,
            output_dir=str(output_dir)
        )

        logger.info(f"From-scratch model saved to {output_dir}")

        return output_dir

    def train_image_generation(
        self,
        image_folder: str,
        config_name: str = "image_gen_fast"
    ):
        """
        Train custom image generation model

        Args:
            image_folder: Path to folder containing training images
            config_name: Training configuration
        """
        logger.info(f"Training image generation model: {config_name}")

        config = MULTIMODAL_CONFIGS.get(config_name)
        output_dir = self.output_base_dir / f"image_gen_{config_name}"

        if config["type"] == "diffusion":
            trainer = ImageGenerationTrainer(output_dir=str(output_dir))
            model, scheduler = trainer.train(
                image_folder=image_folder,
                num_epochs=config["epochs"]
            )
        elif config["type"] == "gan":
            trainer = FastGANTrainer(img_size=config["image_size"])
            model = trainer.train(
                image_folder=image_folder,
                num_epochs=config["epochs"],
                output_dir=str(output_dir)
            )

        logger.info(f"Image generation model saved to {output_dir}")

        return output_dir

    def quick_train_all(self, image_folder: str = None):
        """
        Quick training of all models (for testing/demonstration)
        Uses fastest settings
        """
        logger.info("Starting quick training of all models...")

        results = {}

        # 1. Train chat model (from scratch - fastest)
        logger.info("\n=== Training Chat Model ===")
        results["chat"] = self.train_from_scratch_chat(num_epochs=2)

        # 2. Train image generation (if image folder provided)
        if image_folder:
            logger.info("\n=== Training Image Generation ===")
            results["image_gen"] = self.train_image_generation(
                image_folder=image_folder,
                config_name="gan_ultra_fast"
            )

        logger.info("\n=== All Models Trained! ===")
        logger.info(f"Results: {json.dumps(results, indent=2)}")

        return results

    def create_sample_training_data(self):
        """
        Create sample training data for demonstration
        """
        # Sample conversations
        conversations = [
            {
                "user": "Hello! How are you?",
                "assistant": "Hi! I'm doing great, thanks for asking! How can I help you today?"
            },
            {
                "user": "What is machine learning?",
                "assistant": "Machine learning is a subset of AI where computers learn from data to make predictions or decisions without being explicitly programmed."
            },
            {
                "user": "Can you explain neural networks?",
                "assistant": "Neural networks are computing systems inspired by biological neural networks. They consist of interconnected nodes (neurons) that process and transmit information."
            },
            {
                "user": "Tell me about deep learning",
                "assistant": "Deep learning uses neural networks with multiple layers to learn hierarchical representations of data. It's particularly effective for complex tasks like image and speech recognition."
            },
            {
                "user": "What's the difference between AI and ML?",
                "assistant": "AI (Artificial Intelligence) is the broad concept of machines being able to carry out tasks intelligently. ML (Machine Learning) is a specific approach to achieving AI by learning from data."
            }
        ]

        # Save to file
        data_path = self.output_base_dir / "sample_conversations.json"
        with open(data_path, "w") as f:
            json.dump(conversations, f, indent=2)

        logger.info(f"Sample training data created at {data_path}")

        return str(data_path)


def main():
    parser = argparse.ArgumentParser(description="Train custom AI models")
    parser.add_argument(
        "--task",
        type=str,
        choices=["chat", "image", "all", "demo"],
        default="demo",
        help="What to train"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="ultra_fast",
        help="Training configuration"
    )
    parser.add_argument(
        "--image-folder",
        type=str,
        help="Path to image folder for image generation training"
    )
    parser.add_argument(
        "--custom-data",
        type=str,
        help="Path to custom conversation data (JSON)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./trained_models",
        help="Output directory for trained models"
    )

    args = parser.parse_args()

    # Initialize pipeline
    pipeline = ModelTrainingPipeline(output_base_dir=args.output_dir)

    # Execute training
    if args.task == "chat":
        pipeline.train_chat_model(
            config_name=args.config,
            custom_data_path=args.custom_data
        )

    elif args.task == "image":
        if not args.image_folder:
            logger.error("--image-folder required for image generation training")
            return

        pipeline.train_image_generation(
            image_folder=args.image_folder,
            config_name=args.config
        )

    elif args.task == "all":
        pipeline.quick_train_all(image_folder=args.image_folder)

    elif args.task == "demo":
        logger.info("Running demo training with sample data...")

        # Create sample data
        sample_data = pipeline.create_sample_training_data()

        # Train chat model with sample data
        pipeline.train_from_scratch_chat(num_epochs=2)

        logger.info("Demo training complete!")


if __name__ == "__main__":
    # Quick demo mode
    logger.info("Custom Model Training Pipeline")
    logger.info("=" * 50)

    pipeline = ModelTrainingPipeline()

    # Create sample data
    sample_data = pipeline.create_sample_training_data()

    logger.info("\nTo train models, run:")
    logger.info("  python train_models.py --task chat --config ultra_fast")
    logger.info("  python train_models.py --task image --image-folder ./images")
    logger.info("  python train_models.py --task all")
    logger.info("  python train_models.py --task demo")
