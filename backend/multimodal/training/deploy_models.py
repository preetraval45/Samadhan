"""
Deploy Trained Custom Models
Integrate trained models into the backend for inference
"""

import torch
import shutil
from pathlib import Path
from loguru import logger
import json


class ModelDeployment:
    """
    Deploy trained models to production
    """

    def __init__(
        self,
        trained_models_dir: str = "./trained_models",
        deployment_dir: str = "../models/production"
    ):
        self.trained_models_dir = Path(trained_models_dir)
        self.deployment_dir = Path(deployment_dir)
        self.deployment_dir.mkdir(parents=True, exist_ok=True)

    def deploy_chat_model(self, model_path: str, model_name: str = "custom_chat"):
        """
        Deploy custom chat model
        """
        logger.info(f"Deploying chat model: {model_name}")

        source = Path(model_path)
        dest = self.deployment_dir / model_name

        # Copy model files
        if dest.exists():
            shutil.rmtree(dest)

        shutil.copytree(source, dest)

        # Create model config
        config = {
            "model_name": model_name,
            "type": "chat",
            "path": str(dest),
            "metadata": {
                "version": "1.0.0",
                "framework": "pytorch",
                "optimized": True
            }
        }

        # Save config
        config_path = dest / "config.json"
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)

        logger.info(f"Chat model deployed to {dest}")

        return dest

    def deploy_image_generation_model(
        self,
        model_path: str,
        model_name: str = "custom_image_gen"
    ):
        """
        Deploy custom image generation model
        """
        logger.info(f"Deploying image generation model: {model_name}")

        source = Path(model_path)
        dest = self.deployment_dir / model_name

        # Copy model files
        if dest.exists():
            shutil.rmtree(dest)

        shutil.copytree(source, dest)

        # Create model config
        config = {
            "model_name": model_name,
            "type": "image_generation",
            "path": str(dest),
            "metadata": {
                "version": "1.0.0",
                "framework": "pytorch",
                "image_size": 64  # or whatever size the model was trained on
            }
        }

        # Save config
        config_path = dest / "config.json"
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)

        logger.info(f"Image generation model deployed to {dest}")

        return dest

    def create_model_registry(self):
        """
        Create a registry of all deployed models
        """
        registry = {
            "models": [],
            "last_updated": str(Path.ctime(self.deployment_dir))
        }

        # Scan deployment directory
        for model_dir in self.deployment_dir.iterdir():
            if model_dir.is_dir():
                config_path = model_dir / "config.json"
                if config_path.exists():
                    with open(config_path) as f:
                        config = json.load(f)
                    registry["models"].append(config)

        # Save registry
        registry_path = self.deployment_dir / "model_registry.json"
        with open(registry_path, "w") as f:
            json.dump(registry, f, indent=2)

        logger.info(f"Model registry created with {len(registry['models'])} models")

        return registry

    def deploy_all(self):
        """
        Deploy all trained models
        """
        logger.info("Deploying all trained models...")

        deployed = []

        # Find and deploy chat models
        chat_models = list(self.trained_models_dir.glob("chat_*"))
        for model_path in chat_models:
            try:
                dest = self.deploy_chat_model(
                    str(model_path),
                    model_name=model_path.name
                )
                deployed.append(str(dest))
            except Exception as e:
                logger.error(f"Failed to deploy {model_path}: {e}")

        # Find and deploy image generation models
        image_models = list(self.trained_models_dir.glob("image_gen_*"))
        for model_path in image_models:
            try:
                dest = self.deploy_image_generation_model(
                    str(model_path),
                    model_name=model_path.name
                )
                deployed.append(str(dest))
            except Exception as e:
                logger.error(f"Failed to deploy {model_path}: {e}")

        # Create registry
        self.create_model_registry()

        logger.info(f"Deployed {len(deployed)} models successfully")

        return deployed


if __name__ == "__main__":
    deployer = ModelDeployment()

    # Deploy all models
    deployed = deployer.deploy_all()

    logger.info("Deployment complete!")
    logger.info(f"Deployed models: {json.dumps(deployed, indent=2)}")
