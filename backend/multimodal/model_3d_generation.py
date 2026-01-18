"""
3D Model Generation Module
Supports text-to-3D using FREE open-source models
"""

from typing import Optional, Dict, Any
import torch
import numpy as np
from pathlib import Path
import base64
import trimesh


class Model3DGenerator:
    """
    3D model generation using free models
    """

    def __init__(self, device: str = "cuda"):
        self.device = device
        self.shap_e_model = None
        self.initialized = False

    async def initialize(self):
        """Load 3D generation models"""
        print("Initializing 3D generation models...")

        # Shap-E (FREE - OpenAI open-source)
        try:
            from shap_e.diffusion.sample import sample_latents
            from shap_e.diffusion.gaussian_diffusion import diffusion_from_config
            from shap_e.models.download import load_model, load_config
            from shap_e.util.notebooks import decode_latent_mesh

            self.shap_e_model = {
                'model': load_model('transmitter', device=self.device),
                'diffusion': diffusion_from_config(load_config('diffusion')),
                'decoder': decode_latent_mesh
            }
            print("✓ 3D generation model loaded (Shap-E)")
        except Exception as e:
            print(f"⚠ 3D generation model failed: {e}")

        self.initialized = True
        print("✅ 3D generation models initialized")

    async def generate_3d_model(
        self,
        prompt: str,
        output_format: str = "glb",
        guidance_scale: float = 15.0
    ) -> Dict[str, Any]:
        """
        Generate 3D model from text

        Args:
            prompt: Text description
            output_format: 'glb', 'obj', 'ply', 'stl'
            guidance_scale: Prompt adherence
        """
        if not self.initialized:
            await self.initialize()

        if self.shap_e_model is None:
            raise RuntimeError("3D generation model not available")

        # Generate latent
        from shap_e.diffusion.sample import sample_latents

        latents = sample_latents(
            batch_size=1,
            model=self.shap_e_model['model'],
            diffusion=self.shap_e_model['diffusion'],
            guidance_scale=guidance_scale,
            model_kwargs=dict(texts=[prompt]),
            progress=True,
            clip_denoised=True,
            use_fp16=True if self.device == "cuda" else False,
            device=self.device
        )

        # Decode to mesh
        mesh = self.shap_e_model['decoder'](
            self.shap_e_model['model'],
            latents[0],
            device=self.device
        )

        # Save mesh
        output_path = Path("./temp") / f"model_3d_{id(self)}.{output_format}"
        output_path.parent.mkdir(exist_ok=True)

        # Convert to trimesh and export
        vertices = mesh.verts.cpu().numpy()
        faces = mesh.faces.cpu().numpy()

        tri_mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
        tri_mesh.export(str(output_path))

        # Read and encode
        with open(output_path, 'rb') as f:
            model_bytes = f.read()
            model_base64 = base64.b64encode(model_bytes).decode()

        return {
            "model": model_base64,
            "format": output_format,
            "prompt": prompt,
            "vertex_count": len(vertices),
            "face_count": len(faces)
        }


# Singleton
_model_3d_generator: Optional[Model3DGenerator] = None

async def get_model_3d_generator() -> Model3DGenerator:
    """Get or create 3D model generator instance"""
    global _model_3d_generator
    if _model_3d_generator is None:
        _model_3d_generator = Model3DGenerator()
        await _model_3d_generator.initialize()
    return _model_3d_generator
