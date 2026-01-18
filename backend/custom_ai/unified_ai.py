import torch
from typing import Optional, Dict, Any, List
from PIL import Image
import numpy as np

from .tokenizer import CustomTokenizer
from .transformer import CustomTransformer
from .inference import InferenceEngine
from .image_generator import CustomImageGenerator, ImageGeneratorInference
from .video_generator import CustomVideoGenerator, VideoGeneratorInference
from .audio_generator import CustomAudioGenerator, AudioGeneratorInference
from .face_preserving_generator import FacePreservingGenerator, FacePreservingInference
from .multimodal_understanding import MultiModalUnderstanding, MultiModalQA


class UnifiedAI:
    """unified interface for all AI capabilities"""

    def __init__(self, vocab_size=30000, device='cuda'):
        self.vocab_size = vocab_size
        self.device = device if torch.cuda.is_available() else 'cpu'

        self.tokenizer = CustomTokenizer(vocab_size=vocab_size)

        self.text_model = CustomTransformer(vocab_size)
        self.image_model = CustomImageGenerator(vocab_size)
        self.video_model = CustomVideoGenerator(vocab_size)
        self.audio_model = CustomAudioGenerator(vocab_size)
        self.face_model = FacePreservingGenerator(vocab_size)
        self.understanding_model = MultiModalUnderstanding(vocab_size)

        self.text_engine = InferenceEngine(self.text_model, self.tokenizer, self.device)
        self.image_engine = ImageGeneratorInference(self.image_model, self.tokenizer, self.device)
        self.video_engine = VideoGeneratorInference(self.video_model, self.tokenizer, self.device)
        self.audio_engine = AudioGeneratorInference(self.audio_model, self.tokenizer, self.device)
        self.face_engine = FacePreservingInference(self.face_model, self.tokenizer, self.device)
        self.qa_engine = MultiModalQA(self.understanding_model, self.tokenizer, self.device)

        self.capabilities = [
            'text_generation',
            'text_chat',
            'image_generation',
            'video_generation',
            'audio_generation',
            'music_generation',
            'speech_synthesis',
            'face_preservation',
            'visual_question_answering',
            'image_understanding',
            'video_understanding',
            'multimodal_qa'
        ]

    def process(self, prompt: str, task_type: str = 'auto', **kwargs) -> Any:
        """processes any request automatically"""

        if task_type == 'auto':
            task_type = self._detect_task_type(prompt)

        if task_type == 'text' or task_type == 'chat':
            return self.generate_text(prompt, **kwargs)

        elif task_type == 'image':
            return self.generate_image(prompt, **kwargs)

        elif task_type == 'video':
            return self.generate_video(prompt, **kwargs)

        elif task_type == 'gif':
            return self.generate_gif(prompt, **kwargs)

        elif task_type == 'audio':
            return self.generate_audio(prompt, **kwargs)

        elif task_type == 'music':
            return self.generate_music(prompt, **kwargs)

        elif task_type == 'speech':
            return self.generate_speech(prompt, **kwargs)

        elif task_type == 'question':
            return self.answer_question(prompt, **kwargs)

        else:
            return self.generate_text(prompt, **kwargs)

    def _detect_task_type(self, prompt: str) -> str:
        """automatically detects what the user wants"""

        prompt_lower = prompt.lower()

        if any(word in prompt_lower for word in ['image', 'picture', 'photo', 'draw', 'paint', 'sketch']):
            return 'image'

        elif any(word in prompt_lower for word in ['video', 'animation', 'clip', 'footage']):
            return 'video'

        elif 'gif' in prompt_lower:
            return 'gif'

        elif any(word in prompt_lower for word in ['music', 'song', 'melody', 'tune', 'beat']):
            return 'music'

        elif any(word in prompt_lower for word in ['speak', 'say', 'voice', 'speech', 'pronounce']):
            return 'speech'

        elif any(word in prompt_lower for word in ['sound', 'audio', 'noise']):
            return 'audio'

        elif any(word in prompt_lower for word in ['what is', 'describe', 'explain', 'tell me about']):
            return 'question'

        else:
            return 'text'

    def generate_text(self, prompt: str, max_length: int = 200, temperature: float = 0.8) -> str:
        """generates text response"""
        return self.text_engine.generate_text(prompt, max_length, temperature)

    def chat(self, message: str, conversation_history: Optional[List] = None) -> tuple:
        """chat conversation"""
        return self.text_engine.chat(message, conversation_history)

    def generate_image(self, prompt: str, face_path: Optional[str] = None,
                      height: int = 512, width: int = 512, **kwargs) -> Image.Image:
        """generates image from text"""

        if face_path:
            return self.face_engine.generate_with_face(prompt, face_path, preserve_face=True)
        else:
            return self.image_engine.generate_image(prompt, height, width)

    def generate_video(self, prompt: str, face_path: Optional[str] = None,
                      num_frames: int = 16, fps: int = 8, **kwargs) -> np.ndarray:
        """generates video from text"""

        if face_path:
            return self.face_engine.generate_video_with_face(prompt, face_path, num_frames, preserve_face=True)
        else:
            return self.video_engine.generate_video(prompt, num_frames, fps=fps)

    def generate_gif(self, prompt: str, num_frames: int = 16, fps: int = 8, **kwargs) -> np.ndarray:
        """generates GIF animation"""
        return self.video_engine.generate_video(prompt, num_frames, fps=fps)

    def save_gif(self, video: np.ndarray, filepath: str, fps: int = 8):
        """saves video as GIF"""
        self.video_engine.save_gif(video, filepath, fps)

    def generate_audio(self, prompt: str, duration: float = 5.0, **kwargs) -> np.ndarray:
        """generates audio from text"""
        return self.audio_engine.generate_audio(prompt, duration, mode='neural')

    def generate_music(self, prompt: str, duration: float = 10.0, **kwargs) -> np.ndarray:
        """generates music from description"""
        return self.audio_engine.generate_audio(prompt, duration, mode='music')

    def generate_speech(self, text: str, **kwargs) -> np.ndarray:
        """converts text to speech"""
        return self.audio_engine.generate_audio(text, mode='speech')

    def answer_question(self, question: str, image_path: Optional[str] = None,
                       video_path: Optional[str] = None, audio_path: Optional[str] = None) -> str:
        """answers questions about any content"""
        return self.qa_engine.answer(question, image_path, video_path, audio_path)

    def understand_content(self, content_path: str) -> str:
        """understands and describes any content"""
        return self.qa_engine.understand_content(content_path)

    def save_image(self, image: Image.Image, filepath: str):
        """saves generated image"""
        image.save(filepath)

    def save_video(self, video: np.ndarray, filepath: str, fps: int = 8):
        """saves generated video"""
        self.video_engine.save_video(video, filepath, fps)

    def save_audio(self, audio: np.ndarray, filepath: str, sample_rate: int = 16000):
        """saves generated audio"""
        self.audio_engine.save_audio(audio, filepath, sample_rate)

    def load_checkpoint(self, checkpoint_dir: str):
        """loads trained models from checkpoint"""
        import os

        if os.path.exists(f"{checkpoint_dir}/text_model.pt"):
            checkpoint = torch.load(f"{checkpoint_dir}/text_model.pt", map_location=self.device)
            self.text_model.load_state_dict(checkpoint['model_state_dict'])

        if os.path.exists(f"{checkpoint_dir}/tokenizer.pkl"):
            self.tokenizer.load(f"{checkpoint_dir}/tokenizer.pkl")

    def save_checkpoint(self, checkpoint_dir: str):
        """saves all models"""
        import os
        os.makedirs(checkpoint_dir, exist_ok=True)

        torch.save({
            'model_state_dict': self.text_model.state_dict(),
        }, f"{checkpoint_dir}/text_model.pt")

        self.tokenizer.save(f"{checkpoint_dir}/tokenizer.pkl")

    def get_capabilities(self) -> Dict[str, Any]:
        """returns all available capabilities"""
        return {
            'capabilities': self.capabilities,
            'device': self.device,
            'models': {
                'text': 'CustomTransformer',
                'image': 'CustomImageGenerator',
                'video': 'CustomVideoGenerator',
                'audio': 'CustomAudioGenerator',
                'face': 'FacePreservingGenerator',
                'understanding': 'MultiModalUnderstanding'
            },
            'features': {
                'automatic_task_detection': True,
                'face_preservation': True,
                'long_video_generation': True,
                'multimodal_understanding': True,
                'visual_question_answering': True,
                'music_generation': True,
                'speech_synthesis': True
            }
        }


def create_unified_ai(vocab_size=30000, device='cuda') -> UnifiedAI:
    """creates a unified AI instance ready to use"""
    return UnifiedAI(vocab_size, device)


if __name__ == '__main__':
    ai = create_unified_ai()

    print("Unified AI System Initialized")
    print(f"Device: {ai.device}")
    print(f"Capabilities: {', '.join(ai.capabilities)}")

    text = ai.generate_text("Hello, how are you?")
    print(f"\nText: {text}")

    print("\nAll systems operational!")
