"""
Audio Generation Module
Supports TTS, voice cloning, music generation
All using FREE open-source models
"""

from typing import Optional, Dict, Any, List
import torch
import numpy as np
from pathlib import Path
import io
import base64
import soundfile as sf


class AudioGenerator:
    """
    Audio generation using free models
    """

    def __init__(self, device: str = "cuda"):
        self.device = device
        self.tts_model = None
        self.music_model = None
        self.initialized = False

    async def initialize(self):
        """Load audio generation models"""
        print("Initializing audio generation models...")

        # 1. Text-to-Speech (FREE - Coqui TTS)
        try:
            from TTS.api import TTS
            self.tts_model = TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC")
            print("✓ TTS model loaded (Coqui TTS)")
        except Exception as e:
            print(f"⚠ TTS model failed: {e}")

        self.initialized = True
        print("✅ Audio generation models initialized")

    async def text_to_speech(
        self,
        text: str,
        language: str = "en",
        speaker: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Convert text to speech

        Args:
            text: Text to synthesize
            language: Language code
            speaker: Speaker voice (if multi-speaker model)
        """
        if not self.initialized:
            await self.initialize()

        output_path = Path("./temp") / f"tts_{id(self)}.wav"
        output_path.parent.mkdir(exist_ok=True)

        # Generate speech
        self.tts_model.tts_to_file(
            text=text,
            file_path=str(output_path)
        )

        # Read and encode
        with open(output_path, 'rb') as f:
            audio_bytes = f.read()
            audio_base64 = base64.b64encode(audio_bytes).decode()

        # Get audio info
        data, sample_rate = sf.read(output_path)
        duration = len(data) / sample_rate

        return {
            "audio": audio_base64,
            "format": "wav",
            "sample_rate": sample_rate,
            "duration": duration,
            "text": text
        }

    async def clone_voice(
        self,
        reference_audio_path: str,
        text: str
    ) -> Dict[str, Any]:
        """
        Clone voice from reference audio

        Args:
            reference_audio_path: Audio sample of target voice
            text: Text to synthesize in cloned voice
        """
        try:
            from TTS.api import TTS

            # Load XTTS model for voice cloning (FREE)
            tts_clone = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2")

            output_path = Path("./temp") / f"cloned_voice_{id(self)}.wav"

            # Clone and generate
            tts_clone.tts_to_file(
                text=text,
                file_path=str(output_path),
                speaker_wav=reference_audio_path,
                language="en"
            )

            with open(output_path, 'rb') as f:
                audio_bytes = f.read()
                audio_base64 = base64.b64encode(audio_bytes).decode()

            data, sample_rate = sf.read(output_path)
            duration = len(data) / sample_rate

            return {
                "audio": audio_base64,
                "format": "wav",
                "sample_rate": sample_rate,
                "duration": duration,
                "text": text,
                "cloned": True
            }
        except Exception as e:
            raise RuntimeError(f"Voice cloning failed: {e}")


# Singleton
_audio_generator: Optional[AudioGenerator] = None

async def get_audio_generator() -> AudioGenerator:
    """Get or create audio generator instance"""
    global _audio_generator
    if _audio_generator is None:
        _audio_generator = AudioGenerator()
        await _audio_generator.initialize()
    return _audio_generator
