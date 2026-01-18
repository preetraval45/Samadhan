import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import wave
import struct


class SpeakerEncoder(nn.Module):
    """encodes speaker identity from 3 seconds of audio"""

    def __init__(self, mel_bins=80, embedding_dim=256):
        super().__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv1d(mel_bins, 256, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Conv1d(256, 256, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Conv1d(256, 256, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.BatchNorm1d(256)
        )

        self.lstm = nn.LSTM(256, 256, num_layers=3, batch_first=True)

        self.projection = nn.Linear(256, embedding_dim)

    def forward(self, mel_spectrograms):
        x = self.conv_layers(mel_spectrograms)

        x = x.transpose(1, 2)

        _, (hidden, _) = self.lstm(x)

        embedding = self.projection(hidden[-1])

        embedding = F.normalize(embedding, p=2, dim=1)

        return embedding


class VoiceSynthesizer(nn.Module):
    """synthesizes speech with target voice"""

    def __init__(self, num_chars=256, mel_bins=80, speaker_dim=256):
        super().__init__()

        self.char_embedding = nn.Embedding(num_chars, 512)

        self.encoder = nn.LSTM(512, 512, num_layers=3, batch_first=True, bidirectional=True)

        self.attention = nn.MultiheadAttention(1024, num_heads=8, batch_first=True)

        self.speaker_projection = nn.Linear(speaker_dim, 1024)

        self.decoder_lstm = nn.LSTM(1024 + mel_bins, 1024, num_layers=2, batch_first=True)

        self.mel_projection = nn.Linear(1024, mel_bins)

        self.post_net = nn.Sequential(
            nn.Conv1d(mel_bins, 512, kernel_size=5, padding=2),
            nn.Tanh(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.5),
            nn.Conv1d(512, 512, kernel_size=5, padding=2),
            nn.Tanh(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.5),
            nn.Conv1d(512, mel_bins, kernel_size=5, padding=2)
        )

    def forward(self, text_tokens, speaker_embedding, mel_target=None, max_len=1000):
        text_embed = self.char_embedding(text_tokens)

        encoder_out, _ = self.encoder(text_embed)

        speaker_emb = self.speaker_projection(speaker_embedding)
        speaker_emb = speaker_emb.unsqueeze(1).repeat(1, encoder_out.size(1), 1)

        encoder_out = encoder_out + speaker_emb

        batch_size = text_tokens.size(0)

        mel_outputs = []
        decoder_input = torch.zeros(batch_size, 1, 80, device=text_tokens.device)
        hidden = None

        for t in range(max_len):
            attn_out, _ = self.attention(
                decoder_input,
                encoder_out,
                encoder_out
            )

            lstm_input = torch.cat([attn_out, decoder_input], dim=-1)

            decoder_out, hidden = self.decoder_lstm(lstm_input, hidden)

            mel_frame = self.mel_projection(decoder_out)

            mel_outputs.append(mel_frame)

            decoder_input = mel_frame

            if mel_target is not None and t < mel_target.size(1) - 1:
                decoder_input = mel_target[:, t:t+1, :]

        mel_output = torch.cat(mel_outputs, dim=1)

        mel_postnet = self.post_net(mel_output.transpose(1, 2)).transpose(1, 2)

        mel_output = mel_output + mel_postnet

        return mel_output


class EmotionalSpeechController(nn.Module):
    """controls emotion in synthesized speech"""

    def __init__(self, emotion_dim=64):
        super().__init__()

        self.emotions = {
            'neutral': 0,
            'happy': 1,
            'sad': 2,
            'angry': 3,
            'surprised': 4,
            'fearful': 5,
            'excited': 6,
            'calm': 7
        }

        self.emotion_embedding = nn.Embedding(len(self.emotions), emotion_dim)

        self.emotion_projection = nn.Sequential(
            nn.Linear(emotion_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512)
        )

    def forward(self, emotion_name):
        emotion_id = self.emotions.get(emotion_name, 0)

        emotion_tensor = torch.tensor([emotion_id], device=next(self.parameters()).device)

        emotion_emb = self.emotion_embedding(emotion_tensor)

        emotion_features = self.emotion_projection(emotion_emb)

        return emotion_features


class Vocoder(nn.Module):
    """converts mel spectrogram to waveform"""

    def __init__(self, mel_bins=80):
        super().__init__()

        self.upsample = nn.Sequential(
            nn.ConvTranspose1d(mel_bins, 512, kernel_size=16, stride=8, padding=4),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose1d(512, 256, kernel_size=16, stride=8, padding=4),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose1d(256, 128, kernel_size=8, stride=4, padding=2),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose1d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2)
        )

        self.residual_blocks = nn.ModuleList([
            self._make_residual_block(64) for _ in range(3)
        ])

        self.output = nn.Conv1d(64, 1, kernel_size=7, padding=3)

    def _make_residual_block(self, channels):
        return nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        )

    def forward(self, mel):
        x = self.upsample(mel)

        for block in self.residual_blocks:
            x = x + block(x)

        waveform = self.output(x)

        waveform = torch.tanh(waveform)

        return waveform


class NoiseRemovalModel(nn.Module):
    """removes background noise from audio"""

    def __init__(self):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=15, stride=1, padding=7),
            nn.LeakyReLU(0.2),
            nn.Conv1d(64, 128, kernel_size=15, stride=2, padding=7),
            nn.LeakyReLU(0.2),
            nn.Conv1d(128, 256, kernel_size=15, stride=2, padding=7),
            nn.LeakyReLU(0.2)
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(256, 128, kernel_size=16, stride=2, padding=7),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose1d(128, 64, kernel_size=16, stride=2, padding=7),
            nn.LeakyReLU(0.2),
            nn.Conv1d(64, 1, kernel_size=15, padding=7),
            nn.Tanh()
        )

    def forward(self, noisy_audio):
        encoded = self.encoder(noisy_audio)

        clean_audio = self.decoder(encoded)

        return clean_audio


class RealTimeVoiceConverter(nn.Module):
    """converts voice in real-time"""

    def __init__(self, speaker_dim=256):
        super().__init__()

        self.content_encoder = nn.Sequential(
            nn.Conv1d(80, 256, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(256, 512, kernel_size=5, padding=2),
            nn.ReLU()
        )

        self.speaker_projection = nn.Linear(speaker_dim, 512)

        self.decoder = nn.Sequential(
            nn.Conv1d(1024, 512, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(512, 256, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(256, 80, kernel_size=5, padding=2)
        )

    def forward(self, source_mel, target_speaker):
        content = self.content_encoder(source_mel)

        speaker_emb = self.speaker_projection(target_speaker)

        speaker_broadcast = speaker_emb.unsqueeze(-1).repeat(1, 1, content.size(-1))

        combined = torch.cat([content, speaker_broadcast], dim=1)

        converted = self.decoder(combined)

        return converted


class VoiceCloningSystem:
    """complete zero-shot voice cloning system"""

    def __init__(self, device='cuda'):
        self.device = device

        self.speaker_encoder = SpeakerEncoder().to(device)
        self.synthesizer = VoiceSynthesizer().to(device)
        self.emotion_controller = EmotionalSpeechController().to(device)
        self.vocoder = Vocoder().to(device)
        self.noise_remover = NoiseRemovalModel().to(device)
        self.voice_converter = RealTimeVoiceConverter().to(device)

        self.sample_rate = 22050

    def extract_mel_spectrogram(self, audio_waveform):
        """extracts mel spectrogram from audio"""

        n_fft = 1024
        hop_length = 256
        n_mels = 80

        audio_tensor = torch.from_numpy(audio_waveform).float()

        if audio_tensor.dim() == 1:
            audio_tensor = audio_tensor.unsqueeze(0)

        stft = torch.stft(
            audio_tensor,
            n_fft=n_fft,
            hop_length=hop_length,
            return_complex=True
        )

        magnitude = torch.abs(stft)

        mel_basis = torch.randn(n_mels, n_fft // 2 + 1)
        mel_spec = torch.matmul(mel_basis, magnitude)

        mel_spec = torch.log(mel_spec + 1e-5)

        return mel_spec

    def encode_speaker(self, reference_audio):
        """encodes speaker from 3 seconds of audio"""

        if isinstance(reference_audio, str):
            with wave.open(reference_audio, 'rb') as wav:
                frames = wav.readframes(wav.getnframes())
                audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
        else:
            audio = reference_audio

        if len(audio) > self.sample_rate * 3:
            audio = audio[:self.sample_rate * 3]

        mel_spec = self.extract_mel_spectrogram(audio)
        mel_spec = mel_spec.unsqueeze(0).to(self.device)

        with torch.no_grad():
            speaker_embedding = self.speaker_encoder(mel_spec)

        return speaker_embedding

    def synthesize_speech(self, text, speaker_embedding, emotion='neutral'):
        """synthesizes speech with target voice and emotion"""

        text_tokens = torch.tensor(
            [ord(c) for c in text],
            dtype=torch.long,
            device=self.device
        ).unsqueeze(0)

        emotion_features = self.emotion_controller(emotion)

        speaker_emb_with_emotion = speaker_embedding + emotion_features.unsqueeze(0)

        with torch.no_grad():
            mel_output = self.synthesizer(
                text_tokens,
                speaker_emb_with_emotion,
                max_len=len(text) * 10
            )

            waveform = self.vocoder(mel_output.transpose(1, 2))

        audio = waveform[0, 0].cpu().numpy()

        return audio

    def clone_voice(self, reference_audio, text, emotion='neutral'):
        """clones voice from reference and synthesizes new text"""

        speaker_embedding = self.encode_speaker(reference_audio)

        synthesized_audio = self.synthesize_speech(text, speaker_embedding, emotion)

        return synthesized_audio

    def remove_noise(self, noisy_audio):
        """removes background noise from audio"""

        if isinstance(noisy_audio, str):
            with wave.open(noisy_audio, 'rb') as wav:
                frames = wav.readframes(wav.getnframes())
                audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
        else:
            audio = noisy_audio

        audio_tensor = torch.from_numpy(audio).float().unsqueeze(0).unsqueeze(0).to(self.device)

        with torch.no_grad():
            clean_audio = self.noise_remover(audio_tensor)

        return clean_audio[0, 0].cpu().numpy()

    def convert_voice_realtime(self, source_audio, target_reference):
        """converts voice in real-time"""

        target_embedding = self.encode_speaker(target_reference)

        source_mel = self.extract_mel_spectrogram(source_audio)
        source_mel = source_mel.unsqueeze(0).to(self.device)

        with torch.no_grad():
            converted_mel = self.voice_converter(source_mel, target_embedding)

            converted_audio = self.vocoder(converted_mel)

        return converted_audio[0, 0].cpu().numpy()

    def save_audio(self, audio, filepath, sample_rate=None):
        """saves audio to file"""

        if sample_rate is None:
            sample_rate = self.sample_rate

        audio_int = (audio * 32767).astype(np.int16)

        with wave.open(filepath, 'w') as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)

            for sample in audio_int:
                wav_file.writeframes(struct.pack('h', sample))


class AudioSuperResolution(nn.Module):
    """upsamples audio quality"""

    def __init__(self, upscale_factor=2):
        super().__init__()

        self.upscale_factor = upscale_factor

        self.upsampler = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=9, padding=4),
            nn.PReLU(),
            nn.Conv1d(64, 32, kernel_size=5, padding=2),
            nn.PReLU(),
            nn.Conv1d(32, upscale_factor, kernel_size=5, padding=2)
        )

    def forward(self, low_res_audio):
        x = self.upsampler(low_res_audio)

        batch_size = x.size(0)
        channels = x.size(1)
        length = x.size(2)

        x = x.view(batch_size, channels * length)

        return x.unsqueeze(1)
