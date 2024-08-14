from .model.whisper import Whisper
import torch

class LyricGen:
    def __init__(self, model_id: str = "openai/whisper-large-v3") -> None:
        self.model = Whisper(model_id)
        
    def generate_lyrics(self, audio: str | torch.Tensor, language: str = None) -> str:
        return self.model.transcribe(audio, language=language)