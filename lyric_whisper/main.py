from .model.whisper import Whisper

class LyricGen:
    def __init__(self) -> None:
        self.model = Whisper("openai/whisper-large-v3")
        
    def generate_lyrics(self, audio_path: str, language: str = None) -> str:
        return self.model.transcribe(audio_path, language=language)