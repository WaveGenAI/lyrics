import numpy as np
from huggingface_hub import hf_hub_download
from whisper import load_model, transcribe


class Whisper:
    """Class to transcribe audio to text using the Whisper model.
    """

    def __init__(self, model_id: str = "distil-whisper/distil-large-v3") -> None:
        """Function to initialize the Whisper model.

        :param model_id: the model ID, defaults to "distil-whisper/distil-large-v3"
        :type model_id: str, optional
        """

        model_path = hf_hub_download(
            repo_id="distil-whisper/distil-large-v3-openai", filename="model.bin")
        self.model = load_model(model_path)

    def transcribe(self, audio_path: str, language: str = None) -> tuple:
        """Method to transcribe audio to text using the Whisper model.

        :param audio_path: the path to the audio file
        :type audio_path: str
        :return: the transcribed text with the average log probability
        :rtype: tuple
        """

        pred_out = transcribe(self.model, audio=audio_path, language=language)

        text = ""
        avg_logprob = 0

        for pred in pred_out['segments']:
            avg_logprob += pred["avg_logprob"]
            text += pred["text"].strip() + "\n"

        avg_logprob = avg_logprob / len(pred_out['segments'])

        return np.exp(avg_logprob), text
