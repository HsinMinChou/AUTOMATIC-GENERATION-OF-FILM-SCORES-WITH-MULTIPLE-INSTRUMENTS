from gradio_client import Client
from pathlib import Path
import shutil

class StableAudioGenerator:
    """
    Helper class for generating audio via a Gradio-based API.

    Because the API only accepts a fixed set of file_naming choices, this wrapper
    will always call the API using its default "output.wav", then rename the file
    to the user-specified name.

    Parameters:
        base_url (str): URL of the Gradio server (default: "http://127.0.0.1:7860/").
    """
    def __init__(self, base_url: str = "http://127.0.0.1:7860/"):
        self.client = Client(base_url)

    def generate(self, prompt: str, file_naming: str, seconds_total: int) -> Path:
        """
        Generate audio based on the given prompt and parameters, returning the path
        to the generated .wav file with the desired filename.

        Args:
            prompt (str): Text prompt describing the desired audio.
            file_naming (str): Desired output file name (e.g., "music.wav").
            seconds_total (int): Total duration of the generated audio in seconds.

        Returns:
            Path: Absolute pathlib.Path to the generated .wav audio file, renamed.
        """
        # Prepare parameters, using the API's supported default file name
        params = {
            "prompt": prompt,
            "negative_prompt": None,
            "seconds_start": 0,
            "seconds_total": seconds_total,
            "cfg_scale": 7,
            "steps": 100,
            "preview_every": 0,
            "seed": "-1",
            "sampler_type": "dpmpp-3m-sde",
            "sigma_min": 0.01,
            "sigma_max": 100,
            "rho": 1,
            "cfg_interval_min": 0,
            "cfg_interval_max": 1,
            "cfg_rescale": 0,
            "file_format": "wav",
            # Always use the API's default output name
            "file_naming": "output.wav",
            "cut_to_seconds_total": True,
            "init_audio": None,
            "init_noise_level": 0.1,
            "mask_maskstart": 0,
            "mask_maskend": seconds_total,
            "inpaint_audio": None
        }
        # Call API
        raw_result = self.client.predict(**params, api_name="/generate")
        # Extract the default-generated file path
        wav_path = Path(raw_result[0]).resolve()
        # Rename to desired filename if needed
        desired_path = wav_path.parent / file_naming
        if wav_path != desired_path:
            shutil.move(str(wav_path), str(desired_path))
        return desired_path


if __name__ == "__main__":
    generator = StableAudioGenerator()
    target = generator.generate(
        prompt="Hello!!",
        file_naming="output.wav",
        seconds_total=47
    )
    print(target)
