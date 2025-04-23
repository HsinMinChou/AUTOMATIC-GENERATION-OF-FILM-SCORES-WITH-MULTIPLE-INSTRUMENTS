import os
import subprocess
import torch
import torchaudio
import soundfile as sf
from stable_audio_tools import get_pretrained_model
from stable_audio_tools.inference.generation import generate_diffusion_cond


class StableAudioHelper:
    """
    Helper library for generating audio with Stable Audio Diffusion models.

    Usage:
        from stable_audio_helper import StableAudioHelper
        helper = StableAudioHelper()
        output_path = helper.generate(
            prompt="Your musical prompt here", duration=30.0
        )
        print(f"Generated audio saved to {output_path}")

        push_result = helper.push_to_local(
            file_path=output_path,
            local_user="HsinMin",
            local_host="172.20.10.4",
            local_dir="/Users/hsinmin/Downloads"
        )
        print(push_result)
    """

    def __init__(
        self,
        model_name: str = "stabilityai/stable-audio-open-1.0"
    ):
        # Load model and config
        self.model, self.model_config = get_pretrained_model(model_name)
        # Select device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Move model to device and set to eval
        self.model.to(self.device).eval()
        # Try half precision if supported
        try:
            self.model.half()
        except Exception:
            pass
        # Use soundfile as backend for torchaudio
        torchaudio.set_audio_backend("soundfile")

    def generate(
        self,
        prompt: str,
        duration: float,
        output_path: str = "output.wav",
        steps: int = 100,
        cfg_scale: float = 7.0,
        sampler_type: str = "dpmpp-3m-sde",
        sigma_min: float = 0.03,
        sigma_max: float = 500.0,
        seed: int = 42
    ) -> str:
        """
        Generate audio and save to a WAV file.

        Args:
            prompt: Text description for the music.
            duration: Total duration of the clip in seconds.
            output_path: Path to save the generated WAV file.
            steps: Number of diffusion steps.
            cfg_scale: CFG scale for guidance.
            sampler_type: Sampler type to use.
            sigma_min: Minimum sigma value.
            sigma_max: Maximum sigma value.
            seed: Random seed for reproducibility.

        Returns:
            The output_path of the saved WAV file.
        """
        conditioning = [{
            "prompt": prompt,
            "seconds_start": 0,
            "seconds_total": duration
        }]

        # Run generation
        audio_tensor = generate_diffusion_cond(
            model=self.model,
            steps=steps,
            cfg_scale=cfg_scale,
            conditioning=conditioning,
            sample_size=self.model_config["sample_size"],
            sigma_min=sigma_min,
            sigma_max=sigma_max,
            sampler_type=sampler_type,
            device=self.device,
            seed=seed,
        )

        # Convert to CPU numpy and save
        audio_cpu = audio_tensor.squeeze(0).cpu().to(torch.float32)
        audio_np = audio_cpu.numpy().T  # [samples, channels]
        sample_rate = self.model_config.get("sample_rate", 48000)
        sf.write(output_path, audio_np, sample_rate)

        return output_path

    def push_to_local(
        self,
        file_path: str,
        local_user: str,
        local_host: str,
        local_dir: str
    ) -> dict:
        """
        Push the generated file to a local machine via scp.

        Args:
            file_path: Path of the file to push.
            local_user: Username on the local machine.
            local_host: Host or IP of the local machine.
            local_dir: Destination directory on the local machine.

        Returns:
            A dict with keys: success (bool), stdout, stderr, returncode.
        """
        cmd = ["scp", file_path, f"{local_user}@{local_host}:{local_dir}"]
        result = subprocess.run(cmd, capture_output=True, text=True)
        return {
            "success": result.returncode == 0,
            "stdout": result.stdout.strip(),
            "stderr": result.stderr.strip(),
            "returncode": result.returncode
        }
