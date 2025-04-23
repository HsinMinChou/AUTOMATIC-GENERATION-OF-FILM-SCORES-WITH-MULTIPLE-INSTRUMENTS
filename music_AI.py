#!/usr/bin/env python
# test.py

import os
import subprocess
import torch
import torchaudio
import soundfile as sf
from stable_audio_tools import get_pretrained_model
from stable_audio_tools.inference.generation import generate_diffusion_cond

def main():
    # —— 1. 打印当前目录 & 文件列表，确认输出路径 ——
    cwd = os.getcwd()
    print(f"[1] Working directory: {cwd}")
    print("[1] Before generation:", os.listdir(cwd))

    # —— 2. 强制使用 soundfile 后端 ——
    torchaudio.set_audio_backend("soundfile")

    # —— 3. 载入模型 ——
    model, model_config = get_pretrained_model("stabilityai/stable-audio-open-1.0")

    # —— 4. 设备、eval、半精度 ——
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    try:
        model.half()
    except Exception:
        pass

    # —— 5. 准备 conditioning ——
    conditioning = [{
        "prompt": (
            "Compose a ultra‑gentle cinematic forest score that opens with soft celesta "
            "and muted piano in E major at 80 BPM, flows into warm legato solo cello "
            "and viola lines, introduces delicate harp arpeggios and an ethereal choir pad "
            "mid‑section, and concludes with sustained violin harmonies supported by a "
            "gentle ambient string pad."
        ),
        "seconds_start": 0,
        "seconds_total": 47
    }]

    # —— 6. 生成音频 ——
    print("[2] Generating audio...")
    audio_tensor = generate_diffusion_cond(
        model=model,
        steps=100,
        cfg_scale=7,
        conditioning=conditioning,
        batch_size=1,
        sample_size=model_config["sample_size"],
        sigma_min=0.03,
        sigma_max=500.0,
        sampler_type="dpmpp-3m-sde",
        device=device,
        seed=42,
    )

    # —— 7. 处理输出 ——
    audio_cpu = audio_tensor.squeeze(0).cpu().to(torch.float32)
    audio_np = audio_cpu.numpy().T  # 转成 [samples, channels]

    # —— 8. 保存 WAV ——
    out_filename = "output.wav"  # ← 文件名前后一致
    sample_rate = model_config.get("sample_rate", 48000)
    sf.write(out_filename, audio_np, sample_rate)
    print(f"[3] ✅ Saved WAV: {out_filename}")
    print("[3] After generation:", os.listdir(cwd))

    # —— 9. 调试并推送到本地 ——
    local_user = "HsinMin"
    local_host = "172.20.10.4"
    local_dir  = "/Users/hsinmin/Downloads"

    print(f"[4] About to run scp to {local_user}@{local_host}:{local_dir}")
    cmd = ["scp", out_filename, f"{local_user}@{local_host}:{local_dir}"]
    print("[4] Command:", " ".join(cmd))

    result = subprocess.run(cmd, capture_output=True, text=True)
    print("[4] Return code:", result.returncode)
    print("[4] STDOUT:", result.stdout.strip())
    print("[4] STDERR:", result.stderr.strip())

    if result.returncode == 0:
        print("[4] ✅ Successfully pushed to local machine.")
    else:
        print("[4] ❌ Failed to push. See stderr above.")

if __name__ == "__main__":
    main()
