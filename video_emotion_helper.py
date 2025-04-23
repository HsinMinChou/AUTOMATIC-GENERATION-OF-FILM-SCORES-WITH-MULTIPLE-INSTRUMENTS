#!/usr/bin/env python


import os
import subprocess
import shutil
from pathlib import Path
import torch
import clip
from PIL import Image

class VideoEmotionFeatureExtractor:
    def __init__(self, video_path):
        self.video_path = Path(video_path)
        self.video_dir = self.video_path.parent
        self.video_base = self.video_path.stem

    def _split_video_into_frames(self, frame_dir):
        try:
            frame_dir.mkdir(parents=True, exist_ok=True)
            output_path = str(frame_dir / "%03d.jpg")
            cmd = (
                f'ffmpeg -i "{self.video_path}" '
                f'-vf "select=bitor(gte(t-prev_selected_t\\,1)\\,isnan(prev_selected_t))" '
                f'-vsync 0 -qmin 1 -q:v 1 "{output_path}"'
            )
            ret = subprocess.call(cmd, shell=True)
            if ret != 0:
                print("ffmpeg excaute fail，plz check if ffmpeg installed。")
        except Exception as e:
            print(f"_split_video_into_frames error：{e}")
            raise

    def extract_emotion_feature(self):
        try:
            temp_frame_dir = self.video_dir / f"temp_frames_{self.video_base}"
            if temp_frame_dir.exists():
                shutil.rmtree(temp_frame_dir)
            temp_frame_dir.mkdir(parents=True, exist_ok=True)

            self._split_video_into_frames(temp_frame_dir)

            device = "cuda:0" if torch.cuda.is_available() else "cpu"
            model, preprocess = clip.load("ViT-L/14@336px", device=device)
            text_inputs = clip.tokenize(
                ["exciting", "fearful", "tense", "sad", "relaxing", "neutral"]
            ).to(device)

            file_names = sorted([f for f in os.listdir(temp_frame_dir) if f.endswith('.jpg')])
            emolist = []
            for idx, file_name in enumerate(file_names):
                try:
                    fpath = temp_frame_dir / file_name
                    image = preprocess(Image.open(fpath)).unsqueeze(0).to(device)
                    with torch.no_grad():
                        logits_per_image, _ = model(image, text_inputs)
                        probs = logits_per_image.softmax(dim=-1).cpu().numpy()
                    fp_vals = [format(probs[0][i], ".4f") for i in range(6)]
                    emo_val = " ".join(fp_vals)
                    emolist.append(f"{idx} {emo_val}")
                    print(f"[Emotion] Processed {file_name}: probabilities = {emo_val}")
                except Exception as inner_e:
                    print(f" {file_name} error：{inner_e}")
                    continue

            header = "time exciting_prob fearful_prob tense_prob sad_prob relaxing_prob neutral_prob"
            result_str = header + "\n" + "\n".join(emolist)

            output_txt_path = self.video_dir / f"{self.video_base}.txt"
            with open(output_txt_path, 'w', encoding='utf-8') as f:
                f.write(result_str)
            print(f"emotional feature：{output_txt_path}")

            shutil.rmtree(temp_frame_dir)

            return result_str
        except Exception as e:
            print(f"excaute extract_emotion_feature error：{e}")
            return ""
