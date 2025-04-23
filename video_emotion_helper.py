#!/usr/bin/env python
"""
video_emotion_helper.py

此模組提供 VideoEmotionFeatureExtractor 物件，
供其他 Python 程式透過 import 後呼叫 extract_emotion_feature() 方法：
    - 輸入為 .mp4 影片檔案路徑
    - 切出影格，再利用 CLIP 模型萃取情感特徵（Emotion Feature）
    - 將結果儲存為與輸入檔案同一資料夾下、檔名相同但副檔名為 .txt 的檔案
    - 最後回傳結果字串

使用時不包含 main 測試區塊，僅作為 lib 或 helper 使用。
"""

import os
import subprocess
import shutil
from pathlib import Path
import torch
import clip
from PIL import Image

class VideoEmotionFeatureExtractor:
    def __init__(self, video_path):
        """
        初始化，接收要處理的影片檔案路徑
        """
        self.video_path = Path(video_path)
        self.video_dir = self.video_path.parent
        self.video_base = self.video_path.stem

    def _split_video_into_frames(self, frame_dir):
        """
        利用 ffmpeg 將影片切成每秒一張影格，存放在 frame_dir
        """
        try:
            frame_dir.mkdir(parents=True, exist_ok=True)
            # ffmpeg 指令：每秒抽一張影格
            output_path = str(frame_dir / "%03d.jpg")
            cmd = (
                f'ffmpeg -i "{self.video_path}" '
                f'-vf "select=bitor(gte(t-prev_selected_t\\,1)\\,isnan(prev_selected_t))" '
                f'-vsync 0 -qmin 1 -q:v 1 "{output_path}"'
            )
            ret = subprocess.call(cmd, shell=True)
            if ret != 0:
                print("ffmpeg 執行失敗，請檢查 ffmpeg 是否正確安裝。")
        except Exception as e:
            print(f"_split_video_into_frames 發生錯誤：{e}")
            raise

    def extract_emotion_feature(self):
        """
        萃取情感特徵：
          1. 利用 ffmpeg 切出每秒一張影格
          2. 使用 CLIP 模型搭配情感描述文字萃取每張影格的情感機率
          3. 將結果存檔（檔名與輸入影片同名、附 .txt 副檔名）
          4. 回傳結果字串
        """
        try:
            # 建立暫存影格資料夾（放於與影片相同路徑）
            temp_frame_dir = self.video_dir / f"temp_frames_{self.video_base}"
            if temp_frame_dir.exists():
                shutil.rmtree(temp_frame_dir)
            temp_frame_dir.mkdir(parents=True, exist_ok=True)

            # 切出影格
            self._split_video_into_frames(temp_frame_dir)

            # 載入 CLIP 模型
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
            model, preprocess = clip.load("ViT-L/14@336px", device=device)
            # 定義情感文字描述
            text_inputs = clip.tokenize(
                ["exciting", "fearful", "tense", "sad", "relaxing", "neutral"]
            ).to(device)

            # 讀取所有影格（僅取 jpg 檔）
            file_names = sorted([f for f in os.listdir(temp_frame_dir) if f.endswith('.jpg')])
            emolist = []
            for idx, file_name in enumerate(file_names):
                try:
                    fpath = temp_frame_dir / file_name
                    image = preprocess(Image.open(fpath)).unsqueeze(0).to(device)
                    with torch.no_grad():
                        logits_per_image, _ = model(image, text_inputs)
                        probs = logits_per_image.softmax(dim=-1).cpu().numpy()
                    # 格式化 6 個情感機率（小數點後四位）
                    fp_vals = [format(probs[0][i], ".4f") for i in range(6)]
                    emo_val = " ".join(fp_vals)
                    emolist.append(f"{idx} {emo_val}")
                    print(f"[Emotion] Processed {file_name}: probabilities = {emo_val}")
                except Exception as inner_e:
                    print(f"處理影格 {file_name} 發生錯誤：{inner_e}")
                    continue

            # 組合結果字串（包含標題行）
            header = "time exciting_prob fearful_prob tense_prob sad_prob relaxing_prob neutral_prob"
            result_str = header + "\n" + "\n".join(emolist)

            # 儲存結果至與影片同目錄下，檔名與影片同名（副檔名改為 .txt）
            output_txt_path = self.video_dir / f"{self.video_base}.txt"
            with open(output_txt_path, 'w', encoding='utf-8') as f:
                f.write(result_str)
            print(f"情感特徵已儲存至：{output_txt_path}")

            # 清除暫存資料夾
            shutil.rmtree(temp_frame_dir)

            return result_str
        except Exception as e:
            print(f"執行 extract_emotion_feature 時發生錯誤：{e}")
            return ""
