import os
import math
import shutil
from pathlib import Path
from scenedetect import open_video, SceneManager
from scenedetect.detectors import AdaptiveDetector

class VideoSceneFeatureExtractor:
    def __init__(self, video_path):
        self.video_path = Path(video_path)
        self.video_dir = self.video_path.parent
        self.video_base = self.video_path.stem

    def extract_scene_feature(self):
        try:
            if not self.video_path.is_file():
                print(f"找不到影片檔案：{self.video_path}")
                return "", ""

            video = open_video(str(self.video_path))
            scene_manager = SceneManager()
            scene_manager.add_detector(AdaptiveDetector())
            scene_manager.detect_scenes(video, show_progress=False)
            scene_list = scene_manager.get_scene_list()

            # 構建每秒對應的場景索引
            sec = 0
            scenedict = {}
            for idx, scene in enumerate(scene_list):
                end_sec = math.ceil(scene[1].get_seconds())
                for s in range(sec, end_sec):
                    scenedict[s] = idx
                    sec += 1

            # 準備結果文字
            header = "time_sec scene_index"
            lines = [f"{t} {scenedict[t]}" for t in range(len(scenedict))]
            result = "\n".join([header] + lines)

            # 寫入 txt
            out_path = self.video_dir / f"{self.video_base}_scene.txt"
            with open(out_path, 'w', encoding='utf-8') as f:
                f.write(result)

            print(f"場景特徵已儲存至：{out_path}")
            return result, str(out_path)

        except Exception as e:
            print(f"執行 extract_scene_feature 時發生錯誤：{e}")
            return "", ""

