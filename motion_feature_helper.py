import os
import cv2
from pathlib import Path

class VideoMotionFeatureExtractor:
    def __init__(self, video_path):
        self.video_path = Path(video_path)
        self.video_dir = self.video_path.parent
        self.video_base = self.video_path.stem

    def extract_motion_feature(self):
        try:
            if not self.video_path.is_file():
                print(f"找不到影片檔案：{self.video_path}")
                return "", ""

            cap = cv2.VideoCapture(str(self.video_path))
            if not cap.isOpened():
                print(f"無法開啟影片：{self.video_path}")
                return "", ""

            fps = int(cap.get(cv2.CAP_PROP_FPS))
            prev_frame = None
            prev_time = 0
            motiondict = {0: "0.0000"}  # 預設第0秒為0

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                curr_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0

                if prev_frame is not None and curr_time - prev_time >= 1:
                    diff = cv2.absdiff(frame, prev_frame)
                    diff_rgb = cv2.cvtColor(diff, cv2.COLOR_BGR2RGB)
                    motion_value = format(diff_rgb.mean(), ".4f")
                    sec = int(curr_time)
                    motiondict[sec] = motion_value
                    prev_time = sec

                prev_frame = frame.copy()

            cap.release()
            cv2.destroyAllWindows()

            # 準備輸出內容
            header = "time_sec motion_value"
            lines = [f"{i} {motiondict[i]}" for i in sorted(motiondict)]
            result = "\n".join([header] + lines)

            # 寫入 txt
            out_path = self.video_dir / f"{self.video_base}_motion.txt"
            with open(out_path, 'w', encoding='utf-8') as f:
                f.write(result)

            print(f"畫面變化特徵已儲存至：{out_path}")
            return result, str(out_path)

        except Exception as e:
            print(f"執行 extract_motion_feature 時發生錯誤：{e}")
            return "", ""
