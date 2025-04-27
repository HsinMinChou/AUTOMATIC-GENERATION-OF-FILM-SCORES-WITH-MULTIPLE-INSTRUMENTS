import os
import shutil
from pathlib import Path
import cv2
import math

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename

from moviepy.editor import VideoFileClip, AudioFileClip

from video_emotion_helper import VideoEmotionFeatureExtractor
from openaitool import ChatGPTResponder
from stable_connect import StableAudioGenerator

UPLOAD_FOLDER = "uploads"
OUTPUT_FOLDER = "outputs"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

ALLOWED_EXTENSIONS = {"mp4", "mov", "avi"}

app = Flask(__name__)
CORS(app)

def allowed_file(filename):
    return "." in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/upload', methods=['POST'])
def upload_video():
    if 'video' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['video']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    if file and allowed_file(file.filename):
        fname = secure_filename(file.filename)
        file.save(os.path.join(UPLOAD_FOLDER, fname))
        return jsonify({"status": "success", "filename": fname})
    return jsonify({"error": "Unsupported file type"}), 400

@app.route('/generate', methods=['POST'])
def generate_music():
    data = request.get_json()
    if not data or "filename" not in data:
        return jsonify({"error": "No filename provided"}), 400

    video_fn = data['filename']
    video_path = os.path.join(UPLOAD_FOLDER, video_fn)
    if not os.path.exists(video_path):
        return jsonify({"error": "Video file not found"}), 404

    # 1. 讀取影片長度
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return jsonify({"error": "Cannot open video file"}), 500
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
    duration_sec = math.ceil(frame_count / fps)
    cap.release()

    # 檢查長度限制
    if duration_sec > 47:
        return jsonify({
            "error": f"影片過長（{duration_sec} 秒），請上傳不超過 47 秒的影片。"
        }), 400

    # 2. 提取情緒特徵 & 生成 Prompt
    extractor = VideoEmotionFeatureExtractor(video_path)
    txt_path = extractor.extract_emotion_feature()
    responder = ChatGPTResponder()
    music_prompt = responder.get_response(txt_path)

    # 3. 呼叫 StableAudioGenerator 生成 WAV
    generator = StableAudioGenerator()
    base = Path(video_fn).stem
    wav_name = f"{base}_music.wav"
    wav_temp = generator.generate(
        prompt=music_prompt,
        file_naming=wav_name,
        seconds_total=duration_sec
    )
    wav_path = Path(OUTPUT_FOLDER) / wav_name
    shutil.move(str(wav_temp), str(wav_path))

    # 4. 使用 MoviePy 合併靜音影片與新音軌
    video_clip = VideoFileClip(str(video_path)).without_audio()
    audio_clip = AudioFileClip(str(wav_path)).set_duration(video_clip.duration)
    final_clip = video_clip.set_audio(audio_clip)

    merged_name = f"{base}_with_music.mp4"
    merged_path = Path(OUTPUT_FOLDER) / merged_name
    final_clip.write_videofile(
        str(merged_path),
        codec="libx264",
        audio_codec="aac",
        temp_audiofile="temp-audio.m4a",
        remove_temp=True,
        threads=4,
    )

    # 5. 直接回傳含音軌的 MP4
    return send_from_directory(
        directory=OUTPUT_FOLDER,
        path=merged_name,
        as_attachment=True,
        mimetype="video/mp4"
    )

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000)
