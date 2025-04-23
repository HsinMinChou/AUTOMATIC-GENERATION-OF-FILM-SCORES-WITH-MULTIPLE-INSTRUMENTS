#!/usr/bin/env python
"""
test_video_emotion.py

此腳本示範如何呼叫 video_emotion_helper 模組中的 VideoEmotionFeatureExtractor 物件，
並處理名為 test.mp4 的影片，最後印出萃取結果。
"""
from openaitool import ChatGPTResponder
from video_emotion_helper import VideoEmotionFeatureExtractor
from music_gen_helper import StableAudioHelper
def main():
    video_file = "Ambience.mp4"
    extractor = VideoEmotionFeatureExtractor(video_file)
    result = extractor.extract_emotion_feature()
    print("Emotion feature extraction result:")
    print(result)

    LLM = ChatGPTResponder()                              
    music_prompt =LLM.get_response(result)                              
    print(music_prompt)                              

    helper = StableAudioHelper()
    wav = helper.generate("Your musical prompt here", duration=30.0)
    print(f"Saved to {wav}")
    push_info = helper.push_to_local(wav, "HsinMin", "172.20.10.4", "/Users/hsinmin/Downloads")
    print(push_info)

if __name__ == '__main__':
    main()
