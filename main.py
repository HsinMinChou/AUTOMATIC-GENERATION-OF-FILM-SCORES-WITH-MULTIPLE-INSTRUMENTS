#!/usr/bin/env python

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
