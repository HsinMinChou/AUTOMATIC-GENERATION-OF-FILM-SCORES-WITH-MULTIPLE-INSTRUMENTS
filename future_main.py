from video_emotion_helper import VideoEmotionFeatureExtractor
from openaitool import ChatGPTResponder
from stable_audio_tools import get_pretrained_model
import torch


def generate_audio_from_prompt(prompt, model_config_path, ckpt_path, pretrained_name=None, pretransform_ckpt_path=None,
                               model_half=True):
    # 為了確保實驗可重現，設定固定隨機種子
    torch.manual_seed(42)

    # 取得預訓練模型，這個方法需要你傳入模型的配置和檢查點路徑
    model = get_pretrained_model(
        model_config_path=model_config_path,
        ckpt_path=ckpt_path,
        pretrained_name=pretrained_name,
        pretransform_ckpt_path=pretransform_ckpt_path,
        model_half=model_half
    )

    # 使用模型的生成方法，將 prompt 傳入並獲取生成的音頻
    # 注意：這裡假設 generate 方法存在且回傳的是音頻數據（例如 wav 格式的二進位資料）
    generated_audio = model.generate(prompt)
    return generated_audio


def main():
    # 1. 從影片中提取情緒特徵
    video_file = "Ambience.mp4"
    extractor = VideoEmotionFeatureExtractor(video_file)
    emotion_result = extractor.extract_emotion_feature()
    print("情緒特徵提取結果：")
    print(emotion_result)

    # 2. 使用 ChatGPT 取得生成音樂的 prompt
    LLM = ChatGPTResponder()
    music_prompt = LLM.get_response(emotion_result)
    print("生成音樂的 prompt：")
    print(music_prompt)

    # 3. 設定 stable audio 生成模型的參數
    model_config_path = "path/to/model_config.json"  # 替換為你的 config 路徑
    ckpt_path = "path/to/model_checkpoint.ckpt"  # 替換為你的模型檢查點路徑
    pretrained_name = "your_pretrained_model_name"  # 如果使用預訓練模型，這裡填入名稱
    pretransform_ckpt_path = "path/to/pretransform.ckpt"  # 如果有預處理模型檢查點，否則可設為 None

    # 4. 呼叫 stable audio 生成音頻
    generated_audio = generate_audio_from_prompt(
        prompt=music_prompt,
        model_config_path=model_config_path,
        ckpt_path=ckpt_path,
        pretrained_name=pretrained_name,
        pretransform_ckpt_path=pretransform_ckpt_path,
        model_half=True
    )

    # 5. 將生成的音頻保存成檔案
    output_audio_path = "generated_audio.wav"
    with open(output_audio_path, "wb") as f:
        f.write(generated_audio)
    print("音頻已生成並保存為", output_audio_path)


if __name__ == '__main__':
    main()