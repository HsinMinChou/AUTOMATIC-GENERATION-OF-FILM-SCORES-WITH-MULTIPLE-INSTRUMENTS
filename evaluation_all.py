import os
import sys
import math
import csv
# 檢查所需的第三方套件並導入
import librosa.display
try:
    import moviepy.editor as mp
except ImportError:
    print("正在安裝 moviepy 套件...")
    os.system(sys.executable + " -m pip install moviepy")
    import moviepy.editor as mp

try:
    import librosa
    import numpy as np
    import scipy.stats
except ImportError:
    print("正在安裝 librosa 套件（將同時安裝 numpy 等）...")
    os.system(sys.executable + " -m pip install librosa numpy scipy")
    import librosa
    import numpy as np
    import scipy.stats
    import librosa.display
try:
    import pyloudnorm as pyln
except ImportError:
    print("正在安裝 pyloudnorm 套件...")
    os.system(sys.executable + " -m pip install pyloudnorm")
    import pyloudnorm as pyln

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
except ImportError:
    print("正在安裝 matplotlib 和 seaborn 套件...")
    os.system(sys.executable + " -m pip install matplotlib seaborn")
    import matplotlib.pyplot as plt
    import seaborn as sns

# 設定繪圖樣式
sns.set(style="ticks", font_scale=1.2)
plt.rcParams.update({"axes.grid": False})  # 關閉預設網格

# 影片檔案名稱
video_files = ["Ambience01拷貝.mp4", "Ambiencehi5拷貝.mp4"]
audio_files = ["Ambience01.wav", "Ambiencehi5.wav"]

print("步驟1: 從影片檔提取音訊並轉存為 WAV...")
for vf, af in zip(video_files, audio_files):
    if not os.path.isfile(vf):
        print(f"錯誤: 找不到影片檔 {vf}，請將檔案放置於當前目錄。")
        sys.exit(1)
    # 使用 moviepy 提取音訊
    print(f"  提取 {vf} 的音訊...")
    clip = mp.VideoFileClip(vf)
    try:
        clip.audio.write_audiofile(af, codec='pcm_s16le')  # PCM 16-bit little-endian
    except Exception as e:
        print(f"  使用 moviepy 提取音訊失敗: {e}")
        print("  嘗試使用 ffmpeg 進行提取...")
        # 使用 ffmpeg 命令作為後備方案
        rc = os.system(f"ffmpeg -y -i \"{vf}\" -vn -acodec pcm_s16le -ar 48000 -ac 2 \"{af}\"")
        if rc != 0:
            print(f"  ffmpeg 提取 {vf} 音訊失敗，請確定 ffmpeg 已安裝且影片檔非加密。")
            sys.exit(1)
    finally:
        if 'clip' in locals():
            clip.close()
    print(f"  音訊已保存為 {af}")

print("\n步驟2: 載入 WAV 音訊並計算特徵指標...")
# 用於保存分析結果的字典
analysis_results = {}

for af in audio_files:
    # 讀取音訊數據
    print(f"\n-- 分析 {af} --")
    # preserve original sample rate and stereo data
    y, sr = librosa.load(af, sr=None, mono=False)
    # 獲取通道資訊
    if y.ndim == 2:
        channels = y.shape[0]
        # 轉為單聲道以計算特徵（左右聲道平均）
        y_mono = librosa.to_mono(y)
    else:
        channels = 1
        y_mono = y
    duration = librosa.get_duration(y=y_mono, sr=sr)
    # 使用 wave 模組獲取位元深度
    import wave
    with wave.open(af, 'rb') as wf:
        sampwidth_bytes = wf.getsampwidth()
        bit_depth = sampwidth_bytes * 8
    print(f"音訊時長: {duration:.3f} 秒, 取樣率: {sr} Hz, 通道數: {channels}, 位元深度: {bit_depth}-bit")

    # 計算時域波形的基礎統計
    # 峰值 (Peak) 和 均值 (Mean)
    peak_amp = float(np.max(np.abs(y_mono)))
    mean_amp = float(np.mean(y_mono))
    # 標準差 (Std) 及波形幅度分布偏態、峰度
    std_amp = float(np.std(y_mono))
    skew_amp = float(scipy.stats.skew(y_mono)) if len(y_mono) > 1 else 0.0
    kurt_amp = float(scipy.stats.kurtosis(y_mono)) if len(y_mono) > 1 else 0.0

    # RMS 能量計算
    frame_length = 2048
    hop_length = 512
    rms = librosa.feature.rms(y=y_mono, frame_length=frame_length, hop_length=hop_length)[0]
    rms_mean = float(np.mean(rms))
    rms_max = float(np.max(rms))
    rms_min = float(np.min(rms))
    # 動態範圍（以 dB 表示最大與最小 RMS 比值）
    if rms_min > 0:
        dynamic_range_db = 20 * math.log10(rms_max / rms_min)
    else:
        dynamic_range_db = float('inf')  # 若音訊包含完全靜音帧，動態範圍視為無限大
    # 計算 LUFS 音量 (响度)
    meter = pyln.Meter(sr)
    loudness = float(meter.integrated_loudness(y_mono))
    # 峭度 (Crest Factor = peak/RMS_mean)
    crest_factor = peak_amp / rms_mean if rms_mean > 0 else float('inf')

    # 頻譜特徵計算
    # Zero Crossing Rate
    zcr = librosa.feature.zero_crossing_rate(y=y_mono, frame_length=frame_length, hop_length=hop_length)[0]
    zcr_mean = float(np.mean(zcr))
    zcr_std = float(np.std(zcr))
    # Spectral Centroid
    spec_centroid = librosa.feature.spectral_centroid(y=y_mono, sr=sr, n_fft=2048, hop_length=hop_length)[0]
    spec_centroid_mean = float(np.mean(spec_centroid))
    spec_centroid_std = float(np.std(spec_centroid))
    # Spectral Bandwidth
    spec_bandwidth = librosa.feature.spectral_bandwidth(y=y_mono, sr=sr, n_fft=2048, hop_length=hop_length)[0]
    spec_bandwidth_mean = float(np.mean(spec_bandwidth))
    spec_bandwidth_std = float(np.std(spec_bandwidth))
    # Spectral Rolloff (頻譜滾降，預設參數 roll_percent=0.85)
    spec_rolloff = librosa.feature.spectral_rolloff(y=y_mono, sr=sr, n_fft=2048, hop_length=hop_length)[0]
    spec_rolloff_mean = float(np.mean(spec_rolloff))
    spec_rolloff_std = float(np.std(spec_rolloff))
    # MFCCs
    mfcc = librosa.feature.mfcc(y=y_mono, sr=sr, n_mfcc=13, hop_length=hop_length)
    mfcc_means = list(np.mean(mfcc, axis=1))
    mfcc_stds = list(np.std(mfcc, axis=1))
    # Chroma (使用STFT計算)
    chroma = librosa.feature.chroma_stft(y=y_mono, sr=sr, n_fft=2048, hop_length=hop_length)
    chroma_means = list(np.mean(chroma, axis=1))
    chroma_stds = list(np.std(chroma, axis=1))
    # Spectral Entropy（Shannon entropy of normalized spectrum）
    S = np.abs(librosa.stft(y_mono, n_fft=2048, hop_length=hop_length))**2
    # 對每一列（每幀）計算能量歸一化後的 Shannon entropy
    S_norm = S / np.sum(S, axis=0, keepdims=True)
    # 加入極小值防止 log(0)
    eps = 1e-10
    entropy_per_frame = -np.sum(S_norm * np.log2(S_norm + eps), axis=0)
    spectral_entropy_mean = float(np.mean(entropy_per_frame))
    spectral_entropy_std = float(np.std(entropy_per_frame))

    # 將結果保存到字典
    analysis_results[af] = {
        "duration_s": duration,
        "sample_rate": sr,
        "channels": channels,
        "bit_depth": bit_depth,
        "peak_amplitude": peak_amp,
        "mean_amplitude": mean_amp,
        "std_amplitude": std_amp,
        "amplitude_skewness": skew_amp,
        "amplitude_kurtosis": kurt_amp,
        "rms_mean": rms_mean,
        "rms_max": rms_max,
        "rms_min": rms_min,
        "dynamic_range_db": dynamic_range_db,
        "loudness_lufs": loudness,
        "crest_factor": crest_factor,
        "zcr_mean": zcr_mean,
        "zcr_std": zcr_std,
        "spectral_centroid_mean": spec_centroid_mean,
        "spectral_centroid_std": spec_centroid_std,
        "spectral_bandwidth_mean": spec_bandwidth_mean,
        "spectral_bandwidth_std": spec_bandwidth_std,
        "spectral_rolloff_mean": spec_rolloff_mean,
        "spectral_rolloff_std": spec_rolloff_std,
        "spectral_entropy_mean": spectral_entropy_mean,
        "spectral_entropy_std": spectral_entropy_std,
        "mfcc_means": mfcc_means,
        "mfcc_stds": mfcc_stds,
        "chroma_means": chroma_means,
        "chroma_stds": chroma_stds
    }

    # 將主要結果列印輸出
    print("主要特徵與指標:")
    print(f"  - 平均RMS能量: {rms_mean:.6f}")
    print(f"  - 最大RMS能量: {rms_max:.6f}, 最小RMS能量: {rms_min:.6f}, 動態範圍: {dynamic_range_db:.2f} dB")
    print(f"  - LUFS響度: {loudness:.2f} LUFS")
    print(f"  - 峭度 (Crest Factor): {crest_factor:.2f}")
    print(f"  - 平均過零率 (ZCR): {zcr_mean:.4f}")
    print(f"  - 平均頻譜重心: {spec_centroid_mean:.2f} Hz")
    print(f"  - 平均頻譜帶寬: {spec_bandwidth_mean:.2f} Hz")
    print(f"  - 平均頻譜滾降: {spec_rolloff_mean:.2f} Hz")
    print(f"  - 平均頻譜熵: {spectral_entropy_mean:.4f} bit")

    # 列印 MFCC 前5維的均值作為示例（共13維）
    mfcc_means_rounded = [f"{m:.2f}" for m in mfcc_means[:5]]
    print(f"  - MFCC均值（前5維）: {mfcc_means_rounded} ...")

# 比較兩音訊的 PESQ (如果適用)
print("\n步驟3: 主觀音訊品質評估 (PESQ/ViSQOL)...")
pesq_score = None
try:
    from pesq import pesq as pesq_fn
except ImportError:
    print("Pesq 模組未安裝，跳過 PESQ 分數計算。若需此功能，請執行 `pip install pesq` 安裝。")
except Exception as e:
    print(f"PESQ 模組載入錯誤: {e}，跳過 PESQ 分數計算。")
else:
    try:
        # PESQ 要求輸入為 8kHz 或 16kHz，需轉換取樣率
        # 我們以16kHz寬帶模式來計算，如果原音頻不是16kHz，將重採樣
        import librosa
        ref_signal, _ = librosa.load(audio_files[0], sr=16000, mono=True)
        deg_signal, _ = librosa.load(audio_files[1], sr=16000, mono=True)
        pesq_score = pesq_fn(16000, ref_signal, deg_signal, 'wb')
        print(f"PESQ 分數 (假設 {audio_files[0]} 為參考, {audio_files[1]} 為待測): {pesq_score:.3f}")
    except Exception as e:
        print(f"PESQ 計算失敗: {e}")

print("（註：PESQ 分數範圍約 -0.5~4.5，數值越高表示品質越好）")
print("ViSQOL 評估需額外安裝專用庫，請參考 Google 的 visqol 工具，如需使用請自行安裝配置。")

# 產生圖表並保存
print("\n步驟4: 產生並保存波形和頻譜圖表...")

# 載入兩檔音訊的資料（單聲道）
y1, sr1 = librosa.load(audio_files[0], sr=None, mono=True)
y2, sr2 = librosa.load(audio_files[1], sr=None, mono=True)
dur1 = librosa.get_duration(y=y1, sr=sr1)
dur2 = librosa.get_duration(y=y2, sr=sr2)

# 波形比較圖
fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
t1 = np.linspace(0, dur1, num=len(y1))
t2 = np.linspace(0, dur2, num=len(y2))
axes[0].plot(t1, y1, color='steelblue')
axes[0].set_title(f"{audio_files[0]} 波形")
axes[1].plot(t2, y2, color='orange')
axes[1].set_title(f"{audio_files[1]} 波形")
axes[1].set_xlabel("時間 (秒)")
for ax in axes:
    ax.set_ylabel("幅度")
    ax.set_xlim(0, max(dur1, dur2))
    ax.grid(False)
plt.tight_layout()
fig.savefig("waveform_compare.png", dpi=300)
plt.close(fig)
print("  已保存 waveform_compare.png")

# 頻譜圖比較 (STFT 功率譜密度 dB)
fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
n_fft = 2048
hop = 512
# 音訊1頻譜圖
S1 = np.abs(librosa.stft(y1, n_fft=n_fft, hop_length=hop))**2
S1_db = librosa.power_to_db(S1, ref=np.max)
img1 = librosa.display.specshow(S1_db, sr=sr1, hop_length=hop, x_axis='time', y_axis='log', ax=axes[0])
axes[0].set_title(f"{audio_files[0]} 頻譜圖 (dB)")
# 音訊2頻譜圖
S2 = np.abs(librosa.stft(y2, n_fft=n_fft, hop_length=hop))**2
S2_db = librosa.power_to_db(S2, ref=np.max)
img2 = librosa.display.specshow(S2_db, sr=sr2, hop_length=hop, x_axis='time', y_axis='log', ax=axes[1])
axes[1].set_title(f"{audio_files[1]} 頻譜圖 (dB)")
axes[1].set_xlabel("時間 (秒)")
for ax in axes:
    ax.set_ylabel("頻率 (Hz)")
plt.colorbar(img2, ax=axes, format="%+2.0f dB")
plt.tight_layout()
fig.savefig("spectrogram_compare.png", dpi=300)
plt.close(fig)
print("  已保存 spectrogram_compare.png")

# 梅爾頻譜圖比較
fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
# 音訊1梅爾譜
mel1 = librosa.feature.melspectrogram(y=y1, sr=sr1, n_fft=n_fft, hop_length=hop, n_mels=128)
mel1_db = librosa.power_to_db(mel1, ref=np.max)
img1 = librosa.display.specshow(mel1_db, sr=sr1, hop_length=hop, x_axis='time', y_axis='mel', ax=axes[0])
axes[0].set_title(f"{audio_files[0]} 梅爾頻譜圖")
# 音訊2梅爾譜
mel2 = librosa.feature.melspectrogram(y=y2, sr=sr2, n_fft=n_fft, hop_length=hop, n_mels=128)
mel2_db = librosa.power_to_db(mel2, ref=np.max)
img2 = librosa.display.specshow(mel2_db, sr=sr2, hop_length=hop, x_axis='time', y_axis='mel', ax=axes[1])
axes[1].set_title(f"{audio_files[1]} 梅爾頻譜圖")
axes[1].set_xlabel("時間 (秒)")
for ax in axes:
    ax.set_ylabel("梅爾頻率")
plt.colorbar(img2, ax=axes, format="%+2.0f dB")
plt.tight_layout()
fig.savefig("mel_spectrogram_compare.png", dpi=300)
plt.close(fig)
print("  已保存 mel_spectrogram_compare.png")

# （可選）保存 MFCC 序列熱圖比較
# fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
# librosa.display.specshow(librosa.power_to_db(np.abs(librosa.stft(y1))**2, ref=np.max), sr=sr1, hop_length=hop, x_axis='time', y_axis=None, ax=axes[0])
# axes[0].set_title(f"{audio_files[0]} MFCC 序列")
# librosa.display.specshow(librosa.power_to_db(np.abs(librosa.stft(y2))**2, ref=np.max), sr=sr2, hop_length=hop, x_axis='time', y_axis=None, ax=axes[1])
# axes[1].set_title(f"{audio_files[1]} MFCC 序列")
# axes[1].set_xlabel("時間 (秒)")
# plt.tight_layout()
# fig.savefig("mfcc_compare.png", dpi=300)
# plt.close(fig)
# print("  已保存 mfcc_compare.png")

# 將分析結果輸出為 CSV 檔
print("\n步驟5: 將分析結果輸出為 CSV...")
csv_filename = "analysis_results.csv"
metrics = [
    "duration_s", "sample_rate", "channels", "bit_depth",
    "rms_mean", "rms_max", "rms_min", "dynamic_range_db",
    "loudness_lufs", "crest_factor",
    "zcr_mean", "spectral_centroid_mean", "spectral_bandwidth_mean",
    "spectral_rolloff_mean", "spectral_entropy_mean"
]
# 加入 MFCC 均值 (取前13個)
for i in range(13):
    metrics.append(f"mfcc{i+1}_mean")
# 組織 CSV 行
with open(csv_filename, mode='w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    header = ["Metric", audio_files[0], audio_files[1]]
    writer.writerow(header)
    for metric in metrics:
        if metric.startswith("mfcc") and metric.endswith("_mean"):
            idx = int(metric.replace("mfcc", "").replace("_mean", "")) - 1
            val1 = analysis_results[audio_files[0]]["mfcc_means"][idx]
            val2 = analysis_results[audio_files[1]]["mfcc_means"][idx]
        else:
            val1 = analysis_results[audio_files[0]].get(metric, None)
            val2 = analysis_results[audio_files[1]].get(metric, None)
        # 格式化輸出，浮點取4位小數
        if isinstance(val1, float):
            val1_str = f"{val1:.4f}"
        else:
            val1_str = str(val1)
        if isinstance(val2, float):
            val2_str = f"{val2:.4f}"
        else:
            val2_str = str(val2)
        writer.writerow([metric, val1_str, val2_str])
print(f"  已保存分析結果 CSV 檔: {csv_filename}")

print("\n分析完成！所有數值結果已輸出至 CSV，圖表已保存為 PNG 檔。請查看當前目錄下的檔案以進一步檢視和比較兩段音訊的品質差異。")
