import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav
from faster_whisper import WhisperModel
import keyboard  # キー入力検知用 (pip install keyboard が必要)
import queue
import sys

# --- 設定 ---
MODEL_SIZE = "medium"   # tiny, base, small, medium, large-v3 から選択
DEVICE = "cuda"         # GPUがあるなら "cuda"、なければ "cpu"
COMPUTE_TYPE = "float16"  # int8 (軽量) または float16 (GPU用)
SAMPLE_RATE = 44100    # サンプリングレート
FILENAME = "output_audio.wav"

def record_audio(filename):
    """マイク入力を録音し、WAVファイルに保存する関数"""
    print("\n=== 録音開始 (Enterキーを押して停止) ===")
    
    q = queue.Queue()

    def callback(indata, frames, time, status):
        """音声入力のコールバック関数"""
        if status:
            print(status, file=sys.stderr)
        q.put(indata.copy())

    # 録音データを格納するリスト
    audio_data = []
    
    # ストリーム開始
    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, callback=callback):
        # Enterが押されるまでループ
        input() 
        
    print("=== 録音終了...データ処理中 ===")

    # キューからデータを取り出し結合
    while not q.empty():
        audio_data.append(q.get())
    
    # NumPy配列に変換して保存
    recording = np.concatenate(audio_data, axis=0)
    # float32のデータをint16に変換（WAV保存用）
    recording_int16 = (recording * 32767).astype(np.int16)
    wav.write(filename, SAMPLE_RATE, recording_int16)
    print(f"音声ファイルを保存しました: {filename}")

def transcribe_audio(filename):
    """保存された音声をテキストに変換する関数"""
    print("\n=== 文字起こし開始 ===")
    
    # モデルのロード（初回のみダウンロードが走ります）
    model = WhisperModel(MODEL_SIZE, device=DEVICE, compute_type=COMPUTE_TYPE)

    segments, info = model.transcribe(filename, beam_size=5, language="ja")

    print(f"検出言語: {info.language} (確率: {info.language_probability:.2f})")
    print("-" * 30)

    full_text = ""
    for segment in segments:
        print(f"[{segment.start:.2f}s -> {segment.end:.2f}s] {segment.text}")
        full_text += segment.text + "\n"

    # テキストファイルに保存
    with open("result.txt", "w", encoding="utf-8") as f:
        f.write(full_text)
    
    print("-" * 30)
    print("テキストを result.txt に保存しました。")

if __name__ == "__main__":
    try:
        # 1. 録音
        record_audio(FILENAME)
        # 2. 文字起こし
        transcribe_audio(FILENAME)
    except Exception as e:
        print(f"エラーが発生しました: {e}")
        