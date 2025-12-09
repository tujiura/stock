import sounddevice as sd
import numpy as np
from faster_whisper import WhisperModel
import queue
import sys
from datetime import datetime  # 【追加】時刻処理用

# --- 設定 ---
MODEL_SIZE = "large-v3"   # tiny, base, small, medium, large-v3 から選択
DEVICE = "cuda"         # GPUがあるなら "cuda"、なければ "cpu"
COMPUTE_TYPE = "float16"  # int8 (軽量) または float16 (GPU用)
SAMPLE_RATE = 44100    # サンプリングレート
BLOCK_SIZE = 2000
THRESHOLD = 0.03     # 環境に合わせて調整してください
SILENCE_DURATION = 1.0

class RealTimeTranscriber:
    def __init__(self):
        print("モデルをロード中...")
        self.model = WhisperModel(MODEL_SIZE, device=DEVICE, compute_type=COMPUTE_TYPE)
        self.q = queue.Queue()
        self.buffer = []
        self.is_speaking = True
        self.silence_counter = 0
        self.speech_start_time = None  # 【追加】発話開始時刻を保持する変数

    def callback(self, indata, frames, time, status):
        if status:
            print(status, file=sys.stderr)
        self.q.put(indata.copy())

    def get_rms(self, data):
        return np.sqrt(np.mean(data**2))

    def process_stream(self):
        print("\n=== リアルタイム文字起こし開始 (Ctrl+Cで停止) ===")
        print(f"閾値: {THRESHOLD} / 無音判定: {SILENCE_DURATION}秒")
        
        silence_blocks = int(SILENCE_DURATION * SAMPLE_RATE / BLOCK_SIZE)

        with sd.InputStream(samplerate=SAMPLE_RATE, blocksize=BLOCK_SIZE, channels=1, callback=self.callback):
            while True:
                indata = self.q.get()
                rms = self.get_rms(indata)
                
                if rms > THRESHOLD:
                    # 【変更】発話が始まった「瞬間」の時刻を記録
                    if not self.is_speaking:
                        self.speech_start_time = datetime.now()
                        
                    self.is_speaking = True
                    self.silence_counter = 0
                    self.buffer.append(indata)
                    sys.stdout.write("■") 
                    sys.stdout.flush()
                
                else:
                    if self.is_speaking:
                        self.buffer.append(indata)
                        self.silence_counter += 1
                        
                        if self.silence_counter > silence_blocks:
                            self.transcribe_buffer()
                            self.reset_state()
                    else:
                        pass

    def transcribe_buffer(self):
        print("\n[推論中...]")
        if len(self.buffer) == 0:
            return

        # 時刻のフォーマット作成 (例: 2023-10-27 15:30:05)
        # speech_start_time が何らかの理由で None の場合は現在時刻を使用
        if self.speech_start_time:
            timestamp_str = self.speech_start_time.strftime('%Y-%m-%d %H:%M:%S')
        else:
            timestamp_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        audio_data = np.concatenate(self.buffer, axis=0)
        audio_data = audio_data.flatten().astype(np.float32)

        segments, _ = self.model.transcribe(audio_data, language="ja", beam_size=5)

        # 【変更】時刻付きで出力
        for segment in segments:
            # 画面表示
            output_text = f"[{timestamp_str}] {segment.text}"
            print(output_text)
            
            # ログファイルへの追記保存（append mode）
            with open("transcription_log.txt", "a", encoding="utf-8") as f:
                f.write(output_text + "\n")

        print("---")

    def reset_state(self):
        self.buffer = []
        self.is_speaking = False
        self.silence_counter = 0
        self.speech_start_time = None # 時刻リセット
        sys.stdout.write("待機中...")
        sys.stdout.flush()

if __name__ == "__main__":
    transcriber = RealTimeTranscriber()
    try:
        transcriber.process_stream()
    except KeyboardInterrupt:
        print("\n終了します")